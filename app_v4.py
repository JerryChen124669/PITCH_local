import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SkLinear
from matplotlib.colors import LinearSegmentedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from scipy.stats import pearsonr
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import networkx as nx
import io
from io import BytesIO
import shap
#from imblearn.under_sampling import RandomUnderSampler
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import re

# --- CACHED ML FUNCTIONS TO PREVENT LAG ---

@st.cache_resource(show_spinner="Training PyTorch model...")
def train_pytorch_model(model_type, input_size, hidden_size, dropout, lr, weight_decay, epochs, seed, X_train, y_train):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    
    if model_type == "Multilayers Perception":
        model = MLPRegression(input_size, hidden_size, dropout)
    else:
        model = LinearRegression(input_size)
        
    model.initialize()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return model

# FIX 1: Removed @st.cache_data here and changed _model to model 
# so Streamlit actually evaluates the newly trained model instead of recycling old results.
def evaluate_pytorch_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        corr, _ = pearsonr(y_test, test_preds)
    return test_preds, corr

@st.cache_data(show_spinner="Calculating Integrated Gradients (This may take a moment)...")
def calculate_ig(_model, X_raw):
    _model.eval()
    ig = IntegratedGradients(_model, multiply_by_inputs=False)
    W_tensor = ig.attribute(torch.from_numpy(X_raw), n_steps=50)
    W = W_tensor.detach().numpy()
    attr = (X_raw * W)
    return attr, W

# --- CACHED PLOTTING FUNCTIONS TO PREVENT LAG ---
@st.cache_data(show_spinner="Loading Excel Data...")
def load_data(file):
    return pd.read_excel(file)

@st.cache_data
def get_cached_shap_plot(attr, X_raw, headers):
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(attr, X_raw, feature_names=headers, max_display=30, plot_size=None, show=False, color_bar_label='Feature value')
    plt.subplots_adjust(left=0.3)
    plt.xlabel('Integrated Gradient Attribution Value')
    plt.tight_layout()
    
    # 1. Generate PNG for Streamlit's st.image()
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=150)
    
    # 2. Generate SVG for the download button
    buf_svg = io.BytesIO()
    fig.savefig(buf_svg, format="svg", bbox_inches="tight")
    
    plt.close(fig) 
    
    # Return BOTH byte buffers
    return buf_png.getvalue(), buf_svg.getvalue()

@st.cache_data
def get_cached_bar_plot(avg_abs, headers, top_30_idx):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(30), avg_abs[top_30_idx][::-1], color='steelblue')
    ax.set_yticks(range(30))
    ax.set_yticklabels(headers[top_30_idx][::-1])
    ax.set_xlabel("Mean |Attribution|")
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()

@st.cache_data
def get_cached_reg_plot(y_ig_raw, y_pred, target_prot):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_ig_raw, y_pred, alpha=0.4, color='forestgreen')
    ax.plot([y_ig_raw.min(), y_ig_raw.max()], [y_ig_raw.min(), y_ig_raw.max()], 'r--', lw=2)
    ax.set_xlabel(f"Actual {target_prot} IG Score")
    ax.set_ylabel(f"Predicted {target_prot} IG")
    return fig

@st.cache_data
def get_cached_tree_plot(_clf, tree_features): 
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(_clf, feature_names=tree_features, filled=True, rounded=True, 
              precision=2, fontsize=10, impurity=False, node_ids=True,ax=ax)
    for text in ax.texts:
        t = text.get_text()
        match = re.search(r"value = \[(.*?)\]", t)
        if match:
            vals = match.group(1).split(", ")
            y0, y1 = int(float(vals[0])), int(float(vals[1]))
            lines = t.split("\n")
            
            # lines[0] is "node #X"
            # lines[1] is the condition if it contains "<="
            if "<=" in lines[1]:
                header = f"{lines[0]}\n{lines[1]}"
            else:
                header = f"{lines[0]}" # It's a leaf node, no condition
                
            text.set_text(f"{header}\nTotal: {y0+y1}\nTarget: {y1}")
    return fig


@st.cache_data
def get_cached_network_plot(_G):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # INCREASED k from 0.8 to 1.5 to push nodes further apart. 
    # INCREASED iterations to 300 to let the layout settle better.
    pos = nx.spring_layout(_G, k=2, iterations=400, seed=42, weight = None) 
    edges = _G.edges()
    weights = [_G[u][v]['weight'] for u, v in edges]
    
    if weights:
        min_w, max_w = min(weights), max(weights)
    else:
        min_w, max_w = 1, 1

    hex_colors = ['#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000']
    
    edge_colors = []
    edge_widths = []
    
    for w in weights:
        if max_w == min_w:
            color_idx = 5 
        else:
            norm_w = (w - min_w) / (max_w - min_w)
            color_idx = int(norm_w * 5.99) 
        
        edge_colors.append(hex_colors[color_idx])
        # Slightly scaled down max width to prevent giant thick blobs
        width = 1 + (w / max_w) * 3 
        edge_widths.append(width)

    # REDUCED node size from 2500 to 1200 to give the plot breathing room
    node_size_val = 1200
    
    nx.draw_networkx_nodes(_G, pos, ax=ax, node_color='lightblue', node_size=node_size_val)
    # Reduced font size slightly to fit the smaller nodes
    nx.draw_networkx_labels(_G, pos, ax=ax, font_size=8, font_weight="bold")
    
    nx.draw_networkx_edges(
        _G, pos, ax=ax,
        edgelist=edges,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowstyle='<|-|>',      
        arrowsize=15,            
        node_size=node_size_val, 
        connectionstyle="arc3,rad=0.1" 
    )
    
    ax.axis('off') 
    plt.tight_layout()
    return fig
  
@st.cache_data
def get_cached_umap(X_raw):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=21)
    embedding = reducer.fit_transform(X_scaled)
    return embedding

# --- 1. MODEL DEFINITION ---
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.initialize()

    def forward(self, x):
        return self.linear(x).squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.initialize()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()

    def initialize(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

# --- SIDEBAR CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Interpretable Multimodal Analysis")

with st.sidebar:
    st.header("1. Data & Hyperparameters")
    uploaded_file = st.file_uploader("Upload Excel Data", type=["xlsx"])
    
    with st.form("hyperparameter_form"):
        selected_model_type = st.selectbox("Select Architecture", ["Multilayers Perception", "Linear regression"])
        st.markdown("---")
        seed = st.number_input("Random Seed", value=20)
        
        if selected_model_type == "Multilayers Perception":
            hidden_size = st.slider("Hidden Size (MLP only)", 64, 512, 256)
            dropout = st.slider("Dropout Rate (MLP only)", 0.0, 0.7, 0.5)
        else:
            hidden_size, dropout = 0, 0 # Dummy values for caching
        
        lr = st.number_input("Learning Rate", value=0.0002, format="%.5f")
        weight_decay = st.number_input("Weight Decay (L2)", value=1e-4, format="%.6f")
        epochs = st.number_input("Epochs", value=2000)
        
        submit_params = st.form_submit_button("✅ Update Settings")
        
        # FIX 2: Clear old results from memory when settings are updated
        if submit_params:
            for key in ["val_results", "corr", "model", "trained_model_type"]:
                if key in st.session_state:
                    del st.session_state[key]

    st.markdown("---") 
    if st.button("🔄 Upload New Data / Start Over"):
        st.session_state.clear() 
        st.cache_data.clear()    
        st.cache_resource.clear() 
        st.rerun()

if 'step' not in st.session_state: 
    st.session_state.step = 1

tab1, tab2, tab3 = st.tabs(["Step 1: Model Training & Evaluation", "Step 2: Context-Dependent Attribution Analysis", "Step 3: Dependency Analysis"])

# --- STEP 1: TRAINING ---
with tab1:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        headers = df.columns[:-1].values
        target_name = df.columns[-1]
        X_raw = df.iloc[:, :-1].values.astype(np.float32)
        y_raw = df.iloc[:, -1].values.astype(np.float32)

        st.info(f"Loaded: **{target_name}** with {X_raw.shape[1]} features.")

        if st.button(f"🚀 Start {selected_model_type} Training"):
            X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=seed)
            
            # Use cached training function
            model = train_pytorch_model(
                selected_model_type, X_raw.shape[1], hidden_size, dropout, 
                lr, weight_decay, epochs, seed, X_train, y_train
            )
            
            # Use evaluation function (now uncached!)
            test_preds, corr = evaluate_pytorch_model(model, X_test, y_test)
            
            st.session_state.model = model
            st.session_state.trained_model_type = selected_model_type 
            st.session_state.X_raw = X_raw
            st.session_state.X_test = X_test  # <--- NEW: Saved for UMAP generation
            st.session_state.headers = headers
            st.session_state.target_name = target_name
            st.session_state.corr = corr
            st.session_state.val_results = pd.DataFrame({
                "Measured_Target": y_test,
                "Predicted_Target": test_preds
            })
            
            if selected_model_type == "Multilayers Perception":
                st.session_state.step = 2
                st.success(f"Training Complete! Pearson r: {corr:.4f}. You can now proceed to Step 2.")
            else:
                st.session_state.step = 1 
                st.success(f"Training Complete! Pearson r: {corr:.4f}.")
                st.info("Note: IG Attribution and Dependency Analysis are only available for Multilayers Perception models.")

        

        if "val_results" in st.session_state:
            res = st.session_state.val_results
            st.divider()
            st.subheader("💾 Export Options")
            col1, col2 = st.columns(2)
            
            # Use the saved model type for naming
            trained_type = st.session_state.get("trained_model_type", selected_model_type)
            
            buffer = io.BytesIO()
            torch.save(st.session_state.model.state_dict(), buffer)
            col1.download_button(
                label=f"📥 Download Trained {trained_type} Model (.pth)",
                data=buffer.getvalue(),
                file_name=f"{trained_type.replace(' ', '_').lower()}_weights.pth",
                mime="application/octet-stream"
            )
            
            csv = st.session_state.val_results.to_csv(index=False).encode('utf-8')
            col2.download_button(
                label="📥 Download Validation Data (CSV)",
                data=csv,
                file_name="training_validation_results.csv",
                mime="text/csv"
            )

            st.divider()
            st.subheader("📊 Model Performance Evaluation")

            # --- EXISTING SCATTER PLOT ---
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.scatter(res["Measured_Target"], res["Predicted_Target"], alpha=0.5, color='teal')
            ax1.plot([res["Measured_Target"].min(), res["Measured_Target"].max()], 
                     [res["Measured_Target"].min(), res["Measured_Target"].max()], 'r--', lw=2)
            ax1.set_xlabel("Measured Groundtruth")
            ax1.set_ylabel("Predicted Results")
            ax1.set_title(f"Scatter: {trained_type} Performance (r={st.session_state.corr:.4f})")
            st.pyplot(fig1)
            plt.close(fig1) # <--- Clean up memory
            
            # --- NEW: SIDE-BY-SIDE UMAPS ---
            umap_col1, umap_col2 = st.columns(2)
            
            # Generate UMAP layout using the FULL dataset (X_raw) so it matches Step 2
            embedding_full = get_cached_umap(st.session_state.X_raw)
            
            # Generate predictions for the full dataset for the right-side plot
            st.session_state.model.eval()
            with torch.no_grad():
                full_preds = st.session_state.model(torch.tensor(st.session_state.X_raw, dtype=torch.float32)).numpy()
            
            # Lock color scales so both plots match perfectly
            vmin = min(y_raw.min(), full_preds.min())
            vmax = max(y_raw.max(), full_preds.max())
            
            with umap_col1: # LEFT: Measured Groundtruth
                fig_umap_m, ax_umap_m = plt.subplots(figsize=(6, 5))
                scatter_m = ax_umap_m.scatter(embedding_full[:, 0], embedding_full[:, 1], 
                                              c=y_raw, cmap='viridis', 
                                              s=15, alpha=0.8, vmin=vmin, vmax=vmax)
                plt.colorbar(scatter_m, ax=ax_umap_m, label=f"Measured {target_name}")
                ax_umap_m.set_title("UMAP: Measured Groundtruth (Full Data)")
                ax_umap_m.axis('off')
                st.pyplot(fig_umap_m)
                plt.close(fig_umap_m) # <--- Clean up memory
                
            with umap_col2: # RIGHT: Predicted Results
                fig_umap_p, ax_umap_p = plt.subplots(figsize=(6, 5))
                scatter_p = ax_umap_p.scatter(embedding_full[:, 0], embedding_full[:, 1], 
                                              c=full_preds, cmap='viridis', 
                                              s=15, alpha=0.8, vmin=vmin, vmax=vmax)
                plt.colorbar(scatter_p, ax=ax_umap_p, label=f"Predicted {target_name}")
                ax_umap_p.set_title("UMAP: Predicted Results (Full Data)")
                ax_umap_p.axis('off')
                st.pyplot(fig_umap_p)
                plt.close(fig_umap_p) # <--- Clean up memory

    else:
        st.info("Please upload an Excel file in the sidebar to begin.")

# --- STEP 2: IG ATTRIBUTION ---
with tab2:
    # Explicitly check if a model has been trained AND if it is an MLP
    trained_type = st.session_state.get("trained_model_type", None)
    if trained_type == "Multilayers Perception":
        if st.button("🔍 Calculate Integrated Gradients"):
            # Use cached IG function
            attr, W = calculate_ig(st.session_state.model, st.session_state.X_raw)
            
            avg_abs = np.mean(np.abs(attr), axis=0)
            avg_raw = np.mean(attr, axis=0)
            
            st.session_state.attr = attr
            st.session_state.W_vals = W 
            st.session_state.avg_abs = avg_abs
            st.session_state.avg_raw = avg_raw
            st.session_state.ranking = np.argsort(avg_abs)[::-1]
            st.session_state.step = 3
            st.success("IG Calculation Finished!")
            st.rerun() 

        if 'attr' in st.session_state:
            st.subheader("💾 Export Attribution Data", help = "Each data point (cell) has its own IG score for individual features (10 Features X 100 cells will report 1000 IG score.). Please use the swarm plot or download attribution/IG weight data to identify feature of interest as the focused feature for the next step.\n\n A feature can be positively or negatively attributing to the target response by context.\n A feature can be direct or inverse correlated to the target feature (high value of a feature (color: red) shows positive attribution (X-axis): direct correlation)")
            save_col1, save_col2 = st.columns(2)
            
            df_attr = pd.DataFrame(st.session_state.attr, columns=st.session_state.headers)
            df_w = pd.DataFrame(st.session_state.W_vals, columns=st.session_state.headers)
            
            save_col1.download_button(
                label="📥 Download IG Attribution Values (CSV)",
                data=df_attr.to_csv(index=False).encode('utf-8'),
                file_name="ig_attribution_values.csv",
                mime="text/csv",
                help="Attribution = Input * Gradient weight for every data point."
            )
            
            save_col2.download_button(
                label="📥 Download IG Weights (CSV)",
                data=df_w.to_csv(index=False).encode('utf-8'),
                file_name="ig_w_sensitivity_weights.csv",
                mime="text/csv",
                help="weights of each features for each sample calculated by Integrated Gradient"
            )
            
            st.divider()

            st.subheader("Top 30 Features (Global Importance)")
            top_30_idx = st.session_state.ranking[:30]

            # Unpack the two formats returned by your new function
            shap_png_bytes, shap_svg_bytes = get_cached_shap_plot(st.session_state.attr, st.session_state.X_raw, st.session_state.headers)
            
            # Show the PNG on the screen (PIL loves PNGs)
            st.image(shap_png_bytes)
            
            # Feed the SVG behind the scenes into the download button
            st.download_button(
                label="📥 Download Top 30 Plot (SVG)", 
                data=shap_svg_bytes, 
                file_name="attr_plot.svg", 
                mime="image/svg+xml" 
            )
            
            # --- NEW UMAP SECTION REPLACING BAR PLOT ---
            st.divider()
            st.subheader("🌌 UMAP Cellular Projection by Feature Attribution")
            
            # 1. THE PROJECTION: Uses ALL raw features to determine the X/Y coordinates
            embedding = get_cached_umap(st.session_state.X_raw)
            
            # 2. THE SEARCHABLE BOX: Streamlit's selectbox is natively searchable and scrollable!
            default_feat = st.session_state.headers[st.session_state.ranking[0]]
            selected_ig_feature = st.selectbox(
                "Search and select a target feature to view its IG Attribution:", 
                st.session_state.headers,
                index=st.session_state.headers.tolist().index(default_feat),
                help="Click and type to search for a specific feature."
            )
            
            # 3. THE COLOR: Extract ONLY the selected feature's IG scores for the paint job
            feat_idx = np.where(st.session_state.headers == selected_ig_feature)[0][0]
            ig_scores = st.session_state.attr[:, feat_idx]
            
            # 4. Generate the UMAP Plot
            fig_umap_ig, ax_umap_ig = plt.subplots(figsize=(10, 8))
            
            # Calculate symmetric bounds so 0.0 is ALWAYS perfectly white/gray
            vmax = np.percentile(np.abs(ig_scores), 95)
            if vmax == 0: vmax = 0.001 
            
            # --- NEW CUSTOM COLORMAP ---
            # Your hex codes (ordered Blue -> Light Yellow -> Red)
            custom_hex_colors = [
            "#00441b", "#1b7837", "#5aae61", "#a6dba0", 
            "#d9f0d3",  
            "#e7d4e8", "#c2a5cf", "#9970ab", "#762a83", "#40004b"
            ]
            
            my_cmap = LinearSegmentedColormap.from_list("custom_rdybl", custom_hex_colors)

            # Plot all cells, coloring them strictly by the selected feature's attribution
            scatter = ax_umap_ig.scatter(
                embedding[:, 0], embedding[:, 1], 
                c=ig_scores, cmap=my_cmap, # <--- Apply your custom colormap here!
                s=12, alpha=0.8, edgecolors='none', 
                vmin=-vmax, vmax=vmax 
            )
            
            cbar = plt.colorbar(scatter, ax=ax_umap_ig)
            cbar.set_label(f'IG Attribution of {selected_ig_feature}', rotation=270, labelpad=15)
            ax_umap_ig.set_title(f'UMAP Layout: All Features | Color: Attribution of {selected_ig_feature}')
            ax_umap_ig.set_xlabel('UMAP 1')
            ax_umap_ig.set_ylabel('UMAP 2')
            
            ax_umap_ig.spines['top'].set_visible(False)
            ax_umap_ig.spines['right'].set_visible(False)
            
            st.pyplot(fig_umap_ig)
            
            # 5. UMAP Download Button
            import io
            buf_umap_ig = io.BytesIO()
            fig_umap_ig.savefig(buf_umap_ig, format="svg", bbox_inches="tight")
            st.download_button(
                label="📥 Download IG UMAP Plot (SVG)",
                data=buf_umap_ig.getvalue(),
                file_name=f"umap_ig_{selected_ig_feature}.svg",
                mime="image/svg+xml"
            )
            plt.close(fig_umap_ig) 
            
            st.divider()

    else:
        st.warning("Please train a 'Multilayers Perception' model in Step 1 to process IG Attribution!")

# --- STEP 3: INTERACTION & LOGIC DISCOVERY ---
with tab3:
    if st.session_state.step >= 3:
        st.header("Step 3: Interaction & Logic Discovery", help = "Select a focused feature to explore the conditons (context) where the focused feature would be a very important feature to the target response.\n\n ex: Actin has been known to have a strong correlation with nuclear YAP. To explore how actin amount regulate YAP, selected 'actin amount' as interested feature, select other measured features (such as RhoA, E-cad, LATS1/2, cell morphology, mitochondria activity...) as attributing features.")
        
        if 'reg_data' not in st.session_state: st.session_state.reg_data = None
        if 'dt_data' not in st.session_state: st.session_state.dt_data = None
        
        # 1. THE FORM (Data Selection Only)
        with st.form("logic_discovery_form"):
            st.subheader("Feature selection & Parameters")
            b_left, b_right = st.columns(2)

            with b_left:
                target_prot = st.selectbox(
                    "Select focused feature", 
                    st.session_state.headers[st.session_state.ranking],
                    help="Select the feature you want to explain how it and other features integratively regulate target response. Default = The feature with No.1 absolute attribution ranking."
                )
            
            c_left, c_right = st.columns(2)
            
            with c_left:
                attr_direction = st.radio("Condition Logic", ["Positive", "Negative"], help = "Want to study the positive or negative attribution behavior of the focused feature to target response. Please regard the distribution of attribution at step 2.\n\n ex: For neurons, Fxyd6 regulate ion channel has high possitive attribution to input resistance but never a negative attribution. Select 'Positive'")
                corr_info = st.radio("Correlation", ["Direct", "Inverse"], help = "check what is the correlation between the focused feature and target response. Please see the plot at step 2, it indicates the correlation of the individual features and the target response.")
                percentile_val = st.slider("IG Threshold (%)", 50, 99, 90, help ="Binary the data points/cells based on attribution. If select 'positive' condition logic, the data with highest {100 - X} percentage of the attribution will be labeled as 1, the others are labeled 0, vice versa.\n\n ex: Condition Logic: 'Negative', IG Threshold: '80' -> the data points with lowest 20 percentage of attribution will be markered as 1")
                class_1_weight = st.number_input(
                    "Class 1 Penalty Weight", min_value=1, max_value=10, value=4, step=1,
                    help = "How strict the decision tree would try to find the data labeled as 1 (true positive). Higher weight is more strict. Min = 1, Max = 10. Note: if the weight is too low, the tree can get lazy. If the weight is too high, the tree can overfitting."
                )
            
            with c_right:
                all_prots = st.session_state.headers.tolist()
                default_prots = st.session_state.headers[st.session_state.ranking[:30]].tolist()
                if target_prot in default_prots: default_prots.remove(target_prot)
                
                selected_features = st.multiselect("Attributing Features", all_prots, default=default_prots, help="Investigate the role of these features in 'attributing the focused feature become important to target response'. ")
            
            st.markdown("---")
            # The ONLY button inside the form
            data_confirmed = st.form_submit_button("✅ Confirm Selection & Update Data")

        # 2. DATA PREPARATION (Happens automatically using the confirmed settings)
        if selected_features:

            # Prepare the mathematical arrays
            names = st.session_state.headers
            target_idx = np.where(names == target_prot)[0][0]
            y_ig_raw = st.session_state.attr[:, target_idx]

            reg_features = [f for f in selected_features if f != target_prot]
            reg_indices = [np.where(names == f)[0][0] for f in reg_features]
            X_vals_reg = st.session_state.X_raw[:, reg_indices]

            if attr_direction == "Positive":
                exp_threshold = np.percentile(y_ig_raw, percentile_val)
                exp_y_binary = np.where(y_ig_raw > exp_threshold, 1, 0)
            else:
                exp_threshold = np.percentile(y_ig_raw, 100 - percentile_val)
                exp_y_binary = np.where(y_ig_raw < exp_threshold, 1, 0)

            # Prepare the Excel Data Memory Buffer
            df_export = pd.DataFrame()
            df_export[f"{target_prot}_Binary_Label"] = exp_y_binary
            df_export[f"{target_prot}_Attribution"] = y_ig_raw
            df_export[f"{target_prot}_Raw"] = st.session_state.X_raw[:, target_idx]
            for f, idx in zip(reg_features, reg_indices):
                df_export[f"{f}_Raw"] = st.session_state.X_raw[:, idx]

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Analysis_Data')
            excel_data = output.getvalue()

            # 3. THE THREE PARALLEL ACTION BUTTONS (Outside the form)
            st.markdown("### Choose an Action", help="Export Excell data: Export the binary focused feature label and attribution of the focused feature with the selected Attributing feature value according to the setting.\n\n Run linear regression: Use Attributing feature values to predict IG score of the focused feature. A good correlation indicates that the selected attributing features are sufficient to describe the interaction with the focused feature.\n\nRun Decision Tree: Use the binary focus feature label to explore the conditions where the focus feature is important to the target response. Total: number of total samples. Positive: number of positive samples labeled as 1. Gini/Precision: impurity indicators. Exporting LLM-readable txt is available.")
            action_col1, action_col2, action_col3 = st.columns(3)

            with action_col1:
                st.download_button(
                    label="📥 Export Excel Data",
                    data=excel_data,
                    file_name=f"{target_prot}_Binarylabel.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with action_col2:
                # Changed from form_submit_button to standard button
                run_reg = st.button("🚀 Run Linear Regression")
            
            with action_col3:
                run_dt = st.button("🌳 Run Decision Tree")

            # 4. ALGORITHM EXECUTION LOGIC
            if run_reg:
                st.session_state.dt_data = None 
                reg = SkLinear().fit(X_vals_reg, y_ig_raw)
                y_pred = reg.predict(X_vals_reg)
                total_r, p_val = pearsonr(y_ig_raw, y_pred)
                st.session_state.reg_data = {
                    "r": total_r, "p": p_val, "y_actual": y_ig_raw, "y_pred": y_pred, "target": target_prot
                }

            if run_dt:
                st.session_state.reg_data = None 
                dt_clf = DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=5, max_leaf_nodes=14, random_state=42, 
                    class_weight={0: 1, 1: class_1_weight}
                )
                dt_clf.fit(X_vals_reg, exp_y_binary)
                cv_scores = cross_val_score(dt_clf, X_vals_reg, exp_y_binary, cv=5, scoring='f1_macro')
                
                st.session_state.dt_cv_score = cv_scores.mean()
                st.session_state.dt_data = {
                    "clf": dt_clf, "features": reg_features, "indices": reg_indices, 
                    "direction": attr_direction, "perc": percentile_val, "target": target_prot,
                    "y_binary": exp_y_binary, "corr": corr_info, "weight": class_1_weight
                }

        # --- 2. PERSISTENT RENDERING FROM SESSION STATE ---
        if st.session_state.reg_data is not None:
            r_data = st.session_state.reg_data
            st.subheader(f"📈 Representivity check - linear relationship", 
                         help="use selected features (without focused feature) to predict the IG score of focused feature. If the correlation is high, the selected features are sufficient to describe the attribution trend of the focused feature")
            m1, m2 = st.columns(2)
            m1.metric("Model Pearson r", f"{r_data['r']:.4f}")
            m2.metric("P-value", f"{r_data['p']:.2e}")
            
            fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
            ax_reg.scatter(r_data['y_actual'], r_data['y_pred'], alpha=0.4, color='forestgreen')
            ax_reg.plot([r_data['y_actual'].min(), r_data['y_actual'].max()], 
                        [r_data['y_actual'].min(), r_data['y_actual'].max()], 'r--', lw=2)
            ax_reg.set_xlabel(f"Actual {r_data['target']} IG Score")
            ax_reg.set_ylabel(f"Predicted {r_data['target']} IG")
            st.pyplot(fig_reg)

        if st.session_state.dt_data is not None:
            dt = st.session_state.dt_data
            st.subheader(f"🌳 Decision Logic for {dt['direction']} Attribution", help="use selected features to classify the conditions where the attribution of focused feature is significant. Indicating the critical interactions in the system.")
            
            if 'dt_cv_score' in st.session_state:
                st.metric(
                    label="Tree Rule Robustness (Cross-Validation F1)", 
                    value=f"{st.session_state.dt_cv_score:.1%}",
                    help="A high score means these rules will generalize to new experiments. A low score means the tree is overfitting to the specific dataset."
                )

            
            fig_tree, ax = plt.subplots(figsize=(20, 10))
            plot_tree(dt['clf'], feature_names=dt['features'], filled=True, rounded=True, 
              precision=2, fontsize=6, impurity=False, node_ids=True, ax=ax)
            for text in ax.texts:
                t = text.get_text()
                match = re.search(r"value = \[(.*?)\]", t)
                
                if match:
                    # --- INDENTATION FIXED BELOW ---
                    vals = match.group(1).split(", ")
                    y0= int(float(vals[0])) 
                    y1 = int(round(float(vals[1]) / dt['weight'])) # Divide by the penalty weight
                    lines = t.split("\n")
                
                    # lines[0] is "node #X"
                    # lines[1] is the condition if it contains "<="
                    if len(lines) > 1 and "<=" in lines[1]:
                        header = f"{lines[0]}\n{lines[1]}"
                    else:
                        header = f"{lines[0]}" # It's a leaf node, no condition
                    
                    text.set_text(f"{header}\nTotal: {y0+y1}\nTarget: {y1}")

            # Don't forget to tell Streamlit to actually draw the customized plot!
            st.pyplot(fig_tree)

            # (Removed the old Decision Tree image download button)
            plt.close(fig_tree) # Clean up memory
            
            # --- EXTRACT LIVE TP/FP COUNTS FOR THE TEXT FILE ---
            t_ = dt['clf'].tree_
            
            def get_node_rules(tree, feature_names, target_node):
                left = tree.children_left
                right = tree.children_right
                threshold = tree.threshold
                features = [feature_names[i] if i >= 0 else "undefined" for i in tree.feature]
                
                def recurse(node, path):
                    if node == target_node:
                        return path
                    if left[node] != -1:
                        p = recurse(left[node], path + [f"{features[node]} <= {threshold[node]:.3f}"])
                        if p: return p
                        p = recurse(right[node], path + [f"{features[node]} > {threshold[node]:.3f}"])
                        if p: return p
                    return None
                return recurse(0, [])

            is_leaves = (t_.children_left == -1)
            positive_leaves = [i for i in range(t_.node_count) if is_leaves[i] and np.argmax(t_.value[i][0]) == 1]
            
            X_tree_subset = st.session_state.X_raw[:, dt['indices']]
            full_tree_paths = dt['clf'].apply(X_tree_subset)
            y_binary = dt['y_binary'] 
            
            leaf_summaries = []
            for node_id in positive_leaves:
                in_this_leaf = (full_tree_paths == node_id)
                tp = np.sum(in_this_leaf & (y_binary == 1))
                fp = np.sum(in_this_leaf & (y_binary == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                gini_score = t_.impurity[node_id]

                rules = get_node_rules(t_, dt['features'], node_id)
                rule_str = " AND ".join(rules) if rules else "Root Node"
                
                leaf_summaries.append(
                    f"Node {node_id}:\n"
                    f"  - Logic: {rule_str}\n"
                    f"  - True Positives (TP): {tp} (High Attribution)\n"
                    f"  - False Positives (FP): {fp} (Noise/Other)\n"
                    f"  - Precision: {precision:.1%}"
                    f"  - Gini Impurity: {gini_score:.4f}"
                )
                
            leaf_context = "\n\n".join(leaf_summaries) if leaf_summaries else "No purely positive leaves found."
            
            tree_rules = export_text(dt['clf'], feature_names=dt['features'])

            total_samples = len(y_binary)
            target_samples = np.sum(y_binary == 1)

            # --- NEW UI ROW: LLM EXPORTATION (FORM FORMAT) ---
            st.divider()
            st.markdown("### 🧠 Export LLM Interpretation Prompt")
            
            # 1. Wrap inputs in a form to prevent typing lag
            with st.form("llm_prompt_form"):
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    data_domain_options = [
                        "Immune", "Oncology", "Neuroscience", "Cardiovascular", 
                        "Metabolism & Endocrinology", "Infectious Disease", "Genetics & Genomics", 
                        "Developmental Biology", "Stem Cell Biology", "Microbiome", 
                        "Pharmacology", "Toxicology", "Biophysics", "Structural Biology", 
                        "Systems Biology", "Synthetic Biology", "Plant Biology", 
                        "Evolutionary Biology", "Ecology", "Other"
                    ]
                    selected_domain = st.selectbox("Data Domain", data_domain_options)
                    
                with export_col2:
                    selected_cell_type = st.text_input("Cell Type / Study Tissue", placeholder="e.g., Blood cell, Liver tumor...")

                # This button submits the form, preventing reruns on every keystroke
                generate_btn = st.form_submit_button("⚙️ Generate Prompt File")

            # 2. When the user clicks Generate, build the text and save to Session State
            if generate_btn:
                cell_type_display = selected_cell_type if selected_cell_type.strip() != "" else "Not specifically provided"
                
                ai_readable_text = (
                    f"--- SYSTEM PROMPT ---\n"
                    f"Data Domain Context: {selected_domain}\n"
                    f"Cell Type / Study Tissue: {cell_type_display}\n\n"
                    
                    f"You are an expert computational biologist tasked with interpreting decision tree classification rules "
                    f"that predict specific biomarkers based on RNA gene expression and gene attributes. Your objective is "
                    f"to translate the mathematical logic of the tree, including feature hierarchy, split thresholds, and "
                    f"conditional dependencies, into a cohesive biological narrative that explains the underlying regulatory "
                    f"mechanisms. Using the specific biological context provided, you must synthesize these findings into "
                    f"concrete, experimentally testable hypotheses. "
                    f"CRITICAL INSTRUCTION: Primarily analyze the FOCUSED ZONES section, prioritizing nodes with high Precision and True Positives (TP) "
                    f"to form your hypothesis based on the logic. Use the full raw decision tree only as supplementary context; do not "
                    f"attempt to parse every individual branch.\n\n"



                    f"--- EXPERIMENT CONTEXT & DATA PIPELINE ---\n"
                    f"1. Goal: Forming biological rationale and hypothesis based on the decision tree results to explain why {dt['target']} heavily attributes to the prediction of the present of {st.session_state.target_name}. Focused feature: '{dt['target']}'. Target response: '{st.session_state.target_name}'.\n"
                    f"2. Upstream Model: A PyTorch nonlinear Multi-layer Perception model was trained on the raw dataset.\n"
                    f"3. Feature Selection: Integrated Gradients (IG) was applied. The top {len(dt['features'])} features with the highest Mean Absolute IG Attribution were selected.\n"
                    f"4. Target response and Focus feature Correlation: {dt['corr']} correlation between {dt['target']} and {st.session_state.target_name}.\n"
                    f"5. Label Generation: Samples were labeled as Class 1 ('High focused Attribution') if their IG score for {dt['target']} fell in the extreme 10% according to the analysis type. All other samples are Class 0. Analysis type: the {dt['direction']} Attribution Analysis focusing on conditions of {dt['target']}'s {dt['direction']} attribution effect.\n"
                    f"6. Tree Parameters and classification: max_depth=4, min_samples_leaf=5., class_weight= {{0: 1, 1: {dt['weight']}}}. The decision tree aims to find the conditions that contain most Class 1 data points and mininal Class 0 data points.\n\n"
                    
                    f"--- Focused ZONES (CLASS 1 LEAF NODES) ---\n"
                    f"These are the specific logic zones that successfully isolated 'High Attribution' samples. Prioritize analyzing nodes has high Precision with large cell numbers. True Positive: the numbers of the data points labeled with Class 1. False Positive: the number of the data points labeled with Class 0.\n\n"
                    f"{leaf_context}\n\n"
                    
                    f"--- FULL DECISION TREE RAW RULES ---\n"
                    f"{tree_rules}\n"
                    
                    f"--- TASK ---\n"
                    f"Based on the TARGET ZONES above, forming rationale and hypothesis of the feature relationship (pathways) to explain HOW features integratively regulate {st.session_state.target_name} with {dt['target']}. Highlight which zones are the most biologically robust based on the highest number of True Positives and Precision. Identify the findings into two categories: 1. the findings directly confirm the literature knowledge. 2. suggesting new testable mechanisms supported by literature. List the publications supporting the conclusions. Next, look into the second good zone and repeat the analysis (the conclusions for two highlight zones will be presented separately)."
                )
                
                # Save it to Streamlit's memory so the download button doesn't vanish
                st.session_state.ai_prompt_text = ai_readable_text
                st.session_state.ai_prompt_filename = f"{dt['target']}-{st.session_state.target_name}_Interpretation_Prompt.txt"

            # 3. Show the Download button OUTSIDE the form if the text has been generated
            if "ai_prompt_text" in st.session_state:
                st.success("✅ Prompt successfully generated! Click below to download.")
                
                st.download_button(
                    label="📄 Download LLM Prompt (.txt)",
                    data=st.session_state.ai_prompt_text,
                    file_name=st.session_state.ai_prompt_filename,
                    mime="text/plain",
                    use_container_width=True
                )


            # --- UMAP Projection --- 
            st.divider()
            st.write("#### 🌌 UMAP Analysis")
            
            # --- UMAP FEATURE SELECTION TOGGLE ---
            # By default, we project UMAP using ONLY the features the Decision Tree used (dt['indices']).
            # UNCOMMENT the line below if you want UMAP to look at top 30 background features instead:
            # X_umap = st.session_state.X_raw[:, dt['indices']]
            
            # UNCOMMENT the line below if you want UMAP to look at ALL background features instead:
            X_umap = st.session_state.X_raw 
            
            embedding = get_cached_umap(X_umap)
            is_leaves = (t_.children_left == -1)
            positive_leaves = [i for i in range(t_.node_count) if is_leaves[i] and np.argmax(t_.value[i][0]) == 1]
            
            X_tree_subset = st.session_state.X_raw[:, dt['indices']]
            full_tree_paths = dt['clf'].apply(X_tree_subset)
            y_binary = dt['y_binary'] 

            umap_ctrl, umap_plot = st.columns([1, 2.5])
            

            with umap_ctrl:
                color_mode = st.radio("UMAP Coloring Mode", ["Color by Feature value", "Color by Classification Result"])
                
                if color_mode == "Color by Feature value":
                    selected_umap_feat = st.selectbox("Select Feature to visualize", st.session_state.headers)
                else:
                    if not positive_leaves:
                        st.warning("No purely positive leaf nodes found in this tree.")
                        selected_node = None
                    else:
                        node_options = {}
                        for idx, node_id in enumerate(positive_leaves):
                            in_this_leaf = (full_tree_paths == node_id)
                            tp = np.sum(in_this_leaf & (y_binary == 1))
                            fp = np.sum(in_this_leaf & (y_binary == 0))
                            
                            label = f"Condition {idx + 1} / Node {node_id} (✅ TP: {tp}, ❌ FP: {fp})"
                            node_options[label] = node_id
                            
                        selected_node_label = st.selectbox("Select a 'Good Result Box'", list(node_options.keys()))
                        selected_node = node_options[selected_node_label]
                        
                        if selected_node is not None:
                            rules = get_node_rules(t_, dt['features'], selected_node)
                            st.markdown("##### 📝 Condition Rules:")
                            if rules:
                                for rule in rules:
                                    st.markdown(f"- `{rule}`")
                            else:
                                st.markdown("- Root Node (No rules)")

            with umap_plot:
                fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
                
                if color_mode == "Color by Feature value":
                    feat_idx = np.where(st.session_state.headers == selected_umap_feat)[0][0]
                    c_values = st.session_state.X_raw[:, feat_idx]
                    v_cap = np.percentile(c_values, 95) 
                    
                    # --- NEW CUSTOM SEQUENTIAL COLORMAP ---
                    # Your hex codes (ordered Light Blue -> Dark Blue)
                    blues_hex_colors = [
                         "#deebf7", "#c6dbef", 
                        "#9ecae1", "#6baed6", "#4292c6", 
                        "#2171b5", "#08519c", "#08306b"
                    ]
                    blues_cmap = LinearSegmentedColormap.from_list("custom_blues", blues_hex_colors)
                    
                    # Apply the custom blues colormap here
                    scatter = ax_umap.scatter(
                        embedding[:, 0], embedding[:, 1], 
                        c=c_values, cmap=blues_cmap, 
                        s=8, alpha=0.7, edgecolors='none', 
                        vmin=np.min(c_values), vmax=v_cap
                    )
                    
                    cbar = plt.colorbar(scatter, ax=ax_umap)
                    cbar.set_label(f'Intensity of {selected_umap_feat}', rotation=270, labelpad=15)
                    ax_umap.set_title(f'UMAP Colored by {selected_umap_feat}')
                    
                elif color_mode == "Color by Classification Result" and positive_leaves:
                    in_selected_node = (full_tree_paths == selected_node)
                    is_tp = in_selected_node & (y_binary == 1)
                    is_fp = in_selected_node & (y_binary == 0)
                    
                    ax_umap.scatter(embedding[~in_selected_node, 0], embedding[~in_selected_node, 1], 
                                    c='lightgray', s=8, alpha=0.4, edgecolors='none', label='Other Zones')
                    
                    ax_umap.scatter(embedding[is_fp, 0], embedding[is_fp, 1], 
                                    c='orange', s=12, alpha=0.9, edgecolors='none', 
                                    label=f'False Positives (n={np.sum(is_fp)})')
                    
                    ax_umap.scatter(embedding[is_tp, 0], embedding[is_tp, 1], 
                                    c='green', s=15, alpha=1.0, edgecolors='black', linewidths=0.2, 
                                    label=f'True Positives (n={np.sum(is_tp)})')
                    
                    clean_title = selected_node_label.split(" (")[0]
                    ax_umap.set_title(f'UMAP Highlighting {clean_title}')
                    ax_umap.legend(loc="upper right", markerscale=1.5)

                ax_umap.set_xlabel('UMAP 1')
                ax_umap.set_ylabel('UMAP 2')
                ax_umap.spines['top'].set_visible(False)
                ax_umap.spines['right'].set_visible(False)
                st.pyplot(fig_umap)

                # --- ADD UMAP DOWNLOAD BUTTON HERE ---
                buf_umap = io.BytesIO()
                fig_umap.savefig(buf_umap, format="svg", bbox_inches="tight")
                if color_mode == "Color by Feature value":
                    st.download_button(
                        label="📥 Download UMAP Plot (SVG)",
                        data=buf_umap.getvalue(),
                        file_name=f"umap_feature_value_{selected_umap_feat}.svg",
                        mime="image/svg+xml"
                    )
                else:
                    st.download_button(
                        label="📥 Download UMAP Plot (SVG)",
                        data=buf_umap.getvalue(),
                        file_name=f"umap_nodes_{selected_node}.svg",
                        mime="image/svg+xml"
                    )

                # Assuming your UMAP or Swarm plot figure is named 'fig'

            # --- NETWORK PLOT ---
            st.divider()
            st.write("#### Feature Interaction Network")
            
            G = nx.DiGraph()
            
            def build_net(node):
                if t_.children_left[node] != -1:
                    p_feat = dt['features'][t_.feature[node]]
                    for child in [t_.children_left[node], t_.children_right[node]]:
                        if t_.children_left[child] != -1:
                            c_feat = dt['features'][t_.feature[child]]
                            if p_feat != c_feat:
                                w = int(t_.n_node_samples[child])
                                if G.has_edge(p_feat, c_feat): G[p_feat][c_feat]['weight'] += w
                                else: G.add_edge(p_feat, c_feat, weight=w)
                        build_net(child)
            build_net(0)

            if len(G.edges) > 0:
                fig_net = get_cached_network_plot(G)
                st.pyplot(fig_net)
                
                buf_net = io.BytesIO()
                fig_net.savefig(buf_net, format="svg", bbox_inches="tight")
                st.download_button(label="📥 Download Network Plot (SVG)", data=buf_net.getvalue(), file_name="interaction_network.svg",mime="image/svg+xml")


    else:
        st.warning("Please finish Step 2 first!")