import os
import json
import numpy as np
import tensorflow as tf

def convert_h5_to_numpy(model_name='fashion_mnist'):
    """
    將 Keras .h5 模型轉換為 numpy 格式 (.npz 和 .json)
    """
    TF_MODEL_PATH = f'{model_name}.h5'
    MODEL_WEIGHTS_PATH = f'model/{model_name}.npz'
    MODEL_ARCH_PATH = f'model/{model_name}.json'
    
    # 檢查 .h5 檔案是否存在
    if not os.path.exists(TF_MODEL_PATH):
        print(f"❌ 錯誤：找不到模型檔案 '{TF_MODEL_PATH}'")
        print(f"請確保 {TF_MODEL_PATH} 檔案存在於當前目錄中")
        return False
    
    # 建立 model 資料夾（如果不存在）
    os.makedirs('model', exist_ok=True)
    
    try:
        # === Step 1: 載入 Keras .h5 模型 ===
        print(f"🔄 正在載入模型: {TF_MODEL_PATH}")
        model = tf.keras.models.load_model(TF_MODEL_PATH)
        print("✅ 模型載入成功")
        
        # === Step 2: 提取和收集權重 ===
        params = {}
        print("\n🔍 正在從模型中提取權重...\n")
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"層級: {layer.name}")
                for i, w in enumerate(weights):
                    param_name = f"{layer.name}_{i}"
                    print(f"  {param_name}: shape={w.shape}")
                    params[param_name] = w
                print()
        
        # === Step 3: 儲存為 .npz ===
        np.savez(MODEL_WEIGHTS_PATH, **params)
        print(f"✅ 所有權重已儲存至 {MODEL_WEIGHTS_PATH}")
        
        # === Step 4: 重新載入並驗證 ===
        print("\n🔁 驗證載入的 .npz 權重...\n")
        loaded = np.load(MODEL_WEIGHTS_PATH)
        
        for key in loaded.files:
            print(f"{key}: shape={loaded[key].shape}")
        
        # === Step 5: 提取架構為 JSON ===
        arch = []
        for layer in model.layers:
            config = layer.get_config()
            info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "config": config,
                "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
            }
            arch.append(info)
        
        with open(MODEL_ARCH_PATH, "w") as f:
            json.dump(arch, f, indent=2)
        
        print(f"✅ 模型架構已儲存至 {MODEL_ARCH_PATH}")
        print("\n🎉 轉換完成！")
        return True
        
    except Exception as e:
        print(f"❌ 轉換過程中發生錯誤: {e}")
        return False

if __name__ == "__main__":
    # 檢查是否有 .h5 檔案
    h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not h5_files:
        print("❌ 當前目錄中找不到任何 .h5 檔案")
        print("請確保您有已訓練的 Keras 模型檔案（例如 fashion_mnist.h5）")
        print("\n如果您需要訓練模型，請先執行模型訓練程式")
    else:
        print(f"📁 找到以下 .h5 檔案: {h5_files}")
        model_name = input(f"請輸入要轉換的模型名稱（不含副檔名，預設為 'fashion_mnist'）: ").strip()
        
        if not model_name:
            model_name = 'fashion_mnist'
            
        convert_h5_to_numpy(model_name)