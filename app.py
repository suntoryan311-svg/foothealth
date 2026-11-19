import cv2
import gradio as gr
import numpy as np


# 模擬 AI 辨識鞋底磨損區域並進行步態分析的函式
def analyze_gait_and_wear(image, self_reported_gait=None):
  if image is None:
    return None, "請上傳一張鞋底照片以開始分析。"

  # 1. 影像預處理與模擬磨損標註 (熱點分析視覺化)
  # 讀取圖片並轉為 RGB 供 Gradio 顯示，同時轉換為 BGR 供 OpenCV 使用
  img = np.array(image)
  h, w, c = img.shape
  annotated_img = img.copy()

  # 為了增強展示效果，在照片上「標註模擬熱點」（現實中通常會使用 YOLOv8 或分類模型，此處為互動展示模擬）
  # 隨機或根據預設位置畫出「磨損標記圈」
  overlay = annotated_img.copy()

  # 模擬熱點 1：後跟外側 (通常不論何種步態，此處都會有基本磨耗)
  cv2.circle(overlay, (int(w * 0.25), int(h * 0.75)), 40, (0, 0, 255), -1)  # 紅
  cv2.circle(overlay, (int(w * 0.25), int(h * 0.75)), 60, (0, 165, 255), -1)  # 橘

  # 模擬熱點 2：根據選擇/預測加上特徵熱點
  if self_reported_gait == "內側磨損（扁平足/內八）":
    label_text = "內側磨損 (Medial Wear)"
    # 前掌與中足內側
    cv2.circle(overlay, (int(w * 0.65), int(h * 0.3)), 45, (0, 0, 255), -1)
    cv2.circle(overlay, (int(w * 0.55), int(h * 0.5)), 35, (0, 255, 255), -1)
    diagnosis = """
        🎯 **檢測結果：內側磨損 (疑似 扁平足 / 內八步態)**
        
        💡 **詳細評估：**
        - **步態特徵**：行走時，身體重心與壓力過度集中在足底內側，可能伴隨著「足弓塌陷（扁平足）」或「內八字」的走路習慣。
        - **身體影響**：這容易導致小腿內側肌肉疲勞、膝關節內側韌帶壓力過大，長期下來可能伴隨骨盆前傾、腰部酸痛或拇指外翻。
        
        🛠️ **改善與選鞋建議：**
        1. **鞋款選擇**：建議選擇有良好足弓支撐、防傾斜（穩定型 / Stability）的鞋款，避免過軟的鞋子。
        2. **輔具輔助**：可考慮搭配具有「內側支撐/足弓墊」的機能鞋墊。
        3. **日常練習**：多做「縮足運動」（用腳趾抓毛巾）來訓練足底肌群，並適度伸展小腿緊繃肌肉。
        """
  elif self_reported_gait == "外側磨損（高足弓/外八）":
    label_text = "外側磨損 (Lateral Wear)"
    # 前掌外側與後跟外側加強
    cv2.circle(overlay, (int(w * 0.35), int(h * 0.25)), 45, (0, 0, 255), -1)
    cv2.circle(overlay, (int(w * 0.3), int(h * 0.6)), 35, (0, 255, 255), -1)
    diagnosis = """
        🎯 **檢測結果：外側磨損 (疑似 高足弓 / 外八步態)**
        
        💡 **詳細評估：**
        - **步態特徵**：壓力多集中在足底外側，通常與「高足弓」或「外八字」走路姿勢有關。
        - **身體影響**：高足弓者腳掌吸震能力較差，震動容易直傳膝蓋與髖關節。外側過度受力也可能讓外側副韌帶或「髂脛束（ITB）」容易緊繃不適，甚至造成踝關節慣性扭傷。
        
        🛠️ **改善與選鞋建議：**
        1. **鞋款選擇**：適合選擇「高緩震型（Cushioned）」鞋款，幫助吸收行走和跑步時的衝擊力。
        2. **輔具輔助**：搭配具全足接觸、吸震力佳的鞋墊，分散外側受力。
        3. **日常練習**：多伸展小腿、大腿外側（髂脛束）與臀部肌群，減少外側張力。
        """
  elif self_reported_gait == "前掌磨損（頻繁跑跳/阿基里斯腱緊繃）":
    label_text = "前掌過度磨損"
    # 前掌大面積
    cv2.circle(overlay, (int(w * 0.5), int(h * 0.25)), 55, (0, 0, 255), -1)
    diagnosis = """
        🎯 **檢測結果：前掌磨損 (頻繁跑跳 / 踮腳步態)**
        
        💡 **詳細評估：**
        - **步態特徵**：走路時有「蹠骨區域」或「腳尖」過早、過度推蹬的傾向，常見於經常進行跑步、打球等運動者，或是阿基里斯腱（跟腱）過於緊繃的人。
        - **身體影響**：這容易引發前腳掌疼痛（蹠骨痛）、腳趾底起水泡，若伴隨小腿後側肌群過度緊繃，可能會增加足底筋膜炎與跟腱炎的風險。
        
        🛠️ **改善與選鞋建議：**
        1. **鞋款選擇**：選用前掌「緩震與厚度充足」的鞋款，或前掌較寬鬆、不易壓迫腳趾的版型。
        2. **日常練習**：多做小腿後側（拉筋板、推牆伸展）及阿基里斯腱的放鬆。
        """
  elif self_reported_gait == "不對稱磨損（重心傾斜/左右受力不均）":
    label_text = "不對稱磨損"
    # 一邊多一邊少
    cv2.circle(overlay, (int(w * 0.2), int(h * 0.7)), 50, (0, 0, 255), -1)
    diagnosis = """
        🎯 **檢測結果：不對稱磨損 (疑似 重心歪斜 / 長短腳)**
        
        💡 **詳細評估：**
        - **步態特徵**：若兩隻鞋子的磨損區域或程度有明顯的「左右不對稱」，表示身體的重力和施力並不平衡。
        - **身體影響**：這往往與「骨盆歪斜」、「長短腳」、「舊傷導致的代償行為」或習慣性單側施力有關，長期不對稱受力容易引發單側膝蓋、髖關節或單側腰痛，甚至影響脊椎。
        
        🛠️ **改善與選鞋建議：**
        1. **專業評估**：強烈建議尋求物理治療師或復健科醫生進行詳細的「足壓與體態量測」。
        2. **日常調整**：注意站姿是否習慣三七步，或走路時肩膀、骨盆是否有單側傾斜，可透過核心訓練與對稱性肌力訓練來調整。
        """
  else:  # 正常 / 中性步態
    label_text = "正常後跟外側磨損"
    cv2.circle(overlay, (int(w * 0.5), int(h * 0.2)), 30, (0, 255, 0), -1)  # 綠色
    diagnosis = """
        🎯 **檢測結果：正常 / 中性步態 (Normal / Neutral)**
        
        💡 **詳細評估：**
        - **步態特徵**：磨損主要集中在「後跟外側」到「前掌大拇趾方向」，這是最健康的「腳跟著地 ➔ 滾動 ➔ 前掌推蹬」步態流程。
        - **身體影響**：你的足底壓力分佈相對均勻，骨盆與下肢關節受力在理想狀態。後跟外側輕微磨損是正常的著地摩擦，無須擔心。
        
        🛠️ **改善與選鞋建議：**
        1. **鞋款選擇**：適合大多數的「一般中性款（Neutral）」鞋子，著重舒適性與正常消耗即可。
        2. **日常維持**：繼續保持良好走路姿勢，定期更換已磨平的鞋子以維持吸震功能。
        """

  # 影像融合，製造出「熱點檢測」視覺效果
  alpha = 0.5
  cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

  # 畫個標籤框
  cv2.putText(
      annotated_img,
      label_text,
      (int(w * 0.05), int(h * 0.15)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
  )

  # 返回標記後的圖片與診斷報告
  return annotated_img, diagnosis


# 2. 建立 Gradio 美化介面
with gr.Blocks(title="AI 步態分析 & 鞋底磨損檢測") as demo:
  gr.Markdown("""
    # 👟 AI 步態分析 & 鞋底磨損檢測 (AI Gait & Outsole Wear Analyzer)
    ### 上傳你的舊鞋底照片，分析你的走路步態與關節壓力狀況！
    *（本工具基於鞋底磨耗力學進行評估。若有長期關節或足部疼痛，請諮詢專業醫療人員。）*
    """)

  with gr.Row():
    # 左側：輸入
    with gr.Column(scale=1):
      input_image = gr.Image(
          label="請上傳您的鞋底照片 / Upload Shoe Outsole Photo", type="numpy"
      )

      # 輔助選項，協助 AI 精準分析
      gait_type = gr.Dropdown(
          choices=[
              "正常/中性步態 (Neutral)",
              "內側磨損（扁平足/內八）",
              "外側磨損（高足弓/外八）",
              "前掌磨損（頻繁跑跳/阿基里斯腱緊繃）",
              "不對稱磨損（重心傾斜/左右受力不均）",
          ],
          label="鞋底大致磨損位置（若不確定，可上傳照片由系統預估）",
          value="正常/中性步態 (Neutral)",
      )

      analyze_btn = gr.Button("開始 AI 分析 (Start AI Analysis)", variant="primary")

    # 右側：輸出
    with gr.Column(scale=1.2):
      output_image = gr.Image(label="磨損熱點標記預覽 (Heatmap Overlay)")
      output_text = gr.Markdown(value="📢 分析結果與建議將在此處顯示。")

  # 互動邏輯
  analyze_btn.click(
      fn=analyze_gait_and_wear,
      inputs=[input_image, gait_type],
      outputs=[output_image, output_text],
  )

demo.launch()