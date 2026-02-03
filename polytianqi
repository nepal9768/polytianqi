import streamlit as st
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# --- 配置页面 ---
st.set_page_config(page_title="全球气象监控", layout="wide")

# --- 配色方案 (适配 Streamlit) ---
UI_COLOR = {
    "accent": "#3498db",
    "red": "#e74c3c",
    "green": "#2ecc71",
    "source2": "#d35400",
    "source3": "#8e44ad"
}

# --- 站点列表 ---
STATIONS = {
    "西雅图 (SEA)": ["47.45,-122.30", "America/Los_Angeles"],
    "达拉斯 (DAL)": ["32.84,-96.85", "America/Chicago"],
    "芝加哥 (ORD)": ["41.89,-87.94", "America/Chicago"],
    "多伦多 (YYZ)": ["43.59,-79.64", "America/Toronto"],
    "亚特兰大 (ATL)": ["33.64,-84.42", "America/New_York"],
    "迈阿密 (MIA)": ["25.88,-80.16", "America/New_York"],
    "布宜诺斯艾利斯": ["-34.82,-58.53", "America/Argentina/Buenos_Aires"],
    "伦敦 (LHR)": ["51.51,0.04", "Europe/London"],
    "土耳其安卡拉": ["40.24,33.03", "Europe/Istanbul"],
    "新西兰惠灵顿": ["-41.28,174.77", "Pacific/Auckland"],
    "纽约拉瓜迪亚": ["40.77,-73.87", "America/New_York"],
    "韩国仁川": ["37.46,126.44", "Asia/Seoul"]
}

# --- 侧边栏控制 ---
st.sidebar.title("设置")
selected_city = st.sidebar.selectbox("选择站点", list(STATIONS.keys()))
unit = st.sidebar.radio("温标", ["°C", "°F"], horizontal=True)

# 模拟本地存储 (Session State)
if 'h_off' not in st.session_state: st.session_state.h_off = 0
if 'm_off' not in st.session_state: st.session_state.m_off = 0

st.sidebar.subheader("本地时间调整")
col1, col2 = st.sidebar.columns(2)
h_off = col1.number_input("小时", value=st.session_state.h_off, step=1)
m_off = col2.number_input("分钟", value=st.session_state.m_off, step=1)

# --- 主逻辑 ---
coords, timezone_str = STATIONS[selected_city]
lat, lon = coords.split(',')

# 获取数据函数
@st.cache_data(ttl=600)  # 缓存10分钟，避免频繁请求
def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m&models=best_match,gfs_seamless,icon_seamless&timezone=auto"
    try:
        r = requests.get(url, timeout=10).json()
        hourly = r.get('hourly', {})
        return {
            "main": hourly.get('temperature_2m', [])[:25],
            "gfs": hourly.get('temperature_2m_gfs_seamless', [])[:25],
            "icon": hourly.get('temperature_2m_icon_seamless', [])[:25]
        }
    except:
        return None

data = get_weather_data(lat, lon)

if data:
    # --- 数据处理 ---
    xs = np.linspace(0, 24, 300)
    city_tz = pytz.timezone(timezone_str)
    now = datetime.now(city_tz)
    now_h = now.hour + now.minute/60.0

    st.title(f"{selected_city}")
    st.caption(f"当地时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # 转换温标
    def convert(vals):
        arr = np.array(vals)
        if len(arr) == 0: return np.array([])
        return arr * 9/5 + 32 if unit == "°F" else arr

    y_main = convert(data['main'])
    y_gfs = convert(data['gfs']) if data['gfs'] else y_main
    y_icon = convert(data['icon']) if data['icon'] else y_main

    # 平滑处理
    def smooth(y_raw):
        if len(y_raw) == 0: return np.zeros_like(xs)
        x_raw = np.arange(len(y_raw))
        return make_interp_spline(x_raw, y_raw, k=3)(xs)

    ys_main = smooth(y_main)
    ys_gfs = smooth(y_gfs)
    ys_icon = smooth(y_icon)

    # 计算极值
    max_idx = np.argmax(ys_main)
    min_idx = np.argmin(ys_main)
    max_val = ys_main[max_idx]
    min_val = ys_main[min_idx]
    max_time_h = xs[max_idx]

    # --- 显示指标 ---
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("最高温 (Main)", f"{max_val:.1f}{unit}")
    col_b.metric("最低温 (Main)", f"{min_val:.1f}{unit}")
    
    # 峰值时间计算
    peak_h_int = int(max_time_h)
    peak_m_int = int((max_time_h % 1) * 60)
    peak_dt = now.replace(hour=peak_h_int, minute=peak_m_int, second=0)
    local_peak_dt = peak_dt + timedelta(hours=h_off, minutes=m_off)
    col_c.metric("本地预估峰值", local_peak_dt.strftime('%H:%M'))

    # --- 绘图 (Matplotlib) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('#f0f2f6')

    # 绘制三条线
    ax.plot(xs, ys_icon, color=UI_COLOR["source3"], linestyle="-.", label="ICON (DE)", alpha=0.7)
    ax.plot(xs, ys_gfs, color=UI_COLOR["source2"], linestyle="--", label="GFS (US)", alpha=0.7)
    
    # 主线 (Main) - 过去实线，未来虚线
    ax.plot(xs[xs <= now_h], ys_main[xs <= now_h], color=UI_COLOR["accent"], linewidth=3, label="Main")
    ax.plot(xs[xs >= now_h], ys_main[xs >= now_h], color=UI_COLOR["accent"], linewidth=3, linestyle=":", alpha=0.5)

    # 标记当前时间
    ax.axvline(now_h, color="black", alpha=0.3)
    
    # 标记最高/最低点
    ax.scatter([xs[max_idx]], [max_val], color=UI_COLOR["red"], zorder=5)
    ax.scatter([xs[min_idx]], [min_val], color=UI_COLOR["green"], zorder=5)

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 4)])
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    
    # 去除边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    st.pyplot(fig)

    st.info("💡 提示：这是一个网页应用，您可以将其添加到 iPhone 主屏幕以全屏使用。")

else:
    st.error("无法获取数据，请检查网络。")
