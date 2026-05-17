import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import warnings
import io

warnings.filterwarnings('ignore')

# ── 페이지 설정 ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI 주가 기술분석 도구",
    page_icon="📈",
    layout="wide"
)

# ── 한글 폰트 ───────────────────────────────────────────────
@st.cache_resource
def load_font():
    try:
        import subprocess
        subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], capture_output=True)
        fm._load_fontmanager(try_read_cache=False)
        nanum = [f for f in fm.findSystemFonts() if 'NanumGothic' in f and 'Bold' not in f]
        if nanum:
            return fm.FontProperties(fname=nanum[0]).get_name()
    except:
        pass
    return 'DejaVu Sans'

font_name = load_font()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# ── 종목 사전 ───────────────────────────────────────────────
STOCK_DICT = {
    # 대형주
    '삼성전자':'005930.KS', 'SK하이닉스':'000660.KS',
    '현대차':'005380.KS',   '기아':'000270.KS',
    'LG에너지솔루션':'373220.KS', '삼성바이오로직스':'207940.KS',
    'NAVER':'035420.KS',    '카카오':'035720.KS',
    'POSCO홀딩스':'005490.KS', 'LG화학':'051910.KS',
    '삼성SDI':'006400.KS',  '현대모비스':'012330.KS',
    '삼성물산':'028260.KS', '한국전력':'015760.KS',
    '카카오뱅크':'323410.KS','크래프톤':'259960.KS',
    '두산에너빌리티':'034020.KS','엔씨소프트':'036570.KS',
    'KT':'030200.KS',       'SKT':'017670.KS',  'SK텔레콤':'017670.KS',
    'LG전자':'066570.KS',   '고려아연':'010130.KS',
    '한화에어로스페이스':'012450.KS', '한화':'000880.KS',
    '롯데쇼핑':'023530.KS', '이마트':'139480.KS',
    'CJ제일제당':'097950.KS', 'KT&G':'033780.KS',
    '대한항공':'003490.KS', 'HMM':'011200.KS',
    'HD현대':'267250.KS',   'SK이노베이션':'096770.KS',
    # 금융/증권
    'KB금융':'105560.KS',   '신한지주':'055550.KS',
    '하나금융지주':'086790.KS', '우리금융지주':'316140.KS',
    '미래에셋증권':'006800.KS', '삼성증권':'016360.KS',
    '키움증권':'039490.KS', 'NH투자증권':'005940.KS',
    '메리츠금융지주':'138040.KS', '삼성화재':'000810.KS',
    '현대해상':'001450.KS', 'DB손해보험':'005830.KS',
    '카카오페이':'377300.KS',
    # 바이오/제약
    '셀트리온':'068270.KS', '유한양행':'000100.KS',
    '한미약품':'128940.KS', '대웅제약':'069620.KS',
    '종근당':'185750.KS',   '녹십자':'006280.KS',
    # 미국
    '애플':'AAPL',    '테슬라':'TSLA',
    '엔비디아':'NVDA', '마이크로소프트':'MSFT',
    '구글':'GOOGL',   '알파벳':'GOOGL',
    '아마존':'AMZN',  '메타':'META',
    '넷플릭스':'NFLX','JP모건':'JPM',
    '비자':'V',       '화이자':'PFE',
    '엑슨모빌':'XOM',
}

@st.cache_data(ttl=86400)
def load_krx_tickers():
    """KRX 전체 종목명 → 티커 딕셔너리 (하루 1회 캐싱)"""
    import requests
    result = {}
    try:
        # KRX OpenAPI - 코스피
        for market in ['STK', 'KSQ']:
            url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
            body = {
                'bld': 'dbms/MDC/STAT/standard/MDCSTAT01901',
                'mktId': market,
                'trdDd': pd.Timestamp.now().strftime('%Y%m%d'),
                'money': '1',
                'csvxls_isNo': 'false',
            }
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'http://data.krx.co.kr/',
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            res = requests.post(url, data=body, headers=headers, timeout=10)
            data = res.json()
            suffix = '.KS' if market == 'STK' else '.KQ'
            for item in data.get('OutBlock_1', []):
                name = item.get('ISU_ABBRV', '').strip()
                code = item.get('ISU_SRT_CD', '').strip()
                if name and code:
                    result[name] = code + suffix
    except Exception as e:
        pass
    return result

def resolve_ticker(text):
    t = text.strip()
    # 1. 내장 딕셔너리
    if t in STOCK_DICT: return STOCK_DICT[t]
    for k, v in STOCK_DICT.items():
        if k.lower() == t.lower(): return v
    # 2. KRX 전체 종목 검색
    krx = load_krx_tickers()
    if t in krx: return krx[t]
    for k, v in krx.items():
        if k.lower() == t.lower(): return v
    # 부분 매칭
    matches = [(k,v) for k,v in krx.items() if t in k]
    if len(matches) == 1: return matches[0][1]
    # 3. 6자리 숫자
    if t.isdigit() and len(t) == 6: return t + '.KS'
    # 4. 티커 형식
    if t.upper().endswith('.KS') or t.upper().endswith('.KQ'): return t.upper()
    if t.replace('-','').isalpha(): return t.upper()
    return t

# ── 지표 계산 ───────────────────────────────────────────────
def calc_bb(close, period=20, mult=2):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    return ma, ma + mult*std, ma - mult*std

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain/loss))

def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

# ── 차트 ────────────────────────────────────────────────────
def draw_chart(df, currency):
    close = df['Close']
    ma20, bb_up, bb_dn = calc_bb(close)
    rsi = calc_rsi(close)
    macd, sig_line, hist = calc_macd(close)

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(4, 1, height_ratios=[3,1,1,1], hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor('#0f1117')
        ax.tick_params(colors='#888', labelsize=8)
        for sp in ax.spines.values(): sp.set_color('#2a2a3a')
        ax.grid(color='#1e1e2e', linewidth=0.5)
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

    x = range(len(df))
    ts = df.index
    tp = list(range(0, len(df), max(1, len(df)//8)))
    tl = [ts[i].strftime('%m/%d') for i in tp]

    # 주가 + 볼린저
    ax1 = axes[0]
    ax1.fill_between(x, bb_up, bb_dn, alpha=0.07, color='#4a9eff')
    ax1.plot(x, bb_up, color='#4a9eff', lw=0.8, ls='--', label='볼린저 상단')
    ax1.plot(x, ma20,  color='#f0a500', lw=1.2, label='MA20')
    ax1.plot(x, bb_dn, color='#4a9eff', lw=0.8, ls='--', label='볼린저 하단')
    ax1.plot(x, close, color='#e0e0e0', lw=1.8, label='종가')
    for i, (idx, row) in enumerate(df.iterrows()):
        c = '#ef5350' if row['Close'] >= row['Open'] else '#26a69a'
        ax1.plot([i,i], [row['Low'], row['High']], color=c, lw=0.5, alpha=0.5)
        ax1.add_patch(plt.Rectangle(
            (i-0.3, min(row['Open'], row['Close'])),
            0.6, abs(row['Close']-row['Open']), color=c, alpha=0.65))
    ax1.set_ylabel('가격', color='#888', fontsize=8)
    ax1.set_xticks(tp); ax1.set_xticklabels([])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v,_: f'{v:,.0f}' if currency=='KRW' else f'{v:.1f}'))

    # 거래량
    ax2 = axes[1]
    vc = ['#ef5350' if df['Close'].iloc[i]>=df['Open'].iloc[i] else '#26a69a' for i in range(len(df))]
    ax2.bar(x, df['Volume'], color=vc, alpha=0.7, width=0.8)
    ax2.set_ylabel('거래량', color='#888', fontsize=8)
    ax2.set_xticks(tp); ax2.set_xticklabels([])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v,_: f'{v/1e6:.0f}M' if v>=1e6 else f'{v/1e3:.0f}K'))

    # RSI
    ax3 = axes[2]
    ax3.plot(x, rsi, color='#ab47bc', lw=1.5)
    ax3.axhline(70, color='#ef5350', lw=0.8, ls='--', alpha=0.7)
    ax3.axhline(30, color='#26a69a', lw=0.8, ls='--', alpha=0.7)
    ax3.fill_between(x, rsi, 70, where=(rsi>70), alpha=0.2, color='#ef5350')
    ax3.fill_between(x, rsi, 30, where=(rsi<30), alpha=0.2, color='#26a69a')
    ax3.set_ylim(0, 100)
    ax3.set_ylabel(f'RSI {rsi.iloc[-1]:.1f}', color='#ab47bc', fontsize=8)
    ax3.set_xticks(tp); ax3.set_xticklabels([])

    # MACD
    ax4 = axes[3]
    hc = ['#ef5350' if v>=0 else '#26a69a' for v in hist]
    ax4.bar(x, hist, color=hc, alpha=0.7, width=0.8)
    ax4.plot(x, macd,     color='#4a9eff', lw=1.2, label='MACD')
    ax4.plot(x, sig_line, color='#ff7043', lw=1.2, label='시그널')
    ax4.axhline(0, color='#444', lw=0.5)
    ax4.legend(loc='upper left', fontsize=7, facecolor='#1a1a2e', labelcolor='#ccc', framealpha=0.7)
    ax4.set_ylabel('MACD', color='#888', fontsize=8)
    ax4.set_xticks(tp); ax4.set_xticklabels(tl)

    plt.tight_layout()
    return fig, ma20, bb_up, bb_dn, rsi, macd, sig_line

# ── UI ──────────────────────────────────────────────────────
st.title("📈 AI 주가 기술분석 도구")
st.caption("폴리텍대학 생성형AI 활용 특강")

col1, col2, col3 = st.columns([3, 1.5, 1])
with col1:
    stock_input = st.text_input(
        "종목명 또는 코드",
        placeholder="예: 삼성전자  /  애플  /  TSLA  /  005930",
        label_visibility="collapsed"
    )
with col2:
    period = st.selectbox("기간", ['1개월','3개월','6개월','1년'],
                          index=1, label_visibility="collapsed")
with col3:
    analyze_btn = st.button("▶ 분석하기", type="primary", use_container_width=True)

period_map = {'1개월':'1mo','3개월':'3mo','6개월':'6mo','1년':'1y'}

st.caption("예: 삼성전자 · SK하이닉스 · 현대차 · 카카오 · NAVER · 애플 · 테슬라 · 엔비디아")

if analyze_btn and stock_input:
    ticker = resolve_ticker(stock_input)
    with st.spinner(f'📡 {stock_input} → {ticker} 데이터 불러오는 중...'):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period_map[period])
            if df.empty:
                st.error(f'종목을 찾을 수 없습니다. 입력: {stock_input} → 티커: {ticker}')
                st.stop()
            info = {}
            try: info = tk.info
            except: pass

            name = info.get('longName') or info.get('shortName') or stock_input
            currency = info.get('currency', 'KRW')
            fmt = (lambda v: f'₩{v:,.0f}') if currency=='KRW' else (lambda v: f'${v:.2f}')

            df = df[['Open','High','Low','Close','Volume']].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            close = df['Close']

            last_c = close.iloc[-1]
            prev_c = close.iloc[-2]
            chg = last_c - prev_c
            chg_pct = chg / prev_c * 100
            arrow = '▲' if chg >= 0 else '▼'

        except Exception as e:
            st.error(f'오류: {e}')
            st.stop()

    # 현황
    st.subheader(name)
    st.caption(f'{ticker} · {df.index[0].strftime("%Y-%m-%d")} ~ {df.index[-1].strftime("%Y-%m-%d")}')

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("현재가", fmt(last_c))
    m2.metric("전일 대비", fmt(abs(chg)), f"{arrow} {abs(chg_pct):.2f}%",
              delta_color="normal" if chg >= 0 else "inverse")

    fig, ma20, bb_up, bb_dn, rsi, macd, sig_line = draw_chart(df, currency)

    last_up  = bb_up.iloc[-1]
    last_mid = ma20.iloc[-1]
    last_dn  = bb_dn.iloc[-1]
    last_rsi = rsi.iloc[-1]
    last_mac = macd.iloc[-1]
    last_sig = sig_line.iloc[-1]

    m3.metric("볼린저 상단", fmt(last_up))
    m4.metric("MA20 (중간선)", fmt(last_mid))
    m5.metric("볼린저 하단", fmt(last_dn))
    m6.metric("RSI(14)", f"{last_rsi:.1f}",
              "과매수" if last_rsi>70 else "과매도" if last_rsi<30 else "중립")

    # 차트
    st.pyplot(fig)
    plt.close()

    # 신호
    st.markdown("**🔍 기술분석 신호**")
    sig_cols = st.columns(3)
    if last_c > last_up:
        sig_cols[0].error("볼린저 상단 돌파 — 과매수")
    elif last_c < last_dn:
        sig_cols[0].success("볼린저 하단 이탈 — 과매도")
    elif last_c > last_mid:
        sig_cols[0].info("볼린저 중간선 위 — 상승 추세")
    else:
        sig_cols[0].warning("볼린저 중간선 아래 — 하락 추세")

    if last_rsi >= 70:
        sig_cols[1].error(f"RSI {last_rsi:.1f} — 과매수")
    elif last_rsi <= 30:
        sig_cols[1].success(f"RSI {last_rsi:.1f} — 과매도")
    else:
        sig_cols[1].info(f"RSI {last_rsi:.1f} — 중립")

    if last_mac > last_sig:
        sig_cols[2].success("MACD 골든크로스 — 상승 신호")
    else:
        sig_cols[2].error("MACD 데드크로스 — 하락 신호")

    st.caption("⚠️ 기술적 분석은 참고용입니다. 투자 결정은 본인 판단으로!")

    # 엑셀 다운로드
    st.divider()
    df_ex = df.copy()
    df_ex['MA20']       = ma20.round(2)
    df_ex['볼린저상단'] = bb_up.round(2)
    df_ex['볼린저하단'] = bb_dn.round(2)
    df_ex['RSI']        = rsi.round(2)
    df_ex['MACD']       = macd.round(2)
    df_ex['MACD시그널'] = sig_line.round(2)
    df_ex.index = df_ex.index.strftime('%Y-%m-%d')
    buf = io.BytesIO()
    df_ex.to_excel(buf, index=True)
    st.download_button("💾 엑셀 다운로드", buf.getvalue(),
                       file_name=f"{ticker.replace('.','_')}_{period}.xlsx",
                       mime="application/vnd.ms-excel")

    # 프롬프트
    st.divider()
    st.markdown("**💬 GPT / Claude 프롬프트 자동 생성**")
    rsi_label = '(과매수)' if last_rsi>70 else '(과매도)' if last_rsi<30 else '(중립)'
    macd_label = '골든크로스(상승)' if last_mac>last_sig else '데드크로스(하락)'
    data_str = (
        f"[{name} ({ticker}) 기술분석 데이터]\n"
        f"- 현재가: {fmt(last_c)} (전일 대비 {chg_pct:+.2f}%)\n"
        f"- 볼린저밴드  상단: {fmt(last_up)} / 중간(MA20): {fmt(last_mid)} / 하단: {fmt(last_dn)}\n"
        f"- RSI(14): {last_rsi:.1f} {rsi_label}\n"
        f"- MACD: {last_mac:.2f} / 시그널: {last_sig:.2f} → {macd_label}"
    )
    prompts = {
        "📖 초보자 쉬운 설명": (
            "당신은 친절한 주식 입문 강사입니다. 주식을 처음 접하는 60대 분들도 이해할 수 있게 쉽게 설명해주세요.\n\n"
            + data_str +
            "\n\n위 데이터를 바탕으로:\n"
            "1. 볼린저밴드가 무엇인지 한 줄 설명 (비유 포함)\n"
            "2. RSI가 무엇인지 한 줄 설명 (비유 포함)\n"
            "3. 지금 이 주식 상태를 날씨에 비유해서 설명\n"
            "4. 주식 초보자가 주의할 점 3가지"
        ),
        "⚡ 단기 투자 전략": (
            "당신은 15년 경력 기술적 분석 전문가입니다.\n\n"
            + data_str +
            "\n\n단기(1~4주) 투자 관점 분석:\n"
            "1. 현재 기술적 신호 종합 해석\n"
            "2. 시나리오 3가지 (공격적 / 중립 / 보수적)\n"
            "3. 매수 고려 시 주목할 가격대\n"
            "4. 손절 기준 제안\n"
            "5. 한줄 결론"
        ),
        "⚠️ 리스크 분석": (
            "당신은 리스크 관리 전문 애널리스트입니다.\n\n"
            + data_str +
            "\n\n리스크 분석:\n"
            "1. 기술적 위험 신호\n"
            "2. 과매수/과매도 판단 및 의미\n"
            "3. 볼린저밴드 변동성 위험\n"
            "4. 보수적 투자자를 위한 조언\n"
            "5. 반드시 알아야 할 리스크 3가지"
        ),
        "📋 종합 투자 보고서": (
            "당신은 증권사 리서치센터 수석 애널리스트입니다.\n\n"
            + data_str +
            f"\n\n## {name} 투자 분석 보고서\n"
            "### 1. 현황 요약\n"
            "### 2. 기술적 분석 (볼린저밴드 / RSI / MACD)\n"
            "### 3. 투자 시나리오 (낙관 / 기본 / 비관)\n"
            "### 4. 투자 의견\n"
            "### 5. 리스크 요인"
        ),
    }
    ptype = st.selectbox("프롬프트 유형", list(prompts.keys()), label_visibility="collapsed")
    st.code(prompts[ptype], language=None)
    st.caption("👆 위 텍스트를 복사해서 ChatGPT 또는 Claude에 붙여넣으세요!")
