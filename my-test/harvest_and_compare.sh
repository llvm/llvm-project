#!/usr/bin/env bash
# Stage 1: 掃 llvm-test-suite 的 .c/.cpp，轉成 .ll（只掃 SingleSource）
# Stage 2: 對產生的 .ll 跑 llc，比較 greedy vs segtre，產出 summary.csv

set -euo pipefail

# ====== 可調參數 ======
TEST_ROOT="${TEST_ROOT:-$HOME/work/llvm-project/llvm-test-suite}"
# 只掃這些子樹；需要擴大就加進來（注意 MultiSource 多檔會失敗）
SUBDIRS=( "SingleSource" )
CLANG="${CLANG:-$HOME/work/llvm-project/build/bin/clang}"
LLC="${LLC:-$HOME/work/llvm-project/build/bin/llc}"
TRIPLE="${TRIPLE:-aarch64-unknown-linux-gnu}"
OLEVEL="${OLEVEL:-O3}"
JOBS="${JOBS:-$(nproc)}"
THRESHOLD="${THRESHOLD:-2.0}"   # segtre/greedy 超過就標示為慢

WORKDIR="${WORKDIR:-$PWD/testsuite_scan}"
IRDIR="$WORKDIR/ir"
OUTDIR="$WORKDIR/results"
LOGDIR="$OUTDIR/logs"
ASMDIR="$OUTDIR/asm"
CSV="$WORKDIR/summary.csv"

mkdir -p "$IRDIR" "$LOGDIR/greedy" "$LOGDIR/segtre" "$ASMDIR/greedy" "$ASMDIR/segtre"

echo "[1/3] 掃描與產生 IR (.ll) …"

# 收集來源清單（限定 SingleSource；排除 External 與 MultiSource）
mapfile -t SRC_LIST < <(
  for s in "${SUBDIRS[@]}"; do
    find "$TEST_ROOT/$s" -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.cc" \) \
      ! -path "*/External/*" \
      ! -path "*/MultiSource/*"
  done | sort
)

if (( ${#SRC_LIST[@]} == 0 )); then
  echo "沒有找到任何 .c/.cpp；請確認 TEST_ROOT 路徑：$TEST_ROOT"
  exit 1
fi

# 轉 IR（保留相對路徑結構）
gen_one_ll() {
  local src="$1"
  local rel="${src#"$TEST_ROOT/"}"
  local base="${rel%.*}"
  local out="$IRDIR/$base.ll"
  mkdir -p "$(dirname "$out")"
  # 大多數 SingleSource 可用這組旗標；若個別檔案需要額外巨集/旗標，本法不保證 100% 成功
  "$CLANG" "-$OLEVEL" -S -emit-llvm -target "$TRIPLE" "$src" -o "$out" \
    >/dev/null 2>"$out.err" || { echo "IR 失敗: $src" >&2; rm -f "$out"; }
}
export -f gen_one_ll
export CLANG OLEVEL TRIPLE TEST_ROOT IRDIR

printf "%s\n" "${SRC_LIST[@]}" | xargs -I{} -P "$JOBS" bash -c 'gen_one_ll "$@"' _ {}

# 收集所有成功的 .ll
mapfile -t LL_LIST < <(find "$IRDIR" -type f -name "*.ll" | sort)
echo "IR 產生成功：${#LL_LIST[@]} 檔"

echo "[2/3] llc 比較 greedy vs segtre …"
echo "File,GreedyReal(s),SegtreReal(s),Ratio(segtre/greedy)" > "$CSV"

time_llc() {
  local ra="$1"; shift
  local in="$1"; shift
  local s_out="$1"; shift
  local log="$1"; shift
  /usr/bin/time -p "$LLC" "-$OLEVEL" -mtriple="$TRIPLE" -regalloc="$ra" "$in" -o "$s_out" \
    1>/dev/null 2>"$log"
  awk '/^real /{print $2}' "$log" | tail -n1
}

for ll in "${LL_LIST[@]}"; do
  rel="${ll#"$IRDIR/"}"; name="${rel%.ll}"

  g_log="$LOGDIR/greedy/${name//\//_}.time"
  g_s="$ASMDIR/greedy/${name//\//_}.s"
  s_log="$LOGDIR/segtre/${name//\//_}.time"
  s_s="$ASMDIR/segtre/${name//\//_}.s"

  mkdir -p "$(dirname "$g_log")" "$(dirname "$s_log")" "$(dirname "$g_s")" "$(dirname "$s_s")"

  g_real="$(time_llc greedy "$ll" "$g_s" "$g_log" || echo NaN)"
  s_real="$(time_llc segtre "$ll" "$s_s" "$s_log" || echo NaN)"

  ratio="NaN"
  if [[ "$g_real" != "NaN" && "$s_real" != "NaN" ]]; then
    ratio="$(awk -v a="$s_real" -v b="$g_real" 'BEGIN{ if (b==0) print "Inf"; else printf "%.3f", a/b }')"
  fi
  echo "$rel,$g_real,$s_real,$ratio" >> "$CSV"
done

echo "[3/3] 統計結果"
echo "=== 慢於門檻的案例 (ratio >= $THRESHOLD) ==="
awk -F, -v th="$THRESHOLD" 'NR==1{next} $4!="NaN" && $4>=th {print $0}' "$CSV" | sort -t, -k4,4nr

echo
echo "Summary: $CSV"
echo "IR:      $IRDIR"
echo "ASM:     $ASMDIR/{greedy,segtre}/*.s"
echo "LOGS:    $LOGDIR/{greedy,segtre}/*.time"
