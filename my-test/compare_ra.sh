#!/usr/bin/env bash
# Compare LLVM register allocators (greedy vs segtre) on a batch of .ll files.
# Outputs: results/ (assembly + per-file time logs), summary.csv (final table)
# Optional envs:
#   LLC=</path/to/llc>           (default: ../build/bin/llc)
#   TRIPLE=<target triple>       (default: aarch64-unknown-linux-gnu)
#   OLEVEL=<O0|O1|O2|O3>         (default: O3)
#   THRESHOLD=<ratio>            (default: 2.0)  # flag as slow if segtre/greedy >= THRESHOLD
#   ROUNDS=1                     (optional) also count segtre rounds via -debug-only=regalloc

set -u
LLC="${LLC:-../build/bin/llc}"
TRIPLE="${TRIPLE:-aarch64-unknown-linux-gnu}"
OLEVEL="${OLEVEL:-O3}"
THRESHOLD="${THRESHOLD:-2.0}"

OUTDIR="results"
LOGDIR="$OUTDIR/logs"
ASMDIR="$OUTDIR/asm"
mkdir -p "$LOGDIR/greedy" "$LOGDIR/segtre" "$ASMDIR/greedy" "$ASMDIR/segtre"

CSV="summary.csv"
echo "File,GreedyReal(s),SegtreReal(s),Ratio(segtre/greedy)${ROUNDS:+,SegtreRounds}" > "$CSV"

# function to time a single llc invocation and return 'real' seconds
time_llc () {
  local ra="$1"; shift
  local infile="$1"; shift
  local asmout="$1"; shift
  local logout="$1"; shift
  # -time-passes 會印到 stderr；/usr/bin/time -p 也印到 stderr
  # 我們只抓 /usr/bin/time 的 real/user/sys（其他噪音忽略）
  /usr/bin/time -p "$LLC" "-$OLEVEL" -mtriple="$TRIPLE" -regalloc="$ra" "$infile" -o "$asmout" \
    1>/dev/null 2>"$logout"
  # 解析 real
  awk '/^real /{print $2}' "$logout" | tail -n1
}

# optional: count rounds from debug log (only for segtre)
count_rounds () {
  local infile="$1"; shift
  local dbglog="$1"; shift
  # 以 -debug-only=regalloc 輸出輪次訊息；避免巨量輸出，只在失敗或 outlier 時再開
  "$LLC" "-$OLEVEL" -mtriple="$TRIPLE" -regalloc=segtre -debug-only=regalloc "$infile" -o /dev/null \
    1>/dev/null 2>"$dbglog"
  # 根據你程式裡的輸出樣式調整這行的關鍵字
  grep -c "Segment Tree Regalloc round" "$dbglog" || true
}

shopt -s nullglob
lls=( *.ll )
if (( ${#lls[@]} == 0 )); then
  echo "No .ll files in current directory."
  exit 1
fi

printf "Using llc: %s\nTarget: %s  Opt: -%s  Files: %d  Threshold: %s×\n" \
  "$LLC" "$TRIPLE" "$OLEVEL" "${#lls[@]}" "$THRESHOLD"

for ll in "${lls[@]}"; do
  base="${ll%.ll}"

  # baseline: greedy
  g_log="$LOGDIR/greedy/$base.time"
  g_s="$ASMDIR/greedy/$base.s"
  g_real="$(time_llc greedy "$ll" "$g_s" "$g_log" || echo "NaN")"

  # segtre
  s_log="$LOGDIR/segtre/$base.time"
  s_s="$ASMDIR/segtre/$base.s"
  s_real="$(time_llc segtre "$ll" "$s_s" "$s_log" || echo "NaN")"

  # ratio
  ratio="NaN"
  if [[ "$g_real" != "NaN" && "$s_real" != "NaN" ]]; then
    # 使用 awk 做浮點除法
    ratio="$(awk -v a="$s_real" -v b="$g_real" 'BEGIN{ if (b==0) print "Inf"; else printf "%.3f", a/b }')"
  fi

  # optional rounds
  if [[ -n "${ROUNDS:-}" ]]; then
    dbg="$LOGDIR/segtre/$base.debug"
    rounds="$(count_rounds "$ll" "$dbg" 2>/dev/null || echo 0)"
    echo "$ll,$g_real,$s_real,$ratio,$rounds" >> "$CSV"
  else
    echo "$ll,$g_real,$s_real,$ratio" >> "$CSV"
  fi
done

echo
echo "=== Slow cases (ratio >= $THRESHOLD) ==="
awk -F, -v th="$THRESHOLD" 'NR==1{h=$0; next} $4!="NaN" && $4>=th {print $0}' "$CSV" | sort -t, -k4,4nr

echo
echo "Summary saved to: $CSV"
echo "Artifacts:"
echo "  ASM:   $ASMDIR/{greedy,segtre}/*.s"
echo "  LOGS:  $LOGDIR/{greedy,segtre}/*.time"
[[ -n "${ROUNDS:-}" ]] && echo "  DEBUG: $LOGDIR/segtre/*.debug"
