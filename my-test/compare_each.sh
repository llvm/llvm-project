#!/usr/bin/env bash
# 對每個 .c/.cpp -> .ll 檔，分別用 greedy / segtre 編譯，輸出時間對照
# 結果存到 summary.csv

set -euo pipefail

CLANG="${CLANG:-$HOME/work/llvm-project/build/bin/clang}"
LLC="${LLC:-$HOME/work/llvm-project/build/bin/llc}"
TRIPLE="${TRIPLE:-aarch64-unknown-linux-gnu}"
OLEVEL="${OLEVEL:-O3}"
OUTDIR="${OUTDIR:-$PWD/out}"
CSV="$OUTDIR/summary.csv"

mkdir -p "$OUTDIR/ir" "$OUTDIR/asm/greedy" "$OUTDIR/asm/segtre" "$OUTDIR/logs/greedy" "$OUTDIR/logs/segtre"

echo "File,GreedyReal(s),SegtreReal(s),Ratio(segtre/greedy)" > "$CSV"

# time 一個 llc 執行
time_llc() {
  local ra="$1"; shift
  local infile="$1"; shift
  local outfile="$1"; shift
  local logfile="$1"; shift
  /usr/bin/time -p "$LLC" "-$OLEVEL" -mtriple="$TRIPLE" -regalloc="$ra" "$infile" -o "$outfile" \
    1>/dev/null 2>"$logfile" || return 1
  awk '/^real /{print $2}' "$logfile" | tail -n1
}

for src in "$@"; do
  name="$(basename "$src")"
  base="${name%.*}"
  ll="$OUTDIR/ir/$base.ll"

  echo "[IR] $src -> $ll"
  if ! "$CLANG" "-$OLEVEL" -S -emit-llvm -target "$TRIPLE" "$src" -o "$ll" 2>"$ll.err"; then
    echo "SKIP,$src,IR failed" >&2
    continue
  fi

  g_log="$OUTDIR/logs/greedy/$base.time"
  g_s="$OUTDIR/asm/greedy/$base.s"
  g_real="$(time_llc greedy "$ll" "$g_s" "$g_log" || echo NaN)"

  s_log="$OUTDIR/logs/segtre/$base.time"
  s_s="$OUTDIR/asm/segtre/$base.s"
  s_real="$(time_llc segtre "$ll" "$s_s" "$s_log" || echo NaN)"

  ratio="NaN"
  if [[ "$g_real" != "NaN" && "$s_real" != "NaN" ]]; then
    ratio="$(awk -v a="$s_real" -v b="$g_real" 'BEGIN{ if (b==0) print "Inf"; else printf "%.3f", a/b }')"
  fi

  echo "$src,$g_real,$s_real,$ratio" >> "$CSV"
done

echo
echo "結果已存到 $CSV"
