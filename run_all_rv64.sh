#!/usr/bin/env bash
set -u

# === 可調參數（也可用環境變數覆蓋） ===
ROOT="${ROOT:-ts-build}"                        # 搜尋執行檔的根目錄
SYSROOT="${SYSROOT:-/usr/riscv64-linux-gnu}"    # RISC-V sysroot（qemu -L）
TIMEOUT_SECS="${TIMEOUT_SECS:-60}"              # 每個程式的超時秒數
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)}"  # 併發數

# === 先做基本檢查 ===
command -v qemu-riscv64 >/dev/null 2>&1 || { echo "ERROR: qemu-riscv64 not found."; exit 1; }
[ -d "$SYSROOT" ] || { echo "ERROR: SYSROOT not found: $SYSROOT"; exit 1; }
[ -d "$ROOT" ] || { echo "ERROR: ROOT not found: $ROOT"; exit 1; }

echo "[info] ROOT=$ROOT  SYSROOT=$SYSROOT  TIMEOUT_SECS=$TIMEOUT_SECS  JOBS=$JOBS"

LOGDIR="${LOGDIR:-rv64-logs}"
mkdir -p "$LOGDIR"

# === 找出所有 RISC-V ELF 可執行檔 ===
# 用 'file' 判斷，僅保留 "ELF 64-bit ... RISC-V" 的檔案
mapfile -t ELF_LIST < <(
  find "$ROOT" -type f -perm -111 -print0 \
  | xargs -0 file \
  | grep -E 'ELF 64-bit.*RISC-V' \
  | cut -d: -f1 \
  | sort
)

TOTAL=${#ELF_LIST[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No RISC-V ELF executables found under $ROOT"
  exit 1
fi

echo "[info] Found $TOTAL RISC-V executables."

# === 跑一個檔案的函式 ===
run_one() {
  local elf="$1"
  # 產生穩定可讀的 log 檔名（保留子目錄層級）
  local rel="${elf#$ROOT/}"
  local outdir="$LOGDIR/$(dirname "$rel")"
  mkdir -p "$outdir"
  local base="$(basename "$rel")"
  local out="$outdir/${base}.out"
  local err="$outdir/${base}.err"
  local rcfile="$outdir/${base}.rc"

  # 跑之前先把動態依賴列出來，debug 用（不影響結果）
  # qemu-riscv64 -L "$SYSROOT" /lib/ld-linux-riscv64-lp64d.so.1 --list "$elf" >"$outdir/${base}.deps" 2>&1 || true

  # 加 timeout，避免死鎖/無限迴圈卡住
  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT_SECS" qemu-riscv64 -L "$SYSROOT" "$elf" >"$out" 2>"$err"
    rc=$?
    # 124/137 常見於 timeout（取決於 timeout 實作），標記特別碼
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      echo "TIMEOUT" > "$rcfile"
    else
      echo "$rc" > "$rcfile"
    fi
  else
    qemu-riscv64 -L "$SYSROOT" "$elf" >"$out" 2>"$err"
    echo $? > "$rcfile"
  fi
}

export -f run_one
export ROOT SYSROOT TIMEOUT_SECS LOGDIR

# === 併發執行 ===
# 有 parallel 就用，否則退回 xargs -P
if command -v parallel >/dev/null 2>&1; then
  printf "%s\n" "${ELF_LIST[@]}" | parallel -j "$JOBS" --halt now,fail=1 run_one {}
else
  printf "%s\0" "${ELF_LIST[@]}" \
  | xargs -0 -n 1 -P "$JOBS" bash -c 'run_one "$0"' 
fi

# === 統計結果 ===
PASS=0; FAIL=0; TIMEO=0
while IFS= read -r -d '' rcfile; do
  rc=$(cat "$rcfile")
  if [ "$rc" = "0" ]; then
    PASS=$((PASS+1))
  elif [ "$rc" = "TIMEOUT" ]; then
    TIMEO=$((TIMEO+1))
  else
    FAIL=$((FAIL+1))
  fi
done < <(find "$LOGDIR" -type f -name '*.rc' -print0)

echo
echo "===== SUMMARY ====="
echo "Total:   $TOTAL"
echo "Pass:    $PASS"
echo "Fail:    $FAIL"
echo "Timeout: $TIMEO"
echo "Logs at: $LOGDIR/"
