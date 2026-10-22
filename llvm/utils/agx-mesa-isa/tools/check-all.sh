#!/bin/sh
# Runs every check of strict mode across the projects that implement it and
# prints one line per check (PASS, FAIL or SKIP when its inputs are missing):
#
#   tables     the backend and simulator tables are what mesa_isa.py generates
#   fuzz       mesa_isa.py accepts what Mesa's own packer produces
#   lit        the backend's Mesa ISA lit tests
#   codegen    every CodeGen/AGX test compiles with +agx-strict for G13 and G14
#   mlir       the MLIR AGX tests (ChipInfo, expand-ops in strict mode)
#   fatbin     agx-build-fatbin.py builds the CHIP+agx-strict entries
#   simulator  the AGX Performance Simulator's Mesa ISA tests
#   zluda      ZLUDA's PTX tests compiled in strict mode
#   sail       the Sail AGX model against Mesa's forms and corpus
#   checked    agx-checked kernels on the simulator: each bug caught, correct
#              kernels unchanged (llvm/utils/agx-checked/checked_tests.py)
#   zchecked   checked mode through ZLUDA: PTX built by zoc --agx-checked on
#              the simulator (llvm/utils/agx-checked/zluda_tests.py), and the
#              runtime's record decoding and fault dump replayed in agx_run
#
#   check-all.sh [-k CHECK,...]
#
# Inputs, as environment variables (a check whose input is unset or missing is
# skipped):
#   LLVM_BUILD   LLVM build directory with llc, llvm-mc, opt, llvm-lit,
#                FileCheck, mlir-opt and mlir-translate
#   MESA         a Mesa checkout (read only)
#   SIMULATOR    an AGX-Performance-Simulator checkout
#   ZLUDA        a ZLUDA checkout with the AGX backend
#   SAIL_AGX     PTX-formalization's sail/agx directory, with the model built
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
LLVM=$(cd "$HERE/../.." && pwd)
ROOT=$(cd "$LLVM/.." && pwd)
ONLY=""
[ "${1:-}" = "-k" ] && ONLY=",$2,"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
FAILED=0

have() { [ -n "$1" ] && [ -e "$1" ]; }

run() {
  name=$1
  shift
  case "$ONLY" in "" | *",$name,"*) ;; *) return ;; esac
  if ("$@") >"$TMP/$name.log" 2>&1; then
    printf 'PASS  %s\n' "$name"
  elif [ $? -eq 77 ]; then
    printf 'SKIP  %s (%s)\n' "$name" "$(tail -n 1 "$TMP/$name.log")"
  else
    printf 'FAIL  %s\n' "$name"
    sed 's/^/      /' "$TMP/$name.log" | tail -n 20
    FAILED=1
  fi
}

skip() { echo "$1"; exit 77; }

tables() (
  cd "$HERE"
  python3 -I mesa_isa.py --cpp "$TMP/t.inc" --json "$TMP/t.json" &&
    cmp "$TMP/t.inc" "$LLVM/lib/Target/AGX/MCTargetDesc/AGXMesaISA.inc" &&
    cmp "$TMP/t.json" mesa-isa.json || exit 1
  if have "${SIMULATOR:-}"; then
    python3 -I mesa_isa.py --rust "$TMP/t.rs" &&
      cmp "$TMP/t.rs" "$SIMULATOR/rust/src/isa/mesa_table.rs"
  fi
)

fuzz() {
  have "${MESA:-}" || skip "MESA unset"
  sh "$HERE/tools/fuzz.sh" "$MESA" 300 1
}

lit() {
  have "${LLVM_BUILD:-}" || skip "LLVM_BUILD unset"
  "$LLVM_BUILD/bin/llvm-lit" -q "$LLVM/test/CodeGen/AGX" "$LLVM/test/MC/AGX" \
    "$LLVM/test/MC/Disassembler/AGX"
}

codegen() {
  have "${LLVM_BUILD:-}" || skip "LLVM_BUILD unset"
  n=0
  for f in "$LLVM"/test/CodeGen/AGX/*.ll; do
    # tests of diagnostics, and of other generations' features
    grep -q '^; RUN: not ' "$f" && continue
    for cpu in agx-g13g agx-g14c; do
      if ! "$LLVM_BUILD/bin/llc" -mtriple=agx -mcpu=$cpu -mattr=+agx-strict \
          -filetype=obj -o /dev/null "$f" 2>"$TMP/err"; then
        echo "$(basename "$f") $cpu:"; head -n 3 "$TMP/err"; n=$((n + 1))
      fi
    done
  done
  [ $n -eq 0 ]
}

mlir() {
  have "${LLVM_BUILD:-}" && [ -x "$LLVM_BUILD/bin/mlir-opt" ] || skip "no mlir-opt"
  "$LLVM_BUILD/bin/llvm-lit" -q "$ROOT/mlir/test/Dialect/AGX" \
    "$ROOT/mlir/test/Conversion/AGXToAGXIR" "$ROOT/mlir/test/Conversion/GPUToAGXIR"
}

fatbin() {
  have "${LLVM_BUILD:-}" && [ -x "$LLVM_BUILD/bin/mlir-opt" ] || skip "no mlir-opt"
  k="$ROOT/mlir/utils/agx-kernels"
  # ZLUDA's library kernels
  have "${ZLUDA:-}" || skip "ZLUDA unset (the kernel sources)"
  src="$ZLUDA/agx_libs/kernels/sgemm.mlir"
  have "$src" || skip "no $src"
  python3 -I "$k/agx-build-fatbin.py" --mlir-opt "$LLVM_BUILD/bin/mlir-opt" \
    --mlir-translate "$LLVM_BUILD/bin/mlir-translate" --llc "$LLVM_BUILD/bin/llc" \
    --chips agx-g13g,agx-g14c "$src" -o "$TMP/k.agxfat" &&
    python3 -I - "$TMP/k.agxfat" <<'EOF'
import struct, sys
d = open(sys.argv[1], "rb").read()
assert d[:8] == b"ZAGXFAT1", "bad magic"
n = struct.unpack_from("<I", d, 8)[0]
names = [d[16 + 48 * i:48 + 48 * i].rstrip(b"\0").decode() for i in range(n)]
want = ["agx-g13g", "agx-g14c", "agx-g13g+agx-strict", "agx-g14c+agx-strict"]
assert names == want, names
EOF
}

simulator() {
  have "${SIMULATOR:-}" || skip "SIMULATOR unset"
  (cd "$SIMULATOR/rust" && cargo test -q --release --lib mesa)
}

zluda() {
  have "${ZLUDA:-}" || skip "ZLUDA unset"
  have "${LLVM_BUILD:-}" || skip "LLVM_BUILD unset"
  (cd "$ZLUDA" && ZLUDA_LLVM_BUILD_DIR="$LLVM_BUILD" cargo test -q -p ptx \
    --no-default-features --features agx spirv_run_kernels_compile_agx_strict)
}

sail() {
  have "${SAIL_AGX:-}" || skip "SAIL_AGX unset"
  have "${SIMULATOR:-}" || skip "SIMULATOR unset (the Mesa corpus)"
  MESA_ISA_DIR="$HERE" python3 -I "$SAIL_AGX/gen/mesacheck.py" \
    --corpus "$SIMULATOR/rust/src/isa/mesa_corpus.txt"
}

checked() {
  have "${LLVM_BUILD:-}" || skip "LLVM_BUILD unset"
  have "${SIMULATOR:-}" || skip "SIMULATOR unset"
  run_="$SIMULATOR/rust/target/release/agx_run"
  [ -x "$run_" ] || (cd "$SIMULATOR/rust" && cargo build -q --release --bin agx_run) || exit 1
  python3 -I "$LLVM/utils/agx-checked/checked_tests.py" --llvm-build "$LLVM_BUILD" --agx-run "$run_"
}

zchecked() {
  have "${LLVM_BUILD:-}" || skip "LLVM_BUILD unset"
  have "${SIMULATOR:-}" || skip "SIMULATOR unset"
  have "${ZLUDA:-}" || skip "ZLUDA unset"
  run_="$SIMULATOR/rust/target/release/agx_run"
  [ -x "$run_" ] || (cd "$SIMULATOR/rust" && cargo build -q --release --bin agx_run) || exit 1
  (cd "$ZLUDA" && ZLUDA_LLVM_BUILD_DIR="$LLVM_BUILD" cargo build -q --release -p compiler \
    --no-default-features --features agx) || exit 1
  python3 -I "$LLVM/utils/agx-checked/zluda_tests.py" --zoc "$ZLUDA/target/release/zoc" \
    --llvm-build "$LLVM_BUILD" --agx-run "$run_" || exit 1
  (cd "$ZLUDA" && AGX_RUN="$run_" cargo test -q -p agx_runtime -p zluda_agx)
}

run tables tables
run fuzz fuzz
run lit lit
run codegen codegen
run mlir mlir
run fatbin fatbin
run simulator simulator
run zluda zluda
run sail sail
run checked checked
run zchecked zchecked
exit $FAILED
