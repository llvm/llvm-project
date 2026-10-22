#!/bin/sh
# Builds tools/fuzz.c against a Mesa checkout and checks mesa_isa.py's engine
# against Mesa's packer.
#
#   fuzz.sh MESA_SOURCE_DIR [COUNT [SEED [CORPUS_OUT]]]
#
# CORPUS_OUT keeps the packed instructions (the simulator's tests read one).
#
# Needs python3 with mako and PyYAML (for Mesa's header generators) and a C
# compiler. Nothing is written outside a temporary directory.
set -e
MESA=$(cd "$1" && pwd)
COUNT=${2:-200}
SEED=${3:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
G=$OUT/gen
mkdir -p "$G/util/format"

(cd "$MESA/src/compiler/nir" &&
  python3 nir_opcodes_h.py > "$G/nir_opcodes.h" &&
  python3 nir_builder_opcodes_h.py > "$G/nir_builder_opcodes.h" &&
  python3 nir_intrinsics_h.py --out "$G/nir_intrinsics.h" &&
  python3 nir_intrinsics_indices_h.py --out "$G/nir_intrinsics_indices.h")
(cd "$MESA/src/compiler" && python3 builtin_types_h.py "$G/builtin_types.h")
(cd "$MESA/src/util/format" &&
  python3 u_format_table.py u_format.yaml --enums > "$G/util/format/u_format_gen.h")
(cd "$MESA/src/util" &&
  python3 process_shader_stats.py shader_stats.rnc shader_stats.xml 2>/dev/null |
  sed '/^\*\*\*\*$/,/^\*\*\*\*$/d' > "$G/util/shader_stats.h")
(cd "$MESA/src/asahi/compiler" &&
  python3 agx_opcodes.h.py > "$G/agx_opcodes.h" &&
  python3 agx_opcodes.c.py > "$G/agx_opcodes.c")

CFLAGS="-std=gnu11 -O1 -w -DUTIL_ARCH_LITTLE_ENDIAN=1 -DUTIL_ARCH_BIG_ENDIAN=0
  -DHAVE_TIMESPEC_GET -DHAVE_STRUCT_TIMESPEC -DHAVE_PTHREAD -D_GNU_SOURCE
  -I$G -I$MESA/include -I$MESA/src -I$MESA/src/compiler/nir
  -I$MESA/src/asahi/compiler -I$MESA/src/compiler -I$MESA/src/util"
${CC:-cc} $CFLAGS -o "$OUT/fuzz" "$HERE/fuzz.c" "$G/agx_opcodes.c" "$MESA/src/util/u_dynarray.c"
"$OUT/fuzz" "$COUNT" "$SEED" > "$OUT/cases.txt"
python3 "$HERE/../mesa_isa.py" --fuzz-check "$OUT/cases.txt"
if [ -n "$4" ]; then
  { echo "# Packed by Mesa $(git -C "$MESA" rev-parse --short HEAD 2>/dev/null)'s agx_pack.c:" \
      "llvm/utils/agx-mesa-isa/tools/fuzz.sh MESA $COUNT $SEED"
    cat "$OUT/cases.txt"; } > "$4"
fi
