#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Path to clang and llvm-profdata required!"
  echo "Usage: update_icall_promotion_inputs.sh /path/to/updated/clang /path/to/updated/llvm-profdata"
  exit 1
else
  CLANG=$1
  LLVMPROFDATA=$2
fi

# Allows the script to be invoked from other directories.
OUTDIR=$(dirname $(realpath -s $0))

# Creates trivial header file to expose global_func.
cat > ${OUTDIR}/lib.h << EOF
void global_func();
EOF

# Creates lib.cc. global_func might call one of two indirect callees. One callee
# has internal linkage and the other has external linkage.
cat > ${OUTDIR}/lib.cc << EOF
#include "lib.h"

static void callee0() {}
void callee1() {}

typedef void (*FPT)(); 
FPT calleeAddrs[] = {callee0, callee1};

void global_func() {
    FPT fp = nullptr;
    for (int i = 0; i < 5; i++) {
      fp = calleeAddrs[i % 2];
      fp();
    }
}
EOF

# Create main.cc that calls `global_func` in lib.cc
cat > ${OUTDIR}/main.cc << EOF
#include "lib.h"

int main() {
    global_func();
}
EOF

COMMON_FLAGS="-fuse-ld=lld -O2"

# cd into OUTDIR
cd ${OUTDIR}

# Generate instrumented binary
${CLANG} ${COMMON_FLAGS} -fprofile-generate=. lib.h lib.cc main.cc
# Create raw profiles
env LLVM_PROFILE_FILE=icall_prom.profraw ./a.out
# Create indexed profiles
${LLVMPROFDATA} merge icall_prom.profraw -o thinlto_icall_prom.profdata

# Clean up intermediate files.
rm a.out
rm ${OUTDIR}/icall_prom.profraw
rm ${OUTDIR}/lib.h.pch
rm ${OUTDIR}/lib.h
rm ${OUTDIR}/lib.cc
rm ${OUTDIR}/main.cc

# Go back to original directory
cd -
