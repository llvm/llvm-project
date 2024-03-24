#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Path to clang required!"
  echo "Usage: update_thinlto_indirect_call_promotion_inputs.sh /path/to/updated/clang"
  exit 1
else
  CLANG=$1
fi

# Remember current directory.
CURDIR=$PWD

# Allows the script to be invoked from other directories.
OUTDIR=$(dirname $(realpath -s $0))
cd $OUTDIR

# Creates trivial header file to expose `global_func`.
cat > lib.h << EOF
void global_func();
EOF

# Creates lib.cc. `global_func` might call one of two indirect callees. One
# callee has internal linkage and the other has external linkage.
cat > lib.cc << EOF
#include "lib.h"
static void callee0() {}
void callee1() {}
typedef void (*FPT)();
FPT calleeAddrs[] = {callee0, callee1};
void global_func() {
    FPT fp = nullptr;
    fp = calleeAddrs[0];
    fp();
    fp = calleeAddrs[1];
    fp();
}
EOF

# Create main.cc. Function `main` calls `global_func`.
cat > main.cc << EOF
#include "lib.h"
int main() {
    global_func();
}
EOF

# Clean up temporary files on exit and return to original directory.
cleanup() {
  rm -f $OUTDIR/lib.h
  rm -f $OUTDIR/lib.cc
  rm -f $OUTDIR/main.cc
  rm -f $OUTDIR/lib.h.pch
  rm -f $OUTDIR/a.out
  cd $CURDIR
}
trap cleanup EXIT

# Generate instrumented binary
${CLANG} -fuse-ld=lld -O2 -fprofile-generate=. lib.h lib.cc main.cc
# Create raw profiles
env LLVM_PROFILE_FILE=thinlto_indirect_call_promotion.profraw ./a.out
