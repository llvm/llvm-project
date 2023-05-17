#!/bin/bash

if [ -z $1 ]; then
  echo "Path to clang required!"
  echo "Usage: update_memprof_inputs.sh /path/to/updated/clang"
  exit 1
else
  CLANG=$1
fi

# Allows the script to be invoked from other directories.
OUTDIR=$(dirname $(realpath -s $0))

# Note that changes in the code below which affect relative line number
# offsets of calls from their parent function can affect callsite matching in
# the LLVM IR.
cat > ${OUTDIR}/memprof.cc << EOF
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
char *foo() {
  return new char[10];
}
char *foo2() {
  return foo();
}
char *bar() {
  return foo2();
}
char *baz() {
  return foo2();
}
char *recurse(unsigned n) {
  if (!n)
    return foo();
  return recurse(n-1);
}
int main(int argc, char **argv) {
  // Test allocations with different combinations of stack contexts and
  // coldness (based on lifetime, since they are all accessed a single time
  // per byte via the memset).
  char *a = new char[10];
  char *b = new char[10];
  char *c = foo();
  char *d = foo();
  char *e = bar();
  char *f = baz();
  memset(a, 0, 10);
  memset(b, 0, 10);
  memset(c, 0, 10);
  memset(d, 0, 10);
  memset(e, 0, 10);
  memset(f, 0, 10);
  // a and c have short lifetimes
  delete[] a;
  delete[] c;
  // b, d, e, and f have long lifetimes and will be detected as cold by default.
  sleep(200);
  delete[] b;
  delete[] d;
  delete[] e;
  delete[] f;

  // Loop ensures the two calls to recurse have stack contexts that only differ
  // in one level of recursion. We should get two stack contexts reflecting the
  // different levels of recursion and different allocation behavior (since the
  // first has a very long lifetime and the second has a short lifetime).
  for (unsigned i = 0; i < 2; i++) {
    char *g = recurse(i + 3);
    memset(g, 0, 10);
    if (!i)
      sleep(200);
    delete[] g;
  }
  return 0;
}
EOF

COMMON_FLAGS="-fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie"

${CLANG} ${COMMON_FLAGS} -fmemory-profile ${OUTDIR}/memprof.cc -o ${OUTDIR}/memprof.exe
env MEMPROF_OPTIONS=log_path=stdout ${OUTDIR}/memprof.exe > ${OUTDIR}/memprof.memprofraw

${CLANG} ${COMMON_FLAGS} -fprofile-generate=. \
  ${OUTDIR}/memprof.cc -o ${OUTDIR}/pgo.exe
env LLVM_PROFILE_FILE=${OUTDIR}/memprof_pgo.profraw ${OUTDIR}/pgo.exe

rm ${OUTDIR}/memprof.cc
rm ${OUTDIR}/pgo.exe
