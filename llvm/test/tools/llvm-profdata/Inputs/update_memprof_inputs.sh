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

read -r -d '' BASIC << EOF
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}
EOF

read -r -d '' INLINE << EOF
#include <stdlib.h>
#include <string.h>

__attribute__((always_inline))
void qux(int x) {
  char *ptr = malloc(x);
  memset(ptr, 0, x);
  free(ptr);
}

__attribute__((noinline))
void foo(int x){ qux(x); }

__attribute__((noinline))
void bar(int x) { foo(x); }

int main(int argc, char **argv) {
  bar(argc);
  return 0;
}
EOF

read -r -d '' MULTI << EOF
#include <sanitizer/memprof_interface.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}
EOF

DEFAULT_MEMPROF_FLAGS="-fuse-ld=lld -Wl,--no-rosegment -gmlt -fdebug-info-for-profiling -fmemory-profile -mno-omit-leaf-frame-pointer -fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id -no-pie"

# Map each test to their source and any additional flags separated by ; 
declare -A INPUTS
INPUTS["basic"]="BASIC"
INPUTS["inline"]="INLINE"
INPUTS["multi"]="MULTI"
INPUTS["pic"]="BASIC;-pie"
INPUTS["buildid"]="BASIC;-Wl,-build-id=sha1"

for name in "${!INPUTS[@]}"; do
  IFS=";" read -r src flags <<< "${INPUTS[$name]}"
  echo "${!src}" > ${OUTDIR}/${name}.c
  ${CLANG} ${DEFAULT_MEMPROF_FLAGS} ${flags} ${OUTDIR}/${name}.c -o ${OUTDIR}/${name}.memprofexe
  env MEMPROF_OPTIONS=log_path=stdout ${OUTDIR}/${name}.memprofexe > ${OUTDIR}/${name}.memprofraw
  rm ${OUTDIR}/${name}.c
done
