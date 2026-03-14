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
  char *ptr = (char*) malloc(x);
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

read -r -d '' BASIC_HISTOGRAM << EOF
struct A {
  long int a;
  long int b;
  long int c;
  long int d;
  long int e;
  long int f;
  long int g;
  long int h;
  A() {};
};

void foo() {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;
  acc += a->f;
  acc += a->g;
  acc += a->h;
  delete a;
}
void bar() {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->a;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->b;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->c;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->d;
  acc += a->e;
  acc += a->e;
  acc += a->e;
  acc += a->e;
  acc += a->f;
  acc += a->f;
  acc += a->f;
  acc += a->g;
  acc += a->g;
  acc += a->h;

  delete a;
}

int main(int argc, char **argv) {
  long int acc = 0;
  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;
  acc += a->f;
  acc += a->g;
  acc += a->h;

  delete a;

  A *b = new A();
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->a;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->b;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->c;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->d;
  acc += b->e;
  acc += b->e;
  acc += b->e;
  acc += b->e;
  acc += b->f;
  acc += b->f;
  acc += b->f;
  acc += b->g;
  acc += b->g;
  acc += b->h;

  delete b;

  A *c = new A();
  acc += c->a;

  for (int i = 0; i < 21; ++i) {

    foo();
  }

  for (int i = 0; i < 21; ++i) {

    bar();
  }

  return 0;
}
EOF

read -r -d '' PADDING_HISTOGRAM << EOF
struct A {
  char a;
  char b;
  long int c;
  char d;
  int e;
  A() {};
};

struct B {
  double x;
  double y;
  B() {};
};

struct C {
  A a;
  char z;
  B b;
  C() {};
};

int main(int argc, char **argv) {
  long int acc = 0;

  A *a = new A();
  acc += a->a;
  acc += a->b;
  acc += a->c;
  acc += a->d;
  acc += a->e;

  C *c = new C();
  acc += c->a.a;
  acc += c->a.a;
  acc += c->b.x;
  acc += c->b.y;

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


DEFAULT_HIST_FLAGS="${DEFAULT_MEMPROF_FLAGS} -mllvm -memprof-use-callbacks=true -mllvm -memprof-histogram"


# Map each test to their source and any additional flags separated by ; 
declare -A HISTOGRAM_INPUTS
HISTOGRAM_INPUTS["basic-histogram"]="BASIC_HISTOGRAM"
HISTOGRAM_INPUTS["padding-histogram"]="PADDING_HISTOGRAM"

for name in "${!HISTOGRAM_INPUTS[@]}"; do
  IFS=";" read -r src flags <<< "${HISTOGRAM_INPUTS[$name]}"
  echo "${!src}" > ${OUTDIR}/${name}.c
  ${CLANG} ${DEFAULT_HIST_FLAGS} ${flags} ${OUTDIR}/${name}.c -o ${OUTDIR}/${name}.memprofexe
  env MEMPROF_OPTIONS=log_path=stdout ${OUTDIR}/${name}.memprofexe > ${OUTDIR}/${name}.memprofraw
  rm ${OUTDIR}/${name}.c
done