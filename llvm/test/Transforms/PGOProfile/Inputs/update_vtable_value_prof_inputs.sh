#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Path to clang++ required!"
  echo "Usage: update_vtable_value_prof_inputs.sh /path/to/updated/clang++"
  exit 1
else
  CLANG=$1
fi

OUTDIR=$(dirname $(realpath -s $0))

echo "Outdir is $OUTDIR"

cat > ${OUTDIR}/vtable_prof.cc << EOF
#include <cstdlib>
#include <cstdio>

class Base {
 public:
  virtual int func1(int a, int b) = 0;
  virtual int func2(int a, int b) = 0;
};

class Derived1 : public Base {
    public:
    __attribute__((noinline))
    int func1(int a, int b) override
    {
        return a + b;
    }

    __attribute__((noinline))
    int func2(int a, int b) override {
        return a * b;
    }
};

class Derived2 : public Base {
    public:
    
    __attribute__((noinline))
    int func1(int a, int b) override {
        return a - b;
    }

    __attribute__((noinline))
    int func2(int a, int b) override {
        return a * (a - b);
    }
};

__attribute__((noinline)) Base* createType(int a) {
    Base* base = nullptr;
    if (a % 4 == 0)
      base = new Derived1();
    else
      base = new Derived2();
    return base;
}


int main(int argc, char** argv) {
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        int a = rand();
        int b = rand();
        Base* ptr = createType(i);
        sum += ptr->func1(a, b) + ptr->func2(b, a);
    }
    printf("sum is %d\n", sum);
    return 0;
}
EOF

FLAGS="-fuse-ld=lld -O2 -g -fprofile-generate=. -flto=thin -Xclang -fwhole-program-vtables -Wl,--lto-whole-program-visibility"

${CLANG} ${FLAGS} ${OUTDIR}/vtable_prof.cc -o ${OUTDIR}/vtable_prof
env LLVM_PROFILE_FILE=${OUTDIR}/vtable_prof.profraw ${OUTDIR}/vtable_prof

rm ${OUTDIR}/vtable_prof
rm ${OUTDIR}/vtable_prof.cc

