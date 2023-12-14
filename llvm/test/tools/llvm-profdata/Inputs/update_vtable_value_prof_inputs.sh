#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Path to clang++ required!"
  echo "Usage: update_vtable_value_prof_inputs.sh /path/to/updated/clang++"
  exit 1
else
  CLANG=$1
fi


# Remember current directory.
CURDIR=$PWD

# Allows the script to be invoked from other directories.
OUTDIR=$(dirname $(realpath -s $0))
echo $OUTDIR

cd $OUTDIR

# vtable_prof.cc has the following class hierarchy:
# class Base
# ├── class Derived1
# └── class Derived2
# Derived1 is a class in the global namespace and Derived2 is in anonymous
# namespace for test coverage. Overridden virtual methods are annotated as
# `noinline` so the callsite remains indirect calls for testing purposes.
cat > vtable_prof.cc << EOF
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

namespace {
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
}  // namespace

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


# Clean up temporary files on exit and return to original directory.
cleanup() {
  rm -f vtable_prof
  rm -f vtable_prof.cc
  cd $CURDIR
}
trap cleanup EXIT

FLAGS="-fuse-ld=lld -O2 -g -fprofile-generate=. -mllvm -enable-vtable-value-profiling"

${CLANG} ${FLAGS} vtable_prof.cc -o vtable_prof
env LLVM_PROFILE_FILE=vtable-value-prof-basic.profraw ./vtable_prof
