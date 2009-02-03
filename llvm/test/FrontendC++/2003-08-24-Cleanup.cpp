// RUN: %llvmgxx -xc++ %s -c -o - | llvm-dis | grep unwind

struct S { ~S(); };

int mightthrow();

int test() {
  S s;
  mightthrow();
}
