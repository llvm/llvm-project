// RUN: %llvmgxx %s -emit-llvm -S -o - | not grep eprintf1
// RUN: %llvmgxx %s -emit-llvm -S -o - | grep eprintf

// Only one eprintf should exist in the output

extern "C" 
void __eprintf();

void foo() {

  __eprintf();
}

void *bar() {
  extern void *__eprintf;
  return &__eprintf;
}
