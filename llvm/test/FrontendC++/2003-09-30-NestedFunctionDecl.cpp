// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

// The C++ front-end thinks the two foo's are different, the LLVM emitter
// thinks they are the same.  The disconnect causes problems.

void foo() { }

void bar() {
  void foo();

  foo();
}
