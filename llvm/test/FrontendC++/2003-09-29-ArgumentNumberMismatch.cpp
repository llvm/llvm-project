// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

// Non-POD classes cannot be passed into a function by component, because their
// dtors must be run.  Instead, pass them in by reference.  The C++ front-end
// was mistakenly "thinking" that 'foo' took a structure by component.

struct C {
  int A, B;
  ~C() {}
};

void foo(C b);

void test(C *P) {
  foo(*P);
}

