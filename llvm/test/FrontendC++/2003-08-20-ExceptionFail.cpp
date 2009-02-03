// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

void foo();

void bar() {
  struct local {
    ~local() { foo(); }
  } local_obj;

  foo();
}

