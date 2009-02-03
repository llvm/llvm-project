// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


namespace foo {
  namespace bar {
    struct X { X(); };

    X::X() {}
  }
}


namespace {
  struct Y { Y(); };
  Y::Y() {}
}
