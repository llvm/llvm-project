// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


template<class T>
struct super {
  int Y;
  void foo();
};

template <class T>
struct test : virtual super<int> {};

extern test<int> X;

void foo() {
  X.foo();
}
