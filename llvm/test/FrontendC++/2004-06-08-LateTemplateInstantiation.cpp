// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null



template<typename Ty>
struct normal_iterator {
  int FIELD;
};

void foo(normal_iterator<int>);
normal_iterator<int> baz();

void bar() {
  foo(baz());
}

void *bar2() {
  return (void*)foo;
}
