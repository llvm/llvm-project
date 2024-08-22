#include <functional>
#include <iostream>

void print_num(int i) {
  // break here
  std::cout << i << '\n';
}

int add(int i, int j) {
  // break here
  return i + j;
}

struct PrintAdder {
  PrintAdder(int num) : num_(num) {}
  void operator()(int i) const {
    // break here
    std::cout << i << '\n';
  }
  void print_add(int i) const {
    // break here
    std::cout << num_ + i << '\n';
  }
  int num_;
};

int main() {
  // Invoke a void-returning function
  std::invoke(print_num, -9);

  // Invoke a non-void-returning function
  std::cout << std::invoke(add, 1, 10) << '\n';

  // Invoke a member function
  const PrintAdder foo(314159);
  std::invoke(&PrintAdder::print_add, foo, 1);

  // Invoke a function object
  std::invoke(PrintAdder(12), 18);
}
