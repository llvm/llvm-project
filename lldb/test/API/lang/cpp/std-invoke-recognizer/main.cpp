#include <functional>

void consume_number(int i) { __builtin_printf("break here"); }

int add(int i, int j) {
  // break here
  return i + j;
}

struct Callable {
  Callable(int num) : num_(num) {}
  void operator()(int i) const { __builtin_printf("break here"); }
  void member_function(int i) const { __builtin_printf("break here"); }
  int num_;
};

int main() {
  // Invoke a void-returning function
  std::invoke(consume_number, -9);

  // Invoke a non-void-returning function
  std::invoke(add, 1, 10);

  // Invoke a member function
  const Callable foo(314159);
  std::invoke(&Callable::member_function, foo, 1);

  // Invoke a function object
  std::invoke(Callable(12), 18);
}
