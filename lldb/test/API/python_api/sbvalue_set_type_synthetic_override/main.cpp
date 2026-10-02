#include <vector>

struct Foo {
  int foo;
};
struct Bar {
  bool bar;
};

int main() {
  std::vector<int> vec = {0, 1, 2, 3, 4};

  Foo foo{.foo = 10};
  Bar bar{.bar = false};

  return 0; // break here
}
