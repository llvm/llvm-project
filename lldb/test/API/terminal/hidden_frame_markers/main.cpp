#include <functional>
#include <iostream>

static void target() {
  int a = 0; // break here
}

int main() {
  std::function<void()> fn = [] { target(); };
  fn();
  return 0;
}
