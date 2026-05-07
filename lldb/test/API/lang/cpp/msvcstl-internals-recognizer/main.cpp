#include <functional>

static void target() { __builtin_printf("break here"); }

int main() {
  std::function<void()> fn = [] { target(); };
  fn();
  return 0;
}
