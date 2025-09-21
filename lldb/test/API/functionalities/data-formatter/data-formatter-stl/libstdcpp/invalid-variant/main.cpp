#include <cstdio>
#include <variant>

int main() {
  std::variant<int, double, char> v1;
  v1 = 'x';

  std::puts("// break here");

  return 0;
}
