#include <source_location>

std::source_location foo() { return std::source_location::current(); }

int main() {
  auto loc_main = std::source_location::current();
  auto loc_foo = foo();
  auto loc_empty = std::source_location{};
  return 0; // break here
}
