#include <iostream>
#include <memory>

int main() {
  int not_a_ptr = 666;
  const char *rawptr = "dead";
  std::unique_ptr<int> smartptr(new int(42));
  // Breakpoint
  return 0;
}
