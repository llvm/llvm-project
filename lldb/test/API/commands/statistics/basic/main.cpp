// Test that the lldb command `statistics` works.
#include <string>

void foo() {
  std::string str = "hello world";
  str += "\n"; // stop here
}

int main(void) {
  int patatino = 27;
  foo();
  return 0; // break here
}
