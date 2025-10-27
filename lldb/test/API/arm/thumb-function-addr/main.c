#include <stdint.h>

int a_function() { return 123; }

int main() {
  const uintptr_t a_function_addr = (uintptr_t)a_function;
  // Set break point at this line.
  return a_function();
}
