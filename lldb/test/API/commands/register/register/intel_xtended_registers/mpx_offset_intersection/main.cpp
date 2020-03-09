#include <cstdint>

int main() {
  asm volatile("int3");
  return 0;
}
