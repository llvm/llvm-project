#include <cstdint>
#include <limits>

int main(int argc, char** argv) {
  bool var_true = true;
  bool var_false = false;

  unsigned long long ull_max = std::numeric_limits<unsigned long long>::max();
  unsigned long long ull_zero = 0;

  struct S {
  } s;

  const char* p = nullptr;

  uint32_t mask_ff = 0xFF;

  // BREAK(TestBitwiseOperators)
  return 0; // Set a breakpoint here
}
