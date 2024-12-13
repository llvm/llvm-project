#include <cstdint>
#include <limits>

int
main(int argc, char** argv) {
  char c = 10;
  unsigned char uc = 1;
  int a = 1;
  int int_max = std::numeric_limits<int>::max();
  int int_min = std::numeric_limits<int>::min();
  unsigned int uint_max = std::numeric_limits<unsigned int>::max();
  unsigned int uint_zero = 0;
  long long ll_max = std::numeric_limits<long long>::max();
  long long ll_min = std::numeric_limits<long long>::min();
  unsigned long long ull_max = std::numeric_limits<unsigned long long>::max();
  unsigned long long ull_zero = 0;

  int x = 2;
  int& r = x;
  int* p = &x;

  typedef int& myr;
  myr my_r = x;

  auto fnan = std::numeric_limits<float>::quiet_NaN();
  auto fsnan = std::numeric_limits<float>::signaling_NaN();
  // Smallest positive non-zero float denormal
  auto fdenorm = 0x0.1p-145f;

  // BREAK(TestArithmetic)
  return 0; // Set a breakpoint here
}
