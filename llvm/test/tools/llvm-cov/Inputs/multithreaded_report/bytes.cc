#include "abs.h"
#include "bytes.h"
#include "pow.h"

bool loopBytes() {
  uint64_t totalInt = 0;
  double totalFloat = 0;
  for (uint8_t i = 1; i != 0; ++i) {
    double a = logarithm(i);
    a = abs(a);
    totalInt += abs(pow(i, static_cast<uint8_t>(a)));
    totalFloat += pow(static_cast<decltype(a)>(i), a);
  }
  return totalInt > totalFloat;
}
