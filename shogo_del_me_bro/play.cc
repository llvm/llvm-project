#include <iostream>
using namespace std;

enum class RoundingMode : int { Upward, Downward, TowardZero, Nearest };

string str(RoundingMode mode) {
  switch (mode) {
  case RoundingMode::Upward:
    return "MPFR_RNDU" ;
  case RoundingMode::Downward:
    return "MPFR_RNDD" ;
  case RoundingMode::TowardZero:
    return "MPFR_RNDZ" ;
  default:
    __builtin_unreachable();
  }
}

int main() {
    RoundingMode mode = RoundingMode::Nearest;
    cout << str(mode) << endl;
}