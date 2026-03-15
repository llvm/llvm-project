#include "Debug.h"

std::ostream &GdbColor::operator<<(std::ostream &os, GdbColor::Code code) {
  return os << "\033[" << static_cast<int>(code) << "m";
}
