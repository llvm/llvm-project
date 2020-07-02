#include "onetwo.h"

struct InheritsFromOne : One {
  constexpr InheritsFromOne() = default;
  int member = 47;
} inherits_from_one;

struct InheritsFromTwo : Two {
  constexpr InheritsFromTwo() = default;
  int member = 47;
} inherits_from_two;

int main() { return 0; }
