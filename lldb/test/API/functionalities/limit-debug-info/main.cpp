#include "onetwo.h"

struct InheritsFromOne : One {
  constexpr InheritsFromOne() = default;
  int member = 47;
} inherits_from_one;

struct InheritsFromTwo : Two {
  constexpr InheritsFromTwo() = default;
  int member = 47;
} inherits_from_two;

struct OneAsMember {
  constexpr OneAsMember() = default;
  member::One one;
  int member = 47;
} one_as_member;

struct TwoAsMember {
  constexpr TwoAsMember() = default;
  member::Two two;
  int member = 47;
} two_as_member;

array::One array_of_one[3];
array::Two array_of_two[3];

int main() { return 0; }
