#include "onetwo.h"

struct InheritsFromOne : One {
  int member = 47;
} inherits_from_one;

struct InheritsFromTwo : Two {
  int member = 47;
} inherits_from_two;

struct OneAsMember {
  member::One one;
  int member = 47;
} one_as_member;

struct TwoAsMember {
  member::Two two;
  int member = 47;
} two_as_member;

array::One array_of_one[3];
array::Two array_of_two[3];

result::One get_one() { return result::One(124); }
result::Two get_two() { return result::Two(224); }

// Note that there's also a function with the name func_shadow::One.
struct ShadowedOne : func_shadow::One {
  int member = 47;
} shadowed_one;

int main() { return get_one().member; }
