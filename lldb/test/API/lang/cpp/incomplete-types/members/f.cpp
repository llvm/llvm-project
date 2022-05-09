#include "a.h"

int A::f() {
  struct Af {
    int x, y;
  } af{42, 47};
  return af.x + af.y; // break here
}
