#include "a.h"

int A::g() {
  struct Ag {
    int a, b;
  } ag{47, 42};
  return ag.a + ag.b; // break here
}
