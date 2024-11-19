#include "artificial_location.h"

int A::foo() {
#line 0
  return 42;
}

int main() { return A::foo(); }
