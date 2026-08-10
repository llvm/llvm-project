#include "a.h"

A::A() = default;
void A::anchor() {}

int main() {
  A().f();
  A().g();
}
