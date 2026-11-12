#include "shared.h"

volatile int x;

__attribute__((noinline))
void tail_called_in_b_from_b() {
  ++x; // break here
}

void tail_called_in_b_from_a() {
  tail_called_in_b_from_b();
}
