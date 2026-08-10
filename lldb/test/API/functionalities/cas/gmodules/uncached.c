#include "uncached.h"

int uncached_fn(void) {
  struct Uncached u;
  u.y = 17;
  return u.y; // BREAK UNCACHED
}
