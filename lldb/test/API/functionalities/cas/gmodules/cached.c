#include "cached.h"

int cached_fn(void) {
  struct Cached c;
  c.x = 41;
  return c.x; // BREAK CACHED
}
