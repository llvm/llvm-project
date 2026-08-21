#include "common.h"

void lib2_func(Foo *) {
  // Force definition into lib2.o debug-info.
  Foo{}.foo();
}
