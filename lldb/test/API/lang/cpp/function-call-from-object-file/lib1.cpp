#include "common.h"

// Parameter "Foo*" forces LLDB to parse "Foo" from the object
// file that it is stopped in.
void lib1_func(Foo *) {
  // Force definition into lib1.o debug-info.
  Foo{}.foo();
}
