#include <ptrcheck.h>

// The dylib can't link against the soft trap runtime functions in a.out so
// give it its own copy. This also means that if a disabled BoundsSafety plugin
// is wrongly re-created when this dylib is loaded, the new instance (which only
// sees the newly loaded modules) finds a runtime here and activates.
#include "mockSoftTrapRuntime.c"

int LLDB_DYLIB_EXPORT dylib_soft_trap(void) {
  int pad;
  int buffer[] = {0, 1};
  int pad2;
  return buffer[2]; // dylib soft trap: access past upper bound
}
