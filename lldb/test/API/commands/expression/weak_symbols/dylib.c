#include "dylib.h"

int present_weak_int = 10;
int present_weak_function()
{
  return present_weak_int;
}

#if defined HAS_THEM
int absent_weak_int = 10;
int absent_weak_function() {
  return absent_weak_int;
}
#endif
