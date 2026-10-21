#include <stdio.h>

__thread int var_shared = 33;

int LLDB_DYLIB_EXPORT touch_shared() { return var_shared; }

void LLDB_DYLIB_EXPORT shared_check() {
  var_shared *= 2;
  printf(""); // shared thread breakpoint
}
