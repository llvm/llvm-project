#include <stdint.h>

__attribute__((noinline, optnone)) int dylib_shift(void) { return 33; }

int LLDB_DYLIB_EXPORT dylib_ubsan_issue(void) {
  uint32_t x = 0;
  x = x << dylib_shift(); // dylib ubsan issue
  return x;
}
