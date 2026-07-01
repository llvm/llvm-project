#include <ptrcheck.h>

__attribute__((noinline, optnone)) void test_breakpoint(void) {}

int main(void) {
  int pad;
  int buffer[] = {0, 1};
  int pad2;
  int tmp = buffer[2]; // first soft trap: access past upper bound
  test_breakpoint();
  tmp = buffer[-1]; // second soft trap: access below lower bound
  return 0;
}
