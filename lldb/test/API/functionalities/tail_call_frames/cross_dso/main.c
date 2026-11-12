#include "shared.h"

__attribute__((noinline))
static void helper() {
  tail_called_in_a_from_main();
}

__attribute__((disable_tail_calls))
int main() {
  helper();
  return 0;
}
