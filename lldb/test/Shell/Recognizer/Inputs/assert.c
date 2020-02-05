#include <assert.h>

int main() {
  int a = 42;
  assert(a == 42);
  a--;
  assert(a == 42);
  return 0;
}
