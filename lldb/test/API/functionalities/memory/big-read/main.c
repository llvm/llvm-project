#include <string.h>
int main() {
  int c[2048];
  memset(c, 0, 2048 * sizeof(int));

  c[2047] = 0xfeed;

  return c[2047]; // breakpoint here
}
