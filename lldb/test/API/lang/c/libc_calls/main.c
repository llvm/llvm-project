#include "stdio.h"

char src[64];
char dst[64];
unsigned len64 = 64;

int main(int argc, char **argv) {
  // Call a libc function so that we know there is a real libc in our process.
  puts("s"); // break here
  return 0;
}
