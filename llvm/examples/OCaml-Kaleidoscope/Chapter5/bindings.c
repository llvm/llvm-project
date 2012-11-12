#include <stdio.h>

/* putchard - putchar that takes a double and returns 0. */
extern double putchard(double X) {
  putchar((char)X);
  return 0;
}
