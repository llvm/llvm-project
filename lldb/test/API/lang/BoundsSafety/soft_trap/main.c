#include <ptrcheck.h>

int main(void) {
  int pad;
  int buffer[] = {0, 1};
  int pad2;
  int tmp = buffer[2]; // access past upper bound
  tmp = buffer[-1];    // access below lower bound
  return 0;
}
