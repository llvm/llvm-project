#include <stdint.h>
#include <stdio.h>
int main() {
  union {
    uint8_t buf[8];
    uint64_t val;
  } a;
  a.val = 0; // break here
  for (int i = 0; i < 5; i++) {
    a.val = i;
    printf("a.val is %lu\n", a.val);
  }
}
