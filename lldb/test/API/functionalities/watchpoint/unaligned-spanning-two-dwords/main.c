#include <stdint.h>
#include <stdio.h>
int main() {
  union {
    uint8_t bytebuf[16];
    uint16_t shortbuf[8];
    uint64_t dwordbuf[2];
  } a;
  a.dwordbuf[0] = a.dwordbuf[1] = 0;
  a.bytebuf[0] = 0; // break here
  for (int i = 0; i < 8; i++) {
    a.shortbuf[i] += i;
  }
  for (int i = 0; i < 8; i++) {
    a.shortbuf[i] += i;
  }
}
