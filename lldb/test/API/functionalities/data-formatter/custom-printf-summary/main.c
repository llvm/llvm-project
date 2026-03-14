#include <stdint.h>
#include <stdio.h>

struct Bytes {
  uint8_t ubyte;
  int8_t sbyte;
};

int main() {
  struct Bytes bytes = {0x30, 0x01};
  (void)bytes;
  printf("break here\n");
}
