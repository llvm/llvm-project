#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main() {
  const int count = 65535;
  uint8_t *array = (uint8_t *)malloc(count);
  memset(array, 0, count);

  puts("break here");

  for (int i = 0; i < count; i++)
    array[i]++;

  puts("done, exiting.");
}
