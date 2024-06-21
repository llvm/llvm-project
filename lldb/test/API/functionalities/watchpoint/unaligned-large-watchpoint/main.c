#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct obj {
  uint32_t one;
  uint32_t two;
  uint32_t three;
  uint32_t four;
};

int main() {
  const int count = 16776960;
  uint8_t *array = (uint8_t *)malloc(count);
  memset(array, 0, count);
  struct obj variable;
  variable.one = variable.two = variable.three = variable.four = 0;

  puts("break here");

  for (int i = 0; i < count; i++)
    array[i]++;

  puts("done iterating");

  variable.one = 1;
  variable.two = 2;
  variable.three = 3;
  variable.four = 4;

  printf("variable value is %d\n",
         variable.one + variable.two + variable.three + variable.four);
  puts("exiting.");
}
