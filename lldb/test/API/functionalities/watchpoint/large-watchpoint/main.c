#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main() {
  const int count = 65535;
  int *array = (int*) malloc(sizeof (int) * count);
  memset (array, 0, count * sizeof (int));

  puts ("break here");

  for (int i = 0; i < count - 16; i += 16) 
    array[i] += 10;

  puts ("done, exiting.");
}
