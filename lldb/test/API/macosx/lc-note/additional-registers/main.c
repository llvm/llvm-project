#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {

  char *heap_buf = (char *)malloc(80);
  strcpy(heap_buf, "this is a string on the heap");

  return 0; // break here
}
