#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "present.h"
#include "to-be-removed.h"

const int main_const_data = 5;
int main_dirty_data = 10;
int main(int argc, char **argv) {

  to_be_removed_init(argc);
  present_init(argc);
  main_dirty_data += argc;

  char *heap_buf = (char *)malloc(80);
  strcpy(heap_buf, "this is a string on the heap");

  return to_be_removed(heap_buf, main_const_data, main_dirty_data);
}
