#include <stdio.h>
#include <stdlib.h>

#include "present.h"

const int present_const_data = 5;
int present_dirty_data = 10;

void present_init(int in) { present_dirty_data += 10; }

int present(char *to_be_removed_heap_buf, int to_be_removed_const_data,
            int to_be_removed_dirty_data) {
  char *present_heap_buf = (char *)malloc(256);
  sprintf(present_heap_buf, "have ints %d %d %d %d", to_be_removed_const_data,
          to_be_removed_dirty_data, present_dirty_data, present_const_data);
  printf("%s\n", present_heap_buf);
  puts(to_be_removed_heap_buf);

  puts("break here");

  return present_const_data + present_dirty_data;
}
