#include <stdio.h>
#include <stdlib.h>

#include "present.h"
#include "to-be-removed.h"

const int to_be_removed_const_data = 5;
int to_be_removed_dirty_data = 10;

void to_be_removed_init(int in) { to_be_removed_dirty_data += 10; }

int to_be_removed(char *main_heap_buf, int main_const_data,
                  int main_dirty_data) {
  char *to_be_removed_heap_buf = (char *)malloc(256);
  sprintf(to_be_removed_heap_buf, "got string '%s' have int %d %d %d",
          main_heap_buf, to_be_removed_dirty_data, main_const_data,
          main_dirty_data);
  printf("%s\n", to_be_removed_heap_buf);
  return present(to_be_removed_heap_buf, to_be_removed_const_data,
                 to_be_removed_dirty_data);
}
