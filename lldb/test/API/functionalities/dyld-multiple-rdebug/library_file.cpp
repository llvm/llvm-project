#include "library_file.h"
#include <stdio.h>

int library_function(void) {
  puts(__FUNCTION__); // Library break here
  return 0;
}
