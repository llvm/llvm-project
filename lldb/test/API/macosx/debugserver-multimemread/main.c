#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *memory = malloc(1024);
  memset(memory, '-', 1024);
  // Write "interesting" characters at an offset from the memory filled with
  // `-`. This way, if we read outside the range in either direction, we should
  // find `-`s`.
  int offset = 42;
  for (int i = offset; i < offset + 14; i++)
    memory[i] = 'a' + (i - offset);
  return 0; // break here
}
