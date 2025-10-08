#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *memory = malloc(1024);
  memset(memory, '-', 1024);
  for (int i = 0; i < 50; i++)
    memory[i] = 'a' + i;
  return 0; // break here
}
