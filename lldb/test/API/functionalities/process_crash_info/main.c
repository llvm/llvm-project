#include <stdlib.h>

int main() {
  int *var = malloc(sizeof(int)); // break here
  free(var);
  free(var);
  return 0;
}
