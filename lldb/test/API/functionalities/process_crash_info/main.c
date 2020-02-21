#include <stdlib.h>
int main() {
  int *var = malloc(sizeof(int));
  free(var);
  free(var);
  return 0;
}
