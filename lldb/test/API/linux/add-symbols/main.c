#include <stdio.h>
static int var = 5;
int main() {
  printf("%p is %d\n", &var, var);
  return ++var;
}
