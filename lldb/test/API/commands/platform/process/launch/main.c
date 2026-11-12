#include <stdio.h>

int main(int argc, char const *argv[]) {
  printf("Got %d argument(s).\n", argc);
  for (int i = 0; i < argc; ++i)
    printf("[%d]: %s\n", i, argv[i]);
  return 0;
}
