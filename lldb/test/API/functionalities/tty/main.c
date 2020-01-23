#include <stdio.h>

int
main() { int argc = 0; char **argv = (char **)0;

  for (int i = 0; i < argc; i++) {
    printf("%d: %s.\n", i, argv[i]);
  }
  return 0;
}
