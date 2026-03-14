#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "ERROR: Too few arguments (count: %d).\n", argc - 1);
    exit(1);
  }

#if defined(_WIN32) || defined(_WIN64)
  char *cmd_opt = "/C";
#else
  char *cmd_opt = "-c";
#endif

  if (strncmp(argv[1], cmd_opt, 2)) {
    fprintf(stderr, "ERROR: Missing shell command option ('%s').\n", cmd_opt);
    exit(1);
  }

  printf("SUCCESS: %s\n", argv[0]);
  return 0;
}
