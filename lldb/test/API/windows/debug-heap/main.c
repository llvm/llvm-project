#include <stdio.h>
#include <string.h>

int main(int argc, char **argv, char **envp) {
  for (char **p = envp; *p; ++p) {
    if (strstr(*p, "_NO_DEBUG_HEAP=") == *p)
      printf("%s\n", *p);
  }
  return 0;
}
