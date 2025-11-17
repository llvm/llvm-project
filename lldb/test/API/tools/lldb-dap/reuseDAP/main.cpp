#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdexcept>

int main(int argc, char const *argv[], char const *envp[]) {
  for (int i = 0; i < argc; ++i)
    printf("arg[%i] = \"%s\"\n", i, argv[i]);
  for (int i = 0; envp[i]; ++i)
    printf("env[%i] = \"%s\"\n", i, envp[i]);
  char *cwd = getcwd(NULL, 0);
  printf("cwd = \"%s\"\n", cwd); // breakpoint 1
  free(cwd);
  cwd = NULL; // breakpoint 2
  throw new std::runtime_error("This is a C++ exception");
  return 0;
}
