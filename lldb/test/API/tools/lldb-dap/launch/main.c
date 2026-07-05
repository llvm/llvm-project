#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <direct.h>
#define environ _environ
#else
#include <unistd.h>
extern char **environ;
#endif

// Read the environment from `environ` rather than a third main parameter: not
// every target wires a 3-argument main to the entry point (wasi wires up only
// the 0- and 2-argument forms).
int main(int argc, char const *argv[]) {
  for (int i = 0; i < argc; ++i)
    printf("arg[%i] = \"%s\"\n", i, argv[i]);
  for (int i = 0; environ[i]; ++i)
    printf("env[%i] = \"%s\"\n", i, environ[i]);
  char *cwd = getcwd(NULL, 0);
  printf("cwd = \"%s\"\n", cwd); // breakpoint 1
  free(cwd);
  cwd = NULL;
  return 0; // breakpoint 2
}
