#include <stdio.h>
#include <unistd.h>

int
main (int argc, char **argv)
{
  char buffer[1024];

  fprintf(stdout, "stdout: %s\n", getcwd(buffer, 1024));
  fprintf(stderr, "stderr: %s\n", getcwd(buffer, 1024));

  return 0;
}
