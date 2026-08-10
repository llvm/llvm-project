#include <stdio.h>

#ifdef _MSC_VER
#define _CRT_NONSTDC_NO_WARNINGS
#include <direct.h>
#undef getcwd
#define getcwd(buffer, length) _getcwd(buffer, length)
#else
#include <unistd.h>
#endif

int
main (int argc, char **argv)
{
  char buffer[1024];

  fprintf(stdout, "stdout: %s\n", getcwd(buffer, 1024));
  fprintf(stderr, "stderr: %s\n", getcwd(buffer, 1024));

  return 0;
}
