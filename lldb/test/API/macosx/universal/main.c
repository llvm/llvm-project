#include <stdio.h>
#include <unistd.h>
#include <string.h>

void
call_me()
{
  sleep(1);
}

int
main (int argc, char **argv)
{
  printf ("Hello there!\n"); // Set break point at this line.
  if (argc == 2 && strcmp(argv[1], "keep_waiting") == 0)
    while (1)
      {
        call_me();
      }
  return 0;
}
