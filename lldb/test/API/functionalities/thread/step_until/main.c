#include <stdio.h>

void call_me(int argc)
{
  printf ("At the start, argc: %d.\n", argc);

  if (argc < 2)
    printf("Less than 2.\n");
  else
    printf("Greater than or equal to 2.\n");
}

int
main(int argc, char **argv)
{
  call_me(argc);
  printf("Back out in main.\n");

  return 0;
}
