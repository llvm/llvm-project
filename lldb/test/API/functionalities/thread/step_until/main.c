#include <stdio.h>

/* The return statements are purposefully so simple, and
 * unrelated to the program, just to achieve consistent
 * debug line tables, across platforms, that are not
 * dependent on compiler optimzations. */
int call_me(int argc) {
  printf ("At the start, argc: %d.\n", argc);

  if (argc < 2)
    return 1; /* Less than 2. */
  else
    return argc; /* Greater than or equal to 2. */
}

int
main(int argc, char **argv)
{
  int res = 0;
  res = call_me(argc); /* Back out in main. */
  if (res)
    printf("Result: %d. \n", res);

  return 0;
}
