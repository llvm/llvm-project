#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main (int argc, char **argv)
{
  char *evil = getenv("EVIL");

  return 0;  // Set breakpoint here.
}
