#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() { int argc = 0; char **argv = (char **)0;

  char *evil = getenv("EVIL");

  return 0;  // Set breakpoint here.
}
