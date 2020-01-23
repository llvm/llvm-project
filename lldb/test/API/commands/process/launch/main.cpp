#include <stdio.h>
#include <stdlib.h>

int
main() { int argc = 0; char **argv = (char **)0;

  char buffer[1024];

  fgets (buffer, sizeof (buffer), stdin);
  fprintf (stdout, "%s", buffer);

  
  fgets (buffer, sizeof (buffer), stdin);
  fprintf (stderr, "%s", buffer);

  return 0;
}
