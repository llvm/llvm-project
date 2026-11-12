#include <stdio.h>
#include <stdlib.h>

int
main (int argc, char **argv)
{
  char buffer[1024];

  fgets (buffer, sizeof (buffer), stdin);
  fprintf (stdout, "%s", buffer);

  
  fgets (buffer, sizeof (buffer), stdin);
  fprintf (stderr, "%s", buffer);

  return 0;
}
