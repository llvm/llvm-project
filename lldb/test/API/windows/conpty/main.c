#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Mode "basic"  : print a single well-known line then exit.
// Mode "large"  : print NUM_LINES numbered lines then exit; used to verify
//                 that multi-chunk reads over the ConPTY pipe lose no data.

#define NUM_LINES 500

int main(int argc, char *argv[]) {
  if (strcmp(argv[1], "basic") == 0) {
    printf("Hello from ConPTY\n");
    fflush(stdout);
  } else if (strcmp(argv[1], "large") == 0) {
    for (int i = 0; i < NUM_LINES; i++)
      printf("line %04d\n", i);
    fflush(stdout);
  }

  return 0;
}
