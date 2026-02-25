#include <stdio.h>

int main() {
  // Print a large amount of output to increase the chance that data is still in
  // the ConPTY pipe buffer when the process exits. The test verifies that all
  // lines are received, including the final marker.
  for (int i = 0; i < 100; i++)
    printf("line %d: the quick brown fox jumps over the lazy dog\n", i);
  printf("DONE\n");
  return 0;
}
