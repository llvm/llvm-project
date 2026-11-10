#include <stdio.h>
#include <windows.h>

int main(int argc, char **argv) {
  if (argc > 1) {
    // Create the synchronization token, then wait to be attached.
    FILE *f = fopen(argv[1], "w");
    if (!f)
      return 1;
    fputs("\n", f);
    fclose(f);
    while (1)
      Sleep(1000);
  }
  return 0; // break here
}
