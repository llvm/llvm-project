#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "wx");
  if (f) {
    fputs("\n", f);
    fflush(f);
    fclose(f);
  } else {
    return 1;
  }

  pause();
  return 0;
}
