#include <stdio.h>

int main(int argc, char **argv) {
  FILE *f = fopen("arg0.txt", "w");
  if (f) {
    fputs(argc > 0 ? argv[0] : "", f);
    fclose(f);
  }
  return 0; // break here
}
