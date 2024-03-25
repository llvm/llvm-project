#include <stdio.h>
#include <unistd.h>

int main(int argc, char const *argv[]) {
  lldb_enable_attach();

  if (argc >= 2) {
    // Create the synchronization token.
    FILE *f = fopen(argv[1], "wx");
    if (!f)
      return 1;
    fputs("\n", f);
    fflush(f);
    fclose(f);
  }

  printf("pid = %i\n", getpid());
  sleep(10);
  return 0; // breakpoint 1
}
