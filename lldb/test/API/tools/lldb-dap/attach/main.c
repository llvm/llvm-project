#include "attach.h"
#include <stdio.h>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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

    // Wait on input from stdin.
    // when lldb connects to the process, on MacOS getchar() is interupted
    // and sets the stream's error indicator (EINTR).
    // ignore that and keep waiting until we actually receive a character.
    while (1) {
      int c = getchar();
      if (c == EOF && ferror(stdin)) {
        clearerr(stdin);
        continue;
      }
      printf("char = %c\n", c);
      break;
    }
  }

  printf("pid = %i\n", getpid());
  return 0; // breakpoint 1
}
