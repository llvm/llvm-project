#include <stdio.h>
#ifdef _WIN32
#include <process.h>
#include <windows.h>
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
  }

  printf("pid = %i\n", getpid());
#ifdef _WIN32
  Sleep(10 * 1000);
#else
  sleep(10);
#endif
  return 0; // breakpoint 1
}
