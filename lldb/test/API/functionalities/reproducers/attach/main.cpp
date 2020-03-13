#include <chrono>
#include <stdio.h>
#include <thread>

using std::chrono::seconds;

int main(int argc, char const *argv[]) {
  lldb_enable_attach();

  // Create the synchronization token.
  FILE *f;
  if (f = fopen(argv[1], "wx")) {
    fputs("\n", f);
    fflush(f);
    fclose(f);
  } else
    return 1;

  while (true) {
    std::this_thread::sleep_for(seconds(1));
  }

  return 0;
}
