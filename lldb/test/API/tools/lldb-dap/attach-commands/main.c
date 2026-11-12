#include "attach.h"

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

static volatile int is_ready = 0;

int main(int argc, char const *argv[]) {
  lldb_enable_attach();

  while (!is_ready) {
    sleep_ms(50);
  }

  return 0;
}
