#include "signal_file.h"
#include <signal.h>

int get_signal_crash(void) {
  raise(SIGSEGV);
  return 0;
}
