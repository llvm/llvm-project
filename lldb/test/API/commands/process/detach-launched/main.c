#include <unistd.h>

int main(void) {
  int seconds = 3; // break here
  // Keep running for a while after the debugger detaches, without writing to
  // the terminal.
  sleep(seconds);
  return 0;
}
