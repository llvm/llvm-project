#include "attach.h"
#include <fcntl.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

// Called from the debugger as "(int)marker_function()". It returns the number
// of calls so far, so the n-th attach cycle (counting from 0) sees n + 1 only
// if the expression really ran in this process.
volatile int marker_calls = 0;

int marker_function(void) {
  marker_calls = marker_calls + 1;
  return marker_calls;
}

int main(int argc, char const *argv[]) {
  // Run at background priority, which makes the detach race more likely. If
  // that fails, exit before the heartbeat file exists, so the test fails
  // waiting for it instead of passing without background priority.
  if (setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG) != 0)
    return 1;

  lldb_enable_attach();

  if (argc < 2)
    return 1;

  // The heartbeat file. Its appearance tells the test the process is up, and
  // its growth after a detach proves the process is running again.
  int fd = open(argv[1], O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0)
    return 1;

  for (unsigned long tick = 0;; ++tick) {
    char line[32];
    int len = snprintf(line, sizeof(line), "TICK=%lu\n", tick);
    if (len < 0 || len >= (int)sizeof(line))
      return 1;
    // One write(2) per line, with no stdio buffering in between, so every
    // line is in the file as soon as this call returns.
    if (write(fd, line, (size_t)len) != len)
      return 1;
    usleep(100000);
  }
}
