#include "attach.h"
#include <chrono>
#include <thread>

volatile int g_val = 12345;

int main(int argc, char const *argv[]) {
  lldb_enable_attach();

  // This provides a breakpoint we hit every 50ms that the tests can hit.
  std::chrono::milliseconds poll_time(50);
  unsigned total_wait_in_sec = 60;
  unsigned total_wait_time = total_wait_in_sec * (1000 / poll_time.count());

  for (unsigned wait = 0; wait < total_wait_time; wait++)
    std::this_thread::sleep_for(poll_time); // Waiting to be attached...
}
