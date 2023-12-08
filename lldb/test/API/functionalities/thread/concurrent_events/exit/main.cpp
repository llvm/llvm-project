#include "pseudo_barrier.h"
#include <cstdlib>
#include <thread>

// Use low-level exit functions to achieve predictable timing.
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>

void exit_thread(int status) { syscall(SYS_exit, status); }
void exit_process(int status) { syscall(SYS_exit_group, status); }
#else
#error Unimplemented
#endif

pseudo_barrier_t g_barrier;

void thread_func() {
  pseudo_barrier_wait(g_barrier);
  exit_thread(42);
}

int main() {
  pseudo_barrier_init(g_barrier, 2);
  std::thread(thread_func).detach();

  pseudo_barrier_wait(g_barrier);

  exit_process(47);
}
