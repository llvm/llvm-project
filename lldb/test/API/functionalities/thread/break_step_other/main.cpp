#include <thread>
#include "pseudo_barrier.h"

// Barrier for starting the thread and reaching the loop in main.
pseudo_barrier_t g_barrier;
volatile int g_foo = 0;

void thread_func() {
  // Wait until all the threads are running
  pseudo_barrier_wait(g_barrier);
  g_foo = 1; // thread break here
}

int main() {
  g_foo = 0; // main break here

  pseudo_barrier_init(g_barrier, 2);
  std::thread t(thread_func);
  pseudo_barrier_wait(g_barrier);

  // A dummy loop to have something to step through.
  volatile int i = 0;
  while (g_foo == 0)
    ++i;
  t.join();
  return 0;
}
