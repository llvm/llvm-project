// Threads stop at one breakpoint together. Each one first records the ID the
// system gave it, so the test can check that the frame lldb shows for a thread
// is that thread's own.

#include <cstdio>
#include <thread>
#include <vector>

// thread.h needs pthread_t on Darwin, which <thread> declares.
#include "pseudo_barrier.h"
#include "thread.h"

pseudo_barrier_t g_barrier;

void print_index(int index) {
  uint64_t tid = get_thread_id();
  // Release the threads together, so that several reach the breakpoint at once.
  pseudo_barrier_wait(g_barrier);
  printf("thread %d: %llu\n", index, (unsigned long long)tid); // break here
}

int main() {
  constexpr int num_threads = 32;
  pseudo_barrier_init(g_barrier, num_threads);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i)
    threads.emplace_back(print_index, i);
  for (std::thread &thread : threads)
    thread.join();
  return 0;
}
