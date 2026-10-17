#include <atomic>
#include <thread>
#include <vector>

#define NUM_THREADS 3

std::atomic<int> g_barrier(NUM_THREADS);
volatile bool g_spin = true;

void thread_work() {
  --g_barrier;
  while (g_barrier.load() > 0)
    ;
  // Break in the loop body, which runs on every iteration. On the `while` line,
  // the breakpoint can land on the jump into the loop, which each thread runs
  // only once.
  while (g_spin)
    std::this_thread::yield(); // break in thread
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++)
    threads.emplace_back(thread_work);
  for (auto &t : threads)
    t.join();
  return 0;
}
