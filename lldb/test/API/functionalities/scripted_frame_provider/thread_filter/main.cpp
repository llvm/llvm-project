#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#define NUM_THREADS 3

std::atomic<int> g_barrier(NUM_THREADS);
volatile bool g_spin = true;

void thread_work() {
  --g_barrier;
  while (g_barrier.load() > 0)
    ;
#ifdef _WIN32
  // Avoid a timeout on Windows.
  // It seems the debugger needs some time
  // to init the breakpoint for the thread.
  // This test is flaky on Windows with 1 sec sleep.
  // Moving the break comment below did not help.
  std::this_thread::sleep_for(std::chrono::seconds(3));
#endif
  while (g_spin) // break in thread
    std::this_thread::yield();
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++)
    threads.emplace_back(thread_work);
  for (auto &t : threads)
    t.join();
  return 0;
}
