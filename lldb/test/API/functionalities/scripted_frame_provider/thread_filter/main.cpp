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
  std::this_thread::sleep_for(
      std::chrono::seconds(1)); // avoid timeout on Windows
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
