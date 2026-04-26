#include <thread>
#include <unistd.h>
#include <vector>

#define NUM_THREADS 3

void thread_work() {
  pause(); // break in thread
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < NUM_THREADS; i++)
    threads.emplace_back(thread_work);
  for (auto &t : threads)
    t.join();
  return 0;
}
