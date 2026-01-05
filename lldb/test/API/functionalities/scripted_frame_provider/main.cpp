// Multi-threaded test program for testing frame providers.

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
int ready_count = 0;
constexpr int NUM_THREADS = 2;

void thread_func(int thread_num) {
  std::cout << "Thread " << thread_num << " started\n";

  {
    std::unique_lock<std::mutex> lock(mtx);
    ready_count++;
    if (ready_count == NUM_THREADS + 1) {
      cv.notify_all();
    } else {
      cv.wait(lock, [] { return ready_count == NUM_THREADS + 1; });
    }
  }

  std::cout << "Thread " << thread_num << " at breakpoint\n"; // Break here.
}

int main(int argc, char **argv) {
  std::thread threads[NUM_THREADS];

  for (int i = 0; i < NUM_THREADS; i++) {
    threads[i] = std::thread(thread_func, i);
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    ready_count++;
    if (ready_count == NUM_THREADS + 1) {
      cv.notify_all();
    } else {
      cv.wait(lock, [] { return ready_count == NUM_THREADS + 1; });
    }
  }

  std::cout << "Main thread at barrier\n";

  for (int i = 0; i < NUM_THREADS; i++)
    threads[i].join();

  std::cout << "All threads completed\n";
  return 0;
}
