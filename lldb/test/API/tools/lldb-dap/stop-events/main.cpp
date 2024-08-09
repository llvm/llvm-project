#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
int ready_id = 0;

void worker(int id) {
  std::cout << "Worker " << id << " executing..." << std::endl;
  // Signal the main thread to continue main thread
  {
    std::lock_guard<std::mutex> lock(mtx);
    ready_id = id; // break worker thread here
  }
  cv.notify_one();

  // Simulate some work
  std::this_thread::sleep_for(std::chrono::seconds(10));
  std::cout << "Worker " << id << " finished." << std::endl;
}

void thread_proc(int threadId) {
  std::mutex repro_mtx;
  for (;;) {
    int i = 0;
    ++i; // Set breakpoint1 here
    repro_mtx.lock();
    std::cout << "Thread " << threadId << " is running, " << i << std::endl;
    repro_mtx.unlock();

    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    i += 10; // Set breakpoint2 here
    repro_mtx.lock();
    std::cout << "Thread " << threadId << " finished" << std::endl;
    repro_mtx.unlock(); // Unlock the mutex after printing
  }
}

void run_threads_in_loop(int numThreads) {
  std::thread threads[numThreads];
  // Create threads
  for (int i = 0; i < numThreads; ++i) {
    threads[i] = std::thread(thread_proc, i);
  }
  for (int i = 0; i < numThreads; ++i) {
    threads[i].join();
  }
}

int main() {
  // Create the first worker thread
  std::thread t1(worker, 1);

  // Wait until signaled by the first thread
  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [] { return ready_id == 1; });

  // Create the second worker thread
  std::thread t2(worker, 2);

  // Wait until signaled by the second thread
  cv.wait(lock, [] { return ready_id == 2; });

  // Join the first thread to ensure main waits for it to finish
  t1.join(); // break main thread here
  t2.join();

  run_threads_in_loop(100);
  return 0;
}
