#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
int ready_thread_id = 0;
int signal_main_thread = 0;

void worker(int id) {
  std::cout << "Worker " << id << " executing..." << std::endl;

  // lldb test should change signal_main_thread to true to break the loop.
  while (!signal_main_thread) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Signal the main thread to continue main thread
  {
    std::lock_guard<std::mutex> lock(mtx);
    ready_thread_id = id; // break worker thread here
  }
  cv.notify_one();

  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "Worker " << id << " finished." << std::endl;
}

void deadlock_func(std::unique_lock<std::mutex> &lock) {
  int i = 10;
  ++i;             // Set interrupt breakpoint here
  printf("%d", i); // Finish step-over from inner breakpoint
  auto func = [] { return ready_thread_id == 1; };
  cv.wait(lock, func);
}

int simulate_thread() {
  std::thread t1(worker, 1);

  std::unique_lock<std::mutex> lock(mtx);
  deadlock_func(lock); // Set breakpoint1 here

  std::thread t2(worker, 2); // Finish step-over from breakpoint1

  cv.wait(lock, [] { return ready_thread_id == 2; });

  t1.join();
  t2.join();

  std::cout << "Main thread continues..." << std::endl;

  return 0;
}

int bar() { return 54; }

int foo(const std::string p1, int extra) { return p1.size() + extra; }

int main(int argc, char *argv[]) {
  std::string ss = "this is a string for testing",
              ls = "this is a long string for testing";
  foo(ss.size() % 2 == 0 ? ss : ls, bar()); // Set breakpoint2 here

  simulate_thread(); // Finish step-over from breakpoint2

  return 0;
}
