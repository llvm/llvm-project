// Multi-threaded test program for testing frame providers.

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
int ready_count = 0;
constexpr int NUM_THREADS = 2;

int foo(int x) {
  int foo_local = x * 2;
  int foo_result = foo_local + 1;
  return foo_result; // Break in foo.
}

int bar(int x) {
  int bar_local = x * x;
  int bar_result = bar_local - 3;
  return bar_result; // Break in bar.
}

int baz(int x) {
  int baz_local = x + 7;
  int baz_result = baz_local / 2;
  return baz_result; // Break in baz.
}

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

  // Used as an existing C++ variable we can anchor on.
  int variable_in_main = 123;
  (void)variable_in_main; // Breakpoint for variable tests.

  // Call foo for first breakpoint.
  int result_foo = foo(10);
  (void)result_foo;

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

  // Call bar for second breakpoint.
  int result_bar = bar(5);
  (void)result_bar;

  // Call baz for third breakpoint.
  int result_baz = baz(11);
  (void)result_baz;

  for (int i = 0; i < NUM_THREADS; i++)
    threads[i].join();

  std::cout << "All threads completed\n";
  return 0;
}
