#include <thread>

#include "baz.h"

std::condition_variable cv;
std::mutex mutex;

int bar(int i) {
  int j = i * i;
  return j;
}

int foo(int i) { return bar(i); }

void compute_pow(int &n) {
  std::unique_lock<std::mutex> lock(mutex);
  n = foo(n);
  lock.unlock();
  cv.notify_one(); // waiting thread is notified with n == 42 * 42, cv.wait
                   // returns
}

void call_and_wait(int &n) { baz(n, mutex, cv); }

int main() {
  int n = 42;
  std::thread thread_1(call_and_wait, std::ref(n));
  std::thread thread_2(compute_pow, std::ref(n));

  thread_1.join();
  thread_2.join();

  return 0;
}
