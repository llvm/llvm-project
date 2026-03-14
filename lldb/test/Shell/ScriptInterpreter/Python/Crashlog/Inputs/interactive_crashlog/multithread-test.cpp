#include <iostream>
#include <mutex>
#include <thread>

int bar(int i) {
  int *j = 0;
  *j = 1;
  return i; // break here
}

int foo(int i) { return bar(i); }

void call_and_wait(int &n) {
  std::cout << "waiting for computation!" << std::endl;
  while (n != 42 * 42)
    ;
  std::cout << "finished computation!" << std::endl;
}

void compute_pow(int &n) { n = foo(n); }

int main() {
  int n = 42;
  std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  std::thread thread_1(call_and_wait, std::ref(n));
  std::thread thread_2(compute_pow, std::ref(n));
  lock.unlock();

  thread_1.join();
  thread_2.join();

  return 0;
}
