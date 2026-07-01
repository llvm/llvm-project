#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <thread>

void sleep_for_a_tiny_bit() {
  std::this_thread::sleep_for(std::chrono::seconds(1));
}

int var() {
  // Make all the worker threads stop here, so we can do
  // backtraces with a few stack frames on each thread.
  std::this_thread::sleep_for(std::chrono::seconds(100));
  return 0;
}

int baz() { return 10 + var(); }
int bar() { return 15 + baz(); }
int foo() { return 20 + bar(); }
void work_primary_thread() {
  // Allow all threads to get started, and get to their
  // longer sleep()
  sleep_for_a_tiny_bit();
  foo(); // break here
}
void work() { foo(); }

int main() {
  std::thread thread_1(work_primary_thread);
  std::thread thread_2(work);
  std::thread thread_3(work);
  std::thread thread_4(work);
  std::thread thread_5(work);

  std::this_thread::sleep_for(std::chrono::seconds(20));

  thread_5.join();
  thread_4.join();
  thread_3.join();
  thread_2.join();
  thread_1.join();
  return 0;
}
