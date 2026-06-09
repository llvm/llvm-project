#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <thread>

int var() {
  std::this_thread::sleep_for(std::chrono::seconds(100)); // break here
}
int baz() { return 10 + var(); }
int bar() { return 15 + baz(); }
int foo() { return 20 + bar(); }
void work() { foo(); }

int main() {
  std::thread thread_1(work);
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
}
