// Test simulates situation when suspended thread could stop process
// where thread that is a real reason of stop says process
// should not stop in it's action handler.

#include <chrono>
#include <thread>

void thread1() {
  // Will be suspended at breakpoint stop
  // Set first breakpoint here
}

void thread2() {
  /*
   Prevent threads from stopping simultaneously
   */
  std::this_thread::sleep_for(std::chrono::seconds(1));
  // Set second breakpoint here
}

int main() {
  // Create a thread
  std::thread thread_1(thread1);

  // Create another thread.
  std::thread thread_2(thread2);

  // Wait for the thread that was not suspeneded
  thread_2.join();

  // Wait for thread that was suspended
  thread_1.join(); // Set third breakpoint here

  return 0;
}
