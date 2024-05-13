// This test is intended to create a situation in which two threads are stopped
// at a breakpoint and the debugger issues a step-out command.

#include "pseudo_barrier.h"
#include <thread>

pseudo_barrier_t g_barrier;

volatile int g_test = 0;

void stop_here() {
  g_test += 5; // Set breakpoint here
}

void recurse_a_bit_1(int count) {
  if (count == 50)
    stop_here();
  else
    recurse_a_bit_1(++count);
}

void recurse_a_bit_2(int count) {
  if (count == 50)
    stop_here();
  else
    recurse_a_bit_2(++count);
}

void *thread_func_1() {
  // Wait until both threads are running
  pseudo_barrier_wait(g_barrier);

  // Start the recursion:
  recurse_a_bit_1(0);

  // Return
  return NULL;
}

void *thread_func_2() {
  // Wait until both threads are running
  pseudo_barrier_wait(g_barrier);

  // Start the recursion:
  recurse_a_bit_2(0);

  // Return
  return NULL;
}

int main() {
  // Don't let either thread do anything until they're both ready.
  pseudo_barrier_init(g_barrier, 2);

  // Create two threads
  std::thread thread_1(thread_func_1);
  std::thread thread_2(thread_func_2);

  // Wait for the threads to finish
  thread_1.join();
  thread_2.join();

  return 0;
}
