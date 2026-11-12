// This test will present lldb with two threads one of which the test will
// overlay with an OSPlugin thread.  Then we'll do a step out on the thread_1,
// but arrange to hit a breakpoint in main before the step out completes. At
// that point we will not report an OS plugin thread for thread_1. Then we'll
// run again and hit the step out breakpoint.  Make sure we haven't deleted
// that, and recognize it.

#include <condition_variable>
#include <mutex>
#include <stdio.h>
#include <thread>

static int g_value = 0; // I don't have access to the real threads in the
                        // OS Plugin, and I don't want to have to count
                        // StopID's. So I'm using this value to tell me which
                        // stop point the program has reached.
std::mutex g_mutex;
std::condition_variable g_cv;
static int g_condition = 0; // Using this as the variable backing g_cv
                            // to prevent spurious wakeups.

void step_out_of_here() {
  std::unique_lock<std::mutex> func_lock(g_mutex);
  // Set a breakpoint:first stop in thread - do a step out.
  g_condition = 1;
  g_cv.notify_one();
  g_cv.wait(func_lock, [&] { return g_condition == 2; });
}

void *thread_func() {
  // Do something
  step_out_of_here();

  // Return
  return NULL;
}

int main() {
  // Lock the mutex so we can block the thread:
  std::unique_lock<std::mutex> main_lock(g_mutex);
  // Create the thread
  std::thread thread_1(thread_func);
  g_cv.wait(main_lock, [&] { return g_condition == 1; });
  g_value = 1;
  g_condition = 2;
  // Stop here and do not make a memory thread for thread_1.
  g_cv.notify_one();
  g_value = 2;
  main_lock.unlock();

  // Wait for the threads to finish
  thread_1.join();

  return 0;
}
