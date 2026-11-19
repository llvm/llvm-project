// A thread calls pthread_exit while lldb runs an expression on it.
//
// The test controls when the thread exits by setting g_wait_for_release:
// - 0: exit_current_thread exits immediately, while only its own thread runs.
// - 1: exit_current_thread waits until releaser_func sets g_released.

#include <pthread.h>

volatile int g_wait_for_release = 0;
volatile int g_in_function = 0;
volatile int g_released = 0;

// What main gets from pthread_join: the first means the thread really exited.
#define EXITED_IN_EXPRESSION ((void *)1)
#define RETURNED_NORMALLY ((void *)2)

void exit_current_thread(void) {
  g_in_function = 1;
  if (g_wait_for_release) {
    while (!g_released)
      ;
  }
  pthread_exit(EXITED_IN_EXPRESSION);
}

void *releaser_func(void *unused) {
  while (!g_in_function)
    ;
  g_released = 1;
  return NULL;
}

void *exiting_thread_func(void *unused) {
  exit_current_thread(); // break here
  return RETURNED_NORMALLY;
}

int main(void) {
  pthread_t releaser, exiting_thread;
  pthread_create(&releaser, NULL, releaser_func, NULL);
  pthread_create(&exiting_thread, NULL, exiting_thread_func, NULL);

  void *exit_value;
  pthread_join(exiting_thread, &exit_value);
  int thread_exited = exit_value == EXITED_IN_EXPRESSION;
  // A breakpoint here would be skipped while the expression runs, a trap isn't.
  __builtin_debugtrap();
  pthread_join(releaser, NULL);
  return thread_exited ? 0 : 1;
}
