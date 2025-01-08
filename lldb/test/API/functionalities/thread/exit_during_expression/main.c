#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

static unsigned int g_timeout = 1000000;

extern int usleep(unsigned int);

int function_to_call() {

  errno = 0;
  while (1) {
    int result = usleep(g_timeout);
    if (errno != EINTR)
      break;
  }

  pthread_exit((void *)10);

  return 20; // Prevent warning
}

void *exiting_thread_func(void *unused) {
  function_to_call(); // Break here and cause the thread to exit
  return NULL;
}

int main() {
  char *exit_ptr;
  pthread_t exiting_thread;

  pthread_create(&exiting_thread, NULL, exiting_thread_func, NULL);

  pthread_join(exiting_thread, &exit_ptr);
  int ret_val = (int)exit_ptr;
  usleep(g_timeout * 4); // Make sure in the "run all threads" case
                         // that we don't run past our breakpoint.
  return ret_val;        // Break here to make sure the thread exited.
}
