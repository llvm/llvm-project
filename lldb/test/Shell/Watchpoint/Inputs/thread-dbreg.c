#include <pthread.h>

int g_watchme = 0;

void *thread_func(void *arg) {
  /* watchpoint trigger from subthread */
  g_watchme = 2;
  return 0;
}

int main() {
  pthread_t thread;
  if (pthread_create(&thread, 0, thread_func, 0))
    return 1;

  /* watchpoint trigger from main thread */
  g_watchme = 1;

  if (pthread_join(thread, 0))
    return 2;

  return 0;
}
