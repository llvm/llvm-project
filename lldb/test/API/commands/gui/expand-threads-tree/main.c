#include <pthread.h>

void *thread_start_routine(void *arg) { return NULL; }

int main() {
  pthread_t thread;
  pthread_create(&thread, NULL, thread_start_routine, NULL);
  pthread_join(thread, NULL);
  return 0;
}
