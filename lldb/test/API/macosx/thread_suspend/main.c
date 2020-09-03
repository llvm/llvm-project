#include <pthread.h>
#include <mach/thread_act.h>
#include <unistd.h>

pthread_mutex_t suspend_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t signal_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t signal_cond = PTHREAD_COND_INITIALIZER;

int g_running_count = 0;

int
function_to_call() {
  return g_running_count;
}

void *
suspend_func (void *unused) {
  pthread_setname_np("Look for me");
  pthread_cond_signal(&signal_cond);
  pthread_mutex_lock(&suspend_mutex);

  return NULL; // We allowed the suspend thread to run
}

void *
running_func (void *input) {
  while (g_running_count < 10) {
    usleep (100);
    g_running_count++;  // Break here to show we can handle breakpoints
  }
  return NULL;
}

int
main()
{
  pthread_t suspend_thread; // Stop here to get things going

  pthread_mutex_lock(&suspend_mutex);
  pthread_mutex_lock(&signal_mutex);
  pthread_create(&suspend_thread, NULL, suspend_func, NULL);

  pthread_cond_wait(&signal_cond, &signal_mutex);
  
  mach_port_t th_port = pthread_mach_thread_np(suspend_thread);
  thread_suspend(th_port);

  pthread_mutex_unlock(&suspend_mutex);

  pthread_t running_thread;
  pthread_create(&running_thread, NULL, running_func, NULL);
  
  pthread_join(running_thread, NULL);
  thread_resume(th_port);             // Break here after thread_join
  
  pthread_join(suspend_thread, NULL);
  return 0; // Break here to make sure the thread exited normally
}
