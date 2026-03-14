#include <pthread.h>
#if defined(__OpenBSD__)
#include <pthread_np.h>
#endif
#include <signal.h>

void set_thread_name(const char *name) {
#if defined(__APPLE__)
  ::pthread_setname_np(name);
#elif defined(__FreeBSD__) || defined(__linux__)
  ::pthread_setname_np(::pthread_self(), name);
#elif defined(__NetBSD__)
  ::pthread_setname_np(::pthread_self(), "%s", const_cast<char *>(name));
#elif defined(__OpenBSD__)
  ::pthread_set_name_np(::pthread_self(), name);
#endif
}

int main() {
  set_thread_name("hello world");
  raise(SIGINT);
  set_thread_name("goodbye world");
  raise(SIGINT);
  return 0;
}
