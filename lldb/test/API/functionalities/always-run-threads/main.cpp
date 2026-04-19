#include <chrono>
#include <pthread.h>
#include <thread>

static void set_thread_name(const char *name) {
#if defined(__APPLE__)
  ::pthread_setname_np(name);
#elif defined(__FreeBSD__) || defined(__linux__)
  ::pthread_setname_np(::pthread_self(), name);
#elif defined(__NetBSD__)
  ::pthread_setname_np(::pthread_self(), "%s", const_cast<char *>(name));
#endif
}

volatile int g_helper_count = 0;
volatile bool g_stop = false;

void helper_thread_func() {
  set_thread_name("always-run-helper");
  while (!g_stop) {
    g_helper_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int step_over_me() {
  int result = 0;
  for (int i = 0; i < 1000; i++)
    result += i;
  return result;
}

int main() {
  std::thread helper(helper_thread_func);
  // Give the helper thread time to start and set its name.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  int val = step_over_me(); // break here
  val += step_over_me();    // after step

  g_stop = true;
  helper.join();
  return val > 0 ? 0 : 1;
}
