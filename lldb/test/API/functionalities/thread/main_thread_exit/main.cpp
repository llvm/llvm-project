#include <thread>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>

void exit_thread(int result) { syscall(SYS_exit, result); }
#else
#error Needs OS-specific implementation
#endif

int call_me() { return 12345; }

void thread() {
  std::this_thread::sleep_for(
      std::chrono::seconds(10)); // Let the main thread exit.
  exit_thread(42);               // break here
}

int main() {
  std::thread(thread).detach();
  exit_thread(47);
}
