#include <cassert>
#include <chrono>
#include <cstdlib>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

int main() {
  pid_t fork_result = fork(); // break here
  assert(fork_result >= 0);
  if (fork_result == 0) {
    // child
    _exit(47);
  }
  // parent
  // Use polling to avoid blocking if the child is not actually resumed.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  std::chrono::milliseconds poll_interval{10};
  while (std::chrono::steady_clock::now() < deadline) {
    int status;
    pid_t waitpid_result = waitpid(fork_result, &status, WNOHANG);
    if (waitpid_result == fork_result)
      return 0;
    assert(waitpid_result == 0);
    std::this_thread::sleep_for(poll_interval);
    poll_interval *= 2;
  }
  abort();
}
