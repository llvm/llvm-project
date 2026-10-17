#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "print-pid") == 0) {
      std::fprintf(stderr, "PID: %d\n", getpid());
      std::fflush(stderr);
    } else if (std::strncmp(argv[i], "sleep:", 6) == 0) {
      int seconds = std::atoi(argv[i] + 6);
      std::this_thread::sleep_for(std::chrono::seconds(seconds));
    }
  }
  return 0;
}
