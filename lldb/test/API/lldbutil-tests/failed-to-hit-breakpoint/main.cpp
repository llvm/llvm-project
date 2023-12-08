#include <iostream>
#include <thread>

int main(int argc, char **argv) {
  // Print the string that the test looks for to make sure stdout and stderr
  // got recorded.
  std::cout << "stdout_needle" << std::flush;
  std::cerr << "stderr_needle" << std::flush;

  // Work around a timing issue that sometimes prevents stderr from being
  // captured.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // This is unreachable during normal test execution as we don't pass any
  // (or +100) arguments. This still needs to be theoretically reachable code
  // so that the compiler will generate code for this (that we can set a
  // breakpoint on).
  if (argc > 100)
    return 1; // break here
  return 0;
}
