#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
const char *const SEGFAULT_COMMAND = "segfault";
const char *const ABORT_COMMAND = "abort";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "expected at least one command provided on the command line"
              << std::endl;
  }

  // Process command line args.
  for (int i = 1; i < argc; ++i) {
    const char *const command = argv[i];
    if (std::strstr(command, SEGFAULT_COMMAND)) {
      // Perform a null pointer access.
      int *const null_int_ptr = nullptr;
      *null_int_ptr = 0xDEAD;
    } else if (std::strstr(command, ABORT_COMMAND)) {
      std::abort();
    } else {
      std::cout << "Unsupported command: " << command << std::endl;
    }
  }

  return 0;
}
