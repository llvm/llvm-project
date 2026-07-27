#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
  // Parse args: an optional positional value, plus the flag --read-stdin.
  // The flag is set by tests that have wired up stdin redirection. Without
  // it we never call getline, which would otherwise block on a pipe that's
  // open but empty.
  std::string arg_text;
  bool read_stdin = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--read-stdin") == 0) {
      read_stdin = true;
    } else if (arg_text.empty()) {
      arg_text = argv[i];
    }
  }

  if (!arg_text.empty()) {
    std::cout << "[STDOUT][FROM_ARGV]: " << arg_text << "\n";
    std::cerr << "[STDERR][FROM_ARGV]: " << arg_text << "\n";
  }
  if (const char *env = std::getenv("FROM_ENV")) {
    std::cout << "[STDOUT][FROM_ENV]: " << env << "\n";
    std::cerr << "[STDERR][FROM_ENV]: " << env << "\n";
  }
  if (read_stdin) {
    std::string line;
    if (std::getline(std::cin, line)) {
      std::cout << "[STDOUT][FROM_STDIN]: " << line << "\n";
      std::cerr << "[STDERR][FROM_STDIN]: " << line << "\n";
    }
  }
  return 0;
}
