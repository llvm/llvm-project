#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
  const bool use_stdin = argc <= 1;
  const char *use_env = std::getenv("FROM_ENV");

  if (use_env != nullptr) { // from environment variable
    std::cout << "[STDOUT][FROM_ENV]: " << use_env;
    std::cerr << "[STDERR][FROM_ENV]: " << use_env;

  } else if (use_stdin) { // from standard in
    std::string line;
    std::getline(std::cin, line);
    std::cout << "[STDOUT][FROM_STDIN]: " << line;
    std::cerr << "[STDERR][FROM_STDIN]: " << line;

  } else { // from argv
    const char *first_arg = argv[1];
    std::cout << "[STDOUT][FROM_ARGV]: " << first_arg;
    std::cerr << "[STDERR][FROM_ARGV]: " << first_arg;
  }
  return 0;
}
