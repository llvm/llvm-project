#include <chrono>
#include <iostream>
#include <thread>

void generate_output() {
  for (unsigned i = 1; i < 4; ++i) {
    std::cout << "Hello from stdout line " << i << std::endl;
    std::cerr << "Hello from stderr line " << i << std::endl;
  }
}

int main(int argc, char *argv[]) {
  int test_var = 42;

  // Break before output.
  int break_here = 0; // break here begin

  // Generate stdout/stderr output.
  generate_output();

  // Wait to capture output.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  return 0; // break here end
}
