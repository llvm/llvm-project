// Test program to verify libunwind ret injection feature for execution flow
// rebalancing.
//
// This test creates a multi-frame call stack and throws a C++ exception to
// trigger libunwind's two-phase exception handling. The test verifies that
// libunwind correctly injects the right amount of 'ret' instructions to
// rebalance the execution flow when returning to the landing pad, which is
// important for Apple Processor Trace analysis.

#include <cstdio>
#include <exception>
#include <stdexcept>

// Marker functions with noinline to ensure they appear in the stack.
static void __attribute__((noinline)) func_d() {
  printf("In func_d, about to throw exception\n");
  throw std::runtime_error("test exception");
}

static void __attribute__((noinline)) func_c() {
  printf("In func_c\n");
  func_d();
}

static void __attribute__((noinline)) func_b() {
  printf("In func_b\n");
  func_c();
}

static void __attribute__((noinline)) func_a() {
  printf("In func_a\n");
  func_b();
}

int main(int argc, char *argv[]) {
  try {
    printf("In main, about to call func_a\n");
    func_a();
    printf("ERROR: Should not reach here\n");
    return 1;
  } catch (const std::exception &e) {
    printf("Caught exception in main: %s\n", e.what());
    return 0;
  }
}
