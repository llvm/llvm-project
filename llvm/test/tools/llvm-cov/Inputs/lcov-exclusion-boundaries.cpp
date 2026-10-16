#include "lcov-exclusion-boundaries.h"

int resumes_after_excluded_line() {
  int x = 0; // LCOV_EXCL_LINE
  return x;
}

int partially_excluded_function() { // LCOV_EXCL_LINE
  return 1;
}

int main() {
  return resumes_after_excluded_line() + partially_excluded_function() +
         excluded_header_function();
}
