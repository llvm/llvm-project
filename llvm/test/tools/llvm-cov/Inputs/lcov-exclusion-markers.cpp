#define EXCLUDED_MACRO(x) ((x) ? 1 : 0) // LLVM_COVERAGE_EXCLUDE_LINE

int excluded_line(int x) {
  return x ? 1 : 0; // LCOV_EXCL_LINE
}

int partially_excluded_block(int x) {
  // LLVM_COVERAGE_EXCLUDE_START
  if (x)
    return 1;
  return 0; // LCOV_EXCL_STOP
}

static int excluded_function(int x) { return x; } // LLVM_COVERAGE_EXCLUDE_LINE

int included(int x) {
  if (x)
    return 1;
  return 0;
}

template <typename T> T partially_excluded_template(T x) {
  return x ? 1 : 0; // LCOV_EXCL_LINE
}

int main() {
  return excluded_line(0) + partially_excluded_block(0) +
         excluded_function(0) + included(0) + EXCLUDED_MACRO(0) +
         partially_excluded_template(0) + partially_excluded_template(0L);
}
