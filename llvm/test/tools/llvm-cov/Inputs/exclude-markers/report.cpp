void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

void func() { // LCOV_EXCL_LINE
} // LCOV_EXCL_LINE

int main() {
  foo(false);
  bar();
  return 0;
}
