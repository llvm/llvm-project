void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

// LCOV_EXCL_START
void func() {
}
// LCOV_EXCL_STOP

int main() {
  foo(false);
  bar();
  return 0;
}
