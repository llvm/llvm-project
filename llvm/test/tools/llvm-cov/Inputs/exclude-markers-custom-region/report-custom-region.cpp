void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

// BEGIN_NO_COV
void func() {
}
// END_NO_COV

int main() {
  foo(false);
  bar();
  return 0;
}
