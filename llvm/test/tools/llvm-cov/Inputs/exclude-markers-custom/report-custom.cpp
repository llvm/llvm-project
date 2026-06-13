void foo(bool cond) {
  if (cond) {
  }
}

void bar() {
}

void func() { // MY_SKIP
} // MY_SKIP

int main() {
  foo(false);
  bar();
  return 0;
}
