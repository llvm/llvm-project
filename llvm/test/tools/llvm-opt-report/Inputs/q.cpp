void bar();
void foo(int n) {
  for (int i = 0; i < n; ++i)
    bar();
}

void quack() {
  foo(4);
}

void quack2() {
  foo(4);
}

