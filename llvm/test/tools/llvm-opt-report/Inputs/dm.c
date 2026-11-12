void bar(void);
void foo(int n) {
  if (n) { bar(); } else { while (1) {} }
}

void quack(void) {
  foo(0);
}

void quack2(void) {
  foo(4);
}

