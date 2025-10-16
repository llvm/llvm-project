#include <stdio.h>

void foo(int a, int b) { printf("%d %d\n", a, b); }

void bar(int *ptr) { printf("%d\n", *ptr); }

void nested(int *ptr) { bar(ptr); }

void baz(int *ptr) { nested(ptr); }

int main(int argc, const char *argv[]) {
  foo(42, 56);
  int i = 78;
  bar(&i);
  baz(&i);
  return 0;
}
