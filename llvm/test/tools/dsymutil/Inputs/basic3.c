/* For compilation instructions see basic1.c. */

volatile int val;

extern int foo(int);

int unused2() {
  return foo(val);
}

static int inc() {
  return ++val;
}

__attribute__((noinline))
int bar(int arg) {
  if (arg > 42)
    return inc();
  return foo(val + arg);
}
