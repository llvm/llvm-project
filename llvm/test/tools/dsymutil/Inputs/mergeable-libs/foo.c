int foo_global = 42;
static int foo_static = 7;
_Thread_local int foo_tls = 3;

static inline __attribute__((always_inline)) int foo_helper(int a) {
  return a * foo_global;
}

int foo(int a) {
  static int foo_calls;
  foo_calls++;
  return foo_helper(a) + foo_static++ + foo_tls + foo_calls;
}

int foo_unused(void) { return 1; }
