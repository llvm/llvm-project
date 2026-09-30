volatile int g;

__attribute__((noinline)) void sink(int x) { g = x; }

static inline __attribute__((always_inline)) void level3(int a) {
  g = a; // before sink
  sink(a);
  g = a + 1; // break in level3
  g = a + 2; // until in level3
}

static inline __attribute__((always_inline)) void level2(int b) {
  g = b;
  level3(b + 1);
  g = b + 1; // until in level2
}

static inline __attribute__((always_inline)) void level1(int c) {
  g = c;
  level2(c + 1);
}

__attribute__((noinline)) void outer(void) { level1(42); }

int main(void) {
  outer();
  return 0;
}
