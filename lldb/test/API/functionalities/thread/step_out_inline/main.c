volatile int g;

__attribute__((noinline)) void sink(int x) { g = x; }

static inline __attribute__((always_inline)) void level2(int b) {
  sink(b);
  g = b + 1;
}

static inline __attribute__((always_inline)) void level1(int c) {
  level2(c + 1);
  g = c + 1; // after level2
}

int main(void) {
  level1(42);
  return 0; // after level1
}
