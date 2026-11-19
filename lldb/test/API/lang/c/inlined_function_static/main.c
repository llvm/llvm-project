volatile int g_sink;

static inline __attribute__((always_inline)) void inlined(void) {
  static int calls = 40;
  ++calls;
  g_sink = calls; // break inlined
}

// Has an out-of-line copy and is also inlined into main.
static inline __attribute__((always_inline)) void both(void) {
  static int total = 100;
  ++total;
  {
    static int nested = 200;
    ++nested;
    g_sink = nested; // break nested
  }
  g_sink = total; // break both
}

int main(void) {
  void (*volatile out_of_line)(void) = both;
  inlined();
  both();
  out_of_line();
  return 0;
}
