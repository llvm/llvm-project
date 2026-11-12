volatile int x;
volatile int depth = 3;

void __attribute__((noinline)) sink() { x++; /* break in sink */ }

void __attribute__((disable_tail_calls, noinline)) recurse();

void __attribute__((noinline)) bounce2() {
  x++;
  recurse(); /* tail */
}

void __attribute__((noinline)) bounce1() {
  x++;
  bounce2(); /* tail */
}

void __attribute__((disable_tail_calls, noinline)) recurse() {
  if (--depth == 0)
    sink(); /* regular */
  else
    bounce1(); /* regular */
}

void __attribute__((noinline)) leaf() { x++; /* break in leaf */ }

void __attribute__((noinline)) mid() {
  x++;
  leaf(); /* tail */
}

void __attribute__((disable_tail_calls, noinline)) top() {
  mid(); /* regular */
  mid(); /* regular */
}

int __attribute__((disable_tail_calls)) main() {
  top();     /* regular */
  recurse(); /* regular */
  return 0;
}
