int g1, g2;

__attribute__((noinline)) void step(void) { g1 += g2; }

void loclist(void) {
  int *p = &g1;
  step();
  p = &g2;
  step();
}
