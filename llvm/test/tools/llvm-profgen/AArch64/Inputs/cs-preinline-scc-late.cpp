// G -> R -> F -> G cycle; all out of line so every name has a Pass1 size.
__attribute__((noinline)) unsigned long G(unsigned long n);
__attribute__((noinline)) unsigned long F(unsigned long n) {
  unsigned long x = n * 3 + 1;
  if (n > 3)
    x += G(n - 3);
  return x;
}
__attribute__((noinline)) unsigned long R(unsigned long n) {
  unsigned long x = n ^ (n >> 3);
  x += F(n);
  return x;
}
__attribute__((noinline)) unsigned long G(unsigned long n) {
  unsigned long x = n + 5;
  x += R(n);
  return x;
}
int main(int argc, char **) { return (int)G(argc * 100); }
