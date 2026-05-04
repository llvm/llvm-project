extern int foo(int x) {
  int y = x + 42; // break other
  int z = y + 42;
  return z;
}
