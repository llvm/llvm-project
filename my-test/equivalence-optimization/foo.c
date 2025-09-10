int foo(int x, int y) {
  if ((x & y) == y)
    return x - y;
  return 0;
}
