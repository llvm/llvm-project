extern int fn(int a, int b) {
  if (a < b) {
    int sum = a + b;
    return sum; // breakpoint 2
  }

  return a * b;
}
