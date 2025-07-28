int bar(int x, int y) {
  int sum = x + y;
  int mul = x * y;
  return sum + mul;
}

int foo(int a, int b) {
  int result = bar(a, b);
  return result;
}
