int foo(int x, int y) {
  return x - y + 5; // Set breakpoint here.
}

int main() {
  foo(1, 4);
  foo(5, 1);
  foo(5, 5);
  foo(3, -1);
  foo(6, 6);
  foo(7, 7);
  foo(1, 3);
  foo(3, 1);

  return 0;
}
