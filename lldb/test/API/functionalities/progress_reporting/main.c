int bar(int b) { return b * b; }

int foo(int f) {
  int b = bar(f); // break here
  return b;
}

int main() {
  int f = foo(42);
  return f;
}
