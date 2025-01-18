int bar(int b) { return b * b; }

int foo(int f) {
  int b = bar(f); // Break here
  return b;
}

int main() {
  int f = foo(42);
  return f;
}
