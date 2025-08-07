int foo(int f) {
  int b = f * f; // Break here
  return b;
}

int main() {
  int f = foo(42);
  return f;
}
