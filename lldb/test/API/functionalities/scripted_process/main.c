int bar(int i) {
  int j = i * i;
  return j; // break here
}

int foo(int i) { return bar(i); }

int main() { return foo(42); }
