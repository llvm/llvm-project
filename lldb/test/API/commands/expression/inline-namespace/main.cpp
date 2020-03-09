namespace A {
  inline namespace B {
    int f() { return 3; }
  };
}

int main(int argc, char **argv) {
  // Set break point at this line.
  return A::f();
}
