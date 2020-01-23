namespace A {
  inline namespace B {
    int f() { return 3; }
  };
}

int main() { int argc = 0; char **argv = (char **)0; 
  // Set break point at this line.
  return A::f();
}
