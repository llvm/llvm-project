typedef int Foo;

int main() {
  int lval = 1;
  Foo *x = &lval;
  Foo &y = lval;
  Foo &&z = 1;

  // Test lldb doesn't dereference pointer more than once.
  Foo **xp = &x;
  return 0; // Set breakpoint here
}
