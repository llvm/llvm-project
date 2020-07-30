// Make sure we correctly handle $ in variable names.

int main() {
  // Some variables that might conflict with our variables below.
  int __lldb_expr_result = 2;
  int $$foo = 1;
  int R0 = 2;

  // Some variables with dollar signs that should work (and shadow
  // any built-in LLDB variables).
  int $__lldb_expr_result = 11;
  int $foo = 12;
  int $R0 = 13;
  int $0 = 14;

  return 0; // break here
}
