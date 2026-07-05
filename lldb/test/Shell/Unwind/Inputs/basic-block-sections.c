int k = 1;

int bar() {
  return 0;
}

int foo() {
  return bar();
}

int main() {
  // Control flow to create basic block sections.
  if (k)
    foo();
  else
    bar();
  return 0;
}
