void bar() {
  int val = 32; // breakpoint here
}

void at_line_zero() {}

int foo();

int main(int argc, char const *argv[]) {
  foo();
  return 0;
}

int foo() {
  bar(); // foo call bar
#line 0 "test.cpp"
  at_line_zero();
  return 0;
}
