struct foo {
  int x;
  bool y;
};

void dwo_error_foo() {
  foo f;
  f.x = 1;
  f.y = true;
}
