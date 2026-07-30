struct C {
  int func() { return 111; }
  int func() const { return 222; }

  int const_func() const { return 333; }
  int nonconst_func() { return 444; }
};

int main() {
  C c;
  const C const_c;
  c.func();
  c.nonconst_func();
  const_c.func();
  c.const_func();
  return 0; // break here
}
