struct Foo {
  // Triggers that we emit an artificial constructor for Foo.
  virtual ~Foo() = default;
};

int main() {
  Foo f;
  // Try to construct foo in our expression.
  return 0; //%self.expect("expr Foo()", substrs=["(Foo) $0 = {}"])
}
