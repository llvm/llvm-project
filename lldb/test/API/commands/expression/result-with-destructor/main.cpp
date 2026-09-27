struct Foo {
  Foo() : x(42) {}
  ~Foo() {}
  int x;
};

struct Widget {
  Widget() : x(47) {}
  ~Widget() {}
  int x;
};

Foo make_foo() { return Foo(); }

Widget make_widget() { return Widget(); }

void bar() {
  static Foo f;
  // break here
}

int main() {
  bar();
  return 0;
}
