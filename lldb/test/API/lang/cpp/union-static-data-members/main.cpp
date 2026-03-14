union Foo {
  int val = 42;
  static const int sVal1 = -42;
  static Foo sVal2;
};

Foo Foo::sVal2{};

namespace {
union Bar {
  int val = 137;
  static const int sVal1 = -137;
  static Bar sVal2;
};

Bar Bar::sVal2{};
} // namespace

int main() {
  Foo foo;
  Bar bar;
  auto sum = Bar::sVal1 + Foo::sVal1 + Foo::sVal2.val + Bar::sVal2.val;

  return 0;
}
