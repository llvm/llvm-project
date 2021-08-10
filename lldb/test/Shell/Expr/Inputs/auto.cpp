struct Foo {
  auto i();
};

auto Foo::i() {
  return 1;
}

int main() {
  Foo f;
  f.i();
}
