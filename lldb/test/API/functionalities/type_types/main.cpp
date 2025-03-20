namespace a {
struct Foo {};

unsigned foo() {
  typedef unsigned Foo;
  Foo foo = 12;
  return foo;
}
} // namespace a

int main() {
  a::Foo f = {};
  a::foo();
  return 0;
}
