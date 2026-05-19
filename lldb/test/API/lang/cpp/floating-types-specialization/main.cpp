template <typename T> struct Foo;

template <> struct Foo<__bf16> {};

template <> struct Foo<_Float16> : Foo<__bf16> {};

int main() {
  Foo<__bf16> f0;
  Foo<_Float16> f1;
  return 0; // break here
}
