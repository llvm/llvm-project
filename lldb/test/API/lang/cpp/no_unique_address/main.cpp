struct Empty {};

namespace basic {
struct Foo {
  [[no_unique_address]] Empty a;
};
} // namespace basic

namespace bases {
struct A {
  long c, d;
};

struct B {
  [[no_unique_address]] Empty x;
};

struct C {
  [[no_unique_address]] Empty x;
};

struct Foo : B, A, C {};
struct Bar : B, C, A {};
} // namespace bases

int main() {
  basic::Foo b1;
  bases::Foo b2;
  bases::Bar b3;
  b2.c = 1;
  b2.d = 2;
  b3.c = 5;
  b3.d = 6;
  return 0;
}
