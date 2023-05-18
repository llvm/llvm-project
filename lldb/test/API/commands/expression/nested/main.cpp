namespace a {
namespace b {
namespace c {
static int d = 12;
enum Color { Red, Green, Blue };
} // namespace c
} // namespace b
} // namespace a

struct A {
  int _a = 'a';
  struct B {
    short _b = 'b';
    struct C {
      char _c = 'c';
      enum EnumType : int { Eleven = 11 };
      static EnumType enum_static;
    };
  };
};

A::B::C::EnumType A::B::C::enum_static = A::B::C::Eleven;

int foo() {
  a::b::c::Color color = a::b::c::Blue;
  return A::B::C::enum_static == a::b::c::d && ((int)color == 0);
}

int main() {
  return foo(); // Stop here to evaluate expressions
}
