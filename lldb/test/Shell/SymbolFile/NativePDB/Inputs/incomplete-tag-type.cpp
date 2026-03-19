struct A {
  int x;
  A();
};
A::A() : x(47) {}

struct C {
  C();
};
C::C() = default;

struct E {
  E();
};
E::E() = default;

struct I {
  I();
};
I::I() = default;
