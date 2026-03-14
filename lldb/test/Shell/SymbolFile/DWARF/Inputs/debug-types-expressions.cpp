struct A {
  int i = 47;
  int f() { return i; }
  virtual ~A() = default;
};

struct B: public A {
  int j = 42;
};

namespace ns {
struct A {
  int i = 147;
  ::A getA();
  A();
};
A::A() = default;

::A A::getA() {
  ::A a;
  a.i = i - 1;
  return a;
}

} // namespace ns

int foo(A *a) {
  return a->f();
}

int main() {
  return foo(new B);
}
