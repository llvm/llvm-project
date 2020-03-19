struct A {
  int a;
};

static A a;

struct B {
  B(int b = a.a) {}
};


void foo() {
  B();
}
