struct F {
  int leaf = 42;
};

struct E {
  F f;
};
struct D {
  E e;
};

struct C {
  D d;
};

struct B {
  C c;
};

struct A {
  B b;
};

int main() {
  A a;
  (void)a;
  return 0; // break here
}
