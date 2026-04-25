struct D {
  int value = 42;
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
  return 0; // break here
}
