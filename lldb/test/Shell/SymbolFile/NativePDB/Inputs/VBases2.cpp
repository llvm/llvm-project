struct A {
  char a = 1;
};

struct B {
  int b = 2;
};

struct C : virtual A, virtual B {
  short c = 3;
};

int main() {
  C c{};
  return 0;
}
