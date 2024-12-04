struct A {
  struct {
    int x = 1;
  };
} a;

struct B {
  // Anonymous struct inherits another struct.
  struct : public A {
    int z = 3;
  };
  A a;
} b;

int main(int argc, char **argv) {
  return 0;
}
