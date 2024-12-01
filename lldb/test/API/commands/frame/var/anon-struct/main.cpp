int main(int argc, char** argv) {
  struct A {
    struct {
      int x = 1;
    };
    int y = 2;
  } a;

  struct B {
    // Anonymous struct inherits another struct.
    struct : public A {
      int z = 3;
    };
    int w = 4;
    A a;
  } b;

  return 0; // Set a breakpoint here
}
