int main(int argc, char **argv) {
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


  struct EmptyBase {
  };

  struct : public A {
    struct {
      int z = 13;
    };
  } unnamed_derived;

  struct DerivedB : public B {
    struct {
      // `w` in anonymous struct shadows `w` from `B`.
      int w = 14;
      int k = 15;
    };
  } derb;

  struct MultiBase : public EmptyBase, public A {
    struct {
      int m = 16;
      int y = 30;
    };
  } multi1;

  struct MB2 : public B, EmptyBase, public A {
    int i = 42;
    struct {
      int w = 23;
      int n = 7;
    };
  } multi2;

  return 0; // Set breakpoint here
}
