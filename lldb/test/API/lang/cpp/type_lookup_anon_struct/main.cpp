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

   struct : public A {
     struct {
       int z = 13;
     };
   } unnamed_derived;

   struct DerivedB : public B {
     struct {
       // `w` in anonymous struct overrides `w` from `B`.
       int w = 14;
       int k = 15;
     };
   } derb;

   return 0; // Set breakpoint here
}
