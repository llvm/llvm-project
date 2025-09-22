int main(int argc, char** argv)
{
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

  // Anonymous classes and unions.
  struct C {
    union {
      int x = 5;
    };
    class {
     public:
      int y = 6;
    };
  } c;

  // Multiple levels of anonymous structs.
  struct D {
    struct {
      struct {
        int x = 7;
        struct {
          int y = 8;
        };
      };
      int z = 9;
      struct {
        int w = 10;
      };
    };
  } d;

  struct E {
    struct IsNotAnon {
      int x = 11;
    };
  } e;

  struct F {
    struct {
      int x = 12;
    } named_field;
  } f;

  // Inherited unnamed struct without an enclosing parent class.
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

  return 0; // Set a breakpoint here
}
