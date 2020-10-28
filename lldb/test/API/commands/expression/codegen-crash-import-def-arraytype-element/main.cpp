// This is a reproducer for a crash during codegen. The base issue is when we
// Import the DeclContext we force FieldDecl that are RecordType to be defined
// since we need these to be defined in order to layout the class.
// This case involves an array member whose ElementType are records. In this
// case we need to check the ElementType of an ArrayType and if it is a record
// we need to import the definition.
struct A {
  int x;
};

struct B {
  // When we import the all the FieldDecl we need to check if we have an
  // ArrayType and then check if the ElementType is a RecordDecl and if so
  // import the defintion. Otherwise during codegen we will attempt to layout A
  // but won't be able to.
  A s1[2];
  A s2[2][2][3];
  char o;
};

class FB {
public:
  union {
    struct {
      unsigned char *_s;
    } t;
    char *tt[1];
  } U;

  FB(B *p) : __private(p) {}

  // We import A but we don't import the definition.
  void f(A **bounds) {}

  void init();

private:
  B *__private;
};

void FB::init() {
  return; // break here
}

int main() {
  B b;
  FB fb(&b);

  b.o = 'A';

  fb.init();
}
