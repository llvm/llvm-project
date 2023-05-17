// This is a reproducer for a crash in codegen. It happens when we have a
// RecordDecl used in an expression and one of the FieldDecl are not complete.
// This case happens when:
// - A RecordDecl (E) has a FieldDecl which is a reference member variable
// - The underlying type of the FieldDec is a TypedefDecl
// - The typedef refers to a ClassTemplateSpecialization (DWrapper)
// - The typedef is not present in the DeclContext of B
// - The typedef shows up as a return value of a member function of E (f())
template <typename T> struct DWrapper {};

struct D {};

namespace NS {
typedef DWrapper<D> DW;
}

struct B {
  NS::DW spd;
  int a = 0;
};

struct E {
  E(B &b) : b_ref(b) {}
  NS::DW f() { return {}; };
  void g() {
    return; //%self.expect("expression b_ref", substrs=['(B) $0 =', '(spd = NS::DW', 'a = 0)'])
  }

  B &b_ref;
};

int main() {
  B b;
  E e(b);

  e.g();

  return 0;
}
