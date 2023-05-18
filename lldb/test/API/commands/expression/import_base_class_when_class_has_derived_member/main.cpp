struct B {
  int dump() const;
};

int B::dump() const { return 42; }

// Derived class D obtains dump() method from base class B
struct D : public B {
  // Introduce a TypedefNameDecl
  using Iterator = D *;
};

struct C {
  // This will cause use to invoke VisitTypedefNameDecl(...) when Importing
  // the DeclContext of C.
  // We will invoke ImportContext(...) which should force the From Decl to
  // be defined if it not already defined. We will then Import the definition
  // to the To Decl.
  // This will force processing of the base class of D which allows us to see
  // base class methods such as dump().
  D::Iterator iter;

  bool f(D *DD) {
    return true; //%self.expect_expr("DD->dump()", result_type="int", result_value="42")
  }
};

int main() {
  C c;
  D d;

  c.f(&d);

  return 0;
}
