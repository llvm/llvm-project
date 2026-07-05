namespace A {
  inline namespace B {
    int f() { return 3; }
    int global_var = 0;

    namespace C {
    int global_var = 1;
    }

    inline namespace D {
    int nested_var = 2;
    }
  };

  namespace E {
  inline namespace F {
  int other_var = 3;
  }
  } // namespace E

  int global_var = 4;
}

int main(int argc, char **argv) {
  // Set break point at this line.
  return A::f() + A::B::global_var + A::C::global_var + A::E::F::other_var +
         A::B::D::nested_var;
}
