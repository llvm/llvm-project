struct A {
  int member_var = 1;
  static int static_member_var;
  static const int static_const_member_var;
  static constexpr int static_constexpr_member_var = 4;
  int member_func() { return 5; }
  static int static_func() { return 6; }

  static int context_static_func() {
    int i = static_member_var;
    i += static_func();
    return i; // break in static member function
  }

  int context_member_func() {
    int i = member_var;
    i += member_func();
    return i; // break in member function
  }
};

int A::static_member_var = 2;
const int A::static_const_member_var = 3;
constexpr int A::static_constexpr_member_var;

int main() {
  int i = A::context_static_func();
  A a;
  a.context_member_func();
  return i;
}
