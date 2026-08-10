// This class just serves as an indirection between LLDB and Clang. LLDB might
// be tempted to check the member type of DependsOnParam2 for whether it's
// in some 'currently-loading' state before trying to produce the record layout.
// By inheriting from ToLayout this will make LLDB just check if
// DependsOnParam1 is currently being loaded (which it's not) but it will
template <typename ToLayoutParam> struct DependsOnParam1 : ToLayoutParam {};
// This class forces the memory layout of it's type parameter to be created.
template <typename ToLayoutParam> struct DependsOnParam2 {
  DependsOnParam1<ToLayoutParam> m;
};

// This is the class that LLDB has to generate the record layout for.
struct ToLayout {
  // A static member variable which memory layout depends on the surrounding
  // class. This comes first so that if we accidentially generate the layout
  // for static member types we end up recursively going back to 'ToLayout'
  // before 'some_member' has been loaded.
  static DependsOnParam2<ToLayout> a_static_member;
  // Some dummy member variable. This is only there so that Clang can detect
  // that the record layout is inconsistent (i.e., the number of fields in the
  // layout doesn't fit to the fields in the declaration).
  int some_member;
};
DependsOnParam2<ToLayout> ToLayout::a_static_member;

ToLayout test_var;

int main() { return test_var.some_member; }
