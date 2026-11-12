// This class just serves as an indirection between LLDB and Clang. LLDB might
// be tempted to check the member type of DependsOnParam2 for whether it's
// in some 'currently-loading' state before trying to produce the record layout.
// By inheriting from ToLayout this will make LLDB just check if
// DependsOnParam1 is currently being loaded (which it's not) but it won't
// check if all the types DependsOnParam2 is depending on for its layout are
// currently parsed.
template <typename ToLayoutParam> struct DependsOnParam1 : ToLayoutParam {};
// This class forces the memory layout of it's type parameter to be created.
template <typename ToLayoutParam> struct DependsOnParam2 {
  DependsOnParam1<ToLayoutParam> m;
};

// This is the class that LLDB has to generate the record layout for.
struct ToLayout {
  // The class part of this pointer-to-member type has a memory layout that
  // depends on the surrounding class. If LLDB eagerly tries to layout the
  // class part of a pointer-to-member type while parsing, then layouting this
  // type should cause a test failure (as we aren't done parsing ToLayout
  // at this point).
  int DependsOnParam2<ToLayout>::* pointer_to_member_member;
  // Some dummy member variable. This is only there so that Clang can detect
  // that the record layout is inconsistent (i.e., the number of fields in the
  // layout doesn't fit to the fields in the declaration).
  int some_member;
};

// Emit the definition of DependsOnParam2<ToLayout>. It seems Clang won't
// emit the definition of a class template if it's only used in the class part
// of a pointer-to-member type.
DependsOnParam2<ToLayout> x;

ToLayout test_var;

int main() { return test_var.some_member; }
