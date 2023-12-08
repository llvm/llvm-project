struct TopLevelStruct {
  int i;
};

// Contains a templated nested class with a typedef.
struct StructWithNested {
  template <typename T>
  struct Nested {
    // Typedef in a class. Intended to be referenced directly so that it can
    // trigger the loading of the surrounding classes.
    typedef TopLevelStruct OtherTypedef;
  };
};

// Contains a typedef.
struct StructWithMember {
  // This member pulls in the typedef (and classes) above.
  StructWithNested::Nested<int>::OtherTypedef m;
  // Typedef in a class. Intended to be referenced directly so that it can
  // trigger the loading of the surrounding class.
  typedef int MemberTypedef;
};

// This is printed and will pull in the typedef in StructWithmember.
StructWithMember::MemberTypedef pull_in_classes;


StructWithMember struct_to_print;


int main() {}
