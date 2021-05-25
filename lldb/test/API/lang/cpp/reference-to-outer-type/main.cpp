struct Outer {
  typedef int HookToOuter;
  // When importing this type, we have to import all of it before adding it
  // via the FieldDecl to 'Outer'. If we don't do this, then Clang will
  // while adding 'NestedClassMember' ask for the full type of 'NestedClass'
  // which then will start pulling in the 'RefToOuter' member. That member
  // will import the typedef above and add it to 'Outer'. And adding a
  // Decl to a DeclContext that is currently already in the process of adding
  // another Decl will cause an inconsistent lookup.
  struct NestedClass {
    HookToOuter RefToOuter;
    int SomeMember;
  } NestedClassMember;
};

// We query the members of base classes of a type by doing a lookup via Clang.
// As this tests is trying to find a borked lookup, we need a base class here
// to make our 'GetChildMemberWithName' use the Clang lookup.
struct In : Outer {};

In test_var;

int main() { return test_var.NestedClassMember.SomeMember; }
