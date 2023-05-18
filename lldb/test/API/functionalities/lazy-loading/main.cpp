//----------------------------------------------------------------------------//
// Struct loading declarations.

struct StructFirstMember { int i; };
struct StructBehindPointer { int i; };
struct StructBehindRef { int i; };
struct StructMember { int i; };

StructBehindRef struct_instance;

struct SomeStruct {
  StructFirstMember *first;
  StructBehindPointer *ptr;
  StructMember member;
  StructBehindRef &ref = struct_instance;
};

struct OtherStruct {
  int member_int;
};

//----------------------------------------------------------------------------//
// Class loading declarations.

struct ClassMember { int i; };
struct StaticClassMember { int i; };
struct UnusedClassMember { int i; };
struct UnusedClassMemberPtr { int i; };

namespace NS {
class ClassInNamespace {
  int i;
};
class ClassWeEnter {
public:
  int dummy; // Prevent bug where LLDB always completes first member.
  ClassMember member;
  static StaticClassMember static_member;
  UnusedClassMember unused_member;
  UnusedClassMemberPtr *unused_member_ptr;
  int enteredFunction() {
    return member.i; // Location: class function
  }
};
StaticClassMember ClassWeEnter::static_member;
};

//----------------------------------------------------------------------------//
// Function we can stop in.

int functionWithOtherStruct() {
  OtherStruct other_struct_var;
  other_struct_var.member_int++; // Location: other struct function
  return other_struct_var.member_int;
}

int functionWithMultipleLocals() {
  SomeStruct struct_var;
  OtherStruct other_struct_var;
  NS::ClassInNamespace namespace_class;
  other_struct_var.member_int++; // Location: multiple locals function
  return other_struct_var.member_int;
}

int main(int argc, char **argv) {
  NS::ClassWeEnter c;
  c.enteredFunction();

  functionWithOtherStruct();
  functionWithMultipleLocals();
  return 0;
}
