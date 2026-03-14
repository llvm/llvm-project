// In top-level scope.
struct TopLevelStruct {
  int member;
};
TopLevelStruct top_level_struct;

// Nested in a class.
struct OuterStruct {
  struct InnerStruct {
    int member;
  };
};
OuterStruct::InnerStruct inner_struct;

// Behind typedef.
struct UnderlyingTypedefStruct {
  int member;
};
typedef UnderlyingTypedefStruct TypedefStruct;
TypedefStruct typedef_struct;

// In namespace.
namespace NS {
struct NamespaceStruct {
  int member;
};
} // namespace NS
NS::NamespaceStruct namespace_struct;

// In unnamed namespace.
namespace {
struct UnnamedNamespaceStruct {
  int member;
};
} // namespace
UnnamedNamespaceStruct unnamed_namespace_struct;

// In linkage spec.
extern "C" {
struct ExternCStruct {
  int member;
};
}
ExternCStruct extern_c_struct;

int main() {
  struct DeclInFunc {
    int member;
  };

  DeclInFunc decl_in_func;
  return unnamed_namespace_struct.member; // break here
}
