template <typename T> struct S {
  typedef T V;

  V value;
};

typedef S<float> GlobalTypedef;

namespace ns {
typedef S<float> NamespaceTypedef;
}

struct ST {
  typedef S<float> StructTypedef;
};

// Struct type that is not supposed to be a local variable in the test
// expression evaluation scope. Tests that typedef lookup can actually look
// inside class/struct scopes.
struct NonLocalVarStruct {
  typedef int OtherStructTypedef;
};

int otherFunc() {
  NonLocalVarStruct::OtherStructTypedef i = 3;
  return i;
}

int main(int argc, char const *argv[]) {
  GlobalTypedef s{.5};
  ns::NamespaceTypedef in_ns;
  ST::StructTypedef in_struct;
  return otherFunc(); // Set a breakpoint here
}
