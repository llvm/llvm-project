// This was used to generate `DynamicTypes.pdb`.

// clang-cl /Z7 /GR- /GS- DynamicTypes.cpp -c
// lld-link /NODEFAULTLIB /entry:main /DEBUG DynamicTypes.obj
// rm DynamicTypes.obj

struct Base {
  virtual ~Base() = default;
};

struct UsingBase : public Base {};

struct VBase {};

struct UsingVBase : public virtual VBase {};

struct UsingUsingVBase : public UsingVBase {};

struct NotDynamic : public VBase {};

void operator delete(void *, unsigned __int64 i) throw() {}

int main() {
  UsingBase ub;
  UsingVBase uvb;
  UsingUsingVBase uuvb;
  NotDynamic nd;
  return 0;
}
