// This was used to generate `dynamic-types.yaml` and `dynamic-types-exe.yaml`.

// clang-format off

// clang-cl /Z7 /GR- /GS- .\dynamic-types.cpp -fuse-ld=lld-link -Og /link /nodefaultlib /entry:main
// obj2yaml .\dynamic-types.exe -o dynamic-types-exe.yaml
// llvm-pdbutil pdb2yaml --tpi-stream dynamic-types.pdb > dynamic-types.yaml
// rm dynamic-types.exe
// rm dynamic-types.pdb

// clang-format on

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
