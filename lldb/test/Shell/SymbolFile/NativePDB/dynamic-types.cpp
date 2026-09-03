// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -GS- -GR- -std:c++20 -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb

// RUN: lldb-test symbols --dump-clang-ast --find=type --name=Base            %t.exe | FileCheck %s --check-prefix=DYNAMIC
// RUN: lldb-test symbols --dump-clang-ast --find=type --name=UsingBase       %t.exe | FileCheck %s --check-prefix=DYNAMIC
// RUN: lldb-test symbols --dump-clang-ast --find=type --name=UsingVBase      %t.exe | FileCheck %s --check-prefix=DYNAMIC
// RUN: lldb-test symbols --dump-clang-ast --find=type --name=UsingUsingVBase %t.exe | FileCheck %s --check-prefix=DYNAMIC

// RUN: lldb-test symbols --dump-clang-ast --find=type --name=VBase      %t.exe | FileCheck %s --check-prefix=NOT-DYNAMIC
// RUN: lldb-test symbols --dump-clang-ast --find=type --name=NotDynamic %t.exe | FileCheck %s --check-prefix=NOT-DYNAMIC

// DYNAMIC: Found 1 types:
// DYNAMIC: decl-metadata = uid={{.*}} is_dynamic_cxx=true

// NOT-DYNAMIC: Found 1 types:
// NOT-DYNAMIC: decl-metadata = uid={{.*}} is_dynamic_cxx=false

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
