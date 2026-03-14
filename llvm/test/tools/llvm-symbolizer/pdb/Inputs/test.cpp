// To generate the corresponding EXE/PDB (add -gcolumn-info for line columns):
// clang -cc1 -triple x86_64-pc-windows-msvc -gcodeview -debug-info-kind=constructor -emit-obj test.cpp
// lld-link test.obj -debug -entry:main

namespace NS {
struct Foo {
  void bar() {}
};
}

void foo() {
}

static void private_symbol() {
}

int main() {
  foo();
  
  NS::Foo f;
  f.bar();
  private_symbol();
}

extern "C" {
void __cdecl foo_cdecl() {}
void __stdcall foo_stdcall() {}
void __fastcall foo_fastcall() {}
void __vectorcall foo_vectorcall() {}
}
