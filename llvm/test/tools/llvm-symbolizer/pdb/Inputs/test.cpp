// To generate the corresponding EXE/PDB, run:
// cl /Zi test.cpp
// To generate the PDB with column numbers, run:
// clang-cl /Zi -gcolumn-info test.cpp

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
