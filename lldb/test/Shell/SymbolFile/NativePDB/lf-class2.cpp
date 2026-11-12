// Test that lldb can handle the LF_CLASS2/STRUCTURE2/UNION2 records.

// clang-format off
// REQUIRES: lld, x86

// Test various interesting cases for AST reconstruction.
// RUN: %clang_cl --target=x86_64-windows-msvc -mllvm=-use-codeview-tagrecord2 \
// RUN:   -GS- -GR- -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: llvm-pdbutil dump --types %t.pdb | FileCheck --check-prefix=PDB %s
// RUN: lldb-test symbols --dump-ast %t.exe | FileCheck --check-prefix=STRUCT %s
// RUN: lldb-test symbols --dump-ast %t.exe | FileCheck --check-prefix=CLASS %s

struct AStruct {
  char c1;
  virtual ~AStruct() = default;
  virtual int something() { return 1; }
};

class AClass : public AStruct {
public:
  int i;
  union Inner {
    short s;
    unsigned short u;
  };
  Inner inner;
};

void *operator new(unsigned long long n) { return nullptr; }
void operator delete(void *p, unsigned long long i) {}

int main() {
  AClass a;
  return a.inner.s;
}

// Make sure that we actually check the *2 records.
// PDB-DAG: LF_CLASS2
// PDB-DAG: LF_STRUCTURE2
// PDB-DAG: LF_UNION2

// CLASS:      class AClass : public AStruct {
// CLASS-NEXT:     AClass();
// CLASS-NEXT:     ~AClass();
// CLASS-NEXT:     union Inner {
// CLASS-NEXT:         short s;
// CLASS-NEXT:         unsigned short u;
// CLASS-NEXT:     };
// CLASS-NEXT:     int i;
// CLASS-NEXT:     AClass::Inner inner;
// CLASS-NEXT: };

// STRUCT:      struct AStruct {
// STRUCT-NEXT:     AStruct();
// STRUCT-NEXT:     virtual int something();
// STRUCT-NEXT:     virtual ~AStruct();
// STRUCT-NEXT:     char c1;
// STRUCT-NEXT: };
