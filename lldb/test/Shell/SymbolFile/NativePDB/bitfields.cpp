// clang-format off
// REQUIRES: lld, x86

// Test various interesting cases for AST reconstruction.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/bitfields.lldbinit 2>&1 | FileCheck %s

// Test trivial versions of each tag type.
struct Struct {
  int A : 5 = 6;
  int B : 7 = 8;
  unsigned C : 3 = 2;
  unsigned D : 15 = 12345;
  char E : 1 = 0;
  char F : 2 = 1;
  char G : 3 = 2;
  // H should be at offset 0 of a new byte.
  char H : 3 = 3;
};

constexpr Struct TheStruct;


int main(int argc, char **argv) {
  return TheStruct.A;
}

// CHECK: (lldb) target variable -T TheStruct
// CHECK: (const Struct) TheStruct = {
// CHECK:   (int:5) A = 6
// CHECK:   (int:7) B = 8
// CHECK:   (unsigned int:3) C = 2
// CHECK:   (unsigned int:15) D = 12345
// CHECK:   (char:1) E = '\0'
// CHECK:   (char:2) F = '\x01'
// CHECK:   (char:3) G = '\x02'
// CHECK:   (char:3) H = '\x03'
// CHECK: }
//
// CHECK: target modules dump ast
// CHECK: Dumping clang ast for 1 modules.
// CHECK: TranslationUnitDecl {{.*}}
// CHECK: |-CXXRecordDecl {{.*}} struct Struct definition
// CHECK: | |-FieldDecl {{.*}} A 'int'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 5
// CHECK: | |-FieldDecl {{.*}} B 'int'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 7
// CHECK: | |-FieldDecl {{.*}} C 'unsigned int'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 3
// CHECK: | |-FieldDecl {{.*}} D 'unsigned int'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 15
// CHECK: | |-FieldDecl {{.*}} E 'char'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | |-FieldDecl {{.*}} F 'char'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 2
// CHECK: | |-FieldDecl {{.*}} G 'char'
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 3
// CHECK: | `-FieldDecl {{.*}} H 'char'
// CHECK: |   `-IntegerLiteral {{.*}} 'int' 3
