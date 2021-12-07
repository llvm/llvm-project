// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GR- -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-methods.lldbinit 2>&1 | FileCheck %s --check-prefix=AST

// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symbols --dump-ast %t.exe | FileCheck %s --check-prefix=SYMBOL

struct Struct {
  void simple_method() {}

  virtual void virtual_method() {}

  static void static_method() {}

  int overloaded_method() {}
  int overloaded_method(char c) {}
  int overloaded_method(char c, int i, ...) {}
};

Struct s;

int main(int argc, char **argv) {
  s.simple_method();
  s.static_method();
  s.virtual_method();
  s.overloaded_method();
  s.overloaded_method('a');
  s.overloaded_method('a', 1);
  return 0;
}

// AST: TranslationUnitDecl
// AST: |-CXXRecordDecl {{.*}} struct Struct definition
// AST: | |-CXXMethodDecl {{.*}} simple_method 'void (){{.*}}'
// AST: | |-CXXMethodDecl {{.*}} virtual_method 'void (){{.*}}' virtual
// AST: | |-CXXMethodDecl {{.*}} static_method 'void ()' static
// AST: | |-CXXMethodDecl {{.*}} overloaded_method 'int (){{.*}}'
// AST: | |-CXXMethodDecl {{.*}} overloaded_method 'int (char){{.*}}'
// AST: | | `-ParmVarDecl {{.*}} 'char'
// AST: | `-CXXMethodDecl {{.*}} overloaded_method 'int (char, int, ...)'
// AST: |   |-ParmVarDecl {{.*}} 'char'
// AST: |   `-ParmVarDecl {{.*}} 'int'

// SYMBOL:      int main(int argc, char **argv);
// SYMBOL-NEXT: struct Struct {
// SYMBOL-NEXT:     void simple_method();
// SYMBOL-NEXT:     static void static_method();
// SYMBOL-NEXT:     virtual void virtual_method();
// SYMBOL-NEXT:     int overloaded_method();
// SYMBOL-NEXT:     int overloaded_method(char);
// SYMBOL-NEXT:     int overloaded_method(char, int, ...);
// SYMBOL-NEXT: };
