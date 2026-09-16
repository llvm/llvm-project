// clang-format off
// REQUIRES: lld, x86

// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GR- -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/ast-methods.lldbinit 2>&1 | FileCheck %s --check-prefix=AST

// RUN: lldb-test symbols --dump-ast %t.exe | FileCheck %s --check-prefix=SYMBOL

struct Struct {
  void simple_method() {}

  virtual void virtual_method() {}

  static void static_method() {}

  int overloaded_method() {}
  int overloaded_method(char c) {}
  int overloaded_method(char c, int i, ...) {}

  void const_method() const {}
  void volatile_method() volatile {}
  void const_volatile_method() const volatile {}

  virtual void virtual_const_method() const {}
};

Struct s;

int main(int argc, char **argv) {
  s.simple_method();
  s.static_method();
  s.virtual_method();
  s.overloaded_method();
  s.overloaded_method('a');
  s.overloaded_method('a', 1);
  s.const_method();
  s.volatile_method();
  s.const_volatile_method();
  s.virtual_const_method();
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
// AST: | |-CXXMethodDecl {{.*}} overloaded_method 'int (char, int, ...)'
// AST: | | |-ParmVarDecl {{.*}} 'char'
// AST: | | `-ParmVarDecl {{.*}} 'int'
// AST: | |-CXXMethodDecl {{.*}} const_method 'void () const'
// AST: | |-CXXMethodDecl {{.*}} volatile_method 'void () volatile'
// AST: | |-CXXMethodDecl {{.*}} const_volatile_method 'void () const volatile'
// AST: | `-CXXMethodDecl {{.*}} virtual_const_method 'void () const' virtual

// SYMBOL:      struct Struct {
// SYMBOL-NEXT:     void simple_method();
// SYMBOL-NEXT:     static void static_method();
// SYMBOL-NEXT:     virtual void virtual_method();
// SYMBOL-NEXT:     int overloaded_method();
// SYMBOL-NEXT:     int overloaded_method(char);
// SYMBOL-NEXT:     int overloaded_method(char, int, ...);
// SYMBOL-NEXT:     void const_method() const; 
// SYMBOL-NEXT:     void volatile_method() volatile; 
// SYMBOL-NEXT:     void const_volatile_method() const volatile; 
// SYMBOL-NEXT:     virtual void virtual_const_method() const; 
// SYMBOL-NEXT: };
// SYMBOL-NEXT: Struct s;
// SYMBOL-NEXT: int main(int argc, char **argv);
