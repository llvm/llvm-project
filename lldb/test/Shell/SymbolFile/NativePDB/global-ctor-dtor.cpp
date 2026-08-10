// clang-format off
// REQUIRES: lld, x86

// Global ctor and dtor should be globals decls.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GS- -fno-addrsig -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb -force
// RUN: lldb-test symbols --dump-ast %t.exe | FileCheck %s

struct A {
  ~A() {};
};
struct B {
  static A glob;
};

A B::glob = A();
int main() {
  return 0;
}

// CHECK:      struct A {
// CHECK-NEXT:     ~A();
// CHECK-NEXT: };
// CHECK-NEXT: A B::glob;
// CHECK-NEXT: static void B::`dynamic initializer for 'glob'();
// CHECK-NEXT: static void B::`dynamic atexit destructor for 'glob'();
// CHECK-NEXT: int main();
// CHECK-NEXT: static void _GLOBAL__sub_I_global_ctor_dtor.cpp();
// CHECK-NEXT: struct B {
// CHECK-NEXT:     static A glob;
// CHECK-NEXT: };
