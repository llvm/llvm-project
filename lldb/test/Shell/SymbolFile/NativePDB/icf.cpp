// clang-format off
// REQUIRES: lld, x86

// Test lldb finds the correct parent context decl for functions and class methods when icf happens.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -GS- -fno-addrsig -c /Fo%t.obj -- %s
// RUN: lld-link -opt:icf -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symbols --dump-ast %t.exe | FileCheck %s

struct A {
  int f1(int x) {
    return x * 2;
  }
};
struct B {
  int f2(int x) {
    return x * 2;
  }
};
namespace N1 {
int f3(void*, int x) {
  return  x * 2;
}
} // namespace N1

namespace N2 {
namespace N3 {
  int f4(void*, int x) {
    return  x * 2;
  }
} // namespace N3
} // namespace N2

namespace N4 {
  // Same base name as N1::f3 but different namespaces.
  int f3(void*, int x) {
    return x * 2;
  }
  // Same base name as B::f2 but this is in namespace.
  int f2(void*, int x) {
    return x * 2;
  }
} // namespace N4


int main() {
  A a;
  B b;
  return a.f1(1) + b.f2(1) + N1::f3(nullptr, 1) + N2::N3::f4(nullptr, 1) +
         N4::f3(nullptr, 1);
}


// CHECK:      namespace N1 {
// CHECK-NEXT:     int f3(void *, int x);
// CHECK-NEXT: }
// CHECK-NEXT: namespace N2 {
// CHECK-NEXT:     namespace N3 {
// CHECK-NEXT:         int f4(void *, int x);
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK-NEXT: namespace N4 {
// CHECK-NEXT:     int f3(void *, int x);
// CHECK-NEXT:     int f2(void *, int x);
// CHECK-NEXT: }
// CHECK-NEXT: int main();
// CHECK-NEXT: struct A {
// CHECK-NEXT:     int f1(int);
// CHECK-NEXT: };
// CHECK-NEXT: struct B {
// CHECK-NEXT:     int f2(int);
// CHECK-NEXT: };
