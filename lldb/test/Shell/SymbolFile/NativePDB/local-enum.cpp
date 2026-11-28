// clang-format off
// REQUIRES: lld, x86

// Test that an enum declared in a function is not made a scoped enum. Clang
// sets the Scoped option on function-local types, CodeView has nothing for
// C++11 scoped enums.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -o "type lookup Local" -b 2>&1 | FileCheck %s

int main() {
  enum Local { A = 1, B = 2 };
  Local l = B;
  return l;
}

// CHECK: (lldb) type lookup Local
// CHECK-NEXT: enum Local {
