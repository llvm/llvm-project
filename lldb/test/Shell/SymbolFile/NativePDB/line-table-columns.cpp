// clang-format off
// REQUIRES: lld, x86

// Test that the columns of the line table are read.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -gcolumn-info -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -o "image dump line-table line-table-columns.cpp" -b \
// RUN:     2>&1 | FileCheck %s

int main() {
  int x = 1; int y = x + 2;
  return y;
}

// CHECK:      line-table-columns.cpp:10
// CHECK-NEXT: line-table-columns.cpp:11:7
// CHECK-NEXT: line-table-columns.cpp:11:22
// CHECK-NEXT: line-table-columns.cpp:11:24
// CHECK-NEXT: line-table-columns.cpp:11:18
// CHECK-NEXT: line-table-columns.cpp:12:10
// CHECK-NEXT: line-table-columns.cpp:12:3
