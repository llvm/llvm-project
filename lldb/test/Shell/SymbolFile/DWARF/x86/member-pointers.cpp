// REQUIRES: lld

// Itanium ABI:
// RUN: %clang --target=x86_64-pc-linux -gdwarf -c -o %t_linux.o %s
// RUN: %lldb -f %t_linux.o -b -o "target variable mp" | FileCheck %s
//
// CHECK: (char SI::*) mp = 0x0000000000000000

// Microsoft ABI:
// RUN: %clang_cl --target=x86_64-windows-msvc -c -gdwarf -o %t_win.obj -- %s
// RUN: lld-link /out:%t_win.exe %t_win.obj /nodefaultlib /entry:main /debug
// RUN: %lldb -f %t_win.exe -b -o "target variable mp" | FileCheck --check-prefix=CHECK-MSVC %s
//
// DWARF has no representation of MSInheritanceAttr, so we cannot determine the size
// of member-pointers yet. For the moment, make sure we don't crash on such variables.
// CHECK-MSVC: error: Unable to determine byte size.

struct SI {
  char si;
};

char SI::*mp = &SI::si;

int main() { return 0; }
