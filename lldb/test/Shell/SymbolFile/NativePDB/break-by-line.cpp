// clang-format off
// REQUIRES: lld, x86

// Test that we can set simple breakpoints using PDB on any platform.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: %lldb -f %t.exe -s \
// RUN:     %p/Inputs/break-by-line.lldbinit | FileCheck %s

// This is a separate test from break-by-function.cpp because this test is
// sensitive to edits in the source file.

namespace NS {
  int NamespaceFn(int X) {
    return X + 42;
  }
}

int main(int argc, char **argv) {
  return NS::NamespaceFn(argc);
}


// CHECK:      (lldb) target create "{{.*}}break-by-line.cpp.tmp.exe"
// CHECK:      Current executable set to '{{.*}}break-by-line.cpp.tmp.exe'
// CHECK:      (lldb) break set -f break-by-line.cpp -l 15
// CHECK:      Breakpoint 1: where = break-by-line.cpp.tmp.exe`int NS::NamespaceFn(int) + {{[0-9]+}} at break-by-line.cpp:15
