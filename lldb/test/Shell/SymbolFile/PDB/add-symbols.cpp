// REQUIRES: lld, target-windows

// Test that `target symbols add <pdb>` works.
// RUN: %build --compiler=clang-cl --nodefaultlib --output=%t.exe %s
// RUN: mv %t.pdb %t-renamed.pdb

// RUN: env LLDB_USE_NATIVE_PDB_READER=0 %lldb \
// RUN:   -o "b main" \
// RUN:   -o "target symbols add %t-renamed.pdb" \
// RUN:   -o r \
// RUN:   -o "target variable a" \
// RUN:   -o "target modules dump symtab" \
// RUN:   -b %t.exe | FileCheck %s

// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb \
// RUN:   -o "b main" \
// RUN:   -o "target symbols add %t-renamed.pdb" \
// RUN:   -o r \
// RUN:   -o "target variable a" \
// RUN:   -o "target modules dump symtab" \
// RUN:   -b %t.exe | FileCheck %s

// CHECK: target create
// CHECK: (lldb) b main
// CHECK-NEXT: Breakpoint 1: no locations (pending).
// CHECK: (lldb) target symbols add
// CHECK: 1 location added to breakpoint 1

// CHECK: * thread #1, stop reason = breakpoint 1.1
// CHECK: (lldb) target variable a
// CHECK-NEXT: (A) a = (x = 47)
// CHECK: (lldb) target modules dump symtab
// CHECK: [{{.*}} main

struct A {
  int x = 47;
};
A a;
int main() {}
