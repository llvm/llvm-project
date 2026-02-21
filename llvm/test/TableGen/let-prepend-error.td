// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

// Test that 'let prepend' on unsupported types produces an error.

class Base {
  int count = 0;
}

// CHECK: error: Cannot prepend to field 'count' of type 'int' (expected list, string, code, or dag)
def Bad : Base {
  let prepend count = 1;
}
