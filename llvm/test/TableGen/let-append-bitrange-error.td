// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

// Test that 'let append' with bit range produces an error.

class Base {
  bits<8> flags = 0;
}

// CHECK: error: Cannot use append/prepend with bit range
def Bad : Base {
  let append flags{0-3} = 0;
}
