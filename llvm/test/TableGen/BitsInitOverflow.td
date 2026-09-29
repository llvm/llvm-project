// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// Check that a large integer is not truncated to a small bit sequence.
//
// CHECK: error: Initializer of 'X' in 'anonymous_0' could not be fully resolved:

def {
  bits<2> X = 5;  // bitfield is too small, reject
}
