// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

// This test checks that using def instead of defm gives a meaningful error
multiclass M2 {
  def X;
}

// CHECK: error: Couldn't find class 'M2'. Use 'defm' if you meant to use multiclass 'M2'
def rec1 : M2;
