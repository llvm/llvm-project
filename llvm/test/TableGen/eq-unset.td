// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// CHECK: error: expected value to be a typed value, got '?'

def Z1 {
  // This one caused an assertion because the value was an UnsetInit
  // and !eq() can only accept TypedInit's.
  bit D = !if(!eq(?, 0), 1, 0);
}
