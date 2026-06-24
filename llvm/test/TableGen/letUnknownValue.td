// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s --strict-whitespace

def {
  /// Let can only override something that already exists.
  let abc = [];
// CHECK: error: Value 'abc' unknown!
// CHECK-NEXT:{{^}}  let abc = [];
// CHECK-NEXT:{{^}}      ^
}