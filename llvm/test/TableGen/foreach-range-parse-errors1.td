// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s

// CHECK: [[FILE]]:[[@LINE+2]]:20: error: invalid range, cannot be negative
// CHECK: [[FILE]]:[[@LINE+1]]:20: error: expected declaration in for
foreach Index = -1 - 0 in {

}

