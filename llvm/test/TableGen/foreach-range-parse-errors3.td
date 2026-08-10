// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s

// CHECK: [[FILE]]:[[@LINE+2]]:21: error: expected integer or bitrange
// CHECK: [[FILE]]:[[@LINE+1]]:21: error: expected declaration in for
foreach Index = "0" - 1 in {

}

