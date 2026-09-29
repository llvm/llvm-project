// RUN: not llvm-tblgen %s 2>&1 | FileCheck -DFILE=%s %s

// CHECK: [[FILE]]:[[@LINE+2]]:23: error: expected integer or bitrange
// CHECK: [[FILE]]:[[@LINE+1]]:23: error: expected declaration in for
foreach Index = 0b110 - 0b111 in {

}

