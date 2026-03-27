// Bug: #131279
// RUN: mlir-opt --test-scf-pipelining %s | FileCheck %s
// CHECK: fold_div_index_neg_rhs
// CHECK-NEXT: %c0 = arith.constant 0 : index
// CHECK-NEXT: %0 = shape.div %c0, %c0 : index, index -> index
// CHECK-NEXT: return %0 : index
module {
  func.func @fold_div_index_neg_rhs() -> index {
    %c0 = arith.constant 0 : index
    %0 = shape.div %c0, %c0 : index, index -> index
    return %0 : index
  }
}
