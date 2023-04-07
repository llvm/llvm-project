// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @scf_for(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index, %[[c:.*]]: index
//       CHECK:   "test.some_use"(%[[a]], %[[b]])
func.func @scf_for(%a: index, %b: index, %c: index) {
  scf.for %iv = %a to %b step %c {
    %0 = "test.reify_bound"(%iv) {type = "LB"} : (index) -> (index)
    %1 = "test.reify_bound"(%iv) {type = "UB"} : (index) -> (index)
    "test.some_use"(%0, %1) : (index, index) -> ()
  }
  return
}
