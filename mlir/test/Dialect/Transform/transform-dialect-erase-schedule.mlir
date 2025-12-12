// RUN: mlir-opt %s -test-transform-dialect-erase-schedule | FileCheck %s

module attributes {transform.with_named_sequence} {
    func.func @transform_example(%arg0: !transform.any_op) {
        %transform_copy = transform.structured.match ops{["linalg.copy"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.nvgpu.rewrite_copy_as_tma %transform_copy : (!transform.any_op) -> ()
        transform.yield
    }
}

// CHECK-LABEL: module attributes {transform.with_named_sequence} {
// CHECK-NEXT: func.func @transform_example(%arg0: !transform.any_op) {
// CHECK-NEXT: transform.yield
// CHECK-NEXT: }
// CHECK-NEXT: }