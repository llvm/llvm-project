// RUN: mlir-opt -shape-to-shape-lowering -split-input-file %s | FileCheck %s

// CHECK-LABEL: func @num_elements_to_reduce(
// CHECK-SAME:    [[ARG:%.*]]: !shape.shape) -> !shape.size {
func @num_elements_to_reduce(%shape : !shape.shape) -> !shape.size {
  %num_elements = shape.num_elements %shape
  return %num_elements : !shape.size
}
// CHECK: [[C1:%.*]] = shape.const_size 1
// CHECK: [[NUM_ELEMENTS:%.*]] = shape.reduce([[ARG]], [[C1]]) : !shape.shape -> !shape.size
// CHECK: ^bb0({{.*}}: index, [[DIM:%.*]]: !shape.size, [[ACC:%.*]]: !shape.size
// CHECK:   [[NEW_ACC:%.*]] = shape.mul [[DIM]], [[ACC]]
// CHECK:   shape.yield [[NEW_ACC]] : !shape.size
// CHECK: }
// CHECK: return [[NUM_ELEMENTS]] : !shape.size

