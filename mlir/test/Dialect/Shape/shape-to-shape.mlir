// RUN: mlir-opt -shape-to-shape-lowering -split-input-file %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: func @num_elements_to_reduce(
// CHECK-SAME:    [[ARG:%.*]]: !shape.shape) -> [[SIZE_TY:!.*]] {
func @num_elements_to_reduce(%shape : !shape.shape) -> !shape.size {
  %num_elements = shape.num_elements %shape
  return %num_elements : !shape.size
}
// CHECK: [[C1:%.*]] = shape.const_size 1
// CHECK: [[NUM_ELEMENTS:%.*]] = shape.reduce([[ARG]], [[C1]])  -> [[SIZE_TY]]
// CHECK: ^bb0({{.*}}: index, [[DIM:%.*]]: [[SIZE_TY]], [[ACC:%.*]]: [[SIZE_TY]]
// CHECK:   [[NEW_ACC:%.*]] = shape.mul [[DIM]], [[ACC]]
// CHECK:   shape.yield [[NEW_ACC]] : [[SIZE_TY]]
// CHECK: }
// CHECK: return [[NUM_ELEMENTS]] : [[SIZE_TY]]

