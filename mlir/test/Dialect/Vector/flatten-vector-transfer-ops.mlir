// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

func.func @flatten_transfer_ops(%arg0: memref<16x16xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %b0 = ub.poison : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %b0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<1x8xf32>
  %1 = vector.transfer_read %arg0[%c0, %c8], %b0 {in_bounds = [true, true]} : memref<16x16xf32>, vector<1x8xf32>
  %2 = vector.shape_cast %0 : vector<1x8xf32> to vector<8xf32>
  %3 = vector.shape_cast %1 : vector<1x8xf32> to vector<8xf32>
  %4 = vector.fma %2, %3, %arg1 : vector<8xf32>
  return %4 : vector<8xf32>
}

// CHECK-LABEL: @flatten_transfer_ops
// CHECK-NOT: vector.transfer_read {{.*}},  vector<1x8xf32>
// CHECK-NOT: vector.transfer_read {{.*}},  vector<1x8xf32>
// CHECK: vector.transfer_read {{.*}},  vector<8xf32> 
// CHECK-NEXT: vector.transfer_read {{.*}},  vector<8xf32>
// CHECK-NOT: vector.shape_cast
// CHECK-NOT: vector.shape_cast

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.flatten_vector_transfer_ops
    } : !transform.any_op
    transform.yield
  }
}
