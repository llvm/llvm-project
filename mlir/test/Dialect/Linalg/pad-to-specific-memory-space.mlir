
// RUN: mlir-opt --test-transform-dialect-interpreter -cse -canonicalize -split-input-file -verify-diagnostics %s | FileCheck %s

#map = affine_map<()[s0] -> (-s0 + 12, 7)>

// CHECK-LABEL: func @pad_to_memory_space(
//  CHECK-SAME:     %[[arg0:.*]]: memref<24x12xf32, strided<[?, ?], offset: ?>>,
//  CHECK-SAME:     %[[arg1:.*]]: memref<12x25xf32, strided<[?, ?], offset: ?>>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<24x25xf32, strided<[?, ?], offset: ?>>,
func.func @pad_to_memory_space(%arg0: tensor<24x12xf32>,
                               %arg1: tensor<12x25xf32>,
                               %arg2: tensor<24x25xf32>,
                               %iv0 : index, %iv1 : index,
                               %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map()[%iv2]

  // CHECK: %[[s0:.*]] = memref.subview %[[arg0]]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  // CHECK: %[[s1:.*]] = memref.subview %[[arg1]]
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  // CHECK: %[[s2:.*]] = memref.subview %[[arg2]]
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  // CHECK: %[[alloc0:.*]] = memref.alloc() : memref<4x7xf32, 3>
  // CHECK: linalg.fill {{.*}} outs(%[[alloc0]]
  // CHECK: %[[alloc0_view:.*]] = memref.subview %[[alloc0]][0, 0] [4, %{{.*}}] [1, 1]
  // CHECK: memref.copy %[[s0]], %[[alloc0_view]]

  // CHECK: %[[alloc1:.*]] = memref.alloc() : memref<7x5xf32, 3>
  // CHECK: linalg.fill {{.*}} outs(%[[alloc1]]
  // CHECK: %[[alloc1_view:.*]] = memref.subview %[[alloc1]][0, 0] [%{{.*}}, 5] [1, 1]
  // CHECK: memref.copy %[[s1]], %[[alloc1_view]]

  // CHECK: %[[alloc2:.*]] = memref.alloc() : memref<4x5xf32, 3>
  // CHECK: linalg.fill {{.*}} outs(%[[alloc2]]
  // No subview because there is 0 padding
  // CHECK: memref.copy %[[s2]], %[[alloc2]]

  // CHECK: linalg.matmul ins(%[[alloc0]], %[[alloc1]] : {{.*}}) outs(%[[alloc2]] : {{.*}})
  // Copy back result.
  // CHECK: memref.copy %[[alloc2]], %[[s2]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>

  // insert_slice bufferizes to a no-op.
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %padded, %pad = transform.structured.pad %0 {
    padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
    padding_dimensions=[0, 1, 2],
    pack_paddings=[1, 1, 1]
  } : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %pad_result = transform.get_result %pad[0] : (!transform.any_op) -> !transform.any_value
  %buffer, %replacement = transform.structured.bufferize_to_allocation %pad_result {memory_space = 3}
  %2 = transform.bufferization.one_shot_bufferize %arg1 {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

}
