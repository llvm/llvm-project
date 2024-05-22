// RUN: mlir-opt %s -test-tensor-copy-insertion -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-tensor-copy-insertion="bufferize-function-boundaries" -split-input-file | FileCheck %s --check-prefix=CHECK-FUNC

// CHECK-LABEL: func @extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
// CHECK-FUNC-LABEL: func @extract_slice(
func.func @extract_slice(%t: tensor<?xf32>, %idx: index, %f: f32)
  -> (tensor<5xf32>, tensor<?xf32>)
{
  // CHECK: %[[extract_slice:.*]] = tensor.extract_slice %[[t]][10] [5] [1]
  %0 = tensor.extract_slice %t[10][5][1] : tensor<?xf32> to tensor<5xf32>
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() copy(%[[extract_slice]]) : tensor<5xf32>
  // CHECK-FUNC: bufferization.alloc_tensor() copy(%{{.*}}) : tensor<5xf32>
  // CHECK: %[[insert:.*]] = tensor.insert %{{.*}} into %[[alloc]]
  %1 = tensor.insert %f into %0[%idx] : tensor<5xf32>
  // CHECK: return %[[insert]], %[[t]]
  return %1, %t : tensor<5xf32>, tensor<?xf32>
}
