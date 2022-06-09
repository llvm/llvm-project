// RUN: mlir-opt %s -tensor-copy-insertion -split-input-file | FileCheck %s
// RUN: mlir-opt %s -tensor-copy-insertion="bufferize-function-boundaries allow-return-allocs" -split-input-file | FileCheck %s --check-prefix=CHECK-FUNC

// CHECK-LABEL: func @read_after_write_conflict(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
// CHECK-FUNC-LABEL: func @read_after_write_conflict(
func.func @read_after_write_conflict(%t: tensor<?xf32>, %idx: index, %f: f32)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  // CHECK: %[[copy:.*]] = bufferization.alloc_tensor() copy(%[[t]]) {escape = false} : tensor<?xf32>
  // CHECK-FUNC: bufferization.alloc_tensor() copy(%{{.*}}) {escape = true} : tensor<?xf32>
  // CHECK: %[[insert:.*]] = tensor.insert %{{.*}} into %[[copy]]
  %0 = tensor.insert %f into %t[%idx] : tensor<?xf32>
  // CHECK: return %[[insert]], %[[t]]
  return %0, %t : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @return_alloc_tensor
// CHECK-FUNC-LABEL: func @return_alloc_tensor
func.func @return_alloc_tensor() -> (tensor<5xf32>) {
  // CHECK: bufferization.alloc_tensor() {escape = false} : tensor<5xf32>
  // CHECK-FUNC: bufferization.alloc_tensor() {escape = true} : tensor<5xf32>
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
  return %0 : tensor<5xf32>
}
