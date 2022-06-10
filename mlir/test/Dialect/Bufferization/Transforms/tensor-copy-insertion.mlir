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

// -----

// CHECK-LABEL: func @do_not_copy_undefined_tensor
func.func @do_not_copy_undefined_tensor(%f: f32, %idx: index)
  -> (tensor<5xf32>, tensor<5xf32>)
{
  // CHECK: bufferization.alloc_tensor() {escape = false} : tensor<5xf32>
  // The second alloc_tensor should not have a copy operand.
  // CHECK: bufferization.alloc_tensor() {escape = false} : tensor<5xf32>
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
  %1 = tensor.insert %f into %0[%idx] : tensor<5xf32>
  return %0, %1 : tensor<5xf32>, tensor<5xf32>
}

// -----

// CHECK-LABEL: func @do_not_copy_when_overwritten
func.func @do_not_copy_when_overwritten(%t: tensor<5xf32>, %f: f32)
  -> (tensor<5xf32>, tensor<5xf32>)
{
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() {escape = false} : tensor<5xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[alloc]] : tensor<5xf32>)
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    outs(%t : tensor<5xf32>) {
      ^bb0(%arg0 : f32) :
        linalg.yield %f : f32
    } -> tensor<5xf32>
  return %t, %r : tensor<5xf32>, tensor<5xf32>
}

// -----

// CHECK-LABEL: func @do_not_copy_when_result_not_read
func.func @do_not_copy_when_result_not_read(%t: tensor<5xf32>, %f: f32)
  -> (tensor<3xf32>)
{
  %0 = tensor.extract_slice %t[0][3][1] : tensor<5xf32> to tensor<3xf32>
  // CHECK: %[[alloc:.*]] = bufferization.alloc_tensor() {escape = false} : tensor<3xf32>
  // CHECK: linalg.generic {{.*}} outs(%[[alloc]] : tensor<3xf32>)
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    outs(%0 : tensor<3xf32>) {
      ^bb0(%arg0 : f32) :
        linalg.yield %f : f32
    } -> tensor<3xf32>
  return %r : tensor<3xf32>
}
