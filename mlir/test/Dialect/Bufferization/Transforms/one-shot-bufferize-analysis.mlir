// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only" \
// RUN:     -allow-unregistered-dialect -split-input-file | FileCheck %s

// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only dump-alias-sets" \
// RUN:     -allow-unregistered-dialect -split-input-file | \
// RUN: FileCheck %s --check-prefix=CHECK-ALIAS-SETS

// CHECK-LABEL: func @unknown_op_aliasing(
// CHECK-ALIAS-SETS-LABEL: func @unknown_op_aliasing(
func.func @unknown_op_aliasing(%f: f32, %f2: f32, %pos: index) -> f32 {
  // CHECK-ALIAS-SETS: %[[empty:.*]] = tensor.empty

  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  // CHECK-ALIAS-SETS: %[[fill1:.*]] = linalg.fill
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // Something must bufferize out-of-place because the op may return an alias
  // of %1.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  %alias = "dummy.dummy_op"(%1) : (tensor<10xf32>) -> (tensor<10xf32>)

  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  // CHECK-ALIAS-SETS: %[[fill2:.*]] = linalg.fill
  // CHECK-ALIAS-SETS-SAME: __opresult_alias_set_attr__ = [{{\[}}"%[[fill2]]", "%[[fill1]]", "%[[empty]]"]]
  %2 = linalg.fill ins(%f2 : f32) outs(%1 : tensor<10xf32>) -> tensor<10xf32>
  %3 = tensor.extract %alias[%pos] : tensor<10xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func @unknown_op_bbarg_aliasing(
// CHECK-ALIAS-SETS-LABEL: func @unknown_op_bbarg_aliasing(
func.func @unknown_op_bbarg_aliasing() {
  %0 = tensor.empty() : tensor<7xf32>

  // %arg0 is not aliasing with %0 because it bufferizes out-of-place.
  // CHECK-ALIAS-SETS: "dummy.dummy_op"
  // CHECK-ALIAS-SETS-NEXT: ^{{.*}}(%[[arg:.*]]: tensor<7xf32>):
  // CHECK-ALIAS-SETS-NEXT: }) {__bbarg_alias_set_attr__ = [{{\[}}[{{\[}}"%[[arg]]"]]]], __inplace_operands_attr__ = ["false"]} : (tensor<7xf32>) -> ()
  "dummy.dummy_op"(%0) ({
  ^bb0(%arg1: tensor<7xf32>):
  }) : (tensor<7xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @unknown_op_writing(
func.func @unknown_op_writing(%f: f32, %f2: f32, %pos: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // The op may bufferize to a memory write, so it must bufferize out-of-place.
  // CHECK: "dummy.dummy_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
  "dummy.dummy_op"(%1) : (tensor<10xf32>) -> ()

  %3 = tensor.extract %1[%pos] : tensor<10xf32>
  return %3 : f32
}

// -----

// CHECK-LABEL: func @read_of_undef_is_not_a_conflict(
func.func @read_of_undef_is_not_a_conflict(%f: f32, %idx: index) -> f32 {
  %0 = tensor.empty() : tensor<10xf32>
  // This can be in-place because the read below does reads undefined data.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "true", "none"]}
  %1 = tensor.insert %f into %0[%idx] : tensor<10xf32>
  %2 = tensor.extract %0[%idx] : tensor<10xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @read_of_alloc_tensor_is_not_a_conflict(
func.func @read_of_alloc_tensor_is_not_a_conflict(%f: f32, %idx: index) -> f32 {
  %0 = bufferization.alloc_tensor() : tensor<10xf32>
  // This can be in-place because the read below does reads undefined data.
  // CHECK: tensor.insert {{.*}} {__inplace_operands_attr__ = ["none", "true", "none"]}
  %1 = tensor.insert %f into %0[%idx] : tensor<10xf32>
  %2 = tensor.extract %0[%idx] : tensor<10xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @to_memref_not_read_only(
func.func @to_memref_not_read_only(%idx : index, %f: f32) -> f32 {
  %t = tensor.generate {
  ^bb0(%i : index):
    tensor.yield %f : f32
  } : tensor<5xf32>
  // Some op may write into the result of to_memref later.
  // CHECK: bufferization.to_memref
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  %m = bufferization.to_memref %t : memref<5xf32>
  %2 = tensor.extract %t[%idx] : tensor<5xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @to_memref_read_only(
func.func @to_memref_read_only(%idx : index, %f: f32) -> f32 {
  %t = tensor.generate {
  ^bb0(%i : index):
    tensor.yield %f : f32
  } : tensor<5xf32>
  // Some op may write into the result of to_memref later.
  // CHECK: bufferization.to_memref
  // CHECK-SAME: {__inplace_operands_attr__ = ["true"]}
  %m = bufferization.to_memref %t {read_only} : memref<5xf32>
  %2 = tensor.extract %t[%idx] : tensor<5xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: func @bbarg_of_unknown_op(
func.func @bbarg_of_unknown_op(%f: f32) {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // The op is not bufferizable because %1 is assumed to alias with %arg1.
  // BlockArguments are considered "not writable" by default. So %1 is also
  // considered "not writable".

  // CHECK: "dummy.dummy_op"
  // CHECK: {__inplace_operands_attr__ = ["false"]} : (tensor<10xf32>) -> ()
  "dummy.dummy_op"(%1) ({
  ^bb0(%arg1: tensor<10xf32>):
  }) : (tensor<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @bbarg_of_unknown_op_2(
func.func @bbarg_of_unknown_op_2(%f: f32) {
  %0 = tensor.empty() : tensor<10xf32>
  // CHECK: linalg.fill {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%f : f32) outs(%0 : tensor<10xf32>) -> tensor<10xf32>

  // The op is not bufferizable because %1 is assumed to alias with %arg1.
  // BlockArguments are considered "not writable" by default. So %1 is also
  // considered "not writable".

  // CHECK: "dummy.dummy_op"
  "dummy.dummy_op"(%1) ({
  ^bb0(%arg1: tensor<10xf32>):
    // CHECK: "dummy.another_op"(%{{.*}}) {__inplace_operands_attr__ = ["false"]}
    "dummy.another_op"(%arg1) : (tensor<10xf32>) -> ()
  }) : (tensor<10xf32>) -> ()
  // CHECK: {__inplace_operands_attr__ = ["false"]} : (tensor<10xf32>) -> ()
  return
}

// -----

// CHECK: func @materialize_in_destination_aliasing(
func.func @materialize_in_destination_aliasing(%t: tensor<?xf32>, %p1: index, %p2: index, %sz: index) -> tensor<5xf32> {
  %buffer = tensor.empty(%sz) : tensor<?xf32>
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]}
  %src = tensor.extract_slice %t[%p1][5][1] : tensor<?xf32> to tensor<5xf32>
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none"]}
  %dest = tensor.extract_slice %t[%p2][5][1] : tensor<?xf32> to tensor<5xf32>
  // CHECK: bufferization.materialize_in_destination
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %r = bufferization.materialize_in_destination %src in %dest : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %r : tensor<5xf32>
}

// -----

// CHECK: func @materialize_in_destination(
func.func @materialize_in_destination(%t: tensor<?xf32>, %sz: index) -> tensor<?xf32> {
  %buffer = tensor.empty(%sz) : tensor<?xf32>
  // CHECK: bufferization.materialize_in_destination
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %r = bufferization.materialize_in_destination %buffer in %buffer : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %r : tensor<?xf32>
}
