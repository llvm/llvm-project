// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

func.func @unknown_op() -> index {
  %0 = "test.foo"() : () -> (tensor<?x?xf32>)
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?x?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @cast(
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[c10]]
func.func @cast(%t: tensor<10xf32>) -> index {
  %0 = tensor.cast %t : tensor<10xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

func.func @cast_unranked(%t: tensor<*xf32>) -> index {
  %0 = tensor.cast %t : tensor<*xf32> to tensor<?xf32>
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @dim(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]]
//       CHECK:   return %[[dim]]
func.func @dim(%t: tensor<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %t, %c0 : tensor<?xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @empty(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @empty(%sz: index) -> (index, index) {
  %0 = tensor.empty(%sz) : tensor<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (tensor<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @extract_slice_dynamic(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @extract_slice_dynamic(%t: tensor<?xf32>, %sz: index) -> index {
  %0 = tensor.extract_slice %t[2][%sz][1] : tensor<?xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_slice_static(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @extract_slice_static(%t: tensor<?xf32>) -> index {
  %0 = tensor.extract_slice %t[2][5][1] : tensor<?xf32> to tensor<5xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<5xf32>) -> (index)
  return %1 : index
}

// -----

func.func @extract_slice_dynamic_constant(%t: tensor<?xf32>, %sz: index) -> index {
  %0 = tensor.extract_slice %t[2][%sz][1] : tensor<?xf32> to tensor<?xf32>
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_constant_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_slice_static_constant(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @extract_slice_static_constant(%t: tensor<?xf32>) -> index {
  %0 = tensor.extract_slice %t[2][5][1] : tensor<?xf32> to tensor<5xf32>
  %1 = "test.reify_constant_bound"(%0) {dim = 0} : (tensor<5xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @extract_slice_rank_reduce(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @extract_slice_rank_reduce(%t: tensor<?x?xf32>, %sz: index) -> index {
  %0 = tensor.extract_slice %t[0, 2][1, %sz][1, 1] : tensor<?x?xf32> to tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @insert(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @insert(%t: tensor<?xf32>, %f: f32, %pos: index) -> index {
  %0 = tensor.insert %f into %t[%pos] : tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
// CHECK: #[[$map1:.*]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-LABEL: func @pad(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x7xf32>, %[[a:.*]]: index, %[[b:.*]]: index
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim0:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   %[[bound0:.*]] = affine.apply #[[$map]]()[%[[dim0]], %[[a]]]
//       CHECK:   %[[bound1:.*]] = affine.apply #[[$map1]]()[%[[b]]]
//       CHECK:   return %[[bound0]], %[[bound1]]
func.func @pad(%t: tensor<?x7xf32>, %a: index, %b: index) -> (index, index) {
  %pad = arith.constant 0.0 : f32
  %0 = tensor.pad %t low[%a, 5] high[%a, %b] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad : f32
    } : tensor<?x7xf32> to tensor<?x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (tensor<?x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @rank(
//  CHECK-SAME:     %[[t:.*]]: tensor<5xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @rank(%t: tensor<5xf32>) -> index {
  %0 = tensor.rank %t : tensor<5xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

func.func @dynamic_dims_are_equal(%t: tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %dim0 = tensor.dim %t, %c0 : tensor<?xf32>
  %dim1 = tensor.dim %t, %c0 : tensor<?xf32>
  // expected-remark @below {{equal}}
  "test.are_equal"(%dim0, %dim1) : (index, index) -> ()
  return
}

// -----

func.func @dynamic_dims_are_different(%t: tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %t, %c0 : tensor<?xf32>
  %val = arith.addi %dim0, %c1 : index
  // expected-remark @below {{different}}
  "test.are_equal"(%dim0, %val) : (index, index) -> ()
  return
}

// -----

func.func @dynamic_dims_are_maybe_equal_1(%t: tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %dim0 = tensor.dim %t, %c0 : tensor<?xf32>
  // expected-error @below {{could not determine equality}}
  "test.are_equal"(%dim0, %c5) : (index, index) -> ()
  return
}

// -----

func.func @dynamic_dims_are_maybe_equal_2(%t: tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %t, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %t, %c1 : tensor<?x?xf32>
  // expected-error @below {{could not determine equality}}
  "test.are_equal"(%dim0, %dim1) : (index, index) -> ()
  return
}
