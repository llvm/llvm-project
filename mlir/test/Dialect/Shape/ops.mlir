// RUN: mlir-opt -split-input-file %s | mlir-opt | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: shape_num_elements
func @shape_num_elements(%shape : !shape.shape) -> !shape.size {
  %init = shape.const_size 1
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index : index, %extent : !shape.size, %acc : !shape.size):
      %acc_next = shape.mul %acc, %extent
      shape.yield %acc_next : !shape.size
  }
  return %num_elements : !shape.size
}

// CHECK-LABEL: extent_tensor_num_elements
func @extent_tensor_num_elements(%shape : tensor<?xindex>) -> index {
  %init = constant 1 : index
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index : index, %extent : index, %acc : index):
      %acc_next = muli %acc, %extent : index
      shape.yield %acc_next : index
  }
  return %num_elements : index
}

func @test_shape_num_elements_unknown() {
  %0 = "shape.unknown_shape"() : () -> !shape.shape
  %1 = call @shape_num_elements(%0) : (!shape.shape) -> (!shape.size)
  %2 = "shape.print"(%1) : (!shape.size) -> !shape.size
  return
}

func @test_shape_num_elements_fixed() {
  %0 = shape.const_shape [1, 57, 92]
  %1 = call @shape_num_elements(%0) : (!shape.shape) -> (!shape.size)
  %3 = "shape.print"(%1) : (!shape.size) -> !shape.size
  return
}

func @test_broadcast_fixed() {
  %0 = shape.const_shape [10, 1, 57, 92]
  %1 = shape.const_shape [4, 57, 92]
  %2 = shape.broadcast %0, %1
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_fixed() {
  %0 = shape.const_shape [4, 57, 92]
  %1 = shape.const_shape [4, 57, 92]
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_unknown() {
  %0 = shape.const_shape [4, -1, 92]
  %1 = shape.const_shape [-1, 57, 92]
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_fixed_mismatch() {
  %0 = shape.const_shape [4, 57, 92]
  %1 = shape.const_shape [2, 57, 92]
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_parse_const_shape() {
  %0 = shape.const_shape []
  %1 = shape.const_shape [1, 2, 3]
  return
}

func @test_shape_of(%arg0: tensor<?xf32>) -> !shape.shape {
  %0 = shape.shape_of %arg0 : tensor<?xf32>
  return %0 : !shape.shape
}

func @test_constraints() {
  %0 = shape.const_shape []
  %1 = shape.const_shape [1, 2, 3]
  %w0 = shape.cstr_broadcastable %0, %1 : !shape.shape, !shape.shape
  %w1 = shape.cstr_eq %0, %1
  %w2 = shape.const_witness true
  %w3 = shape.const_witness false
  %w4 = shape.assuming_all %w0, %w1, %w2, %w3
  shape.assuming %w4 -> !shape.shape {
    %2 = shape.any %0, %1
    shape.assuming_yield %2 : !shape.shape
  }
  return
}

func @broadcastable_on_extent_tensors(%lhs : tensor<?xindex>,
                                      %rhs : tensor<?xindex>) {
  %w0 = shape.cstr_broadcastable %lhs, %rhs : tensor<?xindex>, tensor<?xindex>
  return
}

func @test_mul(%lhs: !shape.size, %rhs: !shape.size) -> !shape.size {
  %product = shape.mul %lhs, %rhs
  return %product: !shape.size
}

func @const_size() {
  // CHECK: %c1 = shape.const_size 1
  // CHECK: %c2 = shape.const_size 2
  // CHECK: %c2_0 = shape.const_size 2
  %0 = shape.const_size 1
  %1 = shape.const_size 2
  %2 = shape.const_size 2
  return
}

func @test_to_extent_tensor(%arg: !shape.shape) -> tensor<3xindex> {
  %0 = shape.to_extent_tensor %arg : tensor<3xindex>
  return %0 : tensor<3xindex>
}

func @test_from_extent_tensor(%arg: tensor<?xindex>) -> !shape.shape {
  %0 = shape.from_extent_tensor %arg : tensor<?xindex>
  return %0 : !shape.shape
}

func @rank(%shape : !shape.shape) -> !shape.size {
  %rank = shape.rank %shape : !shape.shape
  return %rank : !shape.size
}

func @rank_on_extent_tensor(%shape : tensor<?xindex>) -> !shape.size {
  %rank = shape.rank %shape : tensor<?xindex>
  return %rank : !shape.size
}


func @shape_eq_on_shapes(%a : !shape.shape, %b : !shape.shape) -> i1 {
  %result = shape.shape_eq %a, %b : !shape.shape, !shape.shape
  return %result : i1
}

func @shape_eq_on_tensors(%a : tensor<?xindex>, %b : tensor<?xindex>) -> i1 {
  %result = shape.shape_eq %a, %b : tensor<?xindex>, tensor<?xindex>
  return %result : i1
}

func @shape_eq_on_mixed(%a : tensor<?xindex>, %b : !shape.shape) -> i1 {
  %result = shape.shape_eq %a, %b : tensor<?xindex>, !shape.shape
  return %result : i1
}
