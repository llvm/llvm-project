// RUN: mlir-opt -split-input-file %s | mlir-opt | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: shape_num_elements
func @shape_num_elements(%shape : !shape.shape) -> !shape.size {
  %0 = shape.const_size 0
  %1 = "shape.reduce"(%shape, %0) ( {
    ^bb0(%index: i32, %dim: !shape.size, %lci: !shape.size):
      %acc = "shape.add"(%lci, %dim) : (!shape.size, !shape.size) -> !shape.size
      "shape.yield"(%acc) : (!shape.size) -> ()
    }) : (!shape.shape, !shape.size) -> (!shape.size)
  return %1 : !shape.size
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

func @test_broadcastable_fixed() {
  %0 = shape.const_shape [10, 1, 57, 92]
  %1 = shape.const_shape [4, 57, 92]
  %2 = "shape.broadcastable"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
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
  %w0 = shape.cstr_broadcastable %0, %1
  %w1 = shape.cstr_eq %0, %1
  %w3 = shape.assuming_all %w0, %w1
  shape.assuming %w3 -> !shape.shape {
    %2 = shape.any %0, %1
    shape.assuming_yield %2 : !shape.shape
  }
  return
}
