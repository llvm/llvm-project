// RUN: mlir-opt -split-input-file %s | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: shape_num_elements
func @shape_num_elements(%shape : !shape.shape) -> !shape.size {
  %0 = shape.constant 0 : !shape.size
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
  %0 = "shape.constant"() { value = [1, 57, 92] }: () -> !shape.shape
  %1 = call @shape_num_elements(%0) : (!shape.shape) -> (!shape.size)
  %3 = "shape.print"(%1) : (!shape.size) -> !shape.size
  return
}

func @test_broadcastable_fixed() {
  %0 = "shape.constant"() { value = [10, 1, 57, 92] }: () -> !shape.shape
  %1 = "shape.constant"() { value = [4, 57, 92] }: () -> !shape.shape
  %2 = "shape.broadcastable"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_fixed() {
  %0 = "shape.constant"() { value = [4, 57, 92] }: () -> !shape.shape
  %1 = "shape.constant"() { value = [4, 57, 92] }: () -> !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_unknown() {
  %0 = "shape.constant"() { value = [4, -1, 92] }: () -> !shape.shape
  %1 = "shape.constant"() { value = [-1, 57, 92] }: () -> !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}

func @test_shape_any_fixed_mismatch() {
  %0 = "shape.constant"() { value = [4, 57, 92] }: () -> !shape.shape
  %1 = "shape.constant"() { value = [2, 57, 92] }: () -> !shape.shape
  %2 = "shape.join"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  %3 = "shape.print"(%2) : (!shape.shape) -> !shape.shape
  return
}
