// RUN: mlir-opt -outline-shape-computation -split-input-file %s 2>%t | FileCheck %s

func.func @func1(%arg0: !shape.value_shape, %arg1: !shape.value_shape) -> !shape.shape {
  %0 = shape.shape_of %arg0 : !shape.value_shape -> !shape.shape
  %1 = shape.shape_of %arg1 : !shape.value_shape -> !shape.shape
  %2 = shape.meet %0, %1 : !shape.shape, !shape.shape -> !shape.shape
  return %2 : !shape.shape
}
// Make sure with_shape used by call not crash.
// CHECK-LABEL:func.func @func
func.func @func(%arg0: !shape.value_shape, %arg1: !shape.value_shape) -> !shape.shape {
  %0 = shape.shape_of %arg0 : !shape.value_shape -> !shape.shape
  %1 = shape.with_shape %arg1, %0 : !shape.value_shape, !shape.shape
  %2 = call @func1(%arg0, %1) : (!shape.value_shape, !shape.value_shape) -> !shape.shape
  return %2 : !shape.shape
}
