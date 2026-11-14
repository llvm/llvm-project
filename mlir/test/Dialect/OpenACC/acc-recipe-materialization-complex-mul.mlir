// RUN: mlir-opt %s -acc-recipe-materialization -split-input-file -verify-diagnostics

// Complex multiplication cannot be split into independent real and imaginary
// reductions. Reject it while the reduction recipe is materialized, for both
// a scalar and an array.

acc.reduction.recipe @reduction_mul_scalar : memref<complex<f32>> reduction_operator <mul> init {
^bb0(%arg0: memref<complex<f32>>):
  %re = arith.constant 1.000000e+00 : f32
  %im = arith.constant 0.000000e+00 : f32
  %one = complex.create %re, %im : complex<f32>
  %0 = memref.alloca() : memref<complex<f32>>
  memref.store %one, %0[] : memref<complex<f32>>
  acc.yield %0 : memref<complex<f32>>
} combiner {
^bb0(%arg0: memref<complex<f32>>, %arg1: memref<complex<f32>>):
  %0 = memref.load %arg0[] : memref<complex<f32>>
  %1 = memref.load %arg1[] : memref<complex<f32>>
  %2 = complex.mul %0, %1 : complex<f32>
  memref.store %2, %arg0[] : memref<complex<f32>>
  acc.yield %arg0 : memref<complex<f32>>
}

func.func @scalar_complex_mul(%arg0: memref<complex<f32>>) {
  // expected-error @below {{not yet implemented: reduction: complex multiply reduction}}
  %red = acc.reduction varPtr(%arg0 : memref<complex<f32>>) recipe(@reduction_mul_scalar) -> memref<complex<f32>>
  acc.parallel reduction(%red : memref<complex<f32>>) {
    acc.yield
  }
  return
}

// -----

acc.reduction.recipe @reduction_mul_array : memref<4xcomplex<f32>> reduction_operator <mul> init {
^bb0(%arg0: memref<4xcomplex<f32>>):
  %0 = memref.alloca() : memref<4xcomplex<f32>>
  acc.yield %0 : memref<4xcomplex<f32>>
} combiner {
^bb0(%arg0: memref<4xcomplex<f32>>, %arg1: memref<4xcomplex<f32>>):
  acc.yield %arg0 : memref<4xcomplex<f32>>
}

func.func @array_complex_mul(%arg0: memref<4xcomplex<f32>>) {
  // expected-error @below {{not yet implemented: reduction: complex multiply reduction}}
  %red = acc.reduction varPtr(%arg0 : memref<4xcomplex<f32>>) recipe(@reduction_mul_array) -> memref<4xcomplex<f32>>
  acc.parallel reduction(%red : memref<4xcomplex<f32>>) {
    acc.yield
  }
  return
}
