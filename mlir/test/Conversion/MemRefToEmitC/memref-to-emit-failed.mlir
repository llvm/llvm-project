// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file -verify-diagnostics

// Unranked memrefs are not converted
// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @memref_unranked(%arg0 : memref<*xf32>) {
  return
}

// -----

// Memrefs with dynamic shapes are not converted
// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @memref_dynamic_shape(%arg0 : memref<2x?xf32>) {
  return
}

// -----

func.func @memref_op(%arg0 : memref<2x4xf32>) {
  // expected-error@+1 {{failed to legalize operation 'memref.copy' that was explicitly marked illegal}}
  memref.copy %arg0, %arg0 : memref<2x4xf32> to memref<2x4xf32>
  return
}

// -----

func.func @alloca_with_dynamic_shape() {
  %0 = index.constant 1
  // expected-error@+1 {{failed to legalize operation 'memref.alloca' that was explicitly marked illegal}}
  %1 = memref.alloca(%0) : memref<4x?xf32>
  return
}

// -----

func.func @alloca_with_alignment() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca' that was explicitly marked illegal}}
  %1 = memref.alloca() {alignment = 64 : i64}: memref<4xf32>
  return
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @non_identity_layout(%arg0 : memref<4x3xf32, affine_map<(d0, d1) -> (d1, d0)>>) {
  return
}

// -----

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @zero_rank(%arg0 : memref<f32>) {
  return
}
