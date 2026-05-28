// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file -verify-diagnostics

func.func @alloca_with_dynamic_shape() {
  %0 = index.constant 1
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %1 = memref.alloca(%0) : memref<4x?xf32>
  return
}

// -----

func.func @alloca_with_alignment() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() {alignment = 64 : i64}: memref<4xf32>
  return
}

// -----

func.func @alloc_and_dealloc_arg(%arg0: memref<999xi32>) {
  // expected-error@+1 {{failed to legalize operation 'memref.dealloc'}}
  memref.dealloc %arg0 : memref<999xi32>
  return
}

// -----

func.func @alloca_and_dealloc() {
  %0 = memref.alloca() : memref<4xf32>
  // expected-error@+1 {{failed to legalize operation 'memref.dealloc'}}
  memref.dealloc %0 : memref<4xf32>
  return
}

// -----

memref.global "private" constant @g_dense : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>

func.func @get_global_dense_and_dealloc() {
  %0 = memref.get_global @g_dense : memref<4xf32>
  // expected-error@+1 {{failed to legalize operation 'memref.dealloc'}}
  memref.dealloc %0 : memref<4xf32>
  return
}

// -----

memref.global "private" @g_uninit : memref<4xf32> = uninitialized

func.func @get_global_uninit_and_dealloc() {
  %0 = memref.get_global @g_uninit : memref<4xf32>
  // expected-error@+1 {{failed to legalize operation 'memref.dealloc'}}
  memref.dealloc %0 : memref<4xf32>
  return
}

// -----

func.func @non_identity_layout() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4x3xf32, affine_map<(d0, d1) -> (d1, d0)>>
  return
}

// -----

func.func @zero_rank() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<f32>
  return
}

// -----

func.func @zero_dim_rank_1() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<0xf32>
  return
}

// -----

func.func @zero_dim_rank_3() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<2x0x4xf32>
  return
}

// -----

// expected-error@+1 {{failed to legalize operation 'memref.global'}}
memref.global "nested" constant @nested_global : memref<3x7xf32>

// -----

func.func @unsupported_type_f128() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4xf128>
  return
}

// -----

func.func @unsupported_type_i4() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4xi4>
  return
}
