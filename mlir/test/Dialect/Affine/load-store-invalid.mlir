// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @load_too_many_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index, %arg3: index) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.load"(%arg0, %arg1, %arg2, %arg3)
    {map = affine_map<(i, j) -> (i, j)> } : (memref<?x?xf32>, index, index, index) -> f32
}

// -----

func.func @load_too_few_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.load"(%arg0, %arg1)
    {map = affine_map<(i, j) -> (i, j)> } : (memref<?x?xf32>, index) -> f32
}

// -----

func.func @store_too_many_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index,
                                    %arg3: index, %val: f32) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.store"(%val, %arg0, %arg1, %arg2, %arg3)
    {map = affine_map<(i, j) -> (i, j)> } : (f32, memref<?x?xf32>, index, index, index) -> ()
}

// -----

func.func @store_too_few_subscripts_map(%arg0: memref<?x?xf32>, %arg1: index, %val: f32) {
  // expected-error@+1 {{op expects as many subscripts as affine map inputs}}
  "affine.store"(%val, %arg0, %arg1)
    {map = affine_map<(i, j) -> (i, j)> } : (f32, memref<?x?xf32>, index) -> ()
}

// -----

func.func @invalid_prefetch_rw(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{rw specifier has to be 'read' or 'write'}}
  affine.prefetch %0[%i], rw, locality<0>, data  : memref<10xf32>
  return
}

// -----

func.func @invalid_prefetch_cache_type(%i : index) {
  %0 = memref.alloc() : memref<10xf32>
  // expected-error@+1 {{cache type has to be 'data' or 'instr'}}
  affine.prefetch %0[%i], read, locality<0>, false  : memref<10xf32>
  return
}
