// RUN: mlir-opt %s -test-progressive-convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Test for special cases of 1D vector transfer ops.

func @transfer_read_1d(%A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

func @transfer_read_1d_broadcast(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

func @transfer_read_1d_in_bounds(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]}
      : memref<?x?xf32>, vector<3xf32>
  vector.print %f: vector<3xf32>
  return
}

func @transfer_read_1d_mask(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = constant -42.0: f32
  %mask = constant dense<[1, 0, 1, 0, 1, 1, 1, 0, 1]> : vector<9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

func @transfer_read_1d_mask_in_bounds(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = constant -42.0: f32
  %mask = constant dense<[1, 0, 1]> : vector<3xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]}
      : memref<?x?xf32>, vector<3xf32>
  vector.print %f: vector<3xf32>
  return
}

func @transfer_write_1d(%A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fn1 = constant -1.0 : f32
  %vf0 = splat %fn1 : vector<7xf32>
  vector.transfer_write %vf0, %A[%base1, %base2]
    {permutation_map = affine_map<(d0, d1) -> (d0)>}
    : vector<7xf32>, memref<?x?xf32>
  return
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c2 = constant 2: index
  %c3 = constant 3: index
  %f10 = constant 10.0: f32
  // work with dims of 4, not of 3
  %first = constant 5: index
  %second = constant 6: index
  %A = memref.alloc(%first, %second) : memref<?x?xf32>
  scf.for %i = %c0 to %first step %c1 {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    %fi10 = mulf %fi, %f10 : f32
    scf.for %j = %c0 to %second step %c1 {
        %j32 = index_cast %j : index to i32
        %fj = sitofp %j32 : i32 to f32
        %fres = addf %fi10, %fj : f32
        memref.store %fres, %A[%i, %j] : memref<?x?xf32>
    }
  }

  // Read from 2D memref on first dimension. Cannot be lowered to an LLVM
  // vector load. Instead, generates scalar loads.
  call @transfer_read_1d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()
  // Write to 2D memref on first dimension. Cannot be lowered to an LLVM
  // vector store. Instead, generates scalar stores.
  call @transfer_write_1d(%A, %c3, %c2) : (memref<?x?xf32>, index, index) -> ()
  // (Same as above.)
  call @transfer_read_1d(%A, %c0, %c2) : (memref<?x?xf32>, index, index) -> ()
  // Read a scalar from a 2D memref and broadcast the value to a 1D vector.
  // Generates a loop with vector.insertelement.
  call @transfer_read_1d_broadcast(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  //  Read from 2D memref on first dimension. Accesses are in-bounds, so no
  // if-check is generated inside the generated loop.
  call @transfer_read_1d_in_bounds(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // Optional mask attribute is specified and, in addition, there may be
  // out-of-bounds accesses.
  call @transfer_read_1d_mask(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // Same as above, but accesses are in-bounds.
  call @transfer_read_1d_mask_in_bounds(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  return
}

// CHECK: ( 12, 22, 32, 42, -42, -42, -42, -42, -42 )
// CHECK: ( 2, 12, 22, -1, -1, -42, -42, -42, -42 )
// CHECK: ( 12, 12, 12, 12, 12, 12, 12, 12, 12 )
// CHECK: ( 12, 22, -1 )
// CHECK: ( 12, -42, -1, -42, -42, -42, -42, -42, -42 )
// CHECK: ( 12, -42, -1 )
