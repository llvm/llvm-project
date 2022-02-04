// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='lower-permutation-maps=true' -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='full-unroll=true' -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='full-unroll=true lower-permutation-maps=true' -lower-affine -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Test for special cases of 1D vector transfer ops.

memref.global "private" @gv : memref<5x6xf32> =
    dense<[[0. , 1. , 2. , 3. , 4. , 5. ],
           [10., 11., 12., 13., 14., 15.],
           [20., 21., 22., 23., 24., 25.],
           [30., 31., 32., 33., 34., 35.],
           [40., 41., 42., 43., 44., 45.]]>

// Non-contiguous, strided load.
func @transfer_read_1d(%A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<(d0, d1) -> (6 * d0 + 2 * d1)>

// Vector load with unit stride only on last dim.
func @transfer_read_1d_unit_stride(%A : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %fm42 = arith.constant -42.0: f32
  scf.for %arg2 = %c1 to %c5 step %c2 {
    scf.for %arg3 = %c0 to %c6 step %c3 {
      %0 = memref.subview %A[%arg2, %arg3] [1, 2] [1, 1]
          : memref<?x?xf32> to memref<1x2xf32, #map0>
      %1 = vector.transfer_read %0[%c0, %c0], %fm42 {in_bounds=[true]}
          : memref<1x2xf32, #map0>, vector<2xf32>
      vector.print %1 : vector<2xf32>
    }
  }
  return
}

// Vector load with unit stride only on last dim. Strides are not static, so
// codegen must go through VectorToSCF 1D lowering.
func @transfer_read_1d_non_static_unit_stride(%A : memref<?x?xf32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %fm42 = arith.constant -42.0: f32
  %1 = memref.reinterpret_cast %A to offset: [%c6], sizes: [%c1, %c2],  strides: [%c6, %c1]
      : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
  %2 = vector.transfer_read %1[%c2, %c1], %fm42 {in_bounds=[true]}
      : memref<?x?xf32, offset: ?, strides: [?, ?]>, vector<4xf32>
  vector.print %2 : vector<4xf32>
  return
}

// Vector load where last dim has non-unit stride.
func @transfer_read_1d_non_unit_stride(%A : memref<?x?xf32>) {
  %B = memref.reinterpret_cast %A to offset: [0], sizes: [4, 3], strides: [6, 2]
      : memref<?x?xf32> to memref<4x3xf32, #map1>
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %fm42 = arith.constant -42.0: f32
  %vec = vector.transfer_read %B[%c2, %c1], %fm42 {in_bounds=[false]} : memref<4x3xf32, #map1>, vector<3xf32>
  vector.print %vec : vector<3xf32>
  return
}

// Broadcast.
func @transfer_read_1d_broadcast(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

// Non-contiguous, strided load.
func @transfer_read_1d_in_bounds(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]}
      : memref<?x?xf32>, vector<3xf32>
  vector.print %f: vector<3xf32>
  return
}

// Non-contiguous, strided load.
func @transfer_read_1d_mask(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1, 0, 1]> : vector<9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d0)>}
      : memref<?x?xf32>, vector<9xf32>
  vector.print %f: vector<9xf32>
  return
}

// Non-contiguous, strided load.
func @transfer_read_1d_mask_in_bounds(
    %A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[1, 0, 1]> : vector<3xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]}
      : memref<?x?xf32>, vector<3xf32>
  vector.print %f: vector<3xf32>
  return
}

// Non-contiguous, strided store.
func @transfer_write_1d(%A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fn1 = arith.constant -1.0 : f32
  %vf0 = vector.splat %fn1 : vector<7xf32>
  vector.transfer_write %vf0, %A[%base1, %base2]
    {permutation_map = affine_map<(d0, d1) -> (d0)>}
    : vector<7xf32>, memref<?x?xf32>
  return
}

// Non-contiguous, strided store.
func @transfer_write_1d_mask(%A : memref<?x?xf32>, %base1 : index, %base2 : index) {
  %fn1 = arith.constant -2.0 : f32
  %vf0 = vector.splat %fn1 : vector<7xf32>
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1]> : vector<7xi1>
  vector.transfer_write %vf0, %A[%base1, %base2], %mask
    {permutation_map = affine_map<(d0, d1) -> (d0)>}
    : vector<7xf32>, memref<?x?xf32>
  return
}

func @entry() {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index
  %0 = memref.get_global @gv : memref<5x6xf32>
  %A = memref.cast %0 : memref<5x6xf32> to memref<?x?xf32>

  // 1. Read from 2D memref on first dimension. Cannot be lowered to an LLVM
  //    vector load. Instead, generates scalar loads.
  call @transfer_read_1d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 12, 22, 32, 42, -42, -42, -42, -42, -42 )

  // 2.a. Read 1D vector from 2D memref with non-unit stride on first dim.
  call @transfer_read_1d_unit_stride(%A) : (memref<?x?xf32>) -> ()
  // CHECK: ( 10, 11 )
  // CHECK: ( 13, 14 )
  // CHECK: ( 30, 31 )
  // CHECK: ( 33, 34 )

  // 2.b. Read 1D vector from 2D memref with non-unit stride on first dim.
  //      Strides are non-static.
  call @transfer_read_1d_non_static_unit_stride(%A) : (memref<?x?xf32>) -> ()
  // CHECK: ( 31, 32, 33, 34 )

  // 3. Read 1D vector from 2D memref with non-unit stride on second dim.
  call @transfer_read_1d_non_unit_stride(%A) : (memref<?x?xf32>) -> ()
  // CHECK: ( 22, 24, -42 )

  // 4. Write to 2D memref on first dimension. Cannot be lowered to an LLVM
  //    vector store. Instead, generates scalar stores.
  call @transfer_write_1d(%A, %c3, %c2) : (memref<?x?xf32>, index, index) -> ()

  // 5. (Same as 1. To check if 4 works correctly.)
  call @transfer_read_1d(%A, %c0, %c2) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 2, 12, 22, -1, -1, -42, -42, -42, -42 )

  // 6. Read a scalar from a 2D memref and broadcast the value to a 1D vector.
  //    Generates a loop with vector.insertelement.
  call @transfer_read_1d_broadcast(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 12, 12, 12, 12, 12, 12, 12, 12, 12 )

  // 7. Read from 2D memref on first dimension. Accesses are in-bounds, so no
  //    if-check is generated inside the generated loop.
  call @transfer_read_1d_in_bounds(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 12, 22, -1 )

  // 8. Optional mask attribute is specified and, in addition, there may be
  //    out-of-bounds accesses.
  call @transfer_read_1d_mask(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 12, -42, -1, -42, -42, -42, -42, -42, -42 )

  // 9. Same as 8, but accesses are in-bounds.
  call @transfer_read_1d_mask_in_bounds(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 12, -42, -1 )

  // 10. Write to 2D memref on first dimension with a mask.
  call @transfer_write_1d_mask(%A, %c1, %c0)
      : (memref<?x?xf32>, index, index) -> ()

  // 11. (Same as 1. To check if 10 works correctly.)
  call @transfer_read_1d(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( 0, -2, 20, -2, 40, -42, -42, -42, -42 )

  return
}
