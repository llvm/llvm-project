// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf,lower-affine,convert-scf-to-cf),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-ub-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-vector-to-scf{full-unroll=true},lower-affine,convert-scf-to-cf),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-ub-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

memref.global "private" @gv : memref<3x4xf32> = dense<[[0. , 1. , 2. , 3. ],
                                                       [10., 11., 12., 13.],
                                                       [20., 21., 22., 23.]]>

// Vector load.
func.func @transfer_read_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Vector load with mask.
func.func @transfer_read_2d_mask(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[[1, 0, 1, 0, 1, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 1, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1, 0, 1]]> : vector<4x9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Vector load with mask + transpose.
func.func @transfer_read_2d_mask_transposed(
    %A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[[1, 0, 1, 0, 1, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 1, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 1, 1, 0, 1]]> : vector<4x9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} :
    memref<?x?xf32>, vector<9x4xf32>
  vector.print %f: vector<9x4xf32>
  return
}

// Vector load with mask + broadcast.
func.func @transfer_read_2d_mask_broadcast(
    %A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1, 0, 1]> : vector<9xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (0, d1)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Transpose + vector load with mask + broadcast.
func.func @transfer_read_2d_mask_transpose_broadcast_last_dim(
    %A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %mask = arith.constant dense<[1, 0, 1, 1]> : vector<4xi1>
  %f = vector.transfer_read %A[%base1, %base2], %fm42, %mask
      {permutation_map = affine_map<(d0, d1) -> (d1, 0)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Load + transpose.
func.func @transfer_read_2d_transposed(
    %A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Load 1D + broadcast to 2D.
func.func @transfer_read_2d_broadcast(
    %A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fm42 = arith.constant -42.0: f32
  %f = vector.transfer_read %A[%base1, %base2], %fm42
      {permutation_map = affine_map<(d0, d1) -> (d1, 0)>} :
    memref<?x?xf32>, vector<4x9xf32>
  vector.print %f: vector<4x9xf32>
  return
}

// Vector store.
func.func @transfer_write_2d(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fn1 = arith.constant -1.0 : f32
  %vf0 = vector.splat %fn1 : vector<1x4xf32>
  vector.transfer_write %vf0, %A[%base1, %base2]
    {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    vector<1x4xf32>, memref<?x?xf32>
  return
}

// Vector store with mask.
func.func @transfer_write_2d_mask(%A : memref<?x?xf32>, %base1: index, %base2: index) {
  %fn1 = arith.constant -2.0 : f32
  %mask = arith.constant dense<[[1, 0, 1, 0]]> : vector<1x4xi1>
  %vf0 = vector.splat %fn1 : vector<1x4xf32>
  vector.transfer_write %vf0, %A[%base1, %base2], %mask
    {permutation_map = affine_map<(d0, d1) -> (d0, d1)>} :
    vector<1x4xf32>, memref<?x?xf32>
  return
}

func.func @entry() {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3: index
  %c10 = arith.constant 10 : index
  %0 = memref.get_global @gv : memref<3x4xf32>
  %A = memref.cast %0 : memref<3x4xf32> to memref<?x?xf32>

  // 1.a. Read 2D vector from 2D memref.
  call @transfer_read_2d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 12, 13, -42, -42, -42, -42, -42, -42, -42 ), ( 22, 23, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 1.b. Read 2D vector from 2D memref. Starting position of first dim is
  //      out-of-bounds.
  call @transfer_read_2d(%A, %c3, %c2) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 1.c. Read 2D vector from 2D memref. Starting position of second dim is
  //      out-of-bounds.
  call @transfer_read_2d(%A, %c1, %c10) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 2. Read 2D vector from 2D memref at specified location and transpose the
  //    result.
  call @transfer_read_2d_transposed(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 12, 22, -42, -42, -42, -42, -42, -42, -42 ), ( 13, 23, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 3. Read 2D vector from 2D memref with a 2D mask. In addition, some
  //    accesses are out-of-bounds.
  call @transfer_read_2d_mask(%A, %c0, %c0)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 0, -42, 2, -42, -42, -42, -42, -42, -42 ), ( -42, -42, 12, 13, -42, -42, -42, -42, -42 ), ( 20, 21, 22, 23, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 4. Same as 3, but transpose the result.
  call @transfer_read_2d_mask_transposed(%A, %c0, %c0)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 0, -42, 20, -42 ), ( -42, -42, 21, -42 ), ( 2, 12, 22, -42 ), ( -42, 13, 23, -42 ), ( -42, -42, -42, -42 ), ( -42, -42, -42, -42 ), ( -42, -42, -42, -42 ), ( -42, -42, -42, -42 ), ( -42, -42, -42, -42 ) )

  // 5. Read 1D vector from 2D memref at specified location and broadcast the
  //    result to 2D.
  call @transfer_read_2d_broadcast(%A, %c1, %c2)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 12, 12, 12, 12, 12, 12, 12, 12, 12 ), ( 13, 13, 13, 13, 13, 13, 13, 13, 13 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 6. Read 1D vector from 2D memref at specified location with mask and
  //    broadcast the result to 2D.
  call @transfer_read_2d_mask_broadcast(%A, %c2, %c1)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 21, -42, 23, -42, -42, -42, -42, -42, -42 ), ( 21, -42, 23, -42, -42, -42, -42, -42, -42 ), ( 21, -42, 23, -42, -42, -42, -42, -42, -42 ), ( 21, -42, 23, -42, -42, -42, -42, -42, -42 ) )

  // 7. Read 1D vector from 2D memref (second dimension) at specified location
  //    with mask and broadcast the result to 2D. In this test case, mask
  //    elements must be evaluated before lowering to an (N>1)-D transfer.
  call @transfer_read_2d_mask_transpose_broadcast_last_dim(%A, %c0, %c1)
      : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 1, 1, 1, 1, 1, 1, 1, 1, 1 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ), ( 3, 3, 3, 3, 3, 3, 3, 3, 3 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 8. Write 2D vector into 2D memref at specified location.
  call @transfer_write_2d(%A, %c1, %c2) : (memref<?x?xf32>, index, index) -> ()

  // 9. Read memref to verify step 8.
  call @transfer_read_2d(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 0, 1, 2, 3, -42, -42, -42, -42, -42 ), ( 10, 11, -1, -1, -42, -42, -42, -42, -42 ), ( 20, 21, 22, 23, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  // 10. Write 2D vector into 2D memref at specified location with mask.
  call @transfer_write_2d_mask(%A, %c0, %c2) : (memref<?x?xf32>, index, index) -> ()

  // 11. Read memref to verify step 10.
  call @transfer_read_2d(%A, %c0, %c0) : (memref<?x?xf32>, index, index) -> ()
  // CHECK: ( ( 0, 1, -2, 3, -42, -42, -42, -42, -42 ), ( 10, 11, -1, -1, -42, -42, -42, -42, -42 ), ( 20, 21, 22, 23, -42, -42, -42, -42, -42 ), ( -42, -42, -42, -42, -42, -42, -42, -42, -42 ) )

  return
}

