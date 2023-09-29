// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -canonicalize -buffer-loop-hoisting -drop-equivalent-buffer-results -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" -drop-equivalent-buffer-results -split-input-file | FileCheck %s --check-prefix=CHECK-NO-LAYOUT-MAP

// TODO: Some test cases from this file should be moved to other dialects.

// CHECK-LABEL: func @fill_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
// CHECK-NO-LAYOUT-MAP-LABEL: func @fill_inplace(%{{.*}}: memref<?xf32>) {
func.func @fill_inplace(
    %A : tensor<?xf32> {bufferization.writable = true})
  -> tensor<?xf32>
{
  //     CHECK: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0 : f32

  /// Inplaceable, no alloc
  // CHECK-NOT: alloc
  //     CHECK: linalg.fill ins(%[[F0]] : f32) outs(%[[A]] : memref<?xf32, strided<[?], offset: ?>>)
  %r = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?xf32>
}

// -----

/// No bufferization.writable flag, must allocate.
// CHECK-LABEL: func @not_inplace(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32> {
// CHECK-NO-LAYOUT-MAP-LABEL: func @not_inplace(%{{.*}}: memref<?xf32>) -> memref<?xf32>
func.func @not_inplace(
    %A : tensor<?xf32> {bufferization.writable = false})
  -> tensor<?xf32>
{
  //     CHECK: %[[F0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0 : f32

  //     CHECK: %[[D0:.*]] = memref.dim %[[A]], {{.*}} : memref<?xf32, strided<[?], offset: ?>>
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[D0]]) {alignment = 64 : i64} : memref<?xf32>
  //     CHECK: linalg.fill ins(%[[F0]] : f32) outs(%[[ALLOC]] : memref<?xf32>)
  %r = linalg.fill ins(%f0 : f32) outs(%A : tensor<?xf32>) -> tensor<?xf32>

  // CHECK-NOT: dealloc
  //     CHECK: return %[[ALLOC]] : memref<?xf32>
  return %r: tensor<?xf32>
}

// -----


// CHECK-LABEL: func @not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?x?xf32, strided<[?, ?], offset: ?>>) {
// CHECK-NO-LAYOUT-MAP-LABEL: func @not_inplace(%{{.*}}: memref<?x?xf32>) {
func.func @not_inplace(
    %A : tensor<?x?xf32> {bufferization.writable = true})
  -> tensor<?x?xf32>
{
  %f0 = arith.constant 0.0 : f32

  /// Cross-op multiple uses of %A, the first op which has interfering reads must alloc.
  //       CHECK: %[[ALLOC:.*]] = memref.alloc
  //       CHECK: linalg.fill ins({{.*}}{{.*}}outs(%[[ALLOC]]
  %f = linalg.fill ins(%f0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>

  /// The second op has no interfering reads and can reuse.
  //   CHECK-NOT: alloc
  //       CHECK: linalg.matmul ins(%[[ALLOC]], %[[ALLOC]]{{.*}}) outs(%[[A]]
  %r = linalg.matmul  ins(%f, %f: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @not_inplace
func.func @not_inplace(
    %A : tensor<?x?xf32> {bufferization.writable = true}) -> tensor<?x?xf32> {
  /// Within op multiple uses of %A, must alloc.
  // CHECK: alloc
  %r = linalg.matmul  ins(%A, %A: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%A: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  // CHECK-NOT: dealloc
  return %r: tensor<?x?xf32>
}
// -----

// CHECK-LABEL: func @vec_inplace
func.func @vec_inplace(
    %A : tensor<?xf32> {bufferization.writable = true}, %vec : vector<4xf32>)
  -> tensor<?xf32>
{
  %c0 = arith.constant 0 : index

  // CHECK-NOT: alloc
  %r = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @vec_not_inplace
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>
func.func @vec_not_inplace(
    %A : tensor<?xf32> {bufferization.writable = true}, %vec : vector<4xf32>)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  /// Cross-op multiple uses of %A, the first vector.transfer which has interfering reads must alloc.
  //      CHECK: %[[ALLOC:.*]] = memref.alloc
  //      CHECK: memref.copy {{.*}}, %[[ALLOC]]
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[ALLOC]]
  %r0 = vector.transfer_write %vec, %A[%c0] : vector<4xf32>, tensor<?xf32>

  /// The second vector.transfer has no interfering reads and can reuse the buffer.
  //  CHECK-NOT: alloc
  // CHECK-NEXT: vector.transfer_write {{.*}}, %[[A]]
  %r1 = vector.transfer_write %vec, %A[%c1] : vector<4xf32>, tensor<?xf32>

  //     CHECK: return
  // CHECK-NOT: tensor
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

//      CHECK: func @matmul(
// CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: memref<128x256xf32>
// CHECK-SAME:   %[[B:[0-9a-zA-Z]*]]: memref<256x192xf32>
// CHECK-SAME:   %[[C:[0-9a-zA-Z]*]]: memref<128x192xf32>
func.func @matmul(
    %A: tensor<128x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %B: tensor<256x192xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = false},
    %C: tensor<128x192xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, bufferization.writable = true})
  -> tensor<128x192xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c192 = arith.constant 192 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // Hoisted alloc.
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x192xf32>
  // CHECK: memref.copy %[[C]], %[[ALLOC]]

  // CHECK: scf.for %[[I:.*]] =
  %0 = scf.for %arg3 = %c0 to %c128 step %c8 iter_args(%arg4 = %C) -> (tensor<128x192xf32>) {
    %1 = tensor.extract_slice %A[%arg3, 0] [8, 256] [1, 1] :
      tensor<128x256xf32> to tensor<8x256xf32>

    // CHECK: scf.for %[[J:.*]] =
    %2 = scf.for %arg5 = %c0 to %c192 step %c16 iter_args(%arg6 = %arg4) -> (tensor<128x192xf32>) {
      %3 = tensor.extract_slice %B[0, %arg5] [256, 16] [1, 1] :
        tensor<256x192xf32> to tensor<256x16xf32>

      // C was already replaced with a copy by preprocessing, so no copy is
      // needed here.
      // CHECK: %[[C_SLICE:.*]] = memref.subview %[[ALLOC]]
      %4 = tensor.extract_slice %C[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<128x192xf32> to tensor<8x16xf32>

      // linalg.fill is inplace.
      // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%[[C_SLICE]]
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<8x16xf32>) -> tensor<8x16xf32>

      // CHECK: scf.for %[[K:.*]] =
      %6 = scf.for %arg7 = %c0 to %c256 step %c32 iter_args(%arg8 = %5) -> (tensor<8x16xf32>) {
        %8 = tensor.extract_slice %1[0, %arg7] [8, 32] [1, 1] :
          tensor<8x256xf32> to tensor<8x32xf32>
        %9 = tensor.extract_slice %3[%arg7, 0] [32, 16] [1, 1] :
          tensor<256x16xf32> to tensor<32x16xf32>

        // linalg.matmul is inplace as well as the enclosing scf.for.
        // CHECK: linalg.matmul ins({{.*}} outs(%[[C_SLICE]]
        %10 = linalg.matmul ins(%8, %9 : tensor<8x32xf32>, tensor<32x16xf32>)
                           outs(%arg8 : tensor<8x16xf32>)
          -> tensor<8x16xf32>
        scf.yield %10 : tensor<8x16xf32>
      }

      // insert_slice is inplace but its source comes from an equivalent buffer
      // that is not in place. So we must insert a copy of the small buffer into
      // the bigger buffer.
      // CHECK: %[[T:.*]] = memref.subview %[[C]][%[[I]], %[[J]]] [8, 16] [1, 1]
      // CHECK: memref.copy %[[C_SLICE]], %[[T]]
      %7 = tensor.insert_slice %6 into %arg6[%arg3, %arg5] [8, 16] [1, 1] :
        tensor<8x16xf32> into tensor<128x192xf32>

      scf.yield %7 : tensor<128x192xf32>
    }
    scf.yield %2 : tensor<128x192xf32>
  }

  return %0 : tensor<128x192xf32>
}

// -----

/// This test just checks the produced IR is valid and does not have dominance
/// errors in the def-use chains.

// CHECK-LABEL: func @dominance_violation_bug_1
func.func @dominance_violation_bug_1(
    %A : tensor<?x?xf32> {bufferization.writable = false},
    %idx : index)
  -> tensor<?x?xf32>
{
  %f0 = arith.constant 0.0 : f32

  %sA = tensor.extract_slice %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %ssA = tensor.extract_slice %sA[0, 0][4, 4][1, 1] : tensor<?x?xf32> to tensor<4x4xf32>
  %FA = linalg.fill ins(%f0 : f32) outs(%ssA : tensor<4x4xf32>) -> tensor<4x4xf32>
  %rsA = tensor.insert_slice %FA into %sA[0, 0][4, 4][1, 1] : tensor<4x4xf32> into tensor<?x?xf32>
  %rA = tensor.insert_slice %rsA into %A[0, 0][%idx, %idx][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  return %rA : tensor<?x?xf32>
}

// -----

func.func @gather_like(
    %arg0 : tensor<?x?xf32> {bufferization.writable = false},
    %arg1 : tensor<?xi32> {bufferization.writable = false},
    %arg2 : tensor<?x?xf32> {bufferization.writable = true})
  -> tensor<?x?xf32>
{
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg1 : tensor<?xi32>) outs(%arg2 : tensor<?x?xf32>) {
      ^bb0(%arg3: i32, %arg4 : f32):
        %iv1 = linalg.index 1 : index
        %1 = arith.index_cast %arg3: i32 to index
        %2 = tensor.extract %arg0[%1, %iv1] : tensor<?x?xf32>
        linalg.yield %2 : f32
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @gather_like(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<?x?xf32,
//  CHECK-SAME:     %[[ARG1:.+]]: memref<?xi32
//  CHECK-SAME:     %[[ARG2:.+]]: memref<?x?xf32
//  CHECK-SAME:   ) {
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[ARG1]] :
//  CHECK-SAME:       outs(%[[ARG2]] :
//       CHECK:     %[[YIELD:.+]] = memref.load %[[ARG0]]
//       CHECK:     linalg.yield %[[YIELD]]

// -----

// CHECK-LABEL: func @linalg_op_bufferizes_inplace_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, strided{{.*}}>, %[[t2:.*]]: memref<?xf32, strided{{.*}}>, %[[t3:.*]]: memref<?x?xf32, strided{{.*}}>
func.func @linalg_op_bufferizes_inplace_with_input(
    %t1: tensor<?x?xf32> {bufferization.writable = true},
    %t2: tensor<?xf32> {bufferization.writable = true},
    %t3: tensor<?x?xf32> {bufferization.writable = true},
    %s1: index, %s2: index, %cst: f32)
  -> tensor<?x?xf32>
{
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t3]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%t1, %t2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%t3 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %add = arith.addf %arg0, %arg1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

#accesses = [
  affine_map<(i) -> (i)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"]
}

// CHECK-LABEL: func @op_is_reading_but_following_ops_are_not
//  CHECK-SAME:     %[[t0:.*]]: memref<?xf32
func.func @op_is_reading_but_following_ops_are_not(
    %t0 : tensor<?xf32> {bufferization.writable = false},
    %cst : f32)
  -> tensor<?xf32>
{
  // Make sure that a copy is inserted here.
  // CHECK: %[[ALLOC:.*]] = memref.alloc
  // CHECK: memref.copy %[[t0]], %[[ALLOC]]
  // CHECK: linalg.generic {{.*}} outs(%[[ALLOC]] : memref
  %r0 =linalg.generic #trait outs (%t0 : tensor<?xf32>) {
      ^bb(%0: f32) :
        %a = arith.addf %cst, %0 : f32
        linalg.yield %a : f32
    } -> (tensor<?xf32>)

  // CHECK: linalg.generic {{.*}} outs(%[[ALLOC]] : memref
  %r1 = linalg.generic #trait outs (%r0 : tensor<?xf32>) {
      ^bb(%0: f32) :
        linalg.yield %cst : f32
    } -> (tensor<?xf32>)

  // CHECK: return %[[ALLOC]]
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @map_binary
// CHECK-SAME:  %[[LHS:[0-9a-zA-Z]*]]: memref<64xf32
// CHECK-SAME:  %[[RHS:[0-9a-zA-Z]*]]: memref<64xf32
func.func @map_binary(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>,
                      %init: tensor<64xf32>) -> tensor<64xf32> {
   // CHECK:      linalg.map { arith.addf } ins(%[[LHS]], %[[RHS]] : memref<64xf32
   %add = linalg.map
          ins(%lhs, %rhs: tensor<64xf32>, tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            linalg.yield %0: f32
          }
  func.return %add : tensor<64xf32>
}

// -----

// CHECK-LABEL: func @reduce
// CHECK-SAME:  %[[INPUT:.*]]: memref<16x32x64xf32
func.func @reduce(%input: tensor<16x32x64xf32>,
                  %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
  // CHECK:     linalg.reduce { arith.addf } ins(%[[INPUT]] : memref<16x32x64xf32
  %reduce = linalg.reduce
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in: f32
        linalg.yield %0: f32
      }
  func.return %reduce : tensor<16x64xf32>
}

// -----

// CHECK-LABEL: func @transpose
// CHECK-SAME:  %[[ARG0:.*]]: memref<16x32x64xf32
func.func @transpose(%input: tensor<16x32x64xf32>,
                     %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // CHECK:      linalg.transpose ins(%[[ARG0]] : memref<16x32x64xf32
  %transpose = linalg.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

// CHECK-LABEL: func @broadcast
// CHECK-SAME:  %[[ARG0:.*]]: memref<8x32xf32
func.func @broadcast(%input: tensor<8x32xf32>,
                     %init: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %bcast = linalg.broadcast
      ins(%input:tensor<8x32xf32>)
      outs(%init:tensor<8x16x32xf32>)
      dimensions = [1]
  func.return %bcast : tensor<8x16x32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// AllocTensorOp elimination would produce SSA violations for the example below.
//===----------------------------------------------------------------------===//

func.func @depthwise_conv_1d_nwc_wc(%arg0: index, %arg1: index, %arg2: tensor<8x18x32xf32>)
    -> tensor<?x1x6x8xf32> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %0 = bufferization.alloc_tensor() : tensor<4x1x6x8xf32>
  %1 = tensor.cast %0 : tensor<4x1x6x8xf32> to tensor<?x1x6x8xf32>
  %2 = bufferization.alloc_tensor() : tensor<1x6x8xf32>
  %3 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %1) -> (tensor<?x1x6x8xf32>) {
    %4 = affine.apply affine_map<(d0) -> (d0 ceildiv 8)>(%arg3)
    %5 = tensor.insert_slice %2 into %arg4[%4,0, 0, 0] [1, 1, 6, 8] [1, 1, 1, 1] :
      tensor<1x6x8xf32> into tensor<?x1x6x8xf32>
    scf.yield %5 : tensor<?x1x6x8xf32>
  }
  return %3 : tensor<?x1x6x8xf32>
}

// -----

// CHECK-LABEL: func @do_not_copy_alloc_tensors(
func.func @do_not_copy_alloc_tensors(%f1: f32, %f2: f32, %idx: index)
  -> (tensor<5xf32>, tensor<5xf32>)
{
  // CHECK: memref.alloc
  // CHECK: memref.alloc
  // CHECK-NOT: copy
  // CHECK: memref.store
  // CHECK: memref.store
  %0 = bufferization.alloc_tensor() : tensor<5xf32>
  %1 = tensor.insert %f1 into %0[%idx] : tensor<5xf32>
  %2 = tensor.insert %f2 into %0[%idx] : tensor<5xf32>
  return %1, %2 : tensor<5xf32>, tensor<5xf32>
}
