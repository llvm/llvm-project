// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=default" \
// RUN: --mlir-print-local-scope | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-outer" \
// RUN: --mlir-print-local-scope | FileCheck %s --check-prefixes=CHECK,OUTER
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-inner" \
// RUN: --mlir-print-local-scope | FileCheck %s --check-prefixes=CHECK,INNER
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=default" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=LOWER,DEFAULT-LOWER --enable-var-scope
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-outer" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=LOWER,OUTER-LOWER --enable-var-scope
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-inner" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=LOWER,INNER-LOWER --enable-var-scope

// A[i] * B[j] * C[j]: i=1, j=2.
// CHECK-LABEL: func.func @intersection(
// CHECK: linalg.generic
// DEFAULT-SAME: indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>]
// OUTER-SAME: indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>]
// INNER-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> ()>]
// CHECK-SAME: sorted = true

// LOWER-LABEL: func.func @intersection(
// LOWER: %[[A:.*]] = sparse_tensor.values %arg0
// DEFAULT-LOWER-NOT: scf.for
// DEFAULT-LOWER: scf.while
// DEFAULT-LOWER: scf.for %[[I:[^ ]+]] =
// DEFAULT-LOWER: memref.load %[[A]][%[[I]]]
// OUTER-LOWER-NOT: scf.for
// OUTER-LOWER: scf.while
// OUTER-LOWER: scf.for %[[I:[^ ]+]] =
// OUTER-LOWER: memref.load %[[A]][%[[I]]]
// INNER-LOWER-NOT: scf.while
// INNER-LOWER: scf.for %[[I:[^ ]+]] =
// INNER-LOWER: memref.load %[[A]][%[[I]]]
// INNER-LOWER: scf.while

#S = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
func.func @intersection(%a: tensor<64xi64, #S>, %b: tensor<64xi64, #S>, %c: tensor<64xi64, #S>, %init: tensor<i64>) -> tensor<i64> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i)>, affine_map<(i,j)->(j)>, affine_map<(i,j)->(j)>, affine_map<(i,j)->()>],
    iterator_types = ["reduction", "reduction"]
  } ins(%a, %b, %c : tensor<64xi64, #S>, tensor<64xi64, #S>, tensor<64xi64, #S>) outs(%init : tensor<i64>) {
    ^bb0(%x: i64, %y: i64, %z: i64, %acc: i64):
      %p = arith.muli %x, %y : i64
      %q = arith.muli %p, %z : i64
      %s = arith.addi %q, %acc : i64
      linalg.yield %s : i64
  } -> tensor<i64>
  return %r : tensor<i64>
}

// -----

// A[2*i+1] * B[i] * C[j]: i=2, j=1
// CHECK-LABEL: func.func @scaled(
// CHECK: linalg.generic
// DEFAULT-SAME: indexing_maps = [affine_map<(d0, d1) -> (d1 * 2 + 1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>]
// OUTER-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0 * 2 + 1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> ()>]
// INNER-SAME: indexing_maps = [affine_map<(d0, d1) -> (d1 * 2 + 1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> ()>]
// CHECK-SAME: sorted = true

// LOWER-LABEL: func.func @scaled(
// LOWER-DAG: %[[C0:.*]] = arith.constant 0 : index
// LOWER-DAG: %[[C8:.*]] = arith.constant 8 : index
// LOWER-DAG: %[[B:.*]] = sparse_tensor.values %arg1
// DEFAULT-LOWER: scf.for %{{[^ ]+}} = %[[C0]] to %[[C8]]
// DEFAULT-LOWER: scf.for %[[I:[^ ]+]] =
// DEFAULT-LOWER: memref.load %[[B]][%[[I]]]
// OUTER-LOWER: scf.for %[[I:[^ ]+]] =
// OUTER-LOWER: memref.load %[[B]][%[[I]]]
// OUTER-LOWER: scf.for %{{[^ ]+}} = %[[C0]] to %[[C8]]
// INNER-LOWER: scf.for %{{[^ ]+}} = %[[C0]] to %[[C8]]
// INNER-LOWER: scf.for %[[I:[^ ]+]] =
// INNER-LOWER: memref.load %[[B]][%[[I]]]

#S = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>
func.func @scaled(%a: tensor<16xi64>, %b: tensor<8xi64, #S>, %c: tensor<8xi64>, %init: tensor<i64>) -> tensor<i64> {
  %r = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(2*i+1)>, affine_map<(i,j)->(i)>, affine_map<(i,j)->(j)>, affine_map<(i,j)->()>],
    iterator_types = ["reduction", "reduction"]
  } ins(%a, %b, %c : tensor<16xi64>, tensor<8xi64, #S>, tensor<8xi64>) outs(%init : tensor<i64>) {
    ^bb0(%x: i64, %y: i64, %z: i64, %acc: i64):
      %p = arith.muli %x, %y : i64
      %q = arith.muli %p, %z : i64
      %s = arith.addi %q, %acc : i64
      linalg.yield %s : i64
  } -> tensor<i64>
  return %r : tensor<i64>
}
