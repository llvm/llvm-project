// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=default" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=CHECK,DEFAULT --enable-var-scope
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-outer" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=CHECK,OUTER --enable-var-scope
// RUN: mlir-opt %s --split-input-file --sparse-reinterpret-map="loop-ordering-strategy=loop-use-inner" \
// RUN: --sparsification --canonicalize | FileCheck %s --check-prefixes=CHECK,INNER --enable-var-scope

// A[i] * B[j] * C[j]: i=1, j=2.
// CHECK-LABEL: func.func @intersection(
// CHECK: %[[A:.*]] = sparse_tensor.values %arg0
// DEFAULT-NOT: scf.for
// DEFAULT: scf.while
// DEFAULT: scf.for %[[I:[^ ]+]] =
// DEFAULT: memref.load %[[A]][%[[I]]]
// OUTER-NOT: scf.for
// OUTER: scf.while
// OUTER: scf.for %[[I:[^ ]+]] =
// OUTER: memref.load %[[A]][%[[I]]]
// INNER-NOT: scf.while
// INNER: scf.for %[[I:[^ ]+]] =
// INNER: memref.load %[[A]][%[[I]]]
// INNER: scf.while

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
// CHECK: %[[B:.*]] = sparse_tensor.values %arg1
// DEFAULT: scf.for %[[J:[^ ]+]] =
// DEFAULT: tensor.extract %arg2[%[[J]]]
// DEFAULT: scf.for %[[I:[^ ]+]] =
// DEFAULT: memref.load %[[B]][%[[I]]]
// OUTER: scf.for %[[I:[^ ]+]] =
// OUTER: memref.load %[[B]][%[[I]]]
// OUTER: scf.for %[[J:[^ ]+]] =
// OUTER: tensor.extract %arg2[%[[J]]]
// INNER: scf.for %[[J:[^ ]+]] =
// INNER: tensor.extract %arg2[%[[J]]]
// INNER: scf.for %[[I:[^ ]+]] =
// INNER: memref.load %[[B]][%[[I]]]

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
