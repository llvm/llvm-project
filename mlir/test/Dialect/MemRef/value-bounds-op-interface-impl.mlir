// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_alloc(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloc(%sz: index) -> (index, index) {
  %0 = memref.alloc(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_alloca(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloca(%sz: index) -> (index, index) {
  %0 = memref.alloca(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_cast(
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[c10]]
func.func @memref_cast(%m: memref<10xf32>) -> index {
  %0 = memref.cast %m : memref<10xf32> to memref<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_dim(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   return %[[dim]]
func.func @memref_dim(%m: memref<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<?xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_dim_all_positive(
func.func @memref_dim_all_positive(%m: memref<?xf32>, %x: index) {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %x : memref<?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%0, %c0) {cmp = "GE"} : (index, index) -> ()
  return
}

// -----

// CHECK-LABEL: func @memref_expand(
//  CHECK-SAME:     %[[m:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECK-SAME:     %[[sz:[a-zA-Z0-9]+]]: index
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[c4]], %[[sz]]
func.func @memref_expand(%m: memref<?xf32>, %sz: index) -> (index, index) {
  %0 = memref.expand_shape %m [[0, 1]] output_shape [4, %sz]: memref<?xf32> into memref<4x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<4x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<4x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

//       CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-LABEL: func @memref_collapse(
//  CHECK-SAME:     %[[sz0:.*]]: index
//   CHECK-DAG:   %[[c2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[c12:.*]] = arith.constant 12 : index
//       CHECK:   %[[dim:.*]] = memref.dim %{{.*}}, %[[c2]] : memref<3x4x?x2xf32>
//       CHECK:   %[[mul:.*]] = affine.apply #[[$MAP]]()[%[[dim]]]
//       CHECK:   return %[[c12]], %[[mul]]
func.func @memref_collapse(%sz0: index) -> (index, index) {
  %0 = memref.alloc(%sz0) : memref<3x4x?x2xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2, 3]] : memref<3x4x?x2xf32> into memref<12x?xf32>
  %2 = "test.reify_bound"(%1) {dim = 0} : (memref<12x?xf32>) -> (index)
  %3 = "test.reify_bound"(%1) {dim = 1} : (memref<12x?xf32>) -> (index)
  return %2, %3 : index, index
}

// -----

// CHECK-LABEL: func @memref_get_global(
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[c4]]
memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
func.func @memref_get_global() -> index {
  %0 = memref.get_global @gv0 : memref<4xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<4xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_rank(
//  CHECK-SAME:     %[[t:.*]]: memref<5xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @memref_rank(%m: memref<5xf32>) -> index {
  %0 = memref.rank %m : memref<5xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_subview(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @memref_subview(%m: memref<?xf32>, %sz: index) -> index {
  %0 = memref.subview %m[2][%sz][1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 2>>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32, strided<[1], offset: 2>>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_view_dynamic_sizes(
// CHECK-SAME:      %[[raw:.*]]: memref<?xi8>, %[[shift:.*]]: index, %[[sz0:.*]]: index, %[[sz1:.*]]: index) -> (index, index) {
//       CHECK:   %[[view:.*]] = memref.view %[[raw]][%[[shift]]][%[[sz0]], %[[sz1]]]
//       CHECK:   return %[[sz0]], %[[sz1]] : index, index
func.func @memref_view_dynamic_sizes(%raw: memref<?xi8>, %shift: index, %sz0: index, %sz1: index) -> (index, index) {
  %0 = memref.view %raw[%shift][%sz0, %sz1] : memref<?xi8> to memref<?x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<?x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// ViewOp OOB: verify shift_bytes + view_size_bytes == src_size (1D f32: view_size = dim0*4).
// Source 256 bytes, shift 0, view 64xf32 -> 0 + 64*4 == 256.
func.func @memref_view_oob_shift_plus_size_eq_src_1d() {
  %c0 = arith.constant 0 : index
  %raw = memref.alloc() : memref<256xi8>
  %0 = memref.view %raw[%c0][] : memref<256xi8> to memref<64xf32>
  %src_size = memref.dim %raw, %c0 : memref<256xi8>
  %dim0 = memref.dim %0, %c0 : memref<64xf32>
  // expected-remark @below{{true}}
  "test.compare"(%c0, %dim0, %src_size)
      {cmp = "EQ", lhs_map = affine_map<(d0, d1) -> (d0 + d1 * 4)>,
       rhs_map = affine_map<(d0) -> (d0)>}
      : (index, index, index) -> ()
  return
}

// -----

// ViewOp OOB: dynamic shift and dynamic size. Reify returns the size operand.
// Source 256 bytes (alloc), view ?xf32. test.compare checks dim0 >= 0 (OOB + size imply non-negative).
// CHECK-LABEL: func @memref_view_oob_shift_plus_size_eq_src_with_shift(
//  CHECK-SAME:     %[[shift:.*]]: index, %[[n:.*]]: index
//       CHECK:   %[[raw:.*]] = memref.alloc()
//       CHECK:   %[[view:.*]] = memref.view %[[raw]][%[[shift]]][%[[n]]]
//       CHECK:   return %[[n]]
func.func @memref_view_oob_shift_plus_size_eq_src_with_shift(%shift: index, %n: index) -> index {
  %c0 = arith.constant 0 : index
  %raw = memref.alloc() : memref<256xi8>
  %0 = memref.view %raw[%shift][%n] : memref<256xi8> to memref<?xf32>
  %dim0 = memref.dim %0, %c0 : memref<?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c0) {cmp = "GE"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// ViewOp OOB (dynamic shift and sizes): 2D view. Source 1024 bytes, dynamic shift and sizes.
// test.compare checks OOB on both dims: dim0 >= 0, dim1 >= 0. Reify returns size operands.
// CHECK-LABEL: func @memref_view_oob_dynamic_2d_in_bounds(
//  CHECK-SAME:     %[[raw:.*]]: memref<1024xi8>, %[[shift:.*]]: index, %[[sz0:.*]]: index, %[[sz1:.*]]: index
//       CHECK:   %[[view:.*]] = memref.view %[[raw]][%[[shift]]][%[[sz0]], %[[sz1]]]
//       CHECK:   return %[[sz0]], %[[sz1]] : index, index
func.func @memref_view_oob_dynamic_2d_in_bounds(%raw: memref<1024xi8>, %shift: index, %sz0: index, %sz1: index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %raw[%shift][%sz0, %sz1] : memref<1024xi8> to memref<?x?xf32>
  %dim0 = memref.dim %0, %c0 : memref<?x?xf32>
  %dim1 = memref.dim %0, %c1 : memref<?x?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c0) {cmp = "GE"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%dim1, %c0) {cmp = "GE"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<?x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// ViewOp OOB (dynamic): 1D view with dynamic size. Reify returns the size operand.
// test.compare checks that dim 0 is non-negative (OOB + mixed size both imply >= 0).
// CHECK-LABEL: func @memref_view_oob_dynamic_1d(
//  CHECK-SAME:     %[[raw:.*]]: memref<?xi8>, %[[shift:.*]]: index, %[[n:.*]]: index
//       CHECK:   %[[view:.*]] = memref.view %[[raw]][%[[shift]]][%[[n]]]
//       CHECK:   return %[[n]]
func.func @memref_view_oob_dynamic_1d(%raw: memref<?xi8>, %shift: index, %n: index) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.view %raw[%shift][%n] : memref<?xi8> to memref<?xf32>
  %dim0 = memref.dim %0, %c0 : memref<?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c0) {cmp = "GE"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// ViewOp OOB (dynamic): 2D view with dynamic sizes. Reify returns size operands.
// test.compare checks OOB on both dims: dim0 >= 0, dim1 >= 0.
// CHECK-LABEL: func @memref_view_oob_dynamic_2d(
//       CHECK:   %[[view:.*]] = memref.view
//       CHECK:   return {{.*}}, {{.*}} : index, index
func.func @memref_view_oob_dynamic_2d(%raw: memref<?xi8>, %shift: index, %a: index, %b: index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.view %raw[%shift][%a, %b] : memref<?xi8> to memref<?x?xf32>
  %dim0 = memref.dim %0, %c0 : memref<?x?xf32>
  %dim1 = memref.dim %0, %c1 : memref<?x?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c0) {cmp = "GE"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%dim1, %c0) {cmp = "GE"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<?x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// ViewOp OOB (dynamic sizes, constant source): alloc 256 bytes, view with dynamic size %n.
// OOB constraint forces dim0 == (256-0)/4 == 64; test.compare proves dim0 == 64.
// CHECK-LABEL: func @memref_view_oob_dynamic_1d_constant_source(
//       CHECK:   %[[c64:.*]] = arith.constant 64 : index
//       CHECK:   return %[[c64]]
func.func @memref_view_oob_dynamic_1d_constant_source(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %raw = memref.alloc() : memref<256xi8>
  %0 = memref.view %raw[%c0][%n] : memref<256xi8> to memref<?xf32>
  %dim0 = memref.dim %0, %c0 : memref<?xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c64) {cmp = "EQ"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// ViewOp OOB (2D, one dynamic dim, constant source): alloc 256 bytes, view with dynamic dim0 %n and static dim1=4.
// OOB constraint forces dim0 == (256-0)/(4*4) == 16; test.compare proves dim0 == 16, dim1 == 4.
// CHECK-LABEL: func @memref_view_oob_dynamic_2d_constant_source(
//       CHECK:   arith.constant 16 : index
//       CHECK:   arith.constant 4 : index
//       CHECK:   return {{.*}}, {{.*}} : index, index
func.func @memref_view_oob_dynamic_2d_constant_source(%n: index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index
  %raw = memref.alloc() : memref<256xi8>
  %0 = memref.view %raw[%c0][%n] : memref<256xi8> to memref<?x4xf32>
  %dim0 = memref.dim %0, %c0 : memref<?x4xf32>
  %dim1 = memref.dim %0, %c1 : memref<?x4xf32>
  // expected-remark @below{{true}}
  "test.compare"(%dim0, %c16) {cmp = "EQ"} : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%dim1, %c4) {cmp = "EQ"} : (index, index) -> ()
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?x4xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<?x4xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// ViewOp value bound failure: constant bound requested but view dim is dynamic (from block arg).
// Only constraint is bound == %size; no constant available.
func.func @memref_view_reify_constant_fails(%raw: memref<?xi8>, %shift: index, %size: index) -> index {
  %0 = memref.view %raw[%shift][%size] : memref<?xi8> to memref<?xf32>
  // expected-error @below {{could not reify bound}}
  %1 = "test.reify_bound"(%0) {dim = 0, constant} : (memref<?xf32>) -> (index)
  return %1 : index
}
