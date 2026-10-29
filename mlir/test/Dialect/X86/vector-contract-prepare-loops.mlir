// RUN: mlir-opt %s -transform-interpreter | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter=entry-point=__transform_nano | FileCheck %s --check-prefix=NANO

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

// CHECK-DAG: #[[$VNNI_LHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$VNNI_RHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
// CHECK-DAG: #[[$VNNI_ACC:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// Fold deinterleave/transpose/interleave/transpose unpacking of non-square
// operands with strided memrefs, aligned nonzero inner read offsets, and
// fastmath flags.
// CHECK-LABEL: func.func @interleaved_vnni_bf16(
// CHECK-SAME: %[[A:[^:]+]]: memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>, %[[B:[^:]+]]: memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>
// NANO-LABEL: func.func @interleaved_vnni_bf16(
// NANO: vector.contract {{.*}} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// NANO: return
// CHECK-DAG: %[[ONE:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[AP:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x?x16x32x2xbf16, strided<[?, 1024, 64, 2, 1]>>
// CHECK-DAG: %[[BP:.+]] = memref.expand_shape %[[B]] {{.*}} into memref<?x?x16x32x2xbf16, strided<[?, 1024, 64, 2, 1]>>
// CHECK: scf.for %[[K:[^ ]+]] =
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[AP]][%{{.+}}, %[[K]], %{{.+}}, %[[ONE]], %{{.+}}]{{.*}} vector<16x16x2xbf16>
// CHECK-NEXT: %[[RHS:.+]] = vector.transfer_read %[[BP]][%{{.+}}, %[[K]], %{{.+}}, %[[ONE]], %{{.+}}]{{.*}} vector<16x16x2xbf16>
// CHECK-NEXT: %[[DOT:.+]] = vector.contract {fastmath = #arith.fastmath<reassoc>, indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// CHECK-NEXT: scf.yield %[[DOT]]
func.func @interleaved_vnni_bf16(%a: memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>,
                                %b: memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>,
                                %init: vector<16x16xf32>, %lb: index, %ub: index,
                                %m: index, %n: index) -> vector<16x16xf32> {
  %zero = arith.constant 0 : index
  %offset = arith.constant 2 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x16xf32>) {
    %lhs = vector.transfer_read %a[%m, %k, %zero, %offset], %pad {in_bounds = [true, true]} : memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%n, %k, %zero, %offset], %pad {in_bounds = [true, true]} : memref<?x?x16x64xbf16, strided<[?, 1024, 64, 1]>>, vector<16x32xbf16>
    %even, %odd = vector.deinterleave %rhs : vector<16x32xbf16> -> vector<16x16xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x16xbf16> to vector<16x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x16xbf16> to vector<16x16xbf16>
    %interleaved = vector.interleave %even_t, %odd_t : vector<16x16xbf16> -> vector<16x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>, fastmath = #arith.fastmath<reassoc>} %lhs, %flat, %accum : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
    scf.yield %dot : vector<16x16xf32>
  }
  return %result : vector<16x16xf32>
}

// The even and odd elements must come from the same packed source.
// CHECK-LABEL: func.func @negative_interleaved_different_sources(
// CHECK: vector.contract {{.*}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_interleaved_different_sources(%a: memref<?x32x32xbf16>, %b: memref<?x16x64xbf16>,
                                                 %c: memref<?x16x64xbf16>, %init: vector<32x32xf32>,
                                                 %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %r1 = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %rhs_c = vector.transfer_read %c[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %even, %unused = vector.deinterleave %rhs : vector<16x64xbf16> -> vector<16x32xbf16>
    %unused_c, %odd = vector.deinterleave %rhs_c : vector<16x64xbf16> -> vector<16x32xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %interleaved = vector.interleave %even_t, %odd_t : vector<32x16xbf16> -> vector<32x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %r1 : vector<32x32xf32>
}

// Fold transpose/extract and slice-insert unpacking despite interspersed unit
// dimensions and odd-first insertion.
// CHECK-LABEL: func.func @split_join_bf16(
// CHECK-DAG: %[[AP:.+]] = memref.expand_shape {{.*}} into memref<?x?x?x16x2xbf16, strided<[?, ?, ?, 2, 1], offset: ?>>
// CHECK-DAG: %[[BP:.+]] = memref.expand_shape {{.*}} into memref<?x?x?x32x2xbf16, strided<[?, ?, ?, 2, 1], offset: ?>>
// CHECK: scf.for %[[IV:[^ ]+]] =
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[AP]][%{{.+}}, %[[IV]], {{.*}} vector<32x16x2xbf16>
// CHECK-NEXT: %[[RHS:.+]] = vector.transfer_read %[[BP]][%{{.+}}, %[[IV]], {{.*}} vector<16x32x2xbf16>
// CHECK-NEXT: vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
// CHECK-NEXT: scf.yield
// NANO-LABEL: func.func @split_join_bf16(
// NANO-NOT: vector.contract
// NANO-COUNT-4: x86.amx.tile_mulf
// NANO-NOT: vector.contract
// NANO-NOT: x86.amx.tile_mulf
// NANO: return
func.func @split_join_bf16(%a: memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>>,
                                  %b: memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>>,
                                  %ub: index, %m: index, %n: index) -> vector<32x32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %poison = ub.poison : vector<32x16x2xbf16>
  %pad = arith.constant 0.0 : bf16
  %init = arith.constant dense<0.0> : vector<32x32xf32>
  %lb = arith.constant 0 : index
  %result = scf.for %iv = %lb to %ub step %c1 iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%m, %iv, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%n, %iv, %c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?x?x?xbf16, strided<[?, ?, ?, 1], offset: ?>>, vector<16x64xbf16>
    %rhs4 = vector.broadcast %rhs : vector<16x64xbf16> to vector<1x1x16x64xbf16>
    %packed = vector.shape_cast %rhs4 : vector<1x1x16x64xbf16> to vector<1x16x1x32x2x1xbf16>
    %even_odd = vector.transpose %packed, [0, 2, 4, 1, 3, 5] : vector<1x16x1x32x2x1xbf16> to vector<1x1x2x16x32x1xbf16>
    %even3 = vector.extract %even_odd[0, 0, 0] : vector<16x32x1xbf16> from vector<1x1x2x16x32x1xbf16>
    %odd3 = vector.extract %even_odd[0, 0, 1] : vector<16x32x1xbf16> from vector<1x1x2x16x32x1xbf16>
    %even = vector.shape_cast %even3 : vector<16x32x1xbf16> to vector<16x32xbf16>
    %odd = vector.shape_cast %odd3 : vector<16x32x1xbf16> to vector<16x32xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %even_t3 = vector.shape_cast %even_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %odd_t3 = vector.shape_cast %odd_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %join0 = vector.insert_strided_slice %odd_t3, %poison offsets = [0, 0, 1], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %join = vector.insert_strided_slice %even_t3, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %interleaved = vector.shape_cast %join : vector<32x16x2xbf16> to vector<32x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// Reordering dimensions other than the VNNI dimension changes element order.
// CHECK-LABEL: func.func @negative_split_join_reordered(
// CHECK: vector.contract {{.*}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_split_join_reordered(%a: memref<?x32x32xbf16>, %b: memref<?x16x64xbf16>,
                                        %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %poison = ub.poison : vector<32x16x2xbf16>
  %r0 = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %packed = vector.shape_cast %rhs : vector<16x64xbf16> to vector<16x32x2xbf16>
    %even_odd = vector.transpose %packed, [2, 1, 0] : vector<16x32x2xbf16> to vector<2x32x16xbf16>
    %even_r = vector.extract %even_odd[0] : vector<32x16xbf16> from vector<2x32x16xbf16>
    %odd_r = vector.extract %even_odd[1] : vector<32x16xbf16> from vector<2x32x16xbf16>
    %even = vector.shape_cast %even_r : vector<32x16xbf16> to vector<16x32xbf16>
    %odd = vector.shape_cast %odd_r : vector<32x16xbf16> to vector<16x32xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %even3 = vector.shape_cast %even_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %odd3 = vector.shape_cast %odd_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %join0 = vector.insert_strided_slice %even3, %poison offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %join = vector.insert_strided_slice %odd3, %join0 offsets = [0, 0, 1], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %interleaved = vector.shape_cast %join : vector<32x16x2xbf16> to vector<32x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %r0 : vector<32x32xf32>
}

// A single insert leaves the odd elements unspecified.
// CHECK-LABEL: func.func @negative_split_join_incomplete(
// CHECK: vector.contract {{.*}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_split_join_incomplete(%a: memref<?x32x32xbf16>, %b: memref<?x16x64xbf16>,
                                         %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %poison = ub.poison : vector<32x16x2xbf16>
  %r1 = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %packed = vector.shape_cast %rhs : vector<16x64xbf16> to vector<16x32x2xbf16>
    %even_odd = vector.transpose %packed, [2, 0, 1] : vector<16x32x2xbf16> to vector<2x16x32xbf16>
    %even = vector.extract %even_odd[0] : vector<16x32xbf16> from vector<2x16x32xbf16>
    %odd = vector.extract %even_odd[1] : vector<16x32xbf16> from vector<2x16x32xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %even3 = vector.shape_cast %even_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %odd3 = vector.shape_cast %odd_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %join = vector.insert_strided_slice %even3, %poison offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %interleaved = vector.shape_cast %join : vector<32x16x2xbf16> to vector<32x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %r1 : vector<32x32xf32>
}

// Two inserts at the same offset still leave the odd positions unspecified.
// CHECK-LABEL: func.func @negative_split_join_duplicate(
// CHECK: vector.contract {{.*}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_split_join_duplicate(%a: memref<?x32x32xbf16>, %b: memref<?x16x64xbf16>,
                                        %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0.0 : bf16
  %poison = ub.poison : vector<32x16x2xbf16>
  %r2 = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %packed = vector.shape_cast %rhs : vector<16x64xbf16> to vector<16x32x2xbf16>
    %even_odd = vector.transpose %packed, [2, 0, 1] : vector<16x32x2xbf16> to vector<2x16x32xbf16>
    %even = vector.extract %even_odd[0] : vector<16x32xbf16> from vector<2x16x32xbf16>
    %odd = vector.extract %even_odd[1] : vector<16x32xbf16> from vector<2x16x32xbf16>
    %even_t = vector.transpose %even, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %odd_t = vector.transpose %odd, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16>
    %even3 = vector.shape_cast %even_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %odd3 = vector.shape_cast %odd_t : vector<32x16xbf16> to vector<32x16x1xbf16>
    %join0 = vector.insert_strided_slice %even3, %poison offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %join = vector.insert_strided_slice %odd3, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xbf16> into vector<32x16x2xbf16>
    %interleaved = vector.shape_cast %join : vector<32x16x2xbf16> to vector<32x32xbf16>
    %flat = vector.transpose %interleaved, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %r2 : vector<32x32xf32>
}

// fold VNNI unpacking of packed i8 rhs read into contraction
// CHECK-LABEL: func.func @direct_vnni_int8(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>, %[[B:[^,]+]]: memref<?x16x32x4xi8>
// NANO-LABEL: func.func @direct_vnni_int8(
// NANO-NOT: vector.contract
// NANO-COUNT-4: x86.amx.tile_muli
// NANO-NOT: vector.contract
// NANO-NOT: x86.amx.tile_muli
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x4xi8>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x4xi8>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x4xi8>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
func.func @direct_vnni_int8(%a: memref<?x32x64xi8>,
                            %b: memref<?x16x32x4xi8>,
                            %init: vector<32x32xi32>, %ub: index,
                            %m: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x4xi8>, vector<16x32x4xi8>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// Fold transpose/shape_cast unpacking of a packed RHS read through a leading
// unit-dimension broadcast, and expand the LHS through a static view of its
// dynamic innermost dimension.
// CHECK-LABEL: func.func @packed_vnni_bf16(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x?xbf16>, %[[B:[^,]+]]: memref<?x16x32x2xbf16>
// NANO-LABEL: func.func @packed_vnni_bf16(
// NANO-NOT: vector.contract
// NANO-COUNT-4: x86.amx.tile_mulf
// NANO-NOT: vector.contract
// NANO-NOT: x86.amx.tile_mulf
// NANO: return
// CHECK: %[[VIEW:.+]] = memref.reinterpret_cast %[[A]] {{.*}} sizes: [%{{.+}}, 32, 32]{{.*}} to memref<?x32x32xbf16, strided<[?, ?, 1]>>
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[VIEW]] {{.*}} into memref<?x32x16x2xbf16, strided<[?, ?, 2, 1]>>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK-NEXT: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK-NEXT: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
// CHECK-NEXT: scf.yield
func.func @packed_vnni_bf16(%a: memref<?x32x?xbf16>,
                            %b: memref<?x16x32x2xbf16>,
                            %init: vector<32x32xf32>, %ub: index,
                            %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x?xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %broadcast = vector.broadcast %rhs : vector<16x32x2xbf16> to vector<1x16x32x2xbf16>
    %transpose = vector.transpose %broadcast, [0, 1, 3, 2] : vector<1x16x32x2xbf16> to vector<1x16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<1x16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: broadcasting creates K, so the source is not VNNI packed -> skip
// CHECK-LABEL: func.func @negative_broadcast_unpacked_rhs(
// CHECK-NOT: memref.expand_shape
// CHECK: %[[RHS:.+]] = vector.transfer_read {{.*}} vector<32x2xbf16>
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[RHS]] : vector<32x2xbf16> to vector<16x32x2xbf16>
// CHECK: %[[T:.+]] = vector.transpose %[[BCAST]], [0, 2, 1]
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %{{.+}}, %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_broadcast_unpacked_rhs(%a: memref<?x32x32xbf16>,
                                          %b: memref<?x32x2xbf16>,
                                          %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x2xbf16>, vector<32x2xbf16>
    %broadcast = vector.broadcast %rhs : vector<32x2xbf16> to vector<16x32x2xbf16>
    %transpose = vector.transpose %broadcast, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost read offset 1 is not a multiple of VNNI factor 2 -> skip
// CHECK-LABEL: func.func @negative_misaligned_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_misaligned_inner_k(%a: memref<?x32x64xbf16>,
                              %b: memref<?x16x32x2xbf16>,
                              %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %one], %pad {in_bounds = [true, true]} : memref<?x32x64xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost dim 33 is not divisible by VNNI factor 2 -> skip
// CHECK-LABEL: func.func @negative_nondivisible_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x33xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_nondivisible_inner_k(%a: memref<?x32x33xbf16>,
                                %b: memref<?x16x32x2xbf16>,
                                %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x33xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: scalable lhs cannot be packed into a fixed VNNI shape -> skip
// CHECK-LABEL: func.func @negative_scalable_vnni(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?x32xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<[32]x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<[32]x32xbf16>, vector<32x32xbf16> into vector<[32]x32xf32>
func.func @negative_scalable_vnni(%a: memref<?x?x32xbf16>,
                         %b: memref<?x16x32x2xbf16>,
                         %init: vector<[32]x32xf32>, %ub: index,
                         %m: index) -> vector<[32]x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<[32]x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x?x32xbf16>, vector<[32]x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<[32]x32xbf16>, vector<32x32xbf16> into vector<[32]x32xf32>
    scf.yield %dot : vector<[32]x32xf32>
  }
  return %result : vector<[32]x32xf32>
}

// negative test: lhs memref with vector elements cannot be expanded into VNNI layout -> skip
// CHECK-LABEL: func.func @negative_vector_element_lhs(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32xvector<32xbf16>>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @negative_vector_element_lhs(%a: memref<?x32xvector<32xbf16>>,
                              %b: memref<?x16x32x2xbf16>,
                              %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %vpad = arith.constant dense<0.0> : vector<32xbf16>
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero], %vpad {in_bounds = [true]} : memref<?x32xvector<32xbf16>>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// Fold i8 unpacking through two levels of binary splits and joins. The first
// split yields VNNI lanes {0, 2} and {1, 3}; mix deinterleave with
// transpose/extract, and interleave with out-of-order slice inserts.
// CHECK-LABEL: func.func @mixed_split_join_int8(
// CHECK-DAG: %[[AP:.+]] = memref.expand_shape {{.*}} into memref<?x32x16x4xi8>
// CHECK-DAG: %[[BP:.+]] = memref.expand_shape {{.*}} into memref<?x16x32x4xi8>
// CHECK: scf.for
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[AP]][{{.*}} vector<32x16x4xi8>
// CHECK-NEXT: %[[RHS:.+]] = vector.transfer_read %[[BP]][{{.*}} vector<16x32x4xi8>
// CHECK-NEXT: vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
// CHECK-NEXT: scf.yield
// NANO-LABEL: func.func @mixed_split_join_int8(
// NANO-NOT: vector.contract
// NANO-COUNT-4: x86.amx.tile_muli
// NANO-NOT: vector.contract
// NANO-NOT: x86.amx.tile_muli
// NANO: return
func.func @mixed_split_join_int8(%a: memref<?x32x64xi8>, %b: memref<?x16x128xi8>,
                                %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %poison = ub.poison : vector<32x16x2xi8>
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %even, %odd = vector.deinterleave %rhs : vector<16x128xi8> -> vector<16x64xi8>
    %lane0, %lane2 = vector.deinterleave %even : vector<16x64xi8> -> vector<16x32xi8>
    %odd_packed = vector.shape_cast %odd : vector<16x64xi8> to vector<16x32x2xi8>
    %odd_split = vector.transpose %odd_packed, [2, 0, 1] : vector<16x32x2xi8> to vector<2x16x32xi8>
    %lane1 = vector.extract %odd_split[0] : vector<16x32xi8> from vector<2x16x32xi8>
    %lane3 = vector.extract %odd_split[1] : vector<16x32xi8> from vector<2x16x32xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %even_t = vector.interleave %t0, %t2 : vector<32x16xi8> -> vector<32x32xi8>
    %s1 = vector.shape_cast %t1 : vector<32x16xi8> to vector<32x16x1xi8>
    %s3 = vector.shape_cast %t3 : vector<32x16xi8> to vector<32x16x1xi8>
    %join0 = vector.insert_strided_slice %s3, %poison offsets = [0, 0, 1], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x2xi8>
    %join1 = vector.insert_strided_slice %s1, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x2xi8>
    %odd_t = vector.shape_cast %join1 : vector<32x16x2xi8> to vector<32x32xi8>
    %joined = vector.interleave %even_t, %odd_t : vector<32x32xi8> -> vector<32x64xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// Joining adjacent VNNI lanes at the inner level swaps VNNI lanes 1 and 2.
// CHECK-LABEL: func.func @negative_int8_lane_order(
// CHECK-NOT: memref.expand_shape
// CHECK: vector.contract {{.*}} : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
func.func @negative_int8_lane_order(%a: memref<?x32x64xi8>, %b: memref<?x16x128xi8>,
                                  %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %even, %odd = vector.deinterleave %rhs : vector<16x128xi8> -> vector<16x64xi8>
    %lane0, %lane2 = vector.deinterleave %even : vector<16x64xi8> -> vector<16x32xi8>
    %lane1, %lane3 = vector.deinterleave %odd : vector<16x64xi8> -> vector<16x32xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %join0 = vector.interleave %t0, %t1 : vector<32x16xi8> -> vector<32x32xi8>
    %join1 = vector.interleave %t2, %t3 : vector<32x16xi8> -> vector<32x32xi8>
    %joined = vector.interleave %join0, %join1 : vector<32x32xi8> -> vector<32x64xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// Fold i8 unpacking through a single 4-way transpose/extract split and a
// 4-way slice-insert join with out-of-order inserts.
// CHECK-LABEL: func.func @fourway_split_join_int8(
// CHECK-DAG: %[[AP:.+]] = memref.expand_shape {{.*}} into memref<?x32x16x4xi8>
// CHECK-DAG: %[[BP:.+]] = memref.expand_shape {{.*}} into memref<?x16x32x4xi8>
// CHECK: scf.for
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[AP]][{{.*}} vector<32x16x4xi8>
// CHECK-NEXT: %[[RHS:.+]] = vector.transfer_read %[[BP]][{{.*}} vector<16x32x4xi8>
// CHECK-NEXT: vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
// CHECK-NEXT: scf.yield
// NANO-LABEL: func.func @fourway_split_join_int8(
// NANO-NOT: vector.contract
// NANO-COUNT-4: x86.amx.tile_muli
// NANO-NOT: vector.contract
// NANO-NOT: x86.amx.tile_muli
// NANO: return
func.func @fourway_split_join_int8(%a: memref<?x32x64xi8>, %b: memref<?x16x128xi8>,
                                   %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %poison = ub.poison : vector<32x16x4xi8>
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %packed = vector.shape_cast %rhs : vector<16x128xi8> to vector<16x32x4xi8>
    %split = vector.transpose %packed, [2, 0, 1] : vector<16x32x4xi8> to vector<4x16x32xi8>
    %lane0 = vector.extract %split[0] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane1 = vector.extract %split[1] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane2 = vector.extract %split[2] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane3 = vector.extract %split[3] : vector<16x32xi8> from vector<4x16x32xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %s0 = vector.shape_cast %t0 : vector<32x16xi8> to vector<32x16x1xi8>
    %s1 = vector.shape_cast %t1 : vector<32x16xi8> to vector<32x16x1xi8>
    %s2 = vector.shape_cast %t2 : vector<32x16xi8> to vector<32x16x1xi8>
    %s3 = vector.shape_cast %t3 : vector<32x16xi8> to vector<32x16x1xi8>
    %join0 = vector.insert_strided_slice %s2, %poison offsets = [0, 0, 2], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join1 = vector.insert_strided_slice %s0, %join0 offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join2 = vector.insert_strided_slice %s3, %join1 offsets = [0, 0, 3], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join3 = vector.insert_strided_slice %s1, %join2 offsets = [0, 0, 1], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %joined = vector.shape_cast %join3 : vector<32x16x4xi8> to vector<32x64xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// Fold a 4-way split combined with two levels of binary joins.
// CHECK-LABEL: func.func @fourway_split_binary_join_int8(
// CHECK-DAG: memref.expand_shape {{.*}} into memref<?x32x16x4xi8>
// CHECK-DAG: memref.expand_shape {{.*}} into memref<?x16x32x4xi8>
// CHECK: vector.contract {{.*}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
func.func @fourway_split_binary_join_int8(%a: memref<?x32x64xi8>, %b: memref<?x16x128xi8>,
                                          %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %packed = vector.shape_cast %rhs : vector<16x128xi8> to vector<16x32x4xi8>
    %split = vector.transpose %packed, [2, 0, 1] : vector<16x32x4xi8> to vector<4x16x32xi8>
    %lane0 = vector.extract %split[0] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane1 = vector.extract %split[1] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane2 = vector.extract %split[2] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane3 = vector.extract %split[3] : vector<16x32xi8> from vector<4x16x32xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %even_t = vector.interleave %t0, %t2 : vector<32x16xi8> -> vector<32x32xi8>
    %odd_t = vector.interleave %t1, %t3 : vector<32x16xi8> -> vector<32x32xi8>
    %joined = vector.interleave %even_t, %odd_t : vector<32x32xi8> -> vector<32x64xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// Inserting lanes 1 and 2 at each other's offsets swaps VNNI lanes.
// CHECK-LABEL: func.func @negative_fourway_join_lane_order(
// CHECK-NOT: memref.expand_shape
// CHECK: vector.contract {{.*}} : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
func.func @negative_fourway_join_lane_order(%a: memref<?x32x64xi8>, %b: memref<?x16x128xi8>,
                                            %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 1 : index
  %pad = arith.constant 0 : i8
  %poison = ub.poison : vector<32x16x4xi8>
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %packed = vector.shape_cast %rhs : vector<16x128xi8> to vector<16x32x4xi8>
    %split = vector.transpose %packed, [2, 0, 1] : vector<16x32x4xi8> to vector<4x16x32xi8>
    %lane0 = vector.extract %split[0] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane1 = vector.extract %split[1] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane2 = vector.extract %split[2] : vector<16x32xi8> from vector<4x16x32xi8>
    %lane3 = vector.extract %split[3] : vector<16x32xi8> from vector<4x16x32xi8>
    %t0 = vector.transpose %lane0, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t1 = vector.transpose %lane1, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t2 = vector.transpose %lane2, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %t3 = vector.transpose %lane3, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
    %s0 = vector.shape_cast %t0 : vector<32x16xi8> to vector<32x16x1xi8>
    %s1 = vector.shape_cast %t1 : vector<32x16xi8> to vector<32x16x1xi8>
    %s2 = vector.shape_cast %t2 : vector<32x16xi8> to vector<32x16x1xi8>
    %s3 = vector.shape_cast %t3 : vector<32x16xi8> to vector<32x16x1xi8>
    %join0 = vector.insert_strided_slice %s0, %poison offsets = [0, 0, 0], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join1 = vector.insert_strided_slice %s2, %join0 offsets = [0, 0, 1], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join2 = vector.insert_strided_slice %s1, %join1 offsets = [0, 0, 2], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %join3 = vector.insert_strided_slice %s3, %join2 offsets = [0, 0, 3], strides = [1, 1, 1] : vector<32x16x1xi8> into vector<32x16x4xi8>
    %joined = vector.shape_cast %join3 : vector<32x16x4xi8> to vector<32x64xi8>
    %flat = vector.transpose %joined, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.apply_dce to %funcs : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.apply_dce to %funcs : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll_for_amx
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}
