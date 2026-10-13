// RUN: mlir-opt %s -transform-interpreter | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter=entry-point=__transform_nano | FileCheck %s --check-prefix=NANO

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

// CHECK-DAG: #[[$VNNI_LHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$VNNI_RHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
// CHECK-DAG: #[[$VNNI_ACC:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// fold VNNI unpacking of packed bf16 rhs read into contraction, keeping fastmath
// CHECK-LABEL: func.func @direct_vnni_bf16(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x16x32x2xbf16>
// NANO-LABEL: func.func @direct_vnni_bf16(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {fastmath = #arith.fastmath<reassoc>, indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
func.func @direct_vnni_bf16(%a: memref<?x32x32xbf16>,
                            %b: memref<?x16x32x2xbf16>,
                            %init: vector<32x32xf32>, %ub: index,
                            %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>, fastmath = #arith.fastmath<reassoc>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of packed bf16 rhs read with leading unit dims
// CHECK-LABEL: func.func @leading_unit_vnni(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x1x1x16x32x2xbf16>
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
// NANO-LABEL: func.func @leading_unit_vnni(
// NANO-NOT: vector.contract
// NANO: return
func.func @leading_unit_vnni(%a: memref<?x32x32xbf16>,
                             %b: memref<?x1x1x16x32x2xbf16>,
                             %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero, %zero, %zero], %pad {in_bounds = [true, true, true, true, true]} : memref<?x1x1x16x32x2xbf16>, vector<1x1x16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 1, 2, 4, 3] : vector<1x1x16x32x2xbf16> to vector<1x1x16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<1x1x16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of flat bf16 rhs read split into lanes by shape_cast
// CHECK-LABEL: func.func @split_vnni_bf16(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x16x64xbf16>
// NANO-LABEL: func.func @split_vnni_bf16(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: %[[B_PACKED:.+]] = memref.expand_shape %[[B]] {{.*}} into memref<?x16x32x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B_PACKED]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
func.func @split_vnni_bf16(%a: memref<?x32x32xbf16>,
                           %b: memref<?x16x64xbf16>,
                           %init: vector<32x32xf32>, %ub: index,
                           %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %packed = vector.shape_cast %rhs : vector<16x64xbf16> to vector<16x32x2xbf16>
    %transpose = vector.transpose %packed, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of packed i8 rhs read into contraction
// CHECK-LABEL: func.func @direct_vnni_int8(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>, %[[B:[^,]+]]: memref<?x16x32x4xi8>
// NANO-LABEL: func.func @direct_vnni_int8(
// NANO-NOT: vector.contract
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

// fold VNNI unpacking of flat i8 rhs read split into lanes by shape_cast
// CHECK-LABEL: func.func @split_vnni_int8(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>, %[[B:[^,]+]]: memref<?x16x128xi8>
// NANO-LABEL: func.func @split_vnni_int8(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x4xi8>
// CHECK: %[[B_PACKED:.+]] = memref.expand_shape %[[B]] {{.*}} into memref<?x16x32x4xi8>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x4xi8>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B_PACKED]][%[[K]], {{.*}} vector<16x32x4xi8>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
func.func @split_vnni_int8(%a: memref<?x32x64xi8>,
                           %b: memref<?x16x128xi8>,
                           %init: vector<32x32xi32>, %ub: index,
                           %m: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %packed = vector.shape_cast %rhs : vector<16x128xi8> to vector<16x32x4xi8>
    %transpose = vector.transpose %packed, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// cast away leading unit dims of reads so that the contraction can be unrolled
// CHECK-LABEL: func.func @unit_dimension_reads(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?x32x32xbf16>, %[[B:[^,]+]]: memref<?x?x32x32xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][%[[K]], {{.*}} vector<32x32xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
// NANO-LABEL: func.func @unit_dimension_reads(
// NANO-NOT: vector.contract
// NANO: return
func.func @unit_dimension_reads(%a: memref<?x?x32x32xbf16>,
                                %b: memref<?x?x32x32xbf16>,
                                %init: vector<32x32xf32>, %ub: index,
                                %m: index, %n: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero, %zero], %pad {in_bounds = [true, true, true, true]} : memref<?x?x32x32xbf16>, vector<1x1x32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %n, %zero, %zero], %pad {in_bounds = [true, true, true, true]} : memref<?x?x32x32xbf16>, vector<1x1x32x32xbf16>
    %left = vector.shape_cast %lhs : vector<1x1x32x32xbf16> to vector<32x32xbf16>
    %right = vector.shape_cast %rhs : vector<1x1x32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %right, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking through a static view of a dynamic lhs innermost dim
// CHECK-LABEL: func.func @dynamic_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x?xbf16>
// NANO-LABEL: func.func @dynamic_inner_k(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[VIEW:.+]] = memref.reinterpret_cast %[[A]] {{.*}} sizes: [%{{.+}}, 32, 32]{{.*}} to memref<?x32x32xbf16, strided<[?, ?, 1]>>
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[VIEW]] {{.*}} into memref<?x32x16x2xbf16, strided<[?, ?, 2, 1]>>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %{{.+}}, %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
func.func @dynamic_inner_k(%a: memref<?x32x?xbf16>,
                           %b: memref<?x16x32x2xbf16>,
                           %init: vector<32x32xf32>, %ub: index,
                           %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x?xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost read offset 1 is not a multiple of VNNI factor 2 -> skip
// CHECK-LABEL: func.func @misaligned_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @misaligned_inner_k(%a: memref<?x32x64xbf16>,
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
// CHECK-LABEL: func.func @nondivisible_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x33xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @nondivisible_inner_k(%a: memref<?x32x33xbf16>,
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

// negative test: rhs VNNI lanes are reordered before unpacking -> skip
// CHECK-LABEL: func.func @interleaved_lanes(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x64xi8>
// CHECK: %[[EVEN:[^,]+]], %[[ODD:.+]] = vector.deinterleave
// CHECK: %[[LANES:.+]] = vector.interleave %[[ODD]], %[[EVEN]]
// CHECK: %[[T:.+]] = vector.transpose %[[LANES]], [0, 2, 1]
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x4x32xi8> to vector<64x32xi8>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
func.func @interleaved_lanes(%a: memref<?x32x64xi8>,
                             %b: memref<?x16x32x4xi8>,
                             %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x4xi8>, vector<16x32x4xi8>
    %even, %odd = vector.deinterleave %rhs : vector<16x32x4xi8> -> vector<16x32x2xi8>
    %reordered = vector.interleave %odd, %even : vector<16x32x2xi8> -> vector<16x32x4xi8>
    %transpose = vector.transpose %reordered, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// negative test: scalable lhs cannot be packed into a fixed VNNI shape -> skip
// CHECK-LABEL: func.func @scalable_vnni(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?x32xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<[32]x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<[32]x32xbf16>, vector<32x32xbf16> into vector<[32]x32xf32>
func.func @scalable_vnni(%a: memref<?x?x32xbf16>,
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
// CHECK-LABEL: func.func @vector_element_lhs(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32xvector<32xbf16>>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @vector_element_lhs(%a: memref<?x32xvector<32xbf16>>,
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
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
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
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