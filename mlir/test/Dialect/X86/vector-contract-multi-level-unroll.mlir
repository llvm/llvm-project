// RUN: mlir-opt %s -transform-interpreter -canonicalize -cse -split-input-file | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter=entry-point=__transform_nano -canonicalize -cse -split-input-file \
// RUN:   | FileCheck %s --check-prefix=NANO --implicit-check-not=vector.contract

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @amx_bf16_flat(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<128x64xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<128x64xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<128x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x64xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<128x32xbf16>, vector<32x64xbf16> into vector<128x64xf32>
    scf.yield %d : vector<128x64xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<128x64xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [128, 64] [1, 1]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK:             %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[ACC2:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[ACC3:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:             %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:             %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C32]]
// CHECK-SAME:          iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]], %[[ARG2:.+]] = %[[ACC2]], %[[ARG3:.+]] = %[[ACC3]])
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[OFF_M]], %[[IV_K]]] [32, 32] [1, 1]
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[OFF_N]]] [32, 32] [1, 1]
// CHECK:               %[[A0:.+]] = vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:               %[[A1:.+]] = vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:               %[[B0:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<32x16xbf16>
// CHECK:               %[[B1:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<32x16xbf16>
// CHECK:               %[[D0:.+]] = vector.contract {{.*}} %[[A0]], %[[B0]], %[[ARG0]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:               %[[D1:.+]] = vector.contract {{.*}} %[[A0]], %[[B1]], %[[ARG1]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>}
// CHECK:               %[[D2:.+]] = vector.contract {{.*}} %[[A1]], %[[B0]], %[[ARG2]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>}
// CHECK:               %[[D3:.+]] = vector.contract {{.*}} %[[A1]], %[[B1]], %[[ARG3]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>}
// CHECK:               scf.yield %[[D0]], %[[D1]], %[[D2]], %[[D3]]
// CHECK:             }
// CHECK:             vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:             vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C0]], %[[C16]]]
// CHECK:             vector.transfer_write %[[RES]]#2, %[[ACC_VIEW]][%[[C16]], %[[C0]]]
// CHECK:             vector.transfer_write %[[RES]]#3, %[[ACC_VIEW]][%[[C16]], %[[C16]]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// M = 48 is an odd multiple of 16, so a 1x2 tile register layout is used.
func.func @amx_bf16_vnni(%A: memref<?x?x2xbf16>, %B: memref<?x?x2xbf16>, %C: memref<?x?xf32>,
                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<48x64xf32>
  %res = scf.for %k = %k_start to %k_end step %c16 iter_args(%acc = %c) -> (vector<48x64xf32>) {
    %a = vector.transfer_read %A[%m, %k, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<48x16x2xbf16>
    %b = vector.transfer_read %B[%k, %n, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<16x64x2xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<48x16x2xbf16>, vector<16x64x2xbf16> into vector<48x64xf32>
    scf.yield %d : vector<48x64xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<48x64xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_vnni(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x2xbf16>, %[[B:.+]]: memref<?x?x2xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [48, 64] [1, 1]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C48]] step %[[C16]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [16, 32] [1, 1]
// CHECK:             %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:             %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:             %[[RES:.+]]:2 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C16]]
// CHECK-SAME:          iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]])
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[OFF_M]], %[[IV_K]], 0] [16, 16, 2] [1, 1, 1]
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[OFF_N]], 0] [16, 32, 2] [1, 1, 1]
// CHECK:               %[[A0:.+]] = vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               %[[B0:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               %[[B1:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               %[[D0:.+]] = vector.contract {{.*}} %[[A0]], %[[B0]], %[[ARG0]] {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 2>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// CHECK:               %[[D1:.+]] = vector.contract {{.*}} %[[A0]], %[[B1]], %[[ARG1]] {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 2>}
// CHECK:               scf.yield %[[D0]], %[[D1]]
// CHECK:             }
// CHECK:             vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:             vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C0]], %[[C16]]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_vnni(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Blocked layout: the two leading memref dims select a block, and the block
// has the same shape as the operand vectors.
func.func @amx_bf16_flat_blocked(%A: memref<?x?x96x32xbf16>, %B: memref<?x?x32x64xbf16>, %C: memref<?x?x96x64xf32>,
                                 %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n, %c0, %c0], %c0_0_f32  {in_bounds = [true, true]} : memref<?x?x96x64xf32>, vector<96x64xf32>
  %res = scf.for %k = %k_start to %k_end step %c1 iter_args(%acc = %c) -> (vector<96x64xf32>) {
    %a = vector.transfer_read %A[%m, %k, %c0, %c0], %c0_0 {in_bounds = [true, true]} : memref<?x?x96x32xbf16>, vector<96x32xbf16>
    %b = vector.transfer_read %B[%n, %k, %c0, %c0], %c0_0 {in_bounds = [true, true]} : memref<?x?x32x64xbf16>, vector<32x64xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<96x32xbf16>, vector<32x64xbf16> into vector<96x64xf32>
    scf.yield %d : vector<96x64xf32>
  }
  vector.transfer_write %res, %C[%m, %n, %c0, %c0] {in_bounds = [true, true]} : vector<96x64xf32>, memref<?x?x96x64xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_blocked(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x96x32xbf16>, %[[B:.+]]: memref<?x?x32x64xbf16>, %[[C:.+]]: memref<?x?x96x64xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]], 0, 0] [1, 1, 96, 64] [1, 1, 1, 1]
// CHECK-SAME:      memref<?x?x96x64xf32> to memref<96x64xf32, strided<[64, 1], offset: ?>>
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C1]]
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[M]], %[[IV_K]], %[[IV_M]], 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:            memref<?x?x96x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[N]], %[[IV_K]], 0, %[[IV_N]]] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:            memref<?x?x32x64xbf16> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<32x16xbf16>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<32x16xbf16>
// CHECK-COUNT-4:       vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat_blocked(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

func.func @amx_bf16_vnni_blocked(%A: memref<?x?x128x16x2xbf16>, %B: memref<?x?x16x96x2xbf16>, %C: memref<?x?x128x96xf32>,
                                 %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n, %c0, %c0], %c0_0_f32  {in_bounds = [true, true]} : memref<?x?x128x96xf32>, vector<128x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c1 iter_args(%acc = %c) -> (vector<128x96xf32>) {
    %a = vector.transfer_read %A[%m, %k, %c0, %c0, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x128x16x2xbf16>, vector<128x16x2xbf16>
    %b = vector.transfer_read %B[%n, %k, %c0, %c0, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x16x96x2xbf16>, vector<16x96x2xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<128x16x2xbf16>, vector<16x96x2xbf16> into vector<128x96xf32>
    scf.yield %d : vector<128x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n, %c0, %c0] {in_bounds = [true, true]} : vector<128x96xf32>, memref<?x?x128x96xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_vnni_blocked(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x128x16x2xbf16>, %[[B:.+]]: memref<?x?x16x96x2xbf16>, %[[C:.+]]: memref<?x?x128x96xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]], 0, 0] [1, 1, 128, 96] [1, 1, 1, 1]
// CHECK-SAME:      memref<?x?x128x96xf32> to memref<128x96xf32, strided<[96, 1], offset: ?>>
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C1]]
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[M]], %[[IV_K]], %[[IV_M]], 0, 0] [1, 1, 32, 16, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:            memref<?x?x128x16x2xbf16> to memref<32x16x2xbf16, strided<[32, 2, 1], offset: ?>>
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[N]], %[[IV_K]], 0, %[[IV_N]], 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:            memref<?x?x16x96x2xbf16> to memref<16x32x2xbf16, strided<[192, 2, 1], offset: ?>>
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK-COUNT-4:       vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 2>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_vnni_blocked(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @amx_int8_flat(%A: memref<?x?xi8>, %B: memref<?x?xi8>, %C: memref<?x?xi32>,
                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c64 = arith.constant 64 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %c = vector.transfer_read %C[%m, %n], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32>, vector<64x128xi32>
  %res = scf.for %k = %k_start to %k_end step %c64 iter_args(%acc = %c) -> (vector<64x128xi32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8>, vector<64x64xi8>
    %b = vector.transfer_read %B[%k, %n], %c0_i8 {in_bounds = [true, true]} : memref<?x?xi8>, vector<64x128xi8>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x64xi8>, vector<64x128xi8> into vector<64x128xi32>
    scf.yield %d : vector<64x128xi32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x128xi32>, memref<?x?xi32>
  func.return
}

// CHECK-LABEL: func.func @amx_int8_flat(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xi8>, %[[B:.+]]: memref<?x?xi8>, %[[C:.+]]: memref<?x?xi32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [64, 128] [1, 1]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xi32>
// CHECK:             %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:             %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:             %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C64]]
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[OFF_M]], %[[IV_K]]] [32, 64] [1, 1]
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[OFF_N]]] [64, 32] [1, 1]
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:               vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<64x16xi8>
// CHECK:               vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<64x16xi8>
// CHECK-COUNT-4:       vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 64>} : vector<16x64xi8>, vector<64x16xi8> into vector<16x16xi32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_int8_flat(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// N = 48 is an odd multiple of 16, so a 2x1 tile register layout is used.
func.func @amx_int8_vnni(%A: memref<?x?x4xi8>, %B: memref<?x?x4xi8>, %C: memref<?x?xi32>,
                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %c = vector.transfer_read %C[%m, %n], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32>, vector<64x48xi32>
  %res = scf.for %k = %k_start to %k_end step %c16 iter_args(%acc = %c) -> (vector<64x48xi32>) {
    %a = vector.transfer_read %A[%m, %k, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<?x?x4xi8>, vector<64x16x4xi8>
    %b = vector.transfer_read %B[%k, %n, %c0], %c0_i8 {in_bounds = [true, true, true]} : memref<?x?x4xi8>, vector<16x48x4xi8>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x16x4xi8>, vector<16x48x4xi8> into vector<64x48xi32>
    scf.yield %d : vector<64x48xi32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x48xi32>, memref<?x?xi32>
  func.return
}

// CHECK-LABEL: func.func @amx_int8_vnni(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x4xi8>, %[[B:.+]]: memref<?x?x4xi8>, %[[C:.+]]: memref<?x?xi32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [64, 48] [1, 1]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C48]] step %[[C16]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 16] [1, 1]
// CHECK:             %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xi32>
// CHECK:             %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x16xi32>
// CHECK:             %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:             %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:             %[[RES:.+]]:2 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C16]]
// CHECK-SAME:          iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]])
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[OFF_M]], %[[IV_K]], 0] [32, 16, 4] [1, 1, 1]
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[OFF_N]], 0] [16, 16, 4] [1, 1, 1]
// CHECK:               %[[A0:.+]] = vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x4xi8>
// CHECK:               %[[A1:.+]] = vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x4xi8>
// CHECK:               %[[B0:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x4xi8>
// CHECK:               %[[D0:.+]] = vector.contract {{.*}} %[[A0]], %[[B0]], %[[ARG0]] {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 4>} : vector<16x16x4xi8>, vector<16x16x4xi8> into vector<16x16xi32>
// CHECK:               %[[D1:.+]] = vector.contract {{.*}} %[[A1]], %[[B0]], %[[ARG1]] {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 4>}
// CHECK:               scf.yield %[[D0]], %[[D1]]
// CHECK:             }
// CHECK:             vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:             vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C16]], %[[C0]]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_int8_vnni(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Zero-initialized accumulator: no reads from the destination, and the K-loop
// starts from a native-shaped zero constant.
func.func @amx_bf16_flat_zero_init(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                   %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c = arith.constant dense<0.0> : vector<96x128xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<96x128xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<96x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x128xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<96x32xbf16>, vector<32x128xbf16> into vector<96x128xf32>
    scf.yield %d : vector<96x128xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<96x128xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_zero_init(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf32>
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [96, 128] [1, 1]
// CHECK-NOT:     vector.transfer_read
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK-NOT:         vector.transfer_read
// CHECK:             %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C32]]
// CHECK-SAME:          iter_args(%{{.+}} = %[[ZERO]], %{{.+}} = %[[ZERO]], %{{.+}} = %[[ZERO]], %{{.+}} = %[[ZERO]])
// CHECK-COUNT-4:       vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat_zero_init(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The initial accumulator is not backed by memory, so it is spilled to a stack
// buffer. The result is still written through a subview of the destination.
func.func @amx_bf16_flat_init_from_arg(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                       %c: vector<64x96xf32>,
                                       %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x96xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_init_from_arg(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[ACC_INIT:.+]]: vector<64x96xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK:         %[[BUF:.+]] = memref.alloca() : memref<64x96xf32>
// CHECK:         vector.transfer_write %[[ACC_INIT]], %[[BUF]][%[[C0]], %[[C0]]]
// CHECK:         %[[ACC_OUT:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [64, 96] [1, 1]
// CHECK-NOT:     memref.alloca
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:             %[[IN_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[IN_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for
// CHECK:             %[[OUT_VIEW:.+]] = memref.subview %[[ACC_OUT]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[OUT_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat_init_from_arg(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The loop result feeds an epilogue instead of a plain store, so it is
// buffered on the stack and rematerialized. The initial accumulator is still
// read through a subview.
func.func @amx_bf16_flat_truncf_result(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>, %D: memref<?x?xbf16>,
                                       %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x96xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  %trunc = arith.truncf %res : vector<64x96xf32> to vector<64x96xbf16>
  vector.transfer_write %trunc, %D[%m, %n] {in_bounds = [true, true]} : vector<64x96xbf16>, memref<?x?xbf16>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_truncf_result(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>, %[[D:.+]]: memref<?x?xbf16>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG:     %[[ACC_IN:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [64, 96] [1, 1]
// CHECK-DAG:     %[[BUF:.+]] = memref.alloca() : memref<64x96xf32>
// CHECK-NOT:     vector.transfer_write
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:             %[[IN_VIEW:.+]] = memref.subview %[[ACC_IN]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[IN_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for
// CHECK:             %[[OUT_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[OUT_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK:         %[[ACC_RES:.+]] = vector.transfer_read %[[BUF]][%[[C0]], %[[C0]]], {{.*}} vector<64x96xf32>
// CHECK:         %[[TRUNC:.+]] = arith.truncf %[[ACC_RES]]
// CHECK:         vector.transfer_write %[[TRUNC]], %[[D]][%[[M]], %[[N]]]

// NANO-LABEL: func.func @amx_bf16_flat_truncf_result(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Neither the initial read nor the final write are fully in-bounds, so a
// single stack buffer is used for both; the original transfers remain.
func.func @amx_bf16_flat_acc_out_of_bounds(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                           %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [false, false]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x96xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [false, false]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_acc_out_of_bounds(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-NOT:     memref.subview %[[C]]
// CHECK:         %[[ACC_INIT:.+]] = vector.transfer_read %[[C]][%[[M]], %[[N]]], {{.*}} : memref<?x?xf32>, vector<64x96xf32>
// CHECK:         %[[BUF:.+]] = memref.alloca() : memref<64x96xf32>
// CHECK:         vector.transfer_write %[[ACC_INIT]], %[[BUF]][%[[C0]], %[[C0]]]
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview %[[C]]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for
// CHECK-NOT:         memref.subview %[[BUF]]
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK:         %[[ACC_RES:.+]] = vector.transfer_read %[[BUF]][%[[C0]], %[[C0]]], {{.*}} vector<64x96xf32>
// CHECK:         vector.transfer_write %[[ACC_RES]], %[[C]][%[[M]], %[[N]]] : vector<64x96xf32>, memref<?x?xf32>

// NANO-LABEL: func.func @amx_bf16_flat_acc_out_of_bounds(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The initial read and the final write target the same memref but at different
// offsets, so the regions partially overlap. Writing the result tiles through
// a view would clobber initial values not yet read by later M/N iterations, so
// the initial value is still read via a subview, but the result goes through a
// stack buffer and the original write remains.
func.func @amx_bf16_flat_acc_partial_overlap(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                             %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %m_off = arith.addi %m, %c32 : index
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x96xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m_off, %n] {in_bounds = [true, true]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_acc_partial_overlap(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG:     %[[M_OFF:.+]] = arith.addi %[[M]], %[[C32]]
// CHECK-DAG:     %[[ACC_IN:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [64, 96] [1, 1]
// CHECK-DAG:     %[[BUF:.+]] = memref.alloca() : memref<64x96xf32>
// CHECK-NOT:     memref.subview %[[C]]
// CHECK-NOT:     vector.transfer_write
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:             %[[IN_VIEW:.+]] = memref.subview %[[ACC_IN]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_read %[[IN_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:             %[[RES:.+]]:4 = scf.for
// CHECK:             %[[OUT_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:     vector.transfer_write %[[RES]]#{{[0-3]}}, %[[OUT_VIEW]]
// CHECK:           }
// CHECK:         }
// CHECK:         %[[ACC_RES:.+]] = vector.transfer_read %[[BUF]][%[[C0]], %[[C0]]], {{.*}} vector<64x96xf32>
// CHECK:         vector.transfer_write %[[ACC_RES]], %[[C]][%[[M_OFF]], %[[N]]] {{.*}} : vector<64x96xf32>, memref<?x?xf32>

// NANO-LABEL: func.func @amx_bf16_flat_acc_partial_overlap(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// M = 48 is an odd multiple of 16, so a 1x2 tile register layout is used.
// Online packing of the flat RHS needs the pair of contracts along N, which
// this layout still provides.
func.func @amx_bf16_flat_1x2_tile(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                  %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<48x64xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<48x64xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<48x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x64xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<48x32xbf16>, vector<32x64xbf16> into vector<48x64xf32>
    scf.yield %d : vector<48x64xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<48x64xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_1x2_tile(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [48, 64] [1, 1]
// CHECK:         scf.for %[[IV_M:.+]] = %[[C0]] to %[[C48]] step %[[C16]] {
// CHECK:           scf.for %[[IV_N:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [16, 32] [1, 1]
// CHECK:             %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:             %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:             %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:             %[[RES:.+]]:2 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C32]]
// CHECK-SAME:          iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]])
// CHECK-NOT:           arith.addi
// CHECK:               %[[A_VIEW:.+]] = memref.subview %[[A]][%[[OFF_M]], %[[IV_K]]] [16, 32] [1, 1]
// CHECK:               %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[OFF_N]]] [32, 32] [1, 1]
// CHECK:               %[[A0:.+]] = vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:               %[[B0:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<32x16xbf16>
// CHECK:               %[[B1:.+]] = vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<32x16xbf16>
// CHECK:               %[[D0:.+]] = vector.contract {{.*}} %[[A0]], %[[B0]], %[[ARG0]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:               %[[D1:.+]] = vector.contract {{.*}} %[[A0]], %[[B1]], %[[ARG1]] {x86_vcmlu_native_shape = array<i64: 16, 16, 32>}
// CHECK:               scf.yield %[[D0]], %[[D1]]
// CHECK:             }
// CHECK:             vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:             vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C0]], %[[C16]]]
// CHECK:           }
// CHECK:         }
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat_1x2_tile(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The original shape equals the register tile shape, so the M-N loop nest has
// a single iteration and folds away.
func.func @amx_bf16_flat_single_reg_tile(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<32x32xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<32x32xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %d : vector<32x32xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_single_reg_tile(
// CHECK-SAME:    %[[A:.+]]: memref<?x?xbf16>, %[[B:.+]]: memref<?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_VIEW:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [32, 32] [1, 1]
// CHECK-NOT:     memref.subview %[[ACC_VIEW]]
// CHECK:         %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC2:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC3:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C32]]
// CHECK-SAME:      iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]], %[[ARG2:.+]] = %[[ACC2]], %[[ARG3:.+]] = %[[ACC3]])
// CHECK-NOT:       arith.addi
// CHECK:           %[[A_VIEW:.+]] = memref.subview %[[A]][%[[M]], %[[IV_K]]] [32, 32] [1, 1]
// CHECK:           %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[N]]] [32, 32] [1, 1]
// CHECK:           vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:           vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:           vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<32x16xbf16>
// CHECK:           vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<32x16xbf16>
// CHECK-COUNT-4:   vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:         vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C0]], %[[C16]]]
// CHECK:         vector.transfer_write %[[RES]]#2, %[[ACC_VIEW]][%[[C16]], %[[C0]]]
// CHECK:         vector.transfer_write %[[RES]]#3, %[[ACC_VIEW]][%[[C16]], %[[C16]]]
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_flat_single_reg_tile(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

func.func @amx_bf16_vnni_single_reg_tile(%A: memref<?x?x2xbf16>, %B: memref<?x?x2xbf16>, %C: memref<?x?xf32>,
                                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<32x32xf32>
  %res = scf.for %k = %k_start to %k_end step %c16 iter_args(%acc = %c) -> (vector<32x32xf32>) {
    %a = vector.transfer_read %A[%m, %k, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<32x16x2xbf16>
    %b = vector.transfer_read %B[%k, %n, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<16x32x2xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
    scf.yield %d : vector<32x32xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_vnni_single_reg_tile(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x2xbf16>, %[[B:.+]]: memref<?x?x2xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index, %[[K_START:.+]]: index, %[[K_END:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-NOT:     memref.alloca
// CHECK:         %[[ACC_VIEW:.+]] = memref.subview %[[C]][%[[M]], %[[N]]] [32, 32] [1, 1]
// CHECK-NOT:     memref.subview %[[ACC_VIEW]]
// CHECK:         %[[ACC0:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC1:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC2:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[ACC3:.+]] = vector.transfer_read %[[ACC_VIEW]][%[[C16]], %[[C16]]], {{.*}} vector<16x16xf32>
// CHECK:         %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[K_START]] to %[[K_END]] step %[[C16]]
// CHECK-SAME:      iter_args(%[[ARG0:.+]] = %[[ACC0]], %[[ARG1:.+]] = %[[ACC1]], %[[ARG2:.+]] = %[[ACC2]], %[[ARG3:.+]] = %[[ACC3]])
// CHECK-NOT:       arith.addi
// CHECK:           %[[A_VIEW:.+]] = memref.subview %[[A]][%[[M]], %[[IV_K]], 0] [32, 16, 2] [1, 1, 1]
// CHECK:           %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_K]], %[[N]], 0] [16, 32, 2] [1, 1, 1]
// CHECK:           vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:           vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:           vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK:           vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]], %[[C0]]], {{.*}} vector<16x16x2xbf16>
// CHECK-COUNT-4:   vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 16, 2>} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// CHECK:           scf.yield
// CHECK:         }
// CHECK:         vector.transfer_write %[[RES]]#0, %[[ACC_VIEW]][%[[C0]], %[[C0]]]
// CHECK:         vector.transfer_write %[[RES]]#1, %[[ACC_VIEW]][%[[C0]], %[[C16]]]
// CHECK:         vector.transfer_write %[[RES]]#2, %[[ACC_VIEW]][%[[C16]], %[[C0]]]
// CHECK:         vector.transfer_write %[[RES]]#3, %[[ACC_VIEW]][%[[C16]], %[[C16]]]
// CHECK-NOT:     vector.transfer_read
// CHECK-NOT:     vector.transfer_write
// CHECK:         return

// NANO-LABEL: func.func @amx_bf16_vnni_single_reg_tile(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The accumulation loop sits inside an existing batch-M-N loop nest.
func.func @amx_int8_flat_batched_loop_nest(%A: memref<?x?x?xi8>, %B: memref<?x?x?xi8>, %C: memref<?x?x?xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %batch = memref.dim %C, %c0 : memref<?x?x?xi32>
  %M = memref.dim %C, %c1 : memref<?x?x?xi32>
  %N = memref.dim %C, %c2 : memref<?x?x?xi32>
  %K = memref.dim %A, %c2 : memref<?x?x?xi8>
  scf.for %bi = %c0 to %batch step %c1 {
    scf.for %m = %c0 to %M step %c64 {
      scf.for %n = %c0 to %N step %c128 {
        %c = vector.transfer_read %C[%bi, %m, %n], %c0_i32 {in_bounds = [true, true]} : memref<?x?x?xi32>, vector<64x128xi32>
        %res = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %c) -> (vector<64x128xi32>) {
          %a = vector.transfer_read %A[%bi, %m, %k], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?xi8>, vector<64x64xi8>
          %b = vector.transfer_read %B[%bi, %k, %n], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?xi8>, vector<64x128xi8>
          %d = vector.contract {
            indexing_maps = [#map, #map1, #map2],
            iterator_types = ["parallel", "parallel", "reduction"],
            kind = #vector.kind<add>}
            %a, %b, %acc : vector<64x64xi8>, vector<64x128xi8> into vector<64x128xi32>
          scf.yield %d : vector<64x128xi32>
        }
        vector.transfer_write %res, %C[%bi, %m, %n] {in_bounds = [true, true]} : vector<64x128xi32>, memref<?x?x?xi32>
      }
    }
  }
  func.return
}

// CHECK-LABEL: func.func @amx_int8_flat_batched_loop_nest(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x?xi8>, %[[B:.+]]: memref<?x?x?xi8>, %[[C:.+]]: memref<?x?x?xi32>)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[BATCH:.+]] = memref.dim %[[C]], %[[C0]]
// CHECK-DAG:     %[[M:.+]] = memref.dim %[[C]], %[[C1]]
// CHECK-DAG:     %[[N:.+]] = memref.dim %[[C]], %[[C2]]
// CHECK-DAG:     %[[K:.+]] = memref.dim %[[A]], %[[C2]]
// CHECK-NOT:     memref.alloca
// CHECK:         scf.for %[[IV_B:.+]] = %[[C0]] to %[[BATCH]] step %[[C1]] {
// CHECK:           scf.for %[[OFF_M:.+]] = %[[C0]] to %[[M]] step %[[C64]] {
// CHECK:             scf.for %[[OFF_N:.+]] = %[[C0]] to %[[N]] step %[[C128]] {
// CHECK:               %[[ACC_BUF:.+]] = memref.subview %[[C]][%[[IV_B]], %[[OFF_M]], %[[OFF_N]]] [1, 64, 128] [1, 1, 1]
// CHECK-SAME:            memref<?x?x?xi32> to memref<64x128xi32, strided<[?, 1], offset: ?>>
// CHECK:               scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:                 scf.for %[[IV_N:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:                   %[[ACC_VIEW:.+]] = memref.subview %[[ACC_BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:           vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xi32>
// CHECK:                   %[[A_OFF_M:.+]] = arith.addi %[[OFF_M]], %[[IV_M]]
// CHECK:                   %[[B_OFF_N:.+]] = arith.addi %[[OFF_N]], %[[IV_N]]
// CHECK:                   %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[C0]] to %[[K]] step %[[C64]]
// CHECK-NOT:                 arith.addi
// CHECK:                     %[[A_VIEW:.+]] = memref.subview %[[A]][%[[IV_B]], %[[A_OFF_M]], %[[IV_K]]] [1, 32, 64] [1, 1, 1]
// CHECK:                     %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_B]], %[[IV_K]], %[[B_OFF_N]]] [1, 64, 32] [1, 1, 1]
// CHECK:                     vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:                     vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:                     vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<64x16xi8>
// CHECK:                     vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<64x16xi8>
// CHECK-COUNT-4:             vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 64>} : vector<16x64xi8>, vector<64x16xi8> into vector<16x16xi32>
// CHECK:                     scf.yield
// CHECK:                   }
// CHECK-COUNT-4:           vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:                 }
// CHECK:               }
// CHECK-NOT:           vector.transfer_read
// CHECK-NOT:           vector.transfer_write
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:         return

// NANO-LABEL: func.func @amx_int8_flat_batched_loop_nest(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// The accumulator is loop-carried through an enclosing batch loop, so it is
// neither read from nor written to memory directly.
// 
// This test is currently intended to demonstrate successful lowering using
// the fallback stack-buffer path. A more desirable lowering would keep the
// accumulator on the tile registers across both loops.
func.func @amx_int8_flat_batch_reduction(%A: memref<?x?x?xi8>, %B: memref<?x?x?xi8>, %C: memref<?x?xi32>,
                                         %m: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c0_i8 = arith.constant 0 : i8
  %c0_i32 = arith.constant 0 : i32
  %batch = memref.dim %A, %c0 : memref<?x?x?xi8>
  %K = memref.dim %A, %c2 : memref<?x?x?xi8>
  %c = vector.transfer_read %C[%m, %n], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32>, vector<64x128xi32>
  %res = scf.for %b = %c0 to %batch step %c1 iter_args(%acc_b = %c) -> (vector<64x128xi32>) {
    %res_b = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %acc_b) -> (vector<64x128xi32>) {
      %a = vector.transfer_read %A[%b, %m, %k], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?xi8>, vector<64x64xi8>
      %bv = vector.transfer_read %B[%b, %k, %n], %c0_i8 {in_bounds = [true, true]} : memref<?x?x?xi8>, vector<64x128xi8>
      %d = vector.contract {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %a, %bv, %acc : vector<64x64xi8>, vector<64x128xi8> into vector<64x128xi32>
      scf.yield %d : vector<64x128xi32>
    }
    scf.yield %res_b : vector<64x128xi32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x128xi32>, memref<?x?xi32>
  func.return
}

// CHECK-LABEL: func.func @amx_int8_flat_batch_reduction(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x?xi8>, %[[B:.+]]: memref<?x?x?xi8>, %[[C:.+]]: memref<?x?xi32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:     %[[BATCH:.+]] = memref.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[K:.+]] = memref.dim %[[A]], %[[C2]]
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK:         %[[ACC_INIT:.+]] = vector.transfer_read %[[C]][%[[M]], %[[N]]]
// CHECK:         %[[ACC_RES:.+]] = scf.for %[[IV_B:.+]] = %[[C0]] to %[[BATCH]] step %[[C1]] iter_args(%[[ACC_B:.+]] = %[[ACC_INIT]])
// CHECK:           %[[BUF:.+]] = memref.alloca() : memref<64x128xi32>
// CHECK:           vector.transfer_write %[[ACC_B]], %[[BUF]][%[[C0]], %[[C0]]]
// CHECK-NOT:       memref.alloca
// CHECK:           scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             scf.for %[[IV_N:.+]] = %[[C0]] to %[[C128]] step %[[C32]] {
// CHECK:               %[[ACC_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:       vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xi32>
// CHECK:               %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:               %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:               %[[RES:.+]]:4 = scf.for %[[IV_K:.+]] = %[[C0]] to %[[K]] step %[[C64]]
// CHECK-NOT:             arith.addi
// CHECK:                 %[[A_VIEW:.+]] = memref.subview %[[A]][%[[IV_B]], %[[OFF_M]], %[[IV_K]]] [1, 32, 64] [1, 1, 1]
// CHECK:                 %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_B]], %[[IV_K]], %[[OFF_N]]] [1, 64, 32] [1, 1, 1]
// CHECK:                 vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:                 vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x64xi8>
// CHECK:                 vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<64x16xi8>
// CHECK:                 vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<64x16xi8>
// CHECK-COUNT-4:         vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 64>} : vector<16x64xi8>, vector<64x16xi8> into vector<16x16xi32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK-NOT:           memref.subview %[[BUF]]
// CHECK-COUNT-4:       vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:             }
// CHECK:           }
// CHECK:           %[[ACC_NEXT:.+]] = vector.transfer_read %[[BUF]][%[[C0]], %[[C0]]], {{.*}} vector<64x128xi32>
// CHECK:           scf.yield %[[ACC_NEXT]]
// CHECK:         }
// CHECK:         vector.transfer_write %[[ACC_RES]], %[[C]][%[[M]], %[[N]]]

// NANO-LABEL: func.func @amx_int8_flat_batch_reduction(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Loop order K-batch, accumulating over both loops: the inner batch loop
// carries the accumulator, its IV appears exactly once in each operand read and
// its step is 1. It is a valid accumulation loop, indistinguishable from a
// K-loop over a blocked layout, and is treated as such.
//
// Again, spilling the accumulator value to memory in the outer loop is correct,
// but not ideal.
func.func @amx_bf16_flat_k_batch_order(%A: memref<?x?x?xbf16>, %B: memref<?x?x?xbf16>, %C: memref<?x?xf32>,
                                       %m: index, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %batch = memref.dim %A, %c0 : memref<?x?x?xbf16>
  %K = memref.dim %A, %c2 : memref<?x?x?xbf16>
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %c0 to %K step %c32 iter_args(%acc_k = %c) -> (vector<64x96xf32>) {
    %res_k = scf.for %bi = %c0 to %batch step %c1 iter_args(%acc = %acc_k) -> (vector<64x96xf32>) {
      %a = vector.transfer_read %A[%bi, %m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?x?xbf16>, vector<64x32xbf16>
      %b = vector.transfer_read %B[%bi, %k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?x?xbf16>, vector<32x96xbf16>
      %d = vector.contract {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel", "reduction"],
        kind = #vector.kind<add>}
        %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
      scf.yield %d : vector<64x96xf32>
    }
    scf.yield %res_k : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @amx_bf16_flat_k_batch_order(
// CHECK-SAME:    %[[A:.+]]: memref<?x?x?xbf16>, %[[B:.+]]: memref<?x?x?xbf16>, %[[C:.+]]: memref<?x?xf32>,
// CHECK-SAME:    %[[M:.+]]: index, %[[N:.+]]: index)
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:     %[[C96:.+]] = arith.constant 96 : index
// CHECK-DAG:     %[[BATCH:.+]] = memref.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[K:.+]] = memref.dim %[[A]], %[[C2]]
// CHECK-NOT:     memref.alloca
// CHECK-NOT:     memref.subview
// CHECK:         %[[ACC_INIT:.+]] = vector.transfer_read %[[C]][%[[M]], %[[N]]]
// CHECK:         %[[ACC_RES:.+]] = scf.for %[[IV_K:.+]] = %[[C0]] to %[[K]] step %[[C32]] iter_args(%[[ACC_K:.+]] = %[[ACC_INIT]])
// CHECK:           %[[BUF:.+]] = memref.alloca() : memref<64x96xf32>
// CHECK:           vector.transfer_write %[[ACC_K]], %[[BUF]][%[[C0]], %[[C0]]]
// CHECK-NOT:       memref.alloca
// CHECK:           scf.for %[[IV_M:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CHECK:             scf.for %[[IV_N:.+]] = %[[C0]] to %[[C96]] step %[[C32]] {
// CHECK:               %[[ACC_VIEW:.+]] = memref.subview %[[BUF]][%[[IV_M]], %[[IV_N]]] [32, 32] [1, 1]
// CHECK-COUNT-4:       vector.transfer_read %[[ACC_VIEW]]{{.*}} vector<16x16xf32>
// CHECK:               %[[OFF_M:.+]] = arith.addi %[[M]], %[[IV_M]]
// CHECK:               %[[OFF_N:.+]] = arith.addi %[[N]], %[[IV_N]]
// CHECK:               %[[RES:.+]]:4 = scf.for %[[IV_B:.+]] = %[[C0]] to %[[BATCH]] step %[[C1]]
// CHECK-NOT:             arith.addi
// CHECK:                 %[[A_VIEW:.+]] = memref.subview %[[A]][%[[IV_B]], %[[OFF_M]], %[[IV_K]]] [1, 32, 32] [1, 1, 1]
// CHECK:                 %[[B_VIEW:.+]] = memref.subview %[[B]][%[[IV_B]], %[[IV_K]], %[[OFF_N]]] [1, 32, 32] [1, 1, 1]
// CHECK:                 vector.transfer_read %[[A_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:                 vector.transfer_read %[[A_VIEW]][%[[C16]], %[[C0]]], {{.*}} vector<16x32xbf16>
// CHECK:                 vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C0]]], {{.*}} vector<32x16xbf16>
// CHECK:                 vector.transfer_read %[[B_VIEW]][%[[C0]], %[[C16]]], {{.*}} vector<32x16xbf16>
// CHECK-COUNT-4:         vector.contract {{.*}} {x86_vcmlu_native_shape = array<i64: 16, 16, 32>} : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK-NOT:           memref.subview %[[BUF]]
// CHECK-COUNT-4:       vector.transfer_write %[[RES]]#{{[0-3]}}, %[[ACC_VIEW]]
// CHECK:             }
// CHECK:           }
// CHECK:           %[[ACC_NEXT:.+]] = vector.transfer_read %[[BUF]][%[[C0]], %[[C0]]], {{.*}} vector<64x96xf32>
// CHECK:           scf.yield %[[ACC_NEXT]]
// CHECK:         }
// CHECK:         vector.transfer_write %[[ACC_RES]], %[[C]][%[[M]], %[[N]]]

// NANO-LABEL: func.func @amx_bf16_flat_k_batch_order(

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

// RHS is transposed (N x K instead of K x N).
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_transposed_operand(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                       %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%n, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<96x32xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<96x32xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @negative_transposed_operand(
// CHECK:         vector.contract {{.*}} : vector<64x32xbf16>, vector<96x32xbf16> into vector<64x96xf32>

// NANO-LABEL:  func.func @negative_transposed_operand(
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Loop order M-K-N, accumulating over both K and N: the IV of the innermost
// loop carrying the accumulator (N) is not present in both operand loads'
// indices.
func.func @negative_k_loop_not_innermost(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %M = memref.dim %C, %c0 : memref<?x?xf32>
  %N = memref.dim %C, %c1 : memref<?x?xf32>
  %K = memref.dim %A, %c1 : memref<?x?xbf16>
  scf.for %m = %c0 to %M step %c64 {
    %c = vector.transfer_read %C[%m, %c0], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x32xf32>
    %res = scf.for %k = %c0 to %K step %c32 iter_args(%acc_k = %c) -> (vector<64x32xf32>) {
      %res_k = scf.for %n = %c0 to %N step %c32 iter_args(%acc = %acc_k) -> (vector<64x32xf32>) {
        %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
        %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
        %d = vector.contract {
          indexing_maps = [#map, #map1, #map2],
          iterator_types = ["parallel", "parallel", "reduction"],
          kind = #vector.kind<add>}
          %a, %b, %acc : vector<64x32xbf16>, vector<32x32xbf16> into vector<64x32xf32>
        scf.yield %d : vector<64x32xf32>
      }
      scf.yield %res_k : vector<64x32xf32>
    }
    vector.transfer_write %res, %C[%m, %c0] {in_bounds = [true, true]} : vector<64x32xf32>, memref<?x?xf32>
  }
  func.return
}

// CHECK-LABEL: func.func @negative_k_loop_not_innermost(
// CHECK:         vector.contract {{.*}} : vector<64x32xbf16>, vector<32x32xbf16> into vector<64x32xf32>

// NANO-LABEL:  func.func @negative_k_loop_not_innermost(
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_operand_read_out_of_bounds(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                               %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x96xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, false]} : memref<?x?xbf16>, vector<32x96xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>
    scf.yield %d : vector<64x96xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x96xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @negative_operand_read_out_of_bounds(
// CHECK:         vector.contract {{.*}} : vector<64x32xbf16>, vector<32x96xbf16> into vector<64x96xf32>

// NANO-LABEL:  func.func @negative_operand_read_out_of_bounds(
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_unsupported_type(%A: memref<?x?xi16>, %B: memref<?x?xi16>, %C: memref<?x?xi32>,
                                     %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_i16 = arith.constant 0 : i16
  %c0_i32 = arith.constant 0 : i32
  %c = vector.transfer_read %C[%m, %n], %c0_i32 {in_bounds = [true, true]} : memref<?x?xi32>, vector<64x96xi32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x96xi32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_i16 {in_bounds = [true, true]} : memref<?x?xi16>, vector<64x32xi16>
    %b = vector.transfer_read %B[%k, %n], %c0_i16 {in_bounds = [true, true]} : memref<?x?xi16>, vector<32x96xi16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xi16>, vector<32x96xi16> into vector<64x96xi32>
    scf.yield %d : vector<64x96xi32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x96xi32>, memref<?x?xi32>
  func.return
}

// CHECK-LABEL: func.func @negative_unsupported_type(
// CHECK:         vector.contract {{.*}} : vector<64x32xi16>, vector<32x96xi16> into vector<64x96xi32>

// NANO-LABEL:  func.func @negative_unsupported_type(
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-int8"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

// N = 48 would require a 2x1 tile register layout, but online packing of the
// flat RHS needs a pair of contracts along N, i.e. N must be a multiple of 32.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @negative_shape_not_unrollable(%A: memref<?x?xbf16>, %B: memref<?x?xbf16>, %C: memref<?x?xf32>,
                                         %m: index, %n: index, %k_start: index, %k_end: index) {
  %c32 = arith.constant 32 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<64x48xf32>
  %res = scf.for %k = %k_start to %k_end step %c32 iter_args(%acc = %c) -> (vector<64x48xf32>) {
    %a = vector.transfer_read %A[%m, %k], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<64x32xbf16>
    %b = vector.transfer_read %B[%k, %n], %c0_0 {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x48xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<64x32xbf16>, vector<32x48xbf16> into vector<64x48xf32>
    scf.yield %d : vector<64x48xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<64x48xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @negative_shape_not_unrollable(
// CHECK:         vector.contract {{.*}} : vector<64x32xbf16>, vector<32x48xbf16> into vector<64x48xf32>

// NANO-LABEL:  func.func @negative_shape_not_unrollable(
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

// VNNI contraction already has the native tile shape; nothing to do.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

func.func @negative_vnni_native_shape(%A: memref<?x?x2xbf16>, %B: memref<?x?x2xbf16>, %C: memref<?x?xf32>,
                                      %m: index, %n: index, %k_start: index, %k_end: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c0_0 = arith.constant 0.0 : bf16
  %c0_0_f32 = arith.constant 0.0 : f32
  %c = vector.transfer_read %C[%m, %n], %c0_0_f32 {in_bounds = [true, true]} : memref<?x?xf32>, vector<16x16xf32>
  %res = scf.for %k = %k_start to %k_end step %c16 iter_args(%acc = %c) -> (vector<16x16xf32>) {
    %a = vector.transfer_read %A[%m, %k, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<16x16x2xbf16>
    %b = vector.transfer_read %B[%k, %n, %c0], %c0_0 {in_bounds = [true, true, true]} : memref<?x?x2xbf16>, vector<16x16x2xbf16>
    %d = vector.contract {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"],
      kind = #vector.kind<add>}
      %a, %b, %acc : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
    scf.yield %d : vector<16x16xf32>
  }
  vector.transfer_write %res, %C[%m, %n] {in_bounds = [true, true]} : vector<16x16xf32>, memref<?x?xf32>
  func.return
}

// CHECK-LABEL: func.func @negative_vnni_native_shape(
// CHECK:         vector.contract {{.*}} : vector<16x16x2xbf16>, vector<16x16x2xbf16> into vector<16x16xf32>
// CHECK-NOT:     x86_vcmlu_native_shape

// NANO-LABEL:  func.func @negative_vnni_native_shape(
// Operands are read directly from a function argument; the AMX lowering
// requires an op (e.g. memref.subview) to define the read source.
// NANO:          vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll target = "amx-bf16"
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}
