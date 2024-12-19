// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 4)>
#map3 = affine_map<(d0) -> (d0 + 3)>

func.func @matmul_f32(%A: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %3 = memref.view %A[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %4 = memref.view %A[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %5 = memref.view %A[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  %6 = memref.dim %3, %c0 : memref<?x?xf32>
  %7 = memref.dim %3, %c1 : memref<?x?xf32>
  %8 = memref.dim %4, %c1 : memref<?x?xf32>
  scf.for %arg4 = %c0 to %6 step %c2 {
    scf.for %arg5 = %c0 to %8 step %c3 {
      scf.for %arg6 = %c0 to %7 step %c4 {
        %11 = memref.subview %3[%arg4, %arg6][%c2, %c4][1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %14 = memref.subview %4[%arg6, %arg5][%c4, %c3][1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        %17 = memref.subview %5[%arg4, %arg5][%c2, %c3][1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        linalg.matmul
          ins(%11, %14: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                        memref<?x?xf32, strided<[?, 1], offset: ?>>)
         outs(%17: memref<?x?xf32, strided<[?, 1], offset: ?>>)
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul_f32(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[vA:.*]] = memref.subview {{.*}} : memref<?x?xf32>
//       CHECK:         %[[vB:.*]] = memref.subview {{.*}} : memref<?x?xf32>
//       CHECK:         %[[vC:.*]] = memref.subview {{.*}} : memref<?x?xf32>
///
//       CHECK:         %[[tmpA:.*]] = memref.alloca() : memref<32xi8>
//       CHECK:         %[[fullA:.*]] = memref.view %[[tmpA]][{{.*}}][{{.*}}] : memref<32xi8> to memref<?x?xf32>
//       CHECK:         %[[partialA:.*]] = memref.subview %[[fullA]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
///
//       CHECK:         %[[tmpB:.*]] = memref.alloca() : memref<48xi8>
//       CHECK:         %[[fullB:.*]] = memref.view %[[tmpB]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf32>
//       CHECK:         %[[partialB:.*]] = memref.subview %[[fullB]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>
///
//       CHECK:         %[[tmpC:.*]] = memref.alloca() : memref<24xi8>
//       CHECK:         %[[fullC:.*]] = memref.view %[[tmpC]][{{.*}}][{{.*}}] : memref<24xi8> to memref<?x?xf32>
//       CHECK:         %[[partialC:.*]] = memref.subview %[[fullC]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1]>>

//       CHECK:         linalg.copy ins(%[[vA]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[partialA]] : memref<?x?xf32, strided<[?, 1]>>)
//       CHECK:         linalg.copy ins(%[[vB]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[partialB]] : memref<?x?xf32, strided<[?, 1]>>)
//       CHECK:         linalg.copy ins(%[[vC]] : memref<?x?xf32, strided<[?, 1], offset: ?>>) outs(%[[partialC]] : memref<?x?xf32, strided<[?, 1]>>)
//
//       CHECK:         linalg.matmul ins(%[[partialA]], %[[partialB]]{{.*}} outs(%[[partialC]]
//
//       CHECK:         linalg.copy ins(%[[partialC]] : memref<?x?xf32, strided<[?, 1]>>) outs(%[[vC]] : memref<?x?xf32, strided<[?, 1], offset: ?>>)
//
//   CHECK-NOT:         memref.dealloc %[[tmpA]] : memref<32xi8>
//   CHECK-NOT:         memref.dealloc %[[tmpB]] : memref<48xi8>
//   CHECK-NOT:         memref.dealloc %[[tmpC]] : memref<24xi8>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { use_alloca } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @matmul_f64(%A: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %3 = memref.view %A[%c0][%M, %K] : memref<?xi8> to memref<?x?xf64>
  %4 = memref.view %A[%c0][%K, %N] : memref<?xi8> to memref<?x?xf64>
  %5 = memref.view %A[%c0][%M, %N] : memref<?xi8> to memref<?x?xf64>
  %6 = memref.dim %3, %c0 : memref<?x?xf64>
  %7 = memref.dim %3, %c1 : memref<?x?xf64>
  %8 = memref.dim %4, %c1 : memref<?x?xf64>
  scf.for %arg4 = %c0 to %6 step %c2 {
    scf.for %arg5 = %c0 to %8 step %c3 {
      scf.for %arg6 = %c0 to %7 step %c4 {
        %11 = memref.subview %3[%arg4, %arg6][%c2, %c4][1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
        %14 = memref.subview %4[%arg6, %arg5][%c4, %c3][1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
        %17 = memref.subview %5[%arg4, %arg5][%c2, %c3][1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
        linalg.matmul
          ins(%11, %14: memref<?x?xf64, strided<[?, 1], offset: ?>>,
                        memref<?x?xf64, strided<[?, 1], offset: ?>>)
         outs(%17: memref<?x?xf64, strided<[?, 1], offset: ?>>)
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul_f64(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[vA_f64:.*]] = memref.subview {{.*}} : memref<?x?xf64>
//       CHECK:         %[[vB_f64:.*]] = memref.subview {{.*}} : memref<?x?xf64>
//       CHECK:         %[[vC_f64:.*]] = memref.subview {{.*}} : memref<?x?xf64>
///
//       CHECK:         %[[tmpA_f64:.*]] = memref.alloc() : memref<64xi8>
//       CHECK:         %[[fullA_f64:.*]] = memref.view %[[tmpA_f64]][{{.*}}][{{.*}}] : memref<64xi8> to memref<?x?xf64>
//       CHECK:         %[[partialA_f64:.*]] = memref.subview %[[fullA_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1]>>
///
//       CHECK:         %[[tmpB_f64:.*]] = memref.alloc() : memref<96xi8>
//       CHECK:         %[[fullB_f64:.*]] = memref.view %[[tmpB_f64]][{{.*}}][{{.*}}] : memref<96xi8> to memref<?x?xf64>
//       CHECK:         %[[partialB_f64:.*]] = memref.subview %[[fullB_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1]>>
///
//       CHECK:         %[[tmpC_f64:.*]] = memref.alloc() : memref<48xi8>
//       CHECK:         %[[fullC_f64:.*]] = memref.view %[[tmpC_f64]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf64>
//       CHECK:         %[[partialC_f64:.*]] = memref.subview %[[fullC_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1]>>

//       CHECK:         linalg.copy ins(%[[vA_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%[[partialA_f64]] : memref<?x?xf64, strided<[?, 1]>>)
//       CHECK:         linalg.copy ins(%[[vB_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%[[partialB_f64]] : memref<?x?xf64, strided<[?, 1]>>)
//       CHECK:         linalg.copy ins(%[[vC_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>>) outs(%[[partialC_f64]] : memref<?x?xf64, strided<[?, 1]>>)
//
//       CHECK:         linalg.matmul ins(%[[partialA_f64]], %[[partialB_f64]]{{.*}} outs(%[[partialC_f64]]
//
//       CHECK:         linalg.copy ins(%[[partialC_f64]] : memref<?x?xf64, strided<[?, 1]>>) outs(%[[vC_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>>)
//
//       CHECK:         memref.dealloc %[[tmpA_f64]] : memref<64xi8>
//       CHECK:         memref.dealloc %[[tmpB_f64]] : memref<96xi8>
//       CHECK:         memref.dealloc %[[tmpC_f64]] : memref<48xi8>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
func.func @gemm_shared(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
   linalg.matmul ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
               outs(%c: memref<?x?xf32>)
   return
}

// CHECK: func @gemm_shared
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK: %[[alloc_A:.*]] = memref.alloc() : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: %[[alloc_B:.*]] = memref.alloc() : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK-DAG: %[[C16:.*]] = arith.constant 16
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[subview_A:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:         %[[subview_B:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:         %[[subview_C:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

// CHECK:         %[[shared_A:.*]] = memref.subview %[[alloc_B]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<16x16xf32, #gpu.address_space<workgroup>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<workgroup>>
// CHECK:         %[[shared_B:.*]] = memref.subview %[[alloc_A]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<16x16xf32, #gpu.address_space<workgroup>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<workgroup>>

// CHECK-NEXT:    gpu.barrier
// CHECK-NEXT:    memref.copy %[[subview_A]], %[[shared_A]] :  memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<workgroup>>
// CHECK-NEXT:    gpu.barrier

// CHECK-NEXT:    gpu.barrier
// CHECK-NEXT:    memref.copy %[[subview_B]], %[[shared_B]] :  memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<workgroup>>
// CHECK-NEXT:    gpu.barrier

// CHECK:         linalg.matmul ins(%[[shared_A]], %[[shared_B]]{{.*}} outs(%[[subview_C]]


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.promote %1 { operands_to_promote = [0, 1], mapping = [#gpu.memory_space<workgroup>] } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}


// -----

func.func @gemm_private(%a : memref<?x?xf32>, %b : memref<?x?xf32>, %c : memref<?x?xf32>)
{
   linalg.matmul ins(%a, %b: memref<?x?xf32>, memref<?x?xf32>)
               outs(%c: memref<?x?xf32>)
   return
}

// CHECK: func @gemm_private
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]+]]: memref<?x?xf32>
// CHECK: %[[alloc_A:.*]] = memref.alloca() : memref<16x16xf32, #gpu.address_space<private>>
// CHECK: %[[alloc_B:.*]] = memref.alloca() : memref<16x16xf32, #gpu.address_space<private>>
// CHECK-DAG: %[[C16:.*]] = arith.constant 16
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[subview_A:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:         %[[subview_B:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:         %[[subview_C:.*]] = memref.subview {{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

// CHECK:         %[[private_A:.*]] = memref.subview %[[alloc_B]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<16x16xf32, #gpu.address_space<private>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<private>>
// CHECK:         %[[private_B:.*]] = memref.subview %[[alloc_A]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<16x16xf32, #gpu.address_space<private>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<private>>

// CHECK-NEXT:    memref.copy %[[subview_A]], %[[private_A]] :  memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<private>>
// CHECK-NEXT:    memref.copy %[[subview_B]], %[[private_B]] :  memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[16, 1]>, #gpu.address_space<private>>

// CHECK:         linalg.matmul ins(%[[private_A]], %[[private_B]]{{.*}} outs(%[[subview_C]]


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 tile_sizes [16, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %2 = transform.structured.promote %1 { operands_to_promote = [0, 1], mapping = [#gpu.memory_space<private>] } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}


// -----

#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: promote_rank_reducing_subviews(%[[arg0:.+]]: memref<{{.*}}>, %[[arg1:.+]]: memref<{{.*}}>, %[[arg2:.+]]: memref<{{.*}}>, %[[lb1:.+]]: index, %[[lb2:.+]]: index, %[[lb3:.+]]: index, %[[lb4:.+]]: index, %[[lb5:.+]]: index, %[[lb6:.+]]: index, %[[ub1:.+]]: index, %[[ub2:.+]]: index
func.func @promote_rank_reducing_subviews(%arg0:  memref<?x?x?x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg1: memref<128x3x3x64xf32, strided<[?, ?, ?, ?], offset: ?>>, %arg2: memref<?x?x?x128xf32>,
                                          %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %ub1: index, %ub2: index) {
  %13 = memref.subview %arg0[%arg3, 0, %arg4, %arg8] [1, 1, %ub1, 32] [1, 1, 1, 1] : memref<?x?x?x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<?x32xf32, strided<[?, ?], offset: ?>>
  %14 = memref.subview %arg1[0, %arg6, %arg7, %arg8] [128, 1, 1, 32] [1, 1, 1, 1] : memref<128x3x3x64xf32, strided<[?, ?, ?, ?], offset: ?>> to memref<128x32xf32, strided<[?, ?], offset: ?>>
  %9 = memref.subview %arg2[%arg3, %arg4, %arg5, 0] [1, 1, %ub2, 128] [1, 1, 1, 1] : memref<?x?x?x128xf32> to memref<?x128xf32, strided<[128, 1], offset: ?>>

  // CHECK: %[[a_alloc:.+]] = memref.alloc
  // CHECK: %[[a_view:.+]] = memref.view %[[a_alloc]]{{.*}}
  // CHECK: %[[a_pro_subview:.+]] = memref.subview %[[a_view]][0, 0] [%[[ub1]], {{.+}}] [1, 1]

  // CHECK: memref.alloc
  // CHECK: %[[b_view:.+]] = memref.view
  // CHECK: %[[b_pro_subview:.+]] = memref.subview %[[b_view]]

  // CHECK: memref.alloc
  // CHECK: %[[c_view:.+]] = memref.view
  // CHECK: %[[c_pro_subview:.+]] = memref.subview %[[c_view]]

  // CHECK-COUNT-3: linalg.copy
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[a_pro_subview]], %[[b_pro_subview]]
  // CHECK-SAME: outs(%[[c_pro_subview]]

  linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x32xf32, strided<[?, ?], offset: ?>>, memref<128x32xf32, strided<[?, ?], offset: ?>>) outs(%9 : memref<?x128xf32, strided<[128, 1], offset: ?>>) {
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %15 = arith.mulf %arg9, %arg10 : f32
    %16 = arith.addf %arg11, %15 : f32
    linalg.yield %16 : f32
  }

  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

  // CHECK-LABEL:   func.func @linalg_generic_update_all_function_inputs_outputs(
  // CHECK-SAME:                                                                 %[[VAL_0:.*]]: memref<3x4xf32, 1>,
  // CHECK-SAME:                                                                 %[[VAL_1:.*]]: memref<3x4xf32, 1>) -> memref<3x4xf32, 1> {
func.func @linalg_generic_update_all_function_inputs_outputs(%arg0: memref<3x4xf32, 1>, %arg1: memref<3x4xf32, 1>) -> memref<3x4xf32, 1> {
  // CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32, 1>
  // CHECK:           %[[VAL_3:.*]] = memref.subview %[[VAL_0]][0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>
  // CHECK:           %[[VAL_4:.*]] = memref.subview %[[VAL_1]][0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>
  // CHECK:           %[[VAL_5:.*]] = memref.subview %[[VAL_2]][0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>

  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32, 1>
  %subview = memref.subview %arg0[0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>
  %subview_0 = memref.subview %arg1[0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>
  %subview_1 = memref.subview %alloc[0, 0] [4, 3] [1, 1] : memref<3x4xf32, 1> to memref<4x3xf32, strided<[4, 1]>, 1>

  // CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_10:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_12:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_14:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_15:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_17:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_20:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_21:.*]] = arith.constant 12 : index
  // CHECK:           %[[VAL_22:.*]] = memref.alloc() : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_23:.*]] = memref.view %[[VAL_22]]{{\[}}%[[VAL_18]]]{{\[}}%[[VAL_12]], %[[VAL_15]]] : memref<48xi8, #gpu.address_space<workgroup>> to memref<?x?xf32, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_24:.*]] = memref.subview %[[VAL_23]][0, 0] {{\[}}%[[VAL_14]], %[[VAL_17]]] [1, 1] : memref<?x?xf32, #gpu.address_space<workgroup>> to memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_25:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_26:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_28:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_29:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_30:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_31:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_32:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_33:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_34:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_35:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_36:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_37:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_38:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_39:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_40:.*]] = arith.constant 12 : index
  // CHECK:           %[[VAL_41:.*]] = memref.alloc() : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_42:.*]] = memref.view %[[VAL_41]]{{\[}}%[[VAL_37]]]{{\[}}%[[VAL_31]], %[[VAL_34]]] : memref<48xi8, #gpu.address_space<workgroup>> to memref<?x?xf32, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_43:.*]] = memref.subview %[[VAL_42]][0, 0] {{\[}}%[[VAL_33]], %[[VAL_36]]] [1, 1] : memref<?x?xf32, #gpu.address_space<workgroup>> to memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_44:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_45:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_46:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_47:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_48:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_49:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_50:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_51:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_52:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_53:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_54:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_55:.*]] = arith.constant 3 : index
  // CHECK:           %[[VAL_56:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_57:.*]] = arith.constant 1 : index
  // CHECK:           %[[VAL_58:.*]] = arith.constant 4 : index
  // CHECK:           %[[VAL_59:.*]] = arith.constant 12 : index
  // CHECK:           %[[VAL_60:.*]] = memref.alloc() : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_61:.*]] = memref.view %[[VAL_60]]{{\[}}%[[VAL_56]]]{{\[}}%[[VAL_50]], %[[VAL_53]]] : memref<48xi8, #gpu.address_space<workgroup>> to memref<?x?xf32, #gpu.address_space<workgroup>>
  // CHECK:           %[[VAL_62:.*]] = memref.subview %[[VAL_61]][0, 0] {{\[}}%[[VAL_52]], %[[VAL_55]]] [1, 1] : memref<?x?xf32, #gpu.address_space<workgroup>> to memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>
// CHECK:           linalg.copy ins(%[[VAL_3]] : memref<4x3xf32, strided<[4, 1]>, 1>) outs(%[[VAL_24]] : memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>)
// CHECK:           linalg.copy ins(%[[VAL_4]] : memref<4x3xf32, strided<[4, 1]>, 1>) outs(%[[VAL_43]] : memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>)
  // CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%[[VAL_24]], %[[VAL_43]] : memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>, memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>) outs(%[[VAL_62]] : memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>) {
  // CHECK:           ^bb0(%[[VAL_63:.*]]: f32, %[[VAL_64:.*]]: f32, %[[VAL_65:.*]]: f32):
  // CHECK:             %[[VAL_66:.*]] = arith.addf %[[VAL_63]], %[[VAL_64]] : f32
  // CHECK:             linalg.yield %[[VAL_66]] : f32
  // CHECK:           }


  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%subview, %subview_0 : memref<4x3xf32, strided<[4, 1]>, 1>, memref<4x3xf32, strided<[4, 1]>, 1>) outs(%subview_1 : memref<4x3xf32, strided<[4, 1]>, 1>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }

  // CHECK:           linalg.copy ins(%[[VAL_62]] : memref<?x?xf32, strided<[?, 1]>, #gpu.address_space<workgroup>>) outs(%[[VAL_5]] : memref<4x3xf32, strided<[4, 1]>, 1>)
  // CHECK:           memref.dealloc %[[VAL_22]] : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           memref.dealloc %[[VAL_41]] : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           memref.dealloc %[[VAL_60]] : memref<48xi8, #gpu.address_space<workgroup>>
  // CHECK:           return %[[VAL_2]] : memref<3x4xf32, 1>
  // CHECK:         }

  return %alloc : memref<3x4xf32, 1>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.promote %0 { memory_space = #gpu.address_space<workgroup> } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
