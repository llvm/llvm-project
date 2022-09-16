// RUN: mlir-opt %s -test-transform-dialect-interpreter -split-input-file | FileCheck %s

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
//       CHECK:         %[[partialA:.*]] = memref.subview %[[fullA]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
///
//       CHECK:         %[[tmpB:.*]] = memref.alloca() : memref<48xi8>
//       CHECK:         %[[fullB:.*]] = memref.view %[[tmpB]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf32>
//       CHECK:         %[[partialB:.*]] = memref.subview %[[fullB]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
///
//       CHECK:         %[[tmpC:.*]] = memref.alloca() : memref<24xi8>
//       CHECK:         %[[fullC:.*]] = memref.view %[[tmpC]][{{.*}}][{{.*}}] : memref<24xi8> to memref<?x?xf32>
//       CHECK:         %[[partialC:.*]] = memref.subview %[[fullC]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>

//       CHECK:         memref.copy %[[vA]], %[[partialA]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//       CHECK:         memref.copy %[[vB]], %[[partialB]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//       CHECK:         memref.copy %[[vC]], %[[partialC]] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
//
//       CHECK:         linalg.matmul ins(%[[partialA]], %[[partialB]]{{.*}} outs(%[[partialC]]
//
//       CHECK:         memref.copy %[[partialC]], %[[vC]] :
//       CHECK:           memref<?x?xf32, strided<[?, 1], offset: ?>> to
//       CHECK:           memref<?x?xf32, strided<[?, 1], offset: ?>>
//
//   CHECK-NOT:         memref.dealloc %[[tmpA]] : memref<32xi8>
//   CHECK-NOT:         memref.dealloc %[[tmpB]] : memref<48xi8>
//   CHECK-NOT:         memref.dealloc %[[tmpC]] : memref<24xi8>

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1 = transform.structured.promote %0 { use_alloca }
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
//       CHECK:         %[[partialA_f64:.*]] = memref.subview %[[fullA_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
///
//       CHECK:         %[[tmpB_f64:.*]] = memref.alloc() : memref<96xi8>
//       CHECK:         %[[fullB_f64:.*]] = memref.view %[[tmpB_f64]][{{.*}}][{{.*}}] : memref<96xi8> to memref<?x?xf64>
//       CHECK:         %[[partialB_f64:.*]] = memref.subview %[[fullB_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>
///
//       CHECK:         %[[tmpC_f64:.*]] = memref.alloc() : memref<48xi8>
//       CHECK:         %[[fullC_f64:.*]] = memref.view %[[tmpC_f64]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf64>
//       CHECK:         %[[partialC_f64:.*]] = memref.subview %[[fullC_f64]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<?x?xf64> to memref<?x?xf64, strided<[?, 1], offset: ?>>

//       CHECK:         memref.copy %[[vA_f64]], %[[partialA_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>> to memref<?x?xf64, strided<[?, 1], offset: ?>>
//       CHECK:         memref.copy %[[vB_f64]], %[[partialB_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>> to memref<?x?xf64, strided<[?, 1], offset: ?>>
//       CHECK:         memref.copy %[[vC_f64]], %[[partialC_f64]] : memref<?x?xf64, strided<[?, 1], offset: ?>> to memref<?x?xf64, strided<[?, 1], offset: ?>>
//
//       CHECK:         linalg.matmul ins(%[[partialA_f64]], %[[partialB_f64]]{{.*}} outs(%[[partialC_f64]]
//
//       CHECK:         memref.copy %[[partialC_f64]], %[[vC_f64]] :
//       CHECK:           memref<?x?xf64, strided<[?, 1], offset: ?>> to
//       CHECK:           memref<?x?xf64, strided<[?, 1], offset: ?>>
//
//       CHECK:         memref.dealloc %[[tmpA_f64]] : memref<64xi8>
//       CHECK:         memref.dealloc %[[tmpB_f64]] : memref<96xi8>
//       CHECK:         memref.dealloc %[[tmpC_f64]] : memref<48xi8>

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
      %1 = transform.structured.promote %0
  }
}


// -----

#map0 = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3 * s4)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
#map5 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: promote_rank_reducing_subviews(%[[arg0:.+]]: memref<{{.*}}>, %[[arg1:.+]]: memref<{{.*}}>, %[[arg2:.+]]: memref<{{.*}}>, %[[lb1:.+]]: index, %[[lb2:.+]]: index, %[[lb3:.+]]: index, %[[lb4:.+]]: index, %[[lb5:.+]]: index, %[[lb6:.+]]: index, %[[ub1:.+]]: index, %[[ub2:.+]]: index
func.func @promote_rank_reducing_subviews(%arg0:  memref<?x?x?x64xf32, #map0>, %arg1: memref<128x3x3x64xf32, #map0>, %arg2: memref<?x?x?x128xf32>,
                                          %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %ub1: index, %ub2: index) {
  %13 = memref.subview %arg0[%arg3, 0, %arg4, %arg8] [1, 1, %ub1, 32] [1, 1, 1, 1] : memref<?x?x?x64xf32, #map0> to memref<?x32xf32, #map5>
  %14 = memref.subview %arg1[0, %arg6, %arg7, %arg8] [128, 1, 1, 32] [1, 1, 1, 1] : memref<128x3x3x64xf32, #map0> to memref<128x32xf32, #map5>
  %9 = memref.subview %arg2[%arg3, %arg4, %arg5, 0] [1, 1, %ub2, 128] [1, 1, 1, 1] : memref<?x?x?x128xf32> to memref<?x128xf32, #map2>

  // CHECK: %[[a_alloc:.+]] = memref.alloc
  // CHECK: %[[a_view:.+]] = memref.view %[[a_alloc]]{{.*}}                 
  // CHECK: %[[a_pro_subview:.+]] = memref.subview %[[a_view]][0, 0] [%[[ub1]], {{.+}}] [1, 1]

  // CHECK: memref.alloc
  // CHECK: %[[b_view:.+]] = memref.view
  // CHECK: %[[b_pro_subview:.+]] = memref.subview %[[b_view]]

  // CHECK: memref.alloc
  // CHECK: %[[c_view:.+]] = memref.view
  // CHECK: %[[c_pro_subview:.+]] = memref.subview %[[c_view]]

  // CHECK-COUNT-3: memref.copy
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[a_pro_subview]], %[[b_pro_subview]]
  // CHECK-SAME: outs(%[[c_pro_subview]]

  linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%13, %14 : memref<?x32xf32, #map5>, memref<128x32xf32, #map5>) outs(%9 : memref<?x128xf32, #map2>) {
  ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
    %15 = arith.mulf %arg9, %arg10 : f32
    %16 = arith.addf %arg11, %15 : f32
    linalg.yield %16 : f32
  }

  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match interface{LinalgOp} in %arg1
      %1 = transform.structured.promote %0
  }
}
