// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#binary2Dpointwise = {
  indexing_maps = [#map0, #map0, #map0],
  iterator_types = ["parallel", "parallel"]
}
#ternary2Dpointwise = {
  indexing_maps = [#map0, #map0, #map0, #map0],
  iterator_types = ["parallel", "parallel"]
}
func @test_fusion_limit(
    %arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>,
    %arg3 : tensor<?x?xf32>, %arg4 : tensor<?x?xf32>, %arg5 : tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %0 = linalg.generic #binary2Dpointwise
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32, %arg8 : f32):
       %1 = arith.mulf %arg6, %arg7 : f32
       linalg.yield %1 : f32
    } -> tensor<?x?xf32>
  %2 = linalg.generic #binary2Dpointwise
      ins(%arg2, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32, %arg8 : f32):
       %3 = arith.mulf %arg6, %arg7 : f32
       linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  %4 = linalg.generic #binary2Dpointwise
      ins(%arg4, %arg5 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32, %arg8 : f32):
       %5 = arith.mulf %arg6, %arg7 : f32
       linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  %6 = linalg.generic #ternary2Dpointwise
      ins(%0, %2, %4 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32, %arg8 : f32, %arg9 : f32):
       %7 = arith.addf %arg6, %arg7 : f32
       %8 = arith.addf %7, %arg8 : f32
       linalg.yield %8 : f32
    } -> tensor<?x?xf32>
  return %6 : tensor<?x?xf32>
}
// CHECK-LABEL: func @test_fusion_limit
//  CHECK-SAME:   %[[ARG0:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG3:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG4:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG5:[a-zA-z0-9_]+]]: tensor<?x?xf32>
//       CHECK:   %[[OP1:.+]] = linalg.generic {{.+}} ins(%[[ARG2]], %[[ARG3]]
//       CHECK:   %[[OP2:.+]] = linalg.generic {{.+}} ins(%[[ARG4]], %[[ARG5]]
//       CHECK:   %[[OP3:.+]] = linalg.generic {{.+}} ins(%[[ARG0]], %[[ARG1]], %[[OP1]], %[[OP2]]
//       CHECK:   return %[[OP3]]
