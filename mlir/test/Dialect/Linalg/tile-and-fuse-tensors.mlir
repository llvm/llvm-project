// RUN: mlir-opt %s -linalg-fusion -split-input-file | FileCheck %s
// RUN: mlir-opt %s -linalg-fusion -canonicalize -cse -split-input-file | FileCheck %s --check-prefix=CANONICALIZED

#map0 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (3, -d0 + s0)>
#map3 = affine_map<(d0, d1) -> (2, d0 - d1)>
#map4 = affine_map<(d0, d1) -> (3, d0 - d1)>

func @matmul_tensors(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %t0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     init(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  %c4 = constant 4 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %0 = dim %t0, %c0 : tensor<?x?xf32>
  %1 = dim %t0, %c1 : tensor<?x?xf32>
  %2 = dim %arg1, %c1 : tensor<?x?xf32>
  %3 = scf.for %arg3 = %c0 to %0 step %c2 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %4 = scf.for %arg5 = %c0 to %2 step %c3 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %5 = scf.for %arg7 = %c0 to %1 step %c4 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = subtensor %t0[%arg3, %arg7][%c2, 4][1, 1] : tensor<?x?xf32> to tensor<?x4xf32>
        %7 = subtensor %arg1[%arg7, %arg5][4, %c3][1, 1] : tensor<?x?xf32> to tensor<4x?xf32>
        %8 = subtensor %arg8[%arg3, %arg5][%c2, %c3][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %9 = linalg.matmul ins(%6, %7 : tensor<?x4xf32>, tensor<4x?xf32>) init(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %10 = subtensor_insert %9 into %arg8[%arg3, %arg5] [%c2, %c3] [1, 1]  : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %10 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

// CHECK-LABEL: func @matmul_tensors(
//  CHECK-SAME: %[[A:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[B:[0-9a-z]*]]: tensor<?x?xf32>
//  CHECK-SAME: %[[C:[0-9a-z]*]]: tensor<?x?xf32>
//       CHECK: %[[C0:.*]] = constant 0 : index
//       CHECK: scf.for %[[I:[0-9a-z]*]]
//  CHECK-NEXT:   scf.for %[[J:[0-9a-z]*]]
//  CHECK-NEXT:     scf.for %[[K:[0-9a-z]*]]
//
// subtensor of the original program, first one refers to the unfused matmul and becomes a dead SSA value.
//       CHECK:     subtensor %{{.*}}[%[[I]], %[[K]]] {{.*}} : tensor<?x?xf32> to tensor<?x4xf32>
//       CHECK:     %[[stB1:.*]] = subtensor %[[B]][%[[K]], %[[J]]] {{.*}} : tensor<?x?xf32> to tensor<4x?xf32>
//       CHECK:     %[[stF:.*]] = subtensor %{{.*}}[%[[I]], %[[J]]] {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
//
// subtensors of the producing matmul.
//       CHECK:     %[[stA:.*]] = subtensor %[[A]][%[[I]], %[[C0]]] {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
//  CHECK-NEXT:     %[[stB2:.*]] = subtensor %[[B]][%[[C0]], %[[K]]] {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
//  CHECK-NEXT:     %[[stC:.*]] = subtensor %[[C]][%[[I]], %[[K]]] {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
//  CHECK-NEXT:     %[[stD:.*]] = linalg.matmul ins(%[[stA]], %[[stB2]] : tensor<?x?xf32>, tensor<?x?xf32>) init(%[[stC]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//  CHECK-NEXT:     %[[stD2:.*]] = tensor_cast %[[stD]] : tensor<?x?xf32> to tensor<?x4xf32>
//  CHECK-NEXT:     %[[stG:.*]] = linalg.matmul ins(%[[stD2]], %[[stB1]] : tensor<?x4xf32>, tensor<4x?xf32>) init(%[[stF]] : tensor<?x?xf32>)  -> tensor<?x?xf32>
//  CHECK-NEXT:     subtensor_insert %[[stG]]


// CANONICALIZED-LABEL: func @matmul_tensors(
//  CANONICALIZED-SAME: %[[A:[0-9a-z]*]]: tensor<?x?xf32>
//  CANONICALIZED-SAME: %[[B:[0-9a-z]*]]: tensor<?x?xf32>
//  CANONICALIZED-SAME: %[[C:[0-9a-z]*]]: tensor<?x?xf32>
//       CANONICALIZED: %[[C0:.*]] = constant 0 : index
//       CANONICALIZED: %[[C1:.*]] = constant 1 : index
//       CANONICALIZED: scf.for %[[I:[0-9a-z]*]]
//  CANONICALIZED-NEXT:   scf.for %[[J:[0-9a-z]*]]
//  CANONICALIZED-NEXT:     scf.for %[[K:[0-9a-z]*]]
//
//       CANONICALIZED:     %[[stB1:.*]] = subtensor %[[B]][%[[K]], %[[J]]] [4, 3] [1, 1]  : tensor<?x?xf32> to tensor<4x3xf32>
//       CANONICALIZED:     %[[stF:.*]] = subtensor %{{.*}}[%[[I]], %[[J]]] [2, 3] [1, 1]  : tensor<?x?xf32> to tensor<2x3xf32>
//
// subtensors of the producing matmul.
//       CANONICALIZED:     %[[dA1:.*]] = dim %[[A]], %[[C1]] : tensor<?x?xf32>
//       CANONICALIZED:     %[[stA:.*]] = subtensor %[[A]][%[[I]], 0] [2, %[[dA1]]] [1, 1]  : tensor<?x?xf32> to tensor<2x?xf32>
//  CANONICALIZED-NEXT:     %[[stB2:.*]] = subtensor %[[B]][0, %[[K]]] [%[[dA1]], 4] [1, 1]  : tensor<?x?xf32> to tensor<?x4xf32>
//  CANONICALIZED-NEXT:     %[[stC:.*]] = subtensor %[[C]][%[[I]], %[[K]]] [2, 4] [1, 1]  : tensor<?x?xf32> to tensor<2x4xf32>
//  CANONICALIZED-NEXT:     %[[stD:.*]] = linalg.matmul ins(%[[stA]], %[[stB2]] : tensor<2x?xf32>, tensor<?x4xf32>) init(%[[stC]] : tensor<2x4xf32>)  -> tensor<2x4xf32>
//  CANONICALIZED-NEXT:     %[[stG:.*]] = linalg.matmul ins(%[[stD]], %[[stB1]] : tensor<2x4xf32>, tensor<4x3xf32>) init(%[[stF]] : tensor<2x3xf32>)  -> tensor<2x3xf32>
//  CANONICALIZED-NEXT:     subtensor_insert %[[stG]]
