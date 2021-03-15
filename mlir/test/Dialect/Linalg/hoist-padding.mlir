// RUN: mlir-opt %s -test-linalg-transform-patterns=test-hoist-padding-2-level -canonicalize | FileCheck %s

#map0 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (3, -d0 + s0)>
#map3 = affine_map<(d0, d1) -> (2, d0 - d1)>
#map4 = affine_map<(d0, d1) -> (3, d0 - d1)>

// CHECK-DAG: #[[$DIV3:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 3)>
// CHECK-DAG: #[[$DIV4:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 4)>
// CHECK-DAG: #[[$DIVS3:[0-9a-z]+]] = affine_map<()[s0] -> (s0 ceildiv 3)>
// CHECK-DAG: #[[$DIVS4:[0-9a-z]+]] = affine_map<()[s0] -> (s0 ceildiv 4)>

// CHECK-LABEL: func @matmul_tensors
//  CHECK-SAME:   %[[TA:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[TB:[0-9a-z]+]]: tensor
//  CHECK-SAME:   %[[TC:[0-9a-z]+]]: tensor
func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
  -> tensor<?x?xf32>
{
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %cst = constant 0.000000e+00 : f32

  //  CHECK-DAG: %[[C0:.*]] = constant 0 : index
  //  CHECK-DAG: %[[C1:.*]] = constant 1 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  //  CHECK-DAG: %[[dM:.*]] = memref.dim %[[TA]], %[[C0]] : tensor<?x?xf32>
  //  CHECK-DAG: %[[dK:.*]] = memref.dim %[[TA]], %[[C1]] : tensor<?x?xf32>
  //  CHECK-DAG: %[[dN:.*]] = memref.dim %[[TB]], %[[C1]] : tensor<?x?xf32>
  %0 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = memref.dim %arg1, %c1 : tensor<?x?xf32>

  //      CHECK: scf.for %[[I:[0-9a-z]+]] =
  // First padded tensor is MxKx2x4 under loop M so Kx2x4
  //      CHECK:   %[[SZpad0_K:[0-9]+]] = affine.apply #[[$DIVS4]]()[%[[dK]]]
  //      CHECK:   linalg.init_tensor [%[[SZpad0_K]], 2, 4] : tensor<?x2x4xf32>
  // 1-D loop
  //      CHECK:   %[[A:.*]] = scf.for %[[J1:[0-9a-z]+]] =
  // Iteration count along J1
  //      CHECK:     %[[IDXpad0_K:[0-9]+]] = affine.apply #[[$DIV4]](%[[J1]])
  //      CHECK:     subtensor %{{.*}} [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:     linalg.pad_tensor %{{.*}}
  //      CHECK:       : tensor<?x?xf32> to tensor<2x4xf32>
  //      CHECK:     subtensor_insert %{{.*}} into %{{.*}}[%[[IDXpad0_K]], 0, 0]
  // CHECK-SAME:       [1, 2, 4] [1, 1, 1] : tensor<2x4xf32> into tensor<?x2x4xf32>
  // Second tensor is KxN but loop order is (M, N, K) so padded tensor is NxKx4x3
  //      CHECK:   %[[SZpad1_N:[0-9]+]] = affine.apply #[[$DIVS3]]()[%[[dN]]]
  //      CHECK:   %[[SZpad1_K:[0-9]+]] = affine.apply #[[$DIVS4]]()[%[[dK]]]
  //      CHECK:   linalg.init_tensor [%[[SZpad1_N]], %[[SZpad1_K]], 4, 3] : tensor<?x?x4x3xf32>
  // 2-D loop
  //      CHECK:   %[[B:.*]] = scf.for %[[K2:[0-9a-z]+]] =
  // Iteration count along K2
  //      CHECK:     %[[IDXpad1_K:[0-9]+]] = affine.apply #[[$DIV3]](%[[K2]])
  //      CHECK:     scf.for %[[J2:[0-9a-z]+]] =
  // Iteration count along J2
  //      CHECK:       %[[IDXpad1_N:[0-9]+]] = affine.apply #[[$DIV4]](%[[J2]])
  //      CHECK:       subtensor %{{.*}} [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:       linalg.pad_tensor %{{.*}}
  //      CHECK:         : tensor<?x?xf32> to tensor<4x3xf32>
  //      CHECK:       subtensor_insert %{{.*}} into %{{.*}}[%[[IDXpad1_K]], %[[IDXpad1_N]], 0, 0]
  // CHECK-SAME:         [1, 1, 4, 3] [1, 1, 1, 1] : tensor<4x3xf32> into tensor<?x?x4x3xf32>
  // 2-D loop
  //      CHECK:   scf.for %[[J:[0-9a-zA-Z]+]]
  //      CHECK:     scf.for %[[K:[0-9a-zA-Z]+]]
  // Iteration count along K
  //      CHECK:       %[[IDXpad0_K:[0-9]+]] = affine.apply #[[$DIV4]](%[[K]])
  //      CHECK:       %[[stA:.*]] = subtensor %[[A]][%[[IDXpad0_K]], 0, 0] [1, 2, 4] [1, 1, 1] :
  // CHECK-SAME:         tensor<?x2x4xf32> to tensor<2x4xf32>
  // Iteration count along J
  //      CHECK:       %[[IDXpad1_N:[0-9]+]] = affine.apply #[[$DIV3]](%[[J]])
  // Iteration count along K
  //      CHECK:       %[[IDXpad1_K:[0-9]+]] = affine.apply #[[$DIV4]](%[[K]])
  //      CHECK:       %[[stB:.*]] = subtensor %[[B]][%[[IDXpad1_N]], %[[IDXpad1_K]], 0, 0] [1, 1, 4, 3] [1, 1, 1, 1] :
  // CHECK-SAME:         tensor<?x?x4x3xf32> to tensor<4x3xf32>
  //      CHECK:       %[[stC:.*]] = linalg.pad_tensor %{{.*}}
  //      CHECK:         : tensor<?x?xf32> to tensor<2x3xf32>
  //      CHECK:       linalg.matmul ins(%[[stA]], %[[stB]] : tensor<2x4xf32>, tensor<4x3xf32>)
  // CHECK-SAME:         outs(%[[stC]] : tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = scf.for %arg3 = %c0 to %0 step %c2 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %4 = scf.for %arg5 = %c0 to %2 step %c3 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %5 = scf.for %arg7 = %c0 to %1 step %c4 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = memref.dim %arg0, %c0 : tensor<?x?xf32>
        %7 = affine.min #map0(%arg3)[%6]
        %8 = memref.dim %arg0, %c1 : tensor<?x?xf32>
        %9 = affine.min #map1(%arg7)[%8]
        %10 = subtensor %arg0[%arg3, %arg7] [%7, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %11 = memref.dim %arg1, %c0 : tensor<?x?xf32>
        %12 = affine.min #map1(%arg7)[%11]
        %13 = memref.dim %arg1, %c1 : tensor<?x?xf32>
        %14 = affine.min #map2(%arg5)[%13]
        %15 = subtensor %arg1[%arg7, %arg5] [%12, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %16 = memref.dim %arg8, %c0 : tensor<?x?xf32>
        %17 = affine.min #map3(%16, %arg3)
        %18 = memref.dim %arg8, %c1 : tensor<?x?xf32>
        %19 = affine.min #map4(%18, %arg5)
        %20 = subtensor %arg8[%arg3, %arg5] [%17, %19] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %21 = subi %c2, %7 : index
        %22 = subi %c4, %9 : index
        %23 = linalg.pad_tensor %10 low[%c0, %c0] high[%21, %22] {
        ^bb0(%arg9: index, %arg10: index):  // no predecessors
          linalg.yield %cst : f32
        } : tensor<?x?xf32> to tensor<2x4xf32>
        %24 = subi %c4, %12 : index
        %25 = subi %c3, %14 : index
        %26 = linalg.pad_tensor %15 low[%c0, %c0] high[%24, %25] {
        ^bb0(%arg9: index, %arg10: index):  // no predecessors
          linalg.yield %cst : f32
        } : tensor<?x?xf32> to tensor<4x3xf32>
        %27 = subi %c2, %17 : index
        %28 = subi %c3, %19 : index
        %29 = linalg.pad_tensor %20 low[%c0, %c0] high[%27, %28] {
        ^bb0(%arg9: index, %arg10: index):  // no predecessors
          linalg.yield %cst : f32
        } : tensor<?x?xf32> to tensor<2x3xf32>
        %30 = linalg.matmul ins(%23, %26 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%29 : tensor<2x3xf32>) -> tensor<2x3xf32>
        %31 = subtensor %30[0, 0] [%7, %14] [1, 1] : tensor<2x3xf32> to tensor<?x?xf32>
        %32 = subtensor_insert %31 into %arg8[%arg3, %arg5] [%17, %19] [%c1, %c1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %32 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}
