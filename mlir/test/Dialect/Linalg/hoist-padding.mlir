// RUN: mlir-opt %s -test-linalg-transform-patterns=test-hoist-padding-2-level -canonicalize | FileCheck %s

#map0 = affine_map<(d0)[s0] -> (2, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (3, -d0 + s0)>
#map3 = affine_map<(d0, d1) -> (2, d0 - d1)>
#map4 = affine_map<(d0, d1) -> (3, d0 - d1)>

// CHECK-LABEL: func @matmul_tensors
func @matmul_tensors(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
  -> tensor<?x?xf32>
{
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %cst = constant 0.000000e+00 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = dim %arg0, %c0 : tensor<?x?xf32>
  %1 = dim %arg0, %c1 : tensor<?x?xf32>
  %2 = dim %arg1, %c1 : tensor<?x?xf32>

  //      CHECK: scf.for
  //      CHECK:   linalg.init_tensor [%{{.*}}, 2, 4] : tensor<?x2x4xf32>
  // 1-D loop
  //      CHECK:   %[[A:.*]] = scf.for
  //  CHECK-NOT:     scf.for
  //      CHECK:     subtensor %{{.*}} [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:     linalg.simple_pad %{{.*}} : tensor<?x?xf32> to tensor<2x4xf32> pad f32
  //      CHECK:     subtensor_insert %{{.*}} into %{{.*}}[%{{.*}}, 0, 0]
  // CHECK-SAME:       [1, 2, 4] [1, 1, 1] : tensor<2x4xf32> into tensor<?x2x4xf32>
  // 2-D loop
  //      CHECK:   linalg.init_tensor [%{{.*}}, %{{.*}}, 4, 3] : tensor<?x?x4x3xf32>
  //      CHECK:   %[[B:.*]] = scf.for
  //      CHECK:     scf.for
  //  CHECK-NOT:       scf.for
  //      CHECK:       subtensor %{{.*}} [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:       linalg.simple_pad %{{.*}} : tensor<?x?xf32> to tensor<4x3xf32> pad f32
  //      CHECK:       subtensor_insert %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}, 0, 0]
  // CHECK-SAME:         [1, 1, 4, 3] [1, 1, 1, 1] : tensor<4x3xf32> into tensor<?x?x4x3xf32>
  // 2-D loop
  //      CHECK:   scf.for %[[J:[0-9a-zA-Z]+]]
  //      CHECK:     scf.for %[[K:[0-9a-zA-Z]+]]
  //  CHECK-NOT:       scf.for
  //      CHECK:       %[[stA:.*]] = subtensor %[[A]][%[[K]], 0, 0] [1, 2, 4] [1, 1, 1] :
  // CHECK-SAME:         tensor<?x2x4xf32> to tensor<2x4xf32>
  //      CHECK:       %[[stB:.*]] = subtensor %[[B]][%[[K]], %[[J]], 0, 0] [1, 1, 4, 3] [1, 1, 1, 1] :
  // CHECK-SAME:         tensor<?x?x4x3xf32> to tensor<4x3xf32>
  //      CHECK:       %[[stC:.*]] = linalg.simple_pad %{{.*}} pad %{{.*}} :
  // CHECK-SAME:         tensor<?x?xf32> to tensor<2x3xf32> pad f32
  //      CHECK:       linalg.matmul ins(%[[stA]], %[[stB]] : tensor<2x4xf32>, tensor<4x3xf32>)
  // CHECK-SAME:         outs(%[[stC]] : tensor<2x3xf32>) -> tensor<2x3xf32>
  %3 = scf.for %arg3 = %c0 to %0 step %c2 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {
    %4 = scf.for %arg5 = %c0 to %2 step %c3 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {
      %5 = scf.for %arg7 = %c0 to %1 step %c4 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = dim %arg0, %c0 : tensor<?x?xf32>
        %7 = affine.min #map0(%arg3)[%6]
        %8 = dim %arg0, %c1 : tensor<?x?xf32>
        %9 = affine.min #map1(%arg7)[%8]
        %10 = subtensor %arg0[%arg3, %arg7] [%7, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %11 = dim %arg1, %c0 : tensor<?x?xf32>
        %12 = affine.min #map1(%arg7)[%11]
        %13 = dim %arg1, %c1 : tensor<?x?xf32>
        %14 = affine.min #map2(%arg5)[%13]
        %15 = subtensor %arg1[%arg7, %arg5] [%12, %14] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %16 = dim %arg8, %c0 : tensor<?x?xf32>
        %17 = affine.min #map3(%16, %arg3)
        %18 = dim %arg8, %c1 : tensor<?x?xf32>
        %19 = affine.min #map4(%18, %arg5)
        %20 = subtensor %arg8[%arg3, %arg5] [%17, %19] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %21 = linalg.simple_pad %10 pad %cst : tensor<?x?xf32> to tensor<2x4xf32> pad f32
        %22 = linalg.simple_pad %15 pad %cst : tensor<?x?xf32> to tensor<4x3xf32> pad f32
        %23 = linalg.simple_pad %20 pad %cst : tensor<?x?xf32> to tensor<2x3xf32> pad f32
        %24 = linalg.matmul ins(%21, %22 : tensor<2x4xf32>, tensor<4x3xf32>) outs(%23 : tensor<2x3xf32>) -> tensor<2x3xf32>
        %25 = subtensor %24[0, 0] [%7, %14] [1, 1] : tensor<2x3xf32> to tensor<?x?xf32>
        %26 = subtensor_insert %25 into %arg8[%arg3, %arg5] [%17, %19] [%c1, %c1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %26 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}
