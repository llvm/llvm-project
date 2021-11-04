// RUN: mlir-opt %s -linalg-tile-and-fuse-tensor-ops="tile-sizes=4,4,0,0 tile-interchange=0,1,2,3" -cse --canonicalize -split-input-file | FileCheck %s

//      CHECK:  fuse_conv_chain
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<2x2xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<11x11xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<10x10xf32>
// CHECK-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<9x9xf32>
// CHECK-SAME:    %[[ARG4:[0-9a-zA-Z]*]]: tensor<8x8xf32>
builtin.func @fuse_conv_chain(%arg0: tensor<2x2xf32>,
                              %arg1: tensor<11x11xf32>,
                              %arg2: tensor<10x10xf32>,
                              %arg3: tensor<9x9xf32>,
                              %arg4: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 1.0 : f32

  // Do not tile the filter fill since the filter dimensions are not tiled.
  //      CHECK:  %[[T0:.*]] = linalg.fill(%{{.*}}, %[[ARG0]])
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<2x2xf32> -> tensor<2x2xf32>

  // Fuse all other operations.
  //      CHECK:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[ARG4]]
  //      CHECK:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG6:.*]] = %[[ARG5]]

  //      CHECK:          %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // CHECK-SAME:                                            %[[IV0]], %[[IV1]]
  //      CHECK:          %[[T2:.*]] = tensor.extract_slice %[[ARG2]]
  // CHECK-SAME:                                            %[[IV0]], %[[IV1]]
  //      CHECK:          %[[T3:.*]] = linalg.fill(%{{.*}}, %[[T2]])
  //      CHECK:          %[[T4:.*]] = linalg.conv_2d ins(%[[T1]], %[[T0]] : {{.*}} outs(%[[T3]]
  %1 = linalg.fill(%cst, %arg2) : f32, tensor<10x10xf32> -> tensor<10x10xf32>
  %2 = linalg.conv_2d ins(%arg1, %0 : tensor<11x11xf32>, tensor<2x2xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>

  //      CHECK:          %[[T5:.*]] = tensor.extract_slice %[[ARG3]]
  // CHECK-SAME:                                            %[[IV0]], %[[IV1]]
  //      CHECK:          %[[T6:.*]] = linalg.fill(%{{.*}}, %[[T5]])
  //      CHECK:          %[[T7:.*]] = linalg.conv_2d ins(%[[T4]], %[[T0]] : {{.*}} outs(%[[T6]]
  %3 = linalg.fill(%cst, %arg3) : f32, tensor<9x9xf32> -> tensor<9x9xf32>
  %4 = linalg.conv_2d ins(%2, %0 : tensor<10x10xf32>, tensor<2x2xf32>) outs(%3 : tensor<9x9xf32>) -> tensor<9x9xf32>

  // Use the argument passed in by iteration argument.
  //      CHECK:          %[[T8:.*]] = tensor.extract_slice %[[ARG6]]
  // CHECK-SAME:                                            %[[IV0]], %[[IV1]]
  //      CHECK:          %[[T9:.*]] = linalg.fill(%{{.*}}, %[[T8]])
  //      CHECK:          %[[T5:.*]] = linalg.conv_2d ins(%[[T7]], %[[T0]] {{.*}} outs(%[[T9]]
  %5 = linalg.fill(%cst, %arg4) : f32, tensor<8x8xf32> -> tensor<8x8xf32>
  %6 = linalg.conv_2d ins(%4, %0 : tensor<9x9xf32>, tensor<2x2xf32>) outs(%5 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// -----

//      CHECK:  fuse_matmul_chain
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<8x8xf32>
builtin.func @fuse_matmul_chain(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32

  // Do not tile rhs fill of the producer matmul since none of its loop dimension is tiled.
  //      CHECK:  %[[T0:.*]] = linalg.fill(%{{.*}}, %[[ARG0]])
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<8x8xf32> -> tensor<8x8xf32>

  //      CHECK:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG1:.*]] = %[[ARG0]]
  //      CHECK:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG2:.*]] = %[[ARG1]]

  // Only the outermost loop of the producer matmul is tiled.
  //      CHECK:      %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME:                                        %[[IV0]], 0
  //      CHECK:      %[[T2:.*]] = linalg.fill(%{{.*}}, %[[T1]])
  //      CHECK:      %[[T3:.*]] = linalg.matmul ins(%[[T2]], %[[T0]] {{.*}}
  %1 = linalg.matmul ins(%0, %0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>

  // Use the argument passed in by iteration argument.
  //      CHECK:      %[[T4:.*]] = tensor.extract_slice %[[ARG2]]
  // CHECK-SAME:                                        %[[IV0]], %[[IV1]]
  //      CHECK:      %[[T5:.*]] = linalg.fill(%{{.*}}, %[[T4]])
  //      CHECK:      %{{.*}} = linalg.matmul ins(%[[T3]], {{.*}} outs(%[[T5]]
  %2 = linalg.matmul ins(%1, %0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
