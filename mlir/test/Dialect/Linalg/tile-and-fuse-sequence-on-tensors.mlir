// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.conv_2d fuse tile-sizes=4,4,0,0 tile-interchange=0,1,2,3 run-enable-pass=false" -split-input-file | FileCheck --check-prefix=CONV %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul fuse tile-sizes=4,4,0 tile-interchange=0,1,2 run-enable-pass=false" -split-input-file | FileCheck --check-prefix=MATMUL %s

//      CONV:  fuse_conv_chain
// CONV-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<2x2xf32>
// CONV-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<11x11xf32>
// CONV-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<10x10xf32>
// CONV-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<9x9xf32>
// CONV-SAME:    %[[ARG4:[0-9a-zA-Z]*]]: tensor<8x8xf32>
func.func @fuse_conv_chain(%arg0: tensor<2x2xf32>,
                              %arg1: tensor<11x11xf32>,
                              %arg2: tensor<10x10xf32>,
                              %arg3: tensor<9x9xf32>,
                              %arg4: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 1.0 : f32

  // Do not tile the filter fill since the filter dimensions are not tiled.
  //      CONV:  %[[T0:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[ARG0]]
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>

  // Fuse all other operations.
  //      CONV:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[ARG4]]
  //      CONV:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG6:.*]] = %[[ARG5]]

  //      CONV:          %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // CONV-SAME:                                            %[[IV0]], %[[IV1]]
  //      CONV:          %[[T2:.*]] = tensor.extract_slice %[[ARG2]]
  // CONV-SAME:                                            %[[IV0]], %[[IV1]]
  //      CONV:          %[[T3:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T2]]
  //      CONV:          %[[T4:.*]] = linalg.conv_2d ins(%[[T1]], %[[T0]] : {{.*}} outs(%[[T3]]
  %1 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = linalg.conv_2d ins(%arg1, %0 : tensor<11x11xf32>, tensor<2x2xf32>) outs(%1 : tensor<10x10xf32>) -> tensor<10x10xf32>

  //      CONV:          %[[T5:.*]] = tensor.extract_slice %[[ARG3]]
  // CONV-SAME:                                            %[[IV0]], %[[IV1]]
  //      CONV:          %[[T6:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T5]]
  //      CONV:          %[[T7:.*]] = linalg.conv_2d ins(%[[T4]], %[[T0]] : {{.*}} outs(%[[T6]]
  %3 = linalg.fill ins(%cst : f32) outs(%arg3 : tensor<9x9xf32>) -> tensor<9x9xf32>
  %4 = linalg.conv_2d ins(%2, %0 : tensor<10x10xf32>, tensor<2x2xf32>) outs(%3 : tensor<9x9xf32>) -> tensor<9x9xf32>

  // Use the argument passed in by iteration argument.
  //      CONV:          %[[T8:.*]] = tensor.extract_slice %[[ARG6]]
  // CONV-SAME:                                            %[[IV0]], %[[IV1]]
  //      CONV:          %[[T9:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T8]]
  //      CONV:          %[[T5:.*]] = linalg.conv_2d ins(%[[T7]], %[[T0]] {{.*}} outs(%[[T9]]
  %5 = linalg.fill ins(%cst : f32) outs(%arg4 : tensor<8x8xf32>) -> tensor<8x8xf32>
  %6 = linalg.conv_2d ins(%4, %0 : tensor<9x9xf32>, tensor<2x2xf32>) outs(%5 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %6 : tensor<8x8xf32>
}

// -----

//      MATMUL:  fuse_matmul_chain
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<8x8xf32>
func.func @fuse_matmul_chain(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32

  // Do not tile rhs fill of the producer matmul since none of its loop dimension is tiled.
  //      MATMUL:  %[[T0:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[ARG0]]
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32>

  //      MATMUL:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG1:.*]] = %[[ARG0]]
  //      MATMUL:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG2:.*]] = %[[ARG1]]

  // Only the outermost loop of the producer matmul is tiled.
  //      MATMUL:      %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // MATMUL-SAME:                                        %[[IV0]], 0
  //      MATMUL:      %[[T2:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T1]]
  //      MATMUL:      %[[T3:.*]] = linalg.matmul ins(%[[T2]], %[[T0]] {{.*}}
  %1 = linalg.matmul ins(%0, %0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>

  // Use the argument passed in by iteration argument.
  //      MATMUL:      %[[T4:.*]] = tensor.extract_slice %[[ARG2]]
  // MATMUL-SAME:                                        %[[IV0]], %[[IV1]]
  //      MATMUL:      %[[T5:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T4]]
  //      MATMUL:      %{{.*}} = linalg.matmul ins(%[[T3]], {{.*}} outs(%[[T5]]
  %2 = linalg.matmul ins(%1, %0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}
