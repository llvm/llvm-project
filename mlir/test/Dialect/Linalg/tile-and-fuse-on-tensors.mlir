// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul fuse tile-sizes=5,4,7 tile-interchange=1,0,2 run-enable-pass=false" -cse -split-input-file | FileCheck --check-prefix=MATMUL %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.generic fuse tile-sizes=5,4,7 tile-interchange=1,0,2 run-enable-pass=false" -cse -split-input-file | FileCheck --check-prefix=GENERIC %s

//  MATMUL-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (5, -d0 + 24)>
//  MATMUL-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (7, -d0 + 12)>
//  MATMUL-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 24)>
//  MATMUL-DAG:  #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 12)>

//      MATMUL:  fuse_input
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
builtin.func @fuse_input(%arg0: tensor<24x12xf32>,
                         %arg1: tensor<12x25xf32>,
                         %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<24x12xf32> -> tensor<24x12xf32>

  //      MATMUL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      MATMUL:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      MATMUL:      %[[TS1:.*]] = affine.min #[[MAP0]](%[[IV1]])
  //      MATMUL:      scf.for %[[IV2:[0-9a-zA-Z]*]] =
  //      MATMUL:        %[[TS2:.*]] = affine.min #[[MAP1]](%[[IV2]])

  // Tile both input operand dimensions.
  //      MATMUL:        %[[UB1:.*]] = affine.min #[[MAP2]](%[[TS1]], %[[IV1]])
  //      MATMUL:        %[[UB2:.*]] = affine.min #[[MAP3]](%[[TS2]], %[[IV2]])
  //      MATMUL:        %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // MATMUL-SAME:                                          %[[IV1]], %[[IV2]]
  // MATMUL-SAME:                                          %[[UB1]], %[[UB2]]
  //      MATMUL:        %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      MATMUL:        %{{.*}} = linalg.matmul ins(%[[T1]]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//  MATMUL-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (5, -d0 + 24)>
//  MATMUL-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (4, -d0 + 25)>

//      MATMUL:  fuse_output
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
builtin.func @fuse_output(%arg0: tensor<24x12xf32>,
                          %arg1: tensor<12x25xf32>,
                          %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  //  MATMUL-DAG:  %[[C0:.*]] = arith.constant 0 : index
  //  MATMUL-DAG:  %[[C1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg2) : f32, tensor<24x25xf32> -> tensor<24x25xf32>

  // Update the iteration argument of the outermost tile loop.
  //      MATMUL:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG3:.*]] = %[[ARG2]]
  //      MATMUL:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG4:.*]] = %[[ARG3]]
  //      MATMUL:      %[[TS1:.*]] = affine.min #[[MAP0]](%[[IV1]])
  //      MATMUL:      %[[TS0:.*]] = affine.min #[[MAP1]](%[[IV0]])

  // Tile the both output operand dimensions.
  //      MATMUL:      %[[T0:.*]] = tensor.extract_slice %[[ARG4]]
  // MATMUL-SAME:                                        %[[IV1]], %[[IV0]]
  // MATMUL-SAME:                                        %[[TS1]], %[[TS0]]
  //      MATMUL:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      MATMUL:        scf.for %[[IV2:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[T1]]

  // Check there is an extract/insert slice pair for the output operand.
  //  MATMUL-DAG:          %[[D0:.*]] = tensor.dim %[[ARG5]], %[[C0]]
  //  MATMUL-DAG:          %[[D1:.*]] = tensor.dim %[[ARG5]], %[[C1]]
  //      MATMUL:          %[[T2:.*]] = tensor.extract_slice %[[ARG5]]
  // MATMUL-SAME:                                            0, 0
  // MATMUL-SAME:                                            %[[D0]], %[[D1]]
  //      MATMUL:          %[[T3:.*]] = linalg.matmul {{.*}} outs(%[[T2]]
  //      MATMUL:          %{{.*}} = tensor.insert_slice %[[T3]] into %[[ARG5]]
  // MATMUL-SAME:                                            0, 0
  // MATMUL-SAME:                                            %[[D0]], %[[D1]]
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%0 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//  MATMUL-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (4, -d0 + 25)>
//  MATMUL-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (7, -d0 + 12)>
//  MATMUL-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 25)>
//  MATMUL-DAG:  #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 12)>
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>

//      MATMUL:  fuse_reduction
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// MATMUL-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<12x7x25xf32>
builtin.func @fuse_reduction(%arg0: tensor<24x12xf32>,
                             %arg1: tensor<12x25xf32>,
                             %arg2: tensor<24x25xf32>,
                             %arg3: tensor<12x7x25xf32>) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3 : tensor<12x7x25xf32>) outs(%arg1 : tensor<12x25xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg4, %arg5 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>

  //      MATMUL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      MATMUL:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      MATMUL:      %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])
  //      MATMUL:      scf.for %[[IV2:[0-9a-zA-Z]*]] =
  //      MATMUL:        %[[TS2:.*]] = affine.min #[[MAP1]](%[[IV2]])
  //      MATMUL:        %[[UB2:.*]] = affine.min #[[MAP3]](%[[TS2]], %[[IV2]])
  //      MATMUL:        %[[UB0:.*]] = affine.min #[[MAP2]](%[[TS0]], %[[IV0]])

  // Tile only the parallel dimensions but not the reduction dimension.
  //      MATMUL:        %[[T0:.*]] = tensor.extract_slice %[[ARG3]]
  // MATMUL-SAME:                                          %[[IV2]], 0, %[[IV0]]
  // MATMUL-SAME:                                          %[[UB2]], 7, %[[UB0]]
  //      MATMUL:        %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // MATMUL-SAME:                                          %[[IV2]], %[[IV0]]
  // MATMUL-SAME:                                          %[[UB2]], %[[UB0]]
  //      MATMUL:        %[[T2:.*]] = linalg.generic {{.*}} ins(%[[T0]] {{.*}} outs(%[[T1]]
  //      MATMUL:        %{{.*}} = linalg.matmul ins(%{{.*}}, %[[T2]]
  %1 = linalg.matmul ins(%arg0, %0 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

//      MATMUL:  fuse_transposed
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// MATMUL-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<12x24xf32>
builtin.func @fuse_transposed(%arg0: tensor<24x12xf32>,
                              %arg1: tensor<12x25xf32>,
                              %arg2: tensor<24x25xf32>,
                              %arg3: tensor<12x24xf32>) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<12x24xf32>) outs(%arg0 : tensor<24x12xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg4, %arg5 : f32
    linalg.yield %2 : f32
  } -> tensor<24x12xf32>

  //      MATMUL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      MATMUL:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      MATMUL:      scf.for %[[IV2:[0-9a-zA-Z]*]] =

  // Swap the input operand slice offsets due to the transposed indexing map.
  //      MATMUL:        %[[T0:.*]] = tensor.extract_slice %[[ARG3]]
  // MATMUL-SAME:                                          %[[IV2]], %[[IV1]]
  //      MATMUL:        %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // MATMUL-SAME:                                          %[[IV1]], %[[IV2]]
  //      MATMUL:        %[[T2:.*]] = linalg.generic {{.*}} ins(%[[T0]] {{.*}} outs(%[[T1]]
  //      MATMUL:        %{{.*}} = linalg.matmul ins(%[[T2]]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//      MATMUL:  fuse_input_and_output
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
builtin.func @fuse_input_and_output(%arg0: tensor<24x12xf32>,
                                    %arg1: tensor<12x25xf32>,
                                    %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<24x12xf32> -> tensor<24x12xf32>
  %1 = linalg.fill(%cst, %arg2) : f32, tensor<24x25xf32> -> tensor<24x25xf32>

  // Fuse both producers to the appropriate tile loops.
  //      MATMUL:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG3:.*]] = %[[ARG2]]
  //      MATMUL:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG4:.*]] = %[[ARG3]]
  //      MATMUL:      %[[T0:.*]] = tensor.extract_slice %[[ARG4]]
  // MATMUL-SAME:                                        %[[IV1]], %[[IV0]]
  //      MATMUL:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      MATMUL:        scf.for %[[IV2:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[T1]]
  //      MATMUL:          %[[T2:.*]] = tensor.extract_slice %[[ARG0]]
  // MATMUL-SAME:                                            %[[IV1]], %[[IV2]]
  //      MATMUL:          %[[T3:.*]] = linalg.fill(%{{.*}}, %[[T2]])
  //      MATMUL:          %[[T4:.*]] = tensor.extract_slice %[[ARG5]]
  //      MATMUL:          %{{.*}} = linalg.matmul ins(%[[T3]], {{.*}} outs(%[[T4]]
  %2 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%1 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %2 : tensor<24x25xf32>
}

// -----

//  MATMUL-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
#map0 = affine_map<(d0, d1) -> (d1, d0)>

//      MATMUL:  fuse_indexed
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xi32>
builtin.func @fuse_indexed(%arg0: tensor<24x12xi32>,
                           %arg1: tensor<12x25xi32>,
                           %arg2: tensor<24x25xi32>) -> tensor<24x25xi32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c4 = arith.constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%arg1 : tensor<12x25xi32>) {
  ^bb0(%arg3: i32):  
    %6 = linalg.index 0 : index
    %7 = linalg.index 1 : index
    %8 = arith.addi %6, %7 : index
    %9 = arith.index_cast %8 : index to i32
    linalg.yield %9 : i32
  } -> tensor<12x25xi32>

  //      MATMUL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      MATMUL:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      MATMUL:      scf.for %[[IV2:[0-9a-zA-Z]*]] =

  // Shift the indexes by the slice offsets and swap the offsets due to the transposed indexing map.
  //      MATMUL:        %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // MATMUL-SAME:                                          %[[IV2]], %[[IV0]]
  //      MATMUL:  linalg.generic {{.*}} outs(%[[T1]]
  //      MATMUL:  %[[IDX0:.*]] = linalg.index 0
  //      MATMUL:  %[[IDX0_SHIFTED:.*]] = affine.apply #[[MAP0]](%[[IDX0]], %[[IV0]])
  //      MATMUL:  %[[IDX1:.*]] = linalg.index 1
  //      MATMUL:  %[[IDX1_SHIFTED:.*]] = affine.apply #[[MAP0]](%[[IDX1]], %[[IV2]])
  //      MATMUL:  %{{.*}} = arith.addi %[[IDX0_SHIFTED]], %[[IDX1_SHIFTED]]
  %1 = linalg.matmul ins(%arg0, %0 : tensor<24x12xi32>, tensor<12x25xi32>) outs(%arg2 : tensor<24x25xi32>) -> tensor<24x25xi32>
  return %1 : tensor<24x25xi32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>

//      GENERIC:  fuse_outermost_reduction
// GENERIC-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<10x17xf32>
// GENERIC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<10xf32>
func @fuse_outermost_reduction(%arg0: tensor<10x17xf32>,
                               %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<10x17xf32> -> tensor<10x17xf32>

  // Cannot fuse the output fill since the reduction loop is the outermost loop.
  //      GENERIC:      %[[T0:.*]] = linalg.fill(%{{.*}}, %[[ARG1]])
  %1 = linalg.fill(%cst, %arg1) : f32, tensor<10xf32> -> tensor<10xf32>

  //      GENERIC:  scf.for %[[IV0:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG2:.*]] = %[[T0]]
  //      GENERIC:    scf.for %[[IV1:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG3:.*]] = %[[ARG2]]

  // MATMUL the input fill has been fused.
  //      GENERIC:      %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // GENERIC-SAME:                                        %[[IV1]], %[[IV0]]
  //      GENERIC:      %[[T2:.*]] = linalg.fill(%{{.*}}, %[[T1]])
  //      GENERIC:      %[[T3:.*]] = tensor.extract_slice %[[ARG3]]
  // GENERIC-SAME:                                        %[[IV1]]
  //      GENERIC:  linalg.generic {{.*}} ins(%[[T2]] {{.*}} outs(%[[T3]]
  %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<10x17xf32>) outs(%1 : tensor<10xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  
    %3 = arith.addf %arg2, %arg3 : f32
    linalg.yield %3 : f32
  } -> tensor<10xf32>
  return %2 : tensor<10xf32>
}

// -----

//  GENERIC-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
//  GENERIC-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (8, -d0 - d1 + 17)>
//  GENERIC-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, -d1 - d2 + 17)>
#map0 = affine_map<(d0, d1) -> (d0, d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

//      GENERIC:  fuse_non_rectangular
// GENERIC-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<10x17xf32>
func @fuse_non_rectangular(%arg0: tensor<10x17xf32>,
                           %arg1: tensor<10x8xf32>) -> tensor<10x8xf32> {

  //  GENERIC-DAG:  %[[C0:.*]] = arith.constant 0 : index
  //  GENERIC-DAG:  %[[C4:.*]] = arith.constant 4 : index
  //  GENERIC-DAG:  %[[C5:.*]] = arith.constant 5 : index
  //  GENERIC-DAG:  %[[C8:.*]] = arith.constant 8 : index
  //  GENERIC-DAG:  %[[C10:.*]] = arith.constant 10 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<10x17xf32> -> tensor<10x17xf32>

  //      GENERIC:  scf.for %[[IV0:[0-9a-zA-Z]*]] = %[[C0]] to %[[C8]] step %[[C4]]
  //      GENERIC:    scf.for %[[IV1:[0-9a-zA-Z]*]] = %[[C0]] to %[[C10]] step %[[C5]]

  // Compute producer on a hyper rectangular bounding box. Along the second dimenson,
  // the offset is set to the sum of the induction variables, and the upper bound
  // to either 8 (tile size) or 17 (sum of max indices (9+7) then + 1) minus the
  // induction variables.
  //  GENERIC-DAG:      %[[SUM:.*]] = affine.apply #[[MAP0]](%[[IV1]], %[[IV0]]
  //  GENERIC-DAG:      %[[TS1:.*]] = affine.min #[[MAP1]](%[[IV1]], %[[IV0]]
  //  GENERIC-DAG:      %[[UB1:.*]] = affine.min #[[MAP2]](%[[TS1]], %[[IV1]], %[[IV0]]
  //      GENERIC:      %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // GENERIC-SAME:                                        %[[IV1]], %[[SUM]]
  // GENERIC-SAME:                                                , %[[UB1]]
  //      GENERIC:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<10x17xf32>) outs(%arg1 : tensor<10x8xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  
    %2 = arith.addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<10x8xf32>
  return %1 : tensor<10x8xf32>
}
