// RUN: mlir-opt %s -linalg-tile-and-fuse-tensor-ops="tile-sizes=5,4,7 tile-interchange=1,0,2" -cse -split-input-file | FileCheck %s

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (5, -d0 + 24)>
//  CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (7, -d0 + 12)>
//  CHECK-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 24)>
//  CHECK-DAG:  #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 12)>

//      CHECK:  fuse_input
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
builtin.func @fuse_input(%arg0: tensor<24x12xf32>,
                         %arg1: tensor<12x25xf32>,
                         %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<24x12xf32> -> tensor<24x12xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      CHECK:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      CHECK:      %[[TS1:.*]] = affine.min #[[MAP0]](%[[IV1]])
  //      CHECK:      scf.for %[[IV2:[0-9a-zA-Z]*]] =
  //      CHECK:        %[[TS2:.*]] = affine.min #[[MAP1]](%[[IV2]])

  // Tile both input operand dimensions.
  //      CHECK:        %[[UB1:.*]] = affine.min #[[MAP2]](%[[TS1]], %[[IV1]])
  //      CHECK:        %[[UB2:.*]] = affine.min #[[MAP3]](%[[TS2]], %[[IV2]])
  //      CHECK:        %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME:                                          %[[IV1]], %[[IV2]]
  // CHECK-SAME:                                          %[[UB1]], %[[UB2]]
  //      CHECK:        %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      CHECK:        %{{.*}} = linalg.matmul ins(%[[T1]]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (5, -d0 + 24)>
//  CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (4, -d0 + 25)>

//      CHECK:  fuse_output
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
builtin.func @fuse_output(%arg0: tensor<24x12xf32>,
                          %arg1: tensor<12x25xf32>,
                          %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg2) : f32, tensor<24x25xf32> -> tensor<24x25xf32>

  // Update the iteration argument of the outermost tile loop.
  //      CHECK:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG3:.*]] = %[[ARG2]]
  //      CHECK:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG4:.*]] = %[[ARG3]]
  //      CHECK:      %[[TS1:.*]] = affine.min #[[MAP0]](%[[IV1]])
  //      CHECK:      %[[TS0:.*]] = affine.min #[[MAP1]](%[[IV0]])

  // Tile the both output operand dimensions.
  //      CHECK:      %[[T0:.*]] = tensor.extract_slice %[[ARG4]]
  // CHECK-SAME:                                        %[[IV1]], %[[IV0]]
  // CHECK-SAME:                                        %[[TS1]], %[[TS0]]
  //      CHECK:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      CHECK:        scf.for %[[IV2:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[T1]]
  //      CHECK:          %{{.*}} = linalg.matmul {{.*}} outs(%[[ARG5]]
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%0 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0) -> (4, -d0 + 25)>
//  CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0) -> (7, -d0 + 12)>
//  CHECK-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 25)>
//  CHECK-DAG:  #[[MAP3:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 12)>
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>

//      CHECK:  fuse_reduction
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// CHECK-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<12x7x25xf32>
builtin.func @fuse_reduction(%arg0: tensor<24x12xf32>,
                             %arg1: tensor<12x25xf32>,
                             %arg2: tensor<24x25xf32>,
                             %arg3: tensor<12x7x25xf32>) -> tensor<24x25xf32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3 : tensor<12x7x25xf32>) outs(%arg1 : tensor<12x25xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
    %2 = addf %arg4, %arg5 : f32
    linalg.yield %2 : f32
  } -> tensor<12x25xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      CHECK:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      CHECK:      %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])
  //      CHECK:      scf.for %[[IV2:[0-9a-zA-Z]*]] =
  //      CHECK:        %[[TS2:.*]] = affine.min #[[MAP1]](%[[IV2]])
  //      CHECK:        %[[UB2:.*]] = affine.min #[[MAP3]](%[[TS2]], %[[IV2]])
  //      CHECK:        %[[UB0:.*]] = affine.min #[[MAP2]](%[[TS0]], %[[IV0]])

  // Tile only the parallel dimensions but not the reduction dimension.
  //      CHECK:        %[[T0:.*]] = tensor.extract_slice %[[ARG3]]
  // CHECK-SAME:                                          %[[IV2]], 0, %[[IV0]]
  // CHECK-SAME:                                          %[[UB2]], 7, %[[UB0]]
  //      CHECK:        %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // CHECK-SAME:                                          %[[IV2]], %[[IV0]]
  // CHECK-SAME:                                          %[[UB2]], %[[UB0]]
  //      CHECK:        %[[T2:.*]] = linalg.generic {{.*}} ins(%[[T0]] {{.*}} outs(%[[T1]]
  //      CHECK:        %{{.*}} = linalg.matmul ins(%{{.*}}, %[[T2]]
  %1 = linalg.matmul ins(%arg0, %0 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

//      CHECK:  fuse_transposed
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: tensor<12x24xf32>
builtin.func @fuse_transposed(%arg0: tensor<24x12xf32>,
                              %arg1: tensor<12x25xf32>,
                              %arg2: tensor<24x25xf32>,
                              %arg3: tensor<12x24xf32>) -> tensor<24x25xf32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<12x24xf32>) outs(%arg0 : tensor<24x12xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
    %2 = addf %arg4, %arg5 : f32
    linalg.yield %2 : f32
  } -> tensor<24x12xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      CHECK:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      CHECK:      scf.for %[[IV2:[0-9a-zA-Z]*]] =

  // Swap the input operand slice offsets due to the transposed indexing map.
  //      CHECK:        %[[T0:.*]] = tensor.extract_slice %[[ARG3]]
  // CHECK-SAME:                                          %[[IV2]], %[[IV1]]
  //      CHECK:        %[[T1:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME:                                          %[[IV1]], %[[IV2]]
  //      CHECK:        %[[T2:.*]] = linalg.generic {{.*}} ins(%[[T0]] {{.*}} outs(%[[T1]]
  //      CHECK:        %{{.*}} = linalg.matmul ins(%[[T2]]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%arg2 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %1 : tensor<24x25xf32>
}

// -----

//      CHECK:  fuse_input_and_output
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
builtin.func @fuse_input_and_output(%arg0: tensor<24x12xf32>,
                                    %arg1: tensor<12x25xf32>,
                                    %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<24x12xf32> -> tensor<24x12xf32>
  %1 = linalg.fill(%cst, %arg2) : f32, tensor<24x25xf32> -> tensor<24x25xf32>

  // Fuse both producers to the appropriate tile loops.
  //      CHECK:  scf.for %[[IV0:.*]] = {{.*}} iter_args(%[[ARG3:.*]] = %[[ARG2]]
  //      CHECK:    scf.for %[[IV1:.*]] = {{.*}} iter_args(%[[ARG4:.*]] = %[[ARG3]]
  //      CHECK:      %[[T0:.*]] = tensor.extract_slice %[[ARG4]]
  // CHECK-SAME:                                        %[[IV1]], %[[IV0]]
  //      CHECK:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  //      CHECK:        scf.for %[[IV2:.*]] = {{.*}} iter_args(%[[ARG5:.*]] = %[[T1]]
  //      CHECK:          %[[T2:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME:                                            %[[IV1]], %[[IV2]]
  //      CHECK:          %[[T3:.*]] = linalg.fill(%{{.*}}, %[[T2]])
  //      CHECK:          %{{.*}} = linalg.matmul ins(%[[T3]], {{.*}} outs(%[[ARG5]]
  %2 = linalg.matmul ins(%0, %arg1 : tensor<24x12xf32>, tensor<12x25xf32>) outs(%1 : tensor<24x25xf32>) -> tensor<24x25xf32>
  return %2 : tensor<24x25xf32>
}

// -----

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
#map0 = affine_map<(d0, d1) -> (d1, d0)>

//      CHECK:  fuse_indexed
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xi32>
builtin.func @fuse_indexed(%arg0: tensor<24x12xi32>,
                           %arg1: tensor<12x25xi32>,
                           %arg2: tensor<24x25xi32>) -> tensor<24x25xi32> {
  %c0 = constant 0 : index
  %c12 = constant 12 : index
  %c25 = constant 25 : index
  %c24 = constant 24 : index
  %c4 = constant 4 : index
  %0 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%arg1 : tensor<12x25xi32>) {
  ^bb0(%arg3: i32):  // no predecessors
    %6 = linalg.index 0 : index
    %7 = linalg.index 1 : index
    %8 = addi %6, %7 : index
    %9 = index_cast %8 : index to i32
    linalg.yield %9 : i32
  } -> tensor<12x25xi32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  //      CHECK:    scf.for %[[IV1:[0-9a-zA-Z]*]] =
  //      CHECK:      scf.for %[[IV2:[0-9a-zA-Z]*]] =

  // Shift the indexes by the slice offsets and swap the offsets due to the transposed indexing map.
  //      CHECK:        %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // CHECK-SAME:                                          %[[IV2]], %[[IV0]]
  //      CHECK:  linalg.generic {{.*}} outs(%[[T1]]
  //      CHECK:  %[[IDX0:.*]] = linalg.index 0
  //      CHECK:  %[[IDX0_SHIFTED:.*]] = affine.apply #[[MAP0]](%[[IDX0]], %[[IV0]])
  //      CHECK:  %[[IDX1:.*]] = linalg.index 1
  //      CHECK:  %[[IDX1_SHIFTED:.*]] = affine.apply #[[MAP0]](%[[IDX1]], %[[IV2]])
  //      CHECK:  %{{.*}} = addi %[[IDX0_SHIFTED]], %[[IDX1_SHIFTED]]
  %1 = linalg.matmul ins(%arg0, %0 : tensor<24x12xi32>, tensor<12x25xi32>) outs(%arg2 : tensor<24x25xi32>) -> tensor<24x25xi32>
  return %1 : tensor<24x25xi32>
}

// -----

//  CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
//  CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (8, -d0 - d1 + 17)>
//  CHECK-DAG:  #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, -d1 - d2 + 17)>
#map0 = affine_map<(d0, d1) -> (d0, d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

//      CHECK:  fuse_non_rectangular
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<10x17xf32>
func @fuse_non_rectangular(%arg0: tensor<10x17xf32>,
                           %arg1: tensor<10x8xf32>) -> tensor<10x8xf32> {

  //  CHECK-DAG:  %[[C0:.*]] = constant 0 : index
  //  CHECK-DAG:  %[[C4:.*]] = constant 4 : index
  //  CHECK-DAG:  %[[C5:.*]] = constant 5 : index
  //  CHECK-DAG:  %[[C8:.*]] = constant 8 : index
  //  CHECK-DAG:  %[[C10:.*]] = constant 10 : index
  %cst = constant 0.000000e+00 : f32
  %0 = linalg.fill(%cst, %arg0) : f32, tensor<10x17xf32> -> tensor<10x17xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] = %[[C0]] to %[[C8]] step %[[C4]]
  //      CHECK:    scf.for %[[IV1:[0-9a-zA-Z]*]] = %[[C0]] to %[[C10]] step %[[C5]]

  // Compute producer on a hyper rectangular bounding box. Along the second dimenson,
  // the offset is set to the sum of the induction variables, and the upper bound
  // to either 8 (tile size) or 17 (sum of max indices (9+7) then + 1) minus the
  // induction variables.
  //      CHECK:      %[[SUM:.*]] = affine.apply #[[MAP0]](%[[IV1]], %[[IV0]]
  //      CHECK:      %[[TS1:.*]] = affine.min #[[MAP1]](%[[IV1]], %[[IV0]]
  //      CHECK:      %[[UB1:.*]] = affine.min #[[MAP2]](%[[TS1]], %[[IV1]], %[[IV0]]
  //      CHECK:      %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // CHECK-SAME:                                        %[[IV1]], %[[SUM]]
  // CHECK-SAME:                                                , %[[UB1]]
  //      CHECK:      %[[T1:.*]] = linalg.fill(%{{.*}}, %[[T0]])
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<10x17xf32>) outs(%arg1 : tensor<10x8xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
    %2 = addf %arg2, %arg3 : f32
    linalg.yield %2 : f32
  } -> tensor<10x8xf32>
  return %1 : tensor<10x8xf32>
}
