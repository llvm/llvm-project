// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matvec pad hoist-paddings=1,1,0 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s --check-prefix=MATVEC
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad hoist-paddings=1,2,1 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s --check-prefix=MATMUL

//  MATVEC-DAG: #[[DIV4:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 4)>

//      MATVEC:  static_size_divisible
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
func @static_size_divisible(%arg0: tensor<24x12xf32>,
                            %arg1: tensor<12xf32>,
                            %arg2: tensor<24xf32>) -> tensor<24xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c4 = arith.constant 4 : index

  // Pack the vector tiles for all values of IV (IVx4).
  //      MATVEC:  = linalg.init_tensor [3, 4]
  //      MATVEC:  %[[T0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
  //        MATVEC:   %[[PIDX0:.*]] = affine.apply #[[DIV4]](%[[PIV0]])
  //        MATVEC:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]][%[[PIV0]]] [4]
  //        MATVEC:   %[[T2:.*]] = linalg.pad_tensor %[[T1]]
  //        MATVEC:   %[[T3:.*]] = tensor.insert_slice %[[T1:.*]]{{.*}}[%[[PIDX0]]

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c12 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg3] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Index the packed vector.
    //  MATVEC-DAG:   %[[IDX0:.*]] = affine.apply #[[DIV4]](%[[IV0]])
    //  MATVEC-DAG:   %[[T4:.*]] = tensor.extract_slice %[[T0]][%[[IDX0]]
    %2 = tensor.extract_slice %arg1[%arg3] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = linalg.pad_tensor %2 nofold low[%c0] high[%c0]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the packed input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T4]]
    %4 = linalg.matvec ins(%1, %3 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg4 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %4 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

// MATVEC-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (5, -d0 + 12)>
// MATVEC-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 5)>
// MATVEC-DAG: #[[DIV5:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 5)>
#map0 = affine_map<(d0) -> (5, -d0 + 12)>
#map1 = affine_map<(d0) -> (-d0 + 5)>

//      MATVEC:  static_size_not_divisible
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
func @static_size_not_divisible(%arg0: tensor<24x12xf32>,
                                %arg1: tensor<12xf32>,
                                %arg2: tensor<24xf32>) -> tensor<24xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c5 = arith.constant 5 : index

  // Pack the vector tiles for all values of IV (IVx5).
  //      MATVEC:  = linalg.init_tensor [3, 5]
  //      MATVEC:  %[[T0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
  //        MATVEC:   %[[PIDX0:.*]] = affine.apply #[[DIV5]](%[[PIV0]])
  //        MATVEC:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[PIV0]])
  //        MATVEC:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]][%[[PIV0]]] [%[[TS0]]]
  //        MATVEC:   %[[HPD0:.*]] = affine.apply #[[MAP1]](%[[TS0]])
  //        MATVEC:   %[[T2:.*]] = linalg.pad_tensor %[[T1]]{{.*}}high[%[[HPD0]]
  //        MATVEC:   %[[T3:.*]] = tensor.insert_slice %[[T1:.*]]{{.*}}[%[[PIDX0]]

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c12 step %c5 iter_args(%arg4 = %arg2) -> (tensor<24xf32>) {
    %1 = affine.min #map0(%arg3)
    %2 = tensor.extract_slice %arg0[0, %arg3] [24, %1] [1, 1] : tensor<24x12xf32> to tensor<24x?xf32>

    // Index the packed vector.
    //  MATVEC-DAG:   %[[IDX0:.*]] = affine.apply #[[DIV5]](%[[IV0]])
    //  MATVEC-DAG:   %[[T4:.*]] = tensor.extract_slice %[[T0]][%[[IDX0]]
    %3 = tensor.extract_slice %arg1[%arg3] [%1] [1] : tensor<12xf32> to tensor<?xf32>
    %4 = affine.apply #map1(%1)
    %5 = linalg.pad_tensor %2 low[%c0, %c0] high[%c0, %4]  {
    ^bb0(%arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<24x?xf32> to tensor<24x5xf32>
    %6 = linalg.pad_tensor %3 low[%c0] high[%4]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<?xf32> to tensor<5xf32>

    // Check matvec uses the packed input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T4]]
    %7 = linalg.matvec ins(%5, %6 : tensor<24x5xf32>, tensor<5xf32>) outs(%arg4 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %7 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

// MATVEC-DAG: #[[SDIV4:[0-9a-z]+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
// MATVEC-DAG: #[[DDIV4:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 4)>
// MATVEC-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
// MATVEC-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 4)>
#map0 = affine_map<(d0)[s0] -> (4, -d0 + s0)>
#map1 = affine_map<(d0) -> (-d0 + 4)>

//      MATVEC:  dynamic_size
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<?xf32>
func @dynamic_size(%arg0: tensor<24x?xf32>,
                   %arg1: tensor<?xf32>,
                   %arg2: tensor<24xf32>) -> tensor<24xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  //      MATVEC:  %[[D0:.*]] = tensor.dim
  %0 = tensor.dim %arg0, %c1 : tensor<24x?xf32>

  // Pack the vector tiles for all values of IV (IVx4).
  //      MATVEC:  %[[PS0:.*]] = affine.apply #[[SDIV4]]()[%[[D0]]]
  //      MATVEC:  = linalg.init_tensor [%[[PS0]], 4]
  //      MATVEC:  %[[T0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
  //        MATVEC:   %[[PIDX0:.*]] = affine.apply #[[DDIV4]](%[[PIV0]])
  //        MATVEC:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[PIV0]])[%[[D0]]]
  //        MATVEC:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]][%[[PIV0]]] [%[[TS0]]]
  //        MATVEC:   %[[HPD0:.*]] = affine.apply #[[MAP1]](%[[TS0]])
  //        MATVEC:   %[[T2:.*]] = linalg.pad_tensor %[[T1]]{{.*}}high[%[[HPD0]]
  //        MATVEC:   %[[T3:.*]] = tensor.insert_slice %[[T1:.*]]{{.*}}[%[[PIDX0]]

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %1 = scf.for %arg3 = %c0 to %0 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24xf32>) {
    %2 = affine.min #map0(%arg3)[%0]
    %3 = tensor.extract_slice %arg0[0, %arg3] [24, %2] [1, 1] : tensor<24x?xf32> to tensor<24x?xf32>

    // Index the packed vector.
    //  MATVEC-DAG:   %[[IDX0:.*]] = affine.apply #[[DDIV4]](%[[IV0]])
    //  MATVEC-DAG:   %[[T4:.*]] = tensor.extract_slice %[[T0]][%[[IDX0]]
    %4 = tensor.extract_slice %arg1[%arg3] [%2] [1] : tensor<?xf32> to tensor<?xf32>
    %5 = affine.apply #map1(%2)
    %6 = linalg.pad_tensor %3 low[%c0, %c0] high[%c0, %5]  {
    ^bb0(%arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<24x?xf32> to tensor<24x4xf32>
    %7 = linalg.pad_tensor %4 nofold low[%c0] high[%5]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<?xf32> to tensor<4xf32>

    // Check matvec uses the packed input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T4]]
    %8 = linalg.matvec ins(%6, %7 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg4 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %8 : tensor<24xf32>
  }
  return %1 : tensor<24xf32>
}

// -----

//      MATVEC:  non_constant_padding
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
func @non_constant_padding(%arg0: tensor<24x12xf32>,
                   %arg1: tensor<12xf32>,
                   %arg2: tensor<24xf32>) -> tensor<24xf32> {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c12 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg3] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Check the non constant padding is not hoisted.
    //      MATVEC:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[T1:.*]] = linalg.pad_tensor %[[T0]]
    %2 = tensor.extract_slice %arg1[%arg3] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = linalg.pad_tensor %2 nofold low[%c0] high[%c0]  {
    ^bb0(%arg5: index):  // no predecessors
      %5 = arith.index_cast %arg3 : index to i32
      %6 = arith.sitofp %5 : i32 to f32
      linalg.yield %6 : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the padded input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T1]]
    %4 = linalg.matvec ins(%1, %3 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg4 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %4 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

//      MATVEC:  non_constant_op_padding
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
func @non_constant_op_padding(%arg0: tensor<24x12xf32>,
                      %arg1: tensor<12xf32>,
                      %arg2: tensor<24xf32>) -> tensor<24xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c4 = arith.constant 4 : index

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c12 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg3] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Check the non constant op padding is not hoisted.
    //      MATVEC:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[V0:.*]] = tensor.extract %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[T1:.*]] = linalg.pad_tensor %[[T0]]
    //        MATVEC:  linalg.yield %[[V0]]
    %2 = tensor.extract_slice %arg1[%arg3] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = tensor.extract %arg1[%arg3] : tensor<12xf32>
    %4 = linalg.pad_tensor %2 nofold low[%c0] high[%c0]  {
    ^bb0(%arg5: index):  // no predecessors
      linalg.yield %3 : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the padded input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T1]]
    %5 = linalg.matvec ins(%1, %4 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg4 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %5 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

//      MATVEC:  non_index_operand
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
// MATVEC-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: i32
func @non_index_operand(%arg0: tensor<24x12xf32>,
                        %arg1: tensor<12xf32>,
                        %arg2: tensor<24xf32>,
                        %arg3: i32) -> tensor<24xf32> {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg4 = %c0 to %c12 step %c4 iter_args(%arg5 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg4] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Check the index_cast prevents hoisting due to its non index operand.
    //      MATVEC:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[IDX0:.*]] = arith.index_cast %[[ARG3]]
    //      MATVEC:  %[[T1:.*]] = linalg.pad_tensor %[[T0]]{{.*}}%[[IDX0]]
    %2 = tensor.extract_slice %arg1[%arg4] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = linalg.pad_tensor %2 nofold low[%3] high[%3]  {
    ^bb0(%arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the padded input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T1]]
    %5 = linalg.matvec ins(%1, %4 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg5 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %5 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

//      MATVEC:  memory_effect
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
// MATVEC-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: memref<?xindex>
func @memory_effect(%arg0: tensor<24x12xf32>,
                    %arg1: tensor<12xf32>,
                    %arg2: tensor<24xf32>,
                    %arg3: memref<?xindex>) -> tensor<24xf32> {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg4 = %c0 to %c12 step %c4 iter_args(%arg5 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg4] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Check the load prevents hoisting due to its memory effect.
    //      MATVEC:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[IDX0:.*]] = memref.load %[[ARG3]]
    //      MATVEC:  %[[T1:.*]] = linalg.pad_tensor %[[T0]]{{.*}}%[[IDX0]]
    %2 = tensor.extract_slice %arg1[%arg4] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = memref.load %arg3[%c0] : memref<?xindex>
    %4 = linalg.pad_tensor %2 nofold low[%3] high[%3]  {
    ^bb0(%arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the padded input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T1]]
    %5 = linalg.matvec ins(%1, %4 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg5 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %5 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

//      MATVEC:  index_result_loop
// MATVEC-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12xf32>
// MATVEC-SAME:    %[[ARG3:[0-9a-zA-Z]*]]: index
func @index_result_loop(%arg0: tensor<24x12xf32>,
                        %arg1: tensor<12xf32>,
                        %arg2: tensor<24xf32>,
                        %arg3: index) -> tensor<24xf32> {
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32

  //      MATVEC:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg4 = %c0 to %c12 step %c4 iter_args(%arg5 = %arg2) -> (tensor<24xf32>) {
    %1 = tensor.extract_slice %arg0[0, %arg4] [24, 4] [1, 1] : tensor<24x12xf32> to tensor<24x4xf32>

    // Check the unexpected operation with a region prevents hoisting.
    //      MATVEC:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]][%[[IV0]]
    //      MATVEC:  %[[IDX0:.*]] = scf.for {{.*}} step %[[ARG3]]
    //      MATVEC:  %[[T1:.*]] = linalg.pad_tensor %[[T0]]{{.*}}%[[IDX0]]
    %2 = tensor.extract_slice %arg1[%arg4] [4] [1] : tensor<12xf32> to tensor<4xf32>
    %3 = scf.for %arg6 = %c0 to %c12 step %arg3 iter_args(%arg7 = %c0) -> (index) {
      %6 = arith.addi %arg3, %arg7 : index
      scf.yield %6 : index
    }
    %4 = linalg.pad_tensor %2 nofold low[%3] high[%3]  {
    ^bb0(%arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<4xf32> to tensor<4xf32>

    // Check matvec uses the padded input vector.
    //      MATVEC:  = linalg.matvec ins(%{{.*}}, %[[T1]]
    %5 = linalg.matvec ins(%1, %4 : tensor<24x4xf32>, tensor<4xf32>) outs(%arg5 : tensor<24xf32>) -> tensor<24xf32>
    scf.yield %5 : tensor<24xf32>
  }
  return %0 : tensor<24xf32>
}

// -----

#map0 = affine_map<(d0) -> (5, -d0 + 12)>
#map1 = affine_map<(d0) -> (-d0 + 5)>

//      MATMUL:  tile_and_fuse
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<12x6xf32>
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<6x24xf32>
func @tile_and_fuse(%arg0: tensor<12x6xf32>,
                    %arg1: tensor<6x24xf32>,
                    %arg2: tensor<12x24xf32>) -> tensor<12x24xf32> {
  %c6 = arith.constant 6 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c5 = arith.constant 5 : index
  %cst = arith.constant 0.000000e+00 : f32

  // Check the second input operand is hoisted by two loop nests.
  //      MATMUL:  %[[T0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
  //        MATMUL:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  //        MATMUL:   %[[T2:.*]] = linalg.pad_tensor %[[T1]]

  //      MATMUL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c12 step %c5 iter_args(%arg4 = %arg2) -> (tensor<12x24xf32>) {
    %1 = affine.min #map0(%arg3)

    // Check the extract_slice op introduced by the double tiling does not prevent the hoisting.
    %2 = tensor.extract_slice %arg4[%arg3, 0] [%1, 24] [1, 1] : tensor<12x24xf32> to tensor<?x24xf32>
    %3 = affine.apply #map1(%1)

    // Check the fused and padded fill op does not prevent hoisting.
    %4 = linalg.pad_tensor %2 nofold low[%c0, %c0] high[%3, %c0]  {
    ^bb0(%arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<?x24xf32> to tensor<5x24xf32>
    %5 = linalg.fill(%cst, %4) : f32, tensor<5x24xf32> -> tensor<5x24xf32>
    %6 = tensor.extract_slice %5[0, 0] [%1, 24] [1, 1] : tensor<5x24xf32> to tensor<?x24xf32>

    // Check the first input operand is hoisted by one loop nest.
    //      MATMUL:  %[[T3:.*]] = scf.for %[[PIV1:[0-9a-z]+]] =
    //        MATMUL:   %[[T4:.*]] = tensor.extract_slice %[[ARG0]]
    //        MATMUL:   %[[T5:.*]] = linalg.pad_tensor %[[T4]]

    //      MATMUL:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %7 = scf.for %arg5 = %c0 to %c6 step %c3 iter_args(%arg6 = %6) -> (tensor<?x24xf32>) {

      // Index the packed operands.
      //    MATMUL-DAG:   %[[T6:.*]] = tensor.extract_slice %[[T3]]
      //    MATMUL-DAG:   %[[T7:.*]] = tensor.extract_slice %[[T0]]
      %9 = tensor.extract_slice %arg0[%arg3, %arg5] [%1, 3] [1, 1] : tensor<12x6xf32> to tensor<?x3xf32>
      %10 = tensor.extract_slice %arg1[%arg5, 0] [3, 24] [1, 1] : tensor<6x24xf32> to tensor<3x24xf32>
      %11 = tensor.extract_slice %arg6[0, 0] [%1, 24] [1, 1] : tensor<?x24xf32> to tensor<?x24xf32>
      %12 = linalg.pad_tensor %9 nofold low[%c0, %c0] high[%3, %c0]  {
      ^bb0(%arg7: index, %arg8: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<?x3xf32> to tensor<5x3xf32>
      %13 = linalg.pad_tensor %10 nofold low[%c0, %c0] high[%c0, %c0]  {
      ^bb0(%arg7: index, %arg8: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<3x24xf32> to tensor<3x24xf32>

      // Check the output padding is not hoisted.
      //      MATMUL:   %[[T8:.*]] = linalg.pad_tensor
      %14 = linalg.pad_tensor %11 nofold low[%c0, %c0] high[%3, %c0]  {
      ^bb0(%arg7: index, %arg8: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<?x24xf32> to tensor<5x24xf32>

      // Check matmul uses the padded operands.
      //      MATMUL:   = linalg.matmul ins(%[[T6]], %[[T7]] {{.*}} outs(%[[T8]]
      %15 = linalg.matmul ins(%12, %13 : tensor<5x3xf32>, tensor<3x24xf32>) outs(%14 : tensor<5x24xf32>) -> tensor<5x24xf32>
      %16 = tensor.extract_slice %15[0, 0] [%1, 24] [1, 1] : tensor<5x24xf32> to tensor<?x24xf32>
      %17 = tensor.insert_slice %16 into %arg6[0, 0] [%1, 24] [1, 1] : tensor<?x24xf32> into tensor<?x24xf32>
      scf.yield %17 : tensor<?x24xf32>
    }
    %8 = tensor.insert_slice %7 into %arg4[%arg3, 0] [%1, 24] [1, 1] : tensor<?x24xf32> into tensor<12x24xf32>
    scf.yield %8 : tensor<12x24xf32>
  }
  return %0 : tensor<12x24xf32>
}
