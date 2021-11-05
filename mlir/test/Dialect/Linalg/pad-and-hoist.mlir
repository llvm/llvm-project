// RUN: mlir-opt %s -test-linalg-transform-patterns="test-pad-pattern pack-paddings=1,1,0 hoist-paddings=2,1,0" -cse -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns="test-pad-pattern pack-paddings=1,1,0 hoist-paddings=4,3,0" -cse -canonicalize -split-input-file | FileCheck %s --check-prefix=CHECK-DOUBLE

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (5, -d0 + 24)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (8, -d0 + 12)>
// CHECK-DAG: #[[DIV6:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 6)>
#map0 = affine_map<(d0) -> (5, -d0 + 24)>
#map1 = affine_map<(d0) -> (8, -d0 + 12)>
#map2 = affine_map<(d0) -> (7, -d0 + 25)>

//      CHECK:  single_tiling
//      CHECK-DOUBLE:  single_tiling

// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
func @single_tiling(%arg0: tensor<24x12xf32>,
                    %arg1: tensor<12x25xf32>,
                    %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C5:.*]] = arith.constant 5
  //  CHECK-DAG: %[[C8:.*]] = arith.constant 8
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c24 step %c5 iter_args(%arg4 = %arg2) -> (tensor<24x25xf32>) {

    // Packing the first input operand for all values of IV2 (IV2x5x6).
    //      CHECK:  = linalg.init_tensor [2, 5, 6]
    //      CHECK:  %[[PT0:.*]] = scf.for %[[P0IV2:[0-9a-z]+]] =
    //        CHECK:   %[[PIDX0:.*]] = affine.apply #[[DIV6]](%[[P0IV2]])
    //        CHECK:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])
    //        CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
    //   CHECK-SAME:                                     %[[IV0]], %[[P0IV2]]
    //   CHECK-SAME:                                     %[[TS0]], 6
    //        CHECK:   %[[V0:.*]] = arith.subi %[[C5]], %[[TS0]]
    //        CHECK:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] nofold {{.*}} high[%[[V0]]
    //        CHECK:   %[[T2:.*]] = tensor.insert_slice %[[T1:.*]] into %{{.*}}[%[[PIDX0]], 0, 0]
    //        CHECK:   scf.yield %[[T2:.*]]

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c7 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {

      // Packing the second input operand for all values of IV2 (IV2x6x8).
      //      CHECK:  = linalg.init_tensor [2, 6, 8]
      //      CHECK:  %[[PT1:.*]] = scf.for %[[P1IV2:[0-9a-z]+]] =
      //        CHECK:   %[[PIDX1:.*]] = affine.apply #[[DIV6]](%[[P1IV2]])
      //        CHECK:   %[[TS1:.*]] = affine.min #[[MAP1]](%[[IV1]])
      //        CHECK:   %[[T3:.*]] = tensor.extract_slice %[[ARG1]]
      //   CHECK-SAME:                                     %[[P1IV2]], %[[IV1]]
      //   CHECK-SAME:                                     6, %[[TS1]]
      //        CHECK:   %[[V1:.*]] = arith.subi %[[C8]], %[[TS1]]
      //        CHECK:   %[[T4:.*]] = linalg.pad_tensor %[[T3]] nofold {{.*}} high[%[[C0]], %[[V1]]
      //        CHECK:   %[[T5:.*]] = tensor.insert_slice %[[T4:.*]] into %{{.*}}[%[[PIDX1]], 0, 0]
      //        CHECK:   scf.yield %[[T5:.*]]

      //      CHECK:  scf.for %[[IV2:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG4:.*]] =
      %2 = scf.for %arg7 = %c0 to %c12 step %c6 iter_args(%arg8 = %arg6) -> (tensor<24x25xf32>) {
        %3 = affine.min #map0(%arg3)
        // Index the packed operands.
        //    CHECK-DAG:   %[[IDX:.*]] = affine.apply #[[DIV6]](%[[IV2]])
        //    CHECK-DAG:   %[[T6:.*]] = tensor.extract_slice %[[PT0]][%[[IDX]]
        //    CHECK-DAG:   %[[T7:.*]] = tensor.extract_slice %[[PT1]][%[[IDX]]
        %4 = tensor.extract_slice %arg0[%arg3, %arg7] [%3, 6] [1, 1] : tensor<24x12xf32> to tensor<?x6xf32>
        %5 = affine.min #map1(%arg5)
        %6 = tensor.extract_slice %arg1[%arg7, %arg5] [6, %5] [1, 1] : tensor<12x25xf32> to tensor<6x?xf32>

        // Pad the output operand without setting the nofold attribute.
        //    CHECK-DAG:   %[[T8:.*]] = tensor.extract_slice %[[ARG4]][%[[IV0]], %[[IV1]]
        //        CHECK:   %[[T9:.*]] = linalg.pad_tensor %[[T8]] low
        %7 = tensor.extract_slice %arg8[%arg3, %arg5] [%3, %5] [1, 1] : tensor<24x25xf32> to tensor<?x?xf32>

        // Check matmul uses the packed input operands and the padded output operand.
        //        CHECK:   = linalg.matmul ins(%[[T6]], %[[T7]]{{.*}} outs(%[[T9]]
        %8 = linalg.matmul {__internal_linalg_transform__ = "pad"} ins(%4, %6 : tensor<?x6xf32>, tensor<6x?xf32>) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %9 = tensor.insert_slice %8 into %arg8[%arg3, %arg5] [%3, %5] [1, 1] : tensor<?x?xf32> into tensor<24x25xf32>
        scf.yield %9 : tensor<24x25xf32>
      }
      scf.yield %2 : tensor<24x25xf32>
    }
    scf.yield %1 : tensor<24x25xf32>
  }
  return %0 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<(d0) -> (15, -d0 + 24)>
#map1 = affine_map<(d0) -> (16, -d0 + 25)>
#map2 = affine_map<(d0, d1) -> (5, -d0 + d1)>
#map3 = affine_map<(d0, d1) -> (d0 + d1)>
#map4 = affine_map<(d0, d1) -> (6, -d0 + d1)>

//      CHECK:  double_tiling
//      CHECK-DOUBLE:  double_tiling

// CHECK-DOUBLE-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-DOUBLE-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// CHECK-DOUBLE-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
func @double_tiling(%arg0: tensor<24x12xf32>,
                    %arg1: tensor<12x25xf32>,
                    %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  %c15 = arith.constant 15 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %c25 = arith.constant 25 : index
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // Packing the first input operand.
  //    CHECK-DOUBLE:  = linalg.init_tensor
  //    CHECK-DOUBLE:  = linalg.pad_tensor {{.*}} nofold

  //    CHECK-DOUBLE:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c24 step %c15 iter_args(%arg4 = %arg2) -> (tensor<24x25xf32>) {

    // Packing the second input operand.
    //    CHECK-DOUBLE:  = linalg.init_tensor
    //    CHECK-DOUBLE:  = linalg.pad_tensor {{.*}} nofold

    //    CHECK-DOUBLE:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c16 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {
      %2 = affine.min #map0(%arg3)
      %3 = affine.min #map1(%arg5)
      %4 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<24x25xf32> to tensor<?x?xf32>

      //    CHECK-DOUBLE:  scf.for %[[IV2:[0-9a-zA-Z]*]] =
      %5 = scf.for %arg7 = %c0 to %2 step %c5 iter_args(%arg8 = %4) -> (tensor<?x?xf32>) {

        //    CHECK-DOUBLE:  scf.for %[[IV3:[0-9a-zA-Z]*]] =
        %7 = scf.for %arg9 = %c0 to %3 step %c6 iter_args(%arg10 = %arg8) -> (tensor<?x?xf32>) {
          %8 = affine.min #map2(%arg7, %2)
          %9 = affine.apply #map3(%arg7, %arg3)
          %10 = tensor.extract_slice %arg0[%9, 0] [%8, 12] [1, 1] : tensor<24x12xf32> to tensor<?x12xf32>
          %11 = affine.min #map4(%arg9, %3)
          %12 = affine.apply #map3(%arg9, %arg5)
          %13 = tensor.extract_slice %arg1[0, %12] [12, %11] [1, 1] : tensor<12x25xf32> to tensor<12x?xf32>
          %14 = affine.min #map2(%arg7, %2)
          %15 = affine.min #map4(%arg9, %3)
          %16 = tensor.extract_slice %arg10[%arg7, %arg9] [%14, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

          // Pad the output operand and perform the multiplication.
          //        CHECK-DOUBLE:   = linalg.pad_tensor
          //        CHECK-DOUBLE:   = linalg.matmul
          %17 = linalg.matmul {__internal_linalg_transform__ = "pad"} ins(%10, %13 : tensor<?x12xf32>, tensor<12x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %18 = tensor.insert_slice %17 into %arg10[%arg7, %arg9] [%14, %15] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
          scf.yield %18 : tensor<?x?xf32>
        }
        scf.yield %7 : tensor<?x?xf32>
      }
      %6 = tensor.insert_slice %5 into %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<?x?xf32> into tensor<24x25xf32>
      scf.yield %6 : tensor<24x25xf32>
    }
    scf.yield %1 : tensor<24x25xf32>
  }
  return %0 : tensor<24x25xf32>
}
