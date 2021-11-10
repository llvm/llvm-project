// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad pack-paddings=1,1,0 hoist-paddings=2,1,0 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad pack-paddings=1,1,0 hoist-paddings=3,2,0 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s --check-prefix=CHECK-DOUBLE

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (5, -d0 + 24)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (7, -d0 + 25)>
// CHECK-DAG: #[[MAP2:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 5)>
// CHECK-DAG: #[[MAP3:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 7)>
// CHECK-DAG: #[[DIV6:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 6)>
#map0 = affine_map<(d0) -> (5, -d0 + 24)>
#map1 = affine_map<(d0) -> (7, -d0 + 25)>

//      CHECK:  static_sizes
//      CHECK-DOUBLE:  static_sizes
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
func @static_sizes(%arg0: tensor<24x12xf32>,
                   %arg1: tensor<12x25xf32>,
                   %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C5:.*]] = arith.constant 5
  //  CHECK-DAG: %[[C7:.*]] = arith.constant 7
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
    //      CHECK:  %[[PT0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
    //        CHECK:   %[[PIDX0:.*]] = affine.apply #[[DIV6]](%[[PIV0]])
    //        CHECK:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])
    //        CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
    //   CHECK-SAME:                                     %[[IV0]], %[[PIV0]]
    //   CHECK-SAME:                                     %[[TS0]], 6
    //        CHECK:   %[[V0:.*]] = affine.apply #[[MAP2]](%[[TS0]])
    //        CHECK:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] nofold {{.*}} high[%[[V0]]
    //        CHECK:   %[[T2:.*]] = tensor.insert_slice %[[T1:.*]] into %{{.*}}[%[[PIDX0]], 0, 0]
    //        CHECK:   scf.yield %[[T2:.*]]

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c7 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {

      // Packing the second input operand for all values of IV2 (IV2x6x7).
      //      CHECK:  = linalg.init_tensor [2, 6, 7]
      //      CHECK:  %[[PT1:.*]] = scf.for %[[PIV1:[0-9a-z]+]] =
      //        CHECK:   %[[PIDX1:.*]] = affine.apply #[[DIV6]](%[[PIV1]])
      //        CHECK:   %[[TS1:.*]] = affine.min #[[MAP1]](%[[IV1]])
      //        CHECK:   %[[T3:.*]] = tensor.extract_slice %[[ARG1]]
      //   CHECK-SAME:                                     %[[PIV1]], %[[IV1]]
      //   CHECK-SAME:                                     6, %[[TS1]]
      //        CHECK:   %[[V1:.*]] = affine.apply #[[MAP3]](%[[TS1]])
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
        %8 = linalg.matmul ins(%4, %6 : tensor<?x6xf32>, tensor<6x?xf32>) outs(%7 : tensor<?x?xf32>) -> tensor<?x?xf32>
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

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0)[s0] -> (5, -d0 + s0)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0)[s0] -> (6, -d0 + s0)>
// CHECK-DAG: #[[MAP2:[0-9a-z]+]] = affine_map<(d0)[s0] -> (7, -d0 + s0)>
// CHECK-DAG: #[[MAP3:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 5)>
// CHECK-DAG: #[[MAP4:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: #[[MAP5:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 7)>
// CHECK-DAG: #[[SDIV6:[0-9a-z]+]] = affine_map<()[s0] -> (s0 ceildiv 6)>
// CHECK-DAG: #[[DDIV6:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 6)>
#map0 = affine_map<(d0)[s0] -> (5, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (6, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (7, -d0 + s0)>

//      CHECK:  dynamic_sizes
//      CHECK-DOUBLE:  dynamic_sizes
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<?x?xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<?x?xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<?x?xf32>
func @dynamic_sizes(%arg0: tensor<?x?xf32>,
                    %arg1: tensor<?x?xf32>,
                    %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C1:.*]] = arith.constant 1
  //  CHECK-DAG: %[[C5:.*]] = arith.constant 5
  //  CHECK-DAG: %[[C6:.*]] = arith.constant 6
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index

  //  CHECK-DAG: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  //  CHECK-DAG: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  //  CHECK-DAG: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %3 = scf.for %arg3 = %c0 to %0 step %c5 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {

    // Packing the first input operand for all values of IV2 (IV2x5x6).
    //      CHECK:  %[[PS0:.*]] = affine.apply #[[SDIV6]]()[%[[D1]]
    //      CHECK:  = linalg.init_tensor [%[[PS0]], 5, 6]
    //      CHECK:  %[[PT0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
    //        CHECK:   %[[PIDX0:.*]] = affine.apply #[[DDIV6]](%[[PIV0]])
    //        CHECK:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0]]
    //        CHECK:   %[[TS1:.*]] = affine.min #[[MAP1]](%[[PIV0]])[%[[D1]]
    //        CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
    //   CHECK-SAME:                                     %[[IV0]], %[[PIV0]]
    //   CHECK-SAME:                                     %[[TS0]], %[[TS1]]
    //        CHECK:   %[[V0:.*]] = affine.apply #[[MAP3]](%[[TS0]])
    //        CHECK:   %[[V1:.*]] = affine.apply #[[MAP4]](%[[TS1]])
    //        CHECK:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] nofold {{.*}} high[%[[V0]], %[[V1]]
    //        CHECK:   %[[T2:.*]] = tensor.insert_slice %[[T1:.*]] into %{{.*}}[%[[PIDX0]], 0, 0]
    //        CHECK:   scf.yield %[[T2:.*]]

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %4 = scf.for %arg5 = %c0 to %2 step %c7 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {

      // Packing the second input operand for all values of IV2 (IV2x6x7).
      //      CHECK:  = linalg.init_tensor [%[[PS0]], 6, 7]
      //      CHECK:  %[[PT1:.*]] = scf.for %[[PIV1:[0-9a-z]+]] =
      //        CHECK:   %[[PIDX1:.*]] = affine.apply #[[DDIV6]](%[[PIV1]])
      //        CHECK:   %[[TS2:.*]] = affine.min #[[MAP1]](%[[PIV1]])[%[[D1]]
      //        CHECK:   %[[TS3:.*]] = affine.min #[[MAP2]](%[[IV1]])[%[[D2]]
      //        CHECK:   %[[T3:.*]] = tensor.extract_slice %[[ARG1]]
      //   CHECK-SAME:                                     %[[PIV1]], %[[IV1]]
      //   CHECK-SAME:                                     %[[TS2]], %[[TS3]]
      //        CHECK:   %[[V2:.*]] = affine.apply #[[MAP4]](%[[TS2]])
      //        CHECK:   %[[V3:.*]] = affine.apply #[[MAP5]](%[[TS3]])
      //        CHECK:   %[[T4:.*]] = linalg.pad_tensor %[[T3]] nofold {{.*}} high[%[[V2]], %[[V3]]
      //        CHECK:   %[[T5:.*]] = tensor.insert_slice %[[T4:.*]] into %{{.*}}[%[[PIDX1]], 0, 0]
      //        CHECK:   scf.yield %[[T5:.*]]

      //      CHECK:  scf.for %[[IV2:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG4:.*]] =
      %5 = scf.for %arg7 = %c0 to %1 step %c6 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {
        %6 = affine.min #map0(%arg3)[%0]
        %7 = affine.min #map1(%arg7)[%1]

        // Index the packed operands.
        //    CHECK-DAG:   %[[IDX:.*]] = affine.apply #[[DDIV6]](%[[IV2]])
        //    CHECK-DAG:   %[[T6:.*]] = tensor.extract_slice %[[PT0]][%[[IDX]]
        //    CHECK-DAG:   %[[T7:.*]] = tensor.extract_slice %[[PT1]][%[[IDX]]
        %8 = tensor.extract_slice %arg0[%arg3, %arg7] [%6, %7] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %9 = affine.min #map2(%arg5)[%2]
        %10 = tensor.extract_slice %arg1[%arg7, %arg5] [%7, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %11 = tensor.extract_slice %arg8[%arg3, %arg5] [%6, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

        // Check matmul uses the packed input operands.
        //        CHECK:   = linalg.matmul ins(%[[T6]], %[[T7]]
        %12 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %13 = tensor.insert_slice %12 into %arg8[%arg3, %arg5] [%6, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %13 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

// -----

// CHECK-DOUBLE-DAG: #[[DIV5:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 5)>
// CHECK-DOUBLE-DAG: #[[DIV6:[0-9a-z]+]] = affine_map<(d0) -> (d0 ceildiv 6)>
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

  //    CHECK-DOUBLE:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c24 step %c15 iter_args(%arg4 = %arg2) -> (tensor<24x25xf32>) {

    // Packing the first input operand.
    //    CHECK-DOUBLE:  = linalg.init_tensor [3, 5, 12]
    //    CHECK-DOUBLE:  %[[PT0:.*]] = scf.for %[[PIV0:[0-9a-z]+]] =
    //      CHECK-DOUBLE:   %[[PIDX0:.*]] = affine.apply #[[DIV5]](%[[PIV0]])
    //      CHECK-DOUBLE:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
    //      CHECK-DOUBLE:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] nofold
    //      CHECK-DOUBLE:   %[[T2:.*]] = tensor.insert_slice %[[T1:.*]] into %{{.*}}[%[[PIDX0]], 0, 0]
    //      CHECK-DOUBLE:   scf.yield %[[T2:.*]]

    //    CHECK-DOUBLE:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c16 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {
      %2 = affine.min #map0(%arg3)
      %3 = affine.min #map1(%arg5)
      %4 = tensor.extract_slice %arg6[%arg3, %arg5] [%2, %3] [1, 1] : tensor<24x25xf32> to tensor<?x?xf32>

      // Packing the second input operand.
      //    CHECK-DOUBLE:  = linalg.init_tensor [3, 12, 6]
      //    CHECK-DOUBLE:  %[[PT1:.*]] = scf.for %[[PIV1:[0-9a-z]+]] =
      //      CHECK-DOUBLE:   %[[PIDX1:.*]] = affine.apply #[[DIV6]](%[[PIV1]])
      //      CHECK-DOUBLE:   %[[T3:.*]] = tensor.extract_slice %[[ARG1]]
      //      CHECK-DOUBLE:   %[[T4:.*]] = linalg.pad_tensor %[[T3]] nofold
      //      CHECK-DOUBLE:   %[[T5:.*]] = tensor.insert_slice %[[T4:.*]] into %{{.*}}[%[[PIDX1]], 0, 0]
      //      CHECK-DOUBLE:   scf.yield %[[T5:.*]]

      //    CHECK-DOUBLE:  scf.for %[[IV2:[0-9a-zA-Z]*]] =
      %5 = scf.for %arg7 = %c0 to %2 step %c5 iter_args(%arg8 = %4) -> (tensor<?x?xf32>) {

        //    CHECK-DOUBLE:  scf.for %[[IV3:[0-9a-zA-Z]*]] =
        %7 = scf.for %arg9 = %c0 to %3 step %c6 iter_args(%arg10 = %arg8) -> (tensor<?x?xf32>) {
          %8 = affine.min #map2(%arg7, %2)
          %9 = affine.apply #map3(%arg7, %arg3)

          // Index the packed operands.
          //    CHECK-DOUBLE-DAG:   %[[IDX0:.*]] = affine.apply #[[DIV5]](%[[IV2]])
          //    CHECK-DOUBLE-DAG:   %[[T6:.*]] = tensor.extract_slice %[[PT0]][%[[IDX0]]
          //    CHECK-DOUBLE-DAG:   %[[IDX1:.*]] = affine.apply #[[DIV6]](%[[IV3]])
          //    CHECK-DOUBLE-DAG:   %[[T7:.*]] = tensor.extract_slice %[[PT1]][%[[IDX1]]
          %10 = tensor.extract_slice %arg0[%9, 0] [%8, 12] [1, 1] : tensor<24x12xf32> to tensor<?x12xf32>
          %11 = affine.min #map4(%arg9, %3)
          %12 = affine.apply #map3(%arg9, %arg5)
          %13 = tensor.extract_slice %arg1[0, %12] [12, %11] [1, 1] : tensor<12x25xf32> to tensor<12x?xf32>
          %14 = affine.min #map2(%arg7, %2)
          %15 = affine.min #map4(%arg9, %3)
          %16 = tensor.extract_slice %arg10[%arg7, %arg9] [%14, %15] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

          // Check matmul uses the packed input operands.
          //    CHECK-DOUBLE:   = linalg.matmul ins(%[[T6]], %[[T7]]
          %17 = linalg.matmul ins(%10, %13 : tensor<?x12xf32>, tensor<12x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
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
