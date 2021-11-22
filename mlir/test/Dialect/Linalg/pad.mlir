// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad pack-paddings=1,1,0 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.fill pad pack-paddings=1,1,0 run-enable-pass=false" -cse -canonicalize -split-input-file | FileCheck %s --check-prefix=CHECK-FILL

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (7, -d0 + 12)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 7)>
#map = affine_map<(d0) -> (7, -d0 + 12)>

//      CHECK:  static_sizes_output_divisible
// CHECK-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
func @static_sizes_output_divisible(%arg0: tensor<24x12xf32>,
                                    %arg1: tensor<12x25xf32>,
                                    %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C7:.*]] = arith.constant 7
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c24 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24x25xf32>) {

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c5 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {

      //      CHECK:  scf.for %[[IV2:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG4:.*]] =
      %2 = scf.for %arg7 = %c0 to %c12 step %c7 iter_args(%arg8 = %arg6) -> (tensor<24x25xf32>) {

        //      CHECK:   %[[TS2:.*]] = affine.min #[[MAP0]](%[[IV2]])
        %3 = affine.min #map(%arg7)

        //      CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
        //      CHECK:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
        //      CHECK:   %[[T2:.*]] = tensor.extract_slice %[[ARG4]]
        %4 = tensor.extract_slice %arg0[%arg3, %arg7] [4, %3] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
        %5 = tensor.extract_slice %arg1[%arg7, %arg5] [%3, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
        %6 = tensor.extract_slice %arg8[%arg3, %arg5] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

        // Check statically sized matmul inputs with partially divisible sizes are padded.
        //      CHECK:   %[[V0:.*]] = affine.apply #[[MAP1]](%[[TS2]])
        //      CHECK:   %[[T3:.*]] = linalg.pad_tensor %[[T0]] nofold
        // CHECK-SAME:                  [%[[C0]], %[[C0]]]
        // CHECK-SAME:                  [%[[C0]], %[[V0]]
        //      CHECK:   %[[T4:.*]] = linalg.pad_tensor %[[T1]] nofold

        // Check the statically sized matmul output with fully divisible sizes is not padded.
        //      CHECK:   %[[T5:.*]] = linalg.matmul
        // CHECK-SAME:                  ins(%[[T3]], %[[T4]] : tensor<4x7xf32>, tensor<7x5xf32>)
        // CHECK-SAME:                  outs(%[[T2]] : tensor<4x5xf32>)
        //      CHECK:   %[[T6:.*]] = tensor.insert_slice %[[T5]]
        %7 = linalg.matmul ins(%4, %5 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%6 : tensor<4x5xf32>) -> tensor<4x5xf32>
        %8 = tensor.insert_slice %7 into %arg8[%arg3, %arg5] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>

        //      CHECK:   scf.yield %[[T6]]
        scf.yield %8 : tensor<24x25xf32>
      }
      scf.yield %2 : tensor<24x25xf32>
    }
    scf.yield %1 : tensor<24x25xf32>
  }
  return %0 : tensor<24x25xf32>
}

// -----

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (7, -d0 + 25)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 7)>
#map = affine_map<(d0) -> (7, -d0 + 25)>

//      CHECK:  static_sizes_input_divisible
func @static_sizes_input_divisible(%arg0: tensor<24x12xf32>,
                                   %arg1: tensor<12x25xf32>,
                                   %arg2: tensor<24x25xf32>) -> tensor<24x25xf32> {
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[C7:.*]] = arith.constant 7
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c25 = arith.constant 25 : index
  %c24 = arith.constant 24 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c4 = arith.constant 4 : index

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg3 = %c0 to %c24 step %c4 iter_args(%arg4 = %arg2) -> (tensor<24x25xf32>) {

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %1 = scf.for %arg5 = %c0 to %c25 step %c7 iter_args(%arg6 = %arg4) -> (tensor<24x25xf32>) {

      //      CHECK:  scf.for %[[IV2:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG4:.*]] =
      %2 = scf.for %arg7 = %c0 to %c12 step %c6 iter_args(%arg8 = %arg6) -> (tensor<24x25xf32>) {
        %3 = tensor.extract_slice %arg0[%arg3, %arg7] [4, 6] [1, 1] : tensor<24x12xf32> to tensor<4x6xf32>

        //      CHECK:   %[[TS1:.*]] = affine.min #[[MAP0]](%[[IV1]])
        %4 = affine.min #map(%arg5)
        %5 = tensor.extract_slice %arg1[%arg7, %arg5] [6, %4] [1, 1] : tensor<12x25xf32> to tensor<6x?xf32>

        //      CHECK:   %[[T0:.*]] = tensor.extract_slice %[[ARG4]]
        %6 = tensor.extract_slice %arg8[%arg3, %arg5] [4, %4] [1, 1] : tensor<24x25xf32> to tensor<4x?xf32>

        // Check the statically sized matmul output with partially divisible sizes is padded.
        //      CHECK:   %[[V0:.*]] = affine.apply #[[MAP1]](%[[TS1]])
        //      CHECK:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] low
        // CHECK-SAME:                  [%[[C0]], %[[C0]]]
        // CHECK-SAME:                  [%[[C0]], %[[V0]]

        //      CHECK:   %[[T2:.*]] = linalg.matmul
        // CHECK-SAME:                  outs(%[[T1]] : tensor<4x7xf32>)
        //      CHECK:   %[[T3:.*]] = tensor.extract_slice %[[T2]]
        //      CHECK:   %[[T4:.*]] = tensor.insert_slice %[[T3]]
        %7 = linalg.matmul ins(%3, %5 : tensor<4x6xf32>, tensor<6x?xf32>) outs(%6 : tensor<4x?xf32>) -> tensor<4x?xf32>
        %8 = tensor.insert_slice %7 into %arg8[%arg3, %arg5] [4, %4] [1, 1] : tensor<4x?xf32> into tensor<24x25xf32>

        //      CHECK:   scf.yield %[[T4]]
        scf.yield %8 : tensor<24x25xf32>
      }
      scf.yield %2 : tensor<24x25xf32>
    }
    scf.yield %1 : tensor<24x25xf32>
  }
  return %0 : tensor<24x25xf32>
}

// -----

// CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0)[s0] -> (5, -d0 + s0)>
// CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0)[s0] -> (7, -d0 + s0)>
// CHECK-DAG: #[[MAP2:[0-9a-z]+]] = affine_map<(d0)[s0] -> (6, -d0 + s0)>
// CHECK-DAG: #[[MAP3:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 5)>
// CHECK-DAG: #[[MAP4:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 6)>

#map0 = affine_map<(d0)[s0] -> (5, -d0 + s0)>
#map1 = affine_map<(d0)[s0] -> (6, -d0 + s0)>
#map2 = affine_map<(d0)[s0] -> (7, -d0 + s0)>

//      CHECK:  dynamic_sizes
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
  //  CHECK-DAG: %[[D2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  //  CHECK-DAG: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>

  //      CHECK:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %3 = scf.for %arg3 = %c0 to %0 step %c5 iter_args(%arg4 = %arg2) -> (tensor<?x?xf32>) {

    //      CHECK:  scf.for %[[IV1:[0-9a-zA-Z]*]] =
    %4 = scf.for %arg5 = %c0 to %2 step %c7 iter_args(%arg6 = %arg4) -> (tensor<?x?xf32>) {

      //      CHECK:  scf.for %[[IV2:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG4:.*]] =
      %5 = scf.for %arg7 = %c0 to %1 step %c6 iter_args(%arg8 = %arg6) -> (tensor<?x?xf32>) {

        //      CHECK:   %[[TS0:.*]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0]]]
        //      CHECK:   %[[TS2:.*]] = affine.min #[[MAP2]](%[[IV2]])[%[[D2]]]
        //      CHECK:   %[[TS1:.*]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1]]]
        %6 = affine.min #map0(%arg3)[%0]
        %7 = affine.min #map1(%arg7)[%1]
        %8 = tensor.extract_slice %arg0[%arg3, %arg7] [%6, %7] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %9 = affine.min #map2(%arg5)[%2]
        %10 = tensor.extract_slice %arg1[%arg7, %arg5] [%7, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %11 = tensor.extract_slice %arg8[%arg3, %arg5] [%6, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

        // Check all matmul operands are padded.
        //      CHECK:   %[[V0:.*]] = affine.apply #[[MAP3]](%[[TS0]])
        //      CHECK:   %[[V1:.*]] = affine.apply #[[MAP4]](%[[TS2]])
        //      CHECK:   %[[T3:.*]] = linalg.pad_tensor %{{.*}} nofold
        // CHECK-SAME:                  [%[[C0]], %[[C0]]]
        // CHECK-SAME:                  [%[[V0]], %[[V1]]
        //      CHECK:   %[[T4:.*]] = linalg.pad_tensor %{{.*}} nofold
        //      CHECK:   %[[T5:.*]] = linalg.pad_tensor %{{.*}} low

        // Check the dynamic matmul has been erased.
        //  CHECK-NOT:   = linalg.matmul {{.*}} tensor<?x?xf32>

        // Check all padded matmul operands are statically sized.
        //      CHECK:   %[[T6:.*]] = linalg.matmul
        // CHECK-SAME:                  ins(%[[T3]], %[[T4]] : tensor<5x6xf32>, tensor<6x7xf32>)
        // CHECK-SAME:                  outs(%[[T5]] : tensor<5x7xf32>)
        //      CHECK:   %[[T7:.*]] = tensor.extract_slice %[[T6]][0, 0] [%[[TS0]], %[[TS1]]]
        //      CHECK:   %[[T8:.*]] = tensor.insert_slice %[[T7]]
        %12 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %13 = tensor.insert_slice %12 into %arg8[%arg3, %arg5] [%6, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

        //      CHECK:   scf.yield %[[T8]]
        scf.yield %13 : tensor<?x?xf32>
      }
      scf.yield %5 : tensor<?x?xf32>
    }
    scf.yield %4 : tensor<?x?xf32>
  }
  return %3 : tensor<?x?xf32>
}

// -----

#map = affine_map<(d0) -> (7, -d0 + 12)>

//      CHECK-FILL:  scalar_operand
// CHECK-FILL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: f32
// CHECK-FILL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<24x12xf32>
func @scalar_operand(%arg0: f32, %arg1: tensor<24x12xf32>) -> tensor<24x12xf32> {
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c24 = arith.constant 24 : index
  %c7 = arith.constant 7 : index
  %c4 = arith.constant 4 : index

  //      CHECK-FILL:  scf.for %[[IV0:[0-9a-zA-Z]*]] =
  %0 = scf.for %arg2 = %c0 to %c24 step %c4 iter_args(%arg3 = %arg1) -> (tensor<24x12xf32>) {

    //      CHECK-FILL:  scf.for %[[IV1:[0-9a-zA-Z]*]] = {{.*}} iter_args(%[[ARG2:.*]] =
    %1 = scf.for %arg4 = %c0 to %c12 step %c7 iter_args(%arg5 = %arg3) -> (tensor<24x12xf32>) {
      %2 = affine.min #map(%arg4)

      //      CHECK-FILL:   %[[T0:.*]] = tensor.extract_slice %[[ARG2]]
      //      CHECK-FILL:   %[[T1:.*]] = linalg.pad_tensor %[[T0]] nofold
      %3 = tensor.extract_slice %arg5[%arg2, %arg4] [4, %2] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>

      // Check only the fill output operand is padded.
      //      CHECK-FILL:   %[[T6:.*]] = linalg.fill(%[[ARG0]], %[[T1]]
      %4 = linalg.fill(%arg0, %3) : f32, tensor<4x?xf32> -> tensor<4x?xf32>
      %5 = tensor.insert_slice %4 into %arg5[%arg2, %arg4] [4, %2] [1, 1] : tensor<4x?xf32> into tensor<24x12xf32>
      scf.yield %5 : tensor<24x12xf32>
    }
    scf.yield %1 : tensor<24x12xf32>
  }
  return %0 : tensor<24x12xf32>
}
