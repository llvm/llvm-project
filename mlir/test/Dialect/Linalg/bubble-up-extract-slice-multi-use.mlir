//RUN: mlir-opt -test-linalg-transform-patterns=test-bubble-up-extract-slice-op-pattern -split-input-file %s | FileCheck %s

func.func @multi_extract_slice(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>,
                   %arg2: index, %arg3: index, %arg4: index, %arg5:index,
                   %arg6: index, %arg7: index, %arg8: index, %arg9:index
                   ) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %add = arith.addf %b0, %b1 : f32
      linalg.yield %add : f32
  } -> tensor<?x?xf32>

  %1 = tensor.extract_slice %0 [%arg2, %arg3] [%arg4, %arg5] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>

  %2 = tensor.extract_slice %0 [%arg6, %arg7] [%arg8, %arg9] [1, 1]
    : tensor<?x?xf32> to tensor<?x?xf32>

  %3 = tensor.concat dim(0) %1, %2 :
    (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

  return %3 : tensor<?x?xf32>
}
//      CHECK: func @multi_extract_slice
//      CHECK: %[[SLICE0:.+]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[SLICE1:.+]] = tensor.extract_slice %arg1[%arg3] [%arg5] [1] : tensor<?xf32> to tensor<?xf32>
//      CHECK: %[[SLICE2:.+]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[GENERIC_0:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE0]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[SLICE2]] : tensor<?x?xf32>)
//
//      CHECK: %[[SLICE3:.+]] = tensor.extract_slice %arg0[%arg6, %arg7] [%arg8, %arg9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[SLICE4:.+]] = tensor.extract_slice %arg1[%arg7] [%arg9] [1] : tensor<?xf32> to tensor<?xf32>
//      CHECK: %[[SLICE5:.+]] = tensor.extract_slice %arg0[%arg6, %arg7] [%arg8, %arg9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK: %[[GENERIC_1:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[SLICE3]], %[[SLICE4]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[SLICE5]] : tensor<?x?xf32>)
//
// CHECK: %[[CONCAT:.+]] = tensor.concat dim(0) %[[GENERIC_0]], %[[GENERIC_1]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
//      CHECK: return %[[CONCAT]] : tensor<?x?xf32>

//-----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @multi_mixed_use(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>,
                     %arg3: tensor<f32>, %arg4: index, %arg5: index, %arg6: index,
                     %arg7: index) -> tensor<f32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
           ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
           outs(%arg0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %2 = arith.addf %in, %in_3 : f32
        linalg.yield %2 : f32
      } -> tensor<?x?xf32>
    %extract = tensor.extract_slice %0[%arg4, %arg5] [%arg6, %arg7] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

    %reduced = linalg.reduce { arith.addf } ins(%extract : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
    %reduced_0 = linalg.reduce { arith.addf } ins(%reduced : tensor<?xf32>) outs(%arg3 : tensor<f32>) dimensions = [0]

    %reduced_1 = linalg.reduce { arith.addf } ins(%0 : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
    %reduced_2 = linalg.reduce { arith.addf } ins(%reduced_1 : tensor<?xf32>) outs(%arg3 : tensor<f32>) dimensions = [0]

    %1 = arith.divf %reduced_0, %reduced_2 : tensor<f32>
    return %1 : tensor<f32>
  }
}

// CHECK: func @multi_mixed_use
// CHECK: %[[GENERIC:.+]] = linalg.generic  {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK: %[[EXTRACT:.+]] = tensor.extract_slice %[[GENERIC]][%arg4, %arg5] [%arg6, %arg7] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//
// CHECK: %[[REDUCED:.+]] = linalg.reduce { arith.addf } ins(%[[EXTRACT]] : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
// CHECK: %[[REDUCED_0:.+]] = linalg.reduce { arith.addf } ins(%[[REDUCED]] : tensor<?xf32>) outs(%arg3 : tensor<f32>) dimensions = [0]
//
// CHECK: %[[REDUCED_1:.+]] = linalg.reduce { arith.addf } ins(%[[GENERIC]] : tensor<?x?xf32>) outs(%arg2 : tensor<?xf32>) dimensions = [1]
// CHECK: %[[REDUCED_2:.+]] = linalg.reduce { arith.addf } ins(%[[REDUCED_1]] : tensor<?xf32>) outs(%arg3 : tensor<f32>) dimensions = [0]
//
// CHECK: %[[DIV:.+]] = arith.divf %[[REDUCED_0]], %[[REDUCED_2]] : tensor<f32>
// CHECK: return %[[DIV]] : tensor<f32>
