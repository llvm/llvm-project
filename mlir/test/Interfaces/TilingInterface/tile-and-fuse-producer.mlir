// RUN: mlir-opt --transform-interpreter --cse --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0 * 128)>
module {
  func.func @gemm_fill_fusion_multi_level_extract_slice(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %dest1 = linalg.fill ins(%cst : f32) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %dest1) -> tensor<256x256xf32> {
      %iv0 = affine.apply #map(%arg3)
      %iv1 = affine.apply #map(%arg4)
      %extracted_slice_1 = tensor.extract_slice %arg5[%iv0, %iv1] [128, 128] [1, 1] : tensor<256x256xf32> to tensor<128x128xf32>
      %extracted_slice_2 = tensor.extract_slice %arg0[%iv0, 0] [128, 512] [1, 1] : tensor<256x512xf32> to tensor<128x512xf32>
      %extracted_slice_3 = tensor.extract_slice %arg1[0, %iv1] [512, 128] [1, 1] : tensor<512x256xf32> to tensor<512x128xf32>
      %2 = scf.for %arg6 = %c0 to %c128 step %c64 iter_args(%arg7 = %extracted_slice_1) -> (tensor<128x128xf32>) {
        %3 = scf.for %arg8 = %c0 to %c128 step %c64 iter_args(%arg9 = %arg7) -> (tensor<128x128xf32>) {
          %extracted_slice_4 = tensor.extract_slice %arg9[%arg6, %arg8] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
          %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[%arg6, 0] [64, 512] [1, 1] : tensor<128x512xf32> to tensor<64x512xf32>
          %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg8] [512, 64] [1, 1] : tensor<512x128xf32> to tensor<512x64xf32>
          %4 = linalg.matmul ins(%extracted_slice_5, %extracted_slice_6 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%extracted_slice_4 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %insert_slice = tensor.insert_slice %4 into %arg9[%arg6, %arg8] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
          scf.yield %insert_slice : tensor<128x128xf32>
        }
        scf.yield %3 : tensor<128x128xf32>
      }
      scf.forall.in_parallel {
         tensor.parallel_insert_slice %2 into %arg5[%iv0, %iv1] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<256x256xf32>
      }
    }
    return %1 : tensor<256x256xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.get_producer_of_operand %matmul[2]
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_producer %yield
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

//      CHECK: #[[MAP0:.*]] =  affine_map<(d0) -> (d0 * 128)>
//      CHECK: func.func @gemm_fill_fusion_multi_level_extract_slice(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<256x512xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<512x256xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<256x256xf32>
//      CHECK:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %[[dest0:.*]] = tensor.empty() : tensor<256x256xf32>
//      CHECK:   %[[FORALL_RESULT:.*]] = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 2)
// CHECK-SAME:      shared_outs(%[[INIT_ARG0:.*]] = %[[dest0]])
// CHECK-SAME:   {
//      CHECK:      %[[AFFINE_IV1:.*]] = affine.apply #[[MAP0]](%[[IV1]])
//      CHECK:      %[[AFFINE_IV2:.*]] = affine.apply #[[MAP0]](%[[IV2]])
//      CHECK:      %[[FILL_OUT_SLICE0:.*]] = tensor.extract_slice %[[INIT_ARG0]][%[[AFFINE_IV1]], %[[AFFINE_IV2]]] [128, 128] [1, 1]
//      CHECK:      %[[INPUT_SLICE0:.*]] = tensor.extract_slice %[[ARG0]][%[[AFFINE_IV1]], 0] [128, 512] [1, 1]
//      CHECK:      %[[WEIGHT_SLICE0:.*]] = tensor.extract_slice %[[ARG1]][0, %[[AFFINE_IV2]]] [512, 128] [1, 1]
//      CHECK:      %[[LOOP_RESULT1:.*]] = scf.for %[[IV3:.*]] = %[[C0]]
// CHECK-SAME:          iter_args(%[[INIT_ARG1:.*]] = %[[FILL_OUT_SLICE0]])
// CHECK-SAME:      {
//      CHECK:          %[[LOOP_RESULT2:.*]] = scf.for %[[IV4:.*]] = %[[C0]]
// CHECK-SAME:            iter_args(%[[INIT_ARG2:.*]] = %[[INIT_ARG1]])
// CHECK-SAME:          {
//      CHECK:            %[[FILL_OUT_SLICE1:.*]] = tensor.extract_slice %[[INIT_ARG2]][%[[IV3]], %[[IV4]]] [64, 64] [1, 1]
//      CHECK:            %[[TILED_FILL_OUT:.*]] = linalg.fill
// CHECK-SAME:                  outs(%[[FILL_OUT_SLICE1]] :
//      CHECK:            %[[INPUT_SLICE1:.*]] = tensor.extract_slice %[[INPUT_SLICE0]][%[[IV3]], 0] [64, 512] [1, 1]
//      CHECK:            %[[WEIGHT_SLICE1:.*]] = tensor.extract_slice %[[WEIGHT_SLICE0]][0, %[[IV4]]] [512, 64] [1, 1]
//      CHECK:            %[[TILED_MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:                  outs(%[[TILED_FILL_OUT]] :
//      CHECK:            %[[INSERT_MAT:.*]] = tensor.insert_slice %[[TILED_MAT_OUT]] into %[[INIT_ARG2]][%[[IV3]], %[[IV4]]] [64, 64] [1, 1]
//      CHECK:            scf.yield %[[INSERT_MAT]] :
//      CHECK:          }
//      CHECK:          scf.yield %[[LOOP_RESULT2]] :
//      CHECK:      }
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[LOOP_RESULT1]] into %[[INIT_ARG0]][%[[AFFINE_IV1]], %[[AFFINE_IV2]]] [128, 128] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FORALL_RESULT]] :