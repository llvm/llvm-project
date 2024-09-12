// RUN: mlir-opt --transform-interpreter --cse --split-input-file %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module {
  func.func @fuse_tileable_consumer_scf_for(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1:2 = scf.for %arg3 = %c0 to %c64 step %c4 iter_args(%arg4 = %arg2, %arg5 = %arg2) -> (tensor<64xf32>, tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg3] [32] [1] : tensor<64xf32> to tensor<32xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %13 = arith.mulf %in, %in_16 : f32
          %14 = arith.addf %out, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<32xf32>
      %4 = tensor.insert_slice %3 into %arg4[%arg3] [32] [1] : tensor<32xf32> into tensor<64xf32>
      scf.yield %arg5, %4 : tensor<64xf32>, tensor<64xf32>
    }
    %in_operand_2 = tensor.empty() : tensor<64xf32>
    %out_operand_3 = tensor.empty() : tensor<64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1#1, %in_operand_2 : tensor<64xf32>, tensor<64xf32>) outs(%out_operand_3 : tensor<64xf32>) -> tensor<64xf32>
    return %2 : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %yield = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_for(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64xf32>)
//      CHECK:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %0 = tensor.empty() : tensor<64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.for %[[IV:.*]] = %[[C0]]
// CHECK-SAME:      iter_args(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[SECOND_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG:.*]] = %0)
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] : tensor<32xf32>)
//      CHECK:      %[[INSERT_MAT:.*]] = tensor.insert_slice %[[MAT_OUT]] into %[[FIRST_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %0[%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OUT:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT]] :
//      CHECK:      %[[INSERT_ELEM:.*]] = tensor.insert_slice %[[ELEM_OUT]] into %[[ELEM_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      scf.yield %[[SECOND_OUT_ARG]], %[[INSERT_MAT]], %[[INSERT_ELEM]] :
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#2 :

// -----

module {
  func.func @fuse_tileable_consumer_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1:2 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %arg2, %arg6 = %arg2) -> (tensor<64x64xf32>, tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg4] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
      %3 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.forall.in_parallel {
         tensor.parallel_insert_slice %3 into %arg6[%arg3, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
         tensor.parallel_insert_slice %extracted_slice_1 into %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
      }
    }
    %in_operand_2 = tensor.empty() : tensor<64x64xf32>
    %out_operand_3 = tensor.empty() : tensor<64x64xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1#1, %in_operand_2 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%out_operand_3 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_ops = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %first_slice_op, %second_slice_op = transform.split_handle %slice_ops
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %first_slice_op
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x64xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<64x64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:3 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 2)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[SECOND_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[SECOND_ARG_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] :
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %[[OUT_INIT]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[SLICE_OUT:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT]] :
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]] into %[[ELEM_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[MAT_OUT]] into %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[SECOND_ARG_SLICE]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#2 :

// -----

#map = affine_map<(d0) -> (d0)>
module {
  func.func @fuse_tileable_consumer_scf_for_multi_yielding_consumer(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1:2 = scf.for %arg3 = %c0 to %c64 step %c4 iter_args(%arg4 = %arg2, %arg5 = %arg2) -> (tensor<64xf32>, tensor<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg3] [32] [1] : tensor<64xf32> to tensor<32xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%extracted_slice : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %13 = arith.mulf %in, %in_16 : f32
          %14 = arith.addf %out, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<32xf32>
      %4 = tensor.insert_slice %3 into %arg4[%arg3] [32] [1] : tensor<32xf32> into tensor<64xf32>
      scf.yield %arg5, %4 : tensor<64xf32>, tensor<64xf32>
    }
    %in_operand_2 = tensor.empty() : tensor<64xf32>
    %out_operand_3 = tensor.empty() : tensor<64xf32>
    %out_operand_4 = tensor.empty() : tensor<64xf32>
    %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%1#1, %in_operand_2 : tensor<64xf32>, tensor<64xf32>) outs(%out_operand_3, %out_operand_4 : tensor<64xf32>, tensor<64xf32>) {
      ^bb0(%in: f32, %in_16: f32, %out_0: f32, %out_1: f32):
          %13 = arith.mulf %in, %in_16 : f32
          %14 = arith.subf %out_0, %13 : f32
          %15 = arith.addf %out_1, %in : f32
          linalg.yield %14, %15 : f32, f32
    } -> (tensor<64xf32>, tensor<64xf32>)
    return %2#1 : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %yield = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_for_multi_yielding_consumer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64xf32>)
//      CHECK:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   %0 = tensor.empty() : tensor<64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:4 = scf.for %[[IV:.*]] = %[[C0]]
// CHECK-SAME:      iter_args(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[SECOND_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG_0:.*]] = %0, %[[ELEM_OUT_ARG_1:.*]] = %0)
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] : tensor<32xf32>)
//      CHECK:      %[[INSERT_MAT:.*]] = tensor.insert_slice %[[MAT_OUT]] into %[[FIRST_OUT_ARG]][%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %0[%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OUT_0:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG_0]][%[[IV]]] [32] [1]
//      CHECK:      %[[SLICE_OUT_1:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG_1]][%[[IV]]] [32] [1]
//      CHECK:      %[[ELEM_OUT:.*]]:2 = linalg.generic
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT_0]], %[[SLICE_OUT_1]] :
//      CHECK:      %[[INSERT_ELEM_0:.*]] = tensor.insert_slice %[[ELEM_OUT]]#0 into %[[ELEM_OUT_ARG_0]][%[[IV]]] [32] [1]
//      CHECK:      %[[INSERT_ELEM_1:.*]] = tensor.insert_slice %[[ELEM_OUT]]#1 into %[[ELEM_OUT_ARG_1]][%[[IV]]] [32] [1]
//      CHECK:      scf.yield %[[SECOND_OUT_ARG]], %[[INSERT_MAT]], %[[INSERT_ELEM_0]], %[[INSERT_ELEM_1]] :
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#3 :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @fuse_tileable_consumer_scf_forall_multi_yielding_consumer(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x32xf32>) -> (tensor<64x64xf32>, tensor<2048xf32>) {
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %0:2 = scf.forall (%arg4, %arg5) in (2, 2) shared_outs(%arg6 = %arg3, %arg7 = %arg2) -> (tensor<64x32xf32>, tensor<64x64xf32>) {
        %extracted_slice = tensor.extract_slice %arg6[%arg4, %arg5] [32, 32] [1, 1] : tensor<64x32xf32> to tensor<32x32xf32>
        %extracted_slice_0 = tensor.extract_slice %arg7[%arg4, %arg5] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
        %6 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %6 into %arg7[%arg4, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x64xf32>
          tensor.parallel_insert_slice %extracted_slice_0 into %arg6[%arg4, %arg5] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x32xf32>
        }
      }
      %1 = tensor.empty() : tensor<64x64xf32>
      %2 = tensor.empty() : tensor<64x64xf32>
      %3 = tensor.empty() : tensor<64x64xf32>
      %4:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0#1, %1 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%2, %3 : tensor<64x64xf32>, tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32, %out_1: f32):
        %6 = arith.mulf %in, %in_0 : f32
        %7 = arith.subf %out, %6 : f32
        %8 = arith.addf %out_1, %in : f32
        linalg.yield %7, %8 : f32, f32
      } -> (tensor<64x64xf32>, tensor<64x64xf32>)
      %5 = tensor.empty() : tensor<2048xf32>
      %unpack = tensor.unpack %0#0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %5 : tensor<64x32xf32> -> tensor<2048xf32>
      return %4#1, %unpack : tensor<64x64xf32>, tensor<2048xf32>
    }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_ops = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %first_slice_op, %second_slice_op = transform.split_handle %slice_ops
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %first_slice_op
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_tileable_consumer_scf_forall_multi_yielding_consumer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x64xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<64x64xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:4 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 2)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG3]], %[[SECOND_OUT_ARG:.*]] = %[[ARG2]], %[[ELEM_OUT_ARG_0:.*]] = %[[OUT_INIT]], %[[ELEM_OUT_ARG_1:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[SECOND_ARG_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:              outs(%[[MAT_OUT_SLICE]] :
//      CHECK:      %[[SLICE_OPERAND2:.*]] = tensor.extract_slice %[[OUT_INIT]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[SLICE_OUT_0:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG_0]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[SLICE_OUT_1:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG_1]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[ELEM_OUT:.*]]:2 = linalg.generic
// CHECK-SAME:              ins(%[[MAT_OUT]], %[[SLICE_OPERAND2]] :
// CHECK-SAME:              outs(%[[SLICE_OUT_0]], %[[SLICE_OUT_1]] :
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]]#0 into %[[ELEM_OUT_ARG_0]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]]#1 into %[[ELEM_OUT_ARG_1]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[MAT_OUT]] into %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[SECOND_ARG_SLICE]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   %[[UNPACK:.*]] = tensor.unpack %[[FINAL_RESULT]]#0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %{{.*}} : tensor<64x32xf32> -> tensor<2048xf32>
//      CHECK:   return %[[FINAL_RESULT]]#3, %[[UNPACK]] :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @fuse_unpack_consumer_into_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<2048xf32> {
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %1 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %arg2) -> (tensor<64x32xf32>) {
            %extracted_slice = tensor.extract_slice %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<64x32xf32> to tensor<32x32xf32>
            %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_16: f32, %out: f32):
                %13 = arith.mulf %in, %in_16 : f32
                %14 = arith.addf %out, %13 : f32
                linalg.yield %14 : f32
            } -> tensor<32x32xf32>
            scf.forall.in_parallel {
                tensor.parallel_insert_slice %3 into %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x32xf32>
            }
        }
        %output = tensor.empty() : tensor<2048xf32>
        %unpack = tensor.unpack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %output : tensor<64x32xf32> -> tensor<2048xf32>
        return %unpack : tensor<2048xf32>
    }
}
  
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
        %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
        %a, %b = transform.test.fuse_consumer %slice_op
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
}
//      CHECK: #[[UNPACK_RESULT_MAP:.*]] = affine_map<(d0) -> (d0 * 32)>
//      CHECK: func.func @fuse_unpack_consumer_into_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<2048xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 2)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[UNPACK_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[GENERIC_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[GENERIC_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[GENERIC_OUT_SLICE]] :
//      CHECK:      %[[UNPACK_RESULT_OFFSET:.*]] = affine.apply #[[UNPACK_RESULT_MAP]](%[[IV1]])
//      CHECK:      %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [1024] [1]
//      CHECK:      %[[TILED_UNPACK_OUT:.*]] = tensor.unpack %[[GENERIC_OUT]]
// CHECK-SAME:                              outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:                              into %[[TILED_UNPACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[TILED_UNPACK_OUT]] into %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [1024] [1]
//      CHECK:          tensor.parallel_insert_slice %[[GENERIC_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#1 :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @fuse_pack_consumer_into_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<4x32x16xf32> {
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %1 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %arg2) -> (tensor<64x32xf32>) {
            %extracted_slice = tensor.extract_slice %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<64x32xf32> to tensor<32x32xf32>
            %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_16: f32, %out: f32):
                %13 = arith.mulf %in, %in_16 : f32
                %14 = arith.addf %out, %13 : f32
                linalg.yield %14 : f32
            } -> tensor<32x32xf32>
            scf.forall.in_parallel {
                tensor.parallel_insert_slice %3 into %arg5[%arg3, %arg4] [32, 32] [1, 1] : tensor<32x32xf32> into tensor<64x32xf32>
            }
        }
        %output = tensor.empty() : tensor<4x32x16xf32>
        %pack = tensor.pack %1 inner_dims_pos = [0] inner_tiles = [16] into %output : tensor<64x32xf32> -> tensor<4x32x16xf32>
        return %pack : tensor<4x32x16xf32>
    }
}
  
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
        %slice_op = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
        %a, %b = transform.test.fuse_consumer %slice_op
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
}
//      CHECK: #[[PACK_RESULT_MAP:.*]] = affine_map<(d0) -> (d0 floordiv 16)>
//      CHECK: func.func @fuse_pack_consumer_into_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<4x32x16xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 2)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[PACK_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[GENERIC_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[GENERIC_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[GENERIC_OUT_SLICE]] :
//      CHECK:      %[[PACK_RESULT_OFFSET:.*]] = affine.apply #[[PACK_RESULT_MAP]](%[[IV1]])
//      CHECK:      %[[TILED_PACK_DEST:.*]] = tensor.extract_slice %[[PACK_OUT_ARG]][%[[PACK_RESULT_OFFSET]], %[[IV2]], 0] [2, 32, 16] [1, 1, 1]
//      CHECK:      %[[TILED_PACK_OUT:.*]] = tensor.pack %[[GENERIC_OUT]]
// CHECK-SAME:                              inner_dims_pos = [0] inner_tiles = [16]
// CHECK-SAME:                              into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[TILED_PACK_OUT]] into %[[PACK_OUT_ARG]][%[[PACK_RESULT_OFFSET]],  %[[IV2]], 0] [2, 32, 16] [1, 1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[GENERIC_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#1 :
