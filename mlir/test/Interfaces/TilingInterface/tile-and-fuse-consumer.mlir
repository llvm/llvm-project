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
//      CHECK:          tensor.parallel_insert_slice %[[MAT_OUT]] into %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[SECOND_ARG_SLICE]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]] into %[[ELEM_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
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
      %unpack = linalg.unpack %0#0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %5 : tensor<64x32xf32> -> tensor<2048xf32>
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
//      CHECK:          tensor.parallel_insert_slice %[[MAT_OUT]] into %[[SECOND_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[SECOND_ARG_SLICE]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]]#0 into %[[ELEM_OUT_ARG_0]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[ELEM_OUT]]#1 into %[[ELEM_OUT_ARG_1]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   %[[UNPACK:.*]] = linalg.unpack %[[FINAL_RESULT]]#0 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %{{.*}} : tensor<64x32xf32> -> tensor<2048xf32>
//      CHECK:   return %[[FINAL_RESULT]]#3, %[[UNPACK]] :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @fuse_unpack_consumer_into_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<2048xf32> {
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %1 = scf.forall (%arg3, %arg4) = (0, 0) to (64, 32) step (32, 32) shared_outs(%arg5 = %arg2) -> (tensor<64x32xf32>) {
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
        %unpack = linalg.unpack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %output : tensor<64x32xf32> -> tensor<2048xf32>
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
//  CHECK-DAG: #[[UNPACK_RESULT_OFFSET_MAP:.*]] = affine_map<(d0) -> (d0 * 32)>
//  CHECK-DAG: #[[UNPACK_RESULT_SIZE_MAP:.*]] = affine_map<(d0) -> (1024, d0 * -32 + 2048)>
//      CHECK: func.func @fuse_unpack_consumer_into_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<2048xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) = (0, 0) to (64, 32) step (32, 32)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[UNPACK_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[GENERIC_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[GENERIC_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[GENERIC_OUT_SLICE]] :
//  CHECK-DAG:      %[[UNPACK_RESULT_OFFSET:.*]] = affine.apply #[[UNPACK_RESULT_OFFSET_MAP]](%[[IV1]])
//  CHECK-DAG:      %[[UNPACK_RESULT_SIZE:.*]] = affine.min #[[UNPACK_RESULT_SIZE_MAP]](%[[IV1]])
//      CHECK:      %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [%[[UNPACK_RESULT_SIZE]]] [1]
//      CHECK:      %[[TILED_UNPACK_OUT:.*]] = linalg.unpack %[[GENERIC_OUT]]
// CHECK-SAME:                              outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:                              into %[[TILED_UNPACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[GENERIC_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[TILED_UNPACK_OUT]] into %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [%[[UNPACK_RESULT_SIZE]]] [1]
//      CHECK:       }
//      CHECK:   }
//      CHECK:   return %[[FINAL_RESULT]]#1 :

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @fuse_unaligned_unpack_consumer_into_scf_forall(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<2047xf32> {
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %1 = scf.forall (%arg3, %arg4) = (0, 0) to (64, 32) step (32, 32) shared_outs(%arg5 = %arg2) -> (tensor<64x32xf32>) {
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
        %output = tensor.empty() : tensor<2047xf32>
        %unpack = linalg.unpack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %output : tensor<64x32xf32> -> tensor<2047xf32>
        return %unpack : tensor<2047xf32>
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
//  CHECK-DAG: #[[UNPACK_RESULT_OFFSET_MAP:.*]] = affine_map<(d0) -> (d0 * 32)>
//  CHECK-DAG: #[[UNPACK_RESULT_SIZE_MAP:.*]] = affine_map<(d0) -> (1024, d0 * -32 + 2047)>
//      CHECK: func.func @fuse_unaligned_unpack_consumer_into_scf_forall(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<2047xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) = (0, 0) to (64, 32) step (32, 32)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG2]], %[[UNPACK_OUT_ARG:.*]] = %[[OUT_INIT]])
// CHECK-SAME:   {
//      CHECK:      %[[GENERIC_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:      %[[GENERIC_OUT:.*]] = linalg.generic
// CHECK-SAME:              outs(%[[GENERIC_OUT_SLICE]] :
//  CHECK-DAG:      %[[UNPACK_RESULT_OFFSET:.*]] = affine.apply #[[UNPACK_RESULT_OFFSET_MAP]](%[[IV1]])
//  CHECK-DAG:      %[[UNPACK_RESULT_SIZE:.*]] = affine.min #[[UNPACK_RESULT_SIZE_MAP]](%[[IV1]])
//      CHECK:      %[[TILED_UNPACK_DEST:.*]] = tensor.extract_slice %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [%[[UNPACK_RESULT_SIZE]]] [1]
//      CHECK:      %[[TILED_UNPACK_OUT:.*]] = linalg.unpack %[[GENERIC_OUT]]
// CHECK-SAME:                              outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32]
// CHECK-SAME:                              into %[[TILED_UNPACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[GENERIC_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[TILED_UNPACK_OUT]] into %[[UNPACK_OUT_ARG]][%[[UNPACK_RESULT_OFFSET]]] [%[[UNPACK_RESULT_SIZE]]] [1]
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
        %pack = linalg.pack %1 inner_dims_pos = [0] inner_tiles = [16] into %output : tensor<64x32xf32> -> tensor<4x32x16xf32>
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
//      CHECK:      %[[TILED_PACK_OUT:.*]] = linalg.pack %[[GENERIC_OUT]]
// CHECK-SAME:                              inner_dims_pos = [0] inner_tiles = [16]
// CHECK-SAME:                              into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[GENERIC_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], %[[IV2]]] [32, 32] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[TILED_PACK_OUT]] into %[[PACK_OUT_ARG]][%[[PACK_RESULT_OFFSET]],  %[[IV2]], 0] [2, 32, 16] [1, 1, 1]

// -----

module {
  func.func @fuse_add_consumer_into_nested_scf_for(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %dest1 = linalg.fill ins(%cst : f32) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %dest1) -> (tensor<256x256xf32>) {
      %2 = scf.for %arg5 = %c0 to %c256 step %c64 iter_args(%arg6 = %arg4) -> (tensor<256x256xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
        %extracted_slice_2 = tensor.extract_slice %arg0[%arg3, 0] [64, 512] [1, 1] : tensor<256x512xf32> to tensor<64x512xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[0, %arg5] [512, 64] [1, 1] : tensor<512x256xf32> to tensor<512x64xf32>
        %3 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%extracted_slice_1 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %insert_slice = tensor.insert_slice %3 into %arg6[%arg3, %arg5] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
        scf.yield %insert_slice : tensor<256x256xf32>
      }
      scf.yield %2 : tensor<256x256xf32>
    }
    %4 = linalg.add ins(%1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %4 : tensor<256x256xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_add_consumer_into_nested_scf_for(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<256x512xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<512x256xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<256x256xf32>
//      CHECK:   %[[dest0:.*]] = tensor.empty() : tensor<256x256xf32>
//      CHECK:   %[[dest1:.*]] = linalg.fill
// CHECK-SAME:          outs(%[[dest0]] :
//      CHECK:   %[[LOOP_RESULT1:.*]]:2 = scf.for %[[IV1:.*]] = %[[C0]]
// CHECK-SAME:       iter_args(%[[FIRST_OUT_ARG1:.*]] = %[[dest1]], %[[SECOND_OUT_ARG1:.*]] = %[[dest0]])
// CHECK-SAME:   {
//      CHECK:       %[[LOOP_RESULT2:.*]]:2 = scf.for %[[IV2:.*]] = %[[C0]]
// CHECK-SAME:         iter_args(%[[FIRST_OUT_ARG2:.*]] = %[[FIRST_OUT_ARG1]], %[[SECOND_OUT_ARG2:.*]] = %[[SECOND_OUT_ARG1]])
// CHECK-SAME:         {
//      CHECK:            %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[IV1]], 0] [64, 512] [1, 1]
//      CHECK:            %[[WEIGHT_SLICE:.*]] = tensor.extract_slice %[[ARG1]][0, %[[IV2]]] [512, 64] [1, 1]
//      CHECK:            %[[TILED_MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:                  outs(%[[MAT_OUT_SLICE]] :
//      CHECK:            %[[INSERT_MAT:.*]] = tensor.insert_slice %[[TILED_MAT_OUT]] into %[[FIRST_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[ADD_OPERAND2_SLICE:.*]] = tensor.extract_slice %[[ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[ADD_OUT_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[TILED_ADD_OUT:.*]] = linalg.add
// CHECK-SAME:              ins(%[[TILED_MAT_OUT]], %[[ADD_OPERAND2_SLICE]] :
// CHECK-SAME:              outs(%[[ADD_OUT_SLICE]] :
//      CHECK:            %[[INSERT_ADD:.*]] = tensor.insert_slice %[[TILED_ADD_OUT]] into %[[SECOND_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            scf.yield %[[INSERT_MAT]], %[[INSERT_ADD]] :
//      CHECK:         }
//      CHECK:         scf.yield %[[LOOP_RESULT2]]#0, %[[LOOP_RESULT2]]#1 :
//      CHECK:   }
//      CHECK:   return %[[LOOP_RESULT1]]#1 :

// -----

// This test case checks fusion of consumer even if the producer has multiple uses.
// The multiple uses of the producer essentially means that besides the consumer
// op in concern, the only other uses of the producer are allowed in :-
// 1. scf.yield
// 2. tensor.parallel_insert_slice

module {
  module {
    func.func @fuse_consumer_for_multi_use_producer(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<256x256xf32>) -> (tensor<256x256xf32>, tensor<256x256xf32>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<256x256xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
      %2:2 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %1, %arg5 = %arg2) -> (tensor<256x256xf32>, tensor<256x256xf32>) {
        %3 = scf.for %arg6 = %c0 to %c256 step %c64 iter_args(%arg7 = %arg4) -> (tensor<256x256xf32>) {
          %extracted_slice = tensor.extract_slice %arg7[%arg3, %arg6] [64, 64] [1, 1] : tensor<256x256xf32> to tensor<64x64xf32>
          %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, 0] [64, 512] [1, 1] : tensor<256x512xf32> to tensor<64x512xf32>
          %extracted_slice_1 = tensor.extract_slice %arg1[0, %arg6] [512, 64] [1, 1] : tensor<512x256xf32> to tensor<512x64xf32>
          %5 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_1 : tensor<64x512xf32>, tensor<512x64xf32>) outs(%extracted_slice : tensor<64x64xf32>) -> tensor<64x64xf32>
          %inserted_slice = tensor.insert_slice %5 into %arg7[%arg3, %arg6] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<256x256xf32>
          scf.yield %inserted_slice : tensor<256x256xf32>
        }
        %4 = linalg.add ins(%3, %arg5 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
        scf.yield %3, %4 : tensor<256x256xf32>, tensor<256x256xf32>
      }
      return %2#0, %2#1 : tensor<256x256xf32>, tensor<256x256xf32>
    }
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %consumer, %fused_consumer = transform.test.fuse_consumer %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield
    }
  }
}
//      CHECK: func.func @fuse_consumer_for_multi_use_producer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<256x512xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<512x256xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<256x256xf32>
//      CHECK:   %[[dest0:.*]] = tensor.empty() : tensor<256x256xf32>
//      CHECK:   %[[dest1:.*]] = linalg.fill
// CHECK-SAME:          outs(%[[dest0]] :
//      CHECK:   %[[LOOP_RESULT1:.*]]:2 = scf.for %[[IV1:.*]] = %[[C0]]
// CHECK-SAME:       iter_args(%[[FIRST_OUT_ARG1:.*]] = %[[dest1]], %[[SECOND_OUT_ARG1:.*]] = %[[ARG2]])
// CHECK-SAME:   {
//      CHECK:       %[[LOOP_RESULT2:.*]]:2 = scf.for %[[IV2:.*]] = %[[C0]]
// CHECK-SAME:         iter_args(%[[FIRST_OUT_ARG2:.*]] = %[[FIRST_OUT_ARG1]], %[[SECOND_OUT_ARG2:.*]] = %[[dest0]])
// CHECK-SAME:         {
//      CHECK:            %[[MAT_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[INPUT_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[IV1]], 0] [64, 512] [1, 1]
//      CHECK:            %[[WEIGHT_SLICE:.*]] = tensor.extract_slice %[[ARG1]][0, %[[IV2]]] [512, 64] [1, 1]
//      CHECK:            %[[TILED_MAT_OUT:.*]] = linalg.matmul
// CHECK-SAME:                  outs(%[[MAT_OUT_SLICE]] :
//      CHECK:            %[[INSERT_MAT:.*]] = tensor.insert_slice %[[TILED_MAT_OUT]] into %[[FIRST_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[ADD_OPERAND2_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG1]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[ADD_OUT_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            %[[TILED_ADD_OUT:.*]] = linalg.add
// CHECK-SAME:              ins(%[[TILED_MAT_OUT]], %[[ADD_OPERAND2_SLICE]] :
// CHECK-SAME:              outs(%[[ADD_OUT_SLICE]] :
//      CHECK:            %[[INSERT_ADD:.*]] = tensor.insert_slice %[[TILED_ADD_OUT]] into %[[SECOND_OUT_ARG2]][%[[IV1]], %[[IV2]]] [64, 64] [1, 1]
//      CHECK:            scf.yield %[[INSERT_MAT]], %[[INSERT_ADD]] :
//      CHECK:         }
//      CHECK:         scf.yield %[[LOOP_RESULT2]]#0, %[[LOOP_RESULT2]]#1 :
//      CHECK:   }
//      CHECK:   return %[[LOOP_RESULT1]]#0, %[[LOOP_RESULT1]]#1 :

// -----

module {
  func.func @fuse_add_multiple_tilable_consumers(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>) -> (tensor<256x256xf32>, tensor<256x256xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %1 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %dest0) -> (tensor<256x256xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %extracted_slice_2 = tensor.extract_slice %arg0[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %3 = linalg.add ins(%extracted_slice_2, %extracted_slice_3 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%extracted_slice_1 : tensor<64x256xf32>) -> tensor<64x256xf32>
        %insert_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<64x256xf32> into tensor<256x256xf32>
        scf.yield %insert_slice : tensor<256x256xf32>
    }
    %4 = linalg.mul ins(%1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    %5 = linalg.exp ins(%1 : tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %4, %5 : tensor<256x256xf32>, tensor<256x256xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op num_consumer_to_fuse = 2
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @fuse_add_multiple_tilable_consumers(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<256x256xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<256x256xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<256x256xf32>
//      CHECK:   %[[dest0:.*]] = tensor.empty() : tensor<256x256xf32>
//      CHECK:   %[[LOOP_RESULT:.*]]:3 = scf.for %[[IV1:.*]] = %[[C0]]
// CHECK-SAME:       iter_args(%[[FIRST_OUT_ARG:.*]] = %[[dest0]], %[[SECOND_OUT_ARG:.*]] = %[[dest0]], %[[THIRD_OUT_ARG:.*]] = %[[dest0]]) 
// CHECK-SAME:   {
//      CHECK:          %[[ADD_OUT_SLICE:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[ADD_INS0_SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[ADD_INS1_SLICE:.*]] = tensor.extract_slice %[[ARG1]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[TILED_ADD_OUT:.*]] = linalg.add
// CHECK-SAME:                ins(%[[ADD_INS0_SLICE]], %[[ADD_INS1_SLICE]] :
// CHECK-SAME:                outs(%[[ADD_OUT_SLICE]] :
//      CHECK:          %[[INSERT_ADD:.*]] = tensor.insert_slice %[[TILED_ADD_OUT]] into %[[FIRST_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[EXP_OUT_SLICE:.*]] = tensor.extract_slice %[[SECOND_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[TILED_EXP_OUT:.*]] = linalg.exp
// CHECK-SAME:                ins(%[[TILED_ADD_OUT]] :
// CHECK-SAME:                outs(%[[EXP_OUT_SLICE]] :
//      CHECK:          %[[MUL_INS2_SLICE:.*]] = tensor.extract_slice %[[ARG2]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[MUL_OUT_SLICE:.*]] = tensor.extract_slice %[[THIRD_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[TILED_MUL_OUT:.*]] = linalg.mul
// CHECK-SAME:                ins(%[[TILED_ADD_OUT]], %[[MUL_INS2_SLICE]] :
// CHECK-SAME:                outs(%[[MUL_OUT_SLICE]] :
//      CHECK:          %[[INSERT_EXP:.*]] = tensor.insert_slice %[[TILED_EXP_OUT]] into %[[SECOND_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          %[[INSERT_MUL:.*]] = tensor.insert_slice %[[TILED_MUL_OUT]] into %[[THIRD_OUT_ARG]][%[[IV1]], 0] [64, 256] [1, 1]
//      CHECK:          scf.yield %[[INSERT_ADD]], %[[INSERT_EXP]], %[[INSERT_MUL]] :
//      CHECK:   }
//      CHECK:   return %[[LOOP_RESULT]]#2, %[[LOOP_RESULT]]#1 :

// -----

module {
  func.func @no_fuse_only_dps_consumer(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>) -> (tensor<256x256xf32>, tensor<258x258xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : f32
    %dest0 = tensor.empty() : tensor<256x256xf32>
    %1 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %dest0) -> (tensor<256x256xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %extracted_slice_2 = tensor.extract_slice %arg0[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %extracted_slice_3 = tensor.extract_slice %arg1[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
        %3 = linalg.add ins(%extracted_slice_2, %extracted_slice_3 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%extracted_slice_1 : tensor<64x256xf32>) -> tensor<64x256xf32>
        %insert_slice = tensor.insert_slice %3 into %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<64x256xf32> into tensor<256x256xf32>
        scf.yield %insert_slice : tensor<256x256xf32>
    }
    %dest1 = tensor.empty() : tensor<258x258xf32>
    %4 = tensor.insert_slice %1 into %dest1[0, 0] [256, 256] [1, 1] : tensor<256x256xf32> into tensor<258x258xf32>
    %5 = linalg.mul ins(%1, %arg2 : tensor<256x256xf32>, tensor<256x256xf32>) outs(%dest0 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %5, %4 : tensor<256x256xf32>, tensor<258x258xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_ops = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %slice_op, %other_slice = transform.split_handle %slice_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %slice_op num_consumer_to_fuse = 1
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: func.func @no_fuse_only_dps_consumer(
//      CHECK:   %[[LOOP_RESULT:.*]]:2 = scf.for {{.*}} {
//      CHECK:     linalg.add
//      CHECK:     linalg.mul
//      CHECK:     scf.yield
//      CHECK:   }
//      CHECK:   %[[RES_SLICE:.+]] = tensor.insert_slice
//      CHECK:   return %[[LOOP_RESULT]]#1, %[[RES_SLICE]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @fuse_with_tilable_consumer_with_projected_permutations(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>, %arg2: tensor<24xf32>) -> tensor<256x256x24xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %0 = tensor.empty() : tensor<256x256xf32>
    %1 = scf.for %arg3 = %c0 to %c256 step %c64 iter_args(%arg4 = %0) -> (tensor<256x256xf32>) {
      %extracted_slice = tensor.extract_slice %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
      %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg3, 0] [64, 256] [1, 1] : tensor<256x256xf32> to tensor<64x256xf32>
      %4 = linalg.add ins(%extracted_slice_0, %extracted_slice_1 : tensor<64x256xf32>, tensor<64x256xf32>) outs(%extracted_slice : tensor<64x256xf32>) -> tensor<64x256xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg4[%arg3, 0] [64, 256] [1, 1] : tensor<64x256xf32> into tensor<256x256xf32>
      scf.yield %inserted_slice : tensor<256x256xf32>
    }
    %2 = tensor.empty() : tensor<256x256x24xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %arg2 : tensor<256x256xf32>, tensor<24xf32>) outs(%2 : tensor<256x256x24xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<256x256x24xf32>
    return %3 : tensor<256x256x24xf32>
  }
}

// CHECK: func.func @fuse_with_tilable_consumer_with_projected_permutations(%[[VAL_0:.*]]: tensor<256x256xf32>, %[[VAL_1:.*]]: tensor<256x256xf32>, %[[VAL_2:.*]]: tensor<24xf32>) -> tensor<256x256x24xf32> {
// CHECK:             %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_5:.*]] = arith.constant 256 : index
// CHECK:             %[[VAL_6:.*]] = tensor.empty() : tensor<256x256xf32>
// CHECK:             %[[VAL_7:.*]] = tensor.empty() : tensor<256x256x24xf32>
// CHECK:             %[[VAL_8:.*]]:2 = scf.for %[[VAL_9:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_10:.*]] = %[[VAL_6]], %[[VAL_11:.*]] = %[[VAL_7]]) -> (tensor<256x256xf32>, tensor<256x256x24xf32>) {
// CHECK:               %[[VAL_12:.*]] = tensor.extract_slice %[[VAL_10]]{{\[}}%[[VAL_9]], 0] [64, 256] [1, 1]
// CHECK:               %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_9]], 0] [64, 256] [1, 1]
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_9]], 0] [64, 256] [1, 1]
// CHECK:               %[[VAL_15:.*]] = linalg.add ins(%[[VAL_13]], %[[VAL_14]] : tensor<64x256xf32>, tensor<64x256xf32>) outs(%[[VAL_12]] : tensor<64x256xf32>) -> tensor<64x256xf32>
// CHECK:               %[[VAL_16:.*]] = tensor.insert_slice %[[VAL_15]] into %[[VAL_10]]{{\[}}%[[VAL_9]], 0] [64, 256] [1, 1]
// CHECK:               %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_2]][0] [24] [1] : tensor<24xf32> to tensor<24xf32>
// CHECK:               %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_11]]{{\[}}%[[VAL_9]], 0, 0] [64, 256, 24] [1, 1, 1]
// CHECK:               %[[VAL_19:.*]] = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[VAL_15]], %[[VAL_17]] : tensor<64x256xf32>, tensor<24xf32>) outs(%[[VAL_18]] : tensor<64x256x24xf32>) {
// CHECK:               ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: f32):
// CHECK:                 %[[VAL_23:.*]] = arith.addf %[[VAL_20]], %[[VAL_21]] : f32
// CHECK:                 linalg.yield %[[VAL_23]] : f32
// CHECK:               } -> tensor<64x256x24xf32>
// CHECK:               %[[VAL_24:.*]] = tensor.insert_slice %[[VAL_25:.*]] into %[[VAL_11]]{{\[}}%[[VAL_9]], 0, 0] [64, 256, 24] [1, 1, 1]
// CHECK:               scf.yield %[[VAL_16]], %[[VAL_24]] : tensor<256x256xf32>, tensor<256x256x24xf32>
// CHECK:             }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_op = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op num_consumer_to_fuse = 1
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
