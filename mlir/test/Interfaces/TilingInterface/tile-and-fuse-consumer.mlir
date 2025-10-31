// RUN: mlir-opt --transform-interpreter --cse --split-input-file --verify-diagnostics %s | FileCheck %s

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
    %2 = linalg.add ins(%1#1, %in_operand_2 : tensor<64xf32>, tensor<64xf32>) outs(%out_operand_3 : tensor<64xf32>) -> tensor<64xf32>
    return %2 : tensor<64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.structured.match ops{["tensor.insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield in (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.add
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
    %2 = linalg.add ins(%1#1, %in_operand_2 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%out_operand_3 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %2 : tensor<64x64xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %slice_ops = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %first_slice_op, %second_slice_op = transform.split_handle %slice_ops
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %first_slice_op in (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
//      CHECK:      %[[ELEM_OUT:.*]] = linalg.add
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
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %yield in (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %first_slice_op, %second_slice_op = transform.split_handle %slice_ops
        : (!transform.any_op)
        -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %first_slice_op in (%loop)
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
  func.func @fuse_perfect_tiling_pack_consumer(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<4x32x16xf32> {
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = scf.forall (%arg3, %arg4) in (2, 1) shared_outs(%arg5 = %arg2) -> (tensor<64x32xf32>) {
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
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop)
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[PACK_RESULT_MAP:.*]] = affine_map<(d0) -> (d0 floordiv 16)>
//      CHECK: func.func @fuse_perfect_tiling_pack_consumer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<32x32xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<64x32xf32>)
//      CHECK:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<4x32x16xf32>
//      CHECK:   %[[FINAL_RESULT:.*]]:2 = scf.forall (%[[IV1:.*]], %[[IV2:.*]]) in (2, 1)
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

#map = affine_map<(d0) -> (-d0 + 4, 16)>
func.func @fuse_pack_consumer_if_single_iteration(%arg0: tensor<4x4xf32>) -> tensor<1x4x16x1xf32> {
  %0 = tensor.empty() : tensor<1x4x16x1xf32>
  %1 = tensor.empty() : tensor<4x4xf32>
  %2 = scf.forall (%arg1) = (0) to (4) step (16) shared_outs(%arg2 = %1) -> (tensor<4x4xf32>) {
    %3 = affine.min #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%arg1, 0] [%3, 4] [1, 1] : tensor<4x4xf32> to tensor<?x4xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%arg1, 0] [%3, 4] [1, 1] : tensor<4x4xf32> to tensor<?x4xf32>
    %4 = linalg.exp ins(%extracted_slice : tensor<?x4xf32>) outs(%extracted_slice_0 : tensor<?x4xf32>) -> tensor<?x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %4 into %arg2[%arg1, 0] [%3, 4] [1, 1] : tensor<?x4xf32> into tensor<4x4xf32>
    }
  }
  %cst = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %2 padding_value(%cst : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %0 : tensor<4x4xf32> -> tensor<1x4x16x1xf32>
  return %pack : tensor<1x4x16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer, %fused_consumer = transform.test.fuse_consumer %0 in(%1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[MAP:.*]] = affine_map<(d0) -> (-d0 + 4, 16)>
//      CHECK: func.func @fuse_pack_consumer_if_single_iteration(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[PACK_INIT:.*]] = tensor.empty() : tensor<1x4x16x1xf32>
//  CHECK-DAG:   %[[ELEM_INIT:.*]] = tensor.empty() : tensor<4x4xf32>
//  CHECK-DAG:   %[[PAD_VAL:.*]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %{{.*}}:2 = scf.forall (%[[IV:.*]]) = (0) to (4) step (16)
// CHECK-SAME:      shared_outs(%[[ELEM_OUT_ARG:.*]] = %[[ELEM_INIT]], %[[PACK_OUT_ARG:.*]] = %[[PACK_INIT]])
//  CHECK-DAG:      %[[SIZE:.+]] = affine.min #[[MAP]](%[[IV]])
//  CHECK-DAG:      %[[ELEM_SRC:.*]] = tensor.extract_slice %[[ARG0]][%[[IV]], 0] [%[[SIZE]], 4] [1, 1]
//  CHECK-DAG:      %[[ELEM_DEST:.*]] = tensor.extract_slice %[[ELEM_OUT_ARG]][%[[IV]], 0] [%[[SIZE]], 4] [1, 1]
//      CHECK:      %[[ELEM:.*]] = linalg.exp
// CHECK-SAME:        ins(%[[ELEM_SRC]]
// CHECK-SAME:        outs(%[[ELEM_DEST]]
//  CHECK-DAG:      %[[TILED_PACK_DEST:.*]] = tensor.extract_slice %[[PACK_OUT_ARG]][%[[IV]], 0, 0, 0] [1, 4, 16, 1] [1, 1, 1, 1]
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[ELEM]]
// CHECK-SAME:        padding_value(%[[PAD_VAL]] : f32)
// CHECK-SAME:        outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 1]
// CHECK-SAME:        into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM]] into %[[ELEM_OUT_ARG]][%[[IV]], 0] [%[[SIZE]], 4] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[PACK]] into %[[PACK_OUT_ARG]][%[[IV]], 0, 0, 0] [1, 4, 16, 1] [1, 1, 1, 1]

// -----

func.func @fuse_perfect_tiling_pack_consumer_with_outer_dims_perm(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<2x64x16x1xf32>) -> tensor<2x64x16x1xf32> {
  %0 = scf.forall (%arg3) = (0) to (32) step (16) shared_outs(%arg4 = %arg1) -> (tensor<64x32xf32>) {
    %src = tensor.extract_slice %arg0[0, %arg3] [64, 16] [1, 1] : tensor<64x32xf32> to tensor<64x16xf32>
    %dest = tensor.extract_slice %arg4[0, %arg3] [64, 16] [1, 1] : tensor<64x32xf32> to tensor<64x16xf32>
    %1 = linalg.exp ins(%src : tensor<64x16xf32>) outs(%dest : tensor<64x16xf32>) -> tensor<64x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg4[0, %arg3] [64, 16] [1, 1] : tensor<64x16xf32> into tensor<64x32xf32>
    }
  }
  %pack = linalg.pack %0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1] into %arg2 : tensor<64x32xf32> -> tensor<2x64x16x1xf32>
  return %pack : tensor<2x64x16x1xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer, %fused_consumer = transform.test.fuse_consumer %0 in(%1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[PACK_RESULT_MAP:.*]] = affine_map<(d0) -> (d0 floordiv 16)>
//      CHECK: func.func @fuse_perfect_tiling_pack_consumer_with_outer_dims_perm(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]
//      CHECK:   %{{.*}}:2 = scf.forall (%[[IV:.*]]) = (0) to (32) step (16)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG1]], %[[PACK_OUT_ARG:.*]] = %[[ARG2]])
//      CHECK:      %[[ELEM_SRC:.*]] = tensor.extract_slice %[[ARG0]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:      %[[ELEM_DEST:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:      %[[ELEM:.*]] = linalg.exp
// CHECK-SAME:        ins(%[[ELEM_SRC]]
// CHECK-SAME:        outs(%[[ELEM_DEST]]
//  CHECK-DAG:      %[[PACK_RESULT_OFFSET:.*]] = affine.apply #[[PACK_RESULT_MAP]](%[[IV]])
//  CHECK-DAG:      %[[TILED_PACK_DEST:.*]] = tensor.extract_slice %[[PACK_OUT_ARG]][%[[PACK_RESULT_OFFSET]], 0, 0, 0] [1, 64, 16, 1] [1, 1, 1, 1]
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[ELEM]]
// CHECK-SAME:        outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 1]
// CHECK-SAME:        into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM]] into %[[FIRST_OUT_ARG]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[PACK]] into %[[PACK_OUT_ARG]][%[[PACK_RESULT_OFFSET]], 0, 0, 0] [1, 64, 16, 1] [1, 1, 1, 1]

// -----

// It is valid to fuse the pack op in perfect tiling scenario when the dimension
// is dynamic and padding is not needed.

func.func @fuse_pack_consumer_with_no_pad_dynamic_dim(%arg0: tensor<64x?xf32>, %arg1: tensor<64x?xf32>, %1: tensor<64x?x16xf32>) -> tensor<64x?x16xf32> {
  %c1 = arith.constant 1 : index
  %d1 = tensor.dim %arg0, %c1 : tensor<64x?xf32>
  %0 = scf.forall (%arg2) = (0) to (%d1) step (16) shared_outs(%arg3 = %arg1) -> (tensor<64x?xf32>) {
    %src = tensor.extract_slice %arg0[0, %arg2] [64, 16] [1, 1] : tensor<64x?xf32> to tensor<64x16xf32>
    %dest = tensor.extract_slice %arg3[0, %arg2] [64, 16] [1, 1] : tensor<64x?xf32> to tensor<64x16xf32>
    %2 = linalg.exp ins(%src : tensor<64x16xf32>) outs(%dest : tensor<64x16xf32>) -> tensor<64x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg3[0, %arg2] [64, 16] [1, 1] : tensor<64x16xf32> into tensor<64x?xf32>
    }
  }
  %pack = linalg.pack %0 inner_dims_pos = [1] inner_tiles = [16] into %1 : tensor<64x?xf32> -> tensor<64x?x16xf32>
  return %pack : tensor<64x?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer, %fused_consumer = transform.test.fuse_consumer %0 in(%1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//      CHECK: #[[PACK_RESULT_MAP:.*]] = affine_map<(d0) -> (d0 floordiv 16)>
//      CHECK: func.func @fuse_pack_consumer_with_no_pad_dynamic_dim(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]
//      CHECK:   %{{.*}}:2 = scf.forall (%[[IV:.*]]) = (0) to (%{{.+}}) step (16)
// CHECK-SAME:      shared_outs(%[[FIRST_OUT_ARG:.*]] = %[[ARG1]], %[[PACK_OUT_ARG:.*]] = %[[ARG2]])
//      CHECK:      %[[ELEM_SRC:.*]] = tensor.extract_slice %[[ARG0]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:      %[[ELEM_DEST:.*]] = tensor.extract_slice %[[FIRST_OUT_ARG]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:      %[[ELEM:.*]] = linalg.exp
// CHECK-SAME:        ins(%[[ELEM_SRC]]
// CHECK-SAME:        outs(%[[ELEM_DEST]]
//  CHECK-DAG:      %[[PACK_RESULT_OFFSET:.*]] = affine.apply #[[PACK_RESULT_MAP]](%[[IV]])
//  CHECK-DAG:      %[[TILED_PACK_DEST:.*]] = tensor.extract_slice %[[PACK_OUT_ARG]][0, %[[PACK_RESULT_OFFSET]], 0] [64, 1, 16] [1, 1, 1]
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[ELEM]]
// CHECK-SAME:        inner_dims_pos = [1] inner_tiles = [16]
// CHECK-SAME:        into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM]] into %[[FIRST_OUT_ARG]][0, %[[IV]]] [64, 16] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[PACK]] into %[[PACK_OUT_ARG]][0, %[[PACK_RESULT_OFFSET]], 0] [64, 1, 16] [1, 1, 1]

// -----

// It is valid to fuse the pack op with padding semantics if it is a perfect
// tiling case.

func.func @fuse_pack_consumer_with_padding_semantics(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<22x2x3x16xf32> {
  %0 = scf.forall (%arg2, %arg3) = (0, 0) to (64, 32) step (15, 16) shared_outs(%arg4 = %arg1) -> (tensor<64x32xf32>) {
    %size = affine.min affine_map<(d0) -> (-d0 + 64, 15)>(%arg2)
    %src = tensor.extract_slice %arg0[%arg2, %arg3] [%size, 16] [1, 1] : tensor<64x32xf32> to tensor<?x16xf32>
    %dest = tensor.extract_slice %arg4[%arg2, %arg3] [%size, 16] [1, 1] : tensor<64x32xf32> to tensor<?x16xf32>
    %2 = linalg.exp ins(%src : tensor<?x16xf32>) outs(%dest : tensor<?x16xf32>) -> tensor<?x16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %2 into %arg4[%arg2, %arg3] [%size, 16] [1, 1] : tensor<?x16xf32> into tensor<64x32xf32>
    }
  }
  %1 = tensor.empty() : tensor<22x2x3x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %0 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [3, 16] into %1 : tensor<64x32xf32> -> tensor<22x2x3x16xf32>
  return %pack : tensor<22x2x3x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer, %fused_consumer = transform.test.fuse_consumer %0 in(%1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
//  CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (-d0 + 64, 15)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 floordiv 3)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0) -> (d0 ceildiv 3)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0) -> (d0 floordiv 16)>
//      CHECK: func.func @fuse_pack_consumer_with_padding_semantics(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]
//  CHECK-DAG:   %[[OUT_INIT:.*]] = tensor.empty() : tensor<22x2x3x16xf32>
//  CHECK-DAG:   %[[PAD_VAL:.*]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %{{.*}}:2 = scf.forall (%[[I:.*]], %[[J:.*]]) = (0, 0) to (64, 32) step (15, 16)
// CHECK-SAME:      shared_outs(%[[ELEM_OUT:.*]] = %[[ARG1]], %[[PACK_OUT:.*]] = %[[OUT_INIT]])
//      CHECK:      %[[SIZE:.+]] = affine.min #[[MAP0]](%[[I]])
//      CHECK:      %[[ELEM_SRC:.*]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:        [%[[I]], %[[J]]] [%[[SIZE]], 16] [1, 1]
//      CHECK:      %[[ELEM_DEST:.*]] = tensor.extract_slice %[[ELEM_OUT]]
// CHECK-SAME:        [%[[I]], %[[J]]] [%[[SIZE]], 16] [1, 1]
//      CHECK:      %[[ELEM:.*]] = linalg.exp
// CHECK-SAME:        ins(%[[ELEM_SRC]]
// CHECK-SAME:        outs(%[[ELEM_DEST]]
//  CHECK-DAG:      %[[D0_OFFSET:.*]] = affine.apply #[[MAP1]](%[[I]])
//  CHECK-DAG:      %[[D0_SIZE:.*]] = affine.apply #[[MAP2]](%[[SIZE]])
//  CHECK-DAG:      %[[D1_OFFSET:.*]] = affine.apply #[[MAP3]](%[[J]])
//  CHECK-DAG:      %[[PACK_INIT:.*]] = tensor.extract_slice %[[PACK_OUT]]
// CHECK-SAME:        [%[[D0_OFFSET]], %[[D1_OFFSET]], 0, 0] [%[[D0_SIZE]], 1, 3, 16] [1, 1, 1, 1]
//      CHECK:      %[[PACK:.*]] = linalg.pack %[[ELEM]]
// CHECK-SAME:        padding_value(%[[PAD_VAL]] : f32)
// CHECK-SAME:        inner_dims_pos = [0, 1] inner_tiles = [3, 16]
// CHECK-SAME:        into %[[TILED_PACK_DEST]]
//      CHECK:      scf.forall.in_parallel {
//      CHECK:          tensor.parallel_insert_slice %[[ELEM]] into %[[ELEM_OUT]]
// CHECK-SAME:            [%[[I]], %[[J]]] [%[[SIZE]], 16] [1, 1]
//      CHECK:          tensor.parallel_insert_slice %[[PACK]] into %[[PACK_OUT]]
// CHECK-SAME:            [%[[D0_OFFSET]], %[[D1_OFFSET]], 0, 0] [%[[D0_SIZE]], 1, 3, 16] [1, 1, 1, 1]

// -----

// Imperfect tiling is not supported in pack op consumer fusion.

#map = affine_map<(d0) -> (d0 * 5)>
#map1 = affine_map<(d0) -> (d0)>
func.func @nofuse_pack_with_imperfect_tiling(%arg0: tensor<30xf32>) -> tensor<5x6xf32> {
  %0 = tensor.empty() : tensor<30xf32>
  %1 = scf.forall (%arg1) in (6) shared_outs(%arg2 = %0) -> (tensor<30xf32>) {
    %3 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %arg0[%3] [5] [1] : tensor<30xf32> to tensor<5xf32>
    %extracted_slice_0 = tensor.extract_slice %arg2[%3] [5] [1] : tensor<30xf32> to tensor<5xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<5xf32>) outs(%extracted_slice_0 : tensor<5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<5xf32>
    scf.forall.in_parallel {
      // expected-error @below {{failed to fuse consumer of slice}}
      tensor.parallel_insert_slice %4 into %arg2[%3] [5] [1] : tensor<5xf32> into tensor<30xf32>
    }
  }
  %2 = tensor.empty() : tensor<5x6xf32>
  %pack = linalg.pack %1 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [6] into %2 : tensor<30xf32> -> tensor<5x6xf32>
  return %pack : tensor<5x6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %consumer, %fused_consumer = transform.test.fuse_consumer %0 in(%1) : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

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
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop) num_consumer_to_fuse = 2
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %loop = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %slice_op, %other_slice = transform.split_handle %slice_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop) num_consumer_to_fuse = 1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
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
    %loop = transform.structured.match ops{["scf.for"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %a, %b = transform.test.fuse_consumer %slice_op in (%loop) num_consumer_to_fuse = 1
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @multi_slice_fusion1(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>, %arg3 : index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %loop:2 = scf.forall (%iv0) =  (%c0) to (%dim0) step (%arg3) shared_outs(%init0 = %arg1, %init1 = %arg2) -> (tensor<?xf32>, tensor<?xf32>) {
    %tilesize = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv0)[%dim0, %arg3]
    %arg0_slice = tensor.extract_slice %arg0[%iv0, 0] [%tilesize, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %init0_slice = tensor.extract_slice %init0[%iv0] [%tilesize] [1] : tensor<?xf32> to tensor<?xf32>
    %init1_slice = tensor.extract_slice %init1[%iv0] [%tilesize] [1] : tensor<?xf32> to tensor<?xf32>
    %generic:2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
	iterator_types = ["parallel", "reduction"]}
	ins(%arg0_slice : tensor<?x?xf32>) outs(%init0_slice, %init1_slice : tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
        %0 = arith.mulf %b0, %b1 : f32
	%1 = arith.addf %b0, %b2 : f32
	linalg.yield %0, %1 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %generic#0 into %init0[%iv0] [%tilesize] [1] : tensor<?xf32> into tensor<?xf32>
      tensor.parallel_insert_slice %generic#1 into %init1[%iv0] [%tilesize] [1] : tensor<?xf32> into tensor<?xf32>
    }
  }
  %empty = tensor.empty(%dim0) : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%loop#0, %loop#1 : tensor<?xf32>, tensor<?xf32>) outs(%empty : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
// CHECK-LABEL: func @multi_slice_fusion1(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[DIM0]])
//       CHECK:   %[[RESULT:.+]]:3 = scf.forall (%[[IV:.+]]) =
//  CHECK-SAME:       , %[[INIT:[a-zA-Z0-9]+]] = %[[EMPTY]])
//       CHECK:     %[[TILESIZE:.+]] = affine.min
//   CHECK-DAG:     %[[GENERIC:.+]]:2 = linalg.generic
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]]] [%[[TILESIZE]]]
//       CHECK:     %[[FUSED:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC]]#0, %[[GENERIC]]#1 :
//       CHECK:     tensor.parallel_insert_slice %[[FUSED]] into %[[INIT]][%[[IV]]] [%[[TILESIZE]]]
//       CHECK:   return %[[RESULT]]#2

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield0, %yield1 = transform.split_handle %yield : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %yield0, %yield1 in (%loop)
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Check that when the given operand tiles are inconsistent, tiling fails.

func.func @multi_slice_fusion2(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>, %arg3 : index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %loop:2 = scf.forall (%iv0) =  (%c0) to (%dim0) step (%arg3) shared_outs(%init0 = %arg1, %init1 = %arg2) -> (tensor<?xf32>, tensor<?xf32>) {
    %tilesize = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv0)[%dim0, %arg3]
    %arg0_slice = tensor.extract_slice %arg0[%iv0, 0] [%tilesize, %dim1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %init0_slice = tensor.extract_slice %init0[%iv0] [%tilesize] [1] : tensor<?xf32> to tensor<?xf32>
    %generic0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
	iterator_types = ["parallel", "reduction"]}
	ins(%arg0_slice : tensor<?x?xf32>) outs(%init0_slice : tensor<?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %b1 : f32
	linalg.yield %0 : f32
    } -> tensor<?xf32>
    %init1_slice = tensor.extract_slice %init1[%iv0] [%tilesize] [1] : tensor<?xf32> to tensor<?xf32>
    %generic1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
	iterator_types = ["parallel", "reduction"]}
	ins(%arg0_slice : tensor<?x?xf32>) outs(%init1_slice: tensor<?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
	%0 = arith.addf %b0, %b1 : f32
	linalg.yield %0: f32
    } -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %generic0 into %init0[%iv0] [%tilesize] [1] : tensor<?xf32> into tensor<?xf32>
      tensor.parallel_insert_slice %generic1 into %init1[%iv0] [%tilesize] [1] : tensor<?xf32> into tensor<?xf32>
    }
  }
  %empty = tensor.empty(%dim0) : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%loop#0, %loop#1 : tensor<?xf32>, tensor<?xf32>) outs(%empty : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
// CHECK-LABEL: func @multi_slice_fusion2(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//       CHECK:   %[[C0:.+]] = arith.constant 0
//       CHECK:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[DIM0]])
//       CHECK:   %[[RESULT:.+]]:3 = scf.forall (%[[IV:.+]]) =
//  CHECK-SAME:       , %[[INIT:[a-zA-Z0-9]+]] = %[[EMPTY]])
//       CHECK:     %[[TILESIZE:.+]] = affine.min
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV]]] [%[[TILESIZE]]]
//       CHECK:     %[[FUSED:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC0]], %[[GENERIC1]] :
//       CHECK:     tensor.parallel_insert_slice %[[FUSED]] into %[[INIT]][%[[IV]]] [%[[TILESIZE]]]
//       CHECK:   return %[[RESULT]]#2

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield0, %yield1 = transform.split_handle %yield : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %yield0, %yield1 in (%loop)
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @multi_slice_fusion_with_broadcast(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>,
    %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %loop:2 = scf.forall (%iv0, %iv1) =  (%c0, %c0) to (%dim0, %dim1) step (%arg3, %arg4)
      shared_outs(%init0 = %arg1, %init1 = %arg2) -> (tensor<?x?xf32>, tensor<?xf32>) {
    %tilesize0 = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv0)[%dim0, %arg3]
    %tilesize1 = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv1)[%dim1, %arg4]
    %arg0_slice = tensor.extract_slice %arg0[%iv0, %iv1, 0] [%tilesize0, %tilesize1, %dim2] [1, 1, 1]
        : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %init0_slice = tensor.extract_slice %init0[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %generic0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
	      iterator_types = ["parallel", "parallel", "reduction"]}
	      ins(%arg0_slice : tensor<?x?x?xf32>) outs(%init0_slice : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %b1 : f32
	      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    %init1_slice = tensor.extract_slice %init1[%iv0] [%tilesize0] [1] : tensor<?xf32> to tensor<?xf32>
    %generic1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
	      iterator_types = ["parallel", "reduction"]}
	      ins(%generic0 : tensor<?x?xf32>) outs(%init1_slice: tensor<?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
      	%0 = arith.addf %b0, %b1 : f32
	      linalg.yield %0: f32
    } -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %generic0 into %init0[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
      tensor.parallel_insert_slice %generic1 into %init1[%iv0] [%tilesize0] [1] : tensor<?xf32> into tensor<?xf32>
    }
  }
  %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%loop#0, %loop#1 : tensor<?x?xf32>, tensor<?xf32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield0, %yield1 = transform.split_handle %yield : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %yield0, %yield1 in (%loop)
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func @multi_slice_fusion_with_broadcast(
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//   CHECK-DAG:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[DIM0]], %[[DIM1]])
//       CHECK:   %[[RESULT:.+]]:3 = scf.forall (%[[IV0:[a-zA-Z0-9]+]], %[[IV1:[a-zA-Z0-9]+]]) =
//  CHECK-SAME:       , %[[INIT:[a-zA-Z0-9]+]] = %[[EMPTY]])
//   CHECK-DAG:     %[[TILESIZE0:.+]] = affine.min {{.+}}(%[[IV0]])
//   CHECK-DAG:     %[[TILESIZE1:.+]] = affine.min {{.+}}(%[[IV1]])
//       CHECK:     %[[GENERIC0:.+]] = linalg.generic
//       CHECK:     %[[GENERIC1:.+]] = linalg.generic
//   CHECK-DAG:     %[[INIT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[IV0]], %[[IV1]]] [%[[TILESIZE0]], %[[TILESIZE1]]]
//       CHECK:     %[[FUSED:.+]] = linalg.generic
//  CHECK-SAME:         ins(%[[GENERIC0]], %[[GENERIC1]] :
//       CHECK:     tensor.parallel_insert_slice %[[FUSED]] into %[[INIT]][%[[IV0]], %[[IV1]]] [%[[TILESIZE0]], %[[TILESIZE1]]]
//       CHECK:   return %[[RESULT]]#2

// -----

func.func @multi_slice_fusion_invalid(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>,
    %arg3 : index, %arg4 : index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %loop:2 = scf.forall (%iv0, %iv1) =  (%c0, %c0) to (%dim0, %dim1) step (%arg3, %arg4)
      shared_outs(%init0 = %arg1, %init1 = %arg2) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %tilesize0 = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv0)[%dim0, %arg3]
    %tilesize1 = affine.min affine_map<(d0)[s0, s1] -> (s1, s0 - d0)>(%iv1)[%dim1, %arg4]
    %arg0_slice = tensor.extract_slice %arg0[%iv0, %iv1, 0] [%tilesize0, %tilesize1, %dim2] [1, 1, 1]
        : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %init0_slice = tensor.extract_slice %init0[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %generic0 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
	      iterator_types = ["parallel", "parallel", "reduction"]}
	      ins(%arg0_slice : tensor<?x?x?xf32>) outs(%init0_slice : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.mulf %b0, %b1 : f32
	      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    %init1_slice = tensor.extract_slice %init1[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
        : tensor<?x?xf32> to tensor<?x?xf32>
    %generic1 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
	      iterator_types = ["parallel", "parallel", "reduction"]}
	      ins(%arg0_slice : tensor<?x?x?xf32>) outs(%init1_slice: tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
      	%0 = arith.addf %b0, %b1 : f32
	      linalg.yield %0: f32
    } -> tensor<?x?xf32>
    scf.forall.in_parallel {
      // expected-error @below {{failed to fuse consumer of slice}}
      tensor.parallel_insert_slice %generic0 into %init0[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
      tensor.parallel_insert_slice %generic1 into %init1[%iv0, %iv1] [%tilesize0, %tilesize1] [1, 1]
          : tensor<?x?xf32> into tensor<?x?xf32>
    }
  }
  %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%loop#0, %loop#1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32):
      %0 = arith.addf %b0, %b1 : f32
      linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield = transform.structured.match ops{["tensor.parallel_insert_slice"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %yield0, %yield1 = transform.split_handle %yield : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a, %b = transform.test.fuse_consumer %yield0, %yield1 in (%loop)
      : (!transform.any_op, !transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
