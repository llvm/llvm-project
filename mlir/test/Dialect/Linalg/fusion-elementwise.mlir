// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-generic-ops-control -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @drop_unused_producer_result(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0:2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg0, %arg0  : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1: f32, %b2: f32):
      %1 = arith.addf %b0, %b0 : f32
      %2 = arith.mulf %b0, %b0 : f32
      linalg.yield %1, %2 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  %3 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0#0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg0  : tensor<?x?xf32>) {
    ^bb0(%b0: f32, %b1: f32, %b2: f32):
      %4 = arith.subf %b0, %b1 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
  return %3 : tensor<?x?xf32>
}
// CHECK-LABEL: func @drop_unused_producer_result
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
//       CHECK:   return %[[FUSED_OP]]

// -----

#map = affine_map<(d0) -> (d0)>
func.func @handle_unused_operands(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0:2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%out: f32, %out_2: f32):
    %1 = linalg.index 0 : index
    %2 = arith.index_cast %1 : index to i64
    %3 = arith.sitofp %2 : i64 to f32
    %4 = arith.divf %3, %cst_0 : f32
    linalg.yield %3, %4 : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} ins(%0#1 : tensor<8xf32>) {
  ^bb0(%in: f32):
    %2 = arith.cmpf one, %in, %cst_1 : f32
    cf.assert %2, "Side effect op"
    linalg.yield
  }
  func.return %arg1 : tensor<8xf32>
}

// CHECK-LABEL: func @handle_unused_operands
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       outs(%[[EMPTY]] :
//   CHECK-NOT:   linalg.generic

// -----

func.func @map_ops(%in1: tensor<8xf32>, %in2: tensor<8xf32>) -> tensor<8xf32> {
    %fill = tensor.empty() : tensor<8xf32>
    %add = linalg.map {arith.addf} ins(%in1, %in2: tensor<8xf32>, tensor<8xf32>) outs(%fill: tensor<8xf32>)
    %sqrt = linalg.map { math.sqrt } ins(%add : tensor<8xf32>) outs(%fill : tensor<8xf32>)
    return %sqrt : tensor<8xf32>
}

// CHECK-LABEL: func @map_ops
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : {{.*}}) outs(%[[EMPTY]] :
//  CHECK-NEXT:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
//  CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[IN0]], %[[IN1]]
//  CHECK-NEXT:     %[[SQRT:.*]] = math.sqrt %[[ADD]]
//  CHECK-NEXT:     linalg.yield %[[SQRT]] 
//   CHECK-NOT:   linalg.map

// -----

func.func @map_ops_mixed_types(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %init = tensor.empty() : tensor<8xi1>
  %initf = tensor.empty() : tensor<8xf32>
  %0 = linalg.map {math.sqrt} ins(%arg0 : tensor<8xf32>) outs(%initf : tensor<8xf32>)
  %1 = linalg.map {math.exp} ins(%arg1 : tensor<8xf32>) outs(%initf : tensor<8xf32>)
  %2 = linalg.map ins(%0, %1 : tensor<8xf32>, tensor<8xf32>) outs (%init : tensor<8xi1>)
    (%in0 : f32, %in1 : f32, %out : i1) {
      %cmp = arith.cmpf olt, %in0, %in1 : f32
      linalg.yield %cmp : i1
  }
  %3 = linalg.map { arith.select } ins(%2, %0, %1 : tensor<8xi1>, tensor<8xf32>, tensor<8xf32>) outs(%initf : tensor<8xf32>) 
  return %3 : tensor<8xf32>
}

// CHECK-LABEL: func @map_ops_mixed_types
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : {{.*}}) outs(%[[EMPTY]] :
//  CHECK-NEXT:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
//  CHECK-NEXT:     %[[EXP0:.*]] = math.exp %[[IN1]]
//  CHECK-NEXT:     %[[SQRT0:.*]] = math.sqrt %[[IN0]]
//  CHECK-NEXT:     %[[EXP1:.*]] = math.exp %[[IN1]]
//  CHECK-NEXT:     %[[SQRT1:.*]] = math.sqrt %[[IN0]]
//  CHECK-NEXT:     %[[CMP:.*]] = arith.cmpf olt, %[[SQRT1]], %[[EXP1]]
//  CHECK-NEXT:     %[[RES:.*]] = arith.select %[[CMP]], %[[SQRT0]], %[[EXP0]]
//  CHECK-NEXT:     linalg.yield %[[RES]] 
//   CHECK-NOT:   linalg.map

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d0)>
func.func @elementwise_ops(%in1: tensor<8xf32>, %in2: tensor<8x10xf32>) -> tensor<8x10xf32> {
    %fill = tensor.empty() : tensor<8x10xf32>
    %add = linalg.elementwise
      kind=#linalg.elementwise_kind<add>
      indexing_maps = [#bcast, #identity, #identity]
      ins(%in1, %in2: tensor<8xf32>, tensor<8x10xf32>) outs(%fill: tensor<8x10xf32>) -> tensor<8x10xf32>
    %sqrt = linalg.elementwise
      kind=#linalg.elementwise_kind<sqrt>
      indexing_maps = [#identity, #identity]
      ins(%add : tensor<8x10xf32>) outs(%fill : tensor<8x10xf32>) -> tensor<8x10xf32>
    return %sqrt : tensor<8x10xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @elementwise_ops
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8x10xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x10xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP1]], #[[$MAP0]], #[[$MAP0]]]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : {{.*}}) outs(%[[EMPTY]] :
//  CHECK-NEXT:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
//  CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[IN0]], %[[IN1]]
//  CHECK-NEXT:     %[[SQRT:.*]] = math.sqrt %[[ADD]]
//  CHECK-NEXT:     linalg.yield %[[SQRT]] 
//   CHECK-NOT:   linalg.map

// -----

func.func @map_multi_ops(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %fill = tensor.empty() : tensor<8xf32>
    %add_exp = linalg.map ins(%arg0, %arg1: tensor<8xf32>, tensor<8xf32>) outs(%fill: tensor<8xf32>)
    (%in0 : f32, %in1 : f32, %out : f32) {
      %add = arith.addf %in0, %in1 : f32
      %exp = math.exp %add : f32
      linalg.yield %exp : f32
  }
    %sqrt_mul = linalg.map ins(%add_exp, %arg2 : tensor<8xf32>, tensor<8xf32>) outs(%fill : tensor<8xf32>)
    (%in0 : f32, %in1 : f32, %out : f32) {
      %sqrt = math.sqrt %in0 : f32
      %mul = arith.mulf %sqrt, %in1 : f32
      linalg.yield %mul : f32
  }
    return %sqrt_mul : tensor<8xf32>
}

// CHECK-LABEL: func @map_multi_ops
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] : {{.*}}) outs(%[[EMPTY]] :
//  CHECK-NEXT:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[IN2:.*]]: f32, %[[OUT:.*]]: f32):
//  CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[IN0]], %[[IN1]]
//  CHECK-NEXT:     %[[EXP:.*]] = math.exp %[[ADD]]
//  CHECK-NEXT:     %[[SQRT:.*]] = math.sqrt %[[EXP]]
//  CHECK-NEXT:     %[[MUL:.*]] = arith.mulf %[[SQRT]], %[[IN2]]
//  CHECK-NEXT:     linalg.yield %[[MUL]] 
//   CHECK-NOT:   linalg.map

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#bcast = affine_map<(d0, d1) -> (d0)>
func.func @map_genric_ops(%arg0: tensor<8xf32>, %arg1: tensor<8x10xf32>) -> tensor<8x10xf32> {
    %fill = tensor.empty() : tensor<8x10xf32>
    %add = linalg.generic
      {indexing_maps = [#bcast, #identity, #identity], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1: tensor<8xf32>, tensor<8x10xf32>) outs(%fill: tensor<8x10xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32 
  } -> tensor<8x10xf32>
    %sqrt = linalg.map { math.sqrt } ins(%add : tensor<8x10xf32>) outs(%fill : tensor<8x10xf32>)
    return %sqrt : tensor<8x10xf32>
}

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @map_genric_ops
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<8xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<8x10xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<8x10xf32>
//       CHECK:   %[[FUSED_OP:.+]] = linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP1]], #[[$MAP0]], #[[$MAP0]]]
//  CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] : {{.*}}) outs(%[[EMPTY]] :
//  CHECK-NEXT:   ^bb0(%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[OUT:.*]]: f32):
//  CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[IN0]], %[[IN1]]
//  CHECK-NEXT:     %[[SQRT:.*]] = math.sqrt %[[ADD]]
//  CHECK-NEXT:     linalg.yield %[[SQRT]] 
//   CHECK-NOT:   linalg.map
