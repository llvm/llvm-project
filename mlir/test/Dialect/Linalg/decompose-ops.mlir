// RUN: mlir-opt -test-linalg-decompose-ops -cse -split-input-file %s | FileCheck %s
// RUN: mlir-opt -test-linalg-decompose-ops -cse -canonicalize -split-input-file %s | FileCheck %s --check-prefix=CANONICALIZECHECK

func.func @simple_op(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d1, %d0] : tensor<?x?xf32>
  %init2 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %result:2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, 
                     affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
    outs(%init1, %init2 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32, %b4 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      %1 = arith.mulf %0, %b2 : f32
      linalg.yield %0, %1 : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
  return %result#0, %result#1 : tensor<?x?xf32>, tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @simple_op(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [%[[D1]], %[[D0]]]
//  CHECK-DAG:   %[[INIT2:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[GENERIC1:.+]]:3 = linalg.generic
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP0]], #[[MAP3]]]
// CHECK-SAME:       ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:       outs(%[[INIT1]], %[[INIT2]], %[[INIT1]] :
// CHECK-NEXT:   ^bb0(
// CHECK-SAME:       %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B1:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B2:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B3:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B4:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B5:[a-zA-Z0-9]+]]: f32):
// CHECK-NEXT:     %[[S0:.+]] = arith.addf %[[B0]], %[[B1]]
// CHECK-NEXT:     linalg.yield %[[S0]], %{{[a-zA-Z0-9]+}}, %[[S0]]
//      CHECK:   %[[GENERIC2:.+]]:2 = linalg.generic
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP3]], #[[MAP0]]]
// CHECK-SAME:       ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[GENERIC1]]#2 :
// CHECK-SAME:       outs(%[[INIT1]], %[[INIT2]] :
// CHECK-NEXT:   ^bb0(
// CHECK-SAME:       %[[B6:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B7:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B8:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B9:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B10:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B11:[a-zA-Z0-9]+]]: f32):
// CHECK-NEXT:     %[[S1:.+]] = arith.mulf %[[B9]], %[[B8]]
// CHECK-NEXT:     linalg.yield %[[B9]], %[[S1]]
//      CHECK:   return %[[GENERIC1]]#0, %[[GENERIC2]]#1

// With cse + canonicalization

//  CANONICALIZECHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CANONICALIZECHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//  CANONICALIZECHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//  CANONICALIZECHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1)>
//      CANONICALIZECHECK: func @simple_op(
// CANONICALIZECHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CANONICALIZECHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CANONICALIZECHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CANONICALIZECHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CANONICALIZECHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CANONICALIZECHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CANONICALIZECHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CANONICALIZECHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [%[[D1]], %[[D0]]]
//  CANONICALIZECHECK-DAG:   %[[INIT2:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CANONICALIZECHECK-DAG:   %[[GENERIC1:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CANONICALIZECHECK-SAME:       ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT1]] :
// CANONICALIZECHECK-NEXT:   ^bb0(
// CANONICALIZECHECK-SAME:       %[[B0:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B1:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B2:[a-zA-Z0-9]+]]: f32):
// CANONICALIZECHECK-NEXT:     %[[S0:.+]] = arith.addf %[[B0]], %[[B1]]
// CANONICALIZECHECK-NEXT:     linalg.yield %[[S0]]
//      CANONICALIZECHECK:   %[[GENERIC2:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       [#[[MAP3]], #[[MAP2]], #[[MAP0]]]
// CANONICALIZECHECK-SAME:       ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG2]], %[[GENERIC1]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT2]] :
// CANONICALIZECHECK-NEXT:   ^bb0(
// CANONICALIZECHECK-SAME:       %[[B3:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B4:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B5:[a-zA-Z0-9]+]]: f32):
// CANONICALIZECHECK-NEXT:     %[[S1:.+]] = arith.mulf %[[B4]], %[[B3]]
// CANONICALIZECHECK-NEXT:     linalg.yield %[[S1]]
//      CANONICALIZECHECK:   return %[[GENERIC1]], %[[GENERIC2]]


// -----

func.func @simple_op_permuted_outputs(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?xf32>)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init1 = linalg.init_tensor [%d1, %d0] : tensor<?x?xf32>
  %init2 = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %result:3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, 
                     affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
    outs(%init1, %init2, %init2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32, %b2 : f32, %b3 : f32, %b4 : f32, %b5 : f32) :
      %0 = arith.addf %b0, %b1 : f32
      %1 = arith.mulf %0, %b2 : f32
      linalg.yield %0, %1, %0 : f32, f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)
  return %result#0, %result#1, %result#2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @simple_op_permuted_outputs(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [%[[D1]], %[[D0]]]
//  CHECK-DAG:   %[[INIT2:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[GENERIC1:.+]]:4 = linalg.generic
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME:       ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] :
// CHECK-SAME:       outs(%[[INIT1]], %[[INIT2]], %[[INIT2]], %[[INIT2]] :
// CHECK-NEXT:   ^bb0(
// CHECK-SAME:       %[[B0:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B1:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B2:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B3:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B4:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B5:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B6:[a-zA-Z0-9]+]]: f32):
// CHECK-NEXT:     %[[S0:.+]] = arith.addf %[[B0]], %[[B1]]
// CHECK-NEXT:     linalg.yield %[[S0]], %{{[a-zA-Z0-9]+}}, %[[S0]]
//      CHECK:   %[[GENERIC2:.+]]:3 = linalg.generic
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]], #[[MAP3]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME:       ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[GENERIC1]]#3 :
// CHECK-SAME:       outs(%[[INIT1]], %[[INIT2]], %[[INIT2]] :
// CHECK-NEXT:   ^bb0(
// CHECK-SAME:       %[[B7:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B8:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B9:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B10:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B11:[a-zA-Z0-9]+]]: f32
// CHECK-SAME:       %[[B12:[a-zA-Z0-9]+]]: f32):
// CHECK-NEXT:     %[[S1:.+]] = arith.mulf %[[B10]], %[[B9]]
// CHECK-NEXT:     linalg.yield %[[B10]], %[[S1]], %[[B10]]
//      CHECK:   return %[[GENERIC1]]#0, %[[GENERIC2]]#1, %[[GENERIC1]]#2

//  CANONICALIZECHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CANONICALIZECHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//  CANONICALIZECHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//  CANONICALIZECHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1)>
//      CANONICALIZECHECK: func @simple_op_permuted_outputs(
// CANONICALIZECHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CANONICALIZECHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?xf32>
// CANONICALIZECHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?xf32>
//  CANONICALIZECHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CANONICALIZECHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CANONICALIZECHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CANONICALIZECHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CANONICALIZECHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [%[[D1]], %[[D0]]]
//  CANONICALIZECHECK-DAG:   %[[INIT2:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CANONICALIZECHECK-DAG:   %[[GENERIC1:.+]]:2 = linalg.generic
// CANONICALIZECHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CANONICALIZECHECK-SAME:       ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT1]], %[[INIT2]] :
// CANONICALIZECHECK-NEXT:   ^bb0(
// CANONICALIZECHECK-SAME:       %[[B0:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B1:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B2:[a-zA-Z0-9]+]]: f32):
// CANONICALIZECHECK-NEXT:     %[[S0:.+]] = arith.addf %[[B0]], %[[B1]]
// CANONICALIZECHECK-NEXT:     linalg.yield %[[S0]], %[[S0]]
//      CANONICALIZECHECK:   %[[GENERIC2:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       [#[[MAP3]], #[[MAP0]], #[[MAP0]]]
// CANONICALIZECHECK-SAME:       ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG2]], %[[GENERIC1]]#1 :
// CANONICALIZECHECK-SAME:       outs(%[[INIT2]] :
// CANONICALIZECHECK-NEXT:   ^bb0(
// CANONICALIZECHECK-SAME:       %[[B4:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B5:[a-zA-Z0-9]+]]: f32
// CANONICALIZECHECK-SAME:       %[[B6:[a-zA-Z0-9]+]]: f32):
// CANONICALIZECHECK-NEXT:     %[[S1:.+]] = arith.mulf %[[B5]], %[[B4]]
// CANONICALIZECHECK-NEXT:     linalg.yield %[[S1]]
//      CANONICALIZECHECK:   return %[[GENERIC1]]#0, %[[GENERIC2]], %[[GENERIC1]]#1

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
func.func @multi_statement(%arg0 : tensor<10x20xf32>, %arg1 : tensor<10xi32>) -> tensor<20x10xf64> {
  %init = linalg.init_tensor [20, 10] : tensor<20x10xf64>
  %0 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<10xi32>)
    outs(%init : tensor<20x10xf64>) {
    ^bb0(%b0 : f32, %b1 : i32, %b2 : f64):
      %1 = arith.sitofp %b1 : i32 to f64
      %2 = arith.extf %b0 : f32 to f64
      %3 = arith.addf %1, %2 : f64
      linalg.yield %3 : f64
    } -> tensor<20x10xf64>
  return %0 : tensor<20x10xf64>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CHECK: func @multi_statement(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<10xi32>)
//  CHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [20, 10] : tensor<20x10xf64>
//  CHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [10, 20] : tensor<10x20xf64>
//      CHECK:   %[[GENERIC0:.+]]:2 = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[INIT0]], %[[INIT1]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B0:.+]]: f32
// CHECK-SAME:         %[[B1:.+]]: i32
// CHECK-SAME:         %[[B2:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:         %[[B3:.+]]: f64
// CHECK-NEXT:       %[[S0:.+]] = arith.sitofp %[[B1]] : i32 to f64
// CHECK-NEXT:       linalg.yield %{{.+}}, %[[S0]]
//      CHECK:   %[[GENERIC1:.+]]:2 = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[GENERIC0]]#1 :
// CHECK-SAME:       outs(%[[INIT0]], %[[INIT1]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B4:.+]]: f32
// CHECK-SAME:         %[[B5:.+]]: i32
// CHECK-SAME:         %[[B6:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:         %[[B7:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:         %[[B8:.+]]: f64
// CHECK-NEXT:       %[[S1:.+]] = arith.extf %[[B4]] : f32 to f64
// CHECK-NEXT:       linalg.yield %{{.+}}, %[[S1]]
//      CHECK:   %[[GENERIC2:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP0]], #[[MAP0]], #[[MAP2]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]], %[[GENERIC0]]#1, %[[GENERIC1]]#1 :
// CHECK-SAME:       outs(%[[INIT0]] :
// CHECK-NEXT:     ^bb0(
// CHECK-SAME:         %[[B9:.+]]: f32
// CHECK-SAME:         %[[B10:.+]]: i32
// CHECK-SAME:         %[[B11:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:         %[[B12:[a-zA-Z0-9]+]]: f64
// CHECK-SAME:         %[[B13:.+]]: f64
// CHECK-NEXT:       %[[S2:.+]] = arith.addf %[[B11]], %[[B12]] : f64
// CHECK-NEXT:       linalg.yield %[[S2]]
//      CHECK:   return %[[GENERIC2]]

//  CANONICALIZECHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0)>
//  CANONICALIZECHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CANONICALIZECHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//      CANONICALIZECHECK: func @multi_statement(
// CANONICALIZECHECK-SAME:     %[[ARG0:.+]]: tensor<10x20xf32>
// CANONICALIZECHECK-SAME:     %[[ARG1:.+]]: tensor<10xi32>)
//  CANONICALIZECHECK-DAG:   %[[INIT0:.+]] = linalg.init_tensor [20, 10] : tensor<20x10xf64>
//  CANONICALIZECHECK-DAG:   %[[INIT1:.+]] = linalg.init_tensor [10, 20] : tensor<10x20xf64>
//      CANONICALIZECHECK:   %[[GENERIC0:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CANONICALIZECHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG1]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT1]] :
// CANONICALIZECHECK-NEXT:     ^bb0(
// CANONICALIZECHECK-SAME:         %[[B0:.+]]: i32
// CANONICALIZECHECK-SAME:         %[[B1:.+]]: f64
// CANONICALIZECHECK-NEXT:       %[[S0:.+]] = arith.sitofp %[[B0]] : i32 to f64
// CANONICALIZECHECK-NEXT:       linalg.yield %[[S0]]
//      CANONICALIZECHECK:   %[[GENERIC1:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP1]]]
// CANONICALIZECHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[ARG0]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT1]] :
// CANONICALIZECHECK-NEXT:     ^bb0(
// CANONICALIZECHECK-SAME:         %[[B2:.+]]: f32
// CANONICALIZECHECK-SAME:         %[[B3:.+]]: f64
// CANONICALIZECHECK-NEXT:       %[[S1:.+]] = arith.extf %[[B2]] : f32 to f64
// CANONICALIZECHECK-NEXT:       linalg.yield %[[S1]]
//      CANONICALIZECHECK:   %[[GENERIC2:.+]] = linalg.generic
// CANONICALIZECHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP1]], #[[MAP2]]]
// CANONICALIZECHECK-SAME:       iterator_types = ["parallel", "parallel"]
// CANONICALIZECHECK-SAME:       ins(%[[GENERIC0]], %[[GENERIC1]] :
// CANONICALIZECHECK-SAME:       outs(%[[INIT0]] :
// CANONICALIZECHECK-NEXT:     ^bb0(
// CANONICALIZECHECK-SAME:         %[[B4:[a-zA-Z0-9]+]]: f64
// CANONICALIZECHECK-SAME:         %[[B5:[a-zA-Z0-9]+]]: f64
// CANONICALIZECHECK-SAME:         %[[B6:.+]]: f64
// CANONICALIZECHECK-NEXT:       %[[S2:.+]] = arith.addf %[[B4]], %[[B5]] : f64
// CANONICALIZECHECK-NEXT:       linalg.yield %[[S2]]
//      CANONICALIZECHECK:   return %[[GENERIC2]]
