// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This applies continuous tiling to the innermost loop in linalg.matmul.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 [4, 4, 4] continuous_tiles=[false, false, true] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (4, -d0 + 130)>
// CHECK: #{{.*}} = affine_map<(d0) -> (d0 - 1)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (2)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (1)>


// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}

// CHECK:    %[[C0:.+]] = arith.constant 0
// CHECK:    %[[C130:.+]] = arith.constant 130 : index
// CHECK:    %[[C4:.+]] = arith.constant 4 : index
// CHECK:    %[[RES0:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C130]] step %[[C4]] iter_args(%[[OUTARG:.+]] = %[[OUT]]) -> (tensor<130x130xf32>)
// CHECK:      %[[AM0:.+]] = affine.min #[[$MAP0]](%[[IV0]])
// CHECK:      {{.*}} = scf.for %[[IV1:.+]] = %[[C0]]{{.*}} to %[[C130]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTARGI:.+]] = %[[OUTARG]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #[[$MAP0]](%[[IV1]])
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C128:.+]] = arith.constant 128 : index
// CHECK:        %[[L2O0:.+]] = scf.for %[[IV2:.+]] = %[[C0]]{{.*}} to %[[C128]] step %[[C4]]{{.*}} iter_args(%[[OUTARG0:.+]] = %[[OUTARGI]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #[[$MAP0]](%[[IV2]])
// CHECK:          %[[XSIN0:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSIN1:.+]] = tensor.extract_slice %[[IN1]][%[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSOUT:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MMRES0:.+]] = linalg.matmul ins(%[[XSIN0]], %[[XSIN1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XSOUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INSL0:.+]] = tensor.insert_slice %[[MMRES0]] into %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INSL0]] : tensor<130x130xf32>
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[L2O1:.+]] = scf.for %[[IV2]] = %[[C128]] to %[[C130]]{{.*}} step %[[C2]] iter_args(%[[OUTARG1:.+]] = %[[L2O0]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2C2:.+]] = affine.min #[[$MAP2]]()
// CHECK:          %[[XSIN0:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV2]]] [%[[AM0]], %[[AM2C2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSIN1:.+]] = tensor.extract_slice %[[IN1]][%[[IV2]], %[[IV1]]] [%[[AM2C2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSOUT:.+]] = tensor.extract_slice %[[OUTARG1]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MMRES1:.+]] = linalg.matmul ins(%[[XSIN0]], %[[XSIN1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XSOUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INSL1:.+]] = tensor.insert_slice %[[MMRES1]] into %[[OUTARG1]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INSL1]] : tensor<130x130xf32>
// CHECK:        %[[RESINMSTTE:.+]] = scf.for %[[IV2]] = %[[C130]]{{.*}}_8 to %[[C130]]{{.*}} step %[[C1]] iter_args(%[[OUTARG2:.+]] = %[[L2O1]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2C1:.+]] = affine.min #[[$MAP1]]()
// CHECK:          %[[XSIN0:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV2]]] [%[[AM0]], %[[AM2C1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSIN1:.+]] = tensor.extract_slice %[[IN1]][%[[IV2]], %[[IV1]]] [%[[AM2C1]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XSOUT:.+]] = tensor.extract_slice %[[OUTARG2]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MMRES2:.+]] = linalg.matmul ins(%[[XSIN0]], %[[XSIN1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XSOUT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INSL2:.+]] = tensor.insert_slice %[[MMRES2]] into %[[OUTARG2]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INSL2]] : tensor<130x130xf32>
// CHECK:        scf.yield %[[RESINMSTTE]] : tensor<130x130xf32>
// CHECK:      scf.yield {{.*}} : tensor<130x130xf32>
// CHECK:    return %[[RES0]] : tensor<130x130xf32>


// -----

// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This applies continuous tiling to the two inner nested loops in linalg.matmul
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 [4, 4, 4] continuous_tiles=[false, true, true] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (4, -d0 + 130)>
// CHECK: #{{.*}} = affine_map<(d0) -> (d0 - 1)>
// CHECK: #[[$MAP2:.+]] = affine_map<() -> (2)>
// CHECK: #[[$MAP1:.+]] = affine_map<() -> (1)>


// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}

// CHECK:    %[[C0:.+]] = arith.constant 0
// CHECK:    %[[C130:.+]] = arith.constant 130 : index
// CHECK:    %[[C4:.+]] = arith.constant 4 : index
// CHECK:    %{{.*}} = scf.for %[[IV0:.+]] = %[[C0]] to %[[C130]] step %[[C4]] iter_args(%[[OUTARG:.+]] = %[[OUT]]) -> (tensor<130x130xf32>) {
// CHECK:      %[[AM0:.+]] = affine.min #map(%[[IV0]])
// CHECK:      %[[C2:.+]] = arith.constant 2 : index
// CHECK:      %[[C128:.+]] = arith.constant 128 : index
// CHECK:      %[[L1RES0:.+]] = scf.for %[[IV1:.+]] = %[[C0]]{{.*}} to %[[C128]] step %[[C4]]{{.*}} iter_args(%[[OUTARGI:.+]] = %[[OUTARG]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #map(%[[IV1]])
// CHECK:        %[[ILRES0:.+]] = scf.for [[IV2:.+]] = %[[C0]]{{.*}} to %[[C128]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTARG0:.+]] = %[[OUTARGI]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map([[IV2]])
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS1:.+]] = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS2:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %[[XS1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XS2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INS:.+]] : tensor<130x130xf32>
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[ILRES1:.+]] = scf.for [[IV2:.+]] = %[[C128]]{{.*}} to %[[C130]]{{.*}} step %[[C2]]{{.*}} iter_args(%[[OUTARG0:.+]] = %[[ILRES0]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map2()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS1:.+]] = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS2:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %[[XS1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XS2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INS:.+]] : tensor<130x130xf32>
// CHECK:        %[[ILRES2:.+]] = scf.for [[IV2:.+]] = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]]{{.*}} iter_args(%[[OUTARG0:.+]] = %[[ILRES1]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map3()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS1:.+]] = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS2:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %[[XS1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XS2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTARG0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INS:.+]] : tensor<130x130xf32>
// CHECK:        scf.yield %[[ILRES2:.+]] : tensor<130x130xf32>
// CHECK:      %[[C1:.+]] = arith.constant 1 : index
// CHECK:      %{{.*}} = scf.for %[[IV1:.+]] = %[[C128]] to %[[C130]]{{.*}} step %[[C2]] iter_args(%[[L1RES0ARG:.+]] = %[[L1RES0]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #map2()
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C0]]{{.*}} to %[[C128]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTARG1:.+]] = %[[L1RES0ARG]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map([[IV2]])
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARG1]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARG1]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C128]]{{.*}} to %[[C130]]{{.*}} step %[[C2]]{{.*}} iter_args(%[[OUTARGX:.+]] = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map2()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]]{{.*}} iter_args(%[[OUTARGX:.+]] = {{.*}}) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map3()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:      %[[RESL1TE:.+]] = scf.for %[[IV1:.+]] = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]] iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #map3()
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C0]]{{.*}} to %[[C128]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTARGX:.+]] = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map([[IV2]])
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C128]]{{.*}} to %[[C130]]{{.*}} step %[[C2]]{{.*}} iter_args(%[[OUTARGX:.+]] = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map2()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        %{{.*}} = scf.for [[IV2:.+]] = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]]{{.*}} iter_args(%[[OUTARGX:.+]] = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #map3()
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], [[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[IN1]][[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.extract_slice %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %{{.*}} : tensor<?x?xf32>, tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %{{.*}} = tensor.insert_slice %[[MM]] into %[[OUTARGX]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:        scf.yield %{{.*}} : tensor<130x130xf32>
// CHECK:      scf.yield %[[RESL1TE]] : tensor<130x130xf32>


// -----

// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This applies continuous tiling to all nested loops in linalg.matmul.
// This test checks that the function return the result of the tail-end
// outermost loop.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 [4, 4, 4] continuous_tiles=[true, true, true] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}


// CHECK:           %[[C0:.+]] = arith.constant 0
// CHECK:           %[[C130:.+]] = arith.constant 130 : index
// CHECK:           %[[C4:.+]] = arith.constant 4 : index
// CHECK:           %[[C2:.+]] = arith.constant 2 : index
// CHECK:           %[[C128:.+]] = arith.constant 128 : index
// CHECK:           %[[OLRES0:.+]] = scf.for %{{.*}} = %[[C0]] to %[[C128]] step %[[C4]] iter_args(%{{.*}} = %[[OUT]]) -> (tensor<130x130xf32>)
// CHECK:           %[[C1:[c][0-9]+]] = arith.constant 1 : index
// CHECK-NEXT:      %{{.*}} = arith.constant 130 : index
// CHECK:           %[[OLRES1:.+]] = scf.for %{{.*}} = %[[C128]] to %[[C130]]{{.*}} step %[[C2]] iter_args(%{{.*}} = %[[OLRES0]]) -> (tensor<130x130xf32>)
// CHECK:           %[[OLRES2:.+]] = scf.for %{{.*}} = %[[C130]]{{.*}} to %[[C130]] step %[[C1]] iter_args(%{{.*}} = %[[OLRES1]]) -> (tensor<130x130xf32>)
// CHECK:           return %[[OLRES2]] : tensor<130x130xf32>


// -----

// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This applies no continuous tiling to any loop in linalg.matmul.
// All values in continuous_tiles are set to false.
// This test checks that the result is equivalent to regular tiling, 
// i.e. when continuous_tiles is none or not supplied.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 [4, 4, 4] continuous_tiles=[false, false, false] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}


// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (4, -d0 + 130)>

// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}

// CHECK:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:    %[[C130:.+]] = arith.constant 130 : index
// CHECK:    %[[C4:.+]] = arith.constant 4 : index
// CHECK:    %[[RL0:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C130]] step %[[C4]] iter_args(%[[OUTL0:.+]] = %[[OUT]]) -> (tensor<130x130xf32>) {
// CHECK:      %[[AM0:.+]] = affine.min #[[$MAP0]](%[[IV0]])
// CHECK:      %[[RL1:.+]] = scf.for %[[IV1:.+]] = %[[C0]]{{.*}} to %[[C130]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTL1:.+]] = %[[OUTL0]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #[[$MAP0]](%[[IV1]])
// CHECK:        %[[RL2:.+]] = scf.for %[[IV2:.+]] = %[[C0]]{{.*}} to %[[C130]]{{.*}} step %[[C4]]{{.*}} iter_args(%[[OUTL2:.+]] = %[[OUTL1]]) -> (tensor<130x130xf32>) {
// CHECK:          %[[AM2:.+]] = affine.min #[[$MAP0]](%[[IV2]])
// CHECK:          %[[XS:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV2]]] [%[[AM0]], %[[AM2]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS1:.+]] = tensor.extract_slice %[[IN1]][%[[IV2]], %[[IV1]]] [%[[AM2]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[XS2:.+]] = tensor.extract_slice %[[OUTL2]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:          %[[MM:.+]] = linalg.matmul ins(%[[XS]], %[[XS1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[XS2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:          %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTL2]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<?x?xf32> into tensor<130x130xf32>
// CHECK:          scf.yield %[[INS]] : tensor<130x130xf32>
// CHECK:        scf.yield %[[RL2]] : tensor<130x130xf32>
// CHECK:      scf.yield %[[RL1]] : tensor<130x130xf32>
// CHECK:    return %[[RL0]] : tensor<130x130xf32>


// -----

// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This tests that continuous tiling works correctly when interchange is applied.
// We only check for loop ordering and that correct results are yielded.
// We use different tile sizes to identify that loops are interchanged
// properly. All loops are moved from their original nesting by supplying [2, 0, 1]
// for interchange.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:3 = transform.structured.tile_using_for %0 [4, 8, 2] interchange=[2, 0, 1] continuous_tiles=[true, true, true] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}


// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (4, -d0 + 130)>

// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}

// CHECK:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:    %[[C130:.+]] = arith.constant 130 : index
// CHECK:    %[[C2:.+]] = arith.constant 2 : index
// CHECK:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:    %[[L0RES:.+]] = scf.for %{{.*}} = %[[C0]] to %[[C130]]{{.*}} step %[[C2]] iter_args(%{{.*}} = %[[OUT]]) -> (tensor<130x130xf32>) {
// CHECK:      %[[C4:.+]] = arith.constant 4 : index
// CHECK:      %[[C128:.+]] = arith.constant 128 : index
// CHECK:      %{{.*}} = scf.for %{{.*}} = %[[C0]]{{.*}} to %[[C128]] step %[[C4]] iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:        %[[C8:.+]] = arith.constant 8 : index
// CHECK:        %{{.*}} = scf.for %{{.*}} = %[[C0]]{{.*}} to %[[C128]]{{.*}} step %[[C8]] iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:        %{{.*}} = scf.for %{{.*}} = %[[C128]]{{.*}} to %[[C128]]{{.*}} step %[[C4]]{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:        %{{.*}} = scf.for %{{.*}} = %[[C128]]{{.*}} to %[[C130]]{{.*}} step %[[C2]]{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:        %{{.*}} = scf.for %{{.*}} = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]]{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:      %{{.*}} = scf.for %{{.*}} = %[[C128]] to %[[C130]]{{.*}} step %[[C2]]{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:      %{{.*}} = scf.for %{{.*}} = %[[C130]]{{.*}} to %[[C130]]{{.*}} step %[[C1]]{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<130x130xf32>) {
// CHECK:    %[[L1RES:.+]] = scf.for %{{.*}} = %[[C130]]{{.*}} to %[[C130]] step %[[C1]] iter_args(%{{.*}} = %[[L0RES]]) -> (tensor<130x130xf32>) {
// CHECK:    return %[[L1RES]] : tensor<130x130xf32>


// -----

// RUN: mlir-opt --transform-interpreter --split-input-file %s | FileCheck %s

// This tests that continuous tiling works correctly when a tile size is zero.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops:2 = transform.structured.tile_using_for %0 [4, 0, 4] continuous_tiles=[false, false, true] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}


// CHECK: #[[$MAP0:.+]] = affine_map<(d0) -> (4, -d0 + 130)>

// CHECK-LABEL: @tile_linalg_matmul
// CHECK-SAME: %[[IN0:.+]]: tensor<130x130xf32>, %[[IN1:.+]]: tensor<130x130xf32>, %[[OUT:.+]]: tensor<130x130xf32>
func.func @tile_linalg_matmul(
  %arg0: tensor<130x130xf32>, %arg1: tensor<130x130xf32>, %arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<130x130xf32>, tensor<130x130xf32>)
                     outs(%arg2: tensor<130x130xf32>)
    -> tensor<130x130xf32>

  return %0 : tensor<130x130xf32>
}

// CHECK:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:    %[[C130:.+]] = arith.constant 130 : index
// CHECK:    %[[C4:.+]] = arith.constant 4 : index
// CHECK:    %[[OLRES:.+]] = scf.for %[[IV0:.+]] = %[[C0]] to %[[C130]] step %[[C4]] iter_args(%[[OUTARG:.+]] = %[[OUT]]) -> (tensor<130x130xf32>) {
// CHECK:      %[[AM0:.+]] = affine.min #map(%[[IV0]])
// CHECK:      %[[C2:.+]] = arith.constant 2 : index
// CHECK:      %[[C128:.+]] = arith.constant 128 : index
// CHECK:      %[[IL0R:.+]] = scf.for %[[IV1:.+]] = %[[C0]]{{.*}} to %[[C128]] step %[[C4]]{{.*}} iter_args(%[[OUTARG0:.+]] = %[[OUTARG]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #map(%[[IV1]])
// CHECK:        %[[XS0:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:        %[[XS1:.+]] = tensor.extract_slice %[[IN1]][%[[IV1]], 0] [%[[AM1]], 130] [1, 1] : tensor<130x130xf32> to tensor<?x130xf32>
// CHECK:        %[[XS2:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], 0] [%[[AM0]], 130] [1, 1] : tensor<130x130xf32> to tensor<?x130xf32>
// CHECK:        %[[MM:.+]] = linalg.matmul ins(%[[XS0]], %[[XS1]] : tensor<?x?xf32>, tensor<?x130xf32>) outs(%[[XS2]] : tensor<?x130xf32>) -> tensor<?x130xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTARG0]][%[[IV0]], 0] [%[[AM0]], 130] [1, 1] : tensor<?x130xf32> into tensor<130x130xf32>
// CHECK:        scf.yield %[[INS]] : tensor<130x130xf32>
// CHECK:      %[[C1:.+]] = arith.constant 1 : index
// CHECK:      %[[IL1R:.+]] = scf.for %[[IV1:.+]] = %c128 to %[[C130]]{{.*}} step %[[C2]] iter_args(%[[OUTARG0]] = %[[IL0R]]) -> (tensor<130x130xf32>) {
// CHECK:        %[[AM1:.+]] = affine.min #map2()
// CHECK:        %[[XS0:.+]] = tensor.extract_slice %[[IN0]][%[[IV0]], %[[IV1]]] [%[[AM0]], %[[AM1]]] [1, 1] : tensor<130x130xf32> to tensor<?x?xf32>
// CHECK:        %[[XS1:.+]] = tensor.extract_slice %[[IN1]][%[[IV1]], 0] [%[[AM1]], 130] [1, 1] : tensor<130x130xf32> to tensor<?x130xf32>
// CHECK:        %[[XS2:.+]] = tensor.extract_slice %[[OUTARG0]][%[[IV0]], 0] [%[[AM0]], 130] [1, 1] : tensor<130x130xf32> to tensor<?x130xf32>
// CHECK:        %[[MM:.+]] = linalg.matmul ins(%[[XS0]], %[[XS1]] : tensor<?x?xf32>, tensor<?x130xf32>) outs(%[[XS2]] : tensor<?x130xf32>) -> tensor<?x130xf32>
// CHECK:        %[[INS:.+]] = tensor.insert_slice %[[MM]] into %[[OUTARG0]][%[[IV0]], 0] [%[[AM0]], 130] [1, 1] : tensor<?x130xf32> into tensor<130x130xf32>
// CHECK:        scf.yield %[[INS]] : tensor<130x130xf32>
// CHECK:      %[[IL2R:.+]] = scf.for %{{.*}} = %[[C130]]{{.*}} to %[[C130]]{{.*}} step  %[[C1]] iter_args(%[[OUTARG0]] = %[[IL1R]]) -> (tensor<130x130xf32>) {
// CHECK:      scf.yield %[[IL2R]] : tensor<130x130xf32>
// CHECK:    return %[[OLRES]] : tensor<130x130xf32>