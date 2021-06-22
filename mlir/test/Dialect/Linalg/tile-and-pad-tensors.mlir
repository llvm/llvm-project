// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-and-pad-pattern tile-sizes-for-padding=2,3,4" -canonicalize | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns="test-tile-and-pad-pattern tile-sizes-for-padding=2,3" -canonicalize | FileCheck %s -check-prefix=CHECK-1DIM-TILE

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
func @matmul_tensors(
  %arg0: tensor<?x?xi8>, %arg1: tensor<?x?xi8>, %arg2: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
//      CHECK: %[[C0:.*]] = constant 0 : index
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xi32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xi32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?xi32>) {
//      CHECK:       %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<?x?xi8> to tensor<?x?xi8>
//      CHECK:       %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<?x?xi8> to tensor<?x?xi8>
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<?x?xi32> to tensor<?x?xi32>

// Dynamic op has been canonicalized away.
//  CHECK-NOT:       linalg.matmul {{.*}} tensor<?x?xi8>

// Padding injects static information.
//      CHECK:       %[[pA:.*]] = linalg.pad_tensor %[[sTA]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xi8> to tensor<2x4xi8>
//      CHECK:       %[[pB:.*]] = linalg.pad_tensor %[[sTB]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xi8> to tensor<4x3xi8>
//      CHECK:       %[[pC:.*]] = linalg.pad_tensor %[[sTC]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK:         : tensor<?x?xi32> to tensor<2x3xi32>
//      CHECK:       %[[pD:.*]] = linalg.matmul_i8_i8_i32 ins(%[[pA]], %[[pB]] : tensor<2x4xi8>, tensor<4x3xi8>)
// CHECK-SAME:                                           outs(%[[pC]] : tensor<2x3xi32>)  -> tensor<2x3xi32>
//      CHECK:       %[[sTD:.*]] = tensor.extract_slice %[[pD]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<2x3xi32> to tensor<?x?xi32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?xi32> into tensor<?x?xi32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?xi32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?xi32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?xi32>
  %0 = linalg.matmul_i8_i8_i32 {__internal_linalg_transform__ = "tile-and-pad"}
      ins(%arg0, %arg1: tensor<?x?xi8>, tensor<?x?xi8>)
     outs(%arg2: tensor<?x?xi32>)
    -> tensor<?x?xi32>

//      CHECK: return %[[TD0]] : tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: func @generic_scalar_and_tensor(
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?x?xf32>
// CHECK-SAME:    %[[VAL:[0-9a-z]+]]: f32) -> tensor<?x?x?xf32> {
func @generic_scalar_and_tensor(
  %arg0: tensor<?x?x?xf32>, %arg1: f32)
    -> tensor<?x?x?xf32> {
//      CHECK: %[[C0:.*]] = constant 0 : index
//      CHECK: %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?x?xf32>) {
//      CHECK:   %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?x?xf32>) {
//      CHECK:     %[[TD2:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC2:.*]] = %[[TC1]]) -> (tensor<?x?x?xf32>) {
//      CHECK:       %[[sTC:.*]] = tensor.extract_slice %[[TC2]][{{.*}}] : tensor<?x?x?xf32> to tensor<?x?x?xf32>

// Padding injects static information.
//      CHECK:       %[[pC:.*]] = linalg.pad_tensor %[[sTC]] low[%[[C0]], %[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}, %{{.*}}]
//      CHECK:        : tensor<?x?x?xf32> to tensor<2x3x4xf32>
//      CHECK:       %[[pD:.*]] = linalg.generic
// CHECK-SAME:         ins(%[[VAL]] : f32) outs(%[[pC]] : tensor<2x3x4xf32>)
//      CHECK:       %[[sTD:.*]] = tensor.extract_slice %[[pD]][0, 0, 0] [%{{.*}}, %{{.*}}, %{{.*}}] [1, 1, 1] : tensor<2x3x4xf32> to tensor<?x?x?xf32>
//      CHECK:       %[[TD:.*]] = tensor.insert_slice %[[sTD]] into %[[TC2]][{{.*}}]  : tensor<?x?x?xf32> into tensor<?x?x?xf32>
//      CHECK:       scf.yield %[[TD]] : tensor<?x?x?xf32>
//      CHECK:     scf.yield %[[TD2]] : tensor<?x?x?xf32>
//      CHECK:   scf.yield %[[TD1]] : tensor<?x?x?xf32>
      %0 = linalg.generic {
      indexing_maps =  [ affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)> ],
      iterator_types = ["parallel", "parallel", "parallel"]}
      {__internal_linalg_transform__ = "tile-and-pad"}
     ins(%arg1 : f32)
    outs(%arg0: tensor<?x?x?xf32>) {
      ^bb(%0: f32, %1: f32) :
        linalg.yield %0 : f32
    } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-1DIM-TILE: func @matmul_tensors(
// CHECK-1DIM-TILE:    %[[TA:[0-9a-z]+]]: tensor<?x?xi8>
// CHECK-1DIM-TILE:    %[[TB:[0-9a-z]+]]: tensor<?x?xi8>
// CHECK-1DIM-TILE:    %[[TC:[0-9a-z]+]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
// CHECK-1DIM-TILE-NOT: scf.for
// CHECK-1DIM-TILE: linalg.matmul_i8_i8_i32 ins(%[[TA]], %[[TB]] : tensor<?x?xi8>, tensor<?x?xi8>) outs(%[[TC]] : tensor<?x?xi32>) -> tensor<?x?xi32>

func @matmul_partially_padded_tensors(
  %arg0: tensor<?x8xi8>, %arg1: tensor<8x?xi8>, %arg2: tensor<?x?xi32>)
    -> tensor<?x?xi32> {
  %0 = linalg.matmul_i8_i8_i32 {__internal_linalg_transform__ = "tile-and-pad"}
      ins(%arg0, %arg1: tensor<?x8xi8>, tensor<8x?xi8>)
     outs(%arg2: tensor<?x?xi32>)
    -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}
// CHECK-LABEL: func @matmul_partially_padded_tensors(
// CHECK: linalg.matmul_i8_i8_i32 ins({{.*}}, {{.*}} : tensor<2x4xi8>, tensor<4x3xi8>) outs({{.*}} : tensor<2x3xi32>) -> tensor<2x3xi32>


// CHECK-1DIM-TILE: func @matmul_partially_padded_tensors(
// CHECK-1DIM-TILE-SAME:    %[[TA:[0-9a-z]+]]: tensor<?x8xi8>
// CHECK-1DIM-TILE-SAME:    %[[TB:[0-9a-z]+]]: tensor<8x?xi8>
// CHECK-1DIM-TILE-SAME:    %[[TC:[0-9a-z]+]]: tensor<?x?xi32>) -> tensor<?x?xi32> {
//      CHECK-1DIM-TILE:        %[[C0:.*]] = constant 0 : index
//      CHECK-1DIM-TILE:        %[[TD0:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC0:.*]] = %[[TC]]) -> (tensor<?x?xi32>) {
//      CHECK-1DIM-TILE:            %[[TD1:.*]] = scf.for {{.*}} to {{.*}} step {{.*}} iter_args(%[[TC1:.*]] = %[[TC0]]) -> (tensor<?x?xi32>) {
//      CHECK-1DIM-TILE:                %[[sTA:.*]] = tensor.extract_slice %[[TA]][{{.*}}] : tensor<?x8xi8> to tensor<?x8xi8>
//      CHECK-1DIM-TILE:                %[[sTAc:.*]] = tensor.cast %[[sTA]] : tensor<?x8xi8> to tensor<?x?xi8>
//      CHECK-1DIM-TILE:                %[[sTB:.*]] = tensor.extract_slice %[[TB]][{{.*}}] : tensor<8x?xi8> to tensor<8x?xi8>
//      CHECK-1DIM-TILE:                %[[sTBc:.*]] = tensor.cast %[[sTB]] : tensor<8x?xi8> to tensor<?x?xi8>
//      CHECK-1DIM-TILE:                %[[sTC:.*]] = tensor.extract_slice %[[TC1]][{{.*}}] : tensor<?x?xi32> to tensor<?x?xi32>
//      CHECK-1DIM-TILE:                %[[pA:.*]] = linalg.pad_tensor %[[sTAc]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK-1DIM-TILE:                   : tensor<?x?xi8> to tensor<2x8xi8>
//      CHECK-1DIM-TILE:                %[[pB:.*]] = linalg.pad_tensor %[[sTBc]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK-1DIM-TILE:                   : tensor<?x?xi8> to tensor<8x3xi8>
//      CHECK-1DIM-TILE:                %[[pC:.*]] = linalg.pad_tensor %[[sTC]] low[%[[C0]], %[[C0]]] high[%{{.*}}, %{{.*}}]
//      CHECK-1DIM-TILE:                   : tensor<?x?xi32> to tensor<2x3xi32>
//      CHECK-1DIM-TILE:               %[[pD:.*]] = linalg.matmul_i8_i8_i32 ins(%[[pA]], %[[pB]] : tensor<2x8xi8>, tensor<8x3xi8>)
//      CHECK-1DIM-TILE:                                           outs(%[[pC]] : tensor<2x3xi32>)  -> tensor<2x3xi32>
