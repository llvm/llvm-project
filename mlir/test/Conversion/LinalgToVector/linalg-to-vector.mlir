// RUN: mlir-opt %s -test-conv-vectorization="tile-sizes=1,3" --cse -split-input-file
// | FileCheck %s

// CHECK-DAG:  #[[$map0:.*]] = affine_map<(d0)[s0] -> (1, -d0 + s0)>
// CHECK-DAG:  #[[$map1:.*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:  #[[$map2:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:  #[[$map3:.*]] = affine_map<(d0, d1)[s0] -> (3, -d0 - d1 + s0)>
// CHECK-DAG:  #[[$map4:.*]] = affine_map<(d0)[s0] -> (3, -d0 + s0)>

// CHECK-LABEL: @conv_1d
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<?xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32
func @conv_1d(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
//   CHECK-DAG:   %[[c12:.*]] = constant 12 : index
//   CHECK-DAG:   %[[c4:.*]] = constant 4 : index
//   CHECK-DAG:   %[[cst:.*]] = constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[c3:.*]] = constant 3 : index
//   CHECK-DAG:   %[[c0:.*]] = constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = constant 1 : index
//       CHECK:   %[[v0:.*]] = dim %[[arg1]], %[[c0]] : memref<?xf32>
//       CHECK:   %[[v1:.*]] = dim %[[arg2]], %[[c0]] : memref<?xf32>
//       CHECK:   %[[v2:.*]] = dim %[[arg0]], %[[c0]] : memref<?xf32>
//       CHECK:   %[[v3:.*]] = memref.alloc(%[[c12]]) : memref<?xi8>
//       CHECK:   %[[v4:.*]] = memref.alloc(%[[c12]]) : memref<?xi8>
//       CHECK:   %[[v5:.*]] = memref.alloc(%[[c4]]) : memref<?xi8>
//       CHECK:   %[[v6:.*]] = memref.view %[[v3]][%[[c0]]][] : memref<?xi8> to memref<3xf32>
//       CHECK:   %[[v7:.*]] = memref.view %[[v4]][%[[c0]]][] : memref<?xi8> to memref<3xf32>
//       CHECK:   %[[v8:.*]] = memref.view %[[v5]][%[[c0]]][] : memref<?xi8> to memref<1xf32>
//       CHECK:   scf.for %[[arg3:.*]] = %[[c0]] to %[[v1]] step %[[c1]] {
//       CHECK:     %[[v9:.*]] = affine.min #[[$map0]](%[[arg3]])[%[[v1]]]
//       CHECK:     %[[v10:.*]] = subview %[[arg2]][%[[arg3]]] [%[[v9]]] [1]  : memref<?xf32> to memref<?xf32, #[[$map1]]>
//       CHECK:     %[[v11:.*]] = subview %[[v8]][0] [%[[v9]]] [1]  : memref<1xf32> to memref<?xf32>
//       CHECK:     scf.for %[[arg4:.*]] = %[[c0]] to %[[v0]] step %[[c3]] {
//       CHECK:       %[[v12:.*]] = affine.apply #[[$map2]](%[[arg3]], %[[arg4]])
//       CHECK:       %[[v13:.*]] = affine.min #[[$map3]](%[[arg3]], %[[arg4]])[%[[v2]]]
//       CHECK:       %[[v14:.*]] = subview %arg0[%12] [%13] [1]  : memref<?xf32> to memref<?xf32, #[[$map1]]>
//       CHECK:       %[[v15:.*]] = affine.min #[[$map4]](%arg4)[%0]
//       CHECK:       %[[v16:.*]] = subview %[[arg1]][%[[arg4]]] [%[[v15]]] [1]  : memref<?xf32> to memref<?xf32, #[[$map1]]>
//       CHECK:       %[[v17:.*]] = subview %[[v6]][0] [%[[v13]]] [1]  : memref<3xf32> to memref<?xf32>
//       CHECK:       %[[v19:.*]] = vector.transfer_read %[[v6]][%[[c0]]], %[[cst]] {masked = [false]} : memref<3xf32>, vector<3xf32>
//       CHECK:       %[[v20:.*]] = vector.transfer_read %[[v7]][%[[c0]]], %[[cst]] {masked = [false]} : memref<3xf32>, vector<3xf32>
//       CHECK:       %[[v21:.*]] = mulf %[[v19]], %[[v20]] : vector<3xf32>
//       CHECK:       %[[v22:.*]] = vector.reduction "add", %[[v21]], %[[cst]] : vector<3xf32> into f32
//       CHECK:       store %[[v22]], %[[v8]][%[[c0]]] : memref<1xf32>
//       CHECK:       scf.for %[[arg5:.*]] = %[[c0]] to %[[v9]] step %[[c1]] {
//       CHECK:         %[[v23:.*]] = load %[[v11]][%[[arg5]]] : memref<?xf32>
//       CHECK:         store %[[v23]], %[[v10]][%[[arg5]]] : memref<?xf32, #[[$map1]]>
  linalg.conv_1d ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>)
                outs(%arg2 : memref<?xf32>)
  return
}

