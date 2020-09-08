// RUN: mlir-opt %s -test-conv-vectorization --cse | FileCheck %s

// CHECK-DAG:  #[[$map0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:  #[[$map1:.*]] = affine_map<(d0) -> ()>
// CHECK-DAG:  #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[$map3:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[$map4:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:  #[[$map5:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG:  #[[$map6:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[$map7:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG:  #[[$map8:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
// CHECK-DAG:  #[[$map9:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:  #[[$map10:.*]] = affine_map<(d0, d1, d2, d3) -> ()>

func @conv_1d(%arg0: memref<3xf32>, %arg1: memref<3xf32>, %arg2: memref<?xf32>) {
  linalg.conv_1d %arg0, %arg1, %arg2 : (memref<3xf32>, memref<3xf32>, memref<?xf32>)
  return
}

// CHECK-LABEL: @conv_1d
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]]], %[[cst]] : memref<3xf32>, vector<3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]]], %[[cst]] : memref<3xf32>, vector<3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map0]], #[[$map1]]], iterator_types = ["reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3xf32>, vector<3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]]] : memref<?xf32>
//       CHECK:   return

func @conv_1d_ncw(%arg0: memref<1x3x3xf32>, %arg1: memref<1x3x3xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_1d_ncw %arg0, %arg1, %arg2 : (memref<1x3x3xf32>, memref<1x3x3xf32>, memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_1d_ncw
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map3]], #[[$map3]], #[[$map4]]], iterator_types = ["reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3xf32>, vector<3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?xf32>
//       CHECK:   return


func @conv_1d_nwc(%arg0: memref<1x3x3xf32>, %arg1: memref<1x3x3xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_1d_nwc %arg0, %arg1, %arg2 : (memref<1x3x3xf32>, memref<1x3x3xf32>, memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_1d_nwc
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map3]], #[[$map3]], #[[$map4]]], iterator_types = ["reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3xf32>, vector<3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?xf32>
//       CHECK:   return

func @conv_2d(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<?x?xf32>) {
  linalg.conv_2d %arg0, %arg1, %arg2 : (memref<3x3xf32>, memref<3x3xf32>, memref<?x?xf32>)
  return
}

// CHECK-LABEL: @conv_2d
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]]], %[[cst]] : memref<3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]]], %[[cst]] : memref<3x3xf32>, vector<3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map3]], #[[$map3]], #[[$map4]]], iterator_types = ["reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3xf32>, vector<3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]]] : memref<?x?xf32>
//       CHECK:   return

func @conv_2d_nchw(%arg0: memref<1x3x3x3xf32>, %arg1: memref<1x3x3x3xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nchw %arg0, %arg1, %arg2 : (memref<1x3x3x3xf32>, memref<1x3x3x3xf32>, memref<?x?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_2d_nchw
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map6]], #[[$map6]], #[[$map7]]], iterator_types = ["reduction", "reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3x3xf32>, vector<3x3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?x?xf32>
//       CHECK:   return

func @conv_2d_nhwc(%arg0: memref<1x3x3x3xf32>, %arg1: memref<1x3x3x3xf32>, %arg2: memref<?x?x?x?xf32>) {
  linalg.conv_2d_nhwc %arg0, %arg1, %arg2 : (memref<1x3x3x3xf32>, memref<1x3x3x3xf32>, memref<?x?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_2d_nhwc
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map6]], #[[$map6]], #[[$map7]]], iterator_types = ["reduction", "reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3x3xf32>, vector<3x3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?x?xf32>
//       CHECK:   return

func @conv_3d(%arg0: memref<3x3x3xf32>, %arg1: memref<3x3x3xf32>, %arg2: memref<?x?x?xf32>) {
  linalg.conv_3d %arg0, %arg1, %arg2 : (memref<3x3x3xf32>, memref<3x3x3xf32>, memref<?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_3d
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<3x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<3x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<3x3x3xf32>, vector<3x3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map6]], #[[$map6]], #[[$map7]]], iterator_types = ["reduction", "reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3x3xf32>, vector<3x3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?xf32>
//       CHECK:   return

func @conv_3d_ncdhw(%arg0: memref<1x3x3x3x3xf32>, %arg1: memref<1x3x3x3x3xf32>, %arg2: memref<?x?x?x?x?xf32>) {
  linalg.conv_3d_ncdhw %arg0, %arg1, %arg2 : (memref<1x3x3x3x3xf32>, memref<1x3x3x3x3xf32>, memref<?x?x?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_3d_ncdhw
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3x3xf32>, vector<3x3x3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3x3xf32>, vector<3x3x3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map9]], #[[$map9]], #[[$map10]]], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3x3x3xf32>, vector<3x3x3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?x?x?xf32>
//       CHECK:   return

func @conv_3d_ndhwc(%arg0: memref<1x3x3x3x3xf32>, %arg1: memref<1x3x3x3x3xf32>, %arg2: memref<?x?x?x?x?xf32>) {
  linalg.conv_3d_ndhwc %arg0, %arg1, %arg2 : (memref<1x3x3x3x3xf32>, memref<1x3x3x3x3xf32>, memref<?x?x?x?x?xf32>)
  return
}

// CHECK-LABEL: @conv_3d_ndhwc
//  CHECK-SAME: %[[arg0:[a-zA-Z0-9]+]]: memref<1x3x3x3x3xf32>
//  CHECK-SAME: %[[arg1:[a-zA-Z0-9]+]]: memref<1x3x3x3x3xf32>
//  CHECK-SAME: %[[arg2:[a-zA-Z0-9]+]]: memref<?x?x?x?x?xf32
//       CHECK:   %[[c0:.*]] = constant 0 : index
//       CHECK:   %[[cst:.*]] = constant 0.000000e+00 : f32
//       CHECK:   %[[v0:.*]] = vector.transfer_read %[[arg0]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3x3xf32>, vector<3x3x3x3xf32>
//       CHECK:   %[[v1:.*]] = vector.transfer_read %[[arg1]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]], %[[cst]] : memref<1x3x3x3x3xf32>, vector<3x3x3x3xf32>
//       CHECK:   %[[v2:.*]] = vector.contract {indexing_maps = [#[[$map9]], #[[$map9]], #[[$map10]]], iterator_types = ["reduction", "reduction", "reduction", "reduction"]} %[[v0]], %[[v1]], %[[cst]] : vector<3x3x3x3xf32>, vector<3x3x3x3xf32> into f32
//       CHECK:   store %[[v2]], %[[arg2]][%[[c0]], %[[c0]], %[[c0]], %[[c0]], %[[c0]]] : memref<?x?x?x?x?xf32>
//       CHECK:   return
