// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-allocs always-aliasing-with-dest=0" -split-input-file | FileCheck %s

// CHECK-LABEL: func @linalg_op_bufferizes_inplace_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_bufferizes_inplace_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = true},
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = false},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t1]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%t1, %t2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%t3 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %add = arith.addf %arg0, %arg1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @linalg_op_bufferizes_out_of_place_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_bufferizes_out_of_place_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = false},
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = false},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[t1]], %[[alloc]]
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[alloc]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%t1, %t2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%t3 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %add = arith.addf %arg0, %arg1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
  // CHECK: return %[[alloc]]
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @linalg_op_output_cannot_alias_with_input
//  CHECK-SAME:     %[[t1:.*]]: memref<?x?xf32, #{{.*}}>, %[[t2:.*]]: memref<?xf32, #{{.*}}>, %[[t3:.*]]: memref<?x?xf32, #{{.*}}>
func @linalg_op_output_cannot_alias_with_input(
    %t1: tensor<?x?xf32> {linalg.inplaceable = true},
    %t2: tensor<?xf32> {linalg.inplaceable = false},
    %t3: tensor<?x?xf32> {linalg.inplaceable = true},
    %s1: index, %s2: index, %cst: f32) -> tensor<?x?xf32> {
  // CHECK: linalg.generic {{.*}} ins(%[[t1]], %[[t2]] : {{.*}}) outs(%[[t3]] : {{.*}})
  %r = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%t1, %t2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%t3 : tensor<?x?xf32>) {
      ^bb0(%arg0 : f32, %arg1 : f32, %arg2 : f32) :
        %add = arith.addf %arg0, %arg1 : f32
        linalg.yield %add : f32
    } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

