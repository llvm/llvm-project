// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file \
// RUN:   --test-vector-warp-distribute=rewrite-warp-ops-to-scf-if | FileCheck %s --check-prefix=CHECK-SCF-IF

// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file \
// RUN:   --test-vector-warp-distribute="hoist-uniform" | FileCheck --check-prefixes=CHECK-HOIST %s

// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file \
// RUN:   --test-vector-warp-distribute="hoist-uniform distribute-transfer-write max-transfer-write-elements=4" \
// RUN:   | FileCheck --check-prefixes=CHECK-D %s

// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file \
// RUN:  --test-vector-warp-distribute=propagate-distribution --canonicalize \
// RUN:  | FileCheck --check-prefixes=CHECK-PROP %s

// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file \
// RUN:   --test-vector-warp-distribute="hoist-uniform distribute-transfer-write propagate-distribution" \
// RUN:   --canonicalize | FileCheck --check-prefixes=CHECK-DIST-AND-PROP %s

// CHECK-SCF-IF-DAG: #[[$TIMES2:.*]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-SCF-IF-DAG: #[[$TIMES4:.*]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-SCF-IF-DAG: #[[$TIMES8:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_32xf32 : memref<32xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_64xf32 : memref<64xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_128xf32 : memref<128xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_256xf32 : memref<256xf32, 3>

// CHECK-SCF-IF-LABEL: func @rewrite_warp_op_to_scf_if(
//  CHECK-SCF-IF-SAME:     %[[laneid:.*]]: index,
//  CHECK-SCF-IF-SAME:     %[[v0:.*]]: vector<4xf32>, %[[v1:.*]]: vector<8xf32>)
func.func @rewrite_warp_op_to_scf_if(%laneid: index,
                                %v0: vector<4xf32>, %v1: vector<8xf32>) {
//   CHECK-SCF-IF-DAG:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK-SCF-IF:   %[[is_lane_0:.*]] = arith.cmpi eq, %[[laneid]], %[[c0]]

//       CHECK-SCF-IF:   %[[buffer_v0:.*]] = memref.get_global @__shared_128xf32
//       CHECK-SCF-IF:   %[[s0:.*]] = affine.apply #[[$TIMES4]]()[%[[laneid]]]
//       CHECK-SCF-IF:   vector.transfer_write %[[v0]], %[[buffer_v0]][%[[s0]]]
//       CHECK-SCF-IF:   %[[buffer_v1:.*]] = memref.get_global @__shared_256xf32
//       CHECK-SCF-IF:   %[[s1:.*]] = affine.apply #[[$TIMES8]]()[%[[laneid]]]
//       CHECK-SCF-IF:   vector.transfer_write %[[v1]], %[[buffer_v1]][%[[s1]]]

//   CHECK-SCF-IF-DAG:   gpu.barrier
//   CHECK-SCF-IF-DAG:   %[[buffer_def_0:.*]] = memref.get_global @__shared_32xf32
//   CHECK-SCF-IF-DAG:   %[[buffer_def_1:.*]] = memref.get_global @__shared_64xf32

//       CHECK-SCF-IF:   scf.if %[[is_lane_0]] {
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32]
      args(%v0, %v1 : vector<4xf32>, vector<8xf32>) -> (vector<1xf32>, vector<2xf32>) {
    ^bb0(%arg0: vector<128xf32>, %arg1: vector<256xf32>):
//       CHECK-SCF-IF:     %[[arg1:.*]] = vector.transfer_read %[[buffer_v1]][%[[c0]]], %{{.*}} {in_bounds = [true]} : memref<256xf32, 3>, vector<256xf32>
//       CHECK-SCF-IF:     %[[arg0:.*]] = vector.transfer_read %[[buffer_v0]][%[[c0]]], %{{.*}} {in_bounds = [true]} : memref<128xf32, 3>, vector<128xf32>
//       CHECK-SCF-IF:     %[[def_0:.*]] = "some_def"(%[[arg0]]) : (vector<128xf32>) -> vector<32xf32>
//       CHECK-SCF-IF:     %[[def_1:.*]] = "some_def"(%[[arg1]]) : (vector<256xf32>) -> vector<64xf32>
    %2 = "some_def"(%arg0) : (vector<128xf32>) -> vector<32xf32>
    %3 = "some_def"(%arg1) : (vector<256xf32>) -> vector<64xf32>
//       CHECK-SCF-IF:     vector.transfer_write %[[def_0]], %[[buffer_def_0]][%[[c0]]]
//       CHECK-SCF-IF:     vector.transfer_write %[[def_1]], %[[buffer_def_1]][%[[c0]]]
    gpu.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
//       CHECK-SCF-IF:   }
//       CHECK-SCF-IF:   gpu.barrier
//       CHECK-SCF-IF:   %[[o1:.*]] = affine.apply #[[$TIMES2]]()[%[[laneid]]]
//       CHECK-SCF-IF:   %[[r1:.*]] = vector.transfer_read %[[buffer_def_1]][%[[o1]]], %{{.*}} {in_bounds = [true]} : memref<64xf32, 3>, vector<2xf32>
//       CHECK-SCF-IF:   %[[r0:.*]] = vector.transfer_read %[[buffer_def_0]][%[[laneid]]], %{{.*}} {in_bounds = [true]} : memref<32xf32, 3>, vector<1xf32>
//       CHECK-SCF-IF:   "some_use"(%[[r0]]) : (vector<1xf32>) -> ()
//       CHECK-SCF-IF:   "some_use"(%[[r1]]) : (vector<2xf32>) -> ()
  "some_use"(%r#0) : (vector<1xf32>) -> ()
  "some_use"(%r#1) : (vector<2xf32>) -> ()
  return
}

// -----

// CHECK-D-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 2 + 32)>

// CHECK-DIST-AND-PROP-LABEL: func @warp(
// CHECK-HOIST: memref.subview
// CHECK-HOIST: memref.subview
// CHECK-HOIST: memref.subview
// CHECK-HOIST: gpu.warp_execute_on_lane_0

//     CHECK-D: %[[R:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<2xf32>, vector<1xf32>) {
//     CHECK-D:   arith.addf {{.*}} : vector<32xf32>
//     CHECK-D:   arith.addf {{.*}} : vector<64xf32>
//     CHECK-D:   gpu.yield %{{.*}}, %{{.*}} : vector<64xf32>, vector<32xf32>
// CHECK-D-DAG: vector.transfer_write %[[R]]#1, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<1xf32>, memref<128xf32
// CHECK-D-DAG: %[[ID1:.*]] = affine.apply #[[MAP1]]()[%{{.*}}]
// CHECK-D-DAG: vector.transfer_write %[[R]]#0, %{{.*}}[%[[ID1]]] {in_bounds = [true]} : vector<2xf32>, memref<128xf32

// CHECK-DIST-AND-PROP-NOT: gpu.warp_execute_on_lane_0
// CHECK-DIST-AND-PROP: vector.transfer_read {{.*}} vector<1xf32>
// CHECK-DIST-AND-PROP: vector.transfer_read {{.*}} vector<1xf32>
// CHECK-DIST-AND-PROP: vector.transfer_read {{.*}} vector<2xf32>
// CHECK-DIST-AND-PROP: vector.transfer_read {{.*}} vector<2xf32>
// CHECK-DIST-AND-PROP: arith.addf {{.*}} : vector<1xf32>
// CHECK-DIST-AND-PROP: arith.addf {{.*}} : vector<2xf32>
// CHECK-DIST-AND-PROP: vector.transfer_write {{.*}} : vector<1xf32>
// CHECK-DIST-AND-PROP: vector.transfer_write {{.*}} : vector<2xf32>

func.func @warp(%laneid: index, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>,
           %arg3: memref<1024xf32>, %gid : index) {
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %sa = memref.subview %arg1[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, strided<[1], offset: ?>>
    %sb = memref.subview %arg2[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, strided<[1], offset: ?>>
    %sc = memref.subview %arg3[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, strided<[1], offset: ?>>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %2 = vector.transfer_read %sa[%c0], %cst : memref<128xf32, strided<[1], offset: ?>>, vector<32xf32>
    %3 = vector.transfer_read %sa[%c32], %cst : memref<128xf32, strided<[1], offset: ?>>, vector<32xf32>
    %4 = vector.transfer_read %sb[%c0], %cst : memref<128xf32, strided<[1], offset: ?>>, vector<64xf32>
    %5 = vector.transfer_read %sb[%c32], %cst : memref<128xf32, strided<[1], offset: ?>>, vector<64xf32>
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    vector.transfer_write %6, %sc[%c0] : vector<32xf32>, memref<128xf32, strided<[1], offset: ?>>
    vector.transfer_write %7, %sc[%c32] : vector<64xf32>, memref<128xf32, strided<[1], offset: ?>>
  }
  return
}

// -----

// CHECK-D-LABEL: func @warp_extract(
//       CHECK-D:   %[[WARPOP:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>, vector<1x1xf32>)
//       CHECK-D:     "test.dummy_op"
//       CHECK-D:     "test.dummy_op"
//       CHECK-D:     gpu.yield %{{.*}}, %{{.*}} : vector<1xf32>, vector<1x1xf32>
//       CHECK-D:   }
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     vector.transfer_write %[[WARPOP]]#1, %{{.*}}[%{{.*}}] {{.*}} : vector<1x1xf32>
//       CHECK-D:   }
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     vector.transfer_write %[[WARPOP]]#0, %{{.*}}[%{{.*}}] {{.*}} : vector<1xf32>
//       CHECK-D:   }

func.func @warp_extract(%laneid: index, %arg1: memref<1024x1024xf32>, %gid : index) {
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %c0 = arith.constant 0 : index
    %v = "test.dummy_op"() : () -> (vector<1xf32>)
    %v1 = "test.dummy_op"() : () -> (vector<1x1xf32>)
    vector.transfer_write %v1, %arg1[%c0, %c0] : vector<1x1xf32>, memref<1024x1024xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] : vector<1xf32>, memref<1024x1024xf32>
  }
  return
}

// -----

// Check that we can distribute writes of the maximum allowed number of elements.

// CHECK-D-LABEL: func @warp_extract_4_elems(
//       CHECK-D:   %[[WARPOP:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>, vector<4x1xf32>)
//       CHECK-D:     "test.dummy_op"
//       CHECK-D:     "test.dummy_op"
//       CHECK-D:     gpu.yield %{{.*}}, %{{.*}} : vector<4xf32>, vector<4x1xf32>
//       CHECK-D:   }
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     vector.transfer_write %[[WARPOP]]#1, %{{.*}}[%{{.*}}] {{.*}} : vector<4x1xf32>
//       CHECK-D:   }
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     vector.transfer_write %[[WARPOP]]#0, %{{.*}}[%{{.*}}] {{.*}} : vector<4xf32>
//       CHECK-D:   }

func.func @warp_extract_4_elems(%laneid: index, %arg1: memref<1024x1024xf32>, %gid : index) {
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %c0 = arith.constant 0 : index
    %v = "test.dummy_op"() : () -> (vector<4xf32>)
    %v1 = "test.dummy_op"() : () -> (vector<4x1xf32>)
    vector.transfer_write %v1, %arg1[%c0, %c0] : vector<4x1xf32>, memref<1024x1024xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] : vector<4xf32>, memref<1024x1024xf32>
  }
  return
}

// -----

// Check that we do not distribute writes larger than the maximum allowed
// number of elements.

// CHECK-D-LABEL: func @warp_extract_5_elems(
//       CHECK-D:   arith.constant 0 : index
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     %[[V:.+]] = "test.dummy_op"
//       CHECK-D:     %[[V1:.+]] = "test.dummy_op"
//       CHECK-D:     vector.transfer_write %[[V1]], %{{.*}}[%{{.*}}] {{.*}} : vector<5x1xf32>
//       CHECK-D:     vector.transfer_write %[[V]], %{{.*}}[%{{.*}}] {{.*}} : vector<5xf32>
//       CHECK-D:   }

func.func @warp_extract_5_elems(%laneid: index, %arg1: memref<1024x1024xf32>, %gid : index) {
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %c0 = arith.constant 0 : index
    %v = "test.dummy_op"() : () -> (vector<5xf32>)
    %v1 = "test.dummy_op"() : () -> (vector<5x1xf32>)
    vector.transfer_write %v1, %arg1[%c0, %c0] : vector<5x1xf32>, memref<1024x1024xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] : vector<5xf32>, memref<1024x1024xf32>
  }
  return
}

// -----

// Check that we do not distribute writes larger than the maximum allowed
// number of elements, or multiples of the maximum number of elements.

// CHECK-D-LABEL: func @warp_extract_8_elems(
//       CHECK-D:   arith.constant 0 : index
//       CHECK-D:   gpu.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     %[[V:.+]] = "test.dummy_op"
//       CHECK-D:     %[[V1:.+]] = "test.dummy_op"
//       CHECK-D:     vector.transfer_write %[[V1]], %{{.*}}[%{{.*}}] {{.*}} : vector<8x1xf32>
//       CHECK-D:     vector.transfer_write %[[V]], %{{.*}}[%{{.*}}] {{.*}} : vector<8xf32>
//       CHECK-D:   }

func.func @warp_extract_8_elems(%laneid: index, %arg1: memref<1024x1024xf32>, %gid : index) {
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %c0 = arith.constant 0 : index
    %v = "test.dummy_op"() : () -> (vector<8xf32>)
    %v1 = "test.dummy_op"() : () -> (vector<8x1xf32>)
    vector.transfer_write %v1, %arg1[%c0, %c0] : vector<8x1xf32>, memref<1024x1024xf32>
    vector.transfer_write %v, %arg1[%c0, %c0] : vector<8xf32>, memref<1024x1024xf32>
  }
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_dead_result(
func.func @warp_dead_result(%laneid: index) -> (vector<1xf32>) {
  // CHECK-PROP: %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>)
  %r:3 = gpu.warp_execute_on_lane_0(%laneid)[32] ->
    (vector<1xf32>, vector<1xf32>, vector<1xf32>) {
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<32xf32>)
  // CHECK-PROP:   gpu.yield %{{.*}} : vector<32xf32>
    gpu.yield %2, %3, %4 : vector<32xf32>, vector<32xf32>, vector<32xf32>
  }
  // CHECK-PROP: return %[[R]] : vector<1xf32>
  return %r#1 : vector<1xf32>
}

// -----

// CHECK-PROP-LABEL:   func @warp_propagate_operand(
//  CHECK-PROP-SAME:   %[[ID:.*]]: index, %[[V:.*]]: vector<4xf32>)
func.func @warp_propagate_operand(%laneid: index, %v0: vector<4xf32>)
  -> (vector<4xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32]
     args(%v0 : vector<4xf32>) -> (vector<4xf32>) {
     ^bb0(%arg0 : vector<128xf32>) :
    gpu.yield %arg0 : vector<128xf32>
  }
  // CHECK-PROP: return %[[V]] : vector<4xf32>
  return %r : vector<4xf32>
}

// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>

// CHECK-PROP-LABEL:   func @warp_propagate_elementwise(
func.func @warp_propagate_elementwise(%laneid: index, %dest: memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK-PROP: %[[R:.*]]:4 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>, vector<1xf32>, vector<2xf32>, vector<2xf32>)
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] ->
    (vector<1xf32>, vector<2xf32>) {
    // CHECK-PROP: %[[V0:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK-PROP: %[[V1:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK-PROP: %[[V2:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK-PROP: %[[V3:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK-PROP: gpu.yield %[[V0]], %[[V1]], %[[V2]], %[[V3]] : vector<32xf32>, vector<32xf32>, vector<64xf32>, vector<64xf32>
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<64xf32>)
    %5 = "some_def"() : () -> (vector<64xf32>)
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    gpu.yield %6, %7 : vector<32xf32>, vector<64xf32>
  }
  // CHECK-PROP: %[[A0:.*]] = arith.addf %[[R]]#2, %[[R]]#3 : vector<2xf32>
  // CHECK-PROP: %[[A1:.*]] = arith.addf %[[R]]#0, %[[R]]#1 : vector<1xf32>
  %id2 = affine.apply #map0()[%laneid]
  // CHECK-PROP: vector.transfer_write %[[A1]], {{.*}} : vector<1xf32>, memref<1024xf32>
  // CHECK-PROP: vector.transfer_write %[[A0]], {{.*}} : vector<2xf32>, memref<1024xf32>
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @warp_propagate_scalar_arith(
//       CHECK-PROP:   %[[r:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} {
//       CHECK-PROP:     %[[some_def0:.*]] = "some_def"
//       CHECK-PROP:     %[[some_def1:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[some_def0]], %[[some_def1]]
//       CHECK-PROP:   }
//       CHECK-PROP:   arith.addf %[[r]]#0, %[[r]]#1 : f32
func.func @warp_propagate_scalar_arith(%laneid: index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (f32)
    %1 = "some_def"() : () -> (f32)
    %2 = arith.addf %0, %1 : f32
    gpu.yield %2 : f32
  }
  vector.print %r : f32
  return
}

// -----

// CHECK-PROP-LABEL: func @warp_propagate_cast(
//   CHECK-PROP-NOT:   gpu.warp_execute_on_lane_0
//       CHECK-PROP:   %[[result:.*]] = arith.sitofp %{{.*}} : i32 to f32
//       CHECK-PROP:   return %[[result]]
func.func @warp_propagate_cast(%laneid : index, %i : i32) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %casted = arith.sitofp %i : i32 to f32
    gpu.yield %casted : f32
  }
  return %r : f32
}

// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>

//  CHECK-PROP-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK-PROP:   func @warp_propagate_read
//  CHECK-PROP-SAME:     (%[[ID:.*]]: index
func.func @warp_propagate_read(%laneid: index, %src: memref<1024xf32>, %dest: memref<1024xf32>) {
// CHECK-PROP-NOT: warp_execute_on_lane_0
// CHECK-PROP-DAG: %[[R0:.*]] = vector.transfer_read %arg1[%[[ID]]], %{{.*}} : memref<1024xf32>, vector<1xf32>
// CHECK-PROP-DAG: %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
// CHECK-PROP-DAG: %[[R1:.*]] = vector.transfer_read %arg1[%[[ID2]]], %{{.*}} : memref<1024xf32>, vector<2xf32>
// CHECK-PROP: vector.transfer_write %[[R0]], {{.*}} : vector<1xf32>, memref<1024xf32>
// CHECK-PROP: vector.transfer_write %[[R1]], {{.*}} : vector<2xf32>, memref<1024xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] ->(vector<1xf32>, vector<2xf32>) {
    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<32xf32>
    %3 = vector.transfer_read %src[%c32], %cst : memref<1024xf32>, vector<64xf32>
    gpu.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
  %id2 = affine.apply #map0()[%laneid]
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @fold_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[some_def]] : vector<1xf32>
//       CHECK-PROP:   vector.print %[[r]] : vector<1xf32>
func.func @fold_vector_broadcast(%laneid: index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<32xf32>
    gpu.yield %1 : vector<32xf32>
  }
  vector.print %r : vector<1xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @extract_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[some_def]] : vector<1xf32>
//       CHECK-PROP:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : vector<1xf32> to vector<2xf32>
//       CHECK-PROP:   vector.print %[[broadcasted]] : vector<2xf32>
func.func @extract_vector_broadcast(%laneid: index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<64xf32>
    gpu.yield %1 : vector<64xf32>
  }
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @extract_scalar_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (f32)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[some_def]] : f32
//       CHECK-PROP:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : f32 to vector<2xf32>
//       CHECK-PROP:   vector.print %[[broadcasted]] : vector<2xf32>
func.func @extract_scalar_vector_broadcast(%laneid: index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (f32)
    %1 = vector.broadcast %0 : f32 to vector<64xf32>
    gpu.yield %1 : vector<64xf32>
  }
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for(
// CHECK-PROP: %[[INI:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>) {
// CHECK-PROP:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   gpu.yield %[[INI1]] : vector<128xf32>
// CHECK-PROP: }
// CHECK-PROP: %[[F:.*]] = scf.for %[[IT:.+]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG:.*]] = %[[INI]]) -> (vector<4xf32>) {
// CHECK-PROP:   %[[A:.*]] = arith.addi %[[IT]], %{{.*}} : index
// CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%[[FARG]] : vector<4xf32>) -> (vector<4xf32>) {
// CHECK-PROP:    ^bb0(%[[ARG:.*]]: vector<128xf32>):
// CHECK-PROP:      %[[ACC:.*]] = "some_def"(%[[A]], %[[ARG]]) : (index, vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      gpu.yield %[[ACC]] : vector<128xf32>
// CHECK-PROP:   }
// CHECK-PROP:   scf.yield %[[W]] : vector<4xf32>
// CHECK-PROP: }
// CHECK-PROP: "some_use"(%[[F]]) : (vector<4xf32>) -> ()
func.func @warp_scf_for(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>) {
    %ini = "some_def"() : () -> (vector<128xf32>)
    %3 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini) -> (vector<128xf32>) {
      %add = arith.addi %arg3, %c1 : index
      %acc = "some_def"(%add, %arg4) : (index, vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc : vector<128xf32>
    }
    gpu.yield %3 : vector<128xf32>
  }
  "some_use"(%0) : (vector<4xf32>) -> ()
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for_use_from_above(
// CHECK-PROP: %[[INI:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   %[[USE:.*]] = "some_def_above"() : () -> vector<128xf32>
// CHECK-PROP:   gpu.yield %[[INI1]], %[[USE]] : vector<128xf32>, vector<128xf32>
// CHECK-PROP: }
// CHECK-PROP: %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG:.*]] = %[[INI]]#0) -> (vector<4xf32>) {
// CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%[[FARG]], %[[INI]]#1 : vector<4xf32>, vector<4xf32>) -> (vector<4xf32>) {
// CHECK-PROP:    ^bb0(%[[ARG0:.*]]: vector<128xf32>, %[[ARG1:.*]]: vector<128xf32>):
// CHECK-PROP:      %[[ACC:.*]] = "some_def"(%[[ARG0]], %[[ARG1]]) : (vector<128xf32>, vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      gpu.yield %[[ACC]] : vector<128xf32>
// CHECK-PROP:   }
// CHECK-PROP:   scf.yield %[[W]] : vector<4xf32>
// CHECK-PROP: }
// CHECK-PROP: "some_use"(%[[F]]) : (vector<4xf32>) -> ()
func.func @warp_scf_for_use_from_above(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>) {
    %ini = "some_def"() : () -> (vector<128xf32>)
    %use_from_above = "some_def_above"() : () -> (vector<128xf32>)
    %3 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini) -> (vector<128xf32>) {
      %acc = "some_def"(%arg4, %use_from_above) : (vector<128xf32>, vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc : vector<128xf32>
    }
    gpu.yield %3 : vector<128xf32>
  }
  "some_use"(%0) : (vector<4xf32>) -> ()
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for_swap(
// CHECK-PROP: %[[INI:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   %[[INI2:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   gpu.yield %[[INI1]], %[[INI2]] : vector<128xf32>, vector<128xf32>
// CHECK-PROP: }
// CHECK-PROP: %[[F:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG1:.*]] = %[[INI]]#0, %[[FARG2:.*]] = %[[INI]]#1) -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%[[FARG1]], %[[FARG2]] : vector<4xf32>, vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:    ^bb0(%[[ARG1:.*]]: vector<128xf32>, %[[ARG2:.*]]: vector<128xf32>):
// CHECK-PROP:      %[[ACC1:.*]] = "some_def"(%[[ARG1]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      %[[ACC2:.*]] = "some_def"(%[[ARG2]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      gpu.yield %[[ACC2]], %[[ACC1]] : vector<128xf32>, vector<128xf32>
// CHECK-PROP:   }
// CHECK-PROP:   scf.yield %[[W]]#0, %[[W]]#1 : vector<4xf32>, vector<4xf32>
// CHECK-PROP: }
// CHECK-PROP: "some_use"(%[[F]]#0) : (vector<4xf32>) -> ()
// CHECK-PROP: "some_use"(%[[F]]#1) : (vector<4xf32>) -> ()
func.func @warp_scf_for_swap(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:2 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>, vector<4xf32>) {
    %ini1 = "some_def"() : () -> (vector<128xf32>)
    %ini2 = "some_def"() : () -> (vector<128xf32>)
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini1, %arg5 = %ini2) -> (vector<128xf32>, vector<128xf32>) {
      %acc1 = "some_def"(%arg4) : (vector<128xf32>) -> (vector<128xf32>)
      %acc2 = "some_def"(%arg5) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc2, %acc1 : vector<128xf32>, vector<128xf32>
    }
    gpu.yield %3#0, %3#1 : vector<128xf32>, vector<128xf32>
  }
  "some_use"(%0#0) : (vector<4xf32>) -> ()
  "some_use"(%0#1) : (vector<4xf32>) -> ()
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for_swap_no_yield(
// CHECK-PROP:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-PROP-NEXT:        gpu.warp_execute_on_lane_0(%{{.*}})[32] {
// CHECK-PROP-NEXT:          "some_op"() : () -> ()
// CHECK-PROP-NEXT:        }
// CHECK-PROP-NEXT:      }
func.func @warp_scf_for_swap_no_yield(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  gpu.warp_execute_on_lane_0(%arg0)[32] {
    scf.for %arg3 = %c0 to %c128 step %c1 {
      "some_op"() : () -> ()
    }
  }
  return
}

// -----
// scf.for result is not distributed in this case.
// CHECK-PROP-LABEL:   func @warp_scf_for_broadcasted_result(
// CHECK-PROP:  %[[W0:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
// CHECK-PROP:    %[[INI:.*]] = "some_def"() : () -> vector<1xf32>
// CHECK-PROP:    gpu.yield %[[INI]] : vector<1xf32>
// CHECK-PROP:  }
// CHECK-PROP:  %[[F:.*]] = scf.for {{.*}} iter_args(%[[ARG2:.*]] = %[[W0]]) -> (vector<1xf32>) {
// CHECK-PROP:    %[[W1:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%[[ARG2]] : vector<1xf32>) -> (vector<1xf32>) {
// CHECK-PROP:    ^bb0(%{{.*}}: vector<1xf32>):
// CHECK-PROP:      %[[T0:.*]] = "some_op"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK-PROP:      gpu.yield %[[T0]] : vector<1xf32>
// CHECK-PROP:    }
// CHECK-PROP:    scf.yield %[[W1]] : vector<1xf32>
func.func @warp_scf_for_broadcasted_result(%arg0: index) -> vector<1xf32> {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %2 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<1xf32>) {
    %ini = "some_def"() : () -> (vector<1xf32>)
    %0 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini) -> (vector<1xf32>) {
      %1 = "some_op"(%arg4) : (vector<1xf32>) -> (vector<1xf32>)
      scf.yield %1 : vector<1xf32>
    }
    gpu.yield %0 : vector<1xf32>
  }
  return %2 : vector<1xf32>
}

// -----

#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 128 + 128)>
#map2 = affine_map<()[s0] -> (s0 * 4 + 128)>

// CHECK-PROP-LABEL:   func @warp_scf_for_multiple_yield(
//       CHECK-PROP:   gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
//  CHECK-PROP-NEXT:     "some_def"() : () -> vector<32xf32>
//  CHECK-PROP-NEXT:     gpu.yield %{{.*}} : vector<32xf32>
//  CHECK-PROP-NEXT:   }
//   CHECK-PROP-NOT:   gpu.warp_execute_on_lane_0
//       CHECK-PROP:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:   %{{.*}}:2 = scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-PROP-NOT:     gpu.warp_execute_on_lane_0
//       CHECK-PROP:     vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:     vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:     arith.addf {{.*}} : vector<4xf32>
//       CHECK-PROP:     arith.addf {{.*}} : vector<4xf32>
//       CHECK-PROP:     scf.yield {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK-PROP:   }
func.func @warp_scf_for_multiple_yield(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0:3 = gpu.warp_execute_on_lane_0(%arg0)[32] ->
  (vector<1xf32>, vector<4xf32>, vector<4xf32>) {
    %def = "some_def"() : () -> (vector<32xf32>)
    %r1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
    %r2 = vector.transfer_read %arg2[%c128], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %r1, %arg5 = %r2)
    -> (vector<128xf32>, vector<128xf32>) {
      %o1 = affine.apply #map1()[%arg3]
      %o2 = affine.apply #map2()[%arg3]
      %4 = vector.transfer_read %arg1[%o1], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %5 = vector.transfer_read %arg1[%o2], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %6 = arith.addf %4, %arg4 : vector<128xf32>
      %7 = arith.addf %5, %arg5 : vector<128xf32>
      scf.yield %6, %7 : vector<128xf32>, vector<128xf32>
    }
    gpu.yield %def, %3#0, %3#1 :  vector<32xf32>, vector<128xf32>, vector<128xf32>
  }
  %1 = affine.apply #map()[%arg0]
  vector.transfer_write %0#1, %arg2[%1] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  %2 = affine.apply #map2()[%arg0]
  vector.transfer_write %0#2, %arg2[%2] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  "some_use"(%0#0) : (vector<1xf32>) -> ()
  return
}

// -----
// CHECK-PROP-LABEL: func.func @warp_scf_for_unused_for_result(
//       CHECK-PROP: %[[W0:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>, vector<4xf32>) {
//       CHECK-PROP:  %[[INI0:.*]] = "some_def"() : () -> vector<128xf32>
//       CHECK-PROP:  %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
//       CHECK-PROP:  gpu.yield %[[INI0]], %[[INI1]] : vector<128xf32>, vector<128xf32>
//       CHECK-PROP: }
//       CHECK-PROP: %[[F:.*]]:2 = scf.for %{{.*}} iter_args(%{{.*}} = %[[W0]]#0, %{{.*}} = %[[W0]]#1) -> (vector<4xf32>, vector<4xf32>) {
//       CHECK-PROP:  %[[W1:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%{{.*}} : vector<4xf32>, vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
//       CHECK-PROP:    %[[ACC0:.*]] = "some_def"(%{{.*}}) : (vector<128xf32>, index) -> vector<128xf32>
//       CHECK-PROP:    %[[ACC1:.*]] = "some_def"(%{{.*}}) : (index, vector<128xf32>, vector<128xf32>) -> vector<128xf32>
//       CHECK-PROP:    gpu.yield %[[ACC1]], %[[ACC0]] : vector<128xf32>, vector<128xf32>
//       CHECK-PROP:  }
//       CHECK-PROP:  scf.yield %[[W1]]#0, %[[W1]]#1 : vector<4xf32>, vector<4xf32>
//       CHECK-PROP: }
//       CHECK-PROP: "some_use"(%[[F]]#0) : (vector<4xf32>) -> ()
func.func @warp_scf_for_unused_for_result(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>) {
    %ini = "some_def"() : () -> (vector<128xf32>)
    %ini1 = "some_def"() : () -> (vector<128xf32>)
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini, %arg5 = %ini1) -> (vector<128xf32>, vector<128xf32>) {
      %add = arith.addi %arg3, %c1 : index
      %1  = "some_def"(%arg5, %add) : (vector<128xf32>, index) -> (vector<128xf32>)
      %acc = "some_def"(%add, %arg4, %1) : (index, vector<128xf32>, vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc, %1 : vector<128xf32>, vector<128xf32>
    }
    gpu.yield %3#0 : vector<128xf32>
  }
  "some_use"(%0) : (vector<4xf32>) -> ()
  return
}

// -----
// CHECK-PROP-LABEL: func.func @warp_scf_for_swapped_for_results(
//       CHECK-PROP:  %[[W0:.*]]:3 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<8xf32>, vector<4xf32>, vector<4xf32>) {
//  CHECK-PROP-NEXT:    %[[INI0:.*]] = "some_def"() : () -> vector<256xf32>
//  CHECK-PROP-NEXT:    %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
//  CHECK-PROP-NEXT:    %[[INI2:.*]] = "some_def"() : () -> vector<128xf32>
//  CHECK-PROP-NEXT:    gpu.yield %[[INI0]], %[[INI1]], %[[INI2]] : vector<256xf32>, vector<128xf32>, vector<128xf32>
//  CHECK-PROP-NEXT:  }
//  CHECK-PROP-NEXT:  %[[F0:.*]]:3 = scf.for {{.*}} iter_args(%{{.*}} = %[[W0]]#0, %{{.*}} = %[[W0]]#1, %{{.*}} = %[[W0]]#2) -> (vector<8xf32>, vector<4xf32>, vector<4xf32>) {
//  CHECK-PROP-NEXT:    %[[W1:.*]]:3 = gpu.warp_execute_on_lane_0(%{{.*}})[32] args(%{{.*}} :
//  CHECK-PROP-SAME:        vector<8xf32>, vector<4xf32>, vector<4xf32>) -> (vector<8xf32>, vector<4xf32>, vector<4xf32>) {
//  CHECK-PROP-NEXT:      ^bb0(%{{.*}}: vector<256xf32>, %{{.*}}: vector<128xf32>, %{{.*}}: vector<128xf32>):
//  CHECK-PROP-NEXT:        %[[T3:.*]] = "some_def_1"(%{{.*}}) : (vector<256xf32>) -> vector<256xf32>
//  CHECK-PROP-NEXT:        %[[T4:.*]] = "some_def_2"(%{{.*}}) : (vector<128xf32>) -> vector<128xf32>
//  CHECK-PROP-NEXT:        %[[T5:.*]] = "some_def_3"(%{{.*}}) : (vector<128xf32>) -> vector<128xf32>
//  CHECK-PROP-NEXT:        gpu.yield %[[T3]], %[[T4]], %[[T5]] : vector<256xf32>, vector<128xf32>, vector<128xf32>
//  CHECK-PROP-NEXT:    }
//  CHECK-PROP-NEXT:    scf.yield %[[W1]]#0, %[[W1]]#1, %[[W1]]#2 : vector<8xf32>, vector<4xf32>, vector<4xf32>
//  CHECK-PROP-NEXT:  }
//  CHECK-PROP-NEXT:  "some_use_1"(%[[F0]]#2) : (vector<4xf32>) -> ()
//  CHECK-PROP-NEXT:  "some_use_2"(%[[F0]]#1) : (vector<4xf32>) -> ()
//  CHECK-PROP-NEXT:  "some_use_3"(%[[F0]]#0) : (vector<8xf32>) -> ()
func.func @warp_scf_for_swapped_for_results(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:3 = gpu.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>, vector<4xf32>, vector<8xf32>) {
    %ini1 = "some_def"() : () -> (vector<256xf32>)
    %ini2 = "some_def"() : () -> (vector<128xf32>)
    %ini3 = "some_def"() : () -> (vector<128xf32>)
    %3:3 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini1, %arg5 = %ini2, %arg6 = %ini3) -> (vector<256xf32>, vector<128xf32>, vector<128xf32>) {
      %acc1 = "some_def_1"(%arg4) : (vector<256xf32>) -> (vector<256xf32>)
      %acc2 = "some_def_2"(%arg5) : (vector<128xf32>) -> (vector<128xf32>)
      %acc3 = "some_def_3"(%arg6) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc1, %acc2, %acc3 : vector<256xf32>, vector<128xf32>, vector<128xf32>
    }
    gpu.yield %3#2, %3#1, %3#0 : vector<128xf32>, vector<128xf32>, vector<256xf32>
  }
  "some_use_1"(%0#0) : (vector<4xf32>) -> ()
  "some_use_2"(%0#1) : (vector<4xf32>) -> ()
  "some_use_3"(%0#2) : (vector<8xf32>) -> ()
  return
}

// -----

// CHECK-PROP-LABEL: func @vector_reduction(
//  CHECK-PROP-SAME:     %[[laneid:.*]]: index)
//   CHECK-PROP-DAG:   %[[c1:.*]] = arith.constant 1 : i32
//   CHECK-PROP-DAG:   %[[c2:.*]] = arith.constant 2 : i32
//   CHECK-PROP-DAG:   %[[c4:.*]] = arith.constant 4 : i32
//   CHECK-PROP-DAG:   %[[c8:.*]] = arith.constant 8 : i32
//   CHECK-PROP-DAG:   %[[c16:.*]] = arith.constant 16 : i32
//   CHECK-PROP-DAG:   %[[c32:.*]] = arith.constant 32 : i32
//       CHECK-PROP:   %[[warp_op:.*]] = gpu.warp_execute_on_lane_0(%[[laneid]])[32] -> (vector<1xf32>) {
//       CHECK-PROP:     gpu.yield %{{.*}} : vector<32xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[a:.*]] = vector.extract %[[warp_op]][0] : f32 from vector<1xf32>
//       CHECK-PROP:   %[[r0:.*]], %{{.*}} = gpu.shuffle  xor %[[a]], %[[c1]], %[[c32]]
//       CHECK-PROP:   %[[a0:.*]] = arith.addf %[[a]], %[[r0]]
//       CHECK-PROP:   %[[r1:.*]], %{{.*}} = gpu.shuffle  xor %[[a0]], %[[c2]], %[[c32]]
//       CHECK-PROP:   %[[a1:.*]] = arith.addf %[[a0]], %[[r1]]
//       CHECK-PROP:   %[[r2:.*]], %{{.*}} = gpu.shuffle  xor %[[a1]], %[[c4]], %[[c32]]
//       CHECK-PROP:   %[[a2:.*]] = arith.addf %[[a1]], %[[r2]]
//       CHECK-PROP:   %[[r3:.*]], %{{.*}} = gpu.shuffle  xor %[[a2]], %[[c8]], %[[c32]]
//       CHECK-PROP:   %[[a3:.*]] = arith.addf %[[a2]], %[[r3]]
//       CHECK-PROP:   %[[r4:.*]], %{{.*}} = gpu.shuffle  xor %[[a3]], %[[c16]], %[[c32]]
//       CHECK-PROP:   %[[a4:.*]] = arith.addf %[[a3]], %[[r4]]
//       CHECK-PROP:   return %[[a4]] : f32
func.func @vector_reduction(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<32xf32>)
    %1 = vector.reduction <add>, %0 : vector<32xf32> into f32
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL: func @warp_distribute(
//  CHECK-PROP-SAME:    %[[ID:[a-zA-Z0-9]+]]
//  CHECK-PROP-SAME:    %[[SRC:[a-zA-Z0-9]+]]
//  CHECK-PROP-SAME:    %[[DEST:[a-zA-Z0-9]+]]
//       CHECK-PROP:    gpu.warp_execute_on_lane_0(%[[ID]])[32]
//  CHECK-PROP-NEXT:      "some_def"() : () -> vector<4096xf32>
//  CHECK-PROP-NEXT:      %{{.*}} = vector.reduction
//       CHECK-PROP:      %[[DEF:.*]] = arith.divf %{{.*}}, %{{.*}} : vector<1xf32>
//   CHECK-PROP-NOT:      gpu.warp_execute_on_lane_0
//       CHECK-PROP:      scf.for
//       CHECK-PROP:        %{{.*}} = arith.subf %{{.*}}, %[[DEF]] : vector<1xf32>
func.func @warp_distribute(%arg0: index, %src: memref<128xf32>, %dest: memref<128xf32>){
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %f0 = arith.constant 0.000000e+00 : f32
  gpu.warp_execute_on_lane_0(%arg0)[32]{
    %cst_1 = arith.constant dense<2.621440e+05> : vector<1xf32>
    %0 = "some_def"() : () -> (vector<4096xf32>)
    %1 = vector.reduction <add>, %0, %cst : vector<4096xf32> into f32
    %2 = vector.broadcast %1 : f32 to vector<1xf32>
    %3 = arith.divf %2, %cst_1 : vector<1xf32>
    scf.for %arg1 = %c0 to %c128 step %c1 {
        %4 = vector.transfer_read %src[%arg1], %f0 {in_bounds = [true]} : memref<128xf32>, vector<1xf32>
        %5 = arith.subf %4, %3 : vector<1xf32>
        vector.transfer_write %5, %dest[%arg1] : vector<1xf32>, memref<128xf32>
    }
  }
  return
}

// -----

func.func @vector_reduction(%laneid: index, %m0: memref<4x2x32xf32>, %m1: memref<f32>) {
  %c0 = arith.constant 0: index
  %f0 = arith.constant 0.0: f32
  //     CHECK-D: %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<f32>) {
  //     CHECK-D: gpu.warp_execute_on_lane_0(%{{.*}})[32] {
  //     CHECK-D:   vector.transfer_write %[[R]], %{{.*}}[] : vector<f32>, memref<f32>
  gpu.warp_execute_on_lane_0(%laneid)[32] {
    %0 = vector.transfer_read %m0[%c0, %c0, %c0], %f0 {in_bounds = [true]} : memref<4x2x32xf32>, vector<32xf32>
    %1 = vector.transfer_read %m1[], %f0 : memref<f32>, vector<f32>
    %2 = vector.extract %1[] : f32 from vector<f32>
    %3 = vector.reduction <add>, %0 : vector<32xf32> into f32
    %4 = arith.addf %3, %2 : f32
    %5 = vector.broadcast %4 : f32 to vector<f32>
    vector.transfer_write %5, %m1[] : vector<f32>, memref<f32>
  }
  return
}

// -----

// CHECK-PROP-LABEL: func @vector_reduction_large(
//  CHECK-PROP-SAME:     %[[laneid:.*]]: index)
//   CHECK-PROP-DAG:   %[[c1:.*]] = arith.constant 1 : i32
//   CHECK-PROP-DAG:   %[[c2:.*]] = arith.constant 2 : i32
//   CHECK-PROP-DAG:   %[[c4:.*]] = arith.constant 4 : i32
//   CHECK-PROP-DAG:   %[[c8:.*]] = arith.constant 8 : i32
//   CHECK-PROP-DAG:   %[[c16:.*]] = arith.constant 16 : i32
//   CHECK-PROP-DAG:   %[[c32:.*]] = arith.constant 32 : i32
//       CHECK-PROP:   %[[warp_op:.*]] = gpu.warp_execute_on_lane_0(%[[laneid]])[32] -> (vector<2xf32>) {
//       CHECK-PROP:     gpu.yield %{{.*}} : vector<64xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[a:.*]] = vector.reduction <add>, %[[warp_op]] : vector<2xf32> into f32
//       CHECK-PROP:   %[[r0:.*]], %{{.*}} = gpu.shuffle  xor %[[a]], %[[c1]], %[[c32]]
//       CHECK-PROP:   %[[a0:.*]] = arith.addf %[[a]], %[[r0]]
//       CHECK-PROP:   %[[r1:.*]], %{{.*}} = gpu.shuffle  xor %[[a0]], %[[c2]], %[[c32]]
//       CHECK-PROP:   %[[a1:.*]] = arith.addf %[[a0]], %[[r1]]
//       CHECK-PROP:   %[[r2:.*]], %{{.*}} = gpu.shuffle  xor %[[a1]], %[[c4]], %[[c32]]
//       CHECK-PROP:   %[[a2:.*]] = arith.addf %[[a1]], %[[r2]]
//       CHECK-PROP:   %[[r3:.*]], %{{.*}} = gpu.shuffle  xor %[[a2]], %[[c8]], %[[c32]]
//       CHECK-PROP:   %[[a3:.*]] = arith.addf %[[a2]], %[[r3]]
//       CHECK-PROP:   %[[r4:.*]], %{{.*}} = gpu.shuffle  xor %[[a3]], %[[c16]], %[[c32]]
//       CHECK-PROP:   %[[a4:.*]] = arith.addf %[[a3]], %[[r4]]
//       CHECK-PROP:   return %[[a4]] : f32
func.func @vector_reduction_large(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<64xf32>)
    %1 = vector.reduction <add>, %0 : vector<64xf32> into f32
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL: func @vector_reduction_acc(
//  CHECK-PROP-SAME:     %[[laneid:.*]]: index)
//   CHECK-PROP-DAG:   %[[c1:.*]] = arith.constant 1 : i32
//   CHECK-PROP-DAG:   %[[c2:.*]] = arith.constant 2 : i32
//   CHECK-PROP-DAG:   %[[c4:.*]] = arith.constant 4 : i32
//   CHECK-PROP-DAG:   %[[c8:.*]] = arith.constant 8 : i32
//   CHECK-PROP-DAG:   %[[c16:.*]] = arith.constant 16 : i32
//   CHECK-PROP-DAG:   %[[c32:.*]] = arith.constant 32 : i32
//       CHECK-PROP:   %[[warp_op:.*]]:2 = gpu.warp_execute_on_lane_0(%[[laneid]])[32] -> (vector<2xf32>, f32) {
//       CHECK-PROP:     gpu.yield %{{.*}}, %{{.*}} : vector<64xf32>, f32
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[a:.*]] = vector.reduction <add>, %[[warp_op]]#0 : vector<2xf32> into f32
//       CHECK-PROP:   %[[r0:.*]], %{{.*}} = gpu.shuffle  xor %[[a]], %[[c1]], %[[c32]]
//       CHECK-PROP:   %[[a0:.*]] = arith.addf %[[a]], %[[r0]]
//       CHECK-PROP:   %[[r1:.*]], %{{.*}} = gpu.shuffle  xor %[[a0]], %[[c2]], %[[c32]]
//       CHECK-PROP:   %[[a1:.*]] = arith.addf %[[a0]], %[[r1]]
//       CHECK-PROP:   %[[r2:.*]], %{{.*}} = gpu.shuffle  xor %[[a1]], %[[c4]], %[[c32]]
//       CHECK-PROP:   %[[a2:.*]] = arith.addf %[[a1]], %[[r2]]
//       CHECK-PROP:   %[[r3:.*]], %{{.*}} = gpu.shuffle  xor %[[a2]], %[[c8]], %[[c32]]
//       CHECK-PROP:   %[[a3:.*]] = arith.addf %[[a2]], %[[r3]]
//       CHECK-PROP:   %[[r4:.*]], %{{.*}} = gpu.shuffle  xor %[[a3]], %[[c16]], %[[c32]]
//       CHECK-PROP:   %[[a4:.*]] = arith.addf %[[a3]], %[[r4]]
//       CHECK-PROP:   %[[a5:.*]] = arith.addf %[[a4]], %[[warp_op]]#1
//       CHECK-PROP:   return %[[a5]] : f32
func.func @vector_reduction_acc(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<64xf32>)
    %1 = "some_def"() : () -> (f32)
    %2 = vector.reduction <add>, %0, %1 : vector<64xf32> into f32
    gpu.yield %2 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL:   func @warp_duplicate_yield(
func.func @warp_duplicate_yield(%laneid: index) -> (vector<1xf32>, vector<1xf32>) {
  //   CHECK-PROP: %{{.*}}:2 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>, vector<1xf32>)
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>, vector<1xf32>) {
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = arith.addf %2, %3 : vector<32xf32>
    %5 = arith.addf %2, %2 : vector<32xf32>
// CHECK-PROP-NOT:   arith.addf
//     CHECK-PROP:   gpu.yield %{{.*}}, %{{.*}} : vector<32xf32>, vector<32xf32>
    gpu.yield %4, %5 : vector<32xf32>, vector<32xf32>
  }
  return %r#0, %r#1 : vector<1xf32>, vector<1xf32>
}

// -----

// CHECK-PROP-LABEL: func @warp_constant(
//       CHECK-PROP:   %[[C:.*]] = arith.constant dense<2.000000e+00> : vector<1xf32>
//       CHECK-PROP:   return %[[C]] : vector<1xf32>
func.func @warp_constant(%laneid: index) -> (vector<1xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
    %cst = arith.constant dense<2.0> : vector<32xf32>
    gpu.yield %cst : vector<32xf32>
  }
  return %r : vector<1xf32>
}

// -----

// TODO: We could use warp shuffles instead of broadcasting the entire vector.

// CHECK-PROP-LABEL: func.func @vector_extract_1d(
//   CHECK-PROP-DAG:   %[[C5_I32:.*]] = arith.constant 5 : i32
//       CHECK-PROP:   %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<2xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"() : () -> vector<64xf32>
//       CHECK-PROP:     gpu.yield %[[V]] : vector<64xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[R]][1] : f32 from vector<2xf32>
//       CHECK-PROP:   %[[SHUFFLED:.*]], %{{.*}} = gpu.shuffle  idx %[[E]], %[[C5_I32]]
//       CHECK-PROP:   return %[[SHUFFLED]] : f32
func.func @vector_extract_1d(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<64xf32>)
    %1 = vector.extract %0[9] : f32 from vector<64xf32>
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_2d(
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<5x3xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[V]] : vector<5x96xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[W]][2] : vector<3xf32> from vector<5x3xf32>
//       CHECK-PROP:   return %[[E]]
func.func @vector_extract_2d(%laneid: index) -> (vector<3xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<3xf32>) {
    %0 = "some_def"() : () -> (vector<5x96xf32>)
    %1 = vector.extract %0[2] : vector<96xf32> from vector<5x96xf32>
    gpu.yield %1 : vector<96xf32>
  }
  return %r : vector<3xf32>
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_2d_broadcast_scalar(
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<5x96xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[V]] : vector<5x96xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[W]][1, 2] : f32 from vector<5x96xf32>
//       CHECK-PROP:   return %[[E]]
func.func @vector_extract_2d_broadcast_scalar(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<5x96xf32>)
    %1 = vector.extract %0[1, 2] : f32 from vector<5x96xf32>
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_2d_broadcast(
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<5x96xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[V]] : vector<5x96xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[W]][2] : vector<96xf32> from vector<5x96xf32>
//       CHECK-PROP:   return %[[E]]
func.func @vector_extract_2d_broadcast(%laneid: index) -> (vector<96xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<96xf32>) {
    %0 = "some_def"() : () -> (vector<5x96xf32>)
    %1 = vector.extract %0[2] : vector<96xf32> from vector<5x96xf32>
    gpu.yield %1 : vector<96xf32>
  }
  return %r : vector<96xf32>
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_3d(
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<8x4x96xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[V]] : vector<8x128x96xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[W]][2] : vector<4x96xf32> from vector<8x4x96xf32>
//       CHECK-PROP:   return %[[E]]
func.func @vector_extract_3d(%laneid: index) -> (vector<4x96xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x96xf32>) {
    %0 = "some_def"() : () -> (vector<8x128x96xf32>)
    %1 = vector.extract %0[2] : vector<128x96xf32> from vector<8x128x96xf32>
    gpu.yield %1 : vector<128x96xf32>
  }
  return %r : vector<4x96xf32>
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_0d(
//       CHECK-PROP:   %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<f32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"() : () -> vector<f32>
//       CHECK-PROP:     gpu.yield %[[V]] : vector<f32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[R]][] : f32 from vector<f32>
//       CHECK-PROP:   return %[[E]] : f32
func.func @vector_extract_0d(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<f32>)
    %1 = vector.extract %0[] : f32 from vector<f32>
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// CHECK-PROP-LABEL: func.func @vector_extract_1element(
//       CHECK-PROP:   %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"() : () -> vector<1xf32>
//       CHECK-PROP:     gpu.yield %[[V]] : vector<1xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[E:.*]] = vector.extract %[[R]][0] : f32 from vector<1xf32>
//       CHECK-PROP:   return %[[E]] : f32
func.func @vector_extract_1element(%laneid: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %c0 = arith.constant 0 : index
    %1 = vector.extract %0[%c0] : f32 from vector<1xf32>
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

//       CHECK-PROP: #[[$map:.*]] = affine_map<()[s0] -> (s0 ceildiv 3)>
//       CHECK-PROP: #[[$map1:.*]] = affine_map<()[s0] -> (s0 mod 3)>
// CHECK-PROP-LABEL: func.func @vector_extract_1d(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index, %[[POS:.*]]: index
//   CHECK-PROP-DAG:   %[[C32:.*]] = arith.constant 32 : i32
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<3xf32>) {
//       CHECK-PROP:     %[[V:.*]] = "some_def"
//       CHECK-PROP:     gpu.yield %[[V]] : vector<96xf32>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[FROM_LANE:.*]] = affine.apply #[[$map]]()[%[[POS]]]
//       CHECK-PROP:   %[[DISTR_POS:.*]] = affine.apply #[[$map1]]()[%[[POS]]]
//       CHECK-PROP:   %[[EXTRACTED:.*]] = vector.extract %[[W]][%[[DISTR_POS]]] : f32 from vector<3xf32>
//       CHECK-PROP:   %[[FROM_LANE_I32:.*]] = arith.index_cast %[[FROM_LANE]] : index to i32
//       CHECK-PROP:   %[[SHUFFLED:.*]], %{{.*}} = gpu.shuffle  idx %[[EXTRACTED]], %[[FROM_LANE_I32]], %[[C32]] : f32
//       CHECK-PROP:   return %[[SHUFFLED]]
func.func @vector_extract_1d(%laneid: index, %pos: index) -> (f32) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (vector<96xf32>)
    %1 = vector.extract %0[%pos] : f32 from vector<96xf32>
    gpu.yield %1 : f32
  }
  return %r : f32
}

// -----

// Index-typed values cannot be shuffled at the moment.

// CHECK-PROP-LABEL: func.func @vector_extract_1d_index(
//       CHECK-PROP:   gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (index) {
//       CHECK-PROP:     "some_def"
//       CHECK-PROP:     vector.extract
//       CHECK-PROP:     gpu.yield {{.*}} : index
//       CHECK-PROP:   }
func.func @vector_extract_1d_index(%laneid: index, %pos: index) -> (index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (index) {
    %0 = "some_def"() : () -> (vector<96xindex>)
    %1 = vector.extract %0[%pos] : index from vector<96xindex>
    gpu.yield %1 : index
  }
  return %r : index
}

// -----

// CHECK-PROP:   func @lane_dependent_warp_propagate_read
//  CHECK-PROP-SAME:   %[[ID:.*]]: index
func.func @lane_dependent_warp_propagate_read(
    %laneid: index, %src: memref<1x1024xf32>, %dest: memref<1x1024xf32>) {
  // CHECK-PROP-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-PROP-NOT: gpu.warp_execute_on_lane_0
  // CHECK-PROP-DAG: %[[R0:.*]] = vector.transfer_read %arg1[%[[C0]], %[[ID]]], %{{.*}} : memref<1x1024xf32>, vector<1x1xf32>
  // CHECK-PROP: vector.transfer_write %[[R0]], {{.*}} : vector<1x1xf32>, memref<1x1024xf32>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1x1xf32>) {
    %2 = vector.transfer_read %src[%c0, %c0], %cst : memref<1x1024xf32>, vector<1x32xf32>
    gpu.yield %2 : vector<1x32xf32>
  }
  vector.transfer_write %r, %dest[%c0, %laneid] : vector<1x1xf32>, memref<1x1024xf32>
  return
}

// -----

func.func @warp_propagate_read_3d(%laneid: index, %src: memref<32x4x32xf32>) -> vector<1x1x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[1024] -> (vector<1x1x4xf32>) {
    %2 = vector.transfer_read %src[%c0, %c0, %c0], %cst : memref<32x4x32xf32>, vector<32x4x32xf32>
    gpu.yield %2 : vector<32x4x32xf32>
  }
  return %r : vector<1x1x4xf32>
}

//   CHECK-PROP-DAG: #[[$ID0MAP:.+]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 8) * 32)>
//   CHECK-PROP-DAG: #[[$ID1MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 8) mod 4)>
//   CHECK-PROP-DAG: #[[$ID2MAP:.+]] = affine_map<()[s0] -> ((s0 floordiv 8) floordiv 32)>
// CHECK-PROP-LABEL: func.func @warp_propagate_read_3d
//  CHECK-PROP-SAME: (%[[LANE:.+]]: index, %[[SRC:.+]]: memref<32x4x32xf32>)
//   CHECK-PROP-DAG: %[[ID0:.+]] = affine.apply #[[$ID0MAP]]()[%[[LANE]]]
//   CHECK-PROP-DAG: %[[ID1:.+]] = affine.apply #[[$ID1MAP]]()[%[[LANE]]]
//   CHECK-PROP-DAG: %[[ID2:.+]] = affine.apply #[[$ID2MAP]]()[%[[LANE]]]
//       CHECK-PROP: %[[READ:.+]] = vector.transfer_read %[[SRC]][%[[ID2]], %[[ID1]], %[[ID0]]], %{{.+}} : memref<32x4x32xf32>, vector<1x1x4xf32>
//       CHECK-PROP: return %[[READ]] : vector<1x1x4xf32>

// -----

func.func @warp_propagate_read_broadcast(%laneid: index, %src: memref<32x1xf32>) -> vector<1x4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[512] -> (vector<1x4xf32>) {
    %2 = vector.transfer_read %src[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d0, 0)>} : memref<32x1xf32>, vector<32x64xf32>
    gpu.yield %2 : vector<32x64xf32>
  }
  return %r : vector<1x4xf32>
}

//   CHECK-PROP-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 16)>
//   CHECK-PROP-DAG: #[[$READMAP:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-PROP-LABEL: func.func @warp_propagate_read_broadcast
//  CHECK-PROP-SAME: (%[[LANE:.+]]: index, %[[SRC:.+]]: memref<32x1xf32>)
//       CHECK-PROP:  %[[C0:.+]] = arith.constant 0 : index
//       CHECK-PROP:  %[[ID:.+]] = affine.apply #[[$MAP]]()[%[[LANE]]]
//       CHECK-PROP:  %[[READ:.+]] = vector.transfer_read %[[SRC]][%[[ID]], %[[C0]]], %{{.+}} {in_bounds = [true, true], permutation_map = #[[$READMAP]]} : memref<32x1xf32>, vector<1x4xf32>
//       CHECK-PROP:  return %[[READ]] : vector<1x4xf32>

// -----

// CHECK-PROP:   func @dont_duplicate_read
func.func @dont_duplicate_read(
  %laneid: index, %src: memref<1024xf32>) -> vector<1xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
//       CHECK-PROP:   gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
//  CHECK-PROP-NEXT:     vector.transfer_read
//  CHECK-PROP-NEXT:     "blocking_use"
//  CHECK-PROP-NEXT:     gpu.yield
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<32xf32>
    "blocking_use"(%2) : (vector<32xf32>) -> ()
    gpu.yield %2 : vector<32xf32>
  }
  return %r : vector<1xf32>
}

// -----

// CHECK-PROP:   func @dedup
func.func @dedup(%laneid: index, %v0: vector<4xf32>, %v1: vector<4xf32>)
    -> (vector<1xf32>, vector<1xf32>) {

  // CHECK-PROP: %[[SINGLE_RES:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>) {
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32]
      args(%v0, %v1 : vector<4xf32>, vector<4xf32>) -> (vector<1xf32>, vector<1xf32>) {
    ^bb0(%arg0: vector<128xf32>, %arg1: vector<128xf32>):

    // CHECK-PROP: %[[SINGLE_VAL:.*]] = "some_def"(%{{.*}}) : (vector<128xf32>) -> vector<32xf32>
    %2 = "some_def"(%arg0) : (vector<128xf32>) -> vector<32xf32>

    // CHECK-PROP: gpu.yield %[[SINGLE_VAL]] : vector<32xf32>
    gpu.yield %2, %2 : vector<32xf32>, vector<32xf32>
  }

  // CHECK-PROP: return %[[SINGLE_RES]], %[[SINGLE_RES]] : vector<1xf32>, vector<1xf32>
  return %r#0, %r#1 : vector<1xf32>, vector<1xf32>
}

// -----

// CHECK-SCF-IF:   func @warp_execute_has_broadcast_semantics
func.func @warp_execute_has_broadcast_semantics(%laneid: index, %s0: f32, %v0: vector<f32>, %v1: vector<1xf32>, %v2: vector<1x1xf32>)
    -> (f32, vector<f32>, vector<1xf32>, vector<1x1xf32>) {
  // CHECK-SCF-IF-DAG: %[[C0:.*]] = arith.constant 0 : index

  // CHECK-SCF-IF: scf.if{{.*}}{
  %r:4 = gpu.warp_execute_on_lane_0(%laneid)[32]
      args(%s0, %v0, %v1, %v2 : f32, vector<f32>, vector<1xf32>, vector<1x1xf32>) -> (f32, vector<f32>, vector<1xf32>, vector<1x1xf32>) {
    ^bb0(%bs0: f32, %bv0: vector<f32>, %bv1: vector<1xf32>, %bv2: vector<1x1xf32>):

      // CHECK-SCF-IF: vector.transfer_read {{.*}}[%[[C0]], %[[C0]]]{{.*}} {in_bounds = [true, true]} : memref<1x1xf32, 3>, vector<1x1xf32>
      // CHECK-SCF-IF: vector.transfer_read {{.*}}[%[[C0]]]{{.*}} {in_bounds = [true]} : memref<1xf32, 3>, vector<1xf32>
      // CHECK-SCF-IF: vector.transfer_read {{.*}}[]{{.*}} : memref<f32, 3>, vector<f32>
      // CHECK-SCF-IF: memref.load {{.*}}[%[[C0]]] : memref<1xf32, 3>
      // CHECK-SCF-IF: "some_def_0"(%{{.*}}) : (f32) -> f32
      // CHECK-SCF-IF: "some_def_1"(%{{.*}}) : (vector<f32>) -> vector<f32>
      // CHECK-SCF-IF: "some_def_1"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
      // CHECK-SCF-IF: "some_def_1"(%{{.*}}) : (vector<1x1xf32>) -> vector<1x1xf32>
      // CHECK-SCF-IF: memref.store {{.*}}[%[[C0]]] : memref<1xf32, 3>
      // CHECK-SCF-IF: vector.transfer_write {{.*}}[] : vector<f32>, memref<f32, 3>
      // CHECK-SCF-IF: vector.transfer_write {{.*}}[%[[C0]]] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, 3>
      // CHECK-SCF-IF: vector.transfer_write {{.*}}[%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<1x1xf32>, memref<1x1xf32, 3>

      %rs0 = "some_def_0"(%bs0) : (f32) -> f32
      %rv0 = "some_def_1"(%bv0) : (vector<f32>) -> vector<f32>
      %rv1 = "some_def_1"(%bv1) : (vector<1xf32>) -> vector<1xf32>
      %rv2 = "some_def_1"(%bv2) : (vector<1x1xf32>) -> vector<1x1xf32>

      // CHECK-SCF-IF-NOT: gpu.yield
      gpu.yield %rs0, %rv0, %rv1, %rv2 : f32, vector<f32>, vector<1xf32>, vector<1x1xf32>
  }

  // CHECK-SCF-IF: gpu.barrier
  // CHECK-SCF-IF: %[[RV2:.*]] = vector.transfer_read {{.*}}[%[[C0]], %[[C0]]]{{.*}} {in_bounds = [true, true]} : memref<1x1xf32, 3>, vector<1x1xf32>
  // CHECK-SCF-IF: %[[RV1:.*]] = vector.transfer_read {{.*}}[%[[C0]]]{{.*}} {in_bounds = [true]} : memref<1xf32, 3>, vector<1xf32>
  // CHECK-SCF-IF: %[[RV0:.*]] = vector.transfer_read {{.*}}[]{{.*}} : memref<f32, 3>, vector<f32>
  // CHECK-SCF-IF: %[[RS0:.*]] = memref.load {{.*}}[%[[C0]]] : memref<1xf32, 3>
  // CHECK-SCF-IF: return %[[RS0]], %[[RV0]], %[[RV1]], %[[RV2]] : f32, vector<f32>, vector<1xf32>, vector<1x1xf32>
  return %r#0, %r#1, %r#2, %r#3 : f32, vector<f32>, vector<1xf32>, vector<1x1xf32>
}

// -----

// CHECK-SCF-IF-DAG: #[[$TIMES2:.*]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK-SCF-IF:   func @warp_execute_nd_distribute
// CHECK-SCF-IF-SAME: (%[[LANEID:.*]]: index
func.func @warp_execute_nd_distribute(%laneid: index, %v0: vector<1x64x1xf32>, %v1: vector<1x2x128xf32>)
    -> (vector<1x64x1xf32>, vector<1x2x128xf32>) {
  // CHECK-SCF-IF-DAG: %[[C0:.*]] = arith.constant 0 : index

  // CHECK-SCF-IF:  vector.transfer_write %{{.*}}, %{{.*}}[%[[LANEID]], %c0, %c0] {in_bounds = [true, true, true]} : vector<1x64x1xf32>, memref<32x64x1xf32, 3>
  // CHECK-SCF-IF:  %[[RID:.*]] = affine.apply #[[$TIMES2]]()[%[[LANEID]]]
  // CHECK-SCF-IF:  vector.transfer_write %{{.*}}, %{{.*}}[%[[C0]], %[[RID]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x2x128xf32>, memref<1x64x128xf32, 3>
  // CHECK-SCF-IF:  gpu.barrier

  // CHECK-SCF-IF: scf.if{{.*}}{
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32]
      args(%v0, %v1 : vector<1x64x1xf32>, vector<1x2x128xf32>) -> (vector<1x64x1xf32>, vector<1x2x128xf32>) {
    ^bb0(%arg0: vector<32x64x1xf32>, %arg1: vector<1x64x128xf32>):

  // CHECK-SCF-IF-DAG: %[[SR0:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : memref<32x64x1xf32, 3>, vector<32x64x1xf32>
  // CHECK-SCF-IF-DAG: %[[SR1:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : memref<1x64x128xf32, 3>, vector<1x64x128xf32>
  //     CHECK-SCF-IF: %[[W0:.*]] = "some_def_0"(%[[SR0]]) : (vector<32x64x1xf32>) -> vector<32x64x1xf32>
  //     CHECK-SCF-IF: %[[W1:.*]] = "some_def_1"(%[[SR1]]) : (vector<1x64x128xf32>) -> vector<1x64x128xf32>
  // CHECK-SCF-IF-DAG: vector.transfer_write %[[W0]], %{{.*}}[%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<32x64x1xf32>, memref<32x64x1xf32, 3>
  // CHECK-SCF-IF-DAG: vector.transfer_write %[[W1]], %{{.*}}[%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x64x128xf32>, memref<1x64x128xf32, 3>

      %r0 = "some_def_0"(%arg0) : (vector<32x64x1xf32>) -> vector<32x64x1xf32>
      %r1 = "some_def_1"(%arg1) : (vector<1x64x128xf32>) -> vector<1x64x128xf32>

      // CHECK-SCF-IF-NOT: gpu.yield
      gpu.yield %r0, %r1 : vector<32x64x1xf32>, vector<1x64x128xf32>
  }

  //     CHECK-SCF-IF: gpu.barrier
  //     CHECK-SCF-IF: %[[WID:.*]] = affine.apply #[[$TIMES2]]()[%[[LANEID]]]
  // CHECK-SCF-IF-DAG: %[[R0:.*]] = vector.transfer_read %{{.*}}[%[[LANEID]], %[[C0]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : memref<32x64x1xf32, 3>, vector<1x64x1xf32>
  // CHECK-SCF-IF-DAG: %[[R1:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[WID]], %[[C0]]], %{{.*}} {in_bounds = [true, true, true]} : memref<1x64x128xf32, 3>, vector<1x2x128xf32>
  //     CHECK-SCF-IF: return %[[R0]], %[[R1]] : vector<1x64x1xf32>, vector<1x2x128xf32>
  return %r#0, %r#1 : vector<1x64x1xf32>, vector<1x2x128xf32>
}

// -----

//       CHECK-PROP:   #[[$MAP:.*]] = affine_map<()[s0] -> (s0 ceildiv 3)>
//       CHECK-PROP:   #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 mod 3)>
// CHECK-PROP-LABEL: func @vector_insert_1d(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index, %[[POS:.*]]: index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<3xf32>, f32)
//       CHECK-PROP:   %[[INSERTING_LANE:.*]] = affine.apply #[[$MAP]]()[%[[POS]]]
//       CHECK-PROP:   %[[INSERTING_POS:.*]] = affine.apply #[[$MAP1]]()[%[[POS]]]
//       CHECK-PROP:   %[[SHOULD_INSERT:.*]] = arith.cmpi eq, %[[LANEID]], %[[INSERTING_LANE]] : index
//       CHECK-PROP:   %[[R:.*]] = scf.if %[[SHOULD_INSERT]] -> (vector<3xf32>) {
//       CHECK-PROP:     %[[INSERT:.*]] = vector.insert %[[W]]#1, %[[W]]#0 [%[[INSERTING_POS]]]
//       CHECK-PROP:     scf.yield %[[INSERT]]
//       CHECK-PROP:   } else {
//       CHECK-PROP:     scf.yield %[[W]]#0
//       CHECK-PROP:   }
//       CHECK-PROP:   return %[[R]]
func.func @vector_insert_1d(%laneid: index, %pos: index) -> (vector<3xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<3xf32>) {
    %0 = "some_def"() : () -> (vector<96xf32>)
    %f = "another_def"() : () -> (f32)
    %1 = vector.insert %f, %0[%pos] : f32 into vector<96xf32>
    gpu.yield %1 : vector<96xf32>
  }
  return %r : vector<3xf32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_1d_broadcast(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index, %[[POS:.*]]: index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<96xf32>, f32)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VEC]], %[[VAL]]
//       CHECK-PROP:   vector.insert %[[W]]#1, %[[W]]#0 [%[[POS]]] : f32 into vector<96xf32>
func.func @vector_insert_1d_broadcast(%laneid: index, %pos: index) -> (vector<96xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<96xf32>) {
    %0 = "some_def"() : () -> (vector<96xf32>)
    %f = "another_def"() : () -> (f32)
    %1 = vector.insert %f, %0[%pos] : f32 into vector<96xf32>
    gpu.yield %1 : vector<96xf32>
  }
  return %r : vector<96xf32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_0d(
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<f32>, f32)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VEC]], %[[VAL]]
//       CHECK-PROP:   vector.broadcast %[[W]]#1 : f32 to vector<f32>
func.func @vector_insert_0d(%laneid: index) -> (vector<f32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<f32>) {
    %0 = "some_def"() : () -> (vector<f32>)
    %f = "another_def"() : () -> (f32)
    %1 = vector.insert %f, %0[] : f32 into vector<f32>
    gpu.yield %1 : vector<f32>
  }
  return %r : vector<f32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_1d(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index
//   CHECK-PROP-DAG:   %[[C26:.*]] = arith.constant 26 : index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<3xf32>, f32)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VEC]], %[[VAL]]
//       CHECK-PROP:   %[[SHOULD_INSERT:.*]] = arith.cmpi eq, %[[LANEID]], %[[C26]]
//       CHECK-PROP:   %[[R:.*]] = scf.if %[[SHOULD_INSERT]] -> (vector<3xf32>) {
//       CHECK-PROP:     %[[INSERT:.*]] = vector.insert %[[W]]#1, %[[W]]#0 [1]
//       CHECK-PROP:     scf.yield %[[INSERT]]
//       CHECK-PROP:   } else {
//       CHECK-PROP:     scf.yield %[[W]]#0
//       CHECK-PROP:   }
//       CHECK-PROP:   return %[[R]]
func.func @vector_insert_1d(%laneid: index) -> (vector<3xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<3xf32>) {
    %0 = "some_def"() : () -> (vector<96xf32>)
    %f = "another_def"() : () -> (f32)
    %1 = vector.insert %f, %0[76] : f32 into vector<96xf32>
    gpu.yield %1 : vector<96xf32>
  }
  return %r : vector<3xf32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_2d_distr_src(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<3xf32>, vector<4x3xf32>)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VAL]], %[[VEC]]
//       CHECK-PROP:   %[[INSERT:.*]] = vector.insert %[[W]]#0, %[[W]]#1 [2] : vector<3xf32> into vector<4x3xf32>
//       CHECK-PROP:   return %[[INSERT]]
func.func @vector_insert_2d_distr_src(%laneid: index) -> (vector<4x3xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x3xf32>) {
    %0 = "some_def"() : () -> (vector<4x96xf32>)
    %s = "another_def"() : () -> (vector<96xf32>)
    %1 = vector.insert %s, %0[2] : vector<96xf32> into vector<4x96xf32>
    gpu.yield %1 : vector<4x96xf32>
  }
  return %r : vector<4x3xf32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_2d_distr_pos(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index
//       CHECK-PROP:   %[[C19:.*]] = arith.constant 19 : index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<96xf32>, vector<4x96xf32>)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VAL]], %[[VEC]]
//       CHECK-PROP:   %[[SHOULD_INSERT:.*]] = arith.cmpi eq, %[[LANEID]], %[[C19]]
//       CHECK-PROP:   %[[R:.*]] = scf.if %[[SHOULD_INSERT]] -> (vector<4x96xf32>) {
//       CHECK-PROP:     %[[INSERT:.*]] = vector.insert %[[W]]#0, %[[W]]#1 [3] : vector<96xf32> into vector<4x96xf32>
//       CHECK-PROP:     scf.yield %[[INSERT]]
//       CHECK-PROP:   } else {
//       CHECK-PROP:     scf.yield %[[W]]#1
//       CHECK-PROP:   }
//       CHECK-PROP:   return %[[R]]
func.func @vector_insert_2d_distr_pos(%laneid: index) -> (vector<4x96xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x96xf32>) {
    %0 = "some_def"() : () -> (vector<128x96xf32>)
    %s = "another_def"() : () -> (vector<96xf32>)
    %1 = vector.insert %s, %0[79] : vector<96xf32> into vector<128x96xf32>
    gpu.yield %1 : vector<128x96xf32>
  }
  return %r : vector<4x96xf32>
}

// -----

// CHECK-PROP-LABEL: func @vector_insert_2d_broadcast(
//  CHECK-PROP-SAME:     %[[LANEID:.*]]: index
//       CHECK-PROP:   %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<96xf32>, vector<4x96xf32>)
//       CHECK-PROP:     %[[VEC:.*]] = "some_def"
//       CHECK-PROP:     %[[VAL:.*]] = "another_def"
//       CHECK-PROP:     gpu.yield %[[VAL]], %[[VEC]]
//       CHECK-PROP:   %[[INSERT:.*]] = vector.insert %[[W]]#0, %[[W]]#1 [2] : vector<96xf32> into vector<4x96xf32>
//       CHECK-PROP:   return %[[INSERT]]
func.func @vector_insert_2d_broadcast(%laneid: index) -> (vector<4x96xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<4x96xf32>) {
    %0 = "some_def"() : () -> (vector<4x96xf32>)
    %s = "another_def"() : () -> (vector<96xf32>)
    %1 = vector.insert %s, %0[2] : vector<96xf32> into vector<4x96xf32>
    gpu.yield %1 : vector<4x96xf32>
  }
  return %r : vector<4x96xf32>
}

// -----
// CHECK-PROP-LABEL: func.func @vector_extract_strided_slice_2d_distr_inner(
//  CHECK-RPOP-SAME: %[[LANEID:.*]]: index
//       CHECK-PROP: %[[W:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<64x1xf32>) {
//       CHECK-PROP: %[[VEC:.*]] = "some_def"() : () -> vector<64x32xf32>
//       CHECK-PROP: gpu.yield %[[VEC]] : vector<64x32xf32>
//       CHECK-PROP: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[W]]
//  CHECK-PROP-SAME: {offsets = [8], sizes = [24], strides = [1]} : vector<64x1xf32> to vector<24x1xf32>
//       CHECK-PROP: return %[[EXTRACT]] : vector<24x1xf32>
func.func @vector_extract_strided_slice_2d_distr_inner(%laneid: index) -> (vector<24x1xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<24x1xf32>) {
    %0 = "some_def"() : () -> (vector<64x32xf32>)
    %1 = vector.extract_strided_slice %0 { offsets = [8], sizes = [24], strides = [1]}
      : vector<64x32xf32> to vector<24x32xf32>
    gpu.yield %1 : vector<24x32xf32>
  }
  return %r : vector<24x1xf32>
}

// -----
// CHECK-PROP-LABEL: func.func @vector_extract_strided_slice_2d_distr_outer(
//  CHECK-PROP-SAME: %[[LANEID:.*]]: index
//       CHECK-PROP: %[[W:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<1x64xf32>) {
//       CHECK-PROP: %[[VEC:.*]] = "some_def"() : () -> vector<32x64xf32>
//       CHECK-PROP: gpu.yield %[[VEC]] : vector<32x64xf32>
//       CHECK-PROP: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[W]]
//  CHECK-PROP-SAME: {offsets = [0, 12], sizes = [1, 8], strides = [1, 1]} : vector<1x64xf32> to vector<1x8xf32>
//       CHECK-PROP: return %[[EXTRACT]] : vector<1x8xf32>
func.func @vector_extract_strided_slice_2d_distr_outer(%laneid: index) -> (vector<1x8xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1x8xf32>) {
    %0 = "some_def"() : () -> (vector<32x64xf32>)
    %1 = vector.extract_strided_slice %0 { offsets = [0, 12], sizes = [32, 8], strides = [1, 1]}
      : vector<32x64xf32> to vector<32x8xf32>
    gpu.yield %1 : vector<32x8xf32>
  }
  return %r : vector<1x8xf32>
}

// -----
// CHECK-PROP-LABEL: func.func @vector_insert_strided_slice_1d_to_2d(
//  CHECK-PROP-SAME: %[[LANEID:.*]]: index)
//       CHECK-PROP: %[[W:.*]]:2 = gpu.warp_execute_on_lane_0({{.*}} -> (vector<1xf32>, vector<64x1xf32>) {
//       CHECK-PROP: %[[SRC:.*]] = "some_def"() : () -> vector<32xf32>
//       CHECK-PROP: %[[DEST:.*]] = "some_def"() : () -> vector<64x32xf32>
//       CHECK-PROP: gpu.yield %[[SRC]], %[[DEST]] : vector<32xf32>, vector<64x32xf32>
//       CHECK-PROP: %[[INSERT:.*]] = vector.insert_strided_slice %[[W]]#0, %[[W]]#1
//  CHECK-PROP-SAME: {offsets = [18, 0], strides = [1]} : vector<1xf32> into vector<64x1xf32>
//       CHECK-PROP: return %[[INSERT]] : vector<64x1xf32>
func.func @vector_insert_strided_slice_1d_to_2d(%laneid: index) -> (vector<64x1xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<64x1xf32>) {
    %0 = "some_def"() : () -> (vector<32xf32>)
    %1 = "some_def"() : () -> (vector<64x32xf32>)
    %2 = vector.insert_strided_slice %0, %1 { offsets = [18, 0], strides = [1]}
      : vector<32xf32> into vector<64x32xf32>
    gpu.yield %2 : vector<64x32xf32>
  }
  return %r : vector<64x1xf32>
}

// -----
// CHECK-PROP-LABEL: func.func @vector_insert_strided_slice_2d_to_2d(
//  CHECK-PROP-SAME: %[[LANEID:.*]]: index)
//       CHECK-PROP: %[[W:.*]]:2 = gpu.warp_execute_on_lane_0{{.*}} -> (vector<16x1xf32>, vector<64x1xf32>) {
//       CHECK-PROP: %[[SRC:.*]] = "some_def"() : () -> vector<16x32xf32>
//       CHECK-PROP: %[[DEST:.*]] = "some_def"() : () -> vector<64x32xf32>
//       CHECK-PROP: gpu.yield %[[SRC]], %[[DEST]] : vector<16x32xf32>, vector<64x32xf32>
//       CHECK-PROP: %[[INSERT:.*]] = vector.insert_strided_slice %[[W]]#0, %[[W]]#1 {offsets = [36, 0], strides = [1, 1]} :
//  CHECK-PROP-SAME: vector<16x1xf32> into vector<64x1xf32>
//       CHECK-PROP: return %[[INSERT]] : vector<64x1xf32>
func.func @vector_insert_strided_slice_2d_to_2d(%laneid: index) -> (vector<64x1xf32>) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<64x1xf32>) {
    %0 = "some_def"() : () -> (vector<16x32xf32>)
    %1 = "some_def"() : () -> (vector<64x32xf32>)
    %2 = vector.insert_strided_slice %0, %1 { offsets = [36, 0],  strides = [1, 1]}
      : vector<16x32xf32> into vector<64x32xf32>
    gpu.yield %2 : vector<64x32xf32>
  }
  return %r : vector<64x1xf32>
}

// -----

// Make sure that all operands of the transfer_read op are properly propagated.
// The vector.extract op cannot be propagated because index-typed
// shuffles are not supported at the moment.

// CHECK-PROP: #[[$MAP:.*]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-PROP-LABEL: func @transfer_read_prop_operands(
//  CHECK-PROP-SAME:     %[[IN2:[^ :]*]]: vector<1x2xindex>,
//  CHECK-PROP-SAME:     %[[AR1:[^ :]*]]: memref<1x4x2xi32>,
//  CHECK-PROP-SAME:     %[[AR2:[^ :]*]]: memref<1x4x1024xf32>)
//   CHECK-PROP-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-PROP-DAG:   %[[THREADID:.*]] = gpu.thread_id  x
//       CHECK-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%[[THREADID]])[32] args(%[[IN2]]
//       CHECK-PROP:     %[[GATHER:.*]] = vector.gather %[[AR1]][{{.*}}]
//       CHECK-PROP:     %[[EXTRACT:.*]] = vector.extract %[[GATHER]][0] : vector<64xi32> from vector<1x64xi32>
//       CHECK-PROP:     %[[CAST:.*]] = arith.index_cast %[[EXTRACT]] : vector<64xi32> to vector<64xindex>
//       CHECK-PROP:     %[[EXTRACTELT:.*]] = vector.extract %[[CAST]][{{.*}}] : index from vector<64xindex>
//       CHECK-PROP:     gpu.yield %[[EXTRACTELT]] : index
//       CHECK-PROP:   %[[APPLY:.*]] = affine.apply #[[$MAP]]()[%[[THREADID]]]
//       CHECK-PROP:   %[[TRANSFERREAD:.*]] = vector.transfer_read %[[AR2]][%[[C0]], %[[W]], %[[APPLY]]],
//       CHECK-PROP:   return %[[TRANSFERREAD]]
func.func @transfer_read_prop_operands(%in2: vector<1x2xindex>, %ar1 :  memref<1x4x2xi32>, %ar2 : memref<1x4x1024xf32>)-> vector<2xf32> {
  %0 = gpu.thread_id  x
  %c0_i32 = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0> : vector<1x64xi32>
  %cst_0 = arith.constant dense<true> : vector<1x64xi1>
  %cst_1 = arith.constant dense<3> : vector<64xindex>
  %cst_2 = arith.constant dense<0> : vector<64xindex>
  %cst_6 = arith.constant 0.000000e+00 : f32

  %18 = gpu.warp_execute_on_lane_0(%0)[32] args(%in2 : vector<1x2xindex>) -> (vector<2xf32>) {
  ^bb0(%arg4: vector<1x64xindex>):
    %28 = vector.gather %ar1[%c0, %c0, %c0] [%arg4], %cst_0, %cst : memref<1x4x2xi32>, vector<1x64xindex>, vector<1x64xi1>, vector<1x64xi32> into vector<1x64xi32>
    %29 = vector.extract %28[0] : vector<64xi32> from vector<1x64xi32>
    %30 = arith.index_cast %29 : vector<64xi32> to vector<64xindex>
    %36 = vector.extract %30[%c0_i32] : index from vector<64xindex>
    %37 = vector.transfer_read %ar2[%c0, %36, %c0], %cst_6 {in_bounds = [true]} : memref<1x4x1024xf32>, vector<64xf32>
    gpu.yield %37 : vector<64xf32>
  }
  return %18 : vector<2xf32>
}

// -----

// Check that we don't fold vector.broadcast when each thread doesn't get the
// same value.

// CHECK-PROP-LABEL: func @dont_fold_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = gpu.warp_execute_on_lane_0{{.*}} -> (vector<1x2xf32>)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     %[[broadcast:.*]] = vector.broadcast %[[some_def]] : vector<64xf32> to vector<1x64xf32>
//       CHECK-PROP:     gpu.yield %[[broadcast]] : vector<1x64xf32>
//       CHECK-PROP:   vector.print %[[r]] : vector<1x2xf32>
func.func @dont_fold_vector_broadcast(%laneid: index) {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1x2xf32>) {
    %0 = "some_def"() : () -> (vector<64xf32>)
    %1 = vector.broadcast %0 : vector<64xf32> to vector<1x64xf32>
    gpu.yield %1 : vector<1x64xf32>
  }
  vector.print %r : vector<1x2xf32>
  return
}

// -----

func.func @warp_propagate_shape_cast(%laneid: index, %src: memref<32x4x32xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[1024] -> (vector<4xf32>) {
    %2 = vector.transfer_read %src[%c0, %c0, %c0], %cst : memref<32x4x32xf32>, vector<32x4x32xf32>
    %3 = vector.shape_cast %2 : vector<32x4x32xf32> to vector<4096xf32>
    gpu.yield %3 : vector<4096xf32>
  }
  return %r : vector<4xf32>
}

// CHECK-PROP-LABEL: func.func @warp_propagate_shape_cast
// CHECK-PROP:   %[[READ:.+]] = vector.transfer_read {{.+}} : memref<32x4x32xf32>, vector<1x1x4xf32>
// CHECK-PROP:   %[[CAST:.+]] = vector.shape_cast %[[READ]] : vector<1x1x4xf32> to vector<4xf32>
// CHECK-PROP:   return %[[CAST]] : vector<4xf32>

// -----

func.func @warp_propagate_uniform_transfer_read(%laneid: index, %src: memref<4096xf32>, %index: index) -> vector<1xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<1xf32>) {
    %1 = vector.transfer_read %src[%index], %f0 {in_bounds = [true]} : memref<4096xf32>, vector<1xf32>
    gpu.yield %1 : vector<1xf32>
  }
  return %r : vector<1xf32>
}

// CHECK-PROP-LABEL: func.func @warp_propagate_uniform_transfer_read
//  CHECK-PROP-SAME: (%{{.+}}: index, %[[SRC:.+]]: memref<4096xf32>, %[[INDEX:.+]]: index)
//       CHECK-PROP:   %[[READ:.+]] = vector.transfer_read %[[SRC]][%[[INDEX]]], %cst {in_bounds = [true]} : memref<4096xf32>, vector<1xf32>
//       CHECK-PROP:   return %[[READ]] : vector<1xf32>

// -----

func.func @warp_propagate_multi_transfer_read(%laneid: index, %src: memref<4096xf32>, %index: index, %index1: index) -> (vector<1xf32>, vector<1xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<1xf32>, vector<1xf32>) {
    %0 = vector.transfer_read %src[%index], %f0 {in_bounds = [true]} : memref<4096xf32>, vector<1xf32>
    "some_use"(%0) : (vector<1xf32>) -> ()
    %1 = vector.transfer_read %src[%index1], %f0 {in_bounds = [true]} : memref<4096xf32>, vector<1xf32>
    gpu.yield %0, %1 : vector<1xf32>, vector<1xf32>
  }
  return %r#0, %r#1 : vector<1xf32>, vector<1xf32>
}

// CHECK-PROP-LABEL: func.func @warp_propagate_multi_transfer_read
//       CHECK-PROP:   gpu.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK-PROP:     %[[INNER_READ:.+]] = vector.transfer_read
//       CHECK-PROP:     "some_use"(%[[INNER_READ]])
//       CHECK-PROP:     gpu.yield %[[INNER_READ]] : vector<1xf32>
//       CHECK-PROP:   vector.transfer_read

// -----

func.func @warp_propagate_dead_user_multi_read(%laneid: index, %src: memref<4096xf32>, %index: index, %index1: index) -> (vector<1xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %r = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<1xf32>) {
    %0 = vector.transfer_read %src[%index], %f0 {in_bounds = [true]} : memref<4096xf32>, vector<64xf32>
    %1 = vector.transfer_read %src[%index1], %f0 {in_bounds = [true]} : memref<4096xf32>, vector<64xf32>
    %max = arith.maximumf %0, %1 : vector<64xf32>
    gpu.yield %max : vector<64xf32>
  }
  return %r : vector<1xf32>
}

//   CHECK-PROP-LABEL: func.func @warp_propagate_dead_user_multi_read
// CHECK-PROP-COUNT-2:   vector.transfer_read {{.*}} vector<1xf32>
//         CHECK-PROP:   arith.maximumf {{.*}} : vector<1xf32>

// -----

func.func @warp_propagate_masked_write(%laneid: index, %dest: memref<4096xf32>) {
  %c0 = arith.constant 0 : index
  gpu.warp_execute_on_lane_0(%laneid)[32] -> () {
    %mask = "mask_def_0"() : () -> (vector<4096xi1>)
    %mask2 = "mask_def_1"() : () -> (vector<32xi1>)
    %0 = "some_def_0"() : () -> (vector<4096xf32>)
    %1 = "some_def_1"() : () -> (vector<32xf32>)
    vector.transfer_write %0, %dest[%c0], %mask : vector<4096xf32>, memref<4096xf32>
    vector.transfer_write %1, %dest[%c0], %mask2 : vector<32xf32>, memref<4096xf32>
    gpu.yield
  }
  return
}

// CHECK-DIST-AND-PROP-LABEL: func.func @warp_propagate_masked_write(
//       CHECK-DIST-AND-PROP:   %[[W:.*]]:4 = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>, vector<1xi1>, vector<128xf32>, vector<128xi1>) {
//       CHECK-DIST-AND-PROP:     %[[M0:.*]] = "mask_def_0"
//       CHECK-DIST-AND-PROP:     %[[M1:.*]] = "mask_def_1"
//       CHECK-DIST-AND-PROP:     %[[V0:.*]] = "some_def_0"
//       CHECK-DIST-AND-PROP:     %[[V1:.*]] = "some_def_1"
//       CHECK-DIST-AND-PROP:     gpu.yield %[[V1]], %[[M1]], %[[V0]], %[[M0]]
//  CHECK-DIST-AND-PROP-SAME:       vector<32xf32>, vector<32xi1>, vector<4096xf32>, vector<4096xi1>
//       CHECK-DIST-AND-PROP:   }
//       CHECK-DIST-AND-PROP:   vector.transfer_write %[[W]]#2, {{.*}}, %[[W]]#3 {in_bounds = [true]} : vector<128xf32>, memref<4096xf32>
//       CHECK-DIST-AND-PROP:   vector.transfer_write %[[W]]#0, {{.*}}, %[[W]]#1 {in_bounds = [true]} : vector<1xf32>, memref<4096xf32>

// -----

func.func @warp_propagate_masked_transfer_read(%laneid: index, %src: memref<4096x4096xf32>, %index: index) -> (vector<2xf32>, vector<2x2xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<2xf32>, vector<2x2xf32>) {
    %mask = "mask_def_0"() : () -> (vector<128xi1>)
    %0 = vector.transfer_read %src[%c0, %index], %f0, %mask {in_bounds = [true]} : memref<4096x4096xf32>, vector<128xf32>
    %mask2 = "mask_def_1"() : () -> (vector<128x2xi1>)
    %1 = vector.transfer_read %src[%c0, %index], %f0, %mask2 {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<128x2xf32>
    gpu.yield %0, %1 : vector<128xf32>, vector<128x2xf32>
  }
  return %r#0, %r#1 : vector<2xf32>, vector<2x2xf32>
}

//   CHECK-PROP-DAG: #[[$MAP0:.+]] = affine_map<()[s0] -> (s0 * 2)>
//   CHECK-PROP-DAG: #[[$MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
// CHECK-PROP-LABEL: func.func @warp_propagate_masked_transfer_read
//  CHECK-PROP-SAME:   %[[ARG0:.+]]: index, {{.*}}, %[[ARG2:.+]]: index
//       CHECK-PROP:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK-PROP:   %[[R:.*]]:2 = gpu.warp_execute_on_lane_0(%{{.*}})[64] -> (vector<2xi1>, vector<2x2xi1>) {
//       CHECK-PROP:     %[[M0:.*]] = "mask_def_0"
//       CHECK-PROP:     %[[M1:.*]] = "mask_def_1"
//       CHECK-PROP:     gpu.yield %[[M0]], %[[M1]] : vector<128xi1>, vector<128x2xi1>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[DIST_READ_IDX0:.+]] = affine.apply #[[$MAP0]]()[%[[ARG0]]]
//       CHECK-PROP:   vector.transfer_read {{.*}}[%[[DIST_READ_IDX0]], %[[ARG2]]], {{.*}}, %[[R]]#1 {{.*}} vector<2x2xf32>
//       CHECK-PROP:   %[[DIST_READ_IDX1:.+]] = affine.apply #[[$MAP1]]()[%[[ARG2]], %[[ARG0]]]
//       CHECK-PROP:   vector.transfer_read {{.*}}[%[[C0]], %[[DIST_READ_IDX1]]], {{.*}}, %[[R]]#0 {{.*}} vector<2xf32>

// -----

func.func @warp_propagate_nontrivial_map_masked_transfer_read(%laneid: index, %src: memref<4096x4096xf32>, %index: index) -> vector<2xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %r = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<2xf32>) {
    %mask = "mask_def_0"() : () -> (vector<128xi1>)
    %0 = vector.transfer_read %src[%index, %c0], %f0, %mask {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4096x4096xf32>, vector<128xf32>
    gpu.yield %0 : vector<128xf32>
  }
  return %r : vector<2xf32>
}

//   CHECK-PROP-DAG: #[[$MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 2)>
//   CHECK-PROP-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-PROP-LABEL: func.func @warp_propagate_nontrivial_map_masked_transfer_read
//  CHECK-PROP-SAME:   %[[ARG0:.+]]: index, {{.*}}, %[[ARG2:.+]]: index
//       CHECK-PROP:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK-PROP:   %[[R:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[64] -> (vector<2xi1>) {
//       CHECK-PROP:     %[[M0:.*]] = "mask_def_0"
//       CHECK-PROP:     gpu.yield %[[M0]] : vector<128xi1>
//       CHECK-PROP:   }
//       CHECK-PROP:   %[[DIST_READ_IDX0:.+]] = affine.apply #[[$MAP0]]()[%[[ARG2]], %[[ARG0]]]
//       CHECK-PROP:   vector.transfer_read {{.*}}[%[[DIST_READ_IDX0]], %[[C0]]], {{.*}}, %[[R]]
//  CHECK-PROP-SAME:   permutation_map = #[[$MAP1]]} {{.*}} vector<2xf32>

// -----

func.func @warp_propagate_masked_transfer_read_shared_mask(%laneid: index, %src: memref<4096x4096xf32>, %index: index, %index2: index, %mask_ub: index) -> (vector<2xf32>, vector<2xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[64] -> (vector<2xf32>, vector<2xf32>) {
    %mask = vector.create_mask %mask_ub: vector<128xi1>
    %0 = vector.transfer_read %src[%c0, %index], %f0, %mask {in_bounds = [true]} : memref<4096x4096xf32>, vector<128xf32>
    %1 = vector.transfer_read %src[%c0, %index2], %f0, %mask {in_bounds = [true]} : memref<4096x4096xf32>, vector<128xf32>
    gpu.yield %0, %1 : vector<128xf32>, vector<128xf32>
  }
  return %r#0, %r#1 : vector<2xf32>, vector<2xf32>
}

// CHECK-PROP-LABEL: func.func @warp_propagate_masked_transfer_read_shared_mask
//       CHECK-PROP:   vector.create_mask %{{.*}} : vector<2xi1>
//       CHECK-PROP:   vector.transfer_read %{{.*}} : memref<4096x4096xf32>, vector<2xf32>
//       CHECK-PROP:   vector.create_mask %{{.*}} : vector<2xi1>
//       CHECK-PROP:   vector.transfer_read %{{.*}} : memref<4096x4096xf32>, vector<2xf32>

// -----

func.func @warp_propagate_unconnected_read_write(%laneid: index, %buffer: memref<128xf32>, %f1: f32) -> (vector<2xf32>, vector<4xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %r:2 = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>, vector<4xf32>) {
    %cst = arith.constant dense<2.0> : vector<128xf32>
    %0 = vector.transfer_read %buffer[%c0], %f0 {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
    vector.transfer_write %cst, %buffer[%c0] : vector<128xf32>, memref<128xf32>
    %1 = vector.broadcast %f1 : f32 to vector<64xf32>
    gpu.yield %1, %0 : vector<64xf32>, vector<128xf32>
  }
  return %r#0, %r#1 : vector<2xf32>, vector<4xf32>
}

// Verify that the write comes after the read
// CHECK-DIST-AND-PROP-LABEL: func.func @warp_propagate_unconnected_read_write(
//       CHECK-DIST-AND-PROP:   %[[CST:.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
//       CHECK-DIST-AND-PROP:   vector.transfer_read {{.*}} : memref<128xf32>, vector<4xf32>
//       CHECK-DIST-AND-PROP:   vector.transfer_write %[[CST]], {{.*}} : vector<4xf32>, memref<128xf32>

// -----

func.func @warp_propagate_create_mask(%laneid: index, %m0: index) -> vector<1xi1> {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xi1>) {
    %1 = vector.create_mask %m0 : vector<32xi1>
    gpu.yield %1 : vector<32xi1>
  }
  return %r : vector<1xi1>
}

//   CHECK-PROP-DAG: #[[$SUB:.*]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK-PROP-LABEL: func @warp_propagate_create_mask
//  CHECK-PROP-SAME: %[[LANEID:.+]]: index, %[[M0:.+]]: index
//       CHECK-PROP:   %[[MDIST:.+]] = affine.apply #[[$SUB]]()[%[[LANEID]], %[[M0]]]
//       CHECK-PROP:   vector.create_mask %[[MDIST]] : vector<1xi1>

// -----

func.func @warp_propagate_multi_dim_create_mask(%laneid: index, %m0: index, %m1: index, %m2: index) -> vector<1x2x4xi1> {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1x2x4xi1>) {
    %1 = vector.create_mask %m0, %m1, %m2 : vector<16x4x4xi1>
    gpu.yield %1 : vector<16x4x4xi1>
  }
  return %r : vector<1x2x4xi1>
}

//   CHECK-PROP-DAG: #[[$SUBM0:.*]] = affine_map<()[s0, s1] -> (s0 - s1 floordiv 2)>
//   CHECK-PROP-DAG: #[[$SUBM1:.*]] = affine_map<()[s0, s1] -> (s0 - s1 * 2 + (s1 floordiv 2) * 4)>
// CHECK-PROP-LABEL: func @warp_propagate_multi_dim_create_mask
//  CHECK-PROP-SAME: %[[LANEID:.+]]: index, %[[M0:.+]]: index, %[[M1:.+]]: index, %[[M2:.+]]: index
//       CHECK-PROP:   %[[DISTM0:.+]] = affine.apply #[[$SUBM0]]()[%[[M0]], %[[LANEID]]]
//       CHECK-PROP:   %[[DISTM1:.+]] = affine.apply #[[$SUBM1]]()[%[[M1]], %[[LANEID]]]
//       CHECK-PROP:   vector.create_mask %[[DISTM0]], %[[DISTM1]], %[[M2]] : vector<1x2x4xi1>

// -----

func.func @warp_propagate_nd_write(%laneid: index, %dest: memref<4x1024xf32>) {
  %c0 = arith.constant 0 : index
  gpu.warp_execute_on_lane_0(%laneid)[32] -> () {
    %0 = "some_def"() : () -> (vector<4x1024xf32>)
    vector.transfer_write %0, %dest[%c0, %c0] : vector<4x1024xf32>, memref<4x1024xf32>
    gpu.yield
  }
  return
}

//       CHECK-DIST-AND-PROP: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>

// CHECK-DIST-AND-PROP-LABEL: func.func @warp_propagate_nd_write(
//       CHECK-DIST-AND-PROP:   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1x128xf32>) {
//       CHECK-DIST-AND-PROP:     %[[V0:.*]] = "some_def"
//       CHECK-DIST-AND-PROP:     gpu.yield %[[V0]]
//  CHECK-DIST-AND-PROP-SAME:       vector<4x1024xf32>
//       CHECK-DIST-AND-PROP:   }

//       CHECK-DIST-AND-PROP:   %[[IDS:.+]]:2 = affine.delinearize_index %{{.*}} into (4, 8) : index, index
//       CHECK-DIST-AND-PROP:   %[[INNER_ID:.+]] = affine.apply #map()[%[[IDS]]#1]
//       CHECK-DIST-AND-PROP:   vector.transfer_write %[[W]], %{{.*}}[%[[IDS]]#0, %[[INNER_ID]]] {{.*}} : vector<1x128xf32>

// -----
func.func @warp_propagate_duplicated_operands_in_yield(%laneid: index)  {
  %r:3 = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>, vector<1xf32>, vector<1xf32>) {
    %0 = "some_def"() : () -> (vector<32xf32>)
    %1 = "some_other_def"() : () -> (vector<32xf32>)
    %2 = math.exp %1 : vector<32xf32>
    gpu.yield %2, %0, %0 : vector<32xf32>, vector<32xf32>, vector<32xf32>
  }
  "some_use"(%r#0) : (vector<1xf32>) -> ()
  return
}

// CHECK-PROP-LABEL : func.func @warp_propagate_duplicated_operands_in_yield(
// CHECK-PROP       :   %[[W:.*]] = gpu.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
// CHECK-PROP       :     %{{.*}} = "some_def"() : () -> vector<32xf32>
// CHECK-PROP       :     %[[T3:.*]] = "some_other_def"() : () -> vector<32xf32>
// CHECK-PROP       :     gpu.yield %[[T3]] : vector<32xf32>
// CHECK-PROP       :   }
// CHECK-PROP       :   %[T1:.*] = math.exp %[[W]] : vector<1xf32>
// CHECK-PROP       :   "some_use"(%[[T1]]) : (vector<1xf32>) -> ()

// -----

func.func @warp_step_distribute(%buffer: memref<128xindex>)  {
  %laneid = gpu.lane_id
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xindex>) {
    %seq = vector.step : vector<32xindex>
    gpu.yield %seq : vector<32xindex>
  }
  vector.transfer_write %r, %buffer[%laneid] : vector<1xindex>, memref<128xindex>
  return
}

// CHECK-PROP-LABEL: func.func @warp_step_distribute(
//       CHECK-PROP:   %[[LANE_ID:.*]] = gpu.lane_id
//       CHECK-PROP:   %[[LANE_ID_VEC:.*]] = vector.broadcast %[[LANE_ID]] : index to vector<1xindex>
//       CHECK-PROP:   vector.transfer_write %[[LANE_ID_VEC]], %{{.*}} : vector<1xindex>, memref<128xindex>

// -----

func.func @negative_warp_step_more_than_warp_size(%laneid: index, %buffer: memref<128xindex>)  {
  %r = gpu.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xindex>) {
    %seq = vector.step : vector<64xindex>
    gpu.yield %seq : vector<64xindex>
  }
  vector.transfer_write %r, %buffer[%laneid] : vector<2xindex>, memref<128xindex>
  return
}

// CHECK-PROP-LABEL: @negative_warp_step_more_than_warp_size
// CHECK-PROP-NOT: vector.broadcast
// CHECK-PROP: vector.step : vector<64xindex>
