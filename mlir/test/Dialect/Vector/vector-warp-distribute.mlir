// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if | FileCheck %s --check-prefix=CHECK-SCF-IF
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute="hoist-uniform" | FileCheck --check-prefixes=CHECK-HOIST %s
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute="hoist-uniform distribute-transfer-write" | FileCheck --check-prefixes=CHECK-D %s
// RUN: mlir-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=propagate-distribution -canonicalize | FileCheck --check-prefixes=CHECK-PROP %s

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
//   CHECK-SCF-IF-DAG:   %[[c2:.*]] = arith.constant 2 : index
//   CHECK-SCF-IF-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-SCF-IF-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK-SCF-IF:   %[[is_lane_0:.*]] = arith.cmpi eq, %[[laneid]], %[[c0]]

//       CHECK-SCF-IF:   %[[buffer_v0:.*]] = memref.get_global @__shared_128xf32
//       CHECK-SCF-IF:   %[[s0:.*]] = arith.muli %[[laneid]], %[[c4]]
//       CHECK-SCF-IF:   vector.store %[[v0]], %[[buffer_v0]][%[[s0]]]
//       CHECK-SCF-IF:   %[[buffer_v1:.*]] = memref.get_global @__shared_256xf32
//       CHECK-SCF-IF:   %[[s1:.*]] = arith.muli %[[laneid]], %[[c8]]
//       CHECK-SCF-IF:   vector.store %[[v1]], %[[buffer_v1]][%[[s1]]]

//   CHECK-SCF-IF-DAG:   gpu.barrier
//   CHECK-SCF-IF-DAG:   %[[buffer_def_0:.*]] = memref.get_global @__shared_32xf32
//   CHECK-SCF-IF-DAG:   %[[buffer_def_1:.*]] = memref.get_global @__shared_64xf32

//       CHECK-SCF-IF:   scf.if %[[is_lane_0]] {
  %r:2 = vector.warp_execute_on_lane_0(%laneid)[32]
      args(%v0, %v1 : vector<4xf32>, vector<8xf32>) -> (vector<1xf32>, vector<2xf32>) {
    ^bb0(%arg0: vector<128xf32>, %arg1: vector<256xf32>):
//       CHECK-SCF-IF:     %[[arg1:.*]] = vector.load %[[buffer_v1]][%[[c0]]] : memref<256xf32, 3>, vector<256xf32>
//       CHECK-SCF-IF:     %[[arg0:.*]] = vector.load %[[buffer_v0]][%[[c0]]] : memref<128xf32, 3>, vector<128xf32>
//       CHECK-SCF-IF:     %[[def_0:.*]] = "some_def"(%[[arg0]]) : (vector<128xf32>) -> vector<32xf32>
//       CHECK-SCF-IF:     %[[def_1:.*]] = "some_def"(%[[arg1]]) : (vector<256xf32>) -> vector<64xf32>
    %2 = "some_def"(%arg0) : (vector<128xf32>) -> vector<32xf32>
    %3 = "some_def"(%arg1) : (vector<256xf32>) -> vector<64xf32>
//       CHECK-SCF-IF:     vector.store %[[def_0]], %[[buffer_def_0]][%[[c0]]]
//       CHECK-SCF-IF:     vector.store %[[def_1]], %[[buffer_def_1]][%[[c0]]]
    vector.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
//       CHECK-SCF-IF:   }
//       CHECK-SCF-IF:   gpu.barrier
//       CHECK-SCF-IF:   %[[o1:.*]] = arith.muli %[[laneid]], %[[c2]]
//       CHECK-SCF-IF:   %[[r1:.*]] = vector.load %[[buffer_def_1]][%[[o1]]] : memref<64xf32, 3>, vector<2xf32>
//       CHECK-SCF-IF:   %[[r0:.*]] = vector.load %[[buffer_def_0]][%[[laneid]]] : memref<32xf32, 3>, vector<1xf32>
//       CHECK-SCF-IF:   "some_use"(%[[r0]]) : (vector<1xf32>) -> ()
//       CHECK-SCF-IF:   "some_use"(%[[r1]]) : (vector<2xf32>) -> ()
  "some_use"(%r#0) : (vector<1xf32>) -> ()
  "some_use"(%r#1) : (vector<2xf32>) -> ()
  return
}

// -----

// CHECK-D-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 2 + 32)>

// CHECK-ALL-LABEL: func @warp(
// CHECK-HOIST: memref.subview
// CHECK-HOIST: memref.subview
// CHECK-HOIST: memref.subview
// CHECK-HOIST: vector.warp_execute_on_lane_0

//     CHECK-D: %[[R:.*]]:2 = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<2xf32>, vector<1xf32>) {
//     CHECK-D:   arith.addf {{.*}} : vector<32xf32>
//     CHECK-D:   arith.addf {{.*}} : vector<64xf32>
//     CHECK-D:   vector.yield %{{.*}}, %{{.*}} : vector<64xf32>, vector<32xf32>
// CHECK-D-DAG: vector.transfer_write %[[R]]#1, %{{.*}}[%{{.*}}] {in_bounds = [true]} : vector<1xf32>, memref<128xf32
// CHECK-D-DAG: %[[ID1:.*]] = affine.apply #[[MAP1]]()[%{{.*}}]
// CHECK-D-DAG: vector.transfer_write %[[R]]#0, %2[%[[ID1]]] {in_bounds = [true]} : vector<2xf32>, memref<128xf32

// CHECK-ALL-NOT: vector.warp_execute_on_lane_0
// CHECK-ALL: vector.transfer_read {{.*}} vector<1xf32>
// CHECK-ALL: vector.transfer_read {{.*}} vector<1xf32>
// CHECK-ALL: vector.transfer_read {{.*}} vector<2xf32>
// CHECK-ALL: vector.transfer_read {{.*}} vector<2xf32>
// CHECK-ALL: arith.addf {{.*}} : vector<1xf32>
// CHECK-ALL: arith.addf {{.*}} : vector<2xf32>
// CHECK-ALL: vector.transfer_write {{.*}} : vector<1xf32>
// CHECK-ALL: vector.transfer_write {{.*}} : vector<2xf32>

#map0 =  affine_map<(d0)[s0] -> (d0 + s0)>
func.func @warp(%laneid: index, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>,
           %arg3: memref<1024xf32>, %gid : index) {
  vector.warp_execute_on_lane_0(%laneid)[32] {
    %sa = memref.subview %arg1[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %sb = memref.subview %arg2[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %sc = memref.subview %arg3[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map0>
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    %2 = vector.transfer_read %sa[%c0], %cst : memref<128xf32, #map0>, vector<32xf32>
    %3 = vector.transfer_read %sa[%c32], %cst : memref<128xf32, #map0>, vector<32xf32>
    %4 = vector.transfer_read %sb[%c0], %cst : memref<128xf32, #map0>, vector<64xf32>
    %5 = vector.transfer_read %sb[%c32], %cst : memref<128xf32, #map0>, vector<64xf32>
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    vector.transfer_write %6, %sc[%c0] : vector<32xf32>, memref<128xf32, #map0>
    vector.transfer_write %7, %sc[%c32] : vector<64xf32>, memref<128xf32, #map0>
  }
  return
}

// -----

// CHECK-D-LABEL: func @warp_extract(
//       CHECK-D:   %[[WARPOP:.*]] = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>)
//       CHECK-D:     "test.dummy_op"
//       CHECK-D:     vector.yield %{{.*}} : vector<1xf32>
//       CHECK-D:   }
//       CHECK-D:   vector.warp_execute_on_lane_0(%{{.*}})[32] {
//       CHECK-D:     vector.transfer_write %[[WARPOP]], %{{.*}}[%{{.*}}] {{.*}} : vector<1xf32>
//       CHECK-D:   }

#map2 =  affine_map<(d0)[s0] -> (d0 + s0)>

func.func @warp_extract(%laneid: index, %arg1: memref<1024xf32>, %gid : index) {
  vector.warp_execute_on_lane_0(%laneid)[32] {
    %sa = memref.subview %arg1[%gid] [128] [1] : memref<1024xf32> to memref<128xf32, #map2>
    %c0 = arith.constant 0 : index
    %v = "test.dummy_op"() : () -> (vector<1xf32>)
    vector.transfer_write %v, %sa[%c0] : vector<1xf32>, memref<128xf32, #map2>
  }
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_dead_result(
func.func @warp_dead_result(%laneid: index) -> (vector<1xf32>) {
  // CHECK-PROP: %[[R:.*]] = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>)
  %r:3 = vector.warp_execute_on_lane_0(%laneid)[32] ->
    (vector<1xf32>, vector<1xf32>, vector<1xf32>) {
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<32xf32>)
  // CHECK-PROP:   vector.yield %{{.*}} : vector<32xf32>
    vector.yield %2, %3, %4 : vector<32xf32>, vector<32xf32>, vector<32xf32>
  }
  // CHECK-PROP: return %[[R]] : vector<1xf32>
  return %r#1 : vector<1xf32>
}

// -----

// CHECK-PROP-LABEL:   func @warp_propagate_operand(
//  CHECK-PROP-SAME:   %[[ID:.*]]: index, %[[V:.*]]: vector<4xf32>)
func.func @warp_propagate_operand(%laneid: index, %v0: vector<4xf32>)
  -> (vector<4xf32>) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32]
     args(%v0 : vector<4xf32>) -> (vector<4xf32>) {
     ^bb0(%arg0 : vector<128xf32>) :
    vector.yield %arg0 : vector<128xf32>
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
  // CHECK-PROP: %[[R:.*]]:4 = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>, vector<1xf32>, vector<2xf32>, vector<2xf32>)
  %r:2 = vector.warp_execute_on_lane_0(%laneid)[32] ->
    (vector<1xf32>, vector<2xf32>) {
    // CHECK-PROP: %[[V0:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK-PROP: %[[V1:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK-PROP: %[[V2:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK-PROP: %[[V3:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK-PROP: vector.yield %[[V0]], %[[V1]], %[[V2]], %[[V3]] : vector<32xf32>, vector<32xf32>, vector<64xf32>, vector<64xf32>
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<64xf32>)
    %5 = "some_def"() : () -> (vector<64xf32>)
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    vector.yield %6, %7 : vector<32xf32>, vector<64xf32>
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
//       CHECK-PROP:   %[[r:.*]]:2 = vector.warp_execute_on_lane_0{{.*}} {
//       CHECK-PROP:     %[[some_def0:.*]] = "some_def"
//       CHECK-PROP:     %[[some_def1:.*]] = "some_def"
//       CHECK-PROP:     vector.yield %[[some_def0]], %[[some_def1]]
//       CHECK-PROP:   }
//       CHECK-PROP:   arith.addf %[[r]]#0, %[[r]]#1 : f32
func.func @warp_propagate_scalar_arith(%laneid: index) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %0 = "some_def"() : () -> (f32)
    %1 = "some_def"() : () -> (f32)
    %2 = arith.addf %0, %1 : f32
    vector.yield %2 : f32
  }
  vector.print %r : f32
  return
}

// -----

// CHECK-PROP-LABEL: func @warp_propagate_cast(
//   CHECK-PROP-NOT:   vector.warp_execute_on_lane_0
//       CHECK-PROP:   %[[result:.*]] = arith.sitofp %{{.*}} : i32 to f32
//       CHECK-PROP:   return %[[result]]
func.func @warp_propagate_cast(%laneid : index, %i : i32) -> (f32) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32] -> (f32) {
    %casted = arith.sitofp %i : i32 to f32
    vector.yield %casted : f32
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
  %r:2 = vector.warp_execute_on_lane_0(%laneid)[32] ->(vector<1xf32>, vector<2xf32>) {
    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<32xf32>
    %3 = vector.transfer_read %src[%c32], %cst : memref<1024xf32>, vector<64xf32>
    vector.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
  %id2 = affine.apply #map0()[%laneid]
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @fold_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = vector.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     vector.yield %[[some_def]] : vector<1xf32>
//       CHECK-PROP:   vector.print %[[r]] : vector<1xf32>
func.func @fold_vector_broadcast(%laneid: index) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32] -> (vector<1xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<32xf32>
    vector.yield %1 : vector<32xf32>
  }
  vector.print %r : vector<1xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @extract_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = vector.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     vector.yield %[[some_def]] : vector<1xf32>
//       CHECK-PROP:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : vector<1xf32> to vector<2xf32>
//       CHECK-PROP:   vector.print %[[broadcasted]] : vector<2xf32>
func.func @extract_vector_broadcast(%laneid: index) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<64xf32>
    vector.yield %1 : vector<64xf32>
  }
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-PROP-LABEL: func @extract_scalar_vector_broadcast(
//       CHECK-PROP:   %[[r:.*]] = vector.warp_execute_on_lane_0{{.*}} -> (f32)
//       CHECK-PROP:     %[[some_def:.*]] = "some_def"
//       CHECK-PROP:     vector.yield %[[some_def]] : f32
//       CHECK-PROP:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : f32 to vector<2xf32>
//       CHECK-PROP:   vector.print %[[broadcasted]] : vector<2xf32>
func.func @extract_scalar_vector_broadcast(%laneid: index) {
  %r = vector.warp_execute_on_lane_0(%laneid)[32] -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (f32)
    %1 = vector.broadcast %0 : f32 to vector<64xf32>
    vector.yield %1 : vector<64xf32>
  }
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for(
// CHECK-PROP: %[[INI:.*]] = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>) {
// CHECK-PROP:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   vector.yield %[[INI1]] : vector<128xf32>
// CHECK-PROP: }
// CHECK-PROP: %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG:.*]] = %[[INI]]) -> (vector<4xf32>) {
// CHECK-PROP:   %[[W:.*]] = vector.warp_execute_on_lane_0(%{{.*}})[32] args(%[[FARG]] : vector<4xf32>) -> (vector<4xf32>) {
// CHECK-PROP:    ^bb0(%[[ARG:.*]]: vector<128xf32>):
// CHECK-PROP:      %[[ACC:.*]] = "some_def"(%[[ARG]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      vector.yield %[[ACC]] : vector<128xf32>
// CHECK-PROP:   }
// CHECK-PROP:   scf.yield %[[W]] : vector<4xf32>
// CHECK-PROP: }
// CHECK-PROP: "some_use"(%[[F]]) : (vector<4xf32>) -> ()
func.func @warp_scf_for(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = vector.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>) {
    %ini = "some_def"() : () -> (vector<128xf32>)
    %3 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini) -> (vector<128xf32>) {
      %acc = "some_def"(%arg4) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc : vector<128xf32>
    }
    vector.yield %3 : vector<128xf32>
  }
  "some_use"(%0) : (vector<4xf32>) -> ()
  return
}

// -----

// CHECK-PROP-LABEL:   func @warp_scf_for_swap(
// CHECK-PROP: %[[INI:.*]]:2 = vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   %[[INI2:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK-PROP:   vector.yield %[[INI1]], %[[INI2]] : vector<128xf32>, vector<128xf32>
// CHECK-PROP: }
// CHECK-PROP: %[[F:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG1:.*]] = %[[INI]]#0, %[[FARG2:.*]] = %[[INI]]#1) -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:   %[[W:.*]]:2 = vector.warp_execute_on_lane_0(%{{.*}})[32] args(%[[FARG1]], %[[FARG2]] : vector<4xf32>, vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
// CHECK-PROP:    ^bb0(%[[ARG1:.*]]: vector<128xf32>, %[[ARG2:.*]]: vector<128xf32>):
// CHECK-PROP:      %[[ACC1:.*]] = "some_def"(%[[ARG1]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      %[[ACC2:.*]] = "some_def"(%[[ARG2]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK-PROP:      vector.yield %[[ACC2]], %[[ACC1]] : vector<128xf32>, vector<128xf32>
// CHECK-PROP:   }
// CHECK-PROP:   scf.yield %[[W]]#0, %[[W]]#1 : vector<4xf32>, vector<4xf32>
// CHECK-PROP: }
// CHECK-PROP: "some_use"(%[[F]]#0) : (vector<4xf32>) -> ()
// CHECK-PROP: "some_use"(%[[F]]#1) : (vector<4xf32>) -> ()
func.func @warp_scf_for_swap(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:2 = vector.warp_execute_on_lane_0(%arg0)[32] -> (vector<4xf32>, vector<4xf32>) {
    %ini1 = "some_def"() : () -> (vector<128xf32>)
    %ini2 = "some_def"() : () -> (vector<128xf32>)
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini1, %arg5 = %ini2) -> (vector<128xf32>, vector<128xf32>) {
      %acc1 = "some_def"(%arg4) : (vector<128xf32>) -> (vector<128xf32>)
      %acc2 = "some_def"(%arg5) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc2, %acc1 : vector<128xf32>, vector<128xf32>
    }
    vector.yield %3#0, %3#1 : vector<128xf32>, vector<128xf32>
  }
  "some_use"(%0#0) : (vector<4xf32>) -> ()
  "some_use"(%0#1) : (vector<4xf32>) -> ()
  return
}

// -----

#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 128 + 128)>
#map2 = affine_map<()[s0] -> (s0 * 4 + 128)>

// CHECK-PROP-LABEL:   func @warp_scf_for_multiple_yield(
//       CHECK-PROP:   vector.warp_execute_on_lane_0(%{{.*}})[32] -> (vector<1xf32>) {
//  CHECK-PROP-NEXT:     "some_def"() : () -> vector<32xf32>
//  CHECK-PROP-NEXT:     vector.yield %{{.*}} : vector<32xf32>
//  CHECK-PROP-NEXT:   }
//   CHECK-PROP-NOT:   vector.warp_execute_on_lane_0
//       CHECK-PROP:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK-PROP:   %{{.*}}:2 = scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-PROP-NOT:     vector.warp_execute_on_lane_0
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
  %0:3 = vector.warp_execute_on_lane_0(%arg0)[32] ->
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
    vector.yield %def, %3#0, %3#1 :  vector<32xf32>, vector<128xf32>, vector<128xf32>
  }
  %1 = affine.apply #map()[%arg0]
  vector.transfer_write %0#1, %arg2[%1] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  %2 = affine.apply #map2()[%arg0]
  vector.transfer_write %0#2, %arg2[%2] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  "some_use"(%0#0) : (vector<1xf32>) -> ()
  return
}
