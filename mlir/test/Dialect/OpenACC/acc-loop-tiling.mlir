// RUN: mlir-opt %s -acc-loop-tiling | FileCheck %s

// Test single-level loop tiling with tile(2)
// Original loop: for i = 0 to 10 step 1
// After tiling: tile loop (step=2) containing element loop (step=1)

// CHECK-LABEL: func.func @single_loop_tile
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C10:.*]] = arith.constant 10 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%[[C0]] : index) to (%[[C10]] : index) step (%[[C2]] : index) {
// CHECK:           %[[NEW_UB:.*]] = arith.addi %[[IV]], %[[C2]] : index
// CHECK:           %[[MIN_UB:.*]] = arith.minsi %[[NEW_UB]], %[[C10]] : index
// CHECK:           acc.loop control(%[[INNER_IV:.*]] : index) = (%[[IV]] : index) to (%[[MIN_UB]] : index) step (%[[C1]] : index) {
// CHECK:             acc.yield
// CHECK:           } attributes {independent = [#acc.device_type<none>]}
// CHECK:           acc.yield
// CHECK:         } attributes {independent = [#acc.device_type<none>]}
func.func @single_loop_tile(%arg0: memref<10xf32>) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  acc.loop tile({%c2 : index}) control(%i : index) = (%c0 : index) to (%c10 : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<10xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// Test 2-level nested loop tiling with tile(4, 8)
// Creates: tile_loop_1 -> tile_loop_2 -> element_loop_1 -> element_loop_2

// CHECK-LABEL: func.func @nested_loop_tile
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C100:.*]] = arith.constant 100 : index
// CHECK-DAG:     %[[C50:.*]] = arith.constant 50 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C8:.*]] = arith.constant 8 : index
// Outer tile loop with gang
// CHECK:         acc.loop gang control(%[[I:.*]] : index) = (%[[C0]] : index) to (%[[C100]] : index) step (%[[C4]] : index) {
// Inner tile loop
// CHECK:           acc.loop control(%[[J:.*]] : index) = (%[[C0]] : index) to (%[[C50]] : index) step (%[[C8]] : index) {
// Outer element loop with vector
// CHECK:             acc.loop vector control({{.*}} : index) = (%[[I]] : index) to ({{.*}} : index) step (%[[C1]] : index) {
// Inner element loop
// CHECK:               acc.loop control({{.*}} : index) = (%[[J]] : index) to ({{.*}} : index) step (%[[C1]] : index) {
// CHECK:                 acc.yield
// CHECK:               }
// CHECK:               acc.yield
// CHECK:             }
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           acc.yield
// CHECK:         }
func.func @nested_loop_tile(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  acc.loop gang vector tile({%c4 : index, %c8 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// Test unknown tile size (*) represented as -1
// Should use default tile size (32)

// CHECK-LABEL: func.func @unknown_tile_size
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1000:.*]] = arith.constant 1000 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : index
// Tile loop with default tile size
// CHECK:         acc.loop control(%[[IV:.*]] : index) = (%[[C0]] : index) to (%[[C1000]] : index) step (%[[C32]] : index) {
// CHECK:           acc.loop control({{.*}} : index) = (%[[IV]] : index) to ({{.*}} : index) step (%[[C1]] : index) {
// CHECK:             acc.yield
// CHECK:           }
// CHECK:           acc.yield
// CHECK:         }
func.func @unknown_tile_size(%arg0: memref<1000xf32>) {
  %c0 = arith.constant 0 : index
  %c1000 = arith.constant 1000 : index
  %c1 = arith.constant 1 : index
  %cm1 = arith.constant -1 : i32  // tile(*) represented as -1
  acc.loop tile({%cm1 : i32}) control(%i : index) = (%c0 : index) to (%c1000 : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<1000xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}
