// RUN: mlir-opt %s -acc-loop-tiling --remarks-filter="(open)?acc.*" 2>&1 | FileCheck %s

// Test that the pass emits remarks for loop tiling

// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=single_loop_remark | Remark="Tiling 1-level loop nest with tile(4)"
func.func @single_loop_remark(%arg0: memref<100xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  acc.loop tile({%c4 : index}) control(%i : index) = (%c0 : index) to (%c100 : index) step (%c1 : index) {
    %val = arith.index_castui %i : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i] : memref<100xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=nested_loop_remark | Remark="Tiling 2-level loop nest with tile(8,16)"
func.func @nested_loop_remark(%arg0: memref<100x50xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  acc.loop tile({%c8 : index, %c16 : index}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c50 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x50xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// Test remark for unknown tile size (*) represented as -1
// Should use default tile size

// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=unknown_tile_remark | Remark="Tiling 1-level loop nest with tile(*)"
// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=unknown_tile_remark | Remark="Picking default tile size {{[0-9]+}} for unknown tile size '*'"
func.func @unknown_tile_remark(%arg0: memref<1000xf32>) {
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

// Test remark for multiple unknown tile sizes

// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=multiple_unknown_tiles_remark | Remark="Tiling 2-level loop nest with tile(*,*)"
// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=multiple_unknown_tiles_remark | Remark="Picking default tile size {{[0-9]+}} for unknown tile size '*'"
// CHECK: remark: [Passed] openacc | Category:acc-loop-tile | Function=multiple_unknown_tiles_remark | Remark="Picking default tile size {{[0-9]+}} for unknown tile size '*'"
func.func @multiple_unknown_tiles_remark(%arg0: memref<100x100xf32>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  %cm1 = arith.constant -1 : i32  // tile(*) represented as -1
  acc.loop tile({%cm1 : i32, %cm1 : i32}) control(%i : index, %j : index) = (%c0, %c0 : index, index) to (%c100, %c100 : index, index) step (%c1, %c1 : index, index) {
    %sum = arith.addi %i, %j : index
    %val = arith.index_castui %sum : index to i32
    %fval = arith.sitofp %val : i32 to f32
    memref.store %fval, %arg0[%i, %j] : memref<100x100xf32>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}
