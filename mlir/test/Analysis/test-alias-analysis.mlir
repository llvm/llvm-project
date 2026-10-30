// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-alias-analysis))' -split-input-file -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "simple"
// CHECK-DAG: func.region0#0 <-> func.region0#1: MayAlias

// CHECK-DAG: alloca_1#0 <-> alloca_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> alloc_1#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloca_2#0 <-> alloc_1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloc_1#0 <-> alloc_2#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0#1: NoAlias

// CHECK-DAG: alloc_2#0 <-> func.region0#0: NoAlias
// CHECK-DAG: alloc_2#0 <-> func.region0#1: NoAlias
func.func @simple(%arg: memref<2xf32>, %arg1: memref<2xf32>) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>
  %3 = memref.alloc() {test.ptr = "alloc_2"} : memref<8x64xf32>
  return
}

// -----

// CHECK-LABEL: Testing : "control_flow"
// CHECK-DAG: alloca_1#0 <-> func.region0.block1#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> func.region0.block2#0: MustAlias

// CHECK-DAG: alloca_2#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: alloc_1#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0#1 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#1 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0.block1#0 <-> func.region0.block2#0: MustAlias
func.func @control_flow(%arg: memref<2xf32>, %cond: i1) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  cf.cond_br %cond, ^bb1(%0 : memref<8x64xf32>), ^bb2(%0 : memref<8x64xf32>)

^bb1(%arg1: memref<8x64xf32>):
  cf.br ^bb2(%arg1 : memref<8x64xf32>)

^bb2(%arg2: memref<8x64xf32>):
  return
}

// -----

// CHECK-LABEL: Testing : "control_flow_merge"
// CHECK-DAG: alloca_1#0 <-> func.region0.block1#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> func.region0.block2#0: MayAlias

// CHECK-DAG: alloca_2#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: alloc_1#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> func.region0.block2#0: MayAlias

// CHECK-DAG: func.region0#0 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#0 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0#1 <-> func.region0.block1#0: NoAlias
// CHECK-DAG: func.region0#1 <-> func.region0.block2#0: NoAlias

// CHECK-DAG: func.region0.block1#0 <-> func.region0.block2#0: MayAlias
func.func @control_flow_merge(%arg: memref<2xf32>, %cond: i1) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  cf.cond_br %cond, ^bb1(%0 : memref<8x64xf32>), ^bb2(%2 : memref<8x64xf32>)

^bb1(%arg1: memref<8x64xf32>):
  cf.br ^bb2(%arg1 : memref<8x64xf32>)

^bb2(%arg2: memref<8x64xf32>):
  return
}

// -----

// CHECK-LABEL: Testing : "region_control_flow"
// CHECK-DAG: alloca_1#0 <-> if_alloca#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> if_alloc#0: NoAlias

// CHECK-DAG: alloca_2#0 <-> if_alloca#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> if_alloc#0: NoAlias

// CHECK-DAG: alloc_1#0 <-> if_alloca#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> if_alloca_merge#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> if_alloc#0: MustAlias

// CHECK-DAG: if_alloca#0 <-> if_alloca_merge#0: MayAlias
// CHECK-DAG: if_alloca#0 <-> if_alloc#0: NoAlias
// CHECK-DAG: if_alloca#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloca#0 <-> func.region0#1: NoAlias

// CHECK-DAG: if_alloca_merge#0 <-> if_alloc#0: NoAlias
// CHECK-DAG: if_alloca_merge#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloca_merge#0 <-> func.region0#1: NoAlias

// CHECK-DAG: if_alloc#0 <-> func.region0#0: NoAlias
// CHECK-DAG: if_alloc#0 <-> func.region0#1: NoAlias
func.func @region_control_flow(%arg: memref<2xf32>, %cond: i1) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %3 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %0 : memref<8x64xf32>
  } else {
    scf.yield %0 : memref<8x64xf32>
  } {test.ptr = "if_alloca"}

  %4 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %0 : memref<8x64xf32>
  } else {
    scf.yield %1 : memref<8x64xf32>
  } {test.ptr = "if_alloca_merge"}

  %5 = scf.if %cond -> (memref<8x64xf32>) {
    scf.yield %2 : memref<8x64xf32>
  } else {
    scf.yield %2 : memref<8x64xf32>
  } {test.ptr = "if_alloc"}
  return
}

// -----

// CHECK-LABEL: Testing : "region_loop_control_flow"
// CHECK-DAG: alloca_1#0 <-> for_alloca#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#1: MustAlias

// CHECK-DAG: alloca_2#0 <-> for_alloca#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#1: NoAlias

// CHECK-DAG: alloc_1#0 <-> for_alloca#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloc_1#0 <-> for_alloca.region0#1: NoAlias

// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#1: MustAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#0: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#1: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#2: NoAlias
// CHECK-DAG: for_alloca#0 <-> func.region0#3: NoAlias

// CHECK-DAG: for_alloca.region0#0 <-> for_alloca.region0#1: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#0: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#1: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#2: MayAlias
// CHECK-DAG: for_alloca.region0#0 <-> func.region0#3: MayAlias

// CHECK-DAG: for_alloca.region0#1 <-> func.region0#0: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#1: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#2: NoAlias
// CHECK-DAG: for_alloca.region0#1 <-> func.region0#3: NoAlias
func.func @region_loop_control_flow(%arg: memref<2xf32>, %loopI0 : index,
                               %loopI1 : index, %loopI2 : index) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8x64xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8x64xf32>
  %2 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %result = scf.for %i0 = %loopI0 to %loopI1 step %loopI2 iter_args(%si = %0) -> (memref<8x64xf32>) {
    scf.yield %si : memref<8x64xf32>
  } {test.ptr = "for_alloca"}
  return
}

// -----

// CHECK-LABEL: Testing : "region_loop_zero_trip_count"
// CHECK-DAG: alloca_1#0 <-> alloca_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> for_alloca.region0#1: MayAlias

// CHECK-DAG: alloca_2#0 <-> for_alloca#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> for_alloca.region0#1: MayAlias

// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#0: MayAlias
// CHECK-DAG: for_alloca#0 <-> for_alloca.region0#1: MayAlias

// CHECK-DAG: for_alloca.region0#0 <-> for_alloca.region0#1: MayAlias
func.func @region_loop_zero_trip_count() attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<i32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<i32>
  %result = affine.for %i = 0 to 0 iter_args(%si = %0) -> (memref<i32>) {
    affine.yield %si : memref<i32>
  } {test.ptr = "for_alloca"}
  return
}

// -----

// CHECK-LABEL: Testing : "view_like"
// CHECK-DAG: alloc_1#0 <-> view#0: NoAlias

// CHECK-DAG: alloca_1#0 <-> view#0: MustAlias

// CHECK-DAG: view#0 <-> func.region0#0: NoAlias
// CHECK-DAG: view#0 <-> func.region0#1: NoAlias
func.func @view_like(%arg: memref<2xf32>, %size: index) attributes {test.ptr = "func"} {
  %1 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %c0 = arith.constant 0 : index
  %2 = memref.alloca (%size) {test.ptr = "alloca_1"} : memref<?xi8>
  %3 = memref.view %2[%c0][] {test.ptr = "view"} : memref<?xi8> to memref<8x64xf32>
  return
}

// -----

// Two addresses traced back to the same value only must alias if no view on
// the way may have moved them away from it.

// CHECK-LABEL: Testing : "view_like_offset"
// CHECK-DAG: alloca_1#0 <-> alloca_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> view_1#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> view_2#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> view_3#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> zero_view#0: MustAlias
// CHECK-DAG: alloca_2#0 <-> view_1#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> view_2#0: NoAlias
// CHECK-DAG: alloca_2#0 <-> view_3#0: MayAlias
// CHECK-DAG: alloca_2#0 <-> zero_view#0: NoAlias
// CHECK-DAG: view_1#0 <-> view_2#0: MayAlias
// CHECK-DAG: view_1#0 <-> view_3#0: NoAlias
// CHECK-DAG: view_1#0 <-> zero_view#0: MayAlias
// CHECK-DAG: view_2#0 <-> view_3#0: NoAlias
// CHECK-DAG: view_2#0 <-> zero_view#0: MayAlias
// CHECK-DAG: view_3#0 <-> zero_view#0: NoAlias
func.func @view_like_offset() {
  %0 = memref.alloca() {test.ptr = "alloca_1"} : memref<8xf32>
  %1 = memref.alloca() {test.ptr = "alloca_2"} : memref<8xf32>
  %2 = test.offset_view %0 {test.ptr = "view_1"} : memref<8xf32> -> memref<4xf32>
  %3 = test.offset_view %0 {test.ptr = "view_2"} : memref<8xf32> -> memref<4xf32>
  %4 = test.offset_view %1 {test.ptr = "view_3"} : memref<8xf32> -> memref<4xf32>
  %5 = test.offset_view %0 zero_offset {test.ptr = "zero_view"} : memref<8xf32> -> memref<4xf32>
  return
}

// -----

// GEPs keep MustAlias with their base only when all indices are zero.

// CHECK-LABEL: Testing : "gep_offset"
// CHECK-DAG: alloca_1#0 <-> alloca_2#0: NoAlias
// CHECK-DAG: alloca_1#0 <-> gep_zero#0: MustAlias
// CHECK-DAG: alloca_1#0 <-> gep_one#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> gep_dyn#0: MayAlias
// CHECK-DAG: alloca_1#0 <-> gep_other#0: NoAlias
// CHECK-DAG: gep_zero#0 <-> gep_one#0: MayAlias
// CHECK-DAG: gep_one#0 <-> gep_dyn#0: MayAlias
// CHECK-DAG: gep_one#0 <-> gep_other#0: NoAlias
func.func @gep_offset(%idx: i64) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %0 = llvm.alloca %c1 x !llvm.array<4 x i32> {test.ptr = "alloca_1"} : (i32) -> !llvm.ptr
  %1 = llvm.alloca %c1 x !llvm.array<4 x i32> {test.ptr = "alloca_2"} : (i32) -> !llvm.ptr
  %2 = llvm.getelementptr %0[0, 0] {test.ptr = "gep_zero"} : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
  %3 = llvm.getelementptr %0[0, 1] {test.ptr = "gep_one"} : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
  %4 = llvm.getelementptr %0[%idx] {test.ptr = "gep_dyn"} : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %5 = llvm.getelementptr %1[0, 1] {test.ptr = "gep_other"} : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
  return
}

// -----

// CHECK-LABEL: Testing : "constants"
// CHECK-DAG: alloc_1#0 <-> constant_1#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> constant_2#0: NoAlias
// CHECK-DAG: alloc_1#0 <-> constant_3#0: NoAlias

// CHECK-DAG: constant_1#0 <-> constant_2#0: MayAlias
// CHECK-DAG: constant_1#0 <-> constant_3#0: MayAlias
// CHECK-DAG: constant_1#0 <-> func.region0#0: MayAlias

// CHECK-DAG: constant_2#0 <-> constant_3#0: MayAlias
// CHECK-DAG: constant_2#0 <-> func.region0#0: MayAlias

// CHECK-DAG: constant_3#0 <-> func.region0#0: MayAlias
func.func @constants(%arg: memref<2xf32>) attributes {test.ptr = "func"} {
  %1 = memref.alloc() {test.ptr = "alloc_1"} : memref<8x64xf32>

  %c0 = arith.constant {test.ptr = "constant_1"} 0 : index
  %c0_2 = arith.constant {test.ptr = "constant_2"} 0 : index
  %c1 = arith.constant {test.ptr = "constant_3"} 1 : index

  return
}

// -----

// CHECK-LABEL: Testing : "distinct_objects"
// CHECK-DAG: func.region0#0 <-> func.region0#1: MayAlias

// CHECK-DAG: distinct#0 <-> distinct#1: NoAlias
// CHECK-DAG: distinct#0 <-> func.region0#0: MustAlias
// CHECK-DAG: distinct#1 <-> func.region0#0: MayAlias
// CHECK-DAG: distinct#0 <-> func.region0#1: MayAlias
// CHECK-DAG: distinct#1 <-> func.region0#1: MustAlias

func.func @distinct_objects(%arg: memref<?xf32>, %arg1: memref<?xf32>) attributes {test.ptr = "func"} {
  %0, %1 = memref.distinct_objects %arg, %arg1 {test.ptr = "distinct"} : memref<?xf32>, memref<?xf32>
  return
}
