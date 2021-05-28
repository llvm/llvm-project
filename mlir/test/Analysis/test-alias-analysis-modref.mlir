// RUN: mlir-opt %s -pass-pipeline='func(test-alias-analysis-modref)' -split-input-file -allow-unregistered-dialect 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "no_side_effects"
// CHECK: alloc -> func.region0#0: NoModRef
// CHECK: dealloc -> func.region0#0: NoModRef
// CHECK: return -> func.region0#0: NoModRef
func @no_side_effects(%arg: memref<2xf32>) attributes {test.ptr = "func"} {
  %1 = memref.alloc() {test.ptr = "alloc"} : memref<8x64xf32>
  memref.dealloc %1 {test.ptr = "dealloc"} : memref<8x64xf32>
  return {test.ptr = "return"}
}

// -----

// CHECK-LABEL: Testing : "simple"
// CHECK-DAG: store -> alloc#0: Mod
// CHECK-DAG: load -> alloc#0: Ref

// CHECK-DAG: store -> func.region0#0: NoModRef
// CHECK-DAG: load -> func.region0#0: NoModRef
func @simple(%arg: memref<i32>, %value: i32) attributes {test.ptr = "func"} {
  %1 = memref.alloca() {test.ptr = "alloc"} : memref<i32>
  memref.store %value, %1[] {test.ptr = "store"} : memref<i32>
  %2 = memref.load %1[] {test.ptr = "load"} : memref<i32>
  return {test.ptr = "return"}
}

// -----

// CHECK-LABEL: Testing : "mayalias"
// CHECK-DAG: store -> func.region0#0: Mod
// CHECK-DAG: load -> func.region0#0: Ref

// CHECK-DAG: store -> func.region0#1: Mod
// CHECK-DAG: load -> func.region0#1: Ref
func @mayalias(%arg0: memref<i32>, %arg1: memref<i32>, %value: i32) attributes {test.ptr = "func"} {
  memref.store %value, %arg1[] {test.ptr = "store"} : memref<i32>
  %1 = memref.load %arg1[] {test.ptr = "load"} : memref<i32>
  return {test.ptr = "return"}
}

// -----

// CHECK-LABEL: Testing : "recursive"
// CHECK-DAG: if -> func.region0#0: ModRef
// CHECK-DAG: if -> func.region0#1: ModRef

// TODO: This is provably NoModRef, but requires handling recursive side
// effects.
// CHECK-DAG: if -> alloc#0: ModRef
func @recursive(%arg0: memref<i32>, %arg1: memref<i32>, %cond: i1, %value: i32) attributes {test.ptr = "func"} {
  %0 = memref.alloca() {test.ptr = "alloc"} : memref<i32>
  scf.if %cond {
    memref.store %value, %arg0[] : memref<i32>
    %1 = memref.load %arg0[] : memref<i32>
  } {test.ptr = "if"}
  return {test.ptr = "return"}
}

// -----

// CHECK-LABEL: Testing : "unknown"
// CHECK-DAG: unknown -> func.region0#0: ModRef
func @unknown(%arg0: memref<i32>) attributes {test.ptr = "func"} {
  "foo.op"() {test.ptr = "unknown"} : () -> ()
  return
}
