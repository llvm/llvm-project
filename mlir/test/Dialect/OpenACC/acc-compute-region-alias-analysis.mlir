// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-alias-analysis))' -split-input-file 2>&1 | FileCheck %s --implicit-check-not='{{(inner|loop)_alloca#0 <-> .*: MayAlias}}' --implicit-check-not='{{<-> (inner|loop)_alloca#0: MayAlias}}'

// acc.compute_region is an automatic allocation scope: allocas created inside
// its body cannot alias the values captured through `ins`, because those are
// defined before the body runs. Without the scope, the allocation scope of
// such an alloca is the enclosing function, which cannot be related to the
// region's entry block arguments, and the answer degrades to MayAlias. The
// implicit checks in the RUN line make sure no query involving an alloca
// created inside the region ever answers MayAlias.

// CHECK-LABEL: Testing : "alloca_vs_captured_arg"
// CHECK-DAG: region.region0#0 <-> region.region0#1: MayAlias
// CHECK-DAG: inner_alloca#0 <-> region.region0#0: NoAlias
// CHECK-DAG: inner_alloca#0 <-> region.region0#1: NoAlias
func.func @alloca_vs_captured_arg(%a: memref<i32>, %b: memref<i32>) {
  acc.compute_region ins(%x = %a, %y = %b) : (memref<i32>, memref<i32>) {
    %0 = memref.alloca() {test.ptr = "inner_alloca"} : memref<i32>
    acc.yield
  } <{origin = "acc.parallel"}> {test.ptr = "region"}
  return
}

// -----

// An alloca outside of the region that is captured by it is seen through a
// block argument. That relationship is not tracked, so it stays conservative,
// while an alloca created inside the region is still distinct from both.

// CHECK-LABEL: Testing : "outer_alloca_captured"
// CHECK-DAG: outer_alloca#0 <-> region.region0#0: MayAlias
// CHECK-DAG: outer_alloca#0 <-> inner_alloca#0: NoAlias
// CHECK-DAG: inner_alloca#0 <-> region.region0#0: NoAlias
func.func @outer_alloca_captured() {
  %0 = memref.alloca() {test.ptr = "outer_alloca"} : memref<i32>
  acc.compute_region ins(%x = %0) : (memref<i32>) {
    %1 = memref.alloca() {test.ptr = "inner_alloca"} : memref<i32>
    acc.yield
  } <{origin = "acc.parallel"}> {test.ptr = "region"}
  return
}

// -----

// An alloca nested in a loop inside the region is bounded by the loop, and is
// likewise distinct from the captured value.

// CHECK-LABEL: Testing : "alloca_in_nested_loop"
// CHECK-DAG: inner_alloca#0 <-> loop_alloca#0: NoAlias
// CHECK-DAG: inner_alloca#0 <-> region.region0#0: NoAlias
// CHECK-DAG: loop_alloca#0 <-> region.region0#0: NoAlias
func.func @alloca_in_nested_loop(%a: memref<i32>) {
  acc.compute_region ins(%x = %a) : (memref<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %0 = memref.alloca() {test.ptr = "inner_alloca"} : memref<i32>
    scf.parallel (%i) = (%c0) to (%c8) step (%c1) {
      %1 = memref.alloca() {test.ptr = "loop_alloca"} : memref<i32>
      scf.reduce
    }
    acc.yield
  } <{origin = "acc.parallel"}> {test.ptr = "region"}
  return
}
