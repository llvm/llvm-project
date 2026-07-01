// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

// A recipe with two allocations. Tests if both allocations correctly get
// the `var_name` attribute applied after materialization.

acc.firstprivate.recipe @firstprivatization_two_allocs : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %c42 = arith.constant 42 : i32
  %0 = memref.alloca() {acc.var_name = #acc.var_name<"<acc.varname.placeholder>">} : memref<i32>
  memref.store %c42, %0[] : memref<i32>
  %1 = memref.alloca() {acc.var_name = #acc.var_name<"<acc.varname.placeholder>">} : memref<i32>
  %v = memref.load %0[] : memref<i32>
  memref.store %v, %1[] : memref<i32>
  acc.yield %1 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}

// CHECK-LABEL: func.func @firstpriv_two_allocs
// CHECK: acc.parallel {
// CHECK: %[[ALLOC0:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"t">} : memref<i32>
// CHECK: %[[ALLOC1:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"t">} : memref<i32>
// CHECK-NOT: acc.varname.placeholder

func.func @firstpriv_two_allocs() {
  %alloc = memref.alloca() : memref<i32>
  %fp = acc.firstprivate varPtr(%alloc : memref<i32>) recipe(@firstprivatization_two_allocs) -> memref<i32> {name = "t"}
  acc.parallel firstprivate(%fp : memref<i32>) {
    acc.yield
  }
  return
}

// When name is empty, `var_name` should be stripped away post materialization
// CHECK-LABEL: func.func @firstpriv_two_allocs_no_name
// CHECK: acc.parallel {
// CHECK-NOT: acc.var_name
// CHECK: memref.alloca()
// CHECK-NOT: acc.var_name
// CHECK: memref.alloca()

func.func @firstpriv_two_allocs_no_name() {
  %alloc = memref.alloca() : memref<i32>
  %fp = acc.firstprivate varPtr(%alloc : memref<i32>) recipe(@firstprivatization_two_allocs) -> memref<i32> {name = ""}
  acc.parallel firstprivate(%fp : memref<i32>) {
    acc.yield
  }
  return
}
