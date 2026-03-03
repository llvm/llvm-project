// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

acc.firstprivate.recipe @firstprivatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}
acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

// Verify that the firstprivate was materialized into a copy outside the kernel
// and an alloca (as per the recipe) inside the region.
// Then ensure that all uses are of the private alloca.
// CHECK-LABEL: func.func @firstpriv
// CHECK: acc.parallel {
// CHECK: %[[ALLOCA:.*]] = memref.alloca() {acc.var_name = #acc.var_name<"t">} : memref<i32>
// CHECK: %[[FIRSTPRIVLOAD:.*]] = memref.load %{{.*}}[] : memref<i32>
// CHECK: memref.store %[[FIRSTPRIVLOAD]], %[[ALLOCA]][] : memref<i32>
// CHECK: %[[ALLOCALOAD:.*]] = memref.load %[[ALLOCA]][] : memref<i32>
// CHECK: %[[ADDI:.*]] = arith.addi %[[ALLOCALOAD]], %c1{{.*}} : i32
// CHECK: memref.store %[[ADDI]], %[[ALLOCA]][] : memref<i32>

func.func @firstpriv() {
  %c1336 = arith.constant 1336 : i32
  %alloc = memref.alloca() : memref<i32>
  memref.store %c1336, %alloc[] : memref<i32>
  %fp = acc.firstprivate varPtr(%alloc : memref<i32>) recipe(@firstprivatization_memref_i32) -> memref<i32> {implicit = true, name = "t"}
  acc.parallel firstprivate(%fp : memref<i32>) {
    %c1 = arith.constant 1 : i32
    %v = memref.load %fp[] : memref<i32>
    %add = arith.addi %v, %c1 : i32
    memref.store %add, %fp[] : memref<i32>
    acc.yield
  }
  return
}
