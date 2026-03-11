// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

// acc.kernels with private: recipe materialized to alloca inside region
// CHECK-NOT: acc.private
// CHECK: acc.kernels dataOperands(
// CHECK: memref.alloca() {acc.var_name = #acc.var_name<"s">} : memref<i32>

acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

func.func @kpriv_(%arg0: memref<i32>, %arg1: memref<32xi32>) {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %iv_alloc = memref.alloca() : memref<i32>
  %start = memref.load %arg0[] : memref<i32>
  memref.store %start, %iv_alloc[] : memref<i32>
  %copy = acc.copyin varPtr(%arg1 : memref<32xi32>) -> memref<32xi32> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "a"}
  %priv = acc.private varPtr(%iv_alloc : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32> {implicit = true, name = "s"}
  acc.kernels dataOperands(%copy : memref<32xi32>) private(%priv : memref<i32>) {
    acc.loop control(%arg2 : index) = (%c1 : index) to (%c32 : index) step (%c1 : index) {
      %iv = arith.index_cast %arg2 : index to i32
      memref.store %iv, %iv_alloc[] : memref<i32>
      %s_val = memref.load %priv[] : memref<i32>
      memref.store %s_val, %copy[%arg2] : memref<32xi32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.copyout accPtr(%copy : memref<32xi32>) to varPtr(%arg1 : memref<32xi32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "a"}
  return
}
