// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

acc.private.recipe @privatization_memref_i64 : memref<i64> init {
^bb0(%arg0: memref<i64>):
  %0 = memref.alloca() : memref<i64>
  acc.yield %0 : memref<i64>
}

// CHECK-LABEL: func.func @private_i64
// CHECK: acc.loop control([[IV:%.+]] : index)
// CHECK: [[ALLOC:%.+]] = memref.alloca() : memref<i64>
// CHECK: memref.store {{.*}}, [[ALLOC]][]

func.func @private_i64(%arg0 : memref<i64>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %priv = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = ""}
  acc.loop private(%priv : memref<i64>) control(%siv : index) = (%c1 : index) to (%c16 : index) step (%c1 : index) {
    %iv_i64 = arith.index_cast %siv : index to i64
    memref.store %iv_i64, %priv[] : memref<i64>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// CHECK-LABEL: func.func @par_private_i64
// CHECK: acc.parallel {
// CHECK: [[ALLOC:%.+]] = memref.alloca() : memref<i64>
// CHECK: acc.loop control([[IV:%.+]] : index)
// CHECK: memref.store {{.*}}, [[ALLOC]][]

func.func @par_private_i64(%arg0 : memref<i64>) {
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %priv = acc.private varPtr(%arg0 : memref<i64>) recipe(@privatization_memref_i64) -> memref<i64> {implicit = true, name = ""}
  acc.parallel private(%priv : memref<i64>) {
    acc.loop control(%siv : index) = (%c1 : index) to (%c16 : index) step (%c1 : index) {
      %iv_i64 = arith.index_cast %siv : index to i64
      memref.store %iv_i64, %priv[] : memref<i64>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}
