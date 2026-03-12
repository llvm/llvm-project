// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

// Test that the reduction recipes are correctly inlined when attached to a
// parallel construct without loop. Verify init and combine materialize in the region.
// CHECK-LABEL: func.func @par_reduction_clause_
// CHECK:       acc.parallel {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init {{.*}} <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:  memref.store [[ZERO]], [[ALLOCA]][]
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.var_name = #acc.var_name<"tmp">
// CHECK:       memref.load [[PRIVATE]][]
// CHECK:       memref.store {{.*}}, [[PRIVATE]][]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into [[REDUCVAR:%.*]] :
// CHECK:       [[LOADVAR:%.*]] = memref.load [[REDUCVAR]][]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = memref.load [[PRIVATE]][]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  memref.store [[COMBINE]], [[REDUCVAR]][]
// CHECK:       acc.yield

acc.reduction.recipe @reduction_add_memref_f64 : memref<f64> reduction_operator <add> init {
^bb0(%arg0: memref<f64>):
  %cst = arith.constant 0.000000e+00 : f64
  %0 = memref.alloca() : memref<f64>
  memref.store %cst, %0[] : memref<f64>
  acc.yield %0 : memref<f64>
} combiner {
^bb0(%arg0: memref<f64>, %arg1: memref<f64>):
  %0 = memref.load %arg0[] : memref<f64>
  %1 = memref.load %arg1[] : memref<f64>
  %2 = arith.addf %0, %1 fastmath<contract> : f64
  memref.store %2, %arg0[] : memref<f64>
  acc.yield %arg0 : memref<f64>
}
func.func @par_reduction_clause_(%arg0: memref<f64>) {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) -> memref<f64> {name = "tmp"}
  acc.parallel reduction(%0 : memref<f64>) {
    %1 = memref.load %0[] : memref<f64>
    %2 = arith.addf %1, %cst fastmath<contract> : f64
    memref.store %2, %0[] : memref<f64>
    acc.yield
  }
  return
}
