// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s

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
} destroy {
^bb0(%arg0: memref<f64>, %arg1: memref<f64>):
  memref.dealloc %arg1 : memref<f64>
  acc.terminator
}

// Verify that the reduction init and combine recipes attached to compute
// ops materialize within the region
// CHECK-LABEL: func.func @par_reduction_clause_
// CHECK:       acc.parallel {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init {{.*}} <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:  memref.store [[ZERO]], [[ALLOCA]][]
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.par_dims = #acc<par_dims[block_x]>, acc.var_name = #acc.var_name<"tmp">
// CHECK:       memref.load [[PRIVATE]][]
// CHECK:       memref.store {{.*}}, [[PRIVATE]][]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into [[REDUCVAR:%.*]] : memref<f64> {
// CHECK:       [[LOADVAR:%.*]] = memref.load [[REDUCVAR]][]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = memref.load [[PRIVATE]][]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  memref.store [[COMBINE]], [[REDUCVAR]][]
// CHECK-NEXT:  } {acc.par_dims = #acc<par_dims[block_x]>}
// CHECK-NEXT:  memref.dealloc [[PRIVATE]] : memref<f64>
// CHECK:       acc.yield

func.func @par_reduction_clause_(%arg0: memref<f64>) {
  %cst = arith.constant 1.000000e+00 : f64
  %red = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) -> memref<f64> {name = "tmp"}
  acc.parallel reduction(%red : memref<f64>) {
    %3 = memref.load %red[] : memref<f64>
    %4 = arith.addf %3, %cst fastmath<contract> : f64
    memref.store %4, %red[] : memref<f64>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @par_reduction_clause_serial
// CHECK:       acc.parallel {{.*}} {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init {{.*}} <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:  memref.store [[ZERO]], [[ALLOCA]][]
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.par_dims = #acc<par_dims[sequential]>, acc.var_name = #acc.var_name<"tmp">
// CHECK:       memref.load [[PRIVATE]][]
// CHECK:       memref.store {{.*}}, [[PRIVATE]][]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into [[REDUCVAR:%.*]] : memref<f64> {
// CHECK:       [[LOADVAR:%.*]] = memref.load [[REDUCVAR]][]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = memref.load [[PRIVATE]][]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  memref.store [[COMBINE]], [[REDUCVAR]][]
// CHECK-NEXT:  } {acc.par_dims = #acc<par_dims[sequential]>}
// CHECK-NEXT:  memref.dealloc [[PRIVATE]] : memref<f64>
// CHECK:       acc.yield

func.func @par_reduction_clause_serial(%arg0: memref<f64>) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f64
  %red = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) -> memref<f64> {name = "tmp"}
  acc.parallel num_gangs({%c1_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) reduction(%red : memref<f64>) {
    %3 = memref.load %red[] : memref<f64>
    %4 = arith.addf %3, %cst fastmath<contract> : f64
    memref.store %4, %red[] : memref<f64>
    acc.yield
  }
  return
}
