// RUN: mlir-opt %s -acc-recipe-materialization | FileCheck %s
// RUN: mlir-opt %s -acc-recipe-materialization -acc-compute-lowering | FileCheck %s --check-prefix=LOWER

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
// CHECK-SAME:  (%[[HOST:.*]]: memref<f64>)
// CHECK:       %[[MAPPED:.*]] = acc.copyin varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")
// CHECK:       acc.parallel dataOperands(%[[MAPPED]] : memref<f64>) {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init %[[MAPPED]] <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:  memref.store [[ZERO]], [[ALLOCA]][]
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.par_dims = #acc<par_dims[block_x]>, acc.var_name = #acc.var_name<"tmp">
// CHECK:       memref.load [[PRIVATE]][]
// CHECK:       memref.store {{.*}}, [[PRIVATE]][]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into %[[MAPPED]] : memref<f64> {
// CHECK:       [[LOADVAR:%.*]] = memref.load %[[MAPPED]][]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = memref.load [[PRIVATE]][]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  memref.store [[COMBINE]], %[[MAPPED]][]
// CHECK-NEXT:  } {acc.par_dims = #acc<par_dims[block_x]>}
// CHECK-NEXT:  memref.dealloc [[PRIVATE]] : memref<f64>
// CHECK:       acc.yield
// CHECK:       acc.copyout accPtr(%[[MAPPED]] : memref<f64>) to varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")

// LOWER-LABEL: func.func @par_reduction_clause_
// LOWER-SAME:  (%[[HOST:.*]]: memref<f64>)
// LOWER:       %[[MAPPED:.*]] = acc.copyin varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")
// LOWER:       acc.kernel_environment dataOperands(%[[MAPPED]] : memref<f64>) {
// LOWER:       acc.compute_region ins(%[[SHARED:.*]] = %[[MAPPED]]) : (memref<f64>) {
// LOWER:       acc.reduction_init %[[SHARED]] <add>
// LOWER:       acc.reduction_combine_region {{.*}} into %[[SHARED]] : memref<f64>
// LOWER:       acc.copyout accPtr(%[[MAPPED]] : memref<f64>) to varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")

func.func @par_reduction_clause_(%arg0: memref<f64>) {
  %cst = arith.constant 1.000000e+00 : f64
  %red = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) name("tmp") -> memref<f64>
  acc.parallel reduction(%red : memref<f64>) {
    %3 = memref.load %red[] : memref<f64>
    %4 = arith.addf %3, %cst fastmath<contract> : f64
    memref.store %4, %red[] : memref<f64>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @par_reduction_clause_serial
// CHECK-SAME:  (%[[HOST:.*]]: memref<f64>)
// CHECK:       %[[MAPPED:.*]] = acc.copyin varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")
// CHECK:       acc.parallel dataOperands(%[[MAPPED]] : memref<f64>) {{.*}} {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init %[[MAPPED]] <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = memref.alloca() : memref<f64>
// CHECK-NEXT:  memref.store [[ZERO]], [[ALLOCA]][]
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.par_dims = #acc<par_dims[sequential]>, acc.var_name = #acc.var_name<"tmp">
// CHECK:       memref.load [[PRIVATE]][]
// CHECK:       memref.store {{.*}}, [[PRIVATE]][]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into %[[MAPPED]] : memref<f64> {
// CHECK:       [[LOADVAR:%.*]] = memref.load %[[MAPPED]][]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = memref.load [[PRIVATE]][]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  memref.store [[COMBINE]], %[[MAPPED]][]
// CHECK-NEXT:  } {acc.par_dims = #acc<par_dims[sequential]>}
// CHECK-NEXT:  memref.dealloc [[PRIVATE]] : memref<f64>
// CHECK:       acc.yield
// CHECK:       acc.copyout accPtr(%[[MAPPED]] : memref<f64>) to varPtr(%[[HOST]] : memref<f64>) dataClause(acc_reduction) implicit(true) name("tmp")

func.func @par_reduction_clause_serial(%arg0: memref<f64>) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f64
  %red = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) name("tmp") -> memref<f64>
  acc.parallel num_gangs({%c1_i32 : i32}) num_workers(%c1_i32 : i32) vector_length(%c1_i32 : i32) reduction(%red : memref<f64>) {
    %3 = memref.load %red[] : memref<f64>
    %4 = arith.addf %3, %cst fastmath<contract> : f64
    memref.store %4, %red[] : memref<f64>
    acc.yield
  }
  return
}

// A reduction whose original variable is already mapped must reuse that
// mapping rather than create another copy pair.
// CHECK-LABEL: func.func @par_reduction_already_mapped
// CHECK:       %[[MAPPED:.*]] = acc.copyin
// CHECK-NOT:   acc.copyin
// CHECK:       acc.parallel dataOperands(%[[MAPPED]] : memref<f64>) {
// CHECK:       acc.reduction_init %[[MAPPED]] <add>
// CHECK:       acc.reduction_combine_region {{.*}} into %[[MAPPED]] : memref<f64>
// CHECK:       acc.copyout accPtr(%[[MAPPED]] : memref<f64>)

func.func @par_reduction_already_mapped(%arg0: memref<f64>) {
  %cst = arith.constant 1.000000e+00 : f64
  %mapped = acc.copyin varPtr(%arg0 : memref<f64>) dataClause(acc_copy) name("tmp") -> memref<f64>
  %red = acc.reduction varPtr(%arg0 : memref<f64>) recipe(@reduction_add_memref_f64) name("tmp") -> memref<f64>
  acc.parallel dataOperands(%mapped : memref<f64>) reduction(%red : memref<f64>) {
    memref.store %cst, %red[] : memref<f64>
    acc.yield
  }
  acc.copyout accPtr(%mapped : memref<f64>) to varPtr(%arg0 : memref<f64>) dataClause(acc_copy) name("tmp")
  return
}

// A nested loop reduction can reduce an outer private value. It must not
// create a host mapping around the compute construct.
// CHECK-LABEL: func.func @loop_reduction_private
// CHECK-NOT:   acc.copyin
// CHECK:       acc.parallel {
// CHECK:       %[[SHARED:.*]] = memref.alloca() : memref<f64>
// CHECK:       %[[PRIVATE:.*]] = acc.reduction_init %[[SHARED]] <add>
// CHECK:       acc.reduction_combine_region %[[PRIVATE]] into %[[SHARED]]
// CHECK-NOT:   acc.copyout

func.func @loop_reduction_private() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : f64
  acc.parallel {
    %shared = memref.alloca() : memref<f64>
    %red = acc.reduction varPtr(%shared : memref<f64>) recipe(@reduction_add_memref_f64) name("tmp") -> memref<f64>
    acc.loop reduction(%red : memref<f64>) control(%iv : index) = (%c0 : index) to (%c1 : index) step (%c1 : index) {
      memref.store %cst, %red[] : memref<f64>
      acc.yield
    } inclusiveUpperbound(array<i1: false>) independent
    acc.yield
  }
  return
}
