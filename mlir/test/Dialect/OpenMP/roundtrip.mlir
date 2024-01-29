// RUN: fir-opt -verify-diagnostics %s | fir-opt | FileCheck %s

// CHECK-LABEL: _QPprivate_clause
func.func @_QPprivate_clause() {
  %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFprivate_clause_allocatableEx"}
  %1 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFprivate_clause_allocatableEy"}

  // CHECK: omp.parallel private(@x.privatizer %0, @y.privatizer %1 : !fir.ref<i32>, !fir.ref<i32>)
  omp.parallel private(@x.privatizer %0, @y.privatizer %1: !fir.ref<i32>, !fir.ref<i32>) {
    omp.terminator
  }
  return
}

// CHECK: "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "x.privatizer"}> ({
"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "x.privatizer"}> ({
// CHECK: ^bb0(%arg0: {{.*}}):
^bb0(%arg0: !fir.ref<i32>):

  // CHECK: %0 = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFprivate_clause_allocatableEx"}
  %0 = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFprivate_clause_allocatableEx"}

  // CHECK: omp.yield(%0 : !fir.ref<i32>)
  omp.yield(%0 : !fir.ref<i32>)
}) : () -> ()

// CHECK: "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "y.privatizer"}> ({
"omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "y.privatizer"}> ({
^bb0(%arg0: !fir.ref<i32>):

  // CHECK: %0 = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFprivate_clause_allocatableEy"}
  %0 = fir.alloca i32 {bindc_name = "y", pinned, uniq_name = "_QFprivate_clause_allocatableEy"}

  // CHECK: omp.yield(%0 : !fir.ref<i32>)
  omp.yield(%0 : !fir.ref<i32>)
}) : () -> ()
