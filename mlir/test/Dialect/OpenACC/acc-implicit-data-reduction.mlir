// RUN: mlir-opt %s -acc-implicit-data=enable-implicit-reduction-copy=true -split-input-file | FileCheck %s --check-prefix=COPY
// RUN: mlir-opt %s -acc-implicit-data=enable-implicit-reduction-copy=false -split-input-file | FileCheck %s --check-prefix=FIRSTPRIVATE

// Test case: scalar reduction variable in parallel loop
// When enable-implicit-reduction-copy=true: expect copyin/copyout for reduction variable
// When enable-implicit-reduction-copy=false: expect firstprivate for reduction variable

acc.reduction.recipe @reduction_add_memref_i32 : memref<i32> reduction_operator <add> init {
^bb0(%arg0: memref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloca() : memref<i32>
  memref.store %c0_i32, %alloc[] : memref<i32>
  acc.yield %alloc : memref<i32>
} combiner {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  %1 = memref.load %arg1[] : memref<i32>
  %2 = arith.addi %0, %1 : i32
  memref.store %2, %arg0[] : memref<i32>
  acc.yield %arg0 : memref<i32>
}

func.func @test_reduction_implicit_copy() {
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %c0_i32 = arith.constant 0 : i32
  %r = memref.alloca() : memref<i32>
  memref.store %c0_i32, %r[] : memref<i32>

  acc.parallel {
    %red_var = acc.reduction varPtr(%r : memref<i32>) -> memref<i32> {name = "r"}
    acc.loop reduction(@reduction_add_memref_i32 -> %red_var : memref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      %load = memref.load %red_var[] : memref<i32>
      %add = arith.addi %load, %c1_i32 : i32
      memref.store %add, %red_var[] : memref<i32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}

// When enable-implicit-reduction-copy=true: expect copyin/copyout for reduction variable
// COPY-LABEL: func.func @test_reduction_implicit_copy
// COPY: %[[COPYIN:.*]] = acc.copyin varPtr({{.*}} : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_reduction>, implicit = true, name = ""}
// COPY: acc.copyout accPtr(%[[COPYIN]] : memref<i32>) to varPtr({{.*}} : memref<i32>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = ""}

// When enable-implicit-reduction-copy=false: expect firstprivate for reduction variable  
// FIRSTPRIVATE-LABEL: func.func @test_reduction_implicit_copy
// FIRSTPRIVATE: acc.firstprivate varPtr({{.*}} : memref<i32>) -> memref<i32> {implicit = true, name = ""}
// FIRSTPRIVATE-NOT: acc.copyin
// FIRSTPRIVATE-NOT: acc.copyout

// -----

// Test case: reduction variable used both in loop and outside
// Should be firstprivate regardless of the flag setting

acc.reduction.recipe @reduction_add_memref_i32_2 : memref<i32> reduction_operator <add> init {
^bb0(%arg0: memref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %alloc = memref.alloca() : memref<i32>
  memref.store %c0_i32, %alloc[] : memref<i32>
  acc.yield %alloc : memref<i32>
} combiner {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  %1 = memref.load %arg1[] : memref<i32>
  %2 = arith.addi %0, %1 : i32
  memref.store %2, %arg0[] : memref<i32>
  acc.yield %arg0 : memref<i32>
}

func.func @test_reduction_with_usage_outside_loop() {
  %c1_i32 = arith.constant 1 : i32
  %c100_i32 = arith.constant 100 : i32
  %c0_i32 = arith.constant 0 : i32
  %r = memref.alloca() : memref<i32>
  %out = memref.alloca() : memref<i32>
  memref.store %c0_i32, %r[] : memref<i32>

  %out_create = acc.create varPtr(%out : memref<i32>) -> memref<i32> {dataClause = #acc<data_clause acc_copyout>, name = "out"}
  acc.parallel dataOperands(%out_create : memref<i32>) {
    %red_var = acc.reduction varPtr(%r : memref<i32>) -> memref<i32> {name = "r"}
    acc.loop reduction(@reduction_add_memref_i32_2 -> %red_var : memref<i32>) control(%iv : i32) = (%c1_i32 : i32) to (%c100_i32 : i32) step (%c1_i32 : i32) {
      %load = memref.load %red_var[] : memref<i32>
      %add = arith.addi %load, %c1_i32 : i32
      memref.store %add, %red_var[] : memref<i32>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    // out = r (usage of r outside the loop)
    %final_r = memref.load %r[] : memref<i32>
    memref.store %final_r, %out_create[] : memref<i32>
    acc.yield
  }
  acc.copyout accPtr(%out_create : memref<i32>) to varPtr(%out : memref<i32>) {dataClause = #acc<data_clause acc_copyout>, name = "out"}
  return
}

// In this case, r should be firstprivate regardless of the flag setting 
// because it's used outside the reduction context
// COPY-LABEL: func.func @test_reduction_with_usage_outside_loop
// COPY: acc.firstprivate varPtr({{.*}} : memref<i32>) -> memref<i32> {implicit = true, name = ""}
// COPY-NOT: acc.copyin varPtr({{.*}} : memref<i32>) -> memref<i32> {{.*}} name = ""

// FIRSTPRIVATE-LABEL: func.func @test_reduction_with_usage_outside_loop
// FIRSTPRIVATE: acc.firstprivate varPtr({{.*}} : memref<i32>) -> memref<i32> {implicit = true, name = ""}
// FIRSTPRIVATE-NOT: acc.copyin varPtr({{.*}} : memref<i32>) -> memref<i32> {{.*}} name = ""

