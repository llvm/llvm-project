// RUN: mlir-opt %s -acc-if-clause-lowering -split-input-file | FileCheck %s

// Test acc.parallel with if condition
// CHECK-LABEL: func.func @test_parallel_if
func.func @test_parallel_if(%arg0: memref<10xi32>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<10xi32>) -> memref<10xi32>
  %create = acc.create varPtr(%arg0 : memref<10xi32>) -> memref<10xi32> {dataClause = #acc<data_clause acc_copyout>}

  // CHECK-NOT: acc.parallel if
  // CHECK: scf.if %{{.*}} {
  // CHECK:   %[[COPYIN:.*]] = acc.copyin varPtr(%{{.*}}) -> memref<10xi32>
  // CHECK:   %[[CREATE:.*]] = acc.create varPtr(%{{.*}}) -> memref<10xi32>
  // CHECK:   acc.parallel dataOperands(%[[COPYIN]], %[[CREATE]] : memref<10xi32>, memref<10xi32>) {
  // CHECK:     scf.for
  // CHECK:     acc.yield
  // CHECK:   }
  // CHECK:   acc.delete accPtr(%[[CREATE]] : memref<10xi32>)
  // CHECK:   acc.copyout accPtr(%[[COPYIN]] : memref<10xi32>) to varPtr(%{{.*}} : memref<10xi32>)
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK: }
  acc.parallel dataOperands(%copyin, %create : memref<10xi32>, memref<10xi32>) if(%cond) {
    scf.for %i = %c1 to %c10 step %c1 {
      memref.store %c0_i32, %arg0[%i] : memref<10xi32>
    }
    acc.yield
  }

  acc.delete accPtr(%create : memref<10xi32>)
  acc.copyout accPtr(%copyin : memref<10xi32>) to varPtr(%arg0 : memref<10xi32>)
  return
}

// -----

// Test acc.kernels with if condition
// CHECK-LABEL: func.func @test_kernels_if
func.func @test_kernels_if(%arg0: memref<5xi32>, %cond: i1) {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<5xi32>) -> memref<5xi32>
  %create = acc.create varPtr(%arg0 : memref<5xi32>) -> memref<5xi32> {dataClause = #acc<data_clause acc_copyout>}

  // CHECK-NOT: acc.kernels if
  // CHECK: scf.if %{{.*}} {
  // CHECK:   %[[COPYIN:.*]] = acc.copyin
  // CHECK:   %[[CREATE:.*]] = acc.create
  // CHECK:   acc.kernels dataOperands(%[[COPYIN]], %[[CREATE]] : memref<5xi32>, memref<5xi32>) {
  // CHECK:     scf.for
  // CHECK:     acc.terminator
  // CHECK:   }
  // CHECK:   acc.delete accPtr(%[[CREATE]] : memref<5xi32>)
  // CHECK:   acc.copyout accPtr(%[[COPYIN]] : memref<5xi32>) to varPtr(%{{.*}} : memref<5xi32>)
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK: }
  acc.kernels dataOperands(%copyin, %create : memref<5xi32>, memref<5xi32>) if(%cond) {
    scf.for %i = %c1 to %c5 step %c1 {
      memref.store %c1_i32, %arg0[%i] : memref<5xi32>
    }
    acc.terminator
  }

  acc.delete accPtr(%create : memref<5xi32>)
  acc.copyout accPtr(%copyin : memref<5xi32>) to varPtr(%arg0 : memref<5xi32>)
  return
}

// -----

// Test acc.serial with if condition
// CHECK-LABEL: func.func @test_serial_if
func.func @test_serial_if(%arg0: memref<8xi32>, %cond: i1) {
  %c2_i32 = arith.constant 2 : i32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<8xi32>) -> memref<8xi32>
  %create = acc.create varPtr(%arg0 : memref<8xi32>) -> memref<8xi32> {dataClause = #acc<data_clause acc_copyout>}

  // CHECK-NOT: acc.serial if
  // CHECK: scf.if %{{.*}} {
  // CHECK:   %[[COPYIN:.*]] = acc.copyin
  // CHECK:   %[[CREATE:.*]] = acc.create
  // CHECK:   acc.serial dataOperands(%[[COPYIN]], %[[CREATE]] : memref<8xi32>, memref<8xi32>) {
  // CHECK:     scf.for
  // CHECK:     acc.yield
  // CHECK:   }
  // CHECK:   acc.delete accPtr(%[[CREATE]] : memref<8xi32>)
  // CHECK:   acc.copyout accPtr(%[[COPYIN]] : memref<8xi32>) to varPtr(%{{.*}} : memref<8xi32>)
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK: }
  acc.serial dataOperands(%copyin, %create : memref<8xi32>, memref<8xi32>) if(%cond) {
    scf.for %i = %c1 to %c8 step %c1 {
      memref.store %c2_i32, %arg0[%i] : memref<8xi32>
    }
    acc.yield
  }

  acc.delete accPtr(%create : memref<8xi32>)
  acc.copyout accPtr(%copyin : memref<8xi32>) to varPtr(%arg0 : memref<8xi32>)
  return
}

// -----

// Test that acc.parallel without if condition is not modified
// CHECK-LABEL: func.func @test_parallel_no_if
func.func @test_parallel_no_if(%arg0: memref<10xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<10xi32>) -> memref<10xi32>

  // CHECK-NOT: scf.if
  // CHECK: acc.parallel dataOperands(%{{.*}}) {
  acc.parallel dataOperands(%copyin : memref<10xi32>) {
    scf.for %i = %c1 to %c10 step %c1 {
      memref.store %c0_i32, %arg0[%i] : memref<10xi32>
    }
    acc.yield
  }

  acc.copyout accPtr(%copyin : memref<10xi32>) to varPtr(%arg0 : memref<10xi32>)
  return
}

// -----

// Test with private and reduction clauses inside compute construct
acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

acc.reduction.recipe @reduction_add_memref_f32 : memref<f32> reduction_operator <add> init {
^bb0(%arg0: memref<f32>):
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloca() : memref<f32>
  memref.store %cst, %0[] : memref<f32>
  acc.yield %0 : memref<f32>
} combiner {
^bb0(%arg0: memref<f32>, %arg1: memref<f32>):
  %0 = memref.load %arg0[] : memref<f32>
  %1 = memref.load %arg1[] : memref<f32>
  %2 = arith.addf %0, %1 : f32
  memref.store %2, %arg0[] : memref<f32>
  acc.yield %arg0 : memref<f32>
}

// CHECK-LABEL: func.func @test_reduction_if
func.func @test_reduction_if(%r: memref<f32>, %a: memref<8xf32>, %cond: i1) {
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32

  %copyin = acc.copyin varPtr(%r : memref<f32>) -> memref<f32> {dataClause = #acc<data_clause acc_reduction>, implicit = true}

  // CHECK: scf.if
  // CHECK:   acc.parallel
  // CHECK: } else {
  // The else branch should have acc ops converted to host
  // CHECK-NOT: acc.loop
  // CHECK-NOT: acc.reduction
  // CHECK-NOT: acc.private
  // CHECK: }
  acc.parallel combined(loop) dataOperands(%copyin : memref<f32>) if(%cond) {
    %red = acc.reduction varPtr(%r : memref<f32>) recipe(@reduction_add_memref_f32) -> memref<f32>
    %iter_var = memref.alloca() : memref<i32>
    %priv = acc.private varPtr(%iter_var : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
    acc.loop combined(parallel) vector private(%priv : memref<i32>) reduction(%red : memref<f32>) control(%iv : i32) = (%c1_i32 : i32) to (%c8_i32 : i32) step (%c1_i32 : i32) {
      memref.store %iv, %priv[] : memref<i32>
      %idx = memref.load %priv[] : memref<i32>
      %idx_cast = arith.index_cast %idx : i32 to index
      %elem = memref.load %a[%idx_cast] : memref<8xf32>
      %r_val = memref.load %r[] : memref<f32>
      %new_r = arith.addf %r_val, %elem : f32
      memref.store %new_r, %r[] : memref<f32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }

  acc.copyout accPtr(%copyin : memref<f32>) to varPtr(%r : memref<f32>) {dataClause = #acc<data_clause acc_reduction>, implicit = true}
  return
}

// -----

// Test that acc variable uses in host path are replaced with host variables
// CHECK-LABEL: func.func @test_acc_var_replacement
func.func @test_acc_var_replacement(%arg0: memref<10xi32>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<10xi32>) -> memref<10xi32>

  // In the else branch, uses of %copyin should be replaced with %arg0
  // CHECK: scf.if
  // CHECK: } else {
  // CHECK:   scf.for
  // CHECK:     memref.store %{{.*}}, %arg0[%{{.*}}]
  // CHECK: }
  acc.parallel dataOperands(%copyin : memref<10xi32>) if(%cond) {
    scf.for %i = %c1 to %c10 step %c1 {
      // Use the acc ptr inside the region
      memref.store %c0_i32, %copyin[%i] : memref<10xi32>
    }
    acc.yield
  }

  acc.copyout accPtr(%copyin : memref<10xi32>) to varPtr(%arg0 : memref<10xi32>)
  return
}

// -----

// Test that acc variable uses in host path are replaced with host variables;
// and the firstprivate operands are cloned
// CHECK-LABEL: func.func @test_acc_firstprivate

acc.firstprivate.recipe @memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
} copy {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg0[] : memref<i32>
  memref.store %0, %arg1[] : memref<i32>
  acc.terminator
}

func.func @test_acc_firstprivate(%arg0: memref<10xi32>, %arg1: memref<i32>, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %copyin = acc.copyin varPtr(%arg0 : memref<10xi32>) -> memref<10xi32>
  %firstprivate = acc.firstprivate varPtr(%arg1 : memref<i32>) recipe(@memref_i32) -> memref<i32>

  // In the else branch, uses of %firstprivate should be replaced with %arg0
  // CHECK: scf.if
  // CHECK: [[FIRSTPRIVATE:%.*]] = acc.firstprivate varPtr(%arg1 : memref<i32>) recipe(@memref_i32) -> memref<i32>
  // CHECK: acc.parallel {{.*}} firstprivate([[FIRSTPRIVATE]] : memref<i32>) {
  // CHECK: } else {
  // CHECK: [[LOAD:%.*]] = memref.load %arg1[] : memref<i32>
  // CHECK: }

  acc.parallel dataOperands(%copyin : memref<10xi32>) firstprivate(%firstprivate : memref<i32>) if(%cond) {
    %load = memref.load %firstprivate[] : memref<i32>
    %ub = arith.index_cast %load : i32 to index
    scf.for %i = %c1 to %ub step %c1 {
      // Use the acc ptr inside the region
      memref.store %c0_i32, %copyin[%i] : memref<10xi32>
    }
    acc.yield
  }

  acc.copyout accPtr(%copyin : memref<10xi32>) to varPtr(%arg0 : memref<10xi32>)
  return
}

// -----

// Test that acc variable uses in host path are replaced with host variables;
// and the reduction operands are cloned
// CHECK-LABEL: func.func @test_acc_reduction

acc.reduction.recipe @reduction_add_memref_i32 : memref<i32> reduction_operator <add> init {
^bb0(%arg0: memref<i32>):
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.alloca() : memref<i32>
  memref.store %c0_i32, %0[] : memref<i32>
  acc.yield %0 : memref<i32>
} combiner {
^bb0(%arg0: memref<i32>, %arg1: memref<i32>):
  %0 = memref.load %arg1[] : memref<i32>
  %1 = memref.load %arg0[] : memref<i32>
  %2 = arith.addi %1, %0 : i32
  memref.store %2, %arg0[] : memref<i32>
  acc.yield %arg0 : memref<i32>
}

func.func @test_acc_reduction(%arg0: memref<i32>, %cond: i1) {

  %c0_i32 = arith.constant 0 : i32
  %reduction = acc.reduction varPtr(%arg0 : memref<i32>) recipe(@reduction_add_memref_i32) -> memref<i32>

  // In the else branch, uses of %reduction should be replaced with %arg0
  // CHECK: scf.if
  // CHECK: [[REDUCTION:%.*]] = acc.reduction varPtr(%arg0 : memref<i32>) recipe(@reduction_add_memref_i32) -> memref<i32>
  // CHECK: acc.parallel reduction([[REDUCTION]] : memref<i32>) {
  // CHECK: } else {
  // CHECK: [[LOAD:%.*]] = memref.load %arg0[] : memref<i32>
  // CHECK: memref.store {{.*}}, %arg0[] : memref<i32>
  // CHECK: }

  acc.parallel reduction(%reduction : memref<i32>) if(%cond) {
    %load = memref.load %reduction[] : memref<i32>
    %add = arith.addi %load, %c0_i32 : i32
    memref.store %add, %reduction[] : memref<i32>
    acc.yield
  }
  return
}

acc.private.recipe @privatization_memref_i32 : memref<i32> init {
^bb0(%arg0: memref<i32>):
  %0 = memref.alloca() : memref<i32>
  acc.yield %0 : memref<i32>
}

func.func @test_acc_private(%arg0: memref<i32>, %cond: i1) {

  %c0_i32 = arith.constant 0 : i32
  %private = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>

  // In the else branch, uses of %private should be replaced with %arg0
  // CHECK: scf.if
  // CHECK: [[PRIVATE:%.*]] = acc.private varPtr(%arg0 : memref<i32>) recipe(@privatization_memref_i32) -> memref<i32>
  // CHECK: acc.parallel private([[PRIVATE]] : memref<i32>) {
  // CHECK: } else {
  // CHECK: memref.store {{.*}}, %arg0[] : memref<i32>
  // CHECK: }

  acc.parallel private(%private : memref<i32>) if(%cond) {
    memref.store %c0_i32, %private[] : memref<i32>
    acc.yield
  }
  return
}