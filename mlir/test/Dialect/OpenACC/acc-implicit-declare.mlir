// RUN: mlir-opt %s --pass-pipeline="builtin.module(acc-implicit-declare)" -split-input-file 2>&1 | FileCheck %s

// -----

// Test that non-constant scalar globals in compute regions are hoisted
// instead of being marked with acc declare

memref.global @gscalar : memref<f32> = dense<0.0>

func.func @test_scalar_in_serial() {
  acc.serial {
    %addr = memref.get_global @gscalar : memref<f32>
    %load = memref.load %addr[] : memref<f32>
    acc.yield
  }
  return
}

// Expected to hoist this global access out of acc region instead of marking
// with `acc declare`.
// CHECK-LABEL: func.func @test_scalar_in_serial
// CHECK: memref.get_global @gscalar
// CHECK: acc.serial
// CHECK-NOT: acc.declare

// -----

// Test that constant globals are marked with acc declare

memref.global constant @gscalarconst : memref<f32> = dense<1.0>

func.func @test_constant_in_serial() {
  acc.serial {
    %addr = memref.get_global @gscalarconst : memref<f32>
    %load = memref.load %addr[] : memref<f32>
    acc.yield
  }
  return
}

// This is expected to be `acc declare`'d since it is a constant.
// CHECK: memref.global constant @gscalarconst {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}

// -----

// Test globals referenced in acc routine functions

memref.global @gscalar_routine : memref<f32> = dense<0.0>

acc.routine @acc_routine_0 func(@test_scalar_in_accroutine)
func.func @test_scalar_in_accroutine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>} {
  %addr = memref.get_global @gscalar_routine : memref<f32>
  %load = memref.load %addr[] : memref<f32>
  return
}

// Global should be acc declare'd because it's in an acc routine
// CHECK: memref.global @gscalar_routine {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}

// -----

// Test constant globals in acc routine

memref.global constant @gscalarconst_routine : memref<f32> = dense<1.0>

acc.routine @acc_routine_0 func(@test_constant_in_accroutine)
func.func @test_constant_in_accroutine() attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>} {
  %addr = memref.get_global @gscalarconst_routine : memref<f32>
  %load = memref.load %addr[] : memref<f32>
  return
}

// CHECK: memref.global constant @gscalarconst_routine {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}

// -----

// Test acc.private.recipe with global reference - referenced variant

memref.global @global_for_private : memref<f32> = dense<0.0>

acc.private.recipe @private_recipe_with_global : memref<f32> init {
^bb0(%arg0: memref<f32>):
  %0 = memref.alloc() : memref<f32>
  %global_addr = memref.get_global @global_for_private : memref<f32>
  %global_val = memref.load %global_addr[] : memref<f32>
  memref.store %global_val, %0[] : memref<f32>
  acc.yield %0 : memref<f32>
} destroy {
^bb0(%arg0: memref<f32>):
  memref.dealloc %arg0 : memref<f32>
  acc.terminator
}

func.func @test_private_recipe_referenced() {
  %var = memref.alloc() : memref<f32>
  %priv = acc.private varPtr(%var : memref<f32>) recipe(@private_recipe_with_global) -> memref<f32>
  acc.parallel private(%priv : memref<f32>) {
    %load = memref.load %var[] : memref<f32>
    acc.yield
  }
  memref.dealloc %var : memref<f32>
  return
}

// Global should be acc declare'd because the recipe is referenced
// CHECK: memref.global @global_for_private {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}

// -----

// Test acc.private.recipe with global reference - unreferenced variant

memref.global @global_for_private_unused : memref<f32> = dense<0.0>

acc.private.recipe @private_recipe_unused : memref<f32> init {
^bb0(%arg0: memref<f32>):
  %0 = memref.alloc() : memref<f32>
  %global_addr = memref.get_global @global_for_private_unused : memref<f32>
  %global_val = memref.load %global_addr[] : memref<f32>
  memref.store %global_val, %0[] : memref<f32>
  acc.yield %0 : memref<f32>
} destroy {
^bb0(%arg0: memref<f32>):
  memref.dealloc %arg0 : memref<f32>
  acc.terminator
}

func.func @test_private_recipe_not_referenced() {
  %var = memref.alloc() : memref<f32>
  acc.parallel {
    %load = memref.load %var[] : memref<f32>
    acc.yield
  }
  memref.dealloc %var : memref<f32>
  return
}

// Global should NOT be acc declare'd because the recipe is not referenced
// CHECK-NOT: memref.global @global_for_private_unused {{.*}} {acc.declare

// -----

// Test globals in different compute constructs (parallel, kernels, serial)

memref.global @global_parallel : memref<f32> = dense<0.0>
memref.global @global_kernels : memref<f32> = dense<0.0>
memref.global constant @global_serial_const : memref<f32> = dense<1.0>

func.func @test_multiple_constructs() {
  acc.parallel {
    %addr = memref.get_global @global_parallel : memref<f32>
    %load = memref.load %addr[] : memref<f32>
    acc.yield
  }
  acc.kernels {
    %addr = memref.get_global @global_kernels : memref<f32>
    %load = memref.load %addr[] : memref<f32>
    acc.terminator
  }
  acc.serial {
    %addr = memref.get_global @global_serial_const : memref<f32>
    %load = memref.load %addr[] : memref<f32>
    acc.yield
  }
  return
}

// Non-constant globals ARE hoisted before their compute regions
// Constant global should be marked with acc.declare
// CHECK: memref.global constant @global_serial_const {{.*}} {acc.declare = #acc.declare<dataClause = acc_copyin>}
// CHECK-LABEL: func.func @test_multiple_constructs
// CHECK: memref.get_global @global_parallel
// CHECK-NEXT: acc.parallel
// CHECK: memref.get_global @global_kernels
// CHECK-NEXT: acc.kernels

