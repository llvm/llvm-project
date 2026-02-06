// RUN: mlir-opt %s -split-input-file -test-acc-support | FileCheck %s

// Test with direct variable names
func.func @test_direct_var_name() {
  // Create a memref with acc.var_name attribute
  %0 = memref.alloca() {acc.var_name = #acc.var_name<"my_variable">} : memref<10xi32>

  %1 = memref.cast %0 {test.var_name} : memref<10xi32> to memref<10xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.var_name} : memref<10xi32> to memref<10xi32>
  // CHECK-NEXT: getVariableName="my_variable"

  return
}

// -----

// Test through memref.cast
func.func @test_through_cast() {
  // Create a 5x2 memref with acc.var_name attribute
  %0 = memref.alloca() {acc.var_name = #acc.var_name<"casted_variable">} : memref<5x2xi32>

  // Cast to dynamic dimensions
  %1 = memref.cast %0 : memref<5x2xi32> to memref<?x?xi32>

  // Mark with test attribute - should find name through cast
  %2 = memref.cast %1 {test.var_name} : memref<?x?xi32> to memref<5x2xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.var_name} : memref<?x?xi32> to memref<5x2xi32>
  // CHECK-NEXT: getVariableName="casted_variable"

  return
}

// -----

// Test with no variable name
func.func @test_no_var_name() {
  // Create a memref without acc.var_name attribute
  %0 = memref.alloca() : memref<10xi32>

  // Mark with test attribute - should find empty string
  %1 = memref.cast %0 {test.var_name} : memref<10xi32> to memref<10xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.var_name} : memref<10xi32> to memref<10xi32>
  // CHECK-NEXT: getVariableName=""

  return
}

// -----

// Test through multiple casts
func.func @test_multiple_casts() {
  // Create a memref with acc.var_name attribute
  %0 = memref.alloca() {acc.var_name = #acc.var_name<"multi_cast">} : memref<10xi32>

  // Multiple casts
  %1 = memref.cast %0 : memref<10xi32> to memref<?xi32>
  %2 = memref.cast %1 : memref<?xi32> to memref<10xi32>

  // Mark with test attribute - should find name through multiple casts
  %3 = memref.cast %2 {test.var_name} : memref<10xi32> to memref<10xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.var_name} : memref<10xi32> to memref<10xi32>
  // CHECK-NEXT: getVariableName="multi_cast"

  return
}

// -----

// Test with acc.copyin operation
func.func @test_copyin_name() {
  // Create a memref
  %0 = memref.alloca() : memref<10xf32>

  // Create an acc.copyin operation with a name
  %1 = acc.copyin varPtr(%0 : memref<10xf32>) -> memref<10xf32> {name = "input_data"}

  // Mark with test attribute - should find name from copyin operation
  %2 = memref.cast %1 {test.var_name} : memref<10xf32> to memref<?xf32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.var_name} : memref<10xf32> to memref<?xf32>
  // CHECK-NEXT: getVariableName="input_data"

  return
}
