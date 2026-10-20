// RUN: mlir-opt %s -split-input-file -test-acc-support | FileCheck %s

// Test private recipe with 2D memref
func.func @test_private_2d_memref() {
  // Create a 2D memref
  %0 = memref.alloca() {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<5x10xf32>

  // CHECK: op=%{{.*}} = memref.alloca() {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<5x10xf32>
  // CHECK-NEXT: getRecipeName(kind=private_recipe, type=memref<5x10xf32>)="privatization_memref_5x10xf32_"

  return
}

// -----

// Test firstprivate recipe with 2D memref
func.func @test_firstprivate_2d_memref() {
  // Create a 2D memref
  %0 = memref.alloca() {test.recipe_name = #acc.recipe_kind<firstprivate_recipe>} : memref<8x16xf64>

  // CHECK: op=%{{.*}} = memref.alloca() {test.recipe_name = #acc.recipe_kind<firstprivate_recipe>} : memref<8x16xf64>
  // CHECK-NEXT: getRecipeName(kind=firstprivate_recipe, type=memref<8x16xf64>)="firstprivatization_memref_8x16xf64_"

  return
}

// -----

// Test reduction recipe with 2D memref
func.func @test_reduction_2d_memref() {
  // Create a 2D memref
  %0 = memref.alloca() {test.recipe_name = #acc.recipe_kind<reduction_recipe>} : memref<4x8xi32>

  // CHECK: op=%{{.*}} = memref.alloca() {test.recipe_name = #acc.recipe_kind<reduction_recipe>} : memref<4x8xi32>
  // CHECK-NEXT: getRecipeName(kind=reduction_recipe, type=memref<4x8xi32>)="reduction_memref_4x8xi32_"

  return
}

// -----

// Test private recipe with dynamic memref
func.func @test_private_dynamic_memref(%arg0: memref<5x10xi32>) {
  // Cast to dynamic dimensions
  %0 = memref.cast %arg0 {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<5x10xi32> to memref<?x10xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<5x10xi32> to memref<?x10xi32>
  // CHECK-NEXT: getRecipeName(kind=private_recipe, type=memref<?x10xi32>)="privatization_memref_Ux10xi32_"

  return
}

// -----

// Test private recipe with scalar memref
func.func @test_private_scalar_memref() {
  // Create a scalar memref (no dimensions)
  %0 = memref.alloca() {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<i32>

  // CHECK: op=%{{.*}} = memref.alloca() {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<i32>
  // CHECK-NEXT: getRecipeName(kind=private_recipe, type=memref<i32>)="privatization_memref_i32_"

  return
}

// -----

// Test private recipe with unranked memref
func.func @test_private_unranked_memref(%arg0: memref<10xi32>) {
  // Cast to unranked memref
  %0 = memref.cast %arg0 {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<10xi32> to memref<*xi32>

  // CHECK: op=%{{.*}} = memref.cast %{{.*}} {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<10xi32> to memref<*xi32>
  // CHECK-NEXT: getRecipeName(kind=private_recipe, type=memref<*xi32>)="privatization_memref_Zxi32_"

  return
}

