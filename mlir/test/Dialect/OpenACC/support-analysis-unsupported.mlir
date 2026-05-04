// RUN: mlir-opt %s --split-input-file -test-acc-support -verify-diagnostics

// Test emitNYI with a simple message
func.func @test_emit_nyi() {
  // expected-error @below {{not yet implemented: Unsupported feature in OpenACC}}
  %0 = memref.alloca() {test.emit_nyi = "Unsupported feature in OpenACC"} : memref<10xi32>
  return
}

// -----

// Test recipe name on load operation from scalar memref
func.func @test_recipe_load_scalar() {
  %0 = memref.alloca() : memref<i32>
  // expected-error @below {{not yet implemented: variable privatization (incomplete recipe name handling)}}
  %1 = memref.load %0[] {test.recipe_name = #acc.recipe_kind<private_recipe>} : memref<i32>
  return
}
