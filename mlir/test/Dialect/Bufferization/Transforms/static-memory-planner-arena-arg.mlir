// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis{arena-mode=arg}))" \
// RUN:     -split-input-file -verify-diagnostics

// -----

// Test 1: Arena from function argument
func.func @arena_from_arg(%arena: memref<8192xi8>) {
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  return
}

// -----

// Test 2: Error when no function argument
// expected-error @+1 {{arena-mode=arg requires at least one function argument}}
func.func @error_no_args() {
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  return
}

// -----

// Test 3: Error when first argument is not i8 memref
// expected-error @+1 {{arena-mode=arg requires first argument to be memref<...xi8>}}
func.func @error_wrong_type(%arena: memref<8192xf32>) {
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  return
}

// -----

// Test 4: Error when function returns memref
// expected-error @+1 {{static-memory-planner does not support functions with memref return types}}
func.func @error_memref_return(%arena: memref<8192xi8>) -> memref<1024xf32> {
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  %result = memref.alloc() : memref<1024xf32>
  return %result : memref<1024xf32>
}
