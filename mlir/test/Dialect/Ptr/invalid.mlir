// RUN: mlir-opt -split-input-file -verify-diagnostics %s

/// Test `to_ptr` verifiers.
func.func @invalid_to_ptr(%v: memref<f32, 0>) {
  // expected-error@+1 {{expected the input and output to have the same memory space}}
  %r = ptr.to_ptr %v : memref<f32, 0> -> !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @invalid_to_ptr(%v: !ptr.ptr<#ptr.generic_space>) {
  // expected-error@+1 {{the input value cannot be of type `!ptr.ptr`}}
  %r = ptr.to_ptr %v : !ptr.ptr<#ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @invalid_load_alignment(%arg0: !ptr.ptr<#ptr.generic_space>) -> i64 {
  // expected-error@+1 {{alignment must be a power of 2}}
  %r = ptr.load %arg0 alignment = 3 : !ptr.ptr<#ptr.generic_space> -> i64
  return %r : i64
}

// -----

func.func @invalid_store_alignment(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: i64) {
  // expected-error@+1 {{alignment must be a power of 2}}
  ptr.store %arg1, %arg0 alignment = 3 : i64, !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @store_const(%arg0: !ptr.ptr<#test.const_memory_space>, %arg1: i64) {
  // expected-error@+1 {{memory space is read-only}}
  ptr.store %arg1, %arg0 atomic monotonic alignment = 8 : i64, !ptr.ptr<#test.const_memory_space>
  return
}

// -----

func.func @llvm_load(%arg0: !ptr.ptr<#llvm.address_space<1>>) -> (memref<f32>) {
  // expected-error@+1 {{type must be LLVM type with size, but got 'memref<f32>'}}
  %0 = ptr.load %arg0 : !ptr.ptr<#llvm.address_space<1>> -> memref<f32>
  return %0 : memref<f32>
}

// -----

func.func @llvm_store(%arg0: !ptr.ptr<#llvm.address_space<1>>, %arg1: memref<f32>) {
  // expected-error@+1 {{type must be LLVM type with size, but got 'memref<f32>'}}
  ptr.store %arg1, %arg0 : memref<f32>, !ptr.ptr<#llvm.address_space<1>>
  return
}

// -----

func.func @ptr_add_mismatch(%ptrs: tensor<8x!ptr.ptr<#ptr.generic_space>>, %offsets: vector<8xi64>) -> tensor<8x!ptr.ptr<#ptr.generic_space>> {
  // expected-error@+1 {{the shaped containers type must match}}
  %res = ptr.ptr_add %ptrs, %offsets : tensor<8x!ptr.ptr<#ptr.generic_space>>, vector<8xi64>
  return %res : tensor<8x!ptr.ptr<#ptr.generic_space>>
}

// -----

func.func @ptr_add_shape_mismatch(%ptrs: tensor<8x!ptr.ptr<#ptr.generic_space>>, %offsets: tensor<4xi64>) -> tensor<8x!ptr.ptr<#ptr.generic_space>> {
  // expected-error@+1 {{shapes of base and offset must match}}
  %res = ptr.ptr_add %ptrs, %offsets : tensor<8x!ptr.ptr<#ptr.generic_space>>, tensor<4xi64>
  return %res : tensor<8x!ptr.ptr<#ptr.generic_space>>
}

// -----

func.func @ptr_diff_mismatch(%lhs: tensor<8x!ptr.ptr<#ptr.generic_space>>, %rhs: tensor<8x!ptr.ptr<#ptr.generic_space>>) -> vector<8xi64> {
  // expected-error@+1 {{the result to have the same container type as the operands when operands are shaped}}
  %res = ptr.ptr_diff %lhs, %rhs : tensor<8x!ptr.ptr<#ptr.generic_space>> -> vector<8xi64>
  return %res : vector<8xi64>
}
