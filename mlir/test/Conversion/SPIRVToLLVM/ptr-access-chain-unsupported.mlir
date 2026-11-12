// RUN: mlir-opt %s -convert-spirv-to-llvm -verify-diagnostics -split-input-file

// Pointer ArrayStride decorations cannot be represented by the SPIR-V dialect.
// Reject storage classes where the decoration may affect the element stride.

spirv.func @ptr_access_chain_uniform(%ptr: !spirv.ptr<f32, Uniform>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.PtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.PtrAccessChain %ptr[%element] : !spirv.ptr<f32, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
  spirv.Return
}

// -----

spirv.func @ptr_access_chain_storage_buffer(%ptr: !spirv.ptr<f32, StorageBuffer>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.PtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.PtrAccessChain %ptr[%element] : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.ptr<f32, StorageBuffer>
  spirv.Return
}

// -----

spirv.func @ptr_access_chain_push_constant(%ptr: !spirv.ptr<f32, PushConstant>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.PtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.PtrAccessChain %ptr[%element] : !spirv.ptr<f32, PushConstant>, i32 -> !spirv.ptr<f32, PushConstant>
  spirv.Return
}

// -----

spirv.func @ptr_access_chain_physical_storage_buffer(%ptr: !spirv.ptr<f32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> }, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.PtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.PtrAccessChain %ptr[%element] : !spirv.ptr<f32, PhysicalStorageBuffer>, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
  spirv.Return
}

// -----

spirv.func @in_bounds_ptr_access_chain_uniform(%ptr: !spirv.ptr<f32, Uniform>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.InBoundsPtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.InBoundsPtrAccessChain %ptr[%element] : !spirv.ptr<f32, Uniform>, i32 -> !spirv.ptr<f32, Uniform>
  spirv.Return
}

// -----

spirv.func @in_bounds_ptr_access_chain_storage_buffer(%ptr: !spirv.ptr<f32, StorageBuffer>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.InBoundsPtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.InBoundsPtrAccessChain %ptr[%element] : !spirv.ptr<f32, StorageBuffer>, i32 -> !spirv.ptr<f32, StorageBuffer>
  spirv.Return
}

// -----

spirv.func @in_bounds_ptr_access_chain_push_constant(%ptr: !spirv.ptr<f32, PushConstant>, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.InBoundsPtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.InBoundsPtrAccessChain %ptr[%element] : !spirv.ptr<f32, PushConstant>, i32 -> !spirv.ptr<f32, PushConstant>
  spirv.Return
}

// -----

spirv.func @in_bounds_ptr_access_chain_physical_storage_buffer(%ptr: !spirv.ptr<f32, PhysicalStorageBuffer> { spirv.decoration = #spirv.decoration<Aliased> }, %element: i32) "None" {
  // expected-error@+1 {{failed to legalize operation 'spirv.InBoundsPtrAccessChain' that was explicitly marked illegal}}
  %0 = spirv.InBoundsPtrAccessChain %ptr[%element] : !spirv.ptr<f32, PhysicalStorageBuffer>, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
  spirv.Return
}
