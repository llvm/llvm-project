// RUN: mlir-opt %s --verify-roundtrip

/// Check op assembly.
func.func @ptr_add_type_offset(%ptr: !ptr.ptr<#ptr.generic_space>) -> !ptr.ptr<#ptr.generic_space> {
  %off = ptr.type_offset f32 : index
  %res = ptr.ptr_add %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res0 = ptr.ptr_add none %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res1 = ptr.ptr_add nusw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res2 = ptr.ptr_add nuw %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  %res3 = ptr.ptr_add inbounds %ptr, %off : !ptr.ptr<#ptr.generic_space>, index
  return %res : !ptr.ptr<#ptr.generic_space>
}

/// Check cast ops assembly.
func.func @cast_ops(%mr: memref<f32, #ptr.generic_space>) -> memref<f32, #ptr.generic_space> {
  %ptr = ptr.to_ptr %mr : memref<f32, #ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  %mda = ptr.get_metadata %mr : memref<f32, #ptr.generic_space>
  %res = ptr.from_ptr %ptr metadata %mda : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  %mr0 = ptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return %res : memref<f32, #ptr.generic_space>
}

/// Check load ops assembly.
func.func @load_ops(%arg0: !ptr.ptr<#ptr.generic_space>) -> (f32, f32, f32, f32, f32, i64, i32) {
  %0 = ptr.load %arg0 : !ptr.ptr<#ptr.generic_space> -> f32
  %1 = ptr.load volatile %arg0 : !ptr.ptr<#ptr.generic_space> -> f32
  %2 = ptr.load %arg0 nontemporal : !ptr.ptr<#ptr.generic_space> -> f32
  %3 = ptr.load %arg0 invariant : !ptr.ptr<#ptr.generic_space> -> f32
  %4 = ptr.load %arg0 invariant_group : !ptr.ptr<#ptr.generic_space> -> f32
  %5 = ptr.load %arg0 atomic monotonic alignment = 8 : !ptr.ptr<#ptr.generic_space> -> i64
  %6 = ptr.load volatile %arg0 atomic syncscope("workgroup") acquire nontemporal alignment = 4 : !ptr.ptr<#ptr.generic_space> -> i32
  return %0, %1, %2, %3, %4, %5, %6 : f32, f32, f32, f32, f32, i64, i32
}

/// Check store ops assembly.
func.func @store_ops(%arg0: !ptr.ptr<#ptr.generic_space>, %arg1: f32, %arg2: i64, %arg3: i32) {
  ptr.store %arg1, %arg0 : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store volatile %arg1, %arg0 : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg1, %arg0 nontemporal : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg1, %arg0 invariant_group : f32, !ptr.ptr<#ptr.generic_space>
  ptr.store %arg2, %arg0 atomic monotonic alignment = 8 : i64, !ptr.ptr<#ptr.generic_space>
  ptr.store volatile %arg3, %arg0 atomic syncscope("workgroup") release nontemporal alignment = 4 : i32, !ptr.ptr<#ptr.generic_space>
  return
}

/// Test load operations with llvm.address_space memory space
func.func @llvm_load(%arg0: !ptr.ptr<#llvm.address_space<1>>) -> (f32, i32) {
  %0 = ptr.load %arg0 : !ptr.ptr<#llvm.address_space<1>> -> f32
  %1 = ptr.load volatile %arg0 atomic acquire alignment = 4 : !ptr.ptr<#llvm.address_space<1>> -> i32
  return %0, %1 : f32, i32
}

/// Test store operations with llvm.address_space memory space
func.func @llvm_store(%arg0: !ptr.ptr<#llvm.address_space<2>>, %arg1: f32, %arg2: i64) {
  ptr.store %arg1, %arg0 : f32, !ptr.ptr<#llvm.address_space<2>>
  ptr.store %arg2, %arg0 atomic release alignment = 8 : i64, !ptr.ptr<#llvm.address_space<2>>
  return
}

/// Test gather operations
func.func @gather_ops(%ptrs: vector<4x!ptr.ptr<#ptr.generic_space>>, %mask: vector<4xi1>, %passthrough: vector<4xf32>) -> vector<4xf32> {
  %0 = ptr.gather %ptrs, %mask, %passthrough : vector<4x!ptr.ptr<#ptr.generic_space>> -> vector<4xf32>
  %1 = ptr.gather %ptrs, %mask, %passthrough alignment = 8 : vector<4x!ptr.ptr<#ptr.generic_space>> -> vector<4xf32>
  return %0 : vector<4xf32>
}

/// Test gather operations with tensors
func.func @gather_ops_tensor(%ptrs: tensor<8x!ptr.ptr<#ptr.generic_space>>, %mask: tensor<8xi1>, %passthrough: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ptr.gather %ptrs, %mask, %passthrough : tensor<8x!ptr.ptr<#ptr.generic_space>> -> tensor<8xi32>
  %1 = ptr.gather %ptrs, %mask, %passthrough alignment = 4 : tensor<8x!ptr.ptr<#ptr.generic_space>> -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

/// Test scatter operations
func.func @scatter_ops(%value: vector<4xf32>, %ptrs: vector<4x!ptr.ptr<#ptr.generic_space>>, %mask: vector<4xi1>) {
  ptr.scatter %value, %ptrs, %mask : vector<4xf32>, vector<4x!ptr.ptr<#ptr.generic_space>>
  ptr.scatter %value, %ptrs, %mask alignment = 16 : vector<4xf32>, vector<4x!ptr.ptr<#ptr.generic_space>>
  return
}

/// Test scatter operations with tensors
func.func @scatter_ops_tensor(%value: tensor<8xi64>, %ptrs: tensor<8x!ptr.ptr<#ptr.generic_space>>, %mask: tensor<8xi1>) {
  ptr.scatter %value, %ptrs, %mask : tensor<8xi64>, tensor<8x!ptr.ptr<#ptr.generic_space>>
  ptr.scatter %value, %ptrs, %mask alignment = 8 : tensor<8xi64>, tensor<8x!ptr.ptr<#ptr.generic_space>>
  return
}

/// Test masked load operations
func.func @masked_load_ops(%ptr: !ptr.ptr<#ptr.generic_space>, %mask: vector<4xi1>, %passthrough: vector<4xf32>) -> vector<4xf32> {
  %0 = ptr.masked_load %ptr, %mask, %passthrough : !ptr.ptr<#ptr.generic_space> -> vector<4xf32>
  %1 = ptr.masked_load %ptr, %mask, %passthrough alignment = 16 : !ptr.ptr<#ptr.generic_space> -> vector<4xf32>
  return %0 : vector<4xf32>
}

/// Test masked load operations with tensors
func.func @masked_load_ops_tensor(%ptr: !ptr.ptr<#ptr.generic_space>, %mask: tensor<8xi1>, %passthrough: tensor<8xi32>) -> tensor<8xi32> {
  %0 = ptr.masked_load %ptr, %mask, %passthrough : !ptr.ptr<#ptr.generic_space> -> tensor<8xi32>
  %1 = ptr.masked_load %ptr, %mask, %passthrough alignment = 4 : !ptr.ptr<#ptr.generic_space> -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

/// Test masked store operations
func.func @masked_store_ops(%value: vector<4xf32>, %ptr: !ptr.ptr<#ptr.generic_space>, %mask: vector<4xi1>) {
  ptr.masked_store %value, %ptr, %mask : vector<4xf32>, !ptr.ptr<#ptr.generic_space>
  ptr.masked_store %value, %ptr, %mask alignment = 32 : vector<4xf32>, !ptr.ptr<#ptr.generic_space>
  return
}

/// Test masked store operations with tensors
func.func @masked_store_ops_tensor(%value: tensor<8xi64>, %ptr: !ptr.ptr<#ptr.generic_space>, %mask: tensor<8xi1>) {
  ptr.masked_store %value, %ptr, %mask : tensor<8xi64>, !ptr.ptr<#ptr.generic_space>
  ptr.masked_store %value, %ptr, %mask alignment = 8 : tensor<8xi64>, !ptr.ptr<#ptr.generic_space>
  return
}

/// Test operations with LLVM address space
func.func @llvm_masked_ops(%ptr: !ptr.ptr<#llvm.address_space<3>>, %ptrs: vector<4x!ptr.ptr<#llvm.address_space<3>>>, 
                           %mask: vector<4xi1>, %value: vector<4xf32>, %passthrough: vector<4xf32>) -> vector<4xf32> {
  // Gather from shared memory (address space 3)
  %0 = ptr.gather %ptrs, %mask, %passthrough alignment = 4 : vector<4x!ptr.ptr<#llvm.address_space<3>>> -> vector<4xf32>
  // Scatter to shared memory
  ptr.scatter %value, %ptrs, %mask alignment = 4 : vector<4xf32>, vector<4x!ptr.ptr<#llvm.address_space<3>>>
  // Masked load from shared memory
  %1 = ptr.masked_load %ptr, %mask, %passthrough alignment = 4 : !ptr.ptr<#llvm.address_space<3>> -> vector<4xf32>
  // Masked store to shared memory
  ptr.masked_store %value, %ptr, %mask alignment = 4 : vector<4xf32>, !ptr.ptr<#llvm.address_space<3>>
  return %0 : vector<4xf32>
}
