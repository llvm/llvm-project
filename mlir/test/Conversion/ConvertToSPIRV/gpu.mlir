// RUN: mlir-opt -test-convert-to-spirv -split-input-file %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @all_reduce
  // CHECK-SAME: (%[[ARG0:.*]]: f32)
  // CHECK: %{{.*}} = spirv.GroupNonUniformFAdd <Workgroup> <Reduce> %[[ARG0]] : f32 -> f32
  gpu.func @all_reduce(%arg0 : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %reduced = gpu.all_reduce add %arg0 {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.3, [Kernel, Addresses, Groups, GroupNonUniformArithmetic, GroupUniformArithmeticKHR], []>, #spirv.resource_limits<>>
} {

gpu.module @kernels {
  // CHECK-LABEL: spirv.func @subgroup_reduce
  // CHECK-SAME: (%[[ARG0:.*]]: f32)
  // CHECK: %{{.*}} = spirv.GroupNonUniformFAdd <Subgroup> <Reduce> %[[ARG0]] : f32 -> f32
  gpu.func @subgroup_reduce(%arg0 : f32) kernel
    attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
    %reduced = gpu.subgroup_reduce add %arg0 {} : (f32) -> (f32)
    gpu.return
  }
}

}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {

  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  // CHECK-LABEL: spirv.func @load_store
  // CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
  // CHECK-SAME: %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
  // CHECK-SAME: %[[ARG2:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}
  gpu.module @kernels {
    gpu.func @load_store(%arg0: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg2: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      // CHECK: %[[PTR1:.*]] = spirv.AccessChain %[[ARG0]]
      // CHECK-NEXT: spirv.Load "StorageBuffer" %[[PTR1]]
      // CHECK: %[[PTR2:.*]] = spirv.AccessChain %[[ARG1]]
      // CHECK-NEXT: spirv.Load "StorageBuffer" %[[PTR2]]
      // CHECK: spirv.FAdd
      // CHECK: %[[PTR3:.*]] = spirv.AccessChain %[[ARG2]]
      // CHECK-NEXT: spirv.Store "StorageBuffer" %[[PTR3]]
      %0 = gpu.block_id x
      %1 = gpu.block_id y
      %2 = gpu.block_id z
      %3 = gpu.thread_id x
      %4 = gpu.thread_id y
      %5 = gpu.thread_id z
      %6 = gpu.grid_dim x
      %7 = gpu.grid_dim y
      %8 = gpu.grid_dim z
      %9 = gpu.block_dim x
      %10 = gpu.block_dim y
      %11 = gpu.block_dim z
      %12 = arith.addi %arg3, %0 : index
      %13 = arith.addi %arg4, %3 : index
      %14 = memref.load %arg0[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      %15 = memref.load %arg1[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      %16 = arith.addf %14, %15 : f32
      memref.store %16, %arg2[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
}
