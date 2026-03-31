// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -convert-gpu-to-spirv %s -o - | FileCheck %s

// Test workgroup memory load/store through gpu.func workgroup attributions.

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    // CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__load_store_workgroup_0 : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32>)>, Workgroup>
    // CHECK-LABEL: spirv.func @load_store_workgroup
    // CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>
    gpu.func @load_store_workgroup(
      %arg0: memref<256xf32, #spirv.storage_class<StorageBuffer>>,
      %arg1: memref<256xf32, #spirv.storage_class<StorageBuffer>>)
      workgroup(%smem: memref<256xf32, #spirv.storage_class<Workgroup>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>} {
      // CHECK: %[[WG:.*]] = spirv.mlir.addressof @__workgroup_mem__load_store_workgroup_0
      // CHECK: spirv.AccessChain %[[ARG0]][{{%.*}}, {{%.*}}]
      // CHECK: spirv.Load "StorageBuffer"
      // CHECK: spirv.AccessChain %[[WG]][{{%.*}}, {{%.*}}]
      // CHECK: spirv.Store "Workgroup"
      // CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
      // CHECK: spirv.AccessChain %[[WG]][{{%.*}}, {{%.*}}]
      // CHECK: spirv.Load "Workgroup"
      // CHECK: spirv.AccessChain %[[ARG1]][{{%.*}}, {{%.*}}]
      // CHECK: spirv.Store "StorageBuffer"
      %c0 = arith.constant 0 : index
      %val = memref.load %arg0[%c0] : memref<256xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %val, %smem[%c0] : memref<256xf32, #spirv.storage_class<Workgroup>>
      gpu.barrier
      %val2 = memref.load %smem[%c0] : memref<256xf32, #spirv.storage_class<Workgroup>>
      memref.store %val2, %arg1[%c0] : memref<256xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    %1 = "op"() : () -> (memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@load_store_workgroup
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<256xf32, #spirv.storage_class<StorageBuffer>>,
             %1 : memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}

// -----

// Test with 2D workgroup memory.
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // 16 x 16 = 256 elements
    // CHECK-DAG: spirv.GlobalVariable @__workgroup_mem__kernel_2d_wg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32>)>, Workgroup>
    // CHECK-LABEL: spirv.func @kernel_2d_wg
    gpu.func @kernel_2d_wg(
      %arg0: memref<16x16xf32, #spirv.storage_class<StorageBuffer>>)
      workgroup(%smem: memref<16x16xf32, #spirv.storage_class<Workgroup>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 16, 1]>} {
      %c0 = arith.constant 0 : index
      %val = memref.load %arg0[%c0, %c0] : memref<16x16xf32, #spirv.storage_class<StorageBuffer>>
      memref.store %val, %smem[%c0, %c0] : memref<16x16xf32, #spirv.storage_class<Workgroup>>
      gpu.return
    }
  }
}
