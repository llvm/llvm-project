// RUN: mlir-opt --split-input-file --convert-gpu-to-spirv %s | FileCheck %s

module attributes {gpu.container_module} {
  // CHECK-LABEL: spirv.module @{{.*}} GLSL450
  gpu.module @kernels [#spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>] {
    // CHECK: spirv.func @load_kernel
    // CHECK-SAME: %[[ARG:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>})
    gpu.func @load_kernel(%arg0: memref<12x4xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %c0 = arith.constant 0 : index
      // CHECK:             %[[PTR:.*]] = spirv.AccessChain %[[ARG]]{{\[}}{{%.*}}, {{%.*}}{{\]}}
      // CHECK-NEXT:        {{%.*}} = spirv.Load "StorageBuffer" %[[PTR]] : f32
      %0 = memref.load %arg0[%c0, %c0] : memref<12x4xf32>
      // CHECK:             spirv.Return
      gpu.return
    }
  }
}

// -----
// Checks that the `-convert-gpu-to-spirv` pass selects the first
// `spirv.target_env` from the `targets` array attribute attached to `gpu.module`.
module attributes {gpu.container_module} {
  // CHECK-LABEL: spirv.module @{{.*}} GLSL450
  // CHECK-SAME: #spirv.target_env<#spirv.vce<v1.4, [Shader], [SPV_KHR_storage_buffer_storage_class]>
  gpu.module @kernels [
    #spirv.target_env<#spirv.vce<v1.4, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>,
    #spirv.target_env<#spirv.vce<v1.0, [Kernel], []>, #spirv.resource_limits<>>,
    #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>] {
    // CHECK: spirv.func @load_kernel
    // CHECK-SAME: %[[ARG:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>})
    gpu.func @load_kernel(%arg0: memref<12x4xf32>) kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %c0 = arith.constant 0 : index
      // CHECK:             %[[PTR:.*]] = spirv.AccessChain %[[ARG]]{{\[}}{{%.*}}, {{%.*}}{{\]}}
      // CHECK-NEXT:        {{%.*}} = spirv.Load "StorageBuffer" %[[PTR]] : f32
      %0 = memref.load %arg0[%c0, %c0] : memref<12x4xf32>
      // CHECK:             spirv.Return
      gpu.return
    }
  }
}
