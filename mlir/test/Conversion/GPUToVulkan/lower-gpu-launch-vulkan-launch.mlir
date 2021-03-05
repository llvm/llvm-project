// RUN: mlir-opt %s -convert-gpu-launch-to-vulkan-launch | FileCheck %s

// CHECK: %[[resource:.*]] = alloc() : memref<12xf32>
// CHECK: %[[index:.*]] = constant 1 : index
// CHECK: call @vulkanLaunch(%[[index]], %[[index]], %[[index]], %[[resource]]) {spirv_blob = "{{.*}}", spirv_entry_point = "kernel"}

module attributes {gpu.container_module} {
  spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]> {
    spv.GlobalVariable @kernel_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer>
    spv.func @kernel() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spv.mlir.addressof @kernel_arg_0 : !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer>
      %2 = spv.Constant 0 : i32
      %3 = spv.mlir.addressof @kernel_arg_0 : !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer>
      %4 = spv.AccessChain %0[%2, %2] : !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      %5 = spv.Load "StorageBuffer" %4 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @kernel
    spv.ExecutionMode @kernel "LocalSize", 1, 1, 1
  }
  gpu.module @kernels {
    gpu.func @kernel(%arg0: memref<12xf32>) kernel {
      gpu.return
    }
  }
  func @foo() {
    %0 = alloc() : memref<12xf32>
    %c1 = constant 1 : index
    gpu.launch_func @kernels::@kernel
        blocks in(%c1, %c1, %c1)
        threads in(%c1, %c1, %c1)
        args(%0 : memref<12xf32>)
    return
  }
}
