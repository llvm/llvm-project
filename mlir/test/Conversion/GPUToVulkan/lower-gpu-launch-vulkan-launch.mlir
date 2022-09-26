// RUN: mlir-opt %s -convert-gpu-launch-to-vulkan-launch | FileCheck %s

// CHECK: %[[resource:.*]] = memref.alloc() : memref<12xf32>
// CHECK: %[[index:.*]] = arith.constant 1 : index
// CHECK: call @vulkanLaunch(%[[index]], %[[index]], %[[index]], %[[resource]]) {spirv_blob = "{{.*}}", spirv_entry_point = "kernel"}

module attributes {gpu.container_module} {
  spirv.module Logical GLSL450 requires #spirv.vce<v1.0, [Shader], [SPIRV_KHR_storage_buffer_storage_class]> {
    spirv.GlobalVariable @kernel_arg_0 bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
    spirv.func @kernel() "None" attributes {workgroup_attributions = 0 : i64} {
      %0 = spirv.mlir.addressof @kernel_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
      %2 = spirv.Constant 0 : i32
      %3 = spirv.mlir.addressof @kernel_arg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
      %4 = spirv.AccessChain %0[%2, %2] : !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>, i32, i32
      %5 = spirv.Load "StorageBuffer" %4 : f32
      spirv.Return
    }
    spirv.EntryPoint "GLCompute" @kernel
    spirv.ExecutionMode @kernel "LocalSize", 1, 1, 1
  }
  gpu.module @kernels {
    gpu.func @kernel(%arg0: memref<12xf32>) kernel {
      gpu.return
    }
  }
  func.func @foo() {
    %0 = memref.alloc() : memref<12xf32>
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel
        blocks in(%c1, %c1, %c1)
        threads in(%c1, %c1, %c1)
        args(%0 : memref<12xf32>)
    return
  }
}
