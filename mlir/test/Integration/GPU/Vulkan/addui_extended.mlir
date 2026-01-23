// Make sure that addition with carry produces expected results
// with and without expansion to primitive add/cmp ops for WebGPU.

// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-runner - \
// RUN:     --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils \
// RUN:     --entry-point-result=void | FileCheck %s

// RUN: mlir-opt %s -test-vulkan-runner-pipeline=spirv-webgpu-prepare \
// RUN:   | mlir-runner - \
// RUN:     --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils \
// RUN:     --entry-point-result=void | FileCheck %s

// CHECK: [0, 42, 0, 42]
// CHECK: [1, 0, 1, 1]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.4, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<4xi32>, %arg1 : memref<4xi32>, %arg2 : memref<4xi32>, %arg3 : memref<4xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %0 = gpu.block_id x
      %lhs = memref.load %arg0[%0] : memref<4xi32>
      %rhs = memref.load %arg1[%0] : memref<4xi32>
      %sum, %carry = arith.addui_extended %lhs, %rhs : i32, i1

      %carry_i32 = arith.extui %carry : i1 to i32

      memref.store %sum, %arg2[%0] : memref<4xi32> memref.store %carry_i32, %arg3[%0] : memref<4xi32>
      gpu.return
    }
  }

  func.func @main() {
    %buf0 = memref.alloc() : memref<4xi32>
    %buf1 = memref.alloc() : memref<4xi32>
    %buf2 = memref.alloc() : memref<4xi32>
    %buf3 = memref.alloc() : memref<4xi32>
    %i32_0 = arith.constant 0 : i32

    // Initialize output buffers.
    %buf4 = memref.cast %buf2 : memref<4xi32> to memref<?xi32>
    %buf5 = memref.cast %buf3 : memref<4xi32> to memref<?xi32>
    call @fillResource1DInt(%buf4, %i32_0) : (memref<?xi32>, i32) -> ()
    call @fillResource1DInt(%buf5, %i32_0) : (memref<?xi32>, i32) -> ()

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %idx_4 = arith.constant 4 : index

    // Initialize input buffers.
    %lhs_vals = arith.constant dense<[-1, 24, 4294967295, 43]> : vector<4xi32>
    %rhs_vals = arith.constant dense<[1, 18, 1, 4294967295]> : vector<4xi32>
    vector.store %lhs_vals, %buf0[%idx_0] : memref<4xi32>, vector<4xi32>
    vector.store %rhs_vals, %buf1[%idx_0] : memref<4xi32>, vector<4xi32>

    gpu.launch_func @kernels::@kernel_add
        blocks in (%idx_4, %idx_1, %idx_1) threads in (%idx_1, %idx_1, %idx_1)
        args(%buf0 : memref<4xi32>, %buf1 : memref<4xi32>, %buf2 : memref<4xi32>, %buf3 : memref<4xi32>)
    %buf_sum = memref.cast %buf4 : memref<?xi32> to memref<*xi32>
    %buf_carry = memref.cast %buf5 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf_sum) : (memref<*xi32>) -> ()
    call @printMemrefI32(%buf_carry) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
