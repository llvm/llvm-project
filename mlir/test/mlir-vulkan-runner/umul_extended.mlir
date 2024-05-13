// Make sure that unsigned extended multiplication produces expected results
// with and without expansion to primitive mul/add ops for WebGPU.

// RUN: mlir-vulkan-runner %s \
// RUN:  --shared-libs=%vulkan-runtime-wrappers,%mlir_runner_utils \
// RUN:  --entry-point-result=void | FileCheck %s

// RUN: mlir-vulkan-runner %s --vulkan-runner-spirv-webgpu-prepare \
// RUN:  --shared-libs=%vulkan-runtime-wrappers,%mlir_runner_utils \
// RUN:  --entry-point-result=void | FileCheck %s

// CHECK: [0, 1, -2,  1, 1048560, -87620295, -131071, -49]
// CHECK: [0, 0,  1, -2,       0,     65534, -131070,   6]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.4, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<8xi32>, %arg1 : memref<8xi32>, %arg2 : memref<8xi32>, %arg3 : memref<8xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %0 = gpu.block_id x
      %lhs = memref.load %arg0[%0] : memref<8xi32>
      %rhs = memref.load %arg1[%0] : memref<8xi32>
      %low, %hi = arith.mului_extended %lhs, %rhs : i32
      memref.store %low, %arg2[%0] : memref<8xi32>
      memref.store %hi, %arg3[%0] : memref<8xi32>
      gpu.return
    }
  }

  func.func @main() {
    %buf0 = memref.alloc() : memref<8xi32>
    %buf1 = memref.alloc() : memref<8xi32>
    %buf2 = memref.alloc() : memref<8xi32>
    %buf3 = memref.alloc() : memref<8xi32>
    %i32_0 = arith.constant 0 : i32

    // Initialize output buffers.
    %buf4 = memref.cast %buf2 : memref<8xi32> to memref<?xi32>
    %buf5 = memref.cast %buf3 : memref<8xi32> to memref<?xi32>
    call @fillResource1DInt(%buf4, %i32_0) : (memref<?xi32>, i32) -> ()
    call @fillResource1DInt(%buf5, %i32_0) : (memref<?xi32>, i32) -> ()

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %idx_8 = arith.constant 8 : index

    // Initialize input buffers.
    %lhs_vals = arith.constant dense<[0, 1, -1,  -1,  65535,  65535, -65535,  7]> : vector<8xi32>
    %rhs_vals = arith.constant dense<[0, 1,  2,  -1,     16,  -1337, -65535, -7]> : vector<8xi32>
    vector.store %lhs_vals, %buf0[%idx_0] : memref<8xi32>, vector<8xi32>
    vector.store %rhs_vals, %buf1[%idx_0] : memref<8xi32>, vector<8xi32>

    gpu.launch_func @kernels::@kernel_add
        blocks in (%idx_8, %idx_1, %idx_1) threads in (%idx_1, %idx_1, %idx_1)
        args(%buf0 : memref<8xi32>, %buf1 : memref<8xi32>, %buf2 : memref<8xi32>, %buf3 : memref<8xi32>)
    %buf_low = memref.cast %buf4 : memref<?xi32> to memref<*xi32>
    %buf_hi = memref.cast %buf5 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf_low) : (memref<*xi32>) -> ()
    call @printMemrefI32(%buf_hi) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
