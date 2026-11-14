// Make sure that arith.ceildivui, arith.ceildivsi, and arith.floordivsi
// produce expected results after conversion to SPIR-V.

// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-runner - \
// RUN:     --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils \
// RUN:     --entry-point-result=void | FileCheck %s

// CHECK: [4, 2147483645, 1, 1, 2, 1431655764, 0, 0, 1, 1073741824, 1073741824, 715827883, 1, 2, -2147483648, 1]
// CHECK: [4, -3, -3, 4, 2, -2, 0, 0, 1, 1073741824, -1073741824, -715827882, -2147483647, -1, 0, -1]
// CHECK: [3, -4, -4, 3, 2, -2, 0, 0, 1, 1073741823, -1073741824, -715827883, -2147483647, -2, -1, -1]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_div(%arg0 : memref<16xi32>, %arg1 : memref<16xi32>, %arg2 : memref<16xi32>, %arg3 : memref<16xi32>, %arg4 : memref<16xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %0 = gpu.block_id x
      %lhs = memref.load %arg0[%0] : memref<16xi32>
      %rhs = memref.load %arg1[%0] : memref<16xi32>
      %ceilu = arith.ceildivui %lhs, %rhs : i32
      %ceils = arith.ceildivsi %lhs, %rhs : i32
      %floors = arith.floordivsi %lhs, %rhs : i32
      memref.store %ceilu, %arg2[%0] : memref<16xi32>
      memref.store %ceils, %arg3[%0] : memref<16xi32>
      memref.store %floors, %arg4[%0] : memref<16xi32>
      gpu.return
    }
  }

  func.func @main() {
    %buf0 = memref.alloc() : memref<16xi32>
    %buf1 = memref.alloc() : memref<16xi32>
    %buf2 = memref.alloc() : memref<16xi32>
    %buf3 = memref.alloc() : memref<16xi32>
    %buf4 = memref.alloc() : memref<16xi32>
    %i32_0 = arith.constant 0 : i32

    // Initialize output buffers.
    %buf5 = memref.cast %buf2 : memref<16xi32> to memref<?xi32>
    %buf6 = memref.cast %buf3 : memref<16xi32> to memref<?xi32>
    %buf7 = memref.cast %buf4 : memref<16xi32> to memref<?xi32>
    call @fillResource1DInt(%buf5, %i32_0) : (memref<?xi32>, i32) -> ()
    call @fillResource1DInt(%buf6, %i32_0) : (memref<?xi32>, i32) -> ()
    call @fillResource1DInt(%buf7, %i32_0) : (memref<?xi32>, i32) -> ()

    %idx_0 = arith.constant 0 : index
    %idx_1 = arith.constant 1 : index
    %idx_16 = arith.constant 16 : index

    // Initialize input buffers. Skip division by zero and INT_MIN / -1.
    %lhs_vals = arith.constant dense<[7, -7,  7, -7, 6, -6, 0,  0, 1, 2147483647, -2147483648, -2147483648, 2147483647, -2147483648, -1,  1]> : vector<16xi32>
    %rhs_vals = arith.constant dense<[2,  2, -2, -2, 3,  3, 5, -5, 1,          2,           2,           3,         -1,  2147483647,  2, -1]> : vector<16xi32>
    vector.store %lhs_vals, %buf0[%idx_0] : memref<16xi32>, vector<16xi32>
    vector.store %rhs_vals, %buf1[%idx_0] : memref<16xi32>, vector<16xi32>

    gpu.launch_func @kernels::@kernel_div
        blocks in (%idx_16, %idx_1, %idx_1) threads in (%idx_1, %idx_1, %idx_1)
        args(%buf0 : memref<16xi32>, %buf1 : memref<16xi32>, %buf2 : memref<16xi32>, %buf3 : memref<16xi32>, %buf4 : memref<16xi32>)
    %buf_ceilu = memref.cast %buf5 : memref<?xi32> to memref<*xi32>
    %buf_ceils = memref.cast %buf6 : memref<?xi32> to memref<*xi32>
    %buf_floors = memref.cast %buf7 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf_ceilu) : (memref<*xi32>) -> ()
    call @printMemrefI32(%buf_ceils) : (memref<*xi32>) -> ()
    call @printMemrefI32(%buf_floors) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
