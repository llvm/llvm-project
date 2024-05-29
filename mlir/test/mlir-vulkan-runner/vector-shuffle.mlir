// RUN: mlir-vulkan-runner %s \
// RUN:  --shared-libs=%vulkan-runtime-wrappers,%mlir_runner_utils \
// RUN:  --entry-point-result=void | FileCheck %s

// CHECK: [2, 1, 3]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_vector_shuffle(%arg0 : memref<2xi32>, %arg1 : memref<2xi32>, %arg2 : memref<3xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %c0 = arith.constant 0 : index
      %vec0 = vector.load %arg0[%c0] : memref<2xi32>, vector<2xi32>
      %vec1 = vector.load %arg1[%c0] : memref<2xi32>, vector<2xi32>
      %result = vector.shuffle %vec0, %vec1[2, 1, 3] : vector<2xi32>, vector<2xi32>
      vector.store %result, %arg2[%c0] : memref<3xi32>, vector<3xi32>
      gpu.return
    }
  }

  func.func @main() {
    // Allocate 3 buffers.
    %buf0 = memref.alloc() : memref<2xi32>
    %buf1 = memref.alloc() : memref<2xi32>
    %buf2 = memref.alloc() : memref<3xi32>
    
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx4 = arith.constant 4 : index

    // Initialize input buffer
    %buf0_vals = arith.constant dense<[0, 1]> : vector<2xi32>
    %buf1_vals = arith.constant dense<[2, 3]> : vector<2xi32>
    vector.store %buf0_vals, %buf0[%idx0] : memref<2xi32>, vector<2xi32>
    vector.store %buf1_vals, %buf1[%idx0] : memref<2xi32>, vector<2xi32>

    // Initialize output buffer.
    %value0 = arith.constant 0 : i32
    %buf3 = memref.cast %buf2 : memref<3xi32> to memref<?xi32>
    call @fillResource1DInt(%buf3, %value0) : (memref<?xi32>, i32) -> ()

    gpu.launch_func @kernels::@kernel_vector_shuffle
        blocks in (%idx4, %idx1, %idx1) threads in (%idx1, %idx1, %idx1)
        args(%buf0 : memref<2xi32>, %buf1 : memref<2xi32>, %buf2 : memref<3xi32>)
    %buf4 = memref.cast %buf3 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf4) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
