// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-runner - \
// RUN:     --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils \
// RUN:     --entry-point-result=void | FileCheck %s

// CHECK: [0, 2, 1, 3]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_vector_interleave(%arg0 : memref<2xi32>, %arg1 : memref<2xi32>, %arg2 : memref<4xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %idx0 = arith.constant 0 : index
      %idx1 = arith.constant 1 : index
      %idx2 = arith.constant 2 : index
      %idx3 = arith.constant 3 : index
      %idx4 = arith.constant 4 : index

      %lhs = arith.constant dense<[0, 0]> : vector<2xi32>
      %rhs = arith.constant dense<[0, 0]> : vector<2xi32>

      %val0 = memref.load %arg0[%idx0] : memref<2xi32>
      %val1 = memref.load %arg0[%idx1] : memref<2xi32>
      %val2 = memref.load %arg1[%idx0] : memref<2xi32>
      %val3 = memref.load %arg1[%idx1] : memref<2xi32>

      %lhs0 = vector.insertelement %val0, %lhs[%idx0 : index] : vector<2xi32>
      %lhs1 = vector.insertelement %val1, %lhs0[%idx1 : index] : vector<2xi32>
      %rhs0 = vector.insertelement %val2, %rhs[%idx0 : index] : vector<2xi32>
      %rhs1 = vector.insertelement %val3, %rhs0[%idx1 : index] : vector<2xi32>

      %interleave = vector.interleave %lhs1, %rhs1 : vector<2xi32> -> vector<4xi32>

      %res0 = vector.extractelement %interleave[%idx0 : index] : vector<4xi32>
      %res1 = vector.extractelement %interleave[%idx1 : index] : vector<4xi32>
      %res2 = vector.extractelement %interleave[%idx2 : index] : vector<4xi32>
      %res3 = vector.extractelement %interleave[%idx3 : index] : vector<4xi32>

      memref.store %res0, %arg2[%idx0]: memref<4xi32>
      memref.store %res1, %arg2[%idx1]: memref<4xi32>
      memref.store %res2, %arg2[%idx2]: memref<4xi32>
      memref.store %res3, %arg2[%idx3]: memref<4xi32>

      gpu.return
    }
  }

  func.func @main() {
    // Allocate 3 buffers.
    %buf0 = memref.alloc() : memref<2xi32>
    %buf1 = memref.alloc() : memref<2xi32>
    %buf2 = memref.alloc() : memref<4xi32>

    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx4 = arith.constant 4 : index

    // Initialize input buffer.
    %buf0_vals = arith.constant dense<[0, 1]> : vector<2xi32>
    %buf1_vals = arith.constant dense<[2, 3]> : vector<2xi32>
    vector.store %buf0_vals, %buf0[%idx0] : memref<2xi32>, vector<2xi32>
    vector.store %buf1_vals, %buf1[%idx0] : memref<2xi32>, vector<2xi32>

    // Initialize output buffer.
    %value0 = arith.constant 0 : i32
    %buf3 = memref.cast %buf2 : memref<4xi32> to memref<?xi32>
    call @fillResource1DInt(%buf3, %value0) : (memref<?xi32>, i32) -> ()

    gpu.launch_func @kernels::@kernel_vector_interleave
        blocks in (%idx4, %idx1, %idx1) threads in (%idx1, %idx1, %idx1)
        args(%buf0 : memref<2xi32>, %buf1 : memref<2xi32>, %buf2 : memref<4xi32>)
    %buf4 = memref.cast %buf3 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf4) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
