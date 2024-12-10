// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-vulkan-runner - \
// RUN:     --shared-libs=%vulkan-runtime-wrappers,%mlir_runner_utils \
// RUN:     --entry-point-result=void | FileCheck %s

// CHECK: [0, 2]
// CHECK: [1, 3]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_vector_deinterleave(%arg0 : memref<4xi32>, %arg1 : memref<2xi32>, %arg2 : memref<2xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {

      %idx0 = arith.constant 0 : index
      %idx1 = arith.constant 1 : index
      %idx2 = arith.constant 2 : index
      %idx3 = arith.constant 3 : index

      %src = arith.constant dense<[0, 0, 0, 0]> : vector<4xi32>

      %val0 = memref.load %arg0[%idx0] : memref<4xi32>
      %val1 = memref.load %arg0[%idx1] : memref<4xi32>
      %val2 = memref.load %arg0[%idx2] : memref<4xi32>
      %val3 = memref.load %arg0[%idx3] : memref<4xi32>

      %src0 = vector.insert %val0, %src[0] : i32 into vector<4xi32>
      %src1 = vector.insert %val1, %src0[1] : i32 into vector<4xi32>
      %src2 = vector.insert %val2, %src1[2] : i32 into vector<4xi32>
      %src3 = vector.insert %val3, %src2[3] : i32 into vector<4xi32>

      %res0, %res1 = vector.deinterleave %src3 : vector<4xi32> -> vector<2xi32>

      %res0_0 = vector.extract %res0[0] : i32 from vector<2xi32>
      %res0_1 = vector.extract %res0[1] : i32 from vector<2xi32>
      %res1_0 = vector.extract %res1[0] : i32 from vector<2xi32>
      %res1_1 = vector.extract %res1[1] : i32 from vector<2xi32>

      memref.store %res0_0, %arg1[%idx0]: memref<2xi32>
      memref.store %res0_1, %arg1[%idx1]: memref<2xi32>
      memref.store %res1_0, %arg2[%idx0]: memref<2xi32>
      memref.store %res1_1, %arg2[%idx1]: memref<2xi32>

      gpu.return
    }
  }

  func.func @main() {
    %idx0 = arith.constant 0 : index
    %idx1 = arith.constant 1 : index
    %idx4 = arith.constant 4 : index

    // Allocate 3 buffers.
    %buf0 = memref.alloc() : memref<4xi32>
    %buf1 = memref.alloc() : memref<2xi32>
    %buf2 = memref.alloc() : memref<2xi32>

    // Initialize input buffer.
    %buf0_vals = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    vector.store %buf0_vals, %buf0[%idx0] : memref<4xi32>, vector<4xi32>

    // Initialize output buffers.
    %value0 = arith.constant 0 : i32
    %buf3 = memref.cast %buf1 : memref<2xi32> to memref<?xi32>
    %buf4 = memref.cast %buf2 : memref<2xi32> to memref<?xi32>
    call @fillResource1DInt(%buf3, %value0) : (memref<?xi32>, i32) -> ()
    call @fillResource1DInt(%buf4, %value0) : (memref<?xi32>, i32) -> ()

    gpu.launch_func @kernels::@kernel_vector_deinterleave
        blocks in (%idx4, %idx1, %idx1) threads in (%idx1, %idx1, %idx1)
        args(%buf0 : memref<4xi32>, %buf1 : memref<2xi32>, %buf2 : memref<2xi32>)
    %buf5 = memref.cast %buf3 : memref<?xi32> to memref<*xi32>
    %buf6 = memref.cast %buf4 : memref<?xi32> to memref<*xi32>
    call @printMemrefI32(%buf5) : (memref<*xi32>) -> ()
    call @printMemrefI32(%buf6) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}
