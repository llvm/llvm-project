// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK-COUNT-64: [3, 3, 3, 3, 3, 3, 3, 3]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_addi(%arg0 : memref<8xi32>, %arg1 : memref<8x8xi32>, %arg2 : memref<8x8x8xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %x = gpu.block_id x
      %y = gpu.block_id y
      %z = gpu.block_id z
      %0 = memref.load %arg0[%x] : memref<8xi32>
      %1 = memref.load %arg1[%y, %x] : memref<8x8xi32>
      %2 = arith.addi %0, %1 : i32
      memref.store %2, %arg2[%z, %y, %x] : memref<8x8x8xi32>
      gpu.return
    }
  }

  func.func @main() {
    %arg0 = memref.alloc() : memref<8xi32>
    %arg1 = memref.alloc() : memref<8x8xi32>
    %arg2 = memref.alloc() : memref<8x8x8xi32>
    %value0 = arith.constant 0 : i32
    %value1 = arith.constant 1 : i32
    %value2 = arith.constant 2 : i32
    %arg3 = memref.cast %arg0 : memref<8xi32> to memref<?xi32>
    %arg4 = memref.cast %arg1 : memref<8x8xi32> to memref<?x?xi32>
    %arg5 = memref.cast %arg2 : memref<8x8x8xi32> to memref<?x?x?xi32>
    call @fillResource1DInt(%arg3, %value1) : (memref<?xi32>, i32) -> ()
    call @fillResource2DInt(%arg4, %value2) : (memref<?x?xi32>, i32) -> ()
    call @fillResource3DInt(%arg5, %value0) : (memref<?x?x?xi32>, i32) -> ()

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@kernel_addi
        blocks in (%cst8, %cst8, %cst8) threads in (%cst1, %cst1, %cst1)
        args(%arg0 : memref<8xi32>, %arg1 : memref<8x8xi32>, %arg2 : memref<8x8x8xi32>)
    %arg6 = memref.cast %arg5 : memref<?x?x?xi32> to memref<*xi32>
    call @printMemrefI32(%arg6) : (memref<*xi32>) -> ()
    return
  }
  func.func private @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func.func private @fillResource2DInt(%0 : memref<?x?xi32>, %1 : i32)
  func.func private @fillResource3DInt(%0 : memref<?x?x?xi32>, %1 : i32)
  func.func private @printMemrefI32(%ptr : memref<*xi32>)
}

