// RUN: mlir-vulkan-runner %s --shared-libs=%mlir_lib_dir/libvulkan-runtime-wrappers%shlibext,%mlir_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK-COUNT-32: [2.2, 2.2, 2.2, 2.2]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
    #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_sub(%arg0 : memref<8x4x4xf32>, %arg1 : memref<4x4xf32>, %arg2 : memref<8x4x4xf32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[1, 1, 1]>: vector<3xi32>>} {
      %x = gpu.block_id x
      %y = gpu.block_id y
      %z = gpu.block_id z
      %1 = memref.load %arg0[%x, %y, %z] : memref<8x4x4xf32>
      %2 = memref.load %arg1[%y, %z] : memref<4x4xf32>
      %3 = arith.subf %1, %2 : f32
      memref.store %3, %arg2[%x, %y, %z] : memref<8x4x4xf32>
      gpu.return
    }
  }

  func.func @main() {
    %arg0 = memref.alloc() : memref<8x4x4xf32>
    %arg1 = memref.alloc() : memref<4x4xf32>
    %arg2 = memref.alloc() : memref<8x4x4xf32>
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    %value0 = arith.constant 0.0 : f32
    %value1 = arith.constant 3.3 : f32
    %value2 = arith.constant 1.1 : f32
    %arg3 = memref.cast %arg0 : memref<8x4x4xf32> to memref<?x?x?xf32>
    %arg4 = memref.cast %arg1 : memref<4x4xf32> to memref<?x?xf32>
    %arg5 = memref.cast %arg2 : memref<8x4x4xf32> to memref<?x?x?xf32>
    call @fillResource3DFloat(%arg3, %value1) : (memref<?x?x?xf32>, f32) -> ()
    call @fillResource2DFloat(%arg4, %value2) : (memref<?x?xf32>, f32) -> ()
    call @fillResource3DFloat(%arg5, %value0) : (memref<?x?x?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    %cst4 = arith.constant 4 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@kernel_sub
        blocks in (%cst8, %cst4, %cst4) threads in (%cst1, %cst1, %cst1)
        args(%arg0 : memref<8x4x4xf32>, %arg1 : memref<4x4xf32>, %arg2 : memref<8x4x4xf32>)
    %arg6 = memref.cast %arg5 : memref<?x?x?xf32> to memref<*xf32>
    call @printMemrefF32(%arg6) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource2DFloat(%0 : memref<?x?xf32>, %1 : f32)
  func.func private @fillResource3DFloat(%0 : memref<?x?x?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
