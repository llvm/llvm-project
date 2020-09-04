// RUN: mlir-vulkan-runner %s --shared-libs=%vulkan_wrapper_library_dir/libvulkan-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: Compute shader execution time
// CHECK: Command buffer submit time
// CHECK: Wait idle time

module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<16384xf32>, %arg1 : memref<16384xf32>, %arg2 : memref<16384xf32>)
      kernel attributes { spv.entry_point_abi = {local_size = dense<[128, 1, 1]>: vector<3xi32> }} {
      %bid = "gpu.block_id"() {dimension = "x"} : () -> index
      %tid = "gpu.thread_id"() {dimension = "x"} : () -> index
      %cst = constant 128 : index
      %b = muli %bid, %cst : index
      %0 = addi %b, %tid : index
      %1 = load %arg0[%0] : memref<16384xf32>
      %2 = load %arg1[%0] : memref<16384xf32>
      %3 = addf %1, %2 : f32
      store %3, %arg2[%0] : memref<16384xf32>
      gpu.return
    }
  }

  func @main() {
    %arg0 = alloc() : memref<16384xf32>
    %arg1 = alloc() : memref<16384xf32>
    %arg2 = alloc() : memref<16384xf32>
    %0 = constant 0 : i32
    %1 = constant 1 : i32
    %2 = constant 2 : i32
    %value0 = constant 0.0 : f32
    %value1 = constant 1.1 : f32
    %value2 = constant 2.2 : f32
    %arg3 = memref_cast %arg0 : memref<16384xf32> to memref<?xf32>
    %arg4 = memref_cast %arg1 : memref<16384xf32> to memref<?xf32>
    %arg5 = memref_cast %arg2 : memref<16384xf32> to memref<?xf32>
    call @fillResource1DFloat(%arg3, %value1) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg4, %value2) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%arg5, %value0) : (memref<?xf32>, f32) -> ()

    %cst1 = constant 1 : index
    %cst128 = constant 128 : index
    "gpu.launch_func"(%cst128, %cst1, %cst1, %cst128, %cst1, %cst1, %arg0, %arg1, %arg2) { kernel = @kernels::@kernel_add }
        : (index, index, index, index, index, index, memref<16384xf32>, memref<16384xf32>, memref<16384xf32>) -> ()
    %arg6 = memref_cast %arg5 : memref<?xf32> to memref<*xf32>
    return
  }
  func @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func @print_memref_f32(%ptr : memref<*xf32>)
}

