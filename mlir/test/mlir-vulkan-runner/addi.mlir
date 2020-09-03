// RUN: mlir-vulkan-runner %s --shared-libs=%vulkan_wrapper_library_dir/libvulkan-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK-COUNT-64: [3, 3, 3, 3, 3, 3, 3, 3]
module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {
  gpu.module @kernels {
    gpu.func @kernel_addi(%arg0 : memref<8xi32>, %arg1 : memref<8x8xi32>, %arg2 : memref<8x8x8xi32>)
      kernel attributes { spv.entry_point_abi = {local_size = dense<[1, 1, 1]>: vector<3xi32>}} {
      %x = "gpu.block_id"() {dimension = "x"} : () -> index
      %y = "gpu.block_id"() {dimension = "y"} : () -> index
      %z = "gpu.block_id"() {dimension = "z"} : () -> index
      %0 = load %arg0[%x] : memref<8xi32>
      %1 = load %arg1[%y, %x] : memref<8x8xi32>
      %2 = addi %0, %1 : i32
      store %2, %arg2[%z, %y, %x] : memref<8x8x8xi32>
      gpu.return
    }
  }

  func @main() {
    %arg0 = alloc() : memref<8xi32>
    %arg1 = alloc() : memref<8x8xi32>
    %arg2 = alloc() : memref<8x8x8xi32>
    %value0 = constant 0 : i32
    %value1 = constant 1 : i32
    %value2 = constant 2 : i32
    %arg3 = memref_cast %arg0 : memref<8xi32> to memref<?xi32>
    %arg4 = memref_cast %arg1 : memref<8x8xi32> to memref<?x?xi32>
    %arg5 = memref_cast %arg2 : memref<8x8x8xi32> to memref<?x?x?xi32>
    call @fillResource1DInt(%arg3, %value1) : (memref<?xi32>, i32) -> ()
    call @fillResource2DInt(%arg4, %value2) : (memref<?x?xi32>, i32) -> ()
    call @fillResource3DInt(%arg5, %value0) : (memref<?x?x?xi32>, i32) -> ()

    %cst1 = constant 1 : index
    %cst8 = constant 8 : index
    "gpu.launch_func"(%cst8, %cst8, %cst8, %cst1, %cst1, %cst1, %arg0, %arg1, %arg2) { kernel = @kernels::@kernel_addi }
        : (index, index, index, index, index, index, memref<8xi32>, memref<8x8xi32>, memref<8x8x8xi32>) -> ()
    %arg6 = memref_cast %arg5 : memref<?x?x?xi32> to memref<*xi32>
    call @print_memref_i32(%arg6) : (memref<*xi32>) -> ()
    return
  }
  func @fillResource1DInt(%0 : memref<?xi32>, %1 : i32)
  func @fillResource2DInt(%0 : memref<?x?xi32>, %1 : i32)
  func @fillResource3DInt(%0 : memref<?x?x?xi32>, %1 : i32)
  func @print_memref_i32(%ptr : memref<*xi32>)
}

