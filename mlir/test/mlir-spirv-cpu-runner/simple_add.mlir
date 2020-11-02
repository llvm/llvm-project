// RUN: mlir-spirv-cpu-runner %s -e main --entry-point-result=void --shared-libs=%mlir_runner_utils_dir/libmlir_runner_utils%shlibext,%spirv_wrapper_library_dir/libmlir_test_spirv_cpu_runner_c_wrappers%shlibext

// CHECK: [[[7.7,    0,    0], [7.7,    0,    0], [7.7,    0,    0]], [[0,    7.7,    0], [0,    7.7,    0], [0,    7.7,    0]], [[0,    0,    7.7], [0,    0,    7.7], [0,    0,    7.7]]]
module attributes {
  gpu.container_module,
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_8bit_storage]>,
    {max_compute_workgroup_invocations = 128 : i32,
     max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>
} {
  gpu.module @kernels {
    gpu.func @sum(%arg0 : memref<3xf32>, %arg1 : memref<3x3xf32>, %arg2 :  memref<3x3x3xf32>)
      kernel attributes { spv.entry_point_abi = {local_size = dense<[1, 1, 1]>: vector<3xi32>}} {
      %i0 = constant 0 : index
      %i1 = constant 1 : index
      %i2 = constant 2 : index

      %x = load %arg0[%i0] : memref<3xf32>
      %y = load %arg1[%i0, %i0] : memref<3x3xf32>
      %sum = addf %x, %y : f32

      store %sum, %arg2[%i0, %i0, %i0] : memref<3x3x3xf32>
      store %sum, %arg2[%i0, %i1, %i0] : memref<3x3x3xf32>
      store %sum, %arg2[%i0, %i2, %i0] : memref<3x3x3xf32>
      store %sum, %arg2[%i1, %i0, %i1] : memref<3x3x3xf32>
      store %sum, %arg2[%i1, %i1, %i1] : memref<3x3x3xf32>
      store %sum, %arg2[%i1, %i2, %i1] : memref<3x3x3xf32>
      store %sum, %arg2[%i2, %i0, %i2] : memref<3x3x3xf32>
      store %sum, %arg2[%i2, %i1, %i2] : memref<3x3x3xf32>
      store %sum, %arg2[%i2, %i2, %i2] : memref<3x3x3xf32>
      gpu.return
    }
  }

  func @main() {
    %input1 = alloc() : memref<3xf32>
    %input2 = alloc() : memref<3x3xf32>
    %output = alloc() : memref<3x3x3xf32>
    %0 = constant 0.0 : f32
    %3 = constant 3.4 : f32
    %4 = constant 4.3 : f32
    %input1_casted = memref_cast %input1 : memref<3xf32> to memref<?xf32>
    %input2_casted = memref_cast %input2 : memref<3x3xf32> to memref<?x?xf32>
    %output_casted = memref_cast %output : memref<3x3x3xf32> to memref<?x?x?xf32>
    call @fillF32Buffer1D(%input1_casted, %3) : (memref<?xf32>, f32) -> ()
    call @fillF32Buffer2D(%input2_casted, %4) : (memref<?x?xf32>, f32) -> ()
    call @fillF32Buffer3D(%output_casted, %0) : (memref<?x?x?xf32>, f32) -> ()

    %one = constant 1 : index
    gpu.launch_func @kernels::@sum
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%input1 : memref<3xf32>, %input2 : memref<3x3xf32>, %output : memref<3x3x3xf32>)
    %result = memref_cast %output : memref<3x3x3xf32> to memref<*xf32>
    call @print_memref_f32(%result) : (memref<*xf32>) -> ()
    return
  }
  func @fillF32Buffer1D(%arg0 : memref<?xf32>, %arg1 : f32)
  func @fillF32Buffer2D(%arg0 : memref<?x?xf32>, %arg1 : f32)
  func @fillF32Buffer3D(%arg0 : memref<?x?x?xf32>, %arg1 : f32)
  func @print_memref_f32(%arg0 : memref<*xf32>)
}
