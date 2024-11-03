// RUN: mlir-spirv-cpu-runner %s -e main --entry-point-result=void --shared-libs=%mlir_runner_utils,%mlir_test_spirv_cpu_runner_c_wrappers

// CHECK: [[[7.7,    0,    0], [7.7,    0,    0], [7.7,    0,    0]], [[0,    7.7,    0], [0,    7.7,    0], [0,    7.7,    0]], [[0,    0,    7.7], [0,    0,    7.7], [0,    0,    7.7]]]
module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_8bit_storage]>,
    #spirv.resource_limits<
     max_compute_workgroup_invocations = 128,
     max_compute_workgroup_size = [128, 128, 64]>>
} {
  gpu.module @kernels {
    gpu.func @sum(%arg0 : memref<3xf32>, %arg1 : memref<3x3xf32>, %arg2 :  memref<3x3x3xf32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %i0 = arith.constant 0 : index
      %i1 = arith.constant 1 : index
      %i2 = arith.constant 2 : index

      %x = memref.load %arg0[%i0] : memref<3xf32>
      %y = memref.load %arg1[%i0, %i0] : memref<3x3xf32>
      %sum = arith.addf %x, %y : f32

      memref.store %sum, %arg2[%i0, %i0, %i0] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i0, %i1, %i0] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i0, %i2, %i0] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i1, %i0, %i1] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i1, %i1, %i1] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i1, %i2, %i1] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i2, %i0, %i2] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i2, %i1, %i2] : memref<3x3x3xf32>
      memref.store %sum, %arg2[%i2, %i2, %i2] : memref<3x3x3xf32>
      gpu.return
    }
  }

  func.func @main() {
    %input1 = memref.alloc() : memref<3xf32>
    %input2 = memref.alloc() : memref<3x3xf32>
    %output = memref.alloc() : memref<3x3x3xf32>
    %0 = arith.constant 0.0 : f32
    %3 = arith.constant 3.4 : f32
    %4 = arith.constant 4.3 : f32
    %input1_casted = memref.cast %input1 : memref<3xf32> to memref<?xf32>
    %input2_casted = memref.cast %input2 : memref<3x3xf32> to memref<?x?xf32>
    %output_casted = memref.cast %output : memref<3x3x3xf32> to memref<?x?x?xf32>
    call @fillF32Buffer1D(%input1_casted, %3) : (memref<?xf32>, f32) -> ()
    call @fillF32Buffer2D(%input2_casted, %4) : (memref<?x?xf32>, f32) -> ()
    call @fillF32Buffer3D(%output_casted, %0) : (memref<?x?x?xf32>, f32) -> ()

    %one = arith.constant 1 : index
    gpu.launch_func @kernels::@sum
        blocks in (%one, %one, %one) threads in (%one, %one, %one)
        args(%input1 : memref<3xf32>, %input2 : memref<3x3xf32>, %output : memref<3x3x3xf32>)
    %result = memref.cast %output : memref<3x3x3xf32> to memref<*xf32>
    call @printMemrefF32(%result) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillF32Buffer1D(%arg0 : memref<?xf32>, %arg1 : f32)
  func.func private @fillF32Buffer2D(%arg0 : memref<?x?xf32>, %arg1 : f32)
  func.func private @fillF32Buffer3D(%arg0 : memref<?x?x?xf32>, %arg1 : f32)
  func.func private @printMemrefF32(%arg0 : memref<*xf32>)
}
