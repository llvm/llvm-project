// RUN: mlir-opt %s \
// RUN: -gpu-kernel-outlining \
// RUN: -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_70})' \
// RUN: --convert-scf-to-std -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s
// Test case to check the working of Tensor cores on Nvidia GPUs. The kernel has already
// been outlined to prevent crashing due to introduction of an empty basic block by --gpu-
// kernel-outling.
module attributes {gpu.container_module}  {
  func @main() {
    %0 = memref.alloc() : memref<16x16xf16>
    %22 = memref.alloc() : memref<16x16xf32>
    %1 = memref.alloc() : memref<16x16xf32>

    %f1 = constant 1.0e+00 : f16
    %f0 = constant 0.0e+00 : f32
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c32 = constant 32 : index
    %c1 = constant 1 : index

    // Intialize the Input matrix with ones.
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        memref.store %f1, %0[%arg0, %arg1] : memref<16x16xf16>
      }
    }
    // Intialize the accumulator matrix with zeros.
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        memref.store %f0, %22[%arg0, %arg1] : memref<16x16xf32>
      }
    }

    %2 = memref.cast %0 : memref<16x16xf16> to memref<*xf16>
    %33 = memref.cast %22 : memref<16x16xf32> to memref<*xf32>
    %3 = memref.cast %1 : memref<16x16xf32> to memref<*xf32>
    gpu.host_register %2 : memref<*xf16>
    gpu.host_register %33 : memref<*xf32>

    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1) args(%0 : memref<16x16xf16>, %22 : memref<16x16xf32>)

    // Print the memref after computation.
    call @print_memref_f32(%33) : (memref<*xf32>) -> ()
    // CHECK: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16],
    // CHECK-NEXT: [16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16,   16]
    return
  }

  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<16x16xf16>, %arg22 : memref<16x16xf32>) kernel {
      %c0 = constant 0 : index

      %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {operand = "AOp", leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
      %1 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {operand = "BOp", leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
      %2 = gpu.subgroup_mma_load_matrix %arg22[%c0, %c0] {operand = "COp", leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

      %3 = gpu.subgroup_mma_compute %0, %1, %2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf32, "COp"> -> !gpu.mma_matrix<16x16xf32, "DOp">

      gpu.subgroup_mma_store_matrix %3, %arg22[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf32, "DOp">, memref<16x16xf32>

      gpu.return
    }
  }

  func private @print_memref_f32(memref<*xf32>)
}
