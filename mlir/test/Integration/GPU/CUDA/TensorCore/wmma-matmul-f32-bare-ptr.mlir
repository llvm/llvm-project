// Tests memref bare pointer lowering convention both host side and kernel-side;
// this works for only statically shaped memrefs.
// Similar to the wmma-matmul-f32 but but with the memref bare pointer lowering convention.
// This test also uses gpu.memcpy operations (instead of gpu.host_register).
// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm="host-bare-ptr-calling-convention=1 kernel-bare-ptr-calling-convention=1 cubin-chip=sm_70 cubin-format=%gpu_compilation_format" \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @main() {
  %h0 = memref.alloc() : memref<16x16xf16>
  %h_out = memref.alloc() : memref<16x16xf32>

  %f1 = arith.constant 1.0e+00 : f16
  %f0 = arith.constant 0.0e+00 : f32
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // Intialize the Input matrix with ones.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      memref.store %f1, %h0[%arg0, %arg1] : memref<16x16xf16>
    }
  }
  // Intialize the accumulator matrix with zeros.
  scf.for %arg0 = %c0 to %c16 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      memref.store %f0, %h_out[%arg0, %arg1] : memref<16x16xf32>
    }
  }

  %token = gpu.wait async
  %0, %t0 = gpu.alloc async [%token] () : memref<16x16xf16>
  %out, %t1 = gpu.alloc async [%token]() : memref<16x16xf32>
  // Copy from host to device.
  %x = gpu.memcpy async [%token] %0, %h0 : memref<16x16xf16>, memref<16x16xf16>
  %y = gpu.memcpy async [%token] %out, %h_out : memref<16x16xf32>, memref<16x16xf32>

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c32, %block_y = %c1, %block_z = %c1) {
    %A = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %B = gpu.subgroup_mma_load_matrix %0[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "BOp">
    %C = gpu.subgroup_mma_load_matrix %out[%c0, %c0] {leadDimension = 16 : index} : memref<16x16xf32> -> !gpu.mma_matrix<16x16xf32, "COp">

    %R = gpu.subgroup_mma_compute %A, %B, %C : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

    gpu.subgroup_mma_store_matrix %R, %out[%c0, %c0] {leadDimension = 16 : index}: !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>
    // Memref print utilities use unranked memrefs. We can't use bare pointer lowering
    // with them. We simply test for successful execution.
    gpu.printf "Success\n"
    // CHECK: Success
    gpu.terminator
  }
  %z = gpu.dealloc async [%token] %0 : memref<16x16xf16>
  %w = gpu.dealloc async [%token] %out : memref<16x16xf32>
  gpu.wait [%token]
  return
}
