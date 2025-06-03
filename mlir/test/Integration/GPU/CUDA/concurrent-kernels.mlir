// Tests multiple kernels running concurrently. Runs two kernels, which
// increment a global atomic counter and wait for the counter to reach 2.
//
// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | env CUDA_MODULE_LOADING=EAGER mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void

// CUDA_MODULE_LOADING=EAGER avoids an implicit context synchronization on first
// use of each kernel. It is technically not needed for this test, because
// there is only one kernel.

module attributes {gpu.container_module} {

gpu.module @kernels {
  gpu.func @kernel(%memref: memref<i32>) kernel {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %block = memref.atomic_rmw addi %c1, %memref[] : (i32, memref<i32>) -> i32
    scf.while: () -> () {
      %value = memref.atomic_rmw addi %c0, %memref[] : (i32, memref<i32>) -> i32
      %cond = arith.cmpi slt, %value, %c2 : i32
      scf.condition(%cond)
    } do {
      scf.yield
    }
    gpu.return
  }
}

func.func @main() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %memref = gpu.alloc host_shared () : memref<i32>
  memref.store %c0, %memref[] : memref<i32>
  %0 = gpu.wait async
  %1 = gpu.wait async
  %2 = gpu.launch_func async [%0] @kernels::@kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1)
      args(%memref: memref<i32>)
  %3 = gpu.launch_func async [%1] @kernels::@kernel
      blocks in (%c1, %c1, %c1)
      threads in (%c1, %c1, %c1)
      args(%memref: memref<i32>)
  gpu.wait [%2, %3]
  return
}

}
