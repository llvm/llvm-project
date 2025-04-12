// Tests that we can run multiple kernels concurrently. Runs two kernels, which
// increment a global atomic counter, then wait for the counter to reach 2.
//
// RUN: mlir-opt %s \
// RUN: | mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void

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
        %memref = gpu.alloc host_shared () : memref<i32>
        %c0 = arith.constant 0 : i32
        memref.store %c0, %memref[] : memref<i32>

        %0 = gpu.wait async
        %1 = gpu.wait async
        %c1 = arith.constant 1 : index
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
