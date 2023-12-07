// RUN: mlir-opt %s \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-rocdl{index-bitwidth=32 runtime=HIP},gpu-to-hsaco{chip=%chip}))' \
// RUN: | mlir-opt -gpu-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   --shared-libs=%mlir_rocm_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: Hello from 0
// CHECK: Hello from 1
module attributes {gpu.container_module} {
    gpu.module @kernels {
        gpu.func @hello() kernel {
            %0 = gpu.thread_id x
            gpu.printf "Hello from %d\n" %0 : index
            gpu.return
        }
    }

    func.func @main() {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        gpu.launch_func @kernels::@hello
            blocks in (%c1, %c1, %c1)
            threads in (%c2, %c1, %c1)
        return
    }
}
