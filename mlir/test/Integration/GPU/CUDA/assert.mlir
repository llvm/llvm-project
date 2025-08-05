// RUN: mlir-opt %s -gpu-lower-to-nvvm-pipeline="cubin-format=%gpu_compilation_format" \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_cuda_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --entry-point-result=void 2>&1 \
// RUN: | FileCheck %s

// CHECK-DAG: thread 0: print after passing assertion
// CHECK-DAG: thread 1: print after passing assertion
// CHECK-DAG: callee_file.cc:7: callee_func_name: block: [0,0,0], thread: [0,0,0] Assertion `failing assertion` failed.
// CHECK-DAG: callee_file.cc:7: callee_func_name: block: [0,0,0], thread: [1,0,0] Assertion `failing assertion` failed.
// CHECK-NOT: print after failing assertion

module attributes {gpu.container_module} {
gpu.module @kernels {
gpu.func @test_assert(%c0: i1, %c1: i1) kernel {
  %0 = gpu.thread_id x
  cf.assert %c1, "passing assertion"
  gpu.printf "thread %lld: print after passing assertion\n", %0 : index
  // Test callsite(callsite(name)) location.
  cf.assert %c0, "failing assertion" loc(callsite(callsite("callee_func_name"("callee_file.cc":7:9) at "caller_file.cc":10:8) at "caller2_file.cc":11:12))
  gpu.printf "thread %lld: print after failing assertion\n", %0 : index
  gpu.return
}
}

func.func @main() {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0_i1 = arith.constant 0 : i1
  %c1_i1 = arith.constant 1 : i1
  gpu.launch_func @kernels::@test_assert
      blocks in (%c1, %c1, %c1)
      threads in (%c2, %c1, %c1)
      args(%c0_i1 : i1, %c1_i1 : i1)
  return
}
}
