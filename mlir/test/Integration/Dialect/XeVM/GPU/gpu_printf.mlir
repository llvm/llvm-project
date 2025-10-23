// RUN: mlir-opt %s \
// RUN: | mlir-opt -pass-pipeline='builtin.module(cse,func.func(gpu-async-region),xevm-attach-target,gpu.module(convert-gpu-to-llvm-spv{use-64bit-index=true},convert-xevm-to-llvm,cse))' \
// RUN: | mlir-opt -convert-scf-to-cf -convert-cf-to-llvm -convert-vector-to-llvm -convert-arith-to-llvm \
// RUN: | mlir-opt -gpu-to-llvm -reconcile-unrealized-casts -cse -gpu-module-to-binary \
// RUN: | mlir-runner \
// RUN:   --shared-libs=%mlir_sycl_runtime \
// RUN:   --shared-libs=%mlir_runner_utils \
// RUN:   --shared-libs=%mlir_c_runner_utils \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module @test attributes {gpu.container_module} {
  gpu.module @test_module {
    gpu.func @test_printf(%arg0: i32, %arg1: f32) kernel {
      gpu.printf "Hello: %d\n", %arg0 : i32
      gpu.printf "Hello: %f\n", %arg1 : f32
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c11 = arith.constant 11 : i32
    %c4 = arith.constant 4.0 : f32
    // CHECK: Hello: 11
    // CHECK: Hello: 4.000000
    gpu.launch_func @test_module::@test_printf blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%c11 : i32, %c4 : f32)
    return
  }
}
