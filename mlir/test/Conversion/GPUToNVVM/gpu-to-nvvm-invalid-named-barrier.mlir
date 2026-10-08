// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file -verify-diagnostics

gpu.module @test_module {
  func.func @too_many_named_barriers() {
    %c1 = arith.constant 1 : i32
    %nb0 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb1 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb2 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb3 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb4 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb5 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb6 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb7 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb8 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb9 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb10 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb11 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb12 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb13 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    %nb14 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    // expected-error@+2 {{NVVM supports at most 15 named barriers per CTA}}
    // expected-error@+1 {{failed to legalize operation 'gpu.initialize_named_barrier'}}
    %nb15 = gpu.initialize_named_barrier %c1 : i32 -> !gpu.named_barrier
    func.return
  }
}
