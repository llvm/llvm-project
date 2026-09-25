// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1100' -split-input-file -verify-diagnostics

gpu.module @test_module {
  func.func @initialize_named_barrier_pre_gfx12(%count : i32) {
    // expected-error@+2 {{named barriers require gfx12+}}
    // expected-error@+1 {{failed to legalize}}
    %nb = gpu.initialize_named_barrier %count : i32 -> !gpu.named_barrier
    func.return
  }
}
