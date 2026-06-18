// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1250' -split-input-file -verify-diagnostics

gpu.module @test_module {
  func.func @non_constant_member_count(%count : i32) {
    // expected-error@+2 {{named barrier member count must be a constant for ROCDL lowering}}
    // expected-error@+1 {{failed to legalize}}
    %nb = gpu.initialize_named_barrier %count : i32 -> !gpu.named_barrier
    func.return
  }
}
