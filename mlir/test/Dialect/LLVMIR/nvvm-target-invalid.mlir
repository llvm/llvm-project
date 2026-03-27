// RUN: mlir-opt -verify-diagnostics -split-input-file %s

module {
  gpu.module @tcgen05_unsupported_sm90 [#nvvm.target<chip = "sm_90">] {
    func.func @tcgen05_alloc(%arg0: !llvm.ptr<7>, %arg1: i32) {
      // expected-error @+1 {{'nvvm.tcgen05.alloc' op is not supported on sm_90}}
      nvvm.tcgen05.alloc %arg0, %arg1 : !llvm.ptr<7>, i32
      return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // expected-error @+1 {{The optimization level must be a number between 0 and 3}}
  gpu.module @nvvm_target_invalid_opt_level [#nvvm.target<chip = "sm_90", O = 4>] {
  }
}
