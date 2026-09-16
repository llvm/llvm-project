// RUN: mlir-opt --lift-cf-to-scf -split-input-file --verify-diagnostics %s

// Regression test: spirv.BranchConditional inside a spirv.mlir.loop region
// must produce a clean diagnostic rather than an assertion crash.

func.func @spirv_loop_no_crash(%cond: i1) {
  spirv.mlir.loop {
    spirv.Branch ^bb1
  ^bb1:
    // expected-error@below {{cannot convert unknown control flow op to structured control flow}}
    spirv.BranchConditional %cond, ^bb2, ^bb3
  ^bb2:
    spirv.Branch ^bb1
  ^bb3:
    spirv.mlir.merge
  }
  return
}
