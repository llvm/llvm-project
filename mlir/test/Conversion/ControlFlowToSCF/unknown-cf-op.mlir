// RUN: mlir-opt --lift-cf-to-scf %s -verify-diagnostics -split-input-file

// Verify that --lift-cf-to-scf does not crash when it encounters an
// unknown control-flow op (not cf.cond_br or cf.switch) in a multi-block
// region. Instead it should emit a clean error.
// See: https://github.com/llvm/llvm-project/issues/120883

// The spirv.BranchConditional op implements BranchOpInterface but is not
// handled by ControlFlowToSCFTransformation::createStructuredBranchRegionOp.
// Previously, blocks were moved into temporary Region objects before the
// failure was detected, and the Region destructor would assert because the
// moved blocks still had live predecessor references.

func.func private @unknown_cf_in_loop(%arg0: f32) -> (i32, i32) {
  %cst0_i32 = spirv.Constant 0 : i32
  %cst-1_i32 = spirv.Constant -1 : i32
  %0 = spirv.Variable : !spirv.ptr<i32, Function>
  %1 = spirv.Variable : !spirv.ptr<i32, Function>
  spirv.mlir.loop {
    spirv.Branch ^bb1(%cst0_i32, %cst-1_i32 : i32, i32)
  ^bb1(%3: i32, %4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = spirv.SLessThan %3, %cst-1_i32 : i32
    // expected-error@below {{'spirv.BranchConditional' op cannot convert unknown control flow op to structured control flow}}
    spirv.BranchConditional %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = spirv.IAdd %3, %4 : i32
    spirv.Store "Function" %0, %6 : i32
    %7 = spirv.IAdd %3, %cst0_i32 : i32
    spirv.Branch ^bb1(%7, %6 : i32, i32)
  ^bb3:  // pred: ^bb1
    spirv.mlir.merge
  }
  %2 = spirv.Load "Function" %1 : i32
  %3 = spirv.Load "Function" %0 : i32
  return %3, %2 : i32, i32
}
