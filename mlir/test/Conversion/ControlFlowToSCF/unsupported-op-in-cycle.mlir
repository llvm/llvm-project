// RUN: mlir-opt %s -lift-cf-to-scf -verify-diagnostics

// verify faliure for unsupported ops in cycles (issue #173566)

module {
  func.func @spirv_loop_crash_repro(%arg0: index) {
    %0 = builtin.unrealized_conversion_cast %arg0 : index to i32
    %cst8 = spirv.Constant 8 : i32
    
    spirv.mlir.loop {
      spirv.Branch ^bb1(%0 : i32)
    ^bb1(%2: i32):
      %3 = spirv.SLessThan %2, %cst8 : i32
      // expected-error @+1 {{Cannot convert unknown control flow op to structured control flow}}
      spirv.BranchConditional %3, ^bb2, ^bb3
    ^bb2:
      %4 = spirv.IAdd %2, %0 : i32
      spirv.Branch ^bb1(%4 : i32)
    ^bb3:
      spirv.mlir.merge
    }
    spirv.Return
  }
}