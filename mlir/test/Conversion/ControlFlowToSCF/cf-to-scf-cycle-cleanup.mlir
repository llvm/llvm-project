// RUN: not mlir-opt %s -lift-cf-to-scf 2>&1 | FileCheck %s

// This test verifies that the pass does not crash when encountering unsupported
// control flow operations within a cycle (e.g., SPIR-V loops). (issue #173566)
// It ensures that temporary regions are cleaned up correctly upon failure.

// CHECK: Cannot convert unknown control flow op to structured control flow
module {
  func.func @spirv_loop_crash_repro(%arg0: index) {
    %0 = builtin.unrealized_conversion_cast %arg0 : index to i32
    %cst8 = spirv.Constant 8 : i32
    
    // spirv.mlir.loop creates a CFG cycle. 
    spirv.mlir.loop {
      spirv.Branch ^bb1(%0 : i32)
    ^bb1(%2: i32):
      %3 = spirv.SLessThan %2, %cst8 : i32
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
