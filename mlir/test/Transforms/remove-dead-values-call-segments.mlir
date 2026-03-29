// RUN: mlir-opt --split-input-file --remove-dead-values --mlir-print-op-generic %s | FileCheck %s --check-prefix=GEN

// -----
// Private callee: both args become dead after internal DCE; RDV drops callee
// args and shrinks the *args* segment on the call-site to zero; sizes kept in
// sync.

module {
  func.func private @callee(%x: i32, %y: i32) {
    %u = arith.addi %x, %x : i32   // %y is dead
    return
  }

  func.func @caller(%a: i32, %b: i32) {
    // args segment initially has 2 operands.
    "test.call_with_segments"(%a, %b) { callee = @callee,
      operandSegmentSizes = array<i32: 0, 2, 0> } : (i32, i32) -> ()
    return
  }
}

// GEN: "test.call_with_segments"() <{callee = @callee, operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> ()
//       ^ args shrank from 2 -> 0
