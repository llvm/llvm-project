// Check that the ceildivsi lowering is correct.
// We do not check any poison or UB values, as it is not possible to catch them.

// RUN: mlir-opt %s --convert-to-llvm

// Put rhs into separate function so that it won't be constant-folded.
func.func @foo() -> f4E2M1FN {
  %cst = arith.constant 5.0 : f4E2M1FN
  return %cst : f4E2M1FN
}

func.func @entry() {
  %a = arith.constant 5.0 : f4E2M1FN
  %b = func.call @foo() : () -> (f4E2M1FN)
  %c = arith.addf %a, %b : f4E2M1FN
  vector.print %c : f4E2M1FN
  return
}

