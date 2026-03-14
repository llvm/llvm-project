// RUN: not mlir-translate %s --allow-unregistered-dialect --mlir-to-llvmir -verify-diagnostics 2>&1 | FileCheck --check-prefix=UNREGOK %s
// RUN: not mlir-translate %s --mlir-to-llvmir -verify-diagnostics 2>&1 | FileCheck --check-prefix=REGONLY %s


// If the parser allows unregistered operations, then the translation fails,
// otherwise the parse fails.

// UNREGOK: cannot be converted to LLVM IR
// REGONLY: operation being parsed with an unregistered dialect

func.func @trivial() {
  "simple.terminator"() : () -> ()
}
