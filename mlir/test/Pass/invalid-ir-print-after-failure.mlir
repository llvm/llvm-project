// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=true})' -mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=CHECK-GENERIC
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=false})' -mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=CHECK-GENERIC
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=false signal-pass-failure=true})' -mlir-print-ir-after-failure 2>&1  | FileCheck %s --check-prefix=CHECK-CUSTOM
// RUN: mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=false signal-pass-failure=false})' -mlir-print-ir-after-failure 2>&1  | FileCheck %s --check-prefix=CHECK-CUSTOM

// Check that `-mlir-print-assume-verified` will print custom even when the IR is invalid.
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=true})' -mlir-print-ir-after-failure 2>&1 -mlir-print-assume-verified | FileCheck %s --check-prefix=CHECK-CUSTOM
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=false})' -mlir-print-ir-after-failure 2>&1 -mlir-print-assume-verified | FileCheck %s --check-prefix=CHECK-CUSTOM

// Test whether we print generically or not on pass failure, depending on whether there is invalid IR or not.

// CHECK-CUSTOM: func @TestCreateInvalidCallInPass
// CHECK-GENERIC: "func.func"
func @TestCreateInvalidCallInPass() {
  return
}
