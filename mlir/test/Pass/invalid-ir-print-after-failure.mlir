// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=true})' -mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=true signal-pass-failure=false})' -mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=false signal-pass-failure=true})' -mlir-print-ir-after-failure 2>&1  | FileCheck %s --check-prefix=CHECK-VALID
// RUN: mlir-opt %s -pass-pipeline='func.func(test-pass-create-invalid-ir{emit-invalid-ir=false signal-pass-failure=false})' -mlir-print-ir-after-failure 2>&1  | FileCheck %s --check-prefix=CHECK-VALID

// Test whether we print generically or not on pass failure, depending on whether there is invalid IR or not.

// CHECK-VALID: func @TestCreateInvalidCallInPass
// CHECK-INVALID: "func.func"
func @TestCreateInvalidCallInPass() {
  return
}
