// RUN: mlir-opt %s --composite-fixed-point-pass="name=Test pipeline='any(test-increment-attr)' max-iterations=3 on-convergence-failure=warn" 2>&1 | FileCheck %s --check-prefix=CHECK-WARN
// RUN: not mlir-opt %s --composite-fixed-point-pass="name=Test pipeline='any(test-increment-attr)' max-iterations=3 on-convergence-failure=error" 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: mlir-opt %s --composite-fixed-point-pass="name=Test pipeline='any(test-increment-attr)' max-iterations=3 on-convergence-failure=silent" 2>&1 | FileCheck %s --check-prefix=CHECK-SILENT

// The "test-increment-attr" pass mutates the op on every run, so the composite
// pass never reaches a fixed point and always exhausts max-iterations,
// regardless of the input IR.

// CHECK-WARN: warning: Composite pass "Test"+ didn't converge in 3 iterations
// CHECK-WARN: test.counter = 4

// CHECK-ERROR: error: Composite pass "Test"+ didn't converge in 3 iterations

// CHECK-SILENT-NOT: didn't converge
// CHECK-SILENT: test.counter = 4
func.func @test() {
  return
}
