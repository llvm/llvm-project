// RUN: mlir-opt %s --test-bytecode-roundtrip=test-dialect-version=2.0 | FileCheck %s

// CHECK-LABEL: func.func @main
func.func @main() {
  return
}
