// RUN: mlir-opt %s -inline | FileCheck %s

// Verifies that regions with operations from the ml_program dialect can
// be inlined.

ml_program.global private @global(dense<4> : tensor<4xi32>) : tensor<4xi32>

// CHECK: @inline_into
func.func @inline_into() -> tensor<4xi32> {
  // CHECK-NOT: @inline_from
  // CHECK: ml_program.global_load_const
  %0 = call @inline_from() : () -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

func.func @inline_from() -> tensor<4xi32> {
  %0 = ml_program.global_load_const @global : tensor<4xi32>
  return %0 : tensor<4xi32>
}
