// RUN: mlir-opt --lower-affine %s 2>&1 | FileCheck %s

// Test that affine.parallel with an unsupported reduction kind ("assign")
// does not crash but emits a proper error message. Previously,
// getIdentityValue would be called with a null TypedAttr and crash inside
// arith::ConstantOp::build with "Failed to infer result type(s)".

// CHECK: Reduction operation type not supported
// CHECK-NOT: Failed to infer result type

func.func @affine_parallel_assign_reduction_no_crash(%n: index) -> i32 {
  %0 = affine.parallel (%i) = (0) to (%n) reduce ("assign") -> i32 {
    %c0 = arith.constant 0 : i32
    affine.yield %c0 : i32
  }
  return %0 : i32
}
