// RUN: mlir-opt %s --arith-expand="include-min-max-f=true" \
// RUN:   --convert-vector-to-scf --convert-scf-to-cf \
// RUN:   --convert-vector-to-llvm --convert-func-to-llvm \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-arith-to-llvm --convert-cf-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:     --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Keep operands as function arguments so constant folding cannot bypass the
// min/max expansion. Print integer bit patterns to distinguish signed zeros.
func.func @check_min_max(%lhs: f32, %rhs: f32) {
  %max = arith.maximumf %lhs, %rhs : f32
  %min = arith.minimumf %lhs, %rhs : f32
  %max_bits = arith.bitcast %max : f32 to i32
  %min_bits = arith.bitcast %min : f32 to i32
  vector.print %max_bits : i32
  vector.print %min_bits : i32
  return
}

// Check NaN classification without requiring a particular NaN bit pattern.
func.func @check_nan(%lhs: f32, %rhs: f32) {
  %max = arith.maximumf %lhs, %rhs : f32
  %min = arith.minimumf %lhs, %rhs : f32
  %max_nan = arith.cmpf uno, %max, %max : f32
  %min_nan = arith.cmpf uno, %min, %min : f32
  %max_result = arith.extui %max_nan : i1 to i32
  %min_result = arith.extui %min_nan : i1 to i32
  vector.print %max_result : i32
  vector.print %min_result : i32
  return
}

func.func @check_vector(%lhs: vector<4xf32>, %rhs: vector<4xf32>) {
  %max = arith.maximumf %lhs, %rhs : vector<4xf32>
  %min = arith.minimumf %lhs, %rhs : vector<4xf32>
  %max_bits = arith.bitcast %max : vector<4xf32> to vector<4xi32>
  %min_bits = arith.bitcast %min : vector<4xf32> to vector<4xi32>
  vector.print %max_bits : vector<4xi32>
  vector.print %min_bits : vector<4xi32>
  return
}

func.func @entry() {
  %pos_zero = arith.constant 0.0 : f32
  %neg_zero = arith.constant -0.0 : f32
  %two = arith.constant 2.0 : f32
  %neg_two = arith.constant -2.0 : f32
  %pos_subnormal = arith.constant 0x00000001 : f32
  %neg_subnormal = arith.constant 0x80000001 : f32
  %pos_inf = arith.constant 0x7F800000 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %nan = arith.constant 0x7FC00000 : f32
  %neg_nan = arith.constant 0xFFC00000 : f32
  %snan = arith.constant 0x7F800001 : f32

  // Opposite zero signs, in both operand orders.
  // CHECK: 0
  // CHECK-NEXT: -2147483648
  func.call @check_min_max(%pos_zero, %neg_zero) : (f32, f32) -> ()
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483648
  func.call @check_min_max(%neg_zero, %pos_zero) : (f32, f32) -> ()

  // Same zero signs must be preserved.
  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  func.call @check_min_max(%pos_zero, %pos_zero) : (f32, f32) -> ()
  // CHECK-NEXT: -2147483648
  // CHECK-NEXT: -2147483648
  func.call @check_min_max(%neg_zero, %neg_zero) : (f32, f32) -> ()

  // Equal nonzero operands retain their value when selected by sign.
  // CHECK-NEXT: 1073741824
  // CHECK-NEXT: 1073741824
  func.call @check_min_max(%two, %two) : (f32, f32) -> ()
  // CHECK-NEXT: -1073741824
  // CHECK-NEXT: -1073741824
  func.call @check_min_max(%neg_two, %neg_two) : (f32, f32) -> ()

  // Unequal operands still use the original comparison, in either order.
  // CHECK-NEXT: 1073741824
  // CHECK-NEXT: -1073741824
  func.call @check_min_max(%two, %neg_two) : (f32, f32) -> ()
  // CHECK-NEXT: 1073741824
  // CHECK-NEXT: -1073741824
  func.call @check_min_max(%neg_two, %two) : (f32, f32) -> ()

  // Subnormals adjacent to zero still compare as unequal.
  // CHECK-NEXT: 1
  // CHECK-NEXT: -2147483648
  func.call @check_min_max(%pos_subnormal, %neg_zero) : (f32, f32) -> ()
  // CHECK-NEXT: 0
  // CHECK-NEXT: -2147483647
  func.call @check_min_max(%pos_zero, %neg_subnormal) : (f32, f32) -> ()

  // Infinities compare against finite values and retain their signs on ties.
  // CHECK-NEXT: 2139095040
  // CHECK-NEXT: 1073741824
  func.call @check_min_max(%pos_inf, %two) : (f32, f32) -> ()
  // CHECK-NEXT: -1073741824
  // CHECK-NEXT: -8388608
  func.call @check_min_max(%neg_two, %neg_inf) : (f32, f32) -> ()
  // CHECK-NEXT: 2139095040
  // CHECK-NEXT: -8388608
  func.call @check_min_max(%neg_inf, %pos_inf) : (f32, f32) -> ()
  // CHECK-NEXT: 2139095040
  // CHECK-NEXT: 2139095040
  func.call @check_min_max(%pos_inf, %pos_inf) : (f32, f32) -> ()
  // CHECK-NEXT: -8388608
  // CHECK-NEXT: -8388608
  func.call @check_min_max(%neg_inf, %neg_inf) : (f32, f32) -> ()

  // Either NaN operand must propagate, including when both operands are NaN.
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%nan, %two) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%two, %nan) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%nan, %neg_nan) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%neg_nan, %neg_zero) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%pos_zero, %neg_nan) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%snan, %pos_inf) : (f32, f32) -> ()
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  func.call @check_nan(%neg_inf, %snan) : (f32, f32) -> ()

  // Check all four signed-zero combinations elementwise.
  %lhs = arith.constant dense<[0.0, -0.0, 0.0, -0.0]> : vector<4xf32>
  %rhs = arith.constant dense<[-0.0, 0.0, 0.0, -0.0]> : vector<4xf32>
  // CHECK-NEXT: ( 0, 0, 0, -2147483648 )
  // CHECK-NEXT: ( -2147483648, -2147483648, 0, -2147483648 )
  func.call @check_vector(%lhs, %rhs) : (vector<4xf32>, vector<4xf32>) -> ()

  // A vector can take the ordered comparison and signed-zero paths in
  // different lanes.
  %mixed_lhs = arith.constant dense<[2.0, -2.0, 0x7F800000, -0.0]> : vector<4xf32>
  %mixed_rhs = arith.constant dense<[-2.0, 2.0, 0xFF800000, 0.0]> : vector<4xf32>
  // CHECK-NEXT: ( 1073741824, 1073741824, 2139095040, 0 )
  // CHECK-NEXT: ( -1073741824, -1073741824, -8388608, -2147483648 )
  func.call @check_vector(%mixed_lhs, %mixed_rhs) : (vector<4xf32>, vector<4xf32>) -> ()
  return
}
