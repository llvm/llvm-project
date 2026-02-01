// RUN: mlir-opt --int-range-optimizations %s | FileCheck %s

// CHECK-LABEL: func @affine_apply_constant
// CHECK: test.reflect_bounds {smax = 42 : index, smin = 42 : index, umax = 42 : index, umin = 42 : index}
func.func @affine_apply_constant() -> index {
  %0 = affine.apply affine_map<() -> (42)>()
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_add
// CHECK: test.reflect_bounds {smax = 15 : index, smin = 6 : index, umax = 15 : index, umin = 6 : index}
func.func @affine_apply_add() -> index {
  %d0 = test.with_bounds { umin = 2 : index, umax = 5 : index,
                           smin = 2 : index, smax = 5 : index } : index
  %d1 = test.with_bounds { umin = 4 : index, umax = 10 : index,
                           smin = 4 : index, smax = 10 : index } : index
  %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%d0, %d1)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mul
// CHECK: test.reflect_bounds {smax = 30 : index, smin = 12 : index, umax = 30 : index, umin = 12 : index}
func.func @affine_apply_mul() -> index {
  %d0 = test.with_bounds { umin = 2 : index, umax = 5 : index,
                           smin = 2 : index, smax = 5 : index } : index
  %s0 = test.with_bounds { umin = 6 : index, umax = 6 : index,
                           smin = 6 : index, smax = 6 : index } : index
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%d0)[%s0]
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_floordiv
// CHECK: test.reflect_bounds {smax = 2 : index, smin = 1 : index, umax = 2 : index, umin = 1 : index}
func.func @affine_apply_floordiv() -> index {
  %d0 = test.with_bounds { umin = 5 : index, umax = 10 : index,
                           smin = 5 : index, smax = 10 : index } : index
  %0 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_ceildiv
// CHECK: test.reflect_bounds {smax = 3 : index, smin = 2 : index, umax = 3 : index, umin = 2 : index}
func.func @affine_apply_ceildiv() -> index {
  %d0 = test.with_bounds { umin = 5 : index, umax = 10 : index,
                           smin = 5 : index, smax = 10 : index } : index
  %0 = affine.apply affine_map<(d0) -> (d0 ceildiv 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mod
// CHECK: test.reflect_bounds {smax = 3 : index, smin = 0 : index, umax = 3 : index, umin = 0 : index}
func.func @affine_apply_mod() -> index {
  %d0 = test.with_bounds { umin = 5 : index, umax = 27 : index,
                           smin = 5 : index, smax = 27 : index } : index
  %0 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_complex
// CHECK: test.reflect_bounds {smax = 13 : index, smin = 5 : index, umax = 13 : index, umin = 5 : index}
func.func @affine_apply_complex() -> index {
  %d0 = test.with_bounds { umin = 10 : index, umax = 20 : index,
                           smin = 10 : index, smax = 20 : index } : index
  %d1 = test.with_bounds { umin = 3 : index, umax = 7 : index,
                           smin = 3 : index, smax = 7 : index } : index
  // (d0 floordiv 2) + (d1 mod 4) = [5, 10] + [0, 3] = [5, 13]
  %0 = affine.apply affine_map<(d0, d1) -> (d0 floordiv 2 + d1 mod 4)>(%d0, %d1)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_with_symbols
// CHECK: test.reflect_bounds {smax = 24 : index, smin = 9 : index, umax = 24 : index, umin = 9 : index}
func.func @affine_apply_with_symbols() -> index {
  %d0 = test.with_bounds { umin = 2 : index, umax = 5 : index,
                           smin = 2 : index, smax = 5 : index } : index
  %s0 = test.with_bounds { umin = 3 : index, umax = 4 : index,
                           smin = 3 : index, smax = 4 : index } : index
  // d0 * s0 + s0 = s0 * (d0 + 1) = [3, 4] * [3, 6] = [9, 24]
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 * s0 + s0)>(%d0)[%s0]
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_sub
// CHECK: test.reflect_bounds {smax = 1 : index, smin = -8 : index
func.func @affine_apply_sub() -> index {
  %d0 = test.with_bounds { umin = 2 : index, umax = 5 : index,
                           smin = 2 : index, smax = 5 : index } : index
  %d1 = test.with_bounds { umin = 4 : index, umax = 10 : index,
                           smin = 4 : index, smax = 10 : index } : index
  // d0 - d1 = [2, 5] - [4, 10] = [2-10, 5-4] = [-8, 1]
  %0 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%d0, %d1)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mul_constant
// CHECK: test.reflect_bounds {smax = 20 : index, smin = 8 : index, umax = 20 : index, umin = 8 : index}
func.func @affine_apply_mul_constant() -> index {
  %d0 = test.with_bounds { umin = 2 : index, umax = 5 : index,
                           smin = 2 : index, smax = 5 : index } : index
  // d0 * 4 = [2, 5] * 4 = [8, 20]
  %0 = affine.apply affine_map<(d0) -> (d0 * 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mod_small_range
// CHECK: test.reflect_bounds {smax = 2 : index, smin = 1 : index, umax = 2 : index, umin = 1 : index}
func.func @affine_apply_mod_small_range() -> index {
  %d0 = test.with_bounds { umin = 5 : index, umax = 6 : index,
                           smin = 5 : index, smax = 6 : index } : index
  // Small range optimization: 5 mod 4 = 1, 6 mod 4 = 2, so [1, 2]
  %0 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mod_already_in_range
// CHECK: test.reflect_bounds {smax = 7 : index, smin = 5 : index, umax = 7 : index, umin = 5 : index}
func.func @affine_apply_mod_already_in_range() -> index {
  %d0 = test.with_bounds { umin = 5 : index, umax = 7 : index,
                           smin = 5 : index, smax = 7 : index } : index
  // Dividend [5, 7] already in [0, 10), result equals dividend: [5, 7]
  %0 = affine.apply affine_map<(d0) -> (d0 mod 10)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mod_variable_divisor
// CHECK: test.reflect_bounds {smax = 4 : index, smin = 0 : index, umax = 4 : index, umin = 0 : index}
func.func @affine_apply_mod_variable_divisor() -> index {
  %d0 = test.with_bounds { umin = 10 : index, umax = 20 : index,
                           smin = 10 : index, smax = 20 : index } : index
  %s0 = test.with_bounds { umin = 3 : index, umax = 5 : index,
                           smin = 3 : index, smax = 5 : index } : index
  // s0 can be 3, 4, or 5, so result is [0, max(s0)-1] = [0, 4]
  %0 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)>(%d0)[%s0]
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}

// CHECK-LABEL: func @affine_apply_mod_negative_dividend
// CHECK: test.reflect_bounds {smax = 3 : index, smin = 0 : index, umax = 3 : index, umin = 0 : index}
func.func @affine_apply_mod_negative_dividend() -> index {
  %d0 = test.with_bounds { umin = 0 : index, umax = 2 : index,
                           smin = -2 : index, smax = 2 : index } : index
  // Negative dividend: signed range [-2, 2] mod 4
  // Actual results: -2->2, -1->3, 0->0, 1->1, 2->2 (Euclidean mod)
  // Range is NOT contiguous, so we return conservative [0, 3]
  %0 = affine.apply affine_map<(d0) -> (d0 mod 4)>(%d0)
  %1 = test.reflect_bounds %0 : index
  func.return %1 : index
}
