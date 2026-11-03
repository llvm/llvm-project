// RUN: mlir-opt %s -int-range-optimizations | FileCheck %s

module @width_16 attributes {dlti.dl_spec = #dlti.dl_spec<index = 16 : i32>} {
  // CHECK-LABEL: func.func @arith_unsigned_cast_16
  // CHECK: %[[C:.*]] = arith.constant true
  // CHECK: return %[[C]]
  func.func @arith_unsigned_cast_16(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = arith.index_castui %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @index_unsigned_cast_16
  // CHECK: %[[C:.*]] = arith.constant true
  // CHECK: return %[[C]]
  func.func @index_unsigned_cast_16(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = index.castu %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // TODO: Constant propagation must use the runtime index width when folding
  // the comparison. The large constant truncates to zero, so the result must
  // be 7. The folder currently compares 64-bit attributes and selects 9.
  // CHECK-LABEL: func.func @branch_16
  // CHECK: %[[R:.*]] = arith.constant 9 : i32
  // CHECK-NEXT: return %[[R]]
  func.func @branch_16() -> i32 {
    %large = arith.constant 4294967296 : index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi eq, %large, %zero : index
    %r = scf.if %cmp -> i32 {
      %a = arith.constant 7 : i32
      scf.yield %a : i32
    } else {
      %b = arith.constant 9 : i32
      scf.yield %b : i32
    }
    return %r : i32
  }

  // CHECK-LABEL: func.func @index_cast_from_index_16
  // CHECK: %[[C:.*]] = arith.constant -1 : i64
  // CHECK: return %[[C]]
  func.func @index_cast_from_index_16() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_cast %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @index_castui_from_index_16
  // CHECK: %[[C:.*]] = arith.constant 65535 : i64
  // CHECK: return %[[C]]
  func.func @index_castui_from_index_16() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_castui %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @zero_divisor_16
  // CHECK: arith.divui
  func.func @zero_divisor_16(%v: i8) -> index {
    %x = arith.index_castui %v : i8 to index
    %d = arith.constant 4294967296 : index
    %r = arith.divui %x, %d : index
    return %r : index
  }

  // CHECK-LABEL: func.func @sizeof_16
  // CHECK: %[[S:.*]] = index.constant 16
  // CHECK: return %[[S]]
  func.func @sizeof_16() -> index {
    %s = index.sizeof
    return %s : index
  }

}

module @width_32 attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  // CHECK-LABEL: func.func @arith_unsigned_cast_32
  // CHECK: %[[C:.*]] = arith.constant true
  // CHECK: return %[[C]]
  func.func @arith_unsigned_cast_32(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = arith.index_castui %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @index_unsigned_cast_32
  // CHECK: %[[C:.*]] = arith.constant true
  // CHECK: return %[[C]]
  func.func @index_unsigned_cast_32(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = index.castu %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // TODO: Constant propagation must use the runtime index width when folding
  // the comparison. The large constant truncates to zero, so the result must
  // be 7. The folder currently compares 64-bit attributes and selects 9.
  // CHECK-LABEL: func.func @branch_32
  // CHECK: %[[R:.*]] = arith.constant 9 : i32
  // CHECK-NEXT: return %[[R]]
  func.func @branch_32() -> i32 {
    %large = arith.constant 4294967296 : index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi eq, %large, %zero : index
    %r = scf.if %cmp -> i32 {
      %a = arith.constant 7 : i32
      scf.yield %a : i32
    } else {
      %b = arith.constant 9 : i32
      scf.yield %b : i32
    }
    return %r : i32
  }

  // CHECK-LABEL: func.func @index_cast_from_index_32
  // CHECK: %[[C:.*]] = arith.constant -1 : i64
  // CHECK: return %[[C]]
  func.func @index_cast_from_index_32() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_cast %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @index_castui_from_index_32
  // CHECK: %[[C:.*]] = arith.constant 4294967295 : i64
  // CHECK: return %[[C]]
  func.func @index_castui_from_index_32() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_castui %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @zero_divisor_32
  // CHECK: arith.divui
  func.func @zero_divisor_32(%v: i8) -> index {
    %x = arith.index_castui %v : i8 to index
    %d = arith.constant 4294967296 : index
    %r = arith.divui %x, %d : index
    return %r : index
  }

  // CHECK-LABEL: func.func @affine_divisor_32
  // CHECK: %[[V:.*]] = arith.constant 5 : index
  // CHECK: return %[[V]]
  func.func @affine_divisor_32() -> index {
    %v = arith.constant 5 : index
    %d = arith.constant -4294967295 : index
    %r = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)>(%v)[%d]
    return %r : index
  }

  // CHECK-LABEL: func.func @sizeof_32
  // CHECK: %[[S:.*]] = index.constant 32
  // CHECK: return %[[S]]
  func.func @sizeof_32() -> index {
    %s = index.sizeof
    return %s : index
  }

}

module @width_64 attributes {dlti.dl_spec = #dlti.dl_spec<index = 64 : i32>} {
  // CHECK-LABEL: func.func @arith_unsigned_cast_64
  // CHECK: %[[C:.*]] = arith.constant false
  // CHECK: return %[[C]]
  func.func @arith_unsigned_cast_64(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = arith.index_castui %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @index_unsigned_cast_64
  // CHECK: %[[C:.*]] = arith.constant false
  // CHECK: return %[[C]]
  func.func @index_unsigned_cast_64(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = index.castu %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @branch_64
  // CHECK: %[[R:.*]] = arith.constant 9 : i32
  // CHECK-NEXT: return %[[R]]
  func.func @branch_64() -> i32 {
    %large = arith.constant 4294967296 : index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi eq, %large, %zero : index
    %r = scf.if %cmp -> i32 {
      %a = arith.constant 7 : i32
      scf.yield %a : i32
    } else {
      %b = arith.constant 9 : i32
      scf.yield %b : i32
    }
    return %r : i32
  }

  // CHECK-LABEL: func.func @index_cast_from_index_64
  // CHECK: %[[C:.*]] = arith.constant 4294967295 : i64
  // CHECK: return %[[C]]
  func.func @index_cast_from_index_64() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_cast %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @index_castui_from_index_64
  // CHECK: %[[C:.*]] = arith.constant 4294967295 : i64
  // CHECK: return %[[C]]
  func.func @index_castui_from_index_64() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_castui %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @zero_divisor_64
  // CHECK: %[[Z:.*]] = arith.constant 0 : index
  // CHECK: return %[[Z]]
  func.func @zero_divisor_64(%v: i8) -> index {
    %x = arith.index_castui %v : i8 to index
    %d = arith.constant 4294967296 : index
    %r = arith.divui %x, %d : index
    return %r : index
  }

  // CHECK-LABEL: func.func @sizeof_64
  // CHECK: %[[S:.*]] = index.constant 64
  // CHECK: return %[[S]]
  func.func @sizeof_64() -> index {
    %s = index.sizeof
    return %s : index
  }

}

module @width_unknown {
  // CHECK-LABEL: func.func @arith_unsigned_cast_unknown
  // CHECK: arith.cmpi slt
  func.func @arith_unsigned_cast_unknown(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = arith.index_castui %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // CHECK-LABEL: func.func @index_unsigned_cast_unknown
  // CHECK: arith.cmpi slt
  func.func @index_unsigned_cast_unknown(%arg: i32) -> i1 {
    %lo = arith.constant -4 : i32
    %hi = arith.constant -1 : i32
    %a = arith.maxsi %arg, %lo : i32
    %b = arith.minsi %a, %hi : i32
    %idx = index.castu %b : i32 to index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi slt, %idx, %zero : index
    return %cmp : i1
  }

  // TODO: Constant propagation must retain both branches when the runtime
  // index width is unknown. The folder currently assumes 64 bits and selects 9.
  // CHECK-LABEL: func.func @branch_unknown
  // CHECK: %[[R:.*]] = arith.constant 9 : i32
  // CHECK-NEXT: return %[[R]]
  func.func @branch_unknown() -> i32 {
    %large = arith.constant 4294967296 : index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi eq, %large, %zero : index
    %r = scf.if %cmp -> i32 {
      %a = arith.constant 7 : i32
      scf.yield %a : i32
    } else {
      %b = arith.constant 9 : i32
      scf.yield %b : i32
    }
    return %r : i32
  }

  // CHECK-LABEL: func.func @index_cast_from_index_unknown
  // CHECK: arith.index_cast
  func.func @index_cast_from_index_unknown() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_cast %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @index_castui_from_index_unknown
  // CHECK: arith.index_castui
  func.func @index_castui_from_index_unknown() -> i64 {
    %v = arith.constant 4294967295 : index
    %r = arith.index_castui %v : index to i64
    return %r : i64
  }

  // CHECK-LABEL: func.func @zero_divisor_unknown
  // CHECK: arith.divui
  func.func @zero_divisor_unknown(%v: i8) -> index {
    %x = arith.index_castui %v : i8 to index
    %d = arith.constant 4294967296 : index
    %r = arith.divui %x, %d : index
    return %r : index
  }

  // CHECK-LABEL: func.func @sizeof_unknown
  // CHECK: index.sizeof
  func.func @sizeof_unknown() -> index {
    %s = index.sizeof
    return %s : index
  }

}

module @inherited attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  module @inner {
    // CHECK-LABEL: func.func @inherited_width
    // CHECK: %[[S:.*]] = index.constant 32
    // CHECK: return %[[S]]
    func.func @inherited_width() -> index {
      %s = index.sizeof
      return %s : index
    }
  }
}
module @other_layout attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.endianness" = "little">} {
  // CHECK-LABEL: func.func @unspecified_index_width
  // CHECK: index.sizeof
  func.func @unspecified_index_width() -> index {
    %s = index.sizeof
    return %s : index
  }
}

module @index128 attributes {dlti.dl_spec = #dlti.dl_spec<index = 128 : i32>} {
  // CHECK-LABEL: func.func @wide_unsigned_cast
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: return %[[TRUE]]
  func.func @wide_unsigned_cast(%arg: i64) -> i1 {
    %lo = arith.constant -4 : i64
    %hi = arith.constant -1 : i64
    %a = arith.maxsi %arg, %lo : i64
    %b = arith.minsi %a, %hi : i64
    %idx = arith.index_castui %b : i64 to index
    %zero = arith.constant 0 : index
    %r = arith.cmpi sgt, %idx, %zero : index
    return %r : i1
  }

  // CHECK-LABEL: func.func @wide_dimension
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: return %[[TRUE]]
  func.func @wide_dimension(%t: tensor<?x?xf32>, %d: index) -> i1 {
    %size = tensor.dim %t, %d : tensor<?x?xf32>
    %zero = arith.constant 0 : index
    %r = arith.cmpi sge, %size, %zero : index
    return %r : i1
  }
}

module @argument_bounds attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i32>} {
  // CHECK-LABEL: func.func @argument_width
  // CHECK: test.reflect_bounds <umin = 0 : index, umax = 4294967295 : index, smin = -2147483648 : index, smax = 2147483647 : index>
  func.func @argument_width(%arg: index) -> index {
    %r = test.reflect_bounds %arg : index
    return %r : index
  }
}

module @wide_bounds attributes {dlti.dl_spec = #dlti.dl_spec<index = 128 : i32>} {
  // CHECK-LABEL: func.func @wide_test_bounds
  // CHECK: test.reflect_bounds <umin = 0 : ui128, umax = 3 : ui128, smin = 0 : si128, smax = 3 : si128>
  func.func @wide_test_bounds() -> index {
    %v = test.with_bounds <umin = 0 : index, umax = 3 : index, smin = 0 : index, smax = 3 : index> : index
    %r = test.reflect_bounds %v : index
    return %r : index
  }
}

module @unknown_input {
  // CHECK-LABEL: func.func @unknown_input_unknown_width
  // CHECK: %[[CMP:.*]] = arith.cmpi
  // CHECK: %[[R:.*]] = scf.if
  // CHECK: scf.yield %[[CMP]]
  // CHECK: return %[[R]]
  func.func @unknown_input_unknown_width(%cond: i1) -> i1 {
    %x = "test.without_bounds"() : () -> index
    %zero = arith.constant 0 : index
    %cmp = arith.cmpi eq, %x, %zero : index
    %r = scf.if %cond -> i1 {
      scf.yield %cmp : i1
    } else {
      %false = arith.constant false
      scf.yield %false : i1
    }
    return %r : i1
  }
}

module @width_independent {
  func.func private @use(i1)

  // CHECK-LABEL: func.func @fixed_iv_with_index_carry
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: scf.for
  // CHECK: func.call @use(%[[TRUE]])
  func.func @fixed_iv_with_index_carry(%x: index) -> index {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %r = scf.for %i = %c0 to %c10 step %c1 iter_args(%carry = %x) -> index : i32 {
      %cmp = arith.cmpi sge, %i, %c0 : i32
      func.call @use(%cmp) : (i1) -> ()
      scf.yield %carry : index
    }
    return %r : index
  }

  // CHECK-LABEL: func.func @zero_cast_unknown_width
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK-NEXT: return %[[ZERO]]
  func.func @zero_cast_unknown_width() -> i32 {
    %zero = arith.constant 0 : index
    %r = arith.index_cast %zero : index to i32
    return %r : i32
  }

  // CHECK-LABEL: func.func @zero_dim_unknown_width
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK-NEXT: return %[[ZERO]]
  func.func @zero_dim_unknown_width(%t: tensor<0x?xf32>) -> i32 {
    %zero = arith.constant 0 : index
    %d = tensor.dim %t, %zero : tensor<0x?xf32>
    %r = arith.index_castui %d : index to i32
    return %r : i32
  }

  // CHECK-LABEL: func.func @width_independent_forwarding
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : vector<4xindex>
  // CHECK-NEXT: return %[[ZERO]]
  func.func @width_independent_forwarding(%cond: i1, %unknown: index) -> vector<4xindex> {
    %zero = arith.constant 0 : index
    %selected = arith.select %cond, %zero, %zero : index
    %min = arith.minui %selected, %unknown : index
    %result = vector.broadcast %min : index to vector<4xindex>
    return %result : vector<4xindex>
  }

}
