// RUN: mlir-opt -int-range-optimizations -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @add_min_max
// CHECK: %[[c3:.*]] = arith.constant 3 : index
// CHECK: return %[[c3]]
func.func @add_min_max(%a: index, %b: index) -> index {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.minsi %a, %c1 : index
    %1 = arith.maxsi %0, %c1 : index
    %2 = arith.minui %b, %c2 : index
    %3 = arith.maxui %2, %c2 : index
    %4 = arith.addi %1, %3 : index
    func.return %4 : index
}

// CHECK-LABEL: func @add_lower_bound
// CHECK: %[[sge:.*]] = arith.cmpi sge
// CHECK: return %[[sge]]
func.func @add_lower_bound(%a : i32, %b : i32) -> i1 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %0 = arith.maxsi %a, %c1 : i32
    %1 = arith.maxsi %b, %c1 : i32
    %2 = arith.addi %0, %1 : i32
    %3 = arith.cmpi sge, %2, %c2 : i32
    %4 = arith.cmpi uge, %2, %c2 : i32
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @sub_signed_vs_unsigned
// CHECK-NOT: arith.cmpi sle
// CHECK: %[[unsigned:.*]] = arith.cmpi ule
// CHECK: return %[[unsigned]] : i1
func.func @sub_signed_vs_unsigned(%v : i64) -> i1 {
    %c0 = arith.constant 0 : i64
    %c2 = arith.constant 2 : i64
    %c-5 = arith.constant -5 : i64
    %0 = arith.minsi %v, %c2 : i64
    %1 = arith.maxsi %0, %c-5 : i64
    %2 = arith.subi %1, %c2 : i64
    %3 = arith.cmpi sle, %2, %c0 : i64
    %4 = arith.cmpi ule, %2, %c0 : i64
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @multiply_negatives
// CHECK: %[[false:.*]] = arith.constant false
// CHECK: return %[[false]]
func.func @multiply_negatives(%a : index, %b : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c_1 = arith.constant -1 : index
    %c_2 = arith.constant -2 : index
    %c_4 = arith.constant -4 : index
    %c_12 = arith.constant -12 : index
    %0 = arith.maxsi %a, %c2 : index
    %1 = arith.minsi %0, %c3 : index
    %2 = arith.minsi %b, %c_1 : index
    %3 = arith.maxsi %2, %c_4 : index
    %4 = arith.muli %1, %3 : index
    %5 = arith.cmpi slt, %4, %c_12 : index
    %6 = arith.cmpi slt, %c_1, %4 : index
    %7 = arith.ori %5, %6 : i1
    func.return %7 : i1
}

// CHECK-LABEL: func @multiply_unsigned_bounds
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @multiply_unsigned_bounds(%a : i16, %b : i16) -> i1 {
    %c0 = arith.constant 0 : i16
    %c4 = arith.constant 4 : i16
    %c_mask = arith.constant 0x3fff : i16
    %c_bound = arith.constant 0xfffc : i16
    %0 = arith.andi %a, %c_mask : i16
    %1 = arith.minui %b, %c4 : i16
    %2 = arith.muli %0, %1 : i16
    %3 = arith.cmpi uge, %2, %c0 : i16
    %4 = arith.cmpi ule, %2, %c_bound : i16
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: @for_loop_with_increasing_arg
// CHECK: %[[ret:.*]] = arith.cmpi ule
// CHECK: return %[[ret]]
func.func @for_loop_with_increasing_arg() -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %0 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %c0) -> index {
        %10 = arith.addi %arg0, %arg1 : index
        scf.yield %10 : index
    }
    %1 = arith.cmpi ule, %0, %c16 : index
    func.return %1 : i1
}

// CHECK-LABEL: @for_loop_with_constant_result
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @for_loop_with_constant_result() -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true
    %0 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %true) -> i1 {
        %10 = arith.cmpi ule, %arg0, %c4 : index
        %11 = arith.andi %10, %arg1 : i1
        scf.yield %11 : i1
    }
    func.return %0 : i1
}

// Test to catch a bug present in some versions of the data flow analysis
// CHECK-LABEL: func @while_false
// CHECK: %[[false:.*]] = arith.constant false
// CHECK: scf.condition(%[[false]])
func.func @while_false(%arg0 : index) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.divui %arg0, %c2 : index
    %1 = scf.while (%arg1 = %0) : (index) -> index {
        %2 = arith.cmpi slt, %arg1, %c0 : index
        scf.condition(%2) %arg1 : index
    } do {
    ^bb0(%arg2 : index):
        scf.yield %c2 : index
    }
    func.return %1 : index
}

// CHECK-LABEL: func @div_bounds_positive
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @div_bounds_positive(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %0 = arith.maxsi %arg0, %c2 : index
    %1 = arith.divsi %c4, %0 : index
    %2 = arith.divui %c4, %0 : index

    %3 = arith.cmpi sge, %1, %c0 : index
    %4 = arith.cmpi sle, %1, %c2 : index
    %5 = arith.cmpi sge, %2, %c0 : index
    %6 = arith.cmpi sle, %1, %c2 : index

    %7 = arith.andi %3, %4 : i1
    %8 = arith.andi %7, %5 : i1
    %9 = arith.andi %8, %6 : i1
    func.return %9 : i1
}

// CHECK-LABEL: func @div_bounds_negative
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @div_bounds_negative(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c_2 = arith.constant -2 : index
    %c4 = arith.constant 4 : index
    %0 = arith.minsi %arg0, %c_2 : index
    %1 = arith.divsi %c4, %0 : index
    %2 = arith.divui %c4, %0 : index

    %3 = arith.cmpi sle, %1, %c0 : index
    %4 = arith.cmpi sge, %1, %c_2 : index
    %5 = arith.cmpi eq, %2, %c0 : index

    %7 = arith.andi %3, %4 : i1
    %8 = arith.andi %7, %5 : i1
    func.return %8 : i1
}

// CHECK-LABEL: func @div_zero_undefined
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @div_zero_undefined(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.andi %arg0, %c1 : index
    %1 = arith.divui %c4, %0 : index
    %2 = arith.cmpi ule, %1, %c4 : index
    func.return %2 : i1
}

// CHECK-LABEL: func @div_refine_min
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @div_refine_min(%arg0 : index) -> i1 {
    %c0 = arith.constant 1 : index
    %c1 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %0 = arith.andi %arg0, %c1 : index
    %1 = arith.divui %c4, %0 : index
    %2 = arith.cmpi uge, %1, %c0 : index
    func.return %2 : i1
}

// CHECK-LABEL: func @ceil_divui
// CHECK: %[[ret:.*]] = arith.cmpi eq
// CHECK: return %[[ret]]
func.func @ceil_divui(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %0 = arith.minui %arg0, %c3 : index
    %1 = arith.maxui %0, %c1 : index
    %2 = arith.ceildivui %1, %c4 : index
    %3 = arith.cmpi eq, %2, %c1 : index

    %4 = arith.maxui %0, %c0 : index
    %5 = arith.ceildivui %4, %c4 : index
    %6 = arith.cmpi eq, %5, %c1 : index
    %7 = arith.andi %3, %6 : i1
    func.return %7 : i1
}

// CHECK-LABEL: func @ceil_divsi
// CHECK: %[[ret:.*]] = arith.cmpi eq
// CHECK: return %[[ret]]
func.func @ceil_divsi(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c-4 = arith.constant -4 : index

    %0 = arith.minsi %arg0, %c3 : index
    %1 = arith.maxsi %0, %c1 : index
    %2 = arith.ceildivsi %1, %c4 : index
    %3 = arith.cmpi eq, %2, %c1 : index
    %4 = arith.ceildivsi %1, %c-4 : index
    %5 = arith.cmpi eq, %4, %c0 : index
    %6 = arith.andi %3, %5 : i1

    %7 = arith.maxsi %0, %c0 : index
    %8 = arith.ceildivsi %7, %c4 : index
    %9 = arith.cmpi eq, %8, %c1 : index
    %10 = arith.andi %6, %9 : i1
    func.return %10 : i1
}

// There was a bug, which was causing this expr errorneously fold to constant
// CHECK-LABEL: func @ceil_divsi_full_range
// CHECK-SAME: (%[[arg:.*]]: index)
// CHECK: %[[c64:.*]] = arith.constant 64 : index
// CHECK: %[[ret:.*]] = arith.ceildivsi %[[arg]], %[[c64]] : index
// CHECK: return %[[ret]]
func.func @ceil_divsi_full_range(%6: index) -> index {
  %c64 = arith.constant 64 : index
  %55 = arith.ceildivsi %6, %c64 : index
  return %55 : index
}

// CHECK-LABEL: func @ceil_divsi_intmin_bug_115293
// CHECK: %[[ret:.*]] = arith.constant true
// CHECK: return %[[ret]]
func.func @ceil_divsi_intmin_bug_115293() -> i1 {
    %intMin_i64 = test.with_bounds { smin = -9223372036854775808 : si64, smax = -9223372036854775808 : si64, umin = 9223372036854775808 : ui64, umax = 9223372036854775808 : ui64 } : i64
    %denom_i64 = test.with_bounds { smin = 1189465982 : si64, smax = 1189465982 : si64, umin = 1189465982 : ui64, umax = 1189465982 : ui64 } : i64
    %res_i64 = test.with_bounds { smin = 7754212542 : si64, smax = 7754212542 : si64, umin = 7754212542 : ui64, umax = 7754212542 : ui64 }  : i64

    %0 = arith.ceildivsi %intMin_i64, %denom_i64 : i64
    %1 = arith.cmpi eq, %0, %res_i64 : i64
    func.return %1 : i1
}

// CHECK-LABEL: func @floor_divsi
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @floor_divsi(%arg0 : index) -> i1 {
    %c4 = arith.constant 4 : index
    %c-1 = arith.constant -1 : index
    %c-3 = arith.constant -3 : index
    %c-4 = arith.constant -4 : index

    %0 = arith.minsi %arg0, %c-1 : index
    %1 = arith.maxsi %0, %c-4 : index
    %2 = arith.floordivsi %1, %c4 : index
    %3 = arith.cmpi eq, %2, %c-1 : index
    func.return %3 : i1
}

// CHECK-LABEL: func @remui_base
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @remui_base(%arg0 : index, %arg1 : index ) -> i1 {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index

    %0 = arith.minui %arg1, %c4 : index
    %1 = arith.maxui %0, %c2 : index
    %2 = arith.remui %arg0, %1 : index
    %3 = arith.cmpi ult, %2, %c4 : index
    func.return %3 : i1
}

// CHECK-LABEL: func @remui_base_maybe_zero
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @remui_base_maybe_zero(%arg0 : index, %arg1 : index ) -> i1 {
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index

    %0 = arith.minui %arg1, %c4 : index
    %1 = arith.remui %arg0, %0 : index
    %2 = arith.cmpi ult, %1, %c5 : index
    func.return %2 : i1
}

// CHECK-LABEL: func @remsi_base
// CHECK: %[[ret:.*]] = arith.cmpi sge
// CHECK: return %[[ret]]
func.func @remsi_base(%arg0 : index, %arg1 : index ) -> i1 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c-4 = arith.constant -4 : index
    %true = arith.constant true

    %0 = arith.minsi %arg1, %c4 : index
    %1 = arith.maxsi %0, %c2 : index
    %2 = arith.remsi %arg0, %1 : index
    %3 = arith.cmpi sgt, %2, %c-4 : index
    %4 = arith.cmpi slt, %2, %c4 : index
    %5 = arith.cmpi sge, %2, %c0 : index
    %6 = arith.andi %3, %4 : i1
    %7 = arith.andi %5, %6 : i1
    func.return %7 : i1
}

// CHECK-LABEL: func @remsi_positive
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @remsi_positive(%arg0 : index, %arg1 : index ) -> i1 {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %true = arith.constant true

    %0 = arith.minsi %arg1, %c4 : index
    %1 = arith.maxsi %0, %c2 : index
    %2 = arith.maxsi %arg0, %c0 : index
    %3 = arith.remsi %2, %1 : index
    %4 = arith.cmpi sge, %3, %c0 : index
    %5 = arith.cmpi slt, %3, %c4 : index
    %6 = arith.andi %4, %5 : i1
    func.return %6 : i1
}

// CHECK-LABEL: func @remui_restricted
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @remui_restricted(%arg0 : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    %0 = arith.minui %arg0, %c3 : index
    %1 = arith.maxui %0, %c2 : index
    %2 = arith.remui %1, %c4 : index
    %3 = arith.cmpi ule, %2, %c3 : index
    %4 = arith.cmpi uge, %2, %c2 : index
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @remsi_restricted
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @remsi_restricted(%arg0 : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c-4 = arith.constant -4 : index

    %0 = arith.minsi %arg0, %c3 : index
    %1 = arith.maxsi %0, %c2 : index
    %2 = arith.remsi %1, %c-4 : index
    %3 = arith.cmpi ule, %2, %c3 : index
    %4 = arith.cmpi uge, %2, %c2 : index
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @remui_restricted_fails
// CHECK: %[[ret:.*]] = arith.cmpi ne
// CHECK: return %[[ret]]
func.func @remui_restricted_fails(%arg0 : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index

    %0 = arith.minui %arg0, %c5 : index
    %1 = arith.maxui %0, %c3 : index
    %2 = arith.remui %1, %c4 : index
    %3 = arith.cmpi ne, %2, %c2 : index
    func.return %3 : i1
}

// CHECK-LABEL: func @remsi_restricted_fails
// CHECK: %[[ret:.*]] = arith.cmpi ne
// CHECK: return %[[ret]]
func.func @remsi_restricted_fails(%arg0 : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c-4 = arith.constant -4 : index

    %0 = arith.minsi %arg0, %c5 : index
    %1 = arith.maxsi %0, %c3 : index
    %2 = arith.remsi %1, %c-4 : index
    %3 = arith.cmpi ne, %2, %c2 : index
    func.return %3 : i1
}

// CHECK-LABEL: func @andi
// CHECK: %[[ret:.*]] = arith.cmpi ugt
// CHECK: return %[[ret]]
func.func @andi(%arg0 : index) -> i1 {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c7 = arith.constant 7 : index

    %0 = arith.minsi %arg0, %c5 : index
    %1 = arith.maxsi %0, %c2 : index
    %2 = arith.andi %1, %c7 : index
    %3 = arith.cmpi ugt, %2, %c5 : index
    %4 = arith.cmpi ule, %2, %c7 : index
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @andi_doesnt_make_nonnegative
// CHECK: %[[ret:.*]] = arith.cmpi sge
// CHECK: return %[[ret]]
func.func @andi_doesnt_make_nonnegative(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.addi %arg0, %c1 : index
    %1 = arith.andi %arg0, %0 : index
    %2 = arith.cmpi sge, %1, %c0 : index
    func.return %2 : i1
}


// CHECK-LABEL: func @ori
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @ori(%arg0 : i128, %arg1 : i128) -> i1 {
    %c-1 = arith.constant -1 : i128
    %c0 = arith.constant 0 : i128

    %0 = arith.minsi %arg1, %c-1 : i128
    %1 = arith.ori %arg0, %0 : i128
    %2 = arith.cmpi slt, %1, %c0 : i128
    func.return %2 : i1
}

// CHECK-LABEL: func @xori_issue_82168
// arith.cmpi was erroneously folded to %false, see Issue #82168.
// CHECK: %[[R:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : i64
// CHECK: return %[[R]]
func.func @xori_issue_82168() -> i1 {
    %c0_i64 = arith.constant 0 : i64
    %c2060639849_i64 = arith.constant 2060639849 : i64
    %2 = test.with_bounds { umin = 2060639849 : i64, umax = 2060639850 : i64, smin = 2060639849 : i64, smax = 2060639850 : i64 } : i64
    %3 = arith.xori %2, %c2060639849_i64 : i64
    %4 = arith.cmpi eq, %3, %c0_i64 : i64
    func.return %4 : i1
}

// CHECK-LABEL: func @xori_i1
//   CHECK-DAG: %[[true:.*]] = arith.constant true
//   CHECK-DAG: %[[false:.*]] = arith.constant false
//       CHECK: return %[[true]], %[[false]]
func.func @xori_i1() -> (i1, i1) {
    %true = arith.constant true
    %1 = test.with_bounds { umin = 0 : i1, umax = 0 : i1, smin = 0 : i1, smax = 0 : i1 } : i1
    %2 = test.with_bounds { umin = 1 : i1, umax = 1 : i1, smin = 1 : i1, smax = 1 : i1 } : i1
    %3 = arith.xori %1, %true : i1
    %4 = arith.xori %2, %true : i1
    func.return %3, %4 : i1, i1
}

// CHECK-LABEL: func @xori
// CHECK: %[[false:.*]] = arith.constant false
// CHECK: return %[[false]]
func.func @xori(%arg0 : i64, %arg1 : i64) -> i1 {
    %c0 = arith.constant 0 : i64
    %c7 = arith.constant 7 : i64
    %c15 = arith.constant 15 : i64
    %true = arith.constant true

    %0 = arith.minui %arg0, %c7 : i64
    %1 = arith.minui %arg1, %c15 : i64
    %2 = arith.xori %0, %1 : i64
    %3 = arith.cmpi sle, %2, %c15 : i64
    %4 = arith.xori %3, %true : i1
    func.return %4 : i1
}

// CHECK-LABEL: func @extui
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @extui(%arg0 : i16) -> i1 {
    %ci16_max = arith.constant 0xffff : i32
    %0 = arith.extui %arg0 : i16 to i32
    %1 = arith.cmpi ule, %0, %ci16_max : i32
    func.return %1 : i1
}

// CHECK-LABEL: func @extsi
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @extsi(%arg0 : i16) -> i1 {
    %ci16_smax = arith.constant 0x7fff : i32
    %ci16_smin = arith.constant 0xffff8000 : i32
    %0 = arith.extsi %arg0 : i16 to i32
    %1 = arith.cmpi sle, %0, %ci16_smax : i32
    %2 = arith.cmpi sge, %0, %ci16_smin : i32
    %3 = arith.andi %1, %2 : i1
    func.return %3 : i1
}

// CHECK-LABEL: func @trunci
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @trunci(%arg0 : i32) -> i1 {
    %c-14_i32 = arith.constant -14 : i32
    %c-14_i16 = arith.constant -14 : i16
    %ci16_smin = arith.constant 0xffff8000 : i32
    %0 = arith.minsi %arg0, %c-14_i32 : i32
    %1 = arith.maxsi %0, %ci16_smin : i32
    %2 = arith.trunci %1 : i32 to i16
    %3 = arith.cmpi sle, %2, %c-14_i16 : i16
    %4 = arith.extsi %2 : i16 to i32
    %5 = arith.cmpi sle, %4, %c-14_i32 : i32
    %6 = arith.cmpi sge, %4, %ci16_smin : i32
    %7 = arith.andi %3, %5 : i1
    %8 = arith.andi %7, %6 : i1
    func.return %8 : i1
}

// CHECK-LABEL: func @index_cast
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @index_cast(%arg0 : index) -> i1 {
    %ci32_smin = arith.constant 0xffffffff80000000 : i64
    %0 = arith.index_cast %arg0 : index to i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = arith.index_cast %ci32_smin : i64 to index
    %3 = arith.cmpi sge, %1, %2 : index
    func.return %3 : i1
}

// CHECK-LABEL: func @shli
// CHECK: %[[ret:.*]] = arith.cmpi sgt
// CHECK: return %[[ret]]
func.func @shli(%arg0 : i32, %arg1 : i1) -> i1 {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c32 = arith.constant 32 : i32
    %c-1 = arith.constant -1 : i32
    %c-16 = arith.constant -16 : i32
    %0 = arith.maxsi %arg0, %c-1 : i32
    %1 = arith.minsi %0, %c2 : i32
    %2 = arith.select %arg1, %c2, %c4 : i32
    %3 = arith.shli %1, %2 : i32
    %4 = arith.cmpi sge, %3, %c-16 : i32
    %5 = arith.cmpi sle, %3, %c32 : i32
    %6 = arith.cmpi sgt, %3, %c8 : i32
    %7 = arith.andi %4, %5 : i1
    %8 = arith.andi %7, %6 : i1
    func.return %8 : i1
}

// CHECK-LABEL: func @shrui
// CHECK: %[[ret:.*]] = arith.cmpi uge
// CHECK: return %[[ret]]
func.func @shrui(%arg0 : i1) -> i1 {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c32 = arith.constant 32 : i32
    %0 = arith.select %arg0, %c2, %c4 : i32
    %1 = arith.shrui %c32, %0 : i32
    %2 = arith.cmpi ule, %1, %c8 : i32
    %3 = arith.cmpi uge, %1, %c2 : i32
    %4 = arith.cmpi uge, %1, %c8 : i32
    %5 = arith.andi %2, %3 : i1
    %6 = arith.andi %5, %4 : i1
    func.return %6 : i1
}

// CHECK-LABEL: func @shrsi
// CHECK: %[[ret:.*]] = arith.cmpi slt
// CHECK: return %[[ret]]
func.func @shrsi(%arg0 : i32, %arg1 : i1) -> i1 {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32
    %c32 = arith.constant 32 : i32
    %c-8 = arith.constant -8 : i32
    %c-32 = arith.constant -32 : i32
    %0 = arith.maxsi %arg0, %c-32 : i32
    %1 = arith.minsi %0, %c32 : i32
    %2 = arith.select %arg1, %c2, %c4 : i32
    %3 = arith.shrsi %1, %2 : i32
    %4 = arith.cmpi sge, %3, %c-8 : i32
    %5 = arith.cmpi sle, %3, %c8 : i32
    %6 = arith.cmpi slt, %3, %c2 : i32
    %7 = arith.andi %4, %5 : i1
    %8 = arith.andi %7, %6 : i1
    func.return %8 : i1
}

// CHECK-LABEL: func @no_aggressive_eq
// CHECK: %[[ret:.*]] = arith.cmpi eq
// CHECK: return %[[ret]]
func.func @no_aggressive_eq(%arg0 : index) -> i1 {
    %c1 = arith.constant 1 : index
    %0 = arith.andi %arg0, %c1 : index
    %1 = arith.minui %arg0, %c1 : index
    %2 = arith.cmpi eq, %0, %1 : index
    func.return %2 : i1
}

// CHECK-LABEL: func @select_union
// CHECK: %[[ret:.*]] = arith.cmpi ne
// CHECK: return %[[ret]]

func.func @select_union(%arg0 : index, %arg1 : i1) -> i1 {
    %c64 = arith.constant 64 : index
    %c100 = arith.constant 100 : index
    %c128 = arith.constant 128 : index
    %c192 = arith.constant 192 : index
    %0 = arith.remui %arg0, %c64 : index
    %1 = arith.addi %0, %c128 : index
    %2 = arith.select %arg1, %0, %1 : index
    %3 = arith.cmpi slt, %2, %c192 : index
    %4 = arith.cmpi ne, %c100, %2 : index
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func @if_union
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @if_union(%arg0 : index, %arg1 : i1) -> i1 {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c-1 = arith.constant -1 : index
    %c-4 = arith.constant -4 : index
    %0 = arith.minui %arg0, %c4 : index
    %1 = scf.if %arg1 -> index {
        %10 = arith.muli %0, %0 : index
        scf.yield %10 : index
    } else {
        %20 = arith.muli %0, %c-1 : index
        scf.yield %20 : index
    }
    %2 = arith.cmpi sle, %1, %c16 : index
    %3 = arith.cmpi sge, %1, %c-4 : index
    %4 = arith.andi %2, %3 : i1
    func.return %4 : i1
}

// CHECK-LABEL: func @branch_union
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @branch_union(%arg0 : index, %arg1 : i1) -> i1 {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c-1 = arith.constant -1 : index
    %c-4 = arith.constant -4 : index
    %0 = arith.minui %arg0, %c4 : index
    cf.cond_br %arg1, ^bb1, ^bb2
^bb1 :
    %1 = arith.muli %0, %0 : index
    cf.br ^bb3(%1 : index)
^bb2 :
    %2 = arith.muli %0, %c-1 : index
    cf.br ^bb3(%2 : index)
^bb3(%3 : index) :
    %4 = arith.cmpi sle, %3, %c16 : index
    %5 = arith.cmpi sge, %3, %c-4 : index
    %6 = arith.andi %4, %5 : i1
    func.return %6 : i1
}

// CHECK-LABEL: func @loop_bound_not_inferred_with_branch
// CHECK-DAG: %[[min:.*]] = arith.cmpi sge
// CHECK-DAG: %[[max:.*]] = arith.cmpi slt
// CHECK-DAG: %[[ret:.*]] = arith.andi %[[min]], %[[max]]
// CHECK: return %[[ret]]
func.func @loop_bound_not_inferred_with_branch(%arg0 : index, %arg1 : i1) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.minui %arg0, %c4 : index
    cf.br ^bb2(%c0 : index)
^bb1(%1 : index) :
    %2 = arith.addi %1, %c1 : index
    cf.br ^bb2(%2 : index)
^bb2(%3 : index):
    %4 = arith.cmpi ult, %3, %c4 : index
    cf.cond_br %4, ^bb1(%3 : index), ^bb3(%3 : index)
^bb3(%5 : index) :
    %6 = arith.cmpi sge, %5, %c0 : index
    %7 = arith.cmpi slt, %5, %c4 : index
    %8 = arith.andi %6, %7 : i1
    func.return %8 : i1
}

// Test fon a bug where the noive implementation of trunctation led to the cast
// value being set to [0, 0].
// CHECK-LABEL: func.func @truncation_spillover
// CHECK: %[[unreplaced:.*]] = arith.index_cast
// CHECK: memref.store %[[unreplaced]]
func.func @truncation_spillover(%arg0 : memref<?xi32>) -> index {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c49 = arith.constant 49 : index
    %0 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %c0) -> index {
        %1 = arith.divsi %arg2, %c49 : index
        %2 = arith.index_cast %1 : index to i32
        memref.store %2, %arg0[%c0] : memref<?xi32>
        %3 = arith.addi %arg2, %arg1 : index
        scf.yield %3 : index
    }
  func.return %0 : index
}

// CHECK-LABEL: func.func @trunc_catches_overflow
// CHECK: %[[sge:.*]] = arith.cmpi sge
// CHECK: return %[[sge]]
func.func @trunc_catches_overflow(%arg0 : i16) -> i1 {
    %c0_i16 = arith.constant 0 : i16
    %c130_i16 = arith.constant 130 : i16
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.maxui %arg0, %c0_i16 : i16
    %1 = arith.minui %0, %c130_i16 : i16
    %2 = arith.trunci %1 : i16 to i8
    %3 = arith.cmpi sge, %2, %c0_i8 : i8
    %4 = arith.cmpi uge, %2, %c0_i8 : i8
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

// CHECK-LABEL: func.func @trunc_respects_same_high_half
// CHECK: %[[false:.*]] = arith.constant false
// CHECK: return %[[false]]
func.func @trunc_respects_same_high_half(%arg0 : i16) -> i1 {
    %c256_i16 = arith.constant 256 : i16
    %c257_i16 = arith.constant 257 : i16
    %c2_i8 = arith.constant 2 : i8
    %0 = arith.maxui %arg0, %c256_i16 : i16
    %1 = arith.minui %0, %c257_i16 : i16
    %2 = arith.trunci %1 : i16 to i8
    %3 = arith.cmpi sge, %2, %c2_i8 : i8
    func.return %3 : i1
}

// CHECK-LABEL: func.func @trunc_handles_small_signed_ranges
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @trunc_handles_small_signed_ranges(%arg0 : i16) -> i1 {
    %c-2_i16 = arith.constant -2 : i16
    %c2_i16 = arith.constant 2 : i16
    %c-2_i8 = arith.constant -2 : i8
    %c2_i8 = arith.constant 2 : i8
    %0 = arith.maxsi %arg0, %c-2_i16 : i16
    %1 = arith.minsi %0, %c2_i16 : i16
    %2 = arith.trunci %1 : i16 to i8
    %3 = arith.cmpi sge, %2, %c-2_i8 : i8
    %4 = arith.cmpi sle, %2, %c2_i8 : i8
    %5 = arith.andi %3, %4 : i1
    func.return %5 : i1
}

/// Catch a bug that crept in during an earlier refactoring that made unsigned
/// extension use the signed ranges

// CHECK-LABEL: func.func @extui_uses_unsigned
// CHECK: %[[true:.*]] = arith.constant true
// CHECK: return %[[true]]
func.func @extui_uses_unsigned(%arg0 : i32) -> i1 {
    %ci32_smin = arith.constant 0x80000000 : i32
    %ci32_smin_64 = arith.constant 0x80000000 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.minui %arg0, %ci32_smin : i32
    %1 = arith.extui %0 : i32 to i64
    %2 = arith.cmpi sge, %1, %c0_i64 : i64
    %3 = arith.cmpi ule, %1, %ci32_smin_64 : i64
    %4 = arith.andi %2, %3 : i1
    func.return %4 : i1
}

/// Catch a bug that caused a crash in getLoopBoundFromFold when
/// SparseConstantPropagation is loaded in the solver.

// CHECK-LABEL:   func.func @caller(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<?xindex, 4>) {
// CHECK:           call @callee(%[[VAL_0]]) : (memref<?xindex, 4>) -> ()
// CHECK:           return
// CHECK:         }
func.func @caller(%arg0: memref<?xindex, 4>) {
  call @callee(%arg0) : (memref<?xindex, 4>) -> ()
  return
}

// CHECK-LABEL:   func.func private @callee(
// CHECK-SAME:                              %[[VAL_0:.*]]: memref<?xindex, 4>) {
// CHECK:           return
// CHECK:         }
func.func private @callee(%arg0: memref<?xindex, 4>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = affine.load %arg0[0] : memref<?xindex, 4>
  scf.for %arg1 = %c0 to %0 step %c1 {
  }
  return
}

// CHECK-LABEL: func @test_i8_bounds
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 255 : ui8, umin = 0 : ui8}
func.func @test_i8_bounds() -> i8 {
  %cst1 = arith.constant 1 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 255 : i8, smin = -128 : i8, smax = 127 : i8 } : i8
  %1 = arith.addi %0, %cst1 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_add_1
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 255 : ui8, umin = 0 : ui8}
func.func @test_add_1() -> i8 {
  %cst1 = arith.constant 1 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 255 : i8, smin = -128 : i8, smax = 127 : i8 } : i8
  %1 = arith.addi %0, %cst1 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// Tests below check inference with overflow flags.

// CHECK-LABEL: func @test_add_i8_wrap1
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 128 : ui8, umin = 1 : ui8}
func.func @test_add_i8_wrap1() -> i8 {
  %cst1 = arith.constant 1 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 127 : i8, smin = 0 : i8, smax = 127 : i8 } : i8
  // smax overflow
  %1 = arith.addi %0, %cst1 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_add_i8_wrap2
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 128 : ui8, umin = 1 : ui8}
func.func @test_add_i8_wrap2() -> i8 {
  %cst1 = arith.constant 1 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 127 : i8, smin = 0 : i8, smax = 127 : i8 } : i8
  // smax overflow
  %1 = arith.addi %0, %cst1 overflow<nuw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_add_i8_nowrap
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = 1 : si8, umax = 127 : ui8, umin = 1 : ui8}
func.func @test_add_i8_nowrap() -> i8 {
  %cst1 = arith.constant 1 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 127 : i8, smin = 0 : i8, smax = 127 : i8 } : i8
  // nsw flag stops smax from overflowing
  %1 = arith.addi %0, %cst1 overflow<nsw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_sub_i8_wrap1
// CHECK: test.reflect_bounds {smax = 5 : si8, smin = -10 : si8, umax = 255 : ui8, umin = 0 : ui8} %1 : i8
func.func @test_sub_i8_wrap1() -> i8 {
  %cst10 = arith.constant 10 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 15 : i8, smin = 0 : i8, smax = 15 : i8 } : i8
  // umin underflows
  %1 = arith.subi %0, %cst10 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_sub_i8_wrap2
// CHECK: test.reflect_bounds {smax = 5 : si8, smin = -10 : si8, umax = 255 : ui8, umin = 0 : ui8} %1 : i8
func.func @test_sub_i8_wrap2() -> i8 {
  %cst10 = arith.constant 10 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 15 : i8, smin = 0 : i8, smax = 15 : i8 } : i8
  // umin underflows
  %1 = arith.subi %0, %cst10 overflow<nsw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_sub_i8_nowrap
// CHECK: test.reflect_bounds {smax = 5 : si8, smin = 0 : si8, umax = 5 : ui8, umin = 0 : ui8}
func.func @test_sub_i8_nowrap() -> i8 {
  %cst10 = arith.constant 10 : i8
  %0 = test.with_bounds { umin = 0 : i8, umax = 15 : i8, smin = 0 : i8, smax = 15 : i8 } : i8
  // nuw flag stops umin from underflowing
  %1 = arith.subi %0, %cst10 overflow<nuw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_mul_i8_wrap
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 200 : ui8, umin = 100 : ui8}
func.func @test_mul_i8_wrap() -> i8 {
  %cst10 = arith.constant 10 : i8
  %0 = test.with_bounds { umin = 10 : i8, umax = 20 : i8, smin = 10 : i8, smax = 20 : i8 } : i8
  // smax overflows
  %1 = arith.muli %0, %cst10 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_mul_i8_nowrap
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = 100 : si8, umax = 127 : ui8, umin = 100 : ui8}
func.func @test_mul_i8_nowrap() -> i8 {
  %cst10 = arith.constant 10 : i8
  %0 = test.with_bounds { umin = 10 : i8, umax = 20 : i8, smin = 10 : i8, smax = 20 : i8 } : i8
  // nsw stops overflow
  %1 = arith.muli %0, %cst10 overflow<nsw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_shl_i8_wrap1
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 160 : ui8, umin = 80 : ui8}
func.func @test_shl_i8_wrap1() -> i8 {
  %cst3 = arith.constant 3 : i8
  %0 = test.with_bounds { umin = 10 : i8, umax = 20 : i8, smin = 10 : i8, smax = 20 : i8 } : i8
  // smax overflows
  %1 = arith.shli %0, %cst3 : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_shl_i8_wrap2
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = -128 : si8, umax = 160 : ui8, umin = 80 : ui8}
func.func @test_shl_i8_wrap2() -> i8 {
  %cst3 = arith.constant 3 : i8
  %0 = test.with_bounds { umin = 10 : i8, umax = 20 : i8, smin = 10 : i8, smax = 20 : i8 } : i8
  // smax overflows
  %1 = arith.shli %0, %cst3 overflow<nuw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

// CHECK-LABEL: func @test_shl_i8_nowrap
// CHECK: test.reflect_bounds {smax = 127 : si8, smin = 80 : si8, umax = 127 : ui8, umin = 80 : ui8}
func.func @test_shl_i8_nowrap() -> i8 {
  %cst3 = arith.constant 3 : i8
  %0 = test.with_bounds { umin = 10 : i8, umax = 20 : ui8, smin = 10 : i8, smax = 20 : i8 } : i8
  // nsw stops smax overflow
  %1 = arith.shli %0, %cst3 overflow<nsw> : i8
  %2 = test.reflect_bounds %1 : i8
  return %2: i8
}

/// A test case to ensure that the ranges for unsupported ops are initialized
/// properly to maxRange, rather than left uninitialized.
/// In this test case, the previous behavior would leave the ranges for %a and
/// %b uninitialized, resulting in arith.cmpf's range not being updated, even
/// though it has an integer valued result.

// CHECK-LABEL: func @test_cmpf_propagates
// CHECK: test.reflect_bounds {smax = 2 : index, smin = 1 : index, umax = 2 : index, umin = 1 : index}
func.func @test_cmpf_propagates(%a: f32, %b: f32) -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %0 = arith.cmpf ueq, %a, %b : f32
  %1 = arith.select %0, %c1, %c2 : index
  %2 = test.reflect_bounds %1 : index
  func.return %2 : index
}

// CHECK-LABEL: func @zero_trip_loop
func.func @zero_trip_loop() {
  %idx1 = arith.constant 1 : index
  scf.for %arg0 = %idx1 to %idx1 step %idx1 {
    %138 = index.floordivs %arg0, %arg0
  }
  return
}

// CHECK-LABEL: func @zero_trip_loop2
func.func @zero_trip_loop2() {
  %idx1 = arith.constant 1 : index
  %idxm1 = arith.constant -1 : index
  scf.for %arg0 = %idx1 to %idx1 step %idxm1 {
    %138 = index.floordivs %arg0, %arg0
  }
  return
}
