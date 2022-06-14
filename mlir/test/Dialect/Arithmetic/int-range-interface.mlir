// RUN: mlir-opt -test-int-range-inference -canonicalize %s | FileCheck %s

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
// CHECK: %[[ret:.*]] = arith.cmpi ule
// CHECK: return %[[ret]]
func.func @div_zero_undefined(%arg0 : index) -> i1 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %0 = arith.andi %arg0, %c1 : index
    %1 = arith.divui %c4, %0 : index
    %2 = arith.cmpi ule, %1, %c4 : index
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
    %1 = arith.trunci %0 : i32 to i16
    %2 = arith.cmpi sle, %1, %c-14_i16 : i16
    %3 = arith.extsi %1 : i16 to i32
    %4 = arith.cmpi sle, %3, %c-14_i32 : i32
    %5 = arith.cmpi sge, %3, %ci16_smin : i32
    %6 = arith.andi %2, %4 : i1
    %7 = arith.andi %6, %5 : i1
    func.return %7 : i1
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

