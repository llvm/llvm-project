// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

def int_range_alias_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, Range<ArgIndex<0>, 0, 9>]>;

def int_rangeset_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [5, 8]]>]>;

def int_rangeset_coalesce_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [3, 4], [6, 8]]>]>;

def int_rangeset_terminal_tst : Intrinsic<
    [llvm_i32_ty], [],
    [RangeSet<RetIndex, [[1, 2147483647]]>]>;

// CHECK-DAG: Attribute::get(C, Attribute::Range, ConstantRange(APInt(BitWidth, 0, {{.*}} APInt(BitWidth, 9,

// CHECK-DAG: Attribute::get(C, Attribute::RangeSet, ArrayRef<ConstantRange>{ConstantRange(APInt(BitWidth, 0, {{.*}} APInt(BitWidth, 3, {{.*}} ConstantRange(APInt(BitWidth, 5, {{.*}} APInt(BitWidth, 9,

// CHECK-DAG: Attribute::get(C, Attribute::RangeSet, ArrayRef<ConstantRange>{ConstantRange(APInt(BitWidth, 0, {{.*}} APInt(BitWidth, 5, {{.*}} ConstantRange(APInt(BitWidth, 6, {{.*}} APInt(BitWidth, 9,

// CHECK-DAG: Attribute::get(C, Attribute::RangeSet, ArrayRef<ConstantRange>{ConstantRange(APInt(BitWidth, 1, {{.*}} APInt(BitWidth, 2147483648,
