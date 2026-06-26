// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

def int_rangeset_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [5, 8]]>]>;

def int_rangeset_coalesce_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [3, 4], [6, 8]]>]>;

// CHECK-NOT: Attribute::RangeSet

// CHECK: bool Intrinsic::isImmArgValueInRangeSet

// CHECK: case rangeset_coalesce_tst:
// CHECK: case 0:
// CHECK: APInt(Value.getBitWidth(), 0, {{.*}}APInt(Value.getBitWidth(), 5,
// CHECK: APInt(Value.getBitWidth(), 6, {{.*}}APInt(Value.getBitWidth(), 9,

// CHECK: case rangeset_tst:
// CHECK: case 0:
// CHECK: APInt(Value.getBitWidth(), 0, {{.*}}APInt(Value.getBitWidth(), 3,
// CHECK: APInt(Value.getBitWidth(), 5, {{.*}}APInt(Value.getBitWidth(), 9,
