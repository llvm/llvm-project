// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include %s | FileCheck %s

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

def int_rangeset_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [5, 8]]>]>;

def int_rangeset_coalesce_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [3, 4], [6, 8]]>]>;

def int_rangeset_multi_tst : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
     RangeSet<ArgIndex<0>, [[1, 1]]>,
     RangeSet<ArgIndex<1>, [[2, 3]]>]>;

// CHECK-NOT: Attribute::RangeSet

// CHECK: static constexpr ImmArgRange ImmArgRangeTable[] = {
// CHECK-DAG: {0, 3}
// CHECK-DAG: {5, 9}
// CHECK-DAG: {0, 5}
// CHECK-DAG: {6, 9}
// CHECK-DAG: {1, 2}
// CHECK-DAG: {2, 4}

// CHECK: static constexpr ImmArgRangeSet ImmArgRangeSetTable[] = {
// CHECK-DAG: {0, {{[0-9]+}}, 2}
// CHECK-DAG: {0, {{[0-9]+}}, 2}
// CHECK-DAG: {0, {{[0-9]+}}, 1}
// CHECK-DAG: {1, {{[0-9]+}}, 1}

// CHECK: static constexpr ImmArgRangeSetList ImmArgRangeSetListTable[] = {
// CHECK-DAG: // llvm.rangeset.tst
// CHECK-DAG: // llvm.rangeset.coalesce.tst
// CHECK-DAG: {1, 2}, // llvm.rangeset.multi.tst

// CHECK: bool Intrinsic::isImmArgValueInRangeSet
// CHECK-NOT: switch
// CHECK: const ImmArgRangeSetList &List = ImmArgRangeSetListTable[IID];
// CHECK: ArrayRef<ImmArgRangeSet> RangeSets(
// CHECK: for (const ImmArgRangeSet &RangeSet : RangeSets) {
// CHECK: ArrayRef<ImmArgRange> Ranges(ImmArgRangeTable + RangeSet.RangesStart,
// CHECK: for (const ImmArgRange &Range : Ranges) {
