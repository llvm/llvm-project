// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DEMPTY %s 2>&1 | FileCheck --check-prefix=EMPTY %s -DFILE=%s
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DBAD_BOUNDS %s 2>&1 | FileCheck --check-prefix=BAD-BOUNDS %s -DFILE=%s
// RUN: not llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DOVERLAP %s 2>&1 | FileCheck --check-prefix=OVERLAP %s -DFILE=%s

#define TEST_INTRINSICS_SUPPRESS_DEFS
include "llvm/IR/Intrinsics.td"

#ifdef EMPTY
// EMPTY: error: assertion failed: RangeSet requires at least one range
def int_rangeset_empty : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, []>]>;
#endif

#ifdef BAD_BOUNDS
// BAD-BOUNDS: error: assertion failed: RangeSet requires lower <= upper for each range
def int_rangeset_bad_bounds : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[4, 3]]>]>;
#endif

#ifdef OVERLAP
// OVERLAP: error: assertion failed: RangeSet requires ordered and non-overlapping ranges
def int_rangeset_overlap : Intrinsic<
    [llvm_i32_ty], [llvm_i16_ty],
    [ImmArg<ArgIndex<0>>, RangeSet<ArgIndex<0>, [[0, 2], [2, 4]]>]>;
#endif
