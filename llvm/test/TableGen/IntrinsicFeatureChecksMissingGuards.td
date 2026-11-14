// RUN: not llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s 2>&1 | FileCheck %s

// CHECK: error: intrinsic int_test_missing_guards must have at least one TargetGuard variant

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

let TargetPrefix = "test" in
def int_test_missing_guards
    : GuardedDefaultIntrinsic<[], [], [IntrNoMem]>;
