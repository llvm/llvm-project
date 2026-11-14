// RUN: not llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: error: type-independent TargetGuard must be the intrinsic's only TargetGuard variant

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;

let TargetPrefix = "test" in
def int_test_mixed
    : GuardedDefaultIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"]>,
                      GuardedVariant<["example"], [[]<LLVMType>]>];
}
