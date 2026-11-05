// RUN: not llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s 2>&1 | FileCheck %s

// CHECK: error: GuardedIntrinsic must not specify TargetFeatures; use TargetGuards instead

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;

let TargetPrefix = "test" in
def int_test_guarded_with_target_features
    : GuardedDefaultIntrinsic<[], [], [IntrNoMem]> {
  let TargetFeatures = "example";
  let TargetGuards = [GuardedVariant<["example"]>];
}
