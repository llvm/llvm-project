// RUN: llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s | FileCheck %s

// A TargetGuard without overload types applies to every valid overload.
// A target can define a single mode and use its own feature enum namespace.
// The generated table is also compiled by IntrinsicFeatureEmitterTest.
// CHECK: static constexpr FeatureBitset TestTargetIntrinsicFeatureClauses[] = {
// CHECK-NEXT:   {TestTarget::FeatureExample},
// CHECK: MVT::i32, MVT::i32, // int_test_typed
// CHECK: static constexpr IntrinsicFeatureVariant TestTargetIntrinsicVariants[] = {
// CHECK-NEXT:   {0, 0, 0, 0, 1, true}, // int_test_one_guard
// CHECK-NEXT:   {0, 1, 1, 1, 1, false}, // int_test_typed
// CHECK: static ArrayRef<IntrinsicFeatureVariant> getTestTargetIntrinsicVariants
// CHECK: case Intrinsic::test_one_guard:
// CHECK-NEXT: return ArrayRef(TestTargetIntrinsicVariants + 0, 1);

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;

let TargetPrefix = "test" in
def int_test_one_guard
    : GuardedDefaultIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
                              [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"]>];
}

let TargetPrefix = "test" in
def int_test_typed
    : GuardedDefaultIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"], [[]<LLVMType>]>];
}
