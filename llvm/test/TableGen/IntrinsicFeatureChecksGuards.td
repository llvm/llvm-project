// RUN: llvm-tblgen -gen-intrinsic-feature-checks \
// RUN:   -intrinsic-feature-prefix=test -intrinsic-feature-namespace=TestTarget \
// RUN:   -I %p/../../include %s | FileCheck %s

// A TargetGuard without overload types applies to every valid overload.
// A target can define its own feature enum namespace.
// The generated table is also compiled by IntrinsicFeatureEmitterTest.
// CHECK: static ArrayRef<FeatureBitset> getTestTargetIntrinsicFeatureAlternatives(
// CHECK-NEXT: static constexpr FeatureBitset TestTargetIntrinsicFeatureClauses[] = {
// CHECK-NEXT: {TestTarget::FeatureExample},
// CHECK-NEXT: {TestTarget::FeatureOther},
// CHECK: TestTargetIntrinsicFeatureClauses + Guard.FirstClause, Guard.NumClauses
// CHECK: static FunctionType *getTestTargetIntrinsicFunctionType(
// CHECK: static constexpr MVT::SimpleValueType TestTargetIntrinsicOverloadTypes[] = {
// CHECK-NEXT: // int_test_one_guard
// CHECK-NEXT: // int_test_two_guards
// CHECK-NEXT: MVT::i32, // int_test_type_specific_guards
// CHECK-NEXT: MVT::i64, // int_test_type_specific_guards
// CHECK-NEXT: // int_test_typed
// CHECK: return Intrinsic::getType(Context, ID, OverloadTypes);
// CHECK: static const IntrinsicFeatureGuard &getTestTargetIntrinsicFeatureGuard(
// CHECK-NEXT: static constexpr unsigned TestTargetIntrinsicGuardIndices[] = {
// CHECK: static constexpr IntrinsicFeatureGuard TestTargetIntrinsicFeatureGuards[] = {
// CHECK: assert(Index < Variant.NumGuards && "invalid intrinsic feature guard index");
// CHECK: TestTargetIntrinsicGuardIndices[Variant.FirstGuard + Index]
// CHECK: static ArrayRef<IntrinsicFeatureVariant> getTestTargetIntrinsicVariants
// CHECK-NEXT: static constexpr IntrinsicFeatureVariant TestTargetIntrinsicVariants[] = {
// CHECK-NEXT: {0, 0, 0, 1, true}, // int_test_one_guard
// CHECK-NEXT: {0, 0, 1, 2, true}, // int_test_two_guards
// CHECK-NEXT: {0, 1, 3, 1, false}, // int_test_type_specific_guards
// CHECK-NEXT: {1, 1, 4, 1, false}, // int_test_type_specific_guards
// CHECK-NEXT: {2, 0, 5, 1, false}, // int_test_typed
// CHECK: case Intrinsic::test_one_guard:
// CHECK-NEXT: return ArrayRef(TestTargetIntrinsicVariants + 0, 1);
// CHECK: case Intrinsic::test_two_guards:
// CHECK-NEXT: return ArrayRef(TestTargetIntrinsicVariants + 1, 1);
// CHECK: case Intrinsic::test_type_specific_guards:
// CHECK-NEXT: return ArrayRef(TestTargetIntrinsicVariants + 2, 2);

include "llvm/IR/Intrinsics.td"
include "llvm/Target/TargetSubtarget.td"

def FeatureExample : SubtargetFeature<"example", "Example", "true", "">;
def FeatureOther : SubtargetFeature<"other", "Other", "true", "">;

let TargetPrefix = "test" in
def int_test_one_guard
    : GuardedDefaultIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
                              [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"]>];
}

let TargetPrefix = "test" in
def int_test_two_guards
    : GuardedDefaultIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
                              [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example", "other"]>];
}

let TargetPrefix = "test" in
def int_test_type_specific_guards
    : GuardedDefaultIntrinsic<[llvm_anyint_ty], [LLVMMatchType<0>],
                              [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"], [[llvm_i32_ty]]>,
                      GuardedVariant<["other"], [[llvm_i64_ty]]>];
}

let TargetPrefix = "test" in
def int_test_typed
    : GuardedDefaultIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]> {
  let TargetGuards = [GuardedVariant<["example"], [[]<LLVMType>]>];
}
