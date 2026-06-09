// RUN: llvm-tblgen -gen-subtarget -I %p/../../include %s | FileCheck %s --check-prefix=SUBTARGET
// RUN: llvm-tblgen -gen-target-features -I %p/../../include %s | FileCheck %s --check-prefix=BASIC

include "llvm/Target/Target.td"

def MyTarget : Target;

def FeatureA : SubtargetFeature<"a", "", "", "">;
def FeatureB : SubtargetFeature<"b", "", "", "", [FeatureA]>;
def FeatureC : SubtargetFeature<"c", "", "", "", [FeatureA]>;
def FeatureD : SubtargetFeature<"d", "", "", "", [FeatureB]>;
def HiddenFeature : SubtargetFeature<"", "", "", "", [FeatureD]>;

def P0 : ProcessorModel<"cpu0", NoSchedModel,
                        /*Features=*/[FeatureA],
                        /*TuneFeatures=*/[FeatureB]>;
def P1 : ProcessorModel<"cpu1", NoSchedModel,
                        /*Features=*/[FeatureA, FeatureB]>;
def P2 : ProcessorModel<"cpu2", NoSchedModel,
                        /*Features=*/[HiddenFeature]>;

// SUBTARGET: extern const llvm::SubtargetFeatureKV MyTargetFeatureKV[] = {
// SUBTARGET-NEXT: { "a", "", MyTarget::FeatureA, 0 },
// SUBTARGET-NEXT: { "b", "", MyTarget::FeatureB, 1 },
// SUBTARGET-NEXT: { "c", "", MyTarget::FeatureC, 1 },
// SUBTARGET-NEXT: { "d", "", MyTarget::FeatureD, 2 },
// SUBTARGET-NEXT: };

// SUBTARGET: extern const llvm::SubtargetSubTypeKV MyTargetSubTypeKV[] = {
// SUBTARGET-NEXT: { "cpu0", 1, 2, &NoSchedModel },
// SUBTARGET-NEXT: { "cpu1", 2, 0, &NoSchedModel },
// SUBTARGET-NEXT: { "cpu2", 3, 0, &NoSchedModel },
// SUBTARGET-NEXT: };

// SUBTARGET: extern const llvm::FeatureBitArray MyTargetFeatureBitsets[] = {
// SUBTARGET-NEXT: { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// SUBTARGET-NEXT: { { { 0x1ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// SUBTARGET-NEXT: { { { 0x3ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// SUBTARGET-NEXT: { { { 0x10ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// SUBTARGET-NEXT: };

// BASIC: extern const llvm::BasicSubtargetFeatureKV BasicMyTargetFeatureKV[] = {
// BASIC-NEXT: { "a", MyTarget::FeatureA, 0 },
// BASIC-NEXT: { "b", MyTarget::FeatureB, 1 },
// BASIC-NEXT: { "c", MyTarget::FeatureC, 1 },
// BASIC-NEXT: { "d", MyTarget::FeatureD, 2 },
// BASIC-NEXT: };

// BASIC: extern const llvm::BasicSubtargetSubTypeKV BasicMyTargetSubTypeKV[] = {
// BASIC-NEXT: { "cpu0", 1 },
// BASIC-NEXT: { "cpu1", 2 },
// BASIC-NEXT: { "cpu2", 3 },
// BASIC-NEXT: };

// BASIC: extern const llvm::FeatureBitArray BasicMyTargetFeatureBitsets[] = {
// BASIC-NEXT: { { { 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// BASIC-NEXT: { { { 0x1ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// BASIC-NEXT: { { { 0x3ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// BASIC-NEXT: { { { 0x10ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, 0x0ULL, } } },
// BASIC-NEXT: };
