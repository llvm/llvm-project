// RUN: llvm-tblgen -gen-intrinsic-feature-checks -intrinsic-feature-prefix=aarch64 -intrinsic-feature-namespace=AArch64 \
// RUN:   -I %p/../../include -I %p/../../lib/Target/AArch64 \
// RUN:   %p/../../lib/Target/AArch64/AArch64IntrinsicFeatureChecks.td | FileCheck %s

// Guards are parsed literally; the verifier supplies the SME base feature.
// CHECK-NOT: AArch64StreamingModeFeatures
// CHECK: static ArrayRef<FeatureBitset> getAArch64IntrinsicFeatureAlternatives
// CHECK-NEXT: static constexpr FeatureBitset AArch64IntrinsicFeatureClauses[] = {
// CHECK: {AArch64::FeatureSME2p1, AArch64::FeatureSVE},
// CHECK-NEXT: {AArch64::FeatureSVE, AArch64::FeatureSVE2p1},
// CHECK: {AArch64::FeatureSVE2},
// CHECK: {AArch64::FeatureLUT, AArch64::FeatureSME2, AArch64::FeatureSVE},
// CHECK-NEXT: {AArch64::FeatureLUT, AArch64::FeatureSVE, AArch64::FeatureSVE2},
// Variants store only the overload types used by Intrinsic::getType().
// CHECK: static FunctionType *getAArch64IntrinsicFunctionType(
// CHECK-NEXT: static constexpr MVT::SimpleValueType AArch64IntrinsicOverloadTypes[] = {
// CHECK: MVT::nxv8bf16, // int_aarch64_sve_fadd
// CHECK: MVT::nxv4f32, // int_aarch64_sve_fcvt_x2
// CHECK: MVT::nxv2i64, MVT::nxv2f64, // int_aarch64_sve_fcvtzs
// CHECK: MVT::nxv16i8, MVT::iPTR, // int_aarch64_sve_ld2_sret
// CHECK: MVT::iPTR, MVT::nxv2i64, // int_aarch64_sve_prfb_gather_index
// CHECK: MVT::nxv16i1, // int_aarch64_sve_punpkhi
// CHECK: MVT::nxv16i8, MVT::iPTR, // int_aarch64_sve_st4
// CHECK: return Intrinsic::getType(Context, ID, OverloadTypes);
// CHECK: static const IntrinsicFeatureGuard &getAArch64IntrinsicFeatureGuard
// CHECK-NEXT: static constexpr unsigned AArch64IntrinsicGuardIndices[] = {
// A streaming-compatible ZT0 intrinsic explicitly uses SME2 in both modes.
// CHECK: [[ZT_GUARD:[0-9]+]], [[ZT_GUARD]], // int_aarch64_sme_ldr_zt
// CHECK: static constexpr IntrinsicFeatureGuard AArch64IntrinsicFeatureGuards[] = {
// CHECK: 2, "sve,(sve2p1|sme2p1)"},
// CHECK-NEXT: {{.*}}"sve2p1|sme2p1"},
// CHECK: 2, "sve,(sve2|sme2),lut"},
// CHECK: static ArrayRef<IntrinsicFeatureVariant> getAArch64IntrinsicVariants
// CHECK-NEXT: static constexpr IntrinsicFeatureVariant AArch64IntrinsicVariants[] = {
// CHECK: {{[0-9]+}}, 1, {{[0-9]+}}, 2, false}, // int_aarch64_sme_ldr_zt
// CHECK: case Intrinsic::aarch64_sve_fadd:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 4);
// The contiguous load and store helpers preserve each tuple width.
// CHECK: case Intrinsic::aarch64_sve_ld1:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 14);
// CHECK: case Intrinsic::aarch64_sve_ld1rq:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
// CHECK: case Intrinsic::aarch64_sve_ld2_sret:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
// CHECK: case Intrinsic::aarch64_sve_ld4q_sret:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
// CHECK: case Intrinsic::aarch64_sve_ldnt1:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
// A single Variant can contain several overload type lists.
// CHECK: case Intrinsic::aarch64_sve_sel:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 9);
// CHECK: case Intrinsic::aarch64_sve_st2:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
// CHECK: case Intrinsic::aarch64_sve_st4q:
// CHECK-NEXT: return ArrayRef(AArch64IntrinsicVariants + {{[0-9]+}}, 8);
