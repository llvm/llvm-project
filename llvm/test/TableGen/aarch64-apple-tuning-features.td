// RUN: llvm-tblgen -print-records %p/../../lib/Target/AArch64/AArch64.td -I %p/../../lib/Target/AArch64 -I %p/../../include \
// RUN:    | awk '{gsub(/\[/,"[\n"); gsub(/, /,",\n"); print}' \
// RUN:    | FileCheck %s --implicit-check-not="def TuneApple"

// Verify the resolved tuning feature lists for Apple CPUs.
// Each generation inherits from the previous and applies add/remove deltas.
// When adding a tuning feature to an Apple core, update the expected list below.

// CHECK-LABEL: def TuneAppleA10 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a10";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA10";
// CHECK-NEXT:    string Desc = "Apple A10";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA11 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a11";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA11";
// CHECK-NEXT:    string Desc = "Apple A11";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA12 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a12";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA12";
// CHECK-NEXT:    string Desc = "Apple A12";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA13 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a13";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA13";
// CHECK-NEXT:    string Desc = "Apple A13";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA14 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a14";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA14";
// CHECK-NEXT:    string Desc = "Apple A14";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAggressiveFMA,
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA15 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a15";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA15";
// CHECK-NEXT:    string Desc = "Apple A15";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAdrpAdd,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA16 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a16";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA16";
// CHECK-NEXT:    string Desc = "Apple A16";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAdrpAdd,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseFCmpFCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA17 {      // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a17";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA17";
// CHECK-NEXT:    string Desc = "Apple A17";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAdrpAdd,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseFCmpFCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleA7 {       // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-a7";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleA7";
// CHECK-NEXT:    string Desc = "Apple A7 (the CPU formerly known as Cyclone)";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureStorePairSuppress,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPWorkaround,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleM4 {       // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-m4";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleM4";
// CHECK-NEXT:    string Desc = "Apple M4";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAdrpAdd,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseFCmpFCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
// CHECK-LABEL: def TuneAppleM5 {       // SubtargetFeature
// CHECK-NEXT:    string Name = "apple-m5";
// CHECK-NEXT:    string FieldName = "ARMProcFamily";
// CHECK-NEXT:    string Value = "AppleM5";
// CHECK-NEXT:    string Desc = "Apple M5";
// CHECK-NEXT:    list<SubtargetFeature> Implies = [
// CHECK-NEXT:    FeatureAlignCmpCSelPairs,
// CHECK-NEXT:    FeatureAlternateSExtLoadCVTF32Pattern,
// CHECK-NEXT:    FeatureArithmeticBccFusion,
// CHECK-NEXT:    FeatureArithmeticCbzFusion,
// CHECK-NEXT:    FeatureDisableLatencySchedHeuristic,
// CHECK-NEXT:    FeatureFastLD1Single,
// CHECK-NEXT:    FeatureFuseAddress,
// CHECK-NEXT:    FeatureFuseAdrpAdd,
// CHECK-NEXT:    FeatureFuseAES,
// CHECK-NEXT:    FeatureFuseArithmeticLogic,
// CHECK-NEXT:    FeatureFuseCryptoEOR,
// CHECK-NEXT:    FeatureFuseCmpCSel,
// CHECK-NEXT:    FeatureFuseFCmpFCSel,
// CHECK-NEXT:    FeatureFuseLiterals,
// CHECK-NEXT:    FeatureMaxInterleaveFactor4,
// CHECK-NEXT:    FeatureNoZCZeroingFPR64,
// CHECK-NEXT:    FeatureZCRegMoveFPR128,
// CHECK-NEXT:    FeatureZCRegMoveGPR64,
// CHECK-NEXT:    FeatureZCZeroingFPR128,
// CHECK-NEXT:    FeatureZCZeroingGPR32,
// CHECK-NEXT:    FeatureZCZeroingGPR64];
// CHECK-NEXT:    InlineBehavior InlineBehavior = InlineDefault;
// CHECK-NEXT: }
