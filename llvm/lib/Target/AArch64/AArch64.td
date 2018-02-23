//=- AArch64.td - Describe the AArch64 Target Machine --------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing.
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// AArch64 Subtarget features.
//

def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8", "true",
                                       "Enable ARMv8 FP">;

def FeatureNEON : SubtargetFeature<"neon", "HasNEON", "true",
  "Enable Advanced SIMD instructions", [FeatureFPARMv8]>;

def FeatureCrypto : SubtargetFeature<"crypto", "HasCrypto", "true",
  "Enable cryptographic instructions", [FeatureNEON]>;

def FeatureCRC : SubtargetFeature<"crc", "HasCRC", "true",
  "Enable ARMv8 CRC-32 checksum instructions">;

def FeatureRAS : SubtargetFeature<"ras", "HasRAS", "true",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions">;

def FeatureLSE : SubtargetFeature<"lse", "HasLSE", "true",
  "Enable ARMv8.1 Large System Extension (LSE) atomic instructions">;

def FeatureRDM : SubtargetFeature<"rdm", "HasRDM", "true",
  "Enable ARMv8.1 Rounding Double Multiply Add/Subtract instructions">;

def FeaturePerfMon : SubtargetFeature<"perfmon", "HasPerfMon", "true",
  "Enable ARMv8 PMUv3 Performance Monitors extension">;

def FeatureFullFP16 : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
  "Full FP16", [FeatureFPARMv8]>;

def FeatureSPE : SubtargetFeature<"spe", "HasSPE", "true",
  "Enable Statistical Profiling extension">;

def FeatureSVE : SubtargetFeature<"sve", "HasSVE", "true",
  "Enable Scalable Vector Extension (SVE) instructions">;

/// Cyclone has register move instructions which are "free".
def FeatureZCRegMove : SubtargetFeature<"zcm", "HasZeroCycleRegMove", "true",
                                        "Has zero-cycle register moves">;

/// Cyclone has instructions which zero registers for "free".
def FeatureZCZeroing : SubtargetFeature<"zcz", "HasZeroCycleZeroing", "true",
                                        "Has zero-cycle zeroing instructions">;

def FeatureStrictAlign : SubtargetFeature<"strict-align",
                                          "StrictAlign", "true",
                                          "Disallow all unaligned memory "
                                          "access">;

def FeatureReserveX18 : SubtargetFeature<"reserve-x18", "ReserveX18", "true",
                                         "Reserve X18, making it unavailable "
                                         "as a GPR">;

def FeatureUseAA : SubtargetFeature<"use-aa", "UseAA", "true",
                                    "Use alias analysis during codegen">;

def FeatureBalanceFPOps : SubtargetFeature<"balance-fp-ops", "BalanceFPOps",
    "true",
    "balance mix of odd and even D-registers for fp multiply(-accumulate) ops">;

def FeaturePredictableSelectIsExpensive : SubtargetFeature<
    "predictable-select-expensive", "PredictableSelectIsExpensive", "true",
    "Prefer likely predicted branches over selects">;

def FeatureCustomCheapAsMoveHandling : SubtargetFeature<"custom-cheap-as-move",
    "CustomAsCheapAsMove", "true",
    "Use custom code for TargetInstrInfo::isAsCheapAsAMove()">;

def FeaturePostRAScheduler : SubtargetFeature<"use-postra-scheduler",
    "UsePostRAScheduler", "true", "Schedule again after register allocation">;

def FeatureSlowMisaligned128Store : SubtargetFeature<"slow-misaligned-128store",
    "Misaligned128StoreIsSlow", "true", "Misaligned 128 bit stores are slow">;

def FeatureSlowPaired128 : SubtargetFeature<"slow-paired-128",
    "Paired128IsSlow", "true", "Paired 128 bit loads and stores are slow">;

def FeatureAlternateSExtLoadCVTF32Pattern : SubtargetFeature<
    "alternate-sextload-cvt-f32-pattern", "UseAlternateSExtLoadCVTF32Pattern",
    "true", "Use alternative pattern for sextload convert to f32">;

def FeatureArithmeticBccFusion : SubtargetFeature<
    "arith-bcc-fusion", "HasArithmeticBccFusion", "true",
    "CPU fuses arithmetic+bcc operations">;

def FeatureArithmeticCbzFusion : SubtargetFeature<
    "arith-cbz-fusion", "HasArithmeticCbzFusion", "true",
    "CPU fuses arithmetic + cbz/cbnz operations">;

def FeatureFuseAES : SubtargetFeature<
    "fuse-aes", "HasFuseAES", "true",
    "CPU fuses AES crypto operations">;

def FeatureFuseLiterals : SubtargetFeature<
    "fuse-literals", "HasFuseLiterals", "true",
    "CPU fuses literal generation operations">;

def FeatureDisableLatencySchedHeuristic : SubtargetFeature<
    "disable-latency-sched-heuristic", "DisableLatencySchedHeuristic", "true",
    "Disable latency scheduling heuristic">;

def FeatureUseRSqrt : SubtargetFeature<
    "use-reciprocal-square-root", "UseRSqrt", "true",
    "Use the reciprocal square root approximation">;

def FeatureNoNegativeImmediates : SubtargetFeature<"no-neg-immediates",
                                        "NegativeImmediates", "false",
                                        "Convert immediates and instructions "
                                        "to their negated or complemented "
                                        "equivalent when the immediate does "
                                        "not fit in the encoding.">;

def FeatureLSLFast : SubtargetFeature<
    "lsl-fast", "HasLSLFast", "true",
    "CPU has a fastpath logical shift of up to 3 places">;
//===----------------------------------------------------------------------===//
// Architectures.
//

def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
  "Support ARM v8.1a instructions", [FeatureCRC, FeatureLSE, FeatureRDM]>;

def HasV8_2aOps : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
  "Support ARM v8.2a instructions", [HasV8_1aOps, FeatureRAS]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AArch64RegisterInfo.td"
include "AArch64RegisterBanks.td"
include "AArch64CallingConvention.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "AArch64Schedule.td"
include "AArch64InstrInfo.td"

def AArch64InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Named operands for MRS/MSR/TLBI/...
//===----------------------------------------------------------------------===//

include "AArch64SystemOperands.td"

//===----------------------------------------------------------------------===//
// AArch64 Processors supported.
//
include "AArch64SchedA53.td"
include "AArch64SchedA57.td"
include "AArch64SchedCyclone.td"
include "AArch64SchedFalkor.td"
include "AArch64SchedKryo.td"
include "AArch64SchedM1.td"
include "AArch64SchedThunderX.td"
include "AArch64SchedThunderX2T99.td"

def ProcA35     : SubtargetFeature<"a35", "ARMProcFamily", "CortexA35",
                                   "Cortex-A35 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors", [
                                   FeatureBalanceFPOps,
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeatureUseAA
                                   ]>;

def ProcA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors", [
                                   FeatureBalanceFPOps,
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureFuseLiterals,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive
                                   ]>;

def ProcA72     : SubtargetFeature<"a72", "ARMProcFamily", "CortexA72",
                                   "Cortex-A72 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureFuseAES,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcCyclone : SubtargetFeature<"cyclone", "ARMProcFamily", "Cyclone",
                                   "Cyclone", [
                                   FeatureAlternateSExtLoadCVTF32Pattern,
                                   FeatureCrypto,
                                   FeatureDisableLatencySchedHeuristic,
                                   FeatureFPARMv8,
                                   FeatureArithmeticBccFusion,
                                   FeatureArithmeticCbzFusion,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeatureSlowMisaligned128Store,
                                   FeatureZCRegMove,
                                   FeatureZCZeroing
                                   ]>;

def ProcExynosM1 : SubtargetFeature<"exynosm1", "ARMProcFamily", "ExynosM1",
                                    "Samsung Exynos-M1 processors",
                                    [FeatureSlowPaired128,
                                     FeatureCRC,
                                     FeatureCrypto,
                                     FeatureCustomCheapAsMoveHandling,
                                     FeatureFPARMv8,
                                     FeatureFuseAES,
                                     FeatureNEON,
                                     FeaturePerfMon,
                                     FeaturePostRAScheduler,
                                     FeatureSlowMisaligned128Store,
                                     FeatureUseRSqrt,
                                     FeatureZCZeroing]>;

def ProcExynosM2 : SubtargetFeature<"exynosm2", "ARMProcFamily", "ExynosM1",
                                    "Samsung Exynos-M2/M3 processors",
                                    [FeatureSlowPaired128,
                                     FeatureCRC,
                                     FeatureCrypto,
                                     FeatureCustomCheapAsMoveHandling,
                                     FeatureFPARMv8,
                                     FeatureFuseAES,
                                     FeatureNEON,
                                     FeaturePerfMon,
                                     FeaturePostRAScheduler,
                                     FeatureSlowMisaligned128Store,
                                     FeatureZCZeroing]>;

def ProcKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing,
                                   FeatureLSLFast
                                   ]>;

def ProcFalkor  : SubtargetFeature<"falkor", "ARMProcFamily", "Falkor",
                                   "Qualcomm Falkor processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureRDM,
                                   FeatureZCZeroing,
                                   FeatureLSLFast
                                   ]>;

def ProcThunderX2T99  : SubtargetFeature<"thunderx2t99", "ARMProcFamily",
                                         "ThunderX2T99",
                                         "Cavium ThunderX2 processors", [
                                          FeatureCRC,
                                          FeatureCrypto,
                                          FeatureFPARMv8,
                                          FeatureArithmeticBccFusion,
                                          FeatureNEON,
                                          FeaturePostRAScheduler,
                                          FeaturePredictableSelectIsExpensive,
                                          FeatureLSE,
                                          HasV8_1aOps]>;

def ProcThunderX : SubtargetFeature<"thunderx", "ARMProcFamily", "ThunderX",
                                    "Cavium ThunderX processors", [
                                    FeatureCRC,
                                    FeatureCrypto,
                                    FeatureFPARMv8,
                                    FeaturePerfMon,
                                    FeaturePostRAScheduler,
                                    FeaturePredictableSelectIsExpensive,
                                    FeatureNEON]>;

def ProcThunderXT88 : SubtargetFeature<"thunderxt88", "ARMProcFamily",
                                       "ThunderXT88",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def ProcThunderXT81 : SubtargetFeature<"thunderxt81", "ARMProcFamily",
                                       "ThunderXT81",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def ProcThunderXT83 : SubtargetFeature<"thunderxt83", "ARMProcFamily",
                                       "ThunderXT83",
                                       "Cavium ThunderX processors", [
                                       FeatureCRC,
                                       FeatureCrypto,
                                       FeatureFPARMv8,
                                       FeaturePerfMon,
                                       FeaturePostRAScheduler,
                                       FeaturePredictableSelectIsExpensive,
                                       FeatureNEON]>;

def : ProcessorModel<"generic", NoSchedModel, [
                     FeatureFPARMv8,
                     FeatureFuseAES,
                     FeatureNEON,
                     FeaturePerfMon,
                     FeaturePostRAScheduler
                     ]>;

// FIXME: Cortex-A35 is currently modeled as a Cortex-A53.
def : ProcessorModel<"cortex-a35", CortexA53Model, [ProcA35]>;
def : ProcessorModel<"cortex-a53", CortexA53Model, [ProcA53]>;
def : ProcessorModel<"cortex-a57", CortexA57Model, [ProcA57]>;
// FIXME: Cortex-A72 and Cortex-A73 are currently modeled as a Cortex-A57.
def : ProcessorModel<"cortex-a72", CortexA57Model, [ProcA72]>;
def : ProcessorModel<"cortex-a73", CortexA57Model, [ProcA73]>;
def : ProcessorModel<"cyclone", CycloneModel, [ProcCyclone]>;
def : ProcessorModel<"exynos-m1", ExynosM1Model, [ProcExynosM1]>;
def : ProcessorModel<"exynos-m2", ExynosM1Model, [ProcExynosM2]>;
def : ProcessorModel<"exynos-m3", ExynosM1Model, [ProcExynosM2]>;
def : ProcessorModel<"falkor", FalkorModel, [ProcFalkor]>;
def : ProcessorModel<"kryo", KryoModel, [ProcKryo]>;
// Cavium ThunderX/ThunderX T8X  Processors
def : ProcessorModel<"thunderx", ThunderXT8XModel,  [ProcThunderX]>;
def : ProcessorModel<"thunderxt88", ThunderXT8XModel,  [ProcThunderXT88]>;
def : ProcessorModel<"thunderxt81", ThunderXT8XModel,  [ProcThunderXT81]>;
def : ProcessorModel<"thunderxt83", ThunderXT8XModel,  [ProcThunderXT83]>;
// Cavium ThunderX2T9X  Processors. Formerly Broadcom Vulcan.
def : ProcessorModel<"thunderx2t99", ThunderX2T99Model, [ProcThunderX2T99]>;

//===----------------------------------------------------------------------===//
// Assembly parser
//===----------------------------------------------------------------------===//

def GenericAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "generic";
  string BreakCharacters = ".";
}

def AppleAsmParserVariant : AsmParserVariant {
  int Variant = 1;
  string Name = "apple-neon";
  string BreakCharacters = ".";
}

//===----------------------------------------------------------------------===//
// Assembly printer
//===----------------------------------------------------------------------===//
// AArch64 Uses the MC printer for asm output, so make sure the TableGen
// AsmWriter bits get associated with the correct class.
def GenericAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def AppleAsmWriter : AsmWriter {
  let AsmWriterClassName = "AppleInstPrinter";
  int PassSubtarget = 1;
  int Variant = 1;
  int isMCAsmWriter = 1;
}

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def AArch64 : Target {
  let InstructionSet = AArch64InstrInfo;
  let AssemblyParserVariants = [GenericAsmParserVariant, AppleAsmParserVariant];
  let AssemblyWriters = [GenericAsmWriter, AppleAsmWriter];
}
