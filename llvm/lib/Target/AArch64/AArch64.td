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
  "Enable cryptographic instructions">;

def FeatureCRC : SubtargetFeature<"crc", "HasCRC", "true",
  "Enable ARMv8 CRC-32 checksum instructions">;

def FeatureRAS : SubtargetFeature<"ras", "HasRAS", "true",
  "Enable ARMv8 Reliability, Availability and Serviceability Extensions">;

def FeaturePerfMon : SubtargetFeature<"perfmon", "HasPerfMon", "true",
  "Enable ARMv8 PMUv3 Performance Monitors extension">;

def FeatureFullFP16 : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
  "Full FP16", [FeatureFPARMv8]>;

def FeatureSPE : SubtargetFeature<"spe", "HasSPE", "true",
  "Enable Statistical Profiling extension">;

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

def FeatureMergeNarrowLd : SubtargetFeature<"merge-narrow-ld",
                                            "MergeNarrowLoads", "true",
                                            "Merge narrow load instructions">;

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

def FeatureAvoidQuadLdStPairs : SubtargetFeature<"no-quad-ldst-pairs",
    "AvoidQuadLdStPairs", "true",
    "Do not form quad load/store pair operations">;

def FeatureAlternateSExtLoadCVTF32Pattern : SubtargetFeature<
    "alternate-sextload-cvt-f32-pattern", "UseAlternateSExtLoadCVTF32Pattern",
    "true", "Use alternative pattern for sextload convert to f32">;

def FeatureMacroOpFusion : SubtargetFeature<
    "macroop-fusion", "HasMacroOpFusion", "true",
    "CPU supports macro op fusion">;

def FeatureDisableLatencySchedHeuristic : SubtargetFeature<
    "disable-latency-sched-heuristic", "DisableLatencySchedHeuristic", "true",
    "Disable latency scheduling heuristic">;

//===----------------------------------------------------------------------===//
// Architectures.
//

def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
  "Support ARM v8.1a instructions", [FeatureCRC]>;

def HasV8_2aOps : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
  "Support ARM v8.2a instructions", [HasV8_1aOps, FeatureRAS]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "AArch64RegisterInfo.td"
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
include "AArch64SchedM1.td"
include "AArch64SchedKryo.td"
include "AArch64SchedVulcan.td"

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
                                   FeatureMergeNarrowLd,
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
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureNEON,
                                   FeaturePerfMon
                                   ]>;

def ProcCyclone : SubtargetFeature<"cyclone", "ARMProcFamily", "Cyclone",
                                   "Cyclone", [
                                   FeatureAlternateSExtLoadCVTF32Pattern,
                                   FeatureCrypto,
                                   FeatureDisableLatencySchedHeuristic,
                                   FeatureFPARMv8,
                                   FeatureMacroOpFusion,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeatureSlowMisaligned128Store,
                                   FeatureZCRegMove,
                                   FeatureZCZeroing
                                   ]>;

def ProcExynosM1 : SubtargetFeature<"exynosm1", "ARMProcFamily", "ExynosM1",
                                    "Samsung Exynos-M1 processors", [
                                    FeatureAvoidQuadLdStPairs,
                                    FeatureCRC,
                                    FeatureCrypto,
                                    FeatureCustomCheapAsMoveHandling,
                                    FeatureFPARMv8,
                                    FeatureNEON,
                                    FeaturePerfMon,
                                    FeaturePostRAScheduler,
                                    FeatureZCZeroing
                                    ]>;

def ProcKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureCustomCheapAsMoveHandling,
                                   FeatureFPARMv8,
                                   FeatureMergeNarrowLd,
                                   FeatureNEON,
                                   FeaturePerfMon,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   FeatureZCZeroing
                                   ]>;

def ProcVulcan  : SubtargetFeature<"vulcan", "ARMProcFamily", "Vulcan",
                                   "Broadcom Vulcan processors", [
                                   FeatureCRC,
                                   FeatureCrypto,
                                   FeatureFPARMv8,
                                   FeatureMacroOpFusion,
                                   FeatureNEON,
                                   FeaturePostRAScheduler,
                                   FeaturePredictableSelectIsExpensive,
                                   HasV8_1aOps]>;

def : ProcessorModel<"generic", NoSchedModel, [
                     FeatureCRC,
                     FeatureFPARMv8,
                     FeatureNEON,
                     FeaturePerfMon,
                     FeaturePostRAScheduler
                     ]>;

// FIXME: Cortex-A35 is currently modelled as a Cortex-A53
def : ProcessorModel<"cortex-a35", CortexA53Model, [ProcA35]>;
def : ProcessorModel<"cortex-a53", CortexA53Model, [ProcA53]>;
def : ProcessorModel<"cortex-a57", CortexA57Model, [ProcA57]>;
// FIXME: Cortex-A72 and Cortex-A73 are currently modelled as an Cortex-A57.
def : ProcessorModel<"cortex-a72", CortexA57Model, [ProcA72]>;
def : ProcessorModel<"cortex-a73", CortexA57Model, [ProcA73]>;
def : ProcessorModel<"cyclone", CycloneModel, [ProcCyclone]>;
def : ProcessorModel<"exynos-m1", ExynosM1Model, [ProcExynosM1]>;
def : ProcessorModel<"exynos-m2", ExynosM1Model, [ProcExynosM1]>;
def : ProcessorModel<"kryo", KryoModel, [ProcKryo]>;
def : ProcessorModel<"vulcan", VulcanModel, [ProcVulcan]>;

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
