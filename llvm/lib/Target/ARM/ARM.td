//===-- ARM.td - Describe the ARM Target Machine -----------*- tablegen -*-===//
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
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// ARM Helper classes.
//

class ProcNoItin<string Name, list<SubtargetFeature> Features>
 : Processor<Name, NoItineraries, Features>;

class Architecture<string fname, string aname, list<SubtargetFeature> features >
  : SubtargetFeature<fname, "ARMArch", aname,
                     !strconcat(aname, " architecture"), features>;

//===----------------------------------------------------------------------===//
// ARM Subtarget state.
//

def ModeThumb  : SubtargetFeature<"thumb-mode", "InThumbMode", "true",
                                  "Thumb mode">;

def ModeSoftFloat : SubtargetFeature<"soft-float", "UseSoftFloat", "true",
                                     "Use software floating point features.">;

//===----------------------------------------------------------------------===//
// ARM Subtarget features.
//

def FeatureVFP2 : SubtargetFeature<"vfp2", "HasVFPv2", "true",
                                   "Enable VFP2 instructions">;
def FeatureVFP3 : SubtargetFeature<"vfp3", "HasVFPv3", "true",
                                   "Enable VFP3 instructions",
                                   [FeatureVFP2]>;
def FeatureNEON : SubtargetFeature<"neon", "HasNEON", "true",
                                   "Enable NEON instructions",
                                   [FeatureVFP3]>;
def FeatureThumb2 : SubtargetFeature<"thumb2", "HasThumb2", "true",
                                     "Enable Thumb2 instructions">;
def FeatureNoARM  : SubtargetFeature<"noarm", "NoARM", "true",
                                     "Does not support ARM mode execution",
                                     [ModeThumb]>;
def FeatureFP16   : SubtargetFeature<"fp16", "HasFP16", "true",
                                     "Enable half-precision floating point">;
def FeatureVFP4   : SubtargetFeature<"vfp4", "HasVFPv4", "true",
                                     "Enable VFP4 instructions",
                                     [FeatureVFP3, FeatureFP16]>;
def FeatureFPARMv8 : SubtargetFeature<"fp-armv8", "HasFPARMv8",
                                   "true", "Enable ARMv8 FP",
                                   [FeatureVFP4]>;
def FeatureFullFP16 : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
                                       "Enable full half-precision floating point",
                                       [FeatureFPARMv8]>;
def FeatureD16    : SubtargetFeature<"d16", "HasD16", "true",
                                     "Restrict FP to 16 double registers">;
def FeatureHWDiv  : SubtargetFeature<"hwdiv", "HasHardwareDivide", "true",
                                     "Enable divide instructions">;
def FeatureHWDivARM  : SubtargetFeature<"hwdiv-arm",
                                        "HasHardwareDivideInARM", "true",
                                      "Enable divide instructions in ARM mode">;
def FeatureT2XtPk : SubtargetFeature<"t2xtpk", "HasT2ExtractPack", "true",
                                 "Enable Thumb2 extract and pack instructions">;
def FeatureDB     : SubtargetFeature<"db", "HasDataBarrier", "true",
                                   "Has data barrier (dmb / dsb) instructions">;
def FeatureV7Clrex : SubtargetFeature<"v7clrex", "HasV7Clrex", "true",
                                      "Has v7 clrex instruction">;
def FeatureAcquireRelease : SubtargetFeature<"acquire-release",
                                             "HasAcquireRelease", "true",
                         "Has v8 acquire/release (lda/ldaex etc) instructions">;
def FeatureSlowFPBrcc : SubtargetFeature<"slow-fp-brcc", "SlowFPBrcc", "true",
                                         "FP compare + branch is slow">;
def FeatureVFPOnlySP : SubtargetFeature<"fp-only-sp", "FPOnlySP", "true",
                          "Floating point unit supports single precision only">;
def FeaturePerfMon : SubtargetFeature<"perfmon", "HasPerfMon", "true",
                           "Enable support for Performance Monitor extensions">;
def FeatureTrustZone : SubtargetFeature<"trustzone", "HasTrustZone", "true",
                          "Enable support for TrustZone security extensions">;
def FeatureCrypto : SubtargetFeature<"crypto", "HasCrypto", "true",
                          "Enable support for Cryptography extensions",
                          [FeatureNEON]>;
def FeatureCRC : SubtargetFeature<"crc", "HasCRC", "true",
                          "Enable support for CRC instructions">;

// Cyclone has preferred instructions for zeroing VFP registers, which can
// execute in 0 cycles.
def FeatureZCZeroing : SubtargetFeature<"zcz", "HasZeroCycleZeroing", "true",
                                        "Has zero-cycle zeroing instructions">;

// Some processors have FP multiply-accumulate instructions that don't
// play nicely with other VFP / NEON instructions, and it's generally better
// to just not use them.
def FeatureHasSlowFPVMLx : SubtargetFeature<"slowfpvmlx", "SlowFPVMLx", "true",
                                         "Disable VFP / NEON MAC instructions">;

// Cortex-A8 / A9 Advanced SIMD has multiplier accumulator forwarding.
def FeatureVMLxForwarding : SubtargetFeature<"vmlx-forwarding",
                                       "HasVMLxForwarding", "true",
                                       "Has multiplier accumulator forwarding">;

// Some processors benefit from using NEON instructions for scalar
// single-precision FP operations.
def FeatureNEONForFP : SubtargetFeature<"neonfp", "UseNEONForSinglePrecisionFP",
                                        "true",
                                        "Use NEON for single precision FP">;

// Disable 32-bit to 16-bit narrowing for experimentation.
def FeaturePref32BitThumb : SubtargetFeature<"32bit", "Pref32BitThumb", "true",
                                             "Prefer 32-bit Thumb instrs">;

/// Some instructions update CPSR partially, which can add false dependency for
/// out-of-order implementation, e.g. Cortex-A9, unless each individual bit is
/// mapped to a separate physical register. Avoid partial CPSR update for these
/// processors.
def FeatureAvoidPartialCPSR : SubtargetFeature<"avoid-partial-cpsr",
                                               "AvoidCPSRPartialUpdate", "true",
                                 "Avoid CPSR partial update for OOO execution">;

def FeatureAvoidMOVsShOp : SubtargetFeature<"avoid-movs-shop",
                                            "AvoidMOVsShifterOperand", "true",
                                "Avoid movs instructions with shifter operand">;

// Some processors perform return stack prediction. CodeGen should avoid issue
// "normal" call instructions to callees which do not return.
def FeatureHasRAS : SubtargetFeature<"ras", "HasRAS", "true",
                                     "Has return address stack">;

/// DSP extension.
def FeatureDSP : SubtargetFeature<"dsp", "HasDSP", "true",
                              "Supports DSP instructions in ARM and/or Thumb2">;

// Multiprocessing extension.
def FeatureMP : SubtargetFeature<"mp", "HasMPExtension", "true",
                                 "Supports Multiprocessing extension">;

// Virtualization extension - requires HW divide (ARMv7-AR ARMARM - 4.4.8).
def FeatureVirtualization : SubtargetFeature<"virtualization",
                                 "HasVirtualization", "true",
                                 "Supports Virtualization extension",
                                 [FeatureHWDiv, FeatureHWDivARM]>;

// M-series ISA
def FeatureMClass : SubtargetFeature<"mclass", "ARMProcClass", "MClass",
                                     "Is microcontroller profile ('M' series)">;

// R-series ISA
def FeatureRClass : SubtargetFeature<"rclass", "ARMProcClass", "RClass",
                                     "Is realtime profile ('R' series)">;

// A-series ISA
def FeatureAClass : SubtargetFeature<"aclass", "ARMProcClass", "AClass",
                                     "Is application profile ('A' series)">;

// Special TRAP encoding for NaCl, which looks like a TRAP in Thumb too.
// See ARMInstrInfo.td for details.
def FeatureNaClTrap : SubtargetFeature<"nacl-trap", "UseNaClTrap", "true",
                                       "NaCl trap">;

def FeatureStrictAlign : SubtargetFeature<"strict-align",
                                          "StrictAlign", "true",
                                          "Disallow all unaligned memory "
                                          "access">;

def FeatureLongCalls : SubtargetFeature<"long-calls", "GenLongCalls", "true",
                                        "Generate calls via indirect call "
                                        "instructions">;

def FeatureReserveR9 : SubtargetFeature<"reserve-r9", "ReserveR9", "true",
                                        "Reserve R9, making it unavailable as "
                                        "GPR">;

def FeatureNoMovt : SubtargetFeature<"no-movt", "NoMovt", "true",
                                     "Don't use movt/movw pairs for 32-bit "
                                     "imms">;


//===----------------------------------------------------------------------===//
// ARM ISAa.
//

def HasV4TOps   : SubtargetFeature<"v4t", "HasV4TOps", "true",
                                   "Support ARM v4T instructions">;
def HasV5TOps   : SubtargetFeature<"v5t", "HasV5TOps", "true",
                                   "Support ARM v5T instructions",
                                   [HasV4TOps]>;
def HasV5TEOps  : SubtargetFeature<"v5te", "HasV5TEOps", "true",
                             "Support ARM v5TE, v5TEj, and v5TExp instructions",
                                   [HasV5TOps]>;
def HasV6Ops    : SubtargetFeature<"v6", "HasV6Ops", "true",
                                   "Support ARM v6 instructions",
                                   [HasV5TEOps]>;
def HasV6MOps   : SubtargetFeature<"v6m", "HasV6MOps", "true",
                                   "Support ARM v6M instructions",
                                   [HasV6Ops]>;
def HasV8MBaselineOps : SubtargetFeature<"v8m", "HasV8MBaselineOps", "true",
                                         "Support ARM v8M Baseline instructions",
                                         [HasV6MOps]>;
def HasV6KOps   : SubtargetFeature<"v6k", "HasV6KOps", "true",
                                   "Support ARM v6k instructions",
                                   [HasV6Ops]>;
def HasV6T2Ops  : SubtargetFeature<"v6t2", "HasV6T2Ops", "true",
                                   "Support ARM v6t2 instructions",
                                   [HasV8MBaselineOps, HasV6KOps, FeatureThumb2]>;
def HasV7Ops    : SubtargetFeature<"v7", "HasV7Ops", "true",
                                   "Support ARM v7 instructions",
                                   [HasV6T2Ops, FeaturePerfMon,
                                    FeatureV7Clrex]>;
def HasV8Ops    : SubtargetFeature<"v8", "HasV8Ops", "true",
                                   "Support ARM v8 instructions",
                                   [HasV7Ops, FeatureAcquireRelease]>;
def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
                                   "Support ARM v8.1a instructions",
                                   [HasV8Ops]>;
def HasV8_2aOps   : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
                                   "Support ARM v8.2a instructions",
                                   [HasV8_1aOps]>;
def HasV8MMainlineOps : SubtargetFeature<"v8m.main", "HasV8MMainlineOps", "true",
                                         "Support ARM v8M Mainline instructions",
                                         [HasV7Ops]>;


//===----------------------------------------------------------------------===//
// ARM Processor subtarget features.
//

def ProcA5      : SubtargetFeature<"a5", "ARMProcFamily", "CortexA5",
                                   "Cortex-A5 ARM processors", []>;
def ProcA7      : SubtargetFeature<"a7", "ARMProcFamily", "CortexA7",
                                   "Cortex-A7 ARM processors", []>;
def ProcA8      : SubtargetFeature<"a8", "ARMProcFamily", "CortexA8",
                                   "Cortex-A8 ARM processors", []>;
def ProcA9      : SubtargetFeature<"a9", "ARMProcFamily", "CortexA9",
                                   "Cortex-A9 ARM processors", []>;
def ProcA12     : SubtargetFeature<"a12", "ARMProcFamily", "CortexA12",
                                   "Cortex-A12 ARM processors", []>;
def ProcA15     : SubtargetFeature<"a15", "ARMProcFamily", "CortexA15",
                                   "Cortex-A15 ARM processors", []>;
def ProcA17     : SubtargetFeature<"a17", "ARMProcFamily", "CortexA17",
                                   "Cortex-A17 ARM processors", []>;
def ProcA35     : SubtargetFeature<"a35", "ARMProcFamily", "CortexA35",
                                   "Cortex-A35 ARM processors", []>;
def ProcA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors", []>;
def ProcA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors", []>;
def ProcA72     : SubtargetFeature<"a72", "ARMProcFamily", "CortexA72",
                                   "Cortex-A72 ARM processors", []>;

def ProcKrait   : SubtargetFeature<"krait", "ARMProcFamily", "Krait",
                                   "Qualcomm ARM processors", []>;
def ProcSwift   : SubtargetFeature<"swift", "ARMProcFamily", "Swift",
                                   "Swift ARM processors", []>;

def ProcExynosM1 : SubtargetFeature<"exynosm1", "ARMProcFamily", "ExynosM1",
                                    "Samsung Exynos-M1 processors", []>;

def ProcR4      : SubtargetFeature<"r4", "ARMProcFamily", "CortexR4",
                                   "Cortex-R4 ARM processors", []>;
def ProcR5      : SubtargetFeature<"r5", "ARMProcFamily", "CortexR5",
                                   "Cortex-R5 ARM processors", []>;
def ProcR7      : SubtargetFeature<"r7", "ARMProcFamily", "CortexR7",
                                   "Cortex-R7 ARM processors", []>;


//===----------------------------------------------------------------------===//
// ARM schedules.
//

include "ARMSchedule.td"


//===----------------------------------------------------------------------===//
// ARM architectures
//

def ARMv2     : Architecture<"armv2",     "ARMv2",    []>;

def ARMv2a    : Architecture<"armv2a",    "ARMv2a",   []>;

def ARMv3     : Architecture<"armv3",     "ARMv3",    []>;

def ARMv3m    : Architecture<"armv3m",    "ARMv3m",   []>;

def ARMv4     : Architecture<"armv4",     "ARMv4",    []>;

def ARMv4t    : Architecture<"armv4t",    "ARMv4t",   [HasV4TOps]>;

def ARMv5t    : Architecture<"armv5t",    "ARMv5t",   [HasV5TOps]>;

def ARMv5te   : Architecture<"armv5te",   "ARMv5te",  [HasV5TEOps]>;

def ARMv5tej  : Architecture<"armv5tej",  "ARMv5tej", [HasV5TEOps]>;

def ARMv6     : Architecture<"armv6",     "ARMv6",    [HasV6Ops]>;

def ARMv6t2   : Architecture<"armv6t2",   "ARMv6t2",  [HasV6T2Ops,
                                                       FeatureDSP]>;

def ARMv6k    : Architecture<"armv6k",    "ARMv6k",   [HasV6KOps]>;

def ARMv6kz   : Architecture<"armv6kz",   "ARMv6kz",  [HasV6KOps,
                                                       FeatureTrustZone]>;

def ARMv6m    : Architecture<"armv6-m",   "ARMv6m",   [HasV6MOps,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureMClass]>;

def ARMv6sm   : Architecture<"armv6s-m",  "ARMv6sm",  [HasV6MOps,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureMClass]>;

def ARMv7a    : Architecture<"armv7-a",   "ARMv7a",   [HasV7Ops,
                                                       FeatureNEON,
                                                       FeatureDB,
                                                       FeatureDSP,
                                                       FeatureAClass]>;

def ARMv7r    : Architecture<"armv7-r",   "ARMv7r",   [HasV7Ops,
                                                       FeatureDB,
                                                       FeatureDSP,
                                                       FeatureHWDiv,
                                                       FeatureRClass]>;

def ARMv7m    : Architecture<"armv7-m",   "ARMv7m",   [HasV7Ops,
                                                       FeatureThumb2,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureHWDiv,
                                                       FeatureMClass]>;

def ARMv7em   : Architecture<"armv7e-m",  "ARMv7em",  [HasV7Ops,
                                                       FeatureThumb2,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureHWDiv,
                                                       FeatureMClass,
                                                       FeatureDSP,
                                                       FeatureT2XtPk]>;

def ARMv8a    : Architecture<"armv8-a",   "ARMv8a",   [HasV8Ops,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC]>;

def ARMv81a   : Architecture<"armv8.1-a", "ARMv81a",  [HasV8_1aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC]>;

def ARMv82a   : Architecture<"armv8.2-a", "ARMv82a",  [HasV8_2aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC]>;

def ARMv8mBaseline : Architecture<"armv8-m.base", "ARMv8mBaseline",
                                                      [HasV8MBaselineOps,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureHWDiv,
                                                       FeatureV7Clrex,
                                                       FeatureAcquireRelease,
                                                       FeatureMClass]>;

def ARMv8mMainline : Architecture<"armv8-m.main", "ARMv8mMainline",
                                                      [HasV8MMainlineOps,
                                                       FeatureNoARM,
                                                       FeatureDB,
                                                       FeatureHWDiv,
                                                       FeatureAcquireRelease,
                                                       FeatureMClass]>;

// Aliases
def IWMMXT   : Architecture<"iwmmxt",      "ARMv5te",  [ARMv5te]>;
def IWMMXT2  : Architecture<"iwmmxt2",     "ARMv5te",  [ARMv5te]>;
def XScale   : Architecture<"xscale",      "ARMv5te",  [ARMv5te]>;
def ARMv6j   : Architecture<"armv6j",      "ARMv7a",   [ARMv6]>;
def ARMv7k   : Architecture<"armv7k",      "ARMv7a",   [ARMv7a]>;
def ARMv7s   : Architecture<"armv7s",      "ARMv7a",   [ARMv7a]>;


//===----------------------------------------------------------------------===//
// ARM processors
//

// Dummy CPU, used to target architectures
def : ProcNoItin<"generic",                             []>;

def : ProcNoItin<"arm8",                                [ARMv4]>;
def : ProcNoItin<"arm810",                              [ARMv4]>;
def : ProcNoItin<"strongarm",                           [ARMv4]>;
def : ProcNoItin<"strongarm110",                        [ARMv4]>;
def : ProcNoItin<"strongarm1100",                       [ARMv4]>;
def : ProcNoItin<"strongarm1110",                       [ARMv4]>;

def : ProcNoItin<"arm7tdmi",                            [ARMv4t]>;
def : ProcNoItin<"arm7tdmi-s",                          [ARMv4t]>;
def : ProcNoItin<"arm710t",                             [ARMv4t]>;
def : ProcNoItin<"arm720t",                             [ARMv4t]>;
def : ProcNoItin<"arm9",                                [ARMv4t]>;
def : ProcNoItin<"arm9tdmi",                            [ARMv4t]>;
def : ProcNoItin<"arm920",                              [ARMv4t]>;
def : ProcNoItin<"arm920t",                             [ARMv4t]>;
def : ProcNoItin<"arm922t",                             [ARMv4t]>;
def : ProcNoItin<"arm940t",                             [ARMv4t]>;
def : ProcNoItin<"ep9312",                              [ARMv4t]>;

def : ProcNoItin<"arm10tdmi",                           [ARMv5t]>;
def : ProcNoItin<"arm1020t",                            [ARMv5t]>;

def : ProcNoItin<"arm9e",                               [ARMv5te]>;
def : ProcNoItin<"arm926ej-s",                          [ARMv5te]>;
def : ProcNoItin<"arm946e-s",                           [ARMv5te]>;
def : ProcNoItin<"arm966e-s",                           [ARMv5te]>;
def : ProcNoItin<"arm968e-s",                           [ARMv5te]>;
def : ProcNoItin<"arm10e",                              [ARMv5te]>;
def : ProcNoItin<"arm1020e",                            [ARMv5te]>;
def : ProcNoItin<"arm1022e",                            [ARMv5te]>;
def : ProcNoItin<"xscale",                              [ARMv5te]>;
def : ProcNoItin<"iwmmxt",                              [ARMv5te]>;

def : Processor<"arm1136j-s",       ARMV6Itineraries,   [ARMv6]>;
def : Processor<"arm1136jf-s",      ARMV6Itineraries,   [ARMv6,
                                                         FeatureVFP2,
                                                         FeatureHasSlowFPVMLx]>;

def : Processor<"cortex-m0",        ARMV6Itineraries,   [ARMv6m]>;
def : Processor<"cortex-m0plus",    ARMV6Itineraries,   [ARMv6m]>;
def : Processor<"cortex-m1",        ARMV6Itineraries,   [ARMv6m]>;
def : Processor<"sc000",            ARMV6Itineraries,   [ARMv6m]>;

def : Processor<"arm1176jz-s",      ARMV6Itineraries,   [ARMv6kz]>;
def : Processor<"arm1176jzf-s",     ARMV6Itineraries,   [ARMv6kz,
                                                         FeatureVFP2,
                                                         FeatureHasSlowFPVMLx]>;

def : Processor<"mpcorenovfp",      ARMV6Itineraries,   [ARMv6k]>;
def : Processor<"mpcore",           ARMV6Itineraries,   [ARMv6k,
                                                         FeatureVFP2,
                                                         FeatureHasSlowFPVMLx]>;

def : Processor<"arm1156t2-s",      ARMV6Itineraries,   [ARMv6t2]>;
def : Processor<"arm1156t2f-s",     ARMV6Itineraries,   [ARMv6t2,
                                                         FeatureVFP2,
                                                         FeatureHasSlowFPVMLx]>;

// FIXME: A5 has currently the same Schedule model as A8
def : ProcessorModel<"cortex-a5",   CortexA8Model,      [ARMv7a, ProcA5,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureMP,
                                                         FeatureVFP4]>;

def : ProcessorModel<"cortex-a7",   CortexA8Model,      [ARMv7a, ProcA7,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureMP,
                                                         FeatureVFP4,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureVirtualization]>;

def : ProcessorModel<"cortex-a8",   CortexA8Model,      [ARMv7a, ProcA8,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk]>;

def : ProcessorModel<"cortex-a9",   CortexA9Model,      [ARMv7a, ProcA9,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureFP16,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureMP]>;

// FIXME: A12 has currently the same Schedule model as A9
def : ProcessorModel<"cortex-a12",  CortexA9Model,      [ARMv7a, ProcA12,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureVFP4,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization,
                                                         FeatureMP]>;

// FIXME: A15 has currently the same Schedule model as A9.
def : ProcessorModel<"cortex-a15",  CortexA9Model,      [ARMv7a, ProcA15,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureT2XtPk,
                                                         FeatureVFP4,
                                                         FeatureMP,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization]>;

// FIXME: A17 has currently the same Schedule model as A9
def : ProcessorModel<"cortex-a17",  CortexA9Model,      [ARMv7a, ProcA17,
                                                         FeatureHasRAS,
                                                         FeatureTrustZone,
                                                         FeatureMP,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureVFP4,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization]>;

// FIXME: krait has currently the same Schedule model as A9
// FIXME: krait has currently the same features as A9 plus VFP4 and hardware
//        division features.
def : ProcessorModel<"krait",       CortexA9Model,      [ARMv7a, ProcKrait,
                                                         FeatureHasRAS,
                                                         FeatureVMLxForwarding,
                                                         FeatureT2XtPk,
                                                         FeatureFP16,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVFP4,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM]>;

def : ProcessorModel<"swift",       SwiftModel,         [ARMv7a, ProcSwift,
                                                         FeatureHasRAS,
                                                         FeatureNEONForFP,
                                                         FeatureT2XtPk,
                                                         FeatureVFP4,
                                                         FeatureMP,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureAvoidMOVsShOp,
                                                         FeatureHasSlowFPVMLx]>;

// FIXME: R4 has currently the same ProcessorModel as A8.
def : ProcessorModel<"cortex-r4",   CortexA8Model,      [ARMv7r, ProcR4,
                                                         FeatureHasRAS,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureT2XtPk]>;

// FIXME: R4F has currently the same ProcessorModel as A8.
def : ProcessorModel<"cortex-r4f",  CortexA8Model,      [ARMv7r, ProcR4,
                                                         FeatureHasRAS,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureVFP3,
                                                         FeatureD16,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureT2XtPk]>;

// FIXME: R5 has currently the same ProcessorModel as A8.
def : ProcessorModel<"cortex-r5",   CortexA8Model,      [ARMv7r, ProcR5,
                                                         FeatureHasRAS,
                                                         FeatureVFP3,
                                                         FeatureD16,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHWDivARM,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureT2XtPk]>;

// FIXME: R7 has currently the same ProcessorModel as A8 and is modelled as R5.
def : ProcessorModel<"cortex-r7",   CortexA8Model,      [ARMv7r, ProcR7,
                                                         FeatureHasRAS,
                                                         FeatureVFP3,
                                                         FeatureVFPOnlySP,
                                                         FeatureD16,
                                                         FeatureFP16,
                                                         FeatureMP,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHWDivARM,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureT2XtPk]>;

def : ProcNoItin<"cortex-m3",                           [ARMv7m]>;
def : ProcNoItin<"sc300",                               [ARMv7m]>;

def : ProcNoItin<"cortex-m4",                           [ARMv7em,
                                                         FeatureVFP4,
                                                         FeatureVFPOnlySP,
                                                         FeatureD16]>;

def : ProcNoItin<"cortex-m7",                           [ARMv7em,
                                                         FeatureFPARMv8,
                                                         FeatureD16]>;


def : ProcNoItin<"cortex-a35",                          [ARMv8a, ProcA35,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureT2XtPk,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a53",                          [ARMv8a, ProcA53,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureT2XtPk,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a57",                          [ARMv8a, ProcA57,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureT2XtPk,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a72",                          [ARMv8a, ProcA72,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureT2XtPk,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

// Cyclone is very similar to swift
def : ProcessorModel<"cyclone",     SwiftModel,         [ARMv8a, ProcSwift,
                                                         FeatureHasRAS,
                                                         FeatureNEONForFP,
                                                         FeatureT2XtPk,
                                                         FeatureVFP4,
                                                         FeatureMP,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureAvoidMOVsShOp,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureCrypto,
                                                         FeatureZCZeroing]>;

def : ProcNoItin<"exynos-m1",                           [ARMv8a, ProcExynosM1,
                                                         FeatureHWDiv,
                                                         FeatureHWDivARM,
                                                         FeatureT2XtPk,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "ARMRegisterInfo.td"

include "ARMCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "ARMInstrInfo.td"

def ARMInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def ARMAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def ARMAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "ARM";
  string BreakCharacters = ".";
}

def ARM : Target {
  // Pull in Instruction Info:
  let InstructionSet = ARMInstrInfo;
  let AssemblyWriters = [ARMAsmWriter];
  let AssemblyParserVariants = [ARMAsmParserVariant];
}
