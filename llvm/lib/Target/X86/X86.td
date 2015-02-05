//===-- X86.td - Target definition file for the Intel X86 --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a target description file for the Intel i386 architecture, referred
// to here as the "X86" architecture.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing...
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// X86 Subtarget state
//

def Mode64Bit : SubtargetFeature<"64bit-mode", "In64BitMode", "true",
                                  "64-bit mode (x86_64)">;
def Mode32Bit : SubtargetFeature<"32bit-mode", "In32BitMode", "true",
                                  "32-bit mode (80386)">;
def Mode16Bit : SubtargetFeature<"16bit-mode", "In16BitMode", "true",
                                  "16-bit mode (i8086)">;

//===----------------------------------------------------------------------===//
// X86 Subtarget features
//===----------------------------------------------------------------------===//

def FeatureCMOV    : SubtargetFeature<"cmov","HasCMov", "true",
                                      "Enable conditional move instructions">;

def FeaturePOPCNT   : SubtargetFeature<"popcnt", "HasPOPCNT", "true",
                                       "Support POPCNT instruction">;


def FeatureMMX     : SubtargetFeature<"mmx","X86SSELevel", "MMX",
                                      "Enable MMX instructions">;
def FeatureSSE1    : SubtargetFeature<"sse", "X86SSELevel", "SSE1",
                                      "Enable SSE instructions",
                                      // SSE codegen depends on cmovs, and all
                                      // SSE1+ processors support them.
                                      [FeatureMMX, FeatureCMOV]>;
def FeatureSSE2    : SubtargetFeature<"sse2", "X86SSELevel", "SSE2",
                                      "Enable SSE2 instructions",
                                      [FeatureSSE1]>;
def FeatureSSE3    : SubtargetFeature<"sse3", "X86SSELevel", "SSE3",
                                      "Enable SSE3 instructions",
                                      [FeatureSSE2]>;
def FeatureSSSE3   : SubtargetFeature<"ssse3", "X86SSELevel", "SSSE3",
                                      "Enable SSSE3 instructions",
                                      [FeatureSSE3]>;
def FeatureSSE41   : SubtargetFeature<"sse4.1", "X86SSELevel", "SSE41",
                                      "Enable SSE 4.1 instructions",
                                      [FeatureSSSE3]>;
def FeatureSSE42   : SubtargetFeature<"sse4.2", "X86SSELevel", "SSE42",
                                      "Enable SSE 4.2 instructions",
                                      [FeatureSSE41]>;
def Feature3DNow   : SubtargetFeature<"3dnow", "X863DNowLevel", "ThreeDNow",
                                      "Enable 3DNow! instructions",
                                      [FeatureMMX]>;
def Feature3DNowA  : SubtargetFeature<"3dnowa", "X863DNowLevel", "ThreeDNowA",
                                      "Enable 3DNow! Athlon instructions",
                                      [Feature3DNow]>;
// All x86-64 hardware has SSE2, but we don't mark SSE2 as an implied
// feature, because SSE2 can be disabled (e.g. for compiling OS kernels)
// without disabling 64-bit mode.
def Feature64Bit   : SubtargetFeature<"64bit", "HasX86_64", "true",
                                      "Support 64-bit instructions",
                                      [FeatureCMOV]>;
def FeatureCMPXCHG16B : SubtargetFeature<"cx16", "HasCmpxchg16b", "true",
                                      "64-bit with cmpxchg16b",
                                      [Feature64Bit]>;
def FeatureSlowBTMem : SubtargetFeature<"slow-bt-mem", "IsBTMemSlow", "true",
                                       "Bit testing of memory is slow">;
def FeatureSlowSHLD : SubtargetFeature<"slow-shld", "IsSHLDSlow", "true",
                                       "SHLD instruction is slow">;
// FIXME: This is a 16-byte (SSE/AVX) feature; we should rename it to make that
// explicit. Also, it seems this would be the default state for most chips
// going forward, so it would probably be better to negate the logic and
// match the 32-byte "slow mem" feature below.
def FeatureFastUAMem : SubtargetFeature<"fast-unaligned-mem",
                                        "IsUAMemFast", "true",
                                        "Fast unaligned memory access">;
def FeatureSlowUAMem32 : SubtargetFeature<"slow-unaligned-mem-32",
                            "IsUAMem32Slow", "true",
                            "Slow unaligned 32-byte memory access">;
def FeatureSSE4A   : SubtargetFeature<"sse4a", "HasSSE4A", "true",
                                      "Support SSE 4a instructions",
                                      [FeatureSSE3]>;

def FeatureAVX     : SubtargetFeature<"avx", "X86SSELevel", "AVX",
                                      "Enable AVX instructions",
                                      [FeatureSSE42]>;
def FeatureAVX2    : SubtargetFeature<"avx2", "X86SSELevel", "AVX2",
                                      "Enable AVX2 instructions",
                                      [FeatureAVX]>;
def FeatureAVX512   : SubtargetFeature<"avx512f", "X86SSELevel", "AVX512F",
                                      "Enable AVX-512 instructions",
                                      [FeatureAVX2]>;
def FeatureERI      : SubtargetFeature<"avx512er", "HasERI", "true",
                      "Enable AVX-512 Exponential and Reciprocal Instructions",
                                      [FeatureAVX512]>;
def FeatureCDI      : SubtargetFeature<"avx512cd", "HasCDI", "true",
                      "Enable AVX-512 Conflict Detection Instructions",
                                      [FeatureAVX512]>;
def FeaturePFI      : SubtargetFeature<"avx512pf", "HasPFI", "true",
                      "Enable AVX-512 PreFetch Instructions",
                                      [FeatureAVX512]>;
def FeatureDQI     : SubtargetFeature<"avx512dq", "HasDQI", "true",
                      "Enable AVX-512 Doubleword and Quadword Instructions",
                                      [FeatureAVX512]>;
def FeatureBWI     : SubtargetFeature<"avx512bw", "HasBWI", "true",
                      "Enable AVX-512 Byte and Word Instructions",
                                      [FeatureAVX512]>;
def FeatureVLX     : SubtargetFeature<"avx512vl", "HasVLX", "true",
                      "Enable AVX-512 Vector Length eXtensions",
                                      [FeatureAVX512]>;
def FeaturePCLMUL  : SubtargetFeature<"pclmul", "HasPCLMUL", "true",
                         "Enable packed carry-less multiplication instructions",
                               [FeatureSSE2]>;
def FeatureFMA     : SubtargetFeature<"fma", "HasFMA", "true",
                                      "Enable three-operand fused multiple-add",
                                      [FeatureAVX]>;
def FeatureFMA4    : SubtargetFeature<"fma4", "HasFMA4", "true",
                                      "Enable four-operand fused multiple-add",
                                      [FeatureAVX, FeatureSSE4A]>;
def FeatureXOP     : SubtargetFeature<"xop", "HasXOP", "true",
                                      "Enable XOP instructions",
                                      [FeatureFMA4]>;
def FeatureSSEUnalignedMem : SubtargetFeature<"sse-unaligned-mem",
                                          "HasSSEUnalignedMem", "true",
                      "Allow unaligned memory operands with SSE instructions">;
def FeatureAES     : SubtargetFeature<"aes", "HasAES", "true",
                                      "Enable AES instructions",
                                      [FeatureSSE2]>;
def FeatureTBM     : SubtargetFeature<"tbm", "HasTBM", "true",
                                      "Enable TBM instructions">;
def FeatureMOVBE   : SubtargetFeature<"movbe", "HasMOVBE", "true",
                                      "Support MOVBE instruction">;
def FeatureRDRAND  : SubtargetFeature<"rdrnd", "HasRDRAND", "true",
                                      "Support RDRAND instruction">;
def FeatureF16C    : SubtargetFeature<"f16c", "HasF16C", "true",
                       "Support 16-bit floating point conversion instructions",
                       [FeatureAVX]>;
def FeatureFSGSBase : SubtargetFeature<"fsgsbase", "HasFSGSBase", "true",
                                       "Support FS/GS Base instructions">;
def FeatureLZCNT   : SubtargetFeature<"lzcnt", "HasLZCNT", "true",
                                      "Support LZCNT instruction">;
def FeatureBMI     : SubtargetFeature<"bmi", "HasBMI", "true",
                                      "Support BMI instructions">;
def FeatureBMI2    : SubtargetFeature<"bmi2", "HasBMI2", "true",
                                      "Support BMI2 instructions">;
def FeatureRTM     : SubtargetFeature<"rtm", "HasRTM", "true",
                                      "Support RTM instructions">;
def FeatureHLE     : SubtargetFeature<"hle", "HasHLE", "true",
                                      "Support HLE">;
def FeatureADX     : SubtargetFeature<"adx", "HasADX", "true",
                                      "Support ADX instructions">;
def FeatureSHA     : SubtargetFeature<"sha", "HasSHA", "true",
                                      "Enable SHA instructions",
                                      [FeatureSSE2]>;
def FeatureSGX     : SubtargetFeature<"sgx", "HasSGX", "true",
                                      "Support SGX instructions">;
def FeaturePRFCHW  : SubtargetFeature<"prfchw", "HasPRFCHW", "true",
                                      "Support PRFCHW instructions">;
def FeatureRDSEED  : SubtargetFeature<"rdseed", "HasRDSEED", "true",
                                      "Support RDSEED instruction">;
def FeatureSMAP    : SubtargetFeature<"smap", "HasSMAP", "true",
                                      "Support SMAP instructions">;
def FeatureLeaForSP : SubtargetFeature<"lea-sp", "UseLeaForSP", "true",
                                     "Use LEA for adjusting the stack pointer">;
def FeatureSlowDivide32 : SubtargetFeature<"idivl-to-divb",
                                     "HasSlowDivide32", "true",
                                     "Use 8-bit divide for positive values less than 256">;
def FeatureSlowDivide64 : SubtargetFeature<"idivq-to-divw",
                                     "HasSlowDivide64", "true",
                                     "Use 16-bit divide for positive values less than 65536">;
def FeaturePadShortFunctions : SubtargetFeature<"pad-short-functions",
                                     "PadShortFunctions", "true",
                                     "Pad short functions">;
def FeatureCallRegIndirect : SubtargetFeature<"call-reg-indirect",
                                     "CallRegIndirect", "true",
                                     "Call register indirect">;
def FeatureLEAUsesAG : SubtargetFeature<"lea-uses-ag", "LEAUsesAG", "true",
                                   "LEA instruction needs inputs at AG stage">;
def FeatureSlowLEA : SubtargetFeature<"slow-lea", "SlowLEA", "true",
                                   "LEA instruction with certain arguments is slow">;
def FeatureSlowIncDec : SubtargetFeature<"slow-incdec", "SlowIncDec", "true",
                                   "INC and DEC instructions are slower than ADD and SUB">;
def FeatureUseSqrtEst : SubtargetFeature<"use-sqrt-est", "UseSqrtEst", "true",
                            "Use RSQRT* to optimize square root calculations">;
def FeatureUseRecipEst : SubtargetFeature<"use-recip-est", "UseReciprocalEst",
                          "true", "Use RCP* to optimize division calculations">;

//===----------------------------------------------------------------------===//
// X86 processors supported.
//===----------------------------------------------------------------------===//

include "X86Schedule.td"

def ProcIntelAtom : SubtargetFeature<"atom", "X86ProcFamily", "IntelAtom",
                    "Intel Atom processors">;
def ProcIntelSLM  : SubtargetFeature<"slm", "X86ProcFamily", "IntelSLM",
                    "Intel Silvermont processors">;

class Proc<string Name, list<SubtargetFeature> Features>
 : ProcessorModel<Name, GenericModel, Features>;

def : Proc<"generic",         []>;
def : Proc<"i386",            []>;
def : Proc<"i486",            []>;
def : Proc<"i586",            []>;
def : Proc<"pentium",         []>;
def : Proc<"pentium-mmx",     [FeatureMMX]>;
def : Proc<"i686",            []>;
def : Proc<"pentiumpro",      [FeatureCMOV]>;
def : Proc<"pentium2",        [FeatureMMX, FeatureCMOV]>;
def : Proc<"pentium3",        [FeatureSSE1]>;
def : Proc<"pentium3m",       [FeatureSSE1, FeatureSlowBTMem]>;
def : Proc<"pentium-m",       [FeatureSSE2, FeatureSlowBTMem]>;
def : Proc<"pentium4",        [FeatureSSE2]>;
def : Proc<"pentium4m",       [FeatureSSE2, FeatureSlowBTMem]>;

// Intel Core Duo.
def : ProcessorModel<"yonah", SandyBridgeModel,
                     [FeatureSSE3, FeatureSlowBTMem]>;

// NetBurst.
def : Proc<"prescott", [FeatureSSE3, FeatureSlowBTMem]>;
def : Proc<"nocona",   [FeatureSSE3, FeatureCMPXCHG16B, FeatureSlowBTMem]>;

// Intel Core 2 Solo/Duo.
def : ProcessorModel<"core2", SandyBridgeModel,
                     [FeatureSSSE3, FeatureCMPXCHG16B, FeatureSlowBTMem]>;
def : ProcessorModel<"penryn", SandyBridgeModel,
                     [FeatureSSE41, FeatureCMPXCHG16B, FeatureSlowBTMem]>;

// Atom CPUs.
class BonnellProc<string Name> : ProcessorModel<Name, AtomModel, [
                                   ProcIntelAtom,
                                   FeatureSSSE3,
                                   FeatureCMPXCHG16B,
                                   FeatureMOVBE,
                                   FeatureSlowBTMem,
                                   FeatureLeaForSP,
                                   FeatureSlowDivide32,
                                   FeatureSlowDivide64,
                                   FeatureCallRegIndirect,
                                   FeatureLEAUsesAG,
                                   FeaturePadShortFunctions
                                 ]>;
def : BonnellProc<"bonnell">;
def : BonnellProc<"atom">; // Pin the generic name to the baseline.

class SilvermontProc<string Name> : ProcessorModel<Name, SLMModel, [
                                      ProcIntelSLM,
                                      FeatureSSE42,
                                      FeatureCMPXCHG16B,
                                      FeatureMOVBE,
                                      FeaturePOPCNT,
                                      FeaturePCLMUL,
                                      FeatureAES,
                                      FeatureSlowDivide64,
                                      FeatureCallRegIndirect,
                                      FeaturePRFCHW,
                                      FeatureSlowLEA,
                                      FeatureSlowIncDec,
                                      FeatureSlowBTMem,
                                      FeatureFastUAMem
                                    ]>;
def : SilvermontProc<"silvermont">;
def : SilvermontProc<"slm">; // Legacy alias.

// "Arrandale" along with corei3 and corei5
class NehalemProc<string Name, list<SubtargetFeature> AdditionalFeatures>
    : ProcessorModel<Name, SandyBridgeModel, !listconcat([
                                                           FeatureSSE42,
                                                           FeatureCMPXCHG16B,
                                                           FeatureSlowBTMem,
                                                           FeatureFastUAMem,
                                                           FeaturePOPCNT
                                                         ],
                                                         AdditionalFeatures)>;
def : NehalemProc<"nehalem", []>;
def : NehalemProc<"corei7", [FeatureAES]>;

// Westmere is a similar machine to nehalem with some additional features.
// Westmere is the corei3/i5/i7 path from nehalem to sandybridge
class WestmereProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
                                    FeatureSSE42,
                                    FeatureCMPXCHG16B,
                                    FeatureSlowBTMem,
                                    FeatureFastUAMem,
                                    FeaturePOPCNT,
                                    FeatureAES,
                                    FeaturePCLMUL
                                  ]>;
def : WestmereProc<"westmere">;

// SSE is not listed here since llvm treats AVX as a reimplementation of SSE,
// rather than a superset.
class SandyBridgeProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
                                       FeatureAVX,
                                       FeatureCMPXCHG16B,
                                       FeatureFastUAMem,
                                       FeatureSlowUAMem32,
                                       FeaturePOPCNT,
                                       FeatureAES,
                                       FeaturePCLMUL
                                     ]>;
def : SandyBridgeProc<"sandybridge">;
def : SandyBridgeProc<"corei7-avx">; // Legacy alias.

class IvyBridgeProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
                                     FeatureAVX,
                                     FeatureCMPXCHG16B,
                                     FeatureFastUAMem,
                                     FeatureSlowUAMem32,
                                     FeaturePOPCNT,
                                     FeatureAES,
                                     FeaturePCLMUL,
                                     FeatureRDRAND,
                                     FeatureF16C,
                                     FeatureFSGSBase
                                   ]>;
def : IvyBridgeProc<"ivybridge">;
def : IvyBridgeProc<"core-avx-i">; // Legacy alias.

class HaswellProc<string Name> : ProcessorModel<Name, HaswellModel, [
                                   FeatureAVX2,
                                   FeatureCMPXCHG16B,
                                   FeatureFastUAMem,
                                   FeaturePOPCNT,
                                   FeatureAES,
                                   FeaturePCLMUL,
                                   FeatureRDRAND,
                                   FeatureF16C,
                                   FeatureFSGSBase,
                                   FeatureMOVBE,
                                   FeatureLZCNT,
                                   FeatureBMI,
                                   FeatureBMI2,
                                   FeatureFMA,
                                   FeatureRTM,
                                   FeatureHLE,
                                   FeatureSlowIncDec
                                 ]>;
def : HaswellProc<"haswell">;
def : HaswellProc<"core-avx2">; // Legacy alias.

class BroadwellProc<string Name> : ProcessorModel<Name, HaswellModel, [
                                     FeatureAVX2,
                                     FeatureCMPXCHG16B,
                                     FeatureFastUAMem,
                                     FeaturePOPCNT,
                                     FeatureAES,
                                     FeaturePCLMUL,
                                     FeatureRDRAND,
                                     FeatureF16C,
                                     FeatureFSGSBase,
                                     FeatureMOVBE,
                                     FeatureLZCNT,
                                     FeatureBMI,
                                     FeatureBMI2,
                                     FeatureFMA,
                                     FeatureRTM,
                                     FeatureHLE,
                                     FeatureADX,
                                     FeatureRDSEED,
                                     FeatureSMAP,
                                     FeatureSlowIncDec
                                   ]>;
def : BroadwellProc<"broadwell">;

// FIXME: define KNL model
class KnightsLandingProc<string Name> : ProcessorModel<Name, HaswellModel,
                     [FeatureAVX512, FeatureERI, FeatureCDI, FeaturePFI,
                      FeatureCMPXCHG16B, FeatureFastUAMem, FeaturePOPCNT,
                      FeatureAES, FeaturePCLMUL, FeatureRDRAND, FeatureF16C,
                      FeatureFSGSBase, FeatureMOVBE, FeatureLZCNT, FeatureBMI,
                      FeatureBMI2, FeatureFMA, FeatureRTM, FeatureHLE,
                      FeatureSlowIncDec]>;
def : KnightsLandingProc<"knl">;

// FIXME: define SKX model
class SkylakeProc<string Name> : ProcessorModel<Name, HaswellModel,
                     [FeatureAVX512, FeatureCDI,
                      FeatureDQI, FeatureBWI, FeatureVLX,
                      FeatureCMPXCHG16B, FeatureFastUAMem, FeaturePOPCNT,
                      FeatureAES, FeaturePCLMUL, FeatureRDRAND, FeatureF16C,
                      FeatureFSGSBase, FeatureMOVBE, FeatureLZCNT, FeatureBMI,
                      FeatureBMI2, FeatureFMA, FeatureRTM, FeatureHLE,
                      FeatureSlowIncDec, FeatureSGX]>;
def : SkylakeProc<"skylake">;
def : SkylakeProc<"skx">; // Legacy alias.


// AMD CPUs.

def : Proc<"k6",              [FeatureMMX]>;
def : Proc<"k6-2",            [Feature3DNow]>;
def : Proc<"k6-3",            [Feature3DNow]>;
def : Proc<"athlon",          [Feature3DNowA, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon-tbird",    [Feature3DNowA, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon-4",        [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon-xp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon-mp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"k8",              [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"opteron",         [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon64",        [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon-fx",       [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"k8-sse3",         [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"opteron-sse3",    [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon64-sse3",   [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"amdfam10",        [FeatureSSE4A,
                               Feature3DNowA, FeatureCMPXCHG16B, FeatureLZCNT,
                               FeaturePOPCNT, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"barcelona",       [FeatureSSE4A,
                               Feature3DNowA, FeatureCMPXCHG16B, FeatureLZCNT,
                               FeaturePOPCNT, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
// Bobcat
def : Proc<"btver1",          [FeatureSSSE3, FeatureSSE4A, FeatureCMPXCHG16B,
                               FeaturePRFCHW, FeatureLZCNT, FeaturePOPCNT,
                               FeatureSlowSHLD]>;

// Jaguar
def : ProcessorModel<"btver2", BtVer2Model,
                     [FeatureAVX, FeatureSSE4A, FeatureCMPXCHG16B,
                      FeaturePRFCHW, FeatureAES, FeaturePCLMUL,
                      FeatureBMI, FeatureF16C, FeatureMOVBE,
                      FeatureLZCNT, FeaturePOPCNT, FeatureFastUAMem,
                      FeatureSlowSHLD, FeatureUseSqrtEst, FeatureUseRecipEst]>;

// TODO: We should probably add 'FeatureFastUAMem' to all of the AMD chips.

// Bulldozer
def : Proc<"bdver1",          [FeatureXOP, FeatureFMA4, FeatureCMPXCHG16B,
                               FeatureAES, FeaturePRFCHW, FeaturePCLMUL,
                               FeatureAVX, FeatureSSE4A, FeatureLZCNT,
                               FeaturePOPCNT, FeatureSlowSHLD]>;
// Piledriver
def : Proc<"bdver2",          [FeatureXOP, FeatureFMA4, FeatureCMPXCHG16B,
                               FeatureAES, FeaturePRFCHW, FeaturePCLMUL,
                               FeatureAVX, FeatureSSE4A, FeatureF16C,
                               FeatureLZCNT, FeaturePOPCNT, FeatureBMI,
                               FeatureTBM, FeatureFMA, FeatureSlowSHLD]>;

// Steamroller
def : Proc<"bdver3",          [FeatureXOP, FeatureFMA4, FeatureCMPXCHG16B,
                               FeatureAES, FeaturePRFCHW, FeaturePCLMUL,
                               FeatureAVX, FeatureSSE4A, FeatureF16C,
                               FeatureLZCNT, FeaturePOPCNT, FeatureBMI,
                               FeatureTBM, FeatureFMA, FeatureSlowSHLD,
                               FeatureFSGSBase]>;

// Excavator
def : Proc<"bdver4",          [FeatureAVX2, FeatureXOP, FeatureFMA4,
                               FeatureCMPXCHG16B, FeatureAES, FeaturePRFCHW,
                               FeaturePCLMUL, FeatureF16C, FeatureLZCNT,
                               FeaturePOPCNT, FeatureBMI, FeatureBMI2,
                               FeatureTBM, FeatureFMA, FeatureSSE4A,
                               FeatureFSGSBase]>;

def : Proc<"geode",           [Feature3DNowA]>;

def : Proc<"winchip-c6",      [FeatureMMX]>;
def : Proc<"winchip2",        [Feature3DNow]>;
def : Proc<"c3",              [Feature3DNow]>;
def : Proc<"c3-2",            [FeatureSSE1]>;

// We also provide a generic 64-bit specific x86 processor model which tries to
// be good for modern chips without enabling instruction set encodings past the
// basic SSE2 and 64-bit ones. It disables slow things from any mainstream and
// modern 64-bit x86 chip, and enables features that are generally beneficial.
//
// We currently use the Sandy Bridge model as the default scheduling model as
// we use it across Nehalem, Westmere, Sandy Bridge, and Ivy Bridge which
// covers a huge swath of x86 processors. If there are specific scheduling
// knobs which need to be tuned differently for AMD chips, we might consider
// forming a common base for them.
def : ProcessorModel<"x86-64", SandyBridgeModel,
                     [FeatureSSE2, Feature64Bit, FeatureSlowBTMem,
                      FeatureFastUAMem]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "X86InstrInfo.td"

def X86InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "X86CallingConv.td"


//===----------------------------------------------------------------------===//
// Assembly Parser
//===----------------------------------------------------------------------===//

def ATTAsmParser : AsmParser {
  string AsmParserClassName = "AsmParser";
}

def ATTAsmParserVariant : AsmParserVariant {
  int Variant = 0;

  // Variant name.
  string Name = "att";

  // Discard comments in assembly strings.
  string CommentDelimiter = "#";

  // Recognize hard coded registers.
  string RegisterPrefix = "%";
}

def IntelAsmParserVariant : AsmParserVariant {
  int Variant = 1;

  // Variant name.
  string Name = "intel";

  // Discard comments in assembly strings.
  string CommentDelimiter = ";";

  // Recognize hard coded registers.
  string RegisterPrefix = "";
}

//===----------------------------------------------------------------------===//
// Assembly Printers
//===----------------------------------------------------------------------===//

// The X86 target supports two different syntaxes for emitting machine code.
// This is controlled by the -x86-asm-syntax={att|intel}
def ATTAsmWriter : AsmWriter {
  string AsmWriterClassName  = "ATTInstPrinter";
  int Variant = 0;
}
def IntelAsmWriter : AsmWriter {
  string AsmWriterClassName  = "IntelInstPrinter";
  int Variant = 1;
}

def X86 : Target {
  // Information about the instructions...
  let InstructionSet = X86InstrInfo;
  let AssemblyParsers = [ATTAsmParser];
  let AssemblyParserVariants = [ATTAsmParserVariant, IntelAsmParserVariant];
  let AssemblyWriters = [ATTAsmWriter, IntelAsmWriter];
}
