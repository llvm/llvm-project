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

def FeatureFXSR    : SubtargetFeature<"fxsr", "HasFXSR", "true",
                                      "Support fxsave/fxrestore instructions">;

def FeatureXSAVE   : SubtargetFeature<"xsave", "HasXSAVE", "true",
                                       "Support xsave instructions">;

def FeatureXSAVEOPT: SubtargetFeature<"xsaveopt", "HasXSAVEOPT", "true",
                                       "Support xsaveopt instructions">;

def FeatureXSAVEC  : SubtargetFeature<"xsavec", "HasXSAVEC", "true",
                                       "Support xsavec instructions">;

def FeatureXSAVES  : SubtargetFeature<"xsaves", "HasXSAVES", "true",
                                       "Support xsaves instructions">;

def FeatureSSE1    : SubtargetFeature<"sse", "X86SSELevel", "SSE1",
                                      "Enable SSE instructions",
                                      // SSE codegen depends on cmovs, and all
                                      // SSE1+ processors support them.
                                      [FeatureCMOV]>;
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
// The MMX subtarget feature is separate from the rest of the SSE features
// because it's important (for odd compatibility reasons) to be able to
// turn it off explicitly while allowing SSE+ to be on.
def FeatureMMX     : SubtargetFeature<"mmx","X863DNowLevel", "MMX",
                                      "Enable MMX instructions">;
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
// FIXME: This should not apply to CPUs that do not have SSE.
def FeatureSlowUAMem16 : SubtargetFeature<"slow-unaligned-mem-16",
                                "IsUAMem16Slow", "true",
                                "Slow unaligned 16-byte memory access">;
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
def FeaturePKU   : SubtargetFeature<"pku", "HasPKU", "true",
                      "Enable protection keys">;
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
def FeaturePRFCHW  : SubtargetFeature<"prfchw", "HasPRFCHW", "true",
                                      "Support PRFCHW instructions">;
def FeatureRDSEED  : SubtargetFeature<"rdseed", "HasRDSEED", "true",
                                      "Support RDSEED instruction">;
def FeatureLAHFSAHF : SubtargetFeature<"sahf", "HasLAHFSAHF", "true",
                                       "Support LAHF and SAHF instructions">;
def FeatureMPX     : SubtargetFeature<"mpx", "HasMPX", "true",
                                      "Support MPX instructions">;
def FeatureLEAForSP : SubtargetFeature<"lea-sp", "UseLeaForSP", "true",
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
// TODO: This feature ought to be renamed.
// What it really refers to are CPUs for which certain instructions
// (which ones besides the example below?) are microcoded.
// The best examples of this are the memory forms of CALL and PUSH
// instructions, which should be avoided in favor of a MOV + register CALL/PUSH.
def FeatureCallRegIndirect : SubtargetFeature<"call-reg-indirect",
                                     "CallRegIndirect", "true",
                                     "Call register indirect">;
def FeatureLEAUsesAG : SubtargetFeature<"lea-uses-ag", "LEAUsesAG", "true",
                                   "LEA instruction needs inputs at AG stage">;
def FeatureSlowLEA : SubtargetFeature<"slow-lea", "SlowLEA", "true",
                                   "LEA instruction with certain arguments is slow">;
def FeatureSlowIncDec : SubtargetFeature<"slow-incdec", "SlowIncDec", "true",
                                   "INC and DEC instructions are slower than ADD and SUB">;
def FeatureSoftFloat
    : SubtargetFeature<"soft-float", "UseSoftFloat", "true",
                       "Use software floating point features.">;

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

def : Proc<"generic",         [FeatureSlowUAMem16]>;
def : Proc<"i386",            [FeatureSlowUAMem16]>;
def : Proc<"i486",            [FeatureSlowUAMem16]>;
def : Proc<"i586",            [FeatureSlowUAMem16]>;
def : Proc<"pentium",         [FeatureSlowUAMem16]>;
def : Proc<"pentium-mmx",     [FeatureSlowUAMem16, FeatureMMX]>;
def : Proc<"i686",            [FeatureSlowUAMem16]>;
def : Proc<"pentiumpro",      [FeatureSlowUAMem16, FeatureCMOV]>;
def : Proc<"pentium2",        [FeatureSlowUAMem16, FeatureMMX, FeatureCMOV,
                               FeatureFXSR]>;
def : Proc<"pentium3",        [FeatureSlowUAMem16, FeatureMMX, FeatureSSE1,
                               FeatureFXSR]>;
def : Proc<"pentium3m",       [FeatureSlowUAMem16, FeatureMMX, FeatureSSE1,
                               FeatureFXSR, FeatureSlowBTMem]>;
def : Proc<"pentium-m",       [FeatureSlowUAMem16, FeatureMMX, FeatureSSE2,
                               FeatureFXSR, FeatureSlowBTMem]>;
def : Proc<"pentium4",        [FeatureSlowUAMem16, FeatureMMX, FeatureSSE2,
                               FeatureFXSR]>;
def : Proc<"pentium4m",       [FeatureSlowUAMem16, FeatureMMX, FeatureSSE2,
                               FeatureFXSR, FeatureSlowBTMem]>;

// Intel Core Duo.
def : ProcessorModel<"yonah", SandyBridgeModel,
                     [FeatureSlowUAMem16, FeatureMMX, FeatureSSE3, FeatureFXSR,
                      FeatureSlowBTMem]>;

// NetBurst.
def : Proc<"prescott",
           [FeatureSlowUAMem16, FeatureMMX, FeatureSSE3, FeatureFXSR,
            FeatureSlowBTMem]>;
def : Proc<"nocona", [
  FeatureSlowUAMem16,
  FeatureMMX,
  FeatureSSE3,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem
]>;

// Intel Core 2 Solo/Duo.
def : ProcessorModel<"core2", SandyBridgeModel, [
  FeatureSlowUAMem16,
  FeatureMMX,
  FeatureSSSE3,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeatureLAHFSAHF
]>;
def : ProcessorModel<"penryn", SandyBridgeModel, [
  FeatureSlowUAMem16,
  FeatureMMX,
  FeatureSSE41,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeatureLAHFSAHF
]>;

// Atom CPUs.
class BonnellProc<string Name> : ProcessorModel<Name, AtomModel, [
  ProcIntelAtom,
  FeatureSlowUAMem16,
  FeatureMMX,
  FeatureSSSE3,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureMOVBE,
  FeatureSlowBTMem,
  FeatureLEAForSP,
  FeatureSlowDivide32,
  FeatureSlowDivide64,
  FeatureCallRegIndirect,
  FeatureLEAUsesAG,
  FeaturePadShortFunctions,
  FeatureLAHFSAHF
]>;
def : BonnellProc<"bonnell">;
def : BonnellProc<"atom">; // Pin the generic name to the baseline.

class SilvermontProc<string Name> : ProcessorModel<Name, SLMModel, [
  ProcIntelSLM,
  FeatureMMX,
  FeatureSSE42,
  FeatureFXSR,
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
  FeatureLAHFSAHF
]>;
def : SilvermontProc<"silvermont">;
def : SilvermontProc<"slm">; // Legacy alias.

// "Arrandale" along with corei3 and corei5
class NehalemProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
  FeatureMMX,
  FeatureSSE42,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeaturePOPCNT,
  FeatureLAHFSAHF
]>;
def : NehalemProc<"nehalem">;
def : NehalemProc<"corei7">;

// Westmere is a similar machine to nehalem with some additional features.
// Westmere is the corei3/i5/i7 path from nehalem to sandybridge
class WestmereProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
  FeatureMMX,
  FeatureSSE42,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureLAHFSAHF
]>;
def : WestmereProc<"westmere">;

// SSE is not listed here since llvm treats AVX as a reimplementation of SSE,
// rather than a superset.
class SandyBridgeProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeatureSlowUAMem32,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureXSAVE,
  FeatureXSAVEOPT,
  FeatureLAHFSAHF
]>;
def : SandyBridgeProc<"sandybridge">;
def : SandyBridgeProc<"corei7-avx">; // Legacy alias.

class IvyBridgeProc<string Name> : ProcessorModel<Name, SandyBridgeModel, [
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeatureSlowUAMem32,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureXSAVE,
  FeatureXSAVEOPT,
  FeatureRDRAND,
  FeatureF16C,
  FeatureFSGSBase,
  FeatureLAHFSAHF
]>;
def : IvyBridgeProc<"ivybridge">;
def : IvyBridgeProc<"core-avx-i">; // Legacy alias.

class HaswellProc<string Name> : ProcessorModel<Name, HaswellModel, [
  FeatureMMX,
  FeatureAVX2,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureRDRAND,
  FeatureXSAVE,
  FeatureXSAVEOPT,
  FeatureF16C,
  FeatureFSGSBase,
  FeatureMOVBE,
  FeatureLZCNT,
  FeatureBMI,
  FeatureBMI2,
  FeatureFMA,
  FeatureRTM,
  FeatureHLE,
  FeatureSlowIncDec,
  FeatureLAHFSAHF
]>;
def : HaswellProc<"haswell">;
def : HaswellProc<"core-avx2">; // Legacy alias.

class BroadwellProc<string Name> : ProcessorModel<Name, HaswellModel, [
  FeatureMMX,
  FeatureAVX2,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureXSAVE,
  FeatureXSAVEOPT,
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
  FeatureSlowIncDec,
  FeatureLAHFSAHF
]>;
def : BroadwellProc<"broadwell">;

// FIXME: define KNL model
class KnightsLandingProc<string Name> : ProcessorModel<Name, HaswellModel, [
  FeatureMMX,
  FeatureAVX512,
  FeatureFXSR,
  FeatureERI,
  FeatureCDI,
  FeaturePFI,
  FeatureCMPXCHG16B,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureXSAVE,
  FeatureXSAVEOPT,
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
  FeatureSlowIncDec,
  FeatureMPX,
  FeatureLAHFSAHF
]>;
def : KnightsLandingProc<"knl">;

// FIXME: define SKX model
class SkylakeProc<string Name> : ProcessorModel<Name, HaswellModel, [
  FeatureMMX,
  FeatureAVX512,
  FeatureFXSR,
  FeatureCDI,
  FeatureDQI,
  FeatureBWI,
  FeatureVLX,
  FeaturePKU,
  FeatureCMPXCHG16B,
  FeatureSlowBTMem,
  FeaturePOPCNT,
  FeatureAES,
  FeaturePCLMUL,
  FeatureXSAVE,
  FeatureXSAVEOPT,
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
  FeatureSlowIncDec,
  FeatureMPX,
  FeatureXSAVEC,
  FeatureXSAVES,
  FeatureLAHFSAHF
]>;
def : SkylakeProc<"skylake">;
def : SkylakeProc<"skx">; // Legacy alias.


// AMD CPUs.

def : Proc<"k6",              [FeatureSlowUAMem16, FeatureMMX]>;
def : Proc<"k6-2",            [FeatureSlowUAMem16, Feature3DNow]>;
def : Proc<"k6-3",            [FeatureSlowUAMem16, Feature3DNow]>;
def : Proc<"athlon",          [FeatureSlowUAMem16, Feature3DNowA,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon-tbird",    [FeatureSlowUAMem16, Feature3DNowA,
                               FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon-4",        [FeatureSlowUAMem16, FeatureSSE1, Feature3DNowA,
                               FeatureFXSR, FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon-xp",       [FeatureSlowUAMem16, FeatureSSE1, Feature3DNowA,
                               FeatureFXSR, FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"athlon-mp",       [FeatureSlowUAMem16, FeatureSSE1, Feature3DNowA,
                               FeatureFXSR, FeatureSlowBTMem, FeatureSlowSHLD]>;
def : Proc<"k8",              [FeatureSlowUAMem16, FeatureSSE2, Feature3DNowA,
                               FeatureFXSR, Feature64Bit, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"opteron",         [FeatureSlowUAMem16, FeatureSSE2, Feature3DNowA,
                               FeatureFXSR, Feature64Bit, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon64",        [FeatureSlowUAMem16, FeatureSSE2,   Feature3DNowA,
                               FeatureFXSR, Feature64Bit, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon-fx",       [FeatureSlowUAMem16, FeatureSSE2,   Feature3DNowA,
                               FeatureFXSR, Feature64Bit, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"k8-sse3",         [FeatureSlowUAMem16, FeatureSSE3,   Feature3DNowA,
                               FeatureFXSR, FeatureCMPXCHG16B, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"opteron-sse3",    [FeatureSlowUAMem16, FeatureSSE3,   Feature3DNowA,
                               FeatureFXSR, FeatureCMPXCHG16B, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"athlon64-sse3",   [FeatureSlowUAMem16, FeatureSSE3,   Feature3DNowA,
                               FeatureFXSR, FeatureCMPXCHG16B, FeatureSlowBTMem,
                               FeatureSlowSHLD]>;
def : Proc<"amdfam10",        [FeatureSSE4A, Feature3DNowA, FeatureFXSR,
                               FeatureCMPXCHG16B, FeatureLZCNT, FeaturePOPCNT,
                               FeatureSlowBTMem, FeatureSlowSHLD, FeatureLAHFSAHF]>;
def : Proc<"barcelona",       [FeatureSSE4A, Feature3DNowA, FeatureFXSR,
                               FeatureCMPXCHG16B, FeatureLZCNT, FeaturePOPCNT,
                               FeatureSlowBTMem, FeatureSlowSHLD, FeatureLAHFSAHF]>;

// Bobcat
def : Proc<"btver1", [
  FeatureMMX,
  FeatureSSSE3,
  FeatureSSE4A,
  FeatureFXSR,
  FeatureCMPXCHG16B,
  FeaturePRFCHW,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureSlowSHLD,
  FeatureLAHFSAHF
]>;

// Jaguar
def : ProcessorModel<"btver2", BtVer2Model, [
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureSSE4A,
  FeatureCMPXCHG16B,
  FeaturePRFCHW,
  FeatureAES,
  FeaturePCLMUL,
  FeatureBMI,
  FeatureF16C,
  FeatureMOVBE,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureXSAVEOPT,
  FeatureSlowSHLD,
  FeatureLAHFSAHF
]>;

// Bulldozer
def : Proc<"bdver1", [
  FeatureXOP,
  FeatureFMA4,
  FeatureCMPXCHG16B,
  FeatureAES,
  FeaturePRFCHW,
  FeaturePCLMUL,
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureSSE4A,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureSlowSHLD,
  FeatureLAHFSAHF
]>;
// Piledriver
def : Proc<"bdver2", [
  FeatureXOP,
  FeatureFMA4,
  FeatureCMPXCHG16B,
  FeatureAES,
  FeaturePRFCHW,
  FeaturePCLMUL,
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureSSE4A,
  FeatureF16C,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureBMI,
  FeatureTBM,
  FeatureFMA,
  FeatureSlowSHLD,
  FeatureLAHFSAHF
]>;

// Steamroller
def : Proc<"bdver3", [
  FeatureXOP,
  FeatureFMA4,
  FeatureCMPXCHG16B,
  FeatureAES,
  FeaturePRFCHW,
  FeaturePCLMUL,
  FeatureMMX,
  FeatureAVX,
  FeatureFXSR,
  FeatureSSE4A,
  FeatureF16C,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureBMI,
  FeatureTBM,
  FeatureFMA,
  FeatureXSAVEOPT,
  FeatureSlowSHLD,
  FeatureFSGSBase,
  FeatureLAHFSAHF
]>;

// Excavator
def : Proc<"bdver4", [
  FeatureMMX,
  FeatureAVX2,
  FeatureFXSR,
  FeatureXOP,
  FeatureFMA4,
  FeatureCMPXCHG16B,
  FeatureAES,
  FeaturePRFCHW,
  FeaturePCLMUL,
  FeatureF16C,
  FeatureLZCNT,
  FeaturePOPCNT,
  FeatureXSAVE,
  FeatureBMI,
  FeatureBMI2,
  FeatureTBM,
  FeatureFMA,
  FeatureXSAVEOPT,
  FeatureFSGSBase,
  FeatureLAHFSAHF
]>;

def : Proc<"geode",           [FeatureSlowUAMem16, Feature3DNowA]>;

def : Proc<"winchip-c6",      [FeatureSlowUAMem16, FeatureMMX]>;
def : Proc<"winchip2",        [FeatureSlowUAMem16, Feature3DNow]>;
def : Proc<"c3",              [FeatureSlowUAMem16, Feature3DNow]>;
def : Proc<"c3-2", [FeatureSlowUAMem16, FeatureMMX, FeatureSSE1, FeatureFXSR]>;

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
                     [FeatureMMX, FeatureSSE2, FeatureFXSR, Feature64Bit,
                      FeatureSlowBTMem ]>;

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
  let AssemblyParserVariants = [ATTAsmParserVariant, IntelAsmParserVariant];
  let AssemblyWriters = [ATTAsmWriter, IntelAsmWriter];
}
