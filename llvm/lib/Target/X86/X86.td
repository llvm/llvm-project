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
def FeatureSSE41   : SubtargetFeature<"sse41", "X86SSELevel", "SSE41",
                                      "Enable SSE 4.1 instructions",
                                      [FeatureSSSE3]>;
def FeatureSSE42   : SubtargetFeature<"sse42", "X86SSELevel", "SSE42",
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
def FeatureCMPXCHG16B : SubtargetFeature<"cmpxchg16b", "HasCmpxchg16b", "true",
                                      "64-bit with cmpxchg16b",
                                      [Feature64Bit]>;
def FeatureSlowBTMem : SubtargetFeature<"slow-bt-mem", "IsBTMemSlow", "true",
                                       "Bit testing of memory is slow">;
def FeatureFastUAMem : SubtargetFeature<"fast-unaligned-mem",
                                        "IsUAMemFast", "true",
                                        "Fast unaligned memory access">;
def FeatureSSE4A   : SubtargetFeature<"sse4a", "HasSSE4A", "true",
                                      "Support SSE 4a instructions",
                                      [FeatureSSE3]>;

def FeatureAVX     : SubtargetFeature<"avx", "X86SSELevel", "AVX",
                                      "Enable AVX instructions",
                                      [FeatureSSE42]>;
def FeatureAVX2    : SubtargetFeature<"avx2", "X86SSELevel", "AVX2",
                                      "Enable AVX2 instructions",
                                      [FeatureAVX]>;
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
def FeatureVectorUAMem : SubtargetFeature<"vector-unaligned-mem",
                                          "HasVectorUAMem", "true",
                 "Allow unaligned memory operands on vector/SIMD instructions">;
def FeatureAES     : SubtargetFeature<"aes", "HasAES", "true",
                                      "Enable AES instructions",
                                      [FeatureSSE2]>;
def FeatureMOVBE   : SubtargetFeature<"movbe", "HasMOVBE", "true",
                                      "Support MOVBE instruction">;
def FeatureRDRAND  : SubtargetFeature<"rdrand", "HasRDRAND", "true",
                                      "Support RDRAND instruction">;
def FeatureF16C    : SubtargetFeature<"f16c", "HasF16C", "true",
                       "Support 16-bit floating point conversion instructions">;
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
def FeatureLeaForSP : SubtargetFeature<"lea-sp", "UseLeaForSP", "true",
                                     "Use LEA for adjusting the stack pointer">;
def FeatureSlowDivide : SubtargetFeature<"idiv-to-divb",
                          "HasSlowDivide", "true",
                          "Use small divide for positive values less than 256">;

//===----------------------------------------------------------------------===//
// X86 processors supported.
//===----------------------------------------------------------------------===//

include "X86Schedule.td"

def ProcIntelAtom : SubtargetFeature<"atom", "X86ProcFamily", "IntelAtom",
                    "Intel Atom processors">;

class Proc<string Name, list<SubtargetFeature> Features>
 : ProcessorModel<Name, GenericModel, Features>;

class AtomProc<string Name, list<SubtargetFeature> Features>
 : ProcessorModel<Name, AtomModel, Features>;

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
def : Proc<"x86-64",          [FeatureSSE2, Feature64Bit, FeatureSlowBTMem]>;
def : Proc<"yonah",           [FeatureSSE3, FeatureSlowBTMem]>;
def : Proc<"prescott",        [FeatureSSE3, FeatureSlowBTMem]>;
def : Proc<"nocona",          [FeatureSSE3, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : Proc<"core2",           [FeatureSSSE3, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : Proc<"penryn",          [FeatureSSE41, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : AtomProc<"atom",        [ProcIntelAtom, FeatureSSSE3, FeatureCMPXCHG16B,
                               FeatureMOVBE, FeatureSlowBTMem, FeatureLeaForSP,
                               FeatureSlowDivide]>;
// "Arrandale" along with corei3 and corei5
def : Proc<"corei7",          [FeatureSSE42, FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureFastUAMem,
                               FeaturePOPCNT, FeatureAES]>;
def : Proc<"nehalem",         [FeatureSSE42,  FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureFastUAMem,
                               FeaturePOPCNT]>;
// Westmere is a similar machine to nehalem with some additional features.
// Westmere is the corei3/i5/i7 path from nehalem to sandybridge
def : Proc<"westmere",        [FeatureSSE42, FeatureCMPXCHG16B,
                               FeatureSlowBTMem, FeatureFastUAMem,
                               FeaturePOPCNT, FeatureAES, FeaturePCLMUL]>;
// Sandy Bridge
// SSE is not listed here since llvm treats AVX as a reimplementation of SSE,
// rather than a superset.
def : Proc<"corei7-avx",      [FeatureAVX, FeatureCMPXCHG16B, FeaturePOPCNT,
                               FeatureAES, FeaturePCLMUL]>;
// Ivy Bridge
def : Proc<"core-avx-i",      [FeatureAVX, FeatureCMPXCHG16B, FeaturePOPCNT,
                               FeatureAES, FeaturePCLMUL,
                               FeatureRDRAND, FeatureF16C, FeatureFSGSBase]>;

// Haswell
def : Proc<"core-avx2",       [FeatureAVX2, FeatureCMPXCHG16B, FeaturePOPCNT,
                               FeatureAES, FeaturePCLMUL, FeatureRDRAND,
                               FeatureF16C, FeatureFSGSBase,
                               FeatureMOVBE, FeatureLZCNT, FeatureBMI,
                               FeatureBMI2, FeatureFMA,
                               FeatureRTM]>;

def : Proc<"k6",              [FeatureMMX]>;
def : Proc<"k6-2",            [Feature3DNow]>;
def : Proc<"k6-3",            [Feature3DNow]>;
def : Proc<"athlon",          [Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-tbird",    [Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-4",        [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-xp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"athlon-mp",       [FeatureSSE1,   Feature3DNowA, FeatureSlowBTMem]>;
def : Proc<"k8",              [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"opteron",         [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"athlon64",        [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"athlon-fx",       [FeatureSSE2,   Feature3DNowA, Feature64Bit,
                               FeatureSlowBTMem]>;
def : Proc<"k8-sse3",         [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : Proc<"opteron-sse3",    [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : Proc<"athlon64-sse3",   [FeatureSSE3,   Feature3DNowA, FeatureCMPXCHG16B,
                               FeatureSlowBTMem]>;
def : Proc<"amdfam10",        [FeatureSSE4A,
                               Feature3DNowA, FeatureCMPXCHG16B, FeatureLZCNT,
                               FeaturePOPCNT, FeatureSlowBTMem]>;
// Bobcat
def : Proc<"btver1",          [FeatureSSSE3, FeatureSSE4A, FeatureCMPXCHG16B,
                               FeatureLZCNT, FeaturePOPCNT]>;
// Bulldozer
def : Proc<"bdver1",          [FeatureXOP, FeatureFMA4, FeatureCMPXCHG16B,
                               FeatureAES, FeaturePCLMUL,
                               FeatureLZCNT, FeaturePOPCNT]>;
// Enhanced Bulldozer
def : Proc<"bdver2",          [FeatureXOP, FeatureFMA4, FeatureCMPXCHG16B,
                               FeatureAES, FeaturePCLMUL,
                               FeatureF16C, FeatureLZCNT,
                               FeaturePOPCNT, FeatureBMI, FeatureFMA]>;
def : Proc<"geode",           [Feature3DNowA]>;

def : Proc<"winchip-c6",      [FeatureMMX]>;
def : Proc<"winchip2",        [Feature3DNow]>;
def : Proc<"c3",              [Feature3DNow]>;
def : Proc<"c3-2",            [FeatureSSE1]>;

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

  // Discard comments in assembly strings.
  string CommentDelimiter = "#";

  // Recognize hard coded registers.
  string RegisterPrefix = "%";
}

def IntelAsmParserVariant : AsmParserVariant {
  int Variant = 1;

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
  bit isMCAsmWriter = 1;
}
def IntelAsmWriter : AsmWriter {
  string AsmWriterClassName  = "IntelInstPrinter";
  int Variant = 1;
  bit isMCAsmWriter = 1;
}

def X86 : Target {
  // Information about the instructions...
  let InstructionSet = X86InstrInfo;
  let AssemblyParsers = [ATTAsmParser];
  let AssemblyParserVariants = [ATTAsmParserVariant, IntelAsmParserVariant];
  let AssemblyWriters = [ATTAsmWriter, IntelAsmWriter];
}
