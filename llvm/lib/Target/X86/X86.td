//===-- X86.td - Target definition file for the Intel X86 --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

def FeatureX87     : SubtargetFeature<"x87","HasX87", "true",
                                      "Enable X87 float instructions">;

def FeatureNOPL    : SubtargetFeature<"nopl", "HasNOPL", "true",
                                      "Enable NOPL instruction">;

def FeatureCMOV    : SubtargetFeature<"cmov","HasCMov", "true",
                                      "Enable conditional move instructions">;

def FeatureCMPXCHG8B : SubtargetFeature<"cx8", "HasCmpxchg8b", "true",
                                        "Support CMPXCHG8B instructions">;

def FeaturePOPCNT   : SubtargetFeature<"popcnt", "HasPOPCNT", "true",
                                       "Support POPCNT instruction">;

def FeatureFXSR    : SubtargetFeature<"fxsr", "HasFXSR", "true",
                                      "Support fxsave/fxrestore instructions">;

def FeatureXSAVE   : SubtargetFeature<"xsave", "HasXSAVE", "true",
                                       "Support xsave instructions">;

def FeatureXSAVEOPT: SubtargetFeature<"xsaveopt", "HasXSAVEOPT", "true",
                                       "Support xsaveopt instructions",
                                       [FeatureXSAVE]>;

def FeatureXSAVEC  : SubtargetFeature<"xsavec", "HasXSAVEC", "true",
                                       "Support xsavec instructions",
                                       [FeatureXSAVE]>;

def FeatureXSAVES  : SubtargetFeature<"xsaves", "HasXSAVES", "true",
                                       "Support xsaves instructions",
                                       [FeatureXSAVE]>;

def FeatureSSE1    : SubtargetFeature<"sse", "X86SSELevel", "SSE1",
                                      "Enable SSE instructions">;
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
// without disabling 64-bit mode. Nothing should imply this feature bit. It
// is used to enforce that only 64-bit capable CPUs are used in 64-bit mode.
def Feature64Bit   : SubtargetFeature<"64bit", "HasX86_64", "true",
                                      "Support 64-bit instructions">;
def FeatureCMPXCHG16B : SubtargetFeature<"cx16", "HasCmpxchg16b", "true",
                                      "64-bit with cmpxchg16b",
                                      [FeatureCMPXCHG8B]>;
def FeatureSlowSHLD : SubtargetFeature<"slow-shld", "IsSHLDSlow", "true",
                                       "SHLD instruction is slow">;
def FeatureSlowPMULLD : SubtargetFeature<"slow-pmulld", "IsPMULLDSlow", "true",
                                        "PMULLD instruction is slow">;
def FeatureSlowPMADDWD : SubtargetFeature<"slow-pmaddwd", "IsPMADDWDSlow",
                                          "true",
                                          "PMADDWD is slower than PMULLD">;
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
def FeatureFMA     : SubtargetFeature<"fma", "HasFMA", "true",
                                      "Enable three-operand fused multiple-add",
                                      [FeatureAVX]>;
def FeatureF16C    : SubtargetFeature<"f16c", "HasF16C", "true",
                       "Support 16-bit floating point conversion instructions",
                       [FeatureAVX]>;
def FeatureAVX512   : SubtargetFeature<"avx512f", "X86SSELevel", "AVX512F",
                                      "Enable AVX-512 instructions",
                                      [FeatureAVX2, FeatureFMA, FeatureF16C]>;
def FeatureERI      : SubtargetFeature<"avx512er", "HasERI", "true",
                      "Enable AVX-512 Exponential and Reciprocal Instructions",
                                      [FeatureAVX512]>;
def FeatureCDI      : SubtargetFeature<"avx512cd", "HasCDI", "true",
                      "Enable AVX-512 Conflict Detection Instructions",
                                      [FeatureAVX512]>;
def FeatureVPOPCNTDQ : SubtargetFeature<"avx512vpopcntdq", "HasVPOPCNTDQ",
                       "true", "Enable AVX-512 Population Count Instructions",
                                      [FeatureAVX512]>;
def FeaturePFI      : SubtargetFeature<"avx512pf", "HasPFI", "true",
                      "Enable AVX-512 PreFetch Instructions",
                                      [FeatureAVX512]>;
def FeaturePREFETCHWT1  : SubtargetFeature<"prefetchwt1", "HasPREFETCHWT1",
                                   "true",
                                   "Prefetch with Intent to Write and T1 Hint">;
def FeatureDQI     : SubtargetFeature<"avx512dq", "HasDQI", "true",
                      "Enable AVX-512 Doubleword and Quadword Instructions",
                                      [FeatureAVX512]>;
def FeatureBWI     : SubtargetFeature<"avx512bw", "HasBWI", "true",
                      "Enable AVX-512 Byte and Word Instructions",
                                      [FeatureAVX512]>;
def FeatureVLX     : SubtargetFeature<"avx512vl", "HasVLX", "true",
                      "Enable AVX-512 Vector Length eXtensions",
                                      [FeatureAVX512]>;
def FeatureVBMI     : SubtargetFeature<"avx512vbmi", "HasVBMI", "true",
                      "Enable AVX-512 Vector Byte Manipulation Instructions",
                                      [FeatureBWI]>;
def FeatureVBMI2    : SubtargetFeature<"avx512vbmi2", "HasVBMI2", "true",
                      "Enable AVX-512 further Vector Byte Manipulation Instructions",
                                      [FeatureBWI]>;
def FeatureIFMA     : SubtargetFeature<"avx512ifma", "HasIFMA", "true",
                      "Enable AVX-512 Integer Fused Multiple-Add",
                                      [FeatureAVX512]>;
def FeaturePKU   : SubtargetFeature<"pku", "HasPKU", "true",
                      "Enable protection keys">;
def FeatureVNNI    : SubtargetFeature<"avx512vnni", "HasVNNI", "true",
                          "Enable AVX-512 Vector Neural Network Instructions",
                                      [FeatureAVX512]>;
def FeatureBF16    : SubtargetFeature<"avx512bf16", "HasBF16", "true",
                           "Support bfloat16 floating point",
                                      [FeatureBWI]>;
def FeatureBITALG  : SubtargetFeature<"avx512bitalg", "HasBITALG", "true",
                       "Enable AVX-512 Bit Algorithms",
                        [FeatureBWI]>;
def FeatureVP2INTERSECT  : SubtargetFeature<"avx512vp2intersect",
                                            "HasVP2INTERSECT", "true",
                                            "Enable AVX-512 vp2intersect",
                                            [FeatureAVX512]>;
def FeaturePCLMUL  : SubtargetFeature<"pclmul", "HasPCLMUL", "true",
                         "Enable packed carry-less multiplication instructions",
                               [FeatureSSE2]>;
def FeatureGFNI    : SubtargetFeature<"gfni", "HasGFNI", "true",
                         "Enable Galois Field Arithmetic Instructions",
                               [FeatureSSE2]>;
def FeatureVPCLMULQDQ : SubtargetFeature<"vpclmulqdq", "HasVPCLMULQDQ", "true",
                                         "Enable vpclmulqdq instructions",
                                         [FeatureAVX, FeaturePCLMUL]>;
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
def FeatureVAES    : SubtargetFeature<"vaes", "HasVAES", "true",
                       "Promote selected AES instructions to AVX512/AVX registers",
                        [FeatureAVX, FeatureAES]>;
def FeatureTBM     : SubtargetFeature<"tbm", "HasTBM", "true",
                                      "Enable TBM instructions">;
def FeatureLWP     : SubtargetFeature<"lwp", "HasLWP", "true",
                                      "Enable LWP instructions">;
def FeatureMOVBE   : SubtargetFeature<"movbe", "HasMOVBE", "true",
                                      "Support MOVBE instruction">;
def FeatureRDRAND  : SubtargetFeature<"rdrnd", "HasRDRAND", "true",
                                      "Support RDRAND instruction">;
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
def FeatureADX     : SubtargetFeature<"adx", "HasADX", "true",
                                      "Support ADX instructions">;
def FeatureSHA     : SubtargetFeature<"sha", "HasSHA", "true",
                                      "Enable SHA instructions",
                                      [FeatureSSE2]>;
def FeatureSHSTK   : SubtargetFeature<"shstk", "HasSHSTK", "true",
                       "Support CET Shadow-Stack instructions">;
def FeaturePRFCHW  : SubtargetFeature<"prfchw", "HasPRFCHW", "true",
                                      "Support PRFCHW instructions">;
def FeatureRDSEED  : SubtargetFeature<"rdseed", "HasRDSEED", "true",
                                      "Support RDSEED instruction">;
def FeatureLAHFSAHF : SubtargetFeature<"sahf", "HasLAHFSAHF", "true",
                                       "Support LAHF and SAHF instructions">;
def FeatureMWAITX  : SubtargetFeature<"mwaitx", "HasMWAITX", "true",
                                      "Enable MONITORX/MWAITX timer functionality">;
def FeatureCLZERO  : SubtargetFeature<"clzero", "HasCLZERO", "true",
                                      "Enable Cache Line Zero">;
def FeatureCLDEMOTE  : SubtargetFeature<"cldemote", "HasCLDEMOTE", "true",
                                      "Enable Cache Demote">;
def FeaturePTWRITE  : SubtargetFeature<"ptwrite", "HasPTWRITE", "true",
                                      "Support ptwrite instruction">;
// FIXME: This feature is deprecated in 10.0 and should not be used for
// anything, but removing it would break IR files that may contain it in a
// target-feature attribute.
def FeatureDeprecatedMPX : SubtargetFeature<"mpx", "DeprecatedHasMPX", "false",
                                      "Deprecated. Support MPX instructions">;
def FeatureAMXTILE     : SubtargetFeature<"amx-tile", "HasAMXTILE", "true",
                                      "Support AMX-TILE instructions">;
def FeatureAMXINT8     : SubtargetFeature<"amx-int8", "HasAMXINT8", "true",
                                      "Support AMX-INT8 instructions",
                                      [FeatureAMXTILE]>;
def FeatureAMXBF16     : SubtargetFeature<"amx-bf16", "HasAMXBF16", "true",
                                      "Support AMX-BF16 instructions",
                                      [FeatureAMXTILE]>;
def FeatureLEAForSP : SubtargetFeature<"lea-sp", "UseLeaForSP", "true",
                                     "Use LEA for adjusting the stack pointer">;
def FeatureSlowDivide32 : SubtargetFeature<"idivl-to-divb",
                                     "HasSlowDivide32", "true",
                                     "Use 8-bit divide for positive values less than 256">;
def FeatureSlowDivide64 : SubtargetFeature<"idivq-to-divl",
                                     "HasSlowDivide64", "true",
                                     "Use 32-bit divide for positive values less than 2^32">;
def FeaturePadShortFunctions : SubtargetFeature<"pad-short-functions",
                                     "PadShortFunctions", "true",
                                     "Pad short functions">;
def FeatureINVPCID : SubtargetFeature<"invpcid", "HasINVPCID", "true",
                                      "Invalidate Process-Context Identifier">;
def FeatureSGX     : SubtargetFeature<"sgx", "HasSGX", "true",
                                      "Enable Software Guard Extensions">;
def FeatureCLFLUSHOPT : SubtargetFeature<"clflushopt", "HasCLFLUSHOPT", "true",
                                      "Flush A Cache Line Optimized">;
def FeatureCLWB    : SubtargetFeature<"clwb", "HasCLWB", "true",
                                      "Cache Line Write Back">;
def FeatureWBNOINVD    : SubtargetFeature<"wbnoinvd", "HasWBNOINVD", "true",
                                      "Write Back No Invalidate">;
def FeatureRDPID : SubtargetFeature<"rdpid", "HasRDPID", "true",
                                    "Support RDPID instructions">;
def FeatureWAITPKG  : SubtargetFeature<"waitpkg", "HasWAITPKG", "true",
                                      "Wait and pause enhancements">;
def FeatureENQCMD : SubtargetFeature<"enqcmd", "HasENQCMD", "true",
                                     "Has ENQCMD instructions">;
def FeatureSERIALIZE : SubtargetFeature<"serialize", "HasSERIALIZE", "true",
                                        "Has serialize instruction">;
def FeatureTSXLDTRK : SubtargetFeature<"tsxldtrk", "HasTSXLDTRK", "true",
                                       "Support TSXLDTRK instructions">;
// On some processors, instructions that implicitly take two memory operands are
// slow. In practice, this means that CALL, PUSH, and POP with memory operands
// should be avoided in favor of a MOV + register CALL/PUSH/POP.
def FeatureSlowTwoMemOps : SubtargetFeature<"slow-two-mem-ops",
                                     "SlowTwoMemOps", "true",
                                     "Two memory operand instructions are slow">;
def FeatureLEAUsesAG : SubtargetFeature<"lea-uses-ag", "LEAUsesAG", "true",
                                   "LEA instruction needs inputs at AG stage">;
def FeatureSlowLEA : SubtargetFeature<"slow-lea", "SlowLEA", "true",
                                   "LEA instruction with certain arguments is slow">;
def FeatureSlow3OpsLEA : SubtargetFeature<"slow-3ops-lea", "Slow3OpsLEA", "true",
                                   "LEA instruction with 3 ops or certain registers is slow">;
def FeatureSlowIncDec : SubtargetFeature<"slow-incdec", "SlowIncDec", "true",
                                   "INC and DEC instructions are slower than ADD and SUB">;
def FeatureSoftFloat
    : SubtargetFeature<"soft-float", "UseSoftFloat", "true",
                       "Use software floating point features">;
def FeaturePOPCNTFalseDeps : SubtargetFeature<"false-deps-popcnt",
                                     "HasPOPCNTFalseDeps", "true",
                                     "POPCNT has a false dependency on dest register">;
def FeatureLZCNTFalseDeps : SubtargetFeature<"false-deps-lzcnt-tzcnt",
                                     "HasLZCNTFalseDeps", "true",
                                     "LZCNT/TZCNT have a false dependency on dest register">;
def FeaturePCONFIG : SubtargetFeature<"pconfig", "HasPCONFIG", "true",
                                      "platform configuration instruction">;
// On recent X86 (port bound) processors, its preferable to combine to a single shuffle
// using a variable mask over multiple fixed shuffles.
def FeatureFastVariableShuffle
    : SubtargetFeature<"fast-variable-shuffle",
                       "HasFastVariableShuffle",
                       "true", "Shuffles with variable masks are fast">;
// On some X86 processors, a vzeroupper instruction should be inserted after
// using ymm/zmm registers before executing code that may use SSE instructions.
def FeatureInsertVZEROUPPER
    : SubtargetFeature<"vzeroupper",
                       "InsertVZEROUPPER",
                       "true", "Should insert vzeroupper instructions">;
// FeatureFastScalarFSQRT should be enabled if scalar FSQRT has shorter latency
// than the corresponding NR code. FeatureFastVectorFSQRT should be enabled if
// vector FSQRT has higher throughput than the corresponding NR code.
// The idea is that throughput bound code is likely to be vectorized, so for
// vectorized code we should care about the throughput of SQRT operations.
// But if the code is scalar that probably means that the code has some kind of
// dependency and we should care more about reducing the latency.
def FeatureFastScalarFSQRT
    : SubtargetFeature<"fast-scalar-fsqrt", "HasFastScalarFSQRT",
                       "true", "Scalar SQRT is fast (disable Newton-Raphson)">;
def FeatureFastVectorFSQRT
    : SubtargetFeature<"fast-vector-fsqrt", "HasFastVectorFSQRT",
                       "true", "Vector SQRT is fast (disable Newton-Raphson)">;
// If lzcnt has equivalent latency/throughput to most simple integer ops, it can
// be used to replace test/set sequences.
def FeatureFastLZCNT
    : SubtargetFeature<
          "fast-lzcnt", "HasFastLZCNT", "true",
          "LZCNT instructions are as fast as most simple integer ops">;
// If the target can efficiently decode NOPs upto 7-bytes in length.
def FeatureFast7ByteNOP
    : SubtargetFeature<
          "fast-7bytenop", "HasFast7ByteNOP", "true",
          "Target can quickly decode up to 7 byte NOPs">;
// If the target can efficiently decode NOPs upto 11-bytes in length.
def FeatureFast11ByteNOP
    : SubtargetFeature<
          "fast-11bytenop", "HasFast11ByteNOP", "true",
          "Target can quickly decode up to 11 byte NOPs">;
// If the target can efficiently decode NOPs upto 15-bytes in length.
def FeatureFast15ByteNOP
    : SubtargetFeature<
          "fast-15bytenop", "HasFast15ByteNOP", "true",
          "Target can quickly decode up to 15 byte NOPs">;
// Sandy Bridge and newer processors can use SHLD with the same source on both
// inputs to implement rotate to avoid the partial flag update of the normal
// rotate instructions.
def FeatureFastSHLDRotate
    : SubtargetFeature<
          "fast-shld-rotate", "HasFastSHLDRotate", "true",
          "SHLD can be used as a faster rotate">;

// Ivy Bridge and newer processors have enhanced REP MOVSB and STOSB (aka
// "string operations"). See "REP String Enhancement" in the Intel Software
// Development Manual. This feature essentially means that REP MOVSB will copy
// using the largest available size instead of copying bytes one by one, making
// it at least as fast as REPMOVS{W,D,Q}.
def FeatureERMSB
    : SubtargetFeature<
          "ermsb", "HasERMSB", "true",
          "REP MOVS/STOS are fast">;

// Bulldozer and newer processors can merge CMP/TEST (but not other
// instructions) with conditional branches.
def FeatureBranchFusion
    : SubtargetFeature<"branchfusion", "HasBranchFusion", "true",
                 "CMP/TEST can be fused with conditional branches">;

// Sandy Bridge and newer processors have many instructions that can be
// fused with conditional branches and pass through the CPU as a single
// operation.
def FeatureMacroFusion
    : SubtargetFeature<"macrofusion", "HasMacroFusion", "true",
                 "Various instructions can be fused with conditional branches">;

// Gather is available since Haswell (AVX2 set). So technically, we can
// generate Gathers on all AVX2 processors. But the overhead on HSW is high.
// Skylake Client processor has faster Gathers than HSW and performance is
// similar to Skylake Server (AVX-512).
def FeatureHasFastGather
    : SubtargetFeature<"fast-gather", "HasFastGather", "true",
                       "Indicates if gather is reasonably fast">;

def FeaturePrefer128Bit
    : SubtargetFeature<"prefer-128-bit", "Prefer128Bit", "true",
                       "Prefer 128-bit AVX instructions">;

def FeaturePrefer256Bit
    : SubtargetFeature<"prefer-256-bit", "Prefer256Bit", "true",
                       "Prefer 256-bit AVX instructions">;

def FeaturePreferMaskRegisters
    : SubtargetFeature<"prefer-mask-registers", "PreferMaskRegisters", "true",
                       "Prefer AVX512 mask registers over PTEST/MOVMSK">;

// Lower indirect calls using a special construct called a `retpoline` to
// mitigate potential Spectre v2 attacks against them.
def FeatureRetpolineIndirectCalls
    : SubtargetFeature<
          "retpoline-indirect-calls", "UseRetpolineIndirectCalls", "true",
          "Remove speculation of indirect calls from the generated code">;

// Lower indirect branches and switches either using conditional branch trees
// or using a special construct called a `retpoline` to mitigate potential
// Spectre v2 attacks against them.
def FeatureRetpolineIndirectBranches
    : SubtargetFeature<
          "retpoline-indirect-branches", "UseRetpolineIndirectBranches", "true",
          "Remove speculation of indirect branches from the generated code">;

// Deprecated umbrella feature for enabling both `retpoline-indirect-calls` and
// `retpoline-indirect-branches` above.
def FeatureRetpoline
    : SubtargetFeature<"retpoline", "DeprecatedUseRetpoline", "true",
                       "Remove speculation of indirect branches from the "
                       "generated code, either by avoiding them entirely or "
                       "lowering them with a speculation blocking construct",
                       [FeatureRetpolineIndirectCalls,
                        FeatureRetpolineIndirectBranches]>;

// Rely on external thunks for the emitted retpoline calls. This allows users
// to provide their own custom thunk definitions in highly specialized
// environments such as a kernel that does boot-time hot patching.
def FeatureRetpolineExternalThunk
    : SubtargetFeature<
          "retpoline-external-thunk", "UseRetpolineExternalThunk", "true",
          "When lowering an indirect call or branch using a `retpoline`, rely "
          "on the specified user provided thunk rather than emitting one "
          "ourselves. Only has effect when combined with some other retpoline "
          "feature", [FeatureRetpolineIndirectCalls]>;

// Mitigate LVI attacks against indirect calls/branches and call returns
def FeatureLVIControlFlowIntegrity
    : SubtargetFeature<
          "lvi-cfi", "UseLVIControlFlowIntegrity", "true",
          "Prevent indirect calls/branches from using a memory operand, and "
          "precede all indirect calls/branches from a register with an "
          "LFENCE instruction to serialize control flow. Also decompose RET "
          "instructions into a POP+LFENCE+JMP sequence.">;

// Enable SESES to mitigate speculative execution attacks
def FeatureSpeculativeExecutionSideEffectSuppression
    : SubtargetFeature<
          "seses", "UseSpeculativeExecutionSideEffectSuppression", "true",
          "Prevent speculative execution side channel timing attacks by "
          "inserting a speculation barrier before memory reads, memory writes, "
          "and conditional branches. Implies LVI Control Flow integrity.",
          [FeatureLVIControlFlowIntegrity]>;

// Mitigate LVI attacks against data loads
def FeatureLVILoadHardening
    : SubtargetFeature<
          "lvi-load-hardening", "UseLVILoadHardening", "true",
          "Insert LFENCE instructions to prevent data speculatively injected "
          "into loads from being used maliciously.">;

// Direct Move instructions.
def FeatureMOVDIRI  : SubtargetFeature<"movdiri", "HasMOVDIRI", "true",
                                       "Support movdiri instruction">;
def FeatureMOVDIR64B : SubtargetFeature<"movdir64b", "HasMOVDIR64B", "true",
                                        "Support movdir64b instruction">;

def FeatureFastBEXTR : SubtargetFeature<"fast-bextr", "HasFastBEXTR", "true",
          "Indicates that the BEXTR instruction is implemented as a single uop "
          "with good throughput">;

// Combine vector math operations with shuffles into horizontal math
// instructions if a CPU implements horizontal operations (introduced with
// SSE3) with better latency/throughput than the alternative sequence.
def FeatureFastHorizontalOps
    : SubtargetFeature<
        "fast-hops", "HasFastHorizontalOps", "true",
        "Prefer horizontal vector math instructions (haddp, phsub, etc.) over "
        "normal vector instructions with shuffles">;

def FeatureFastScalarShiftMasks
    : SubtargetFeature<
        "fast-scalar-shift-masks", "HasFastScalarShiftMasks", "true",
        "Prefer a left/right scalar logical shift pair over a shift+and pair">;

def FeatureFastVectorShiftMasks
    : SubtargetFeature<
        "fast-vector-shift-masks", "HasFastVectorShiftMasks", "true",
        "Prefer a left/right vector logical shift pair over a shift+and pair">;

def FeatureUseGLMDivSqrtCosts
    : SubtargetFeature<"use-glm-div-sqrt-costs", "UseGLMDivSqrtCosts", "true",
        "Use Goldmont specific floating point div/sqrt costs">;

// Merge branches using three-way conditional code.
def FeatureMergeToThreeWayBranch : SubtargetFeature<"merge-to-threeway-branch",
                                        "ThreewayBranchProfitable", "true",
                                        "Merge branches to a three-way "
                                        "conditional branch">;

// Enable use of alias analysis during code generation.
def FeatureUseAA : SubtargetFeature<"use-aa", "UseAA", "true",
                                    "Use alias analysis during codegen">;

// Bonnell
def ProcIntelAtom : SubtargetFeature<"", "X86ProcFamily", "IntelAtom", "">;
// Silvermont
def ProcIntelSLM  : SubtargetFeature<"", "X86ProcFamily", "IntelSLM", "">;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "X86RegisterInfo.td"
include "X86RegisterBanks.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "X86Schedule.td"
include "X86InstrInfo.td"
include "X86SchedPredicates.td"

def X86InstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// X86 Scheduler Models
//===----------------------------------------------------------------------===//

include "X86ScheduleAtom.td"
include "X86SchedSandyBridge.td"
include "X86SchedHaswell.td"
include "X86SchedBroadwell.td"
include "X86ScheduleSLM.td"
include "X86ScheduleZnver1.td"
include "X86ScheduleZnver2.td"
include "X86ScheduleBdVer2.td"
include "X86ScheduleBtVer2.td"
include "X86SchedSkylakeClient.td"
include "X86SchedSkylakeServer.td"

//===----------------------------------------------------------------------===//
// X86 Processor Feature Lists
//===----------------------------------------------------------------------===//

def ProcessorFeatures {
  // Nehalem
  list<SubtargetFeature> NHMInheritableFeatures = [FeatureX87,
                                                   FeatureCMPXCHG8B,
                                                   FeatureCMOV,
                                                   FeatureMMX,
                                                   FeatureSSE42,
                                                   FeatureFXSR,
                                                   FeatureNOPL,
                                                   Feature64Bit,
                                                   FeatureCMPXCHG16B,
                                                   FeaturePOPCNT,
                                                   FeatureLAHFSAHF,
                                                   FeatureMacroFusion,
                                                   FeatureInsertVZEROUPPER];
  list<SubtargetFeature> NHMSpecificFeatures = [];
  list<SubtargetFeature> NHMFeatures =
    !listconcat(NHMInheritableFeatures, NHMSpecificFeatures);

  // Westmere
  list<SubtargetFeature> WSMAdditionalFeatures = [FeaturePCLMUL];
  list<SubtargetFeature> WSMSpecificFeatures = [];
  list<SubtargetFeature> WSMInheritableFeatures =
    !listconcat(NHMInheritableFeatures, WSMAdditionalFeatures);
  list<SubtargetFeature> WSMFeatures =
    !listconcat(WSMInheritableFeatures, WSMSpecificFeatures);

  // Sandybridge
  list<SubtargetFeature> SNBAdditionalFeatures = [FeatureAVX,
                                                  FeatureSlowDivide64,
                                                  FeatureXSAVE,
                                                  FeatureXSAVEOPT,
                                                  FeatureSlow3OpsLEA,
                                                  FeatureFastScalarFSQRT,
                                                  FeatureFastSHLDRotate,
                                                  FeatureMergeToThreeWayBranch,
                                                  FeatureFast15ByteNOP];
  list<SubtargetFeature> SNBSpecificFeatures = [FeatureSlowUAMem32,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> SNBInheritableFeatures =
    !listconcat(WSMInheritableFeatures, SNBAdditionalFeatures);
  list<SubtargetFeature> SNBFeatures =
    !listconcat(SNBInheritableFeatures, SNBSpecificFeatures);

  // Ivybridge
  list<SubtargetFeature> IVBAdditionalFeatures = [FeatureRDRAND,
                                                  FeatureF16C,
                                                  FeatureFSGSBase];
  list<SubtargetFeature> IVBSpecificFeatures = [FeatureSlowUAMem32,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> IVBInheritableFeatures =
    !listconcat(SNBInheritableFeatures, IVBAdditionalFeatures);
  list<SubtargetFeature> IVBFeatures =
    !listconcat(IVBInheritableFeatures, IVBSpecificFeatures);

  // Haswell
  list<SubtargetFeature> HSWAdditionalFeatures = [FeatureAVX2,
                                                  FeatureBMI,
                                                  FeatureBMI2,
                                                  FeatureERMSB,
                                                  FeatureFMA,
                                                  FeatureINVPCID,
                                                  FeatureLZCNT,
                                                  FeatureMOVBE,
                                                  FeatureFastVariableShuffle];
  list<SubtargetFeature> HSWSpecificFeatures = [FeaturePOPCNTFalseDeps,
                                                FeatureLZCNTFalseDeps];
  list<SubtargetFeature> HSWInheritableFeatures =
    !listconcat(IVBInheritableFeatures, HSWAdditionalFeatures);
  list<SubtargetFeature> HSWFeatures =
    !listconcat(HSWInheritableFeatures, HSWSpecificFeatures);

  // Broadwell
  list<SubtargetFeature> BDWAdditionalFeatures = [FeatureADX,
                                                  FeatureRDSEED,
                                                  FeaturePRFCHW];
  list<SubtargetFeature> BDWSpecificFeatures = [FeaturePOPCNTFalseDeps,
                                                FeatureLZCNTFalseDeps];
  list<SubtargetFeature> BDWInheritableFeatures =
    !listconcat(HSWInheritableFeatures, BDWAdditionalFeatures);
  list<SubtargetFeature> BDWFeatures =
    !listconcat(BDWInheritableFeatures, BDWSpecificFeatures);

  // Skylake
  list<SubtargetFeature> SKLAdditionalFeatures = [FeatureAES,
                                                  FeatureXSAVEC,
                                                  FeatureXSAVES,
                                                  FeatureCLFLUSHOPT,
                                                  FeatureFastVectorFSQRT];
  list<SubtargetFeature> SKLSpecificFeatures = [FeatureHasFastGather,
                                                FeaturePOPCNTFalseDeps,
                                                FeatureSGX];
  list<SubtargetFeature> SKLInheritableFeatures =
    !listconcat(BDWInheritableFeatures, SKLAdditionalFeatures);
  list<SubtargetFeature> SKLFeatures =
    !listconcat(SKLInheritableFeatures, SKLSpecificFeatures);

  // Skylake-AVX512
  list<SubtargetFeature> SKXAdditionalFeatures = [FeatureAVX512,
                                                  FeaturePrefer256Bit,
                                                  FeatureCDI,
                                                  FeatureDQI,
                                                  FeatureBWI,
                                                  FeatureVLX,
                                                  FeaturePKU,
                                                  FeatureCLWB];
  list<SubtargetFeature> SKXSpecificFeatures = [FeatureHasFastGather,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> SKXInheritableFeatures =
    !listconcat(SKLInheritableFeatures, SKXAdditionalFeatures);
  list<SubtargetFeature> SKXFeatures =
    !listconcat(SKXInheritableFeatures, SKXSpecificFeatures);

  // Cascadelake
  list<SubtargetFeature> CLXAdditionalFeatures = [FeatureVNNI];
  list<SubtargetFeature> CLXSpecificFeatures = [FeatureHasFastGather,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> CLXInheritableFeatures =
    !listconcat(SKXInheritableFeatures, CLXAdditionalFeatures);
  list<SubtargetFeature> CLXFeatures =
    !listconcat(CLXInheritableFeatures, CLXSpecificFeatures);

  // Cooperlake
  list<SubtargetFeature> CPXAdditionalFeatures = [FeatureBF16];
  list<SubtargetFeature> CPXSpecificFeatures = [FeatureHasFastGather,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> CPXInheritableFeatures =
    !listconcat(CLXInheritableFeatures, CPXAdditionalFeatures);
  list<SubtargetFeature> CPXFeatures =
    !listconcat(CPXInheritableFeatures, CPXSpecificFeatures);

  // Cannonlake
  list<SubtargetFeature> CNLAdditionalFeatures = [FeatureAVX512,
                                                  FeaturePrefer256Bit,
                                                  FeatureCDI,
                                                  FeatureDQI,
                                                  FeatureBWI,
                                                  FeatureVLX,
                                                  FeaturePKU,
                                                  FeatureVBMI,
                                                  FeatureIFMA,
                                                  FeatureSHA,
                                                  FeatureSGX];
  list<SubtargetFeature> CNLSpecificFeatures = [FeatureHasFastGather];
  list<SubtargetFeature> CNLInheritableFeatures =
    !listconcat(SKLInheritableFeatures, CNLAdditionalFeatures);
  list<SubtargetFeature> CNLFeatures =
    !listconcat(CNLInheritableFeatures, CNLSpecificFeatures);

  // Icelake
  list<SubtargetFeature> ICLAdditionalFeatures = [FeatureBITALG,
                                                  FeatureVAES,
                                                  FeatureVBMI2,
                                                  FeatureVNNI,
                                                  FeatureVPCLMULQDQ,
                                                  FeatureVPOPCNTDQ,
                                                  FeatureGFNI,
                                                  FeatureCLWB,
                                                  FeatureRDPID];
  list<SubtargetFeature> ICLSpecificFeatures = [FeatureHasFastGather];
  list<SubtargetFeature> ICLInheritableFeatures =
    !listconcat(CNLInheritableFeatures, ICLAdditionalFeatures);
  list<SubtargetFeature> ICLFeatures =
    !listconcat(ICLInheritableFeatures, ICLSpecificFeatures);

  // Icelake Server
  list<SubtargetFeature> ICXSpecificFeatures = [FeaturePCONFIG,
                                                FeatureWBNOINVD,
                                                FeatureHasFastGather];
  list<SubtargetFeature> ICXFeatures =
    !listconcat(ICLInheritableFeatures, ICXSpecificFeatures);

  //Tigerlake
  list<SubtargetFeature> TGLAdditionalFeatures = [FeatureVP2INTERSECT,
                                                  FeatureMOVDIRI,
                                                  FeatureMOVDIR64B,
                                                  FeatureSHSTK];
  list<SubtargetFeature> TGLSpecificFeatures = [FeatureHasFastGather];
  list<SubtargetFeature> TGLInheritableFeatures =
    !listconcat(TGLAdditionalFeatures ,TGLSpecificFeatures);
  list<SubtargetFeature> TGLFeatures =
    !listconcat(ICLFeatures, TGLInheritableFeatures );

  // Atom
  list<SubtargetFeature> AtomInheritableFeatures = [FeatureX87,
                                                    FeatureCMPXCHG8B,
                                                    FeatureCMOV,
                                                    FeatureMMX,
                                                    FeatureSSSE3,
                                                    FeatureFXSR,
                                                    FeatureNOPL,
                                                    Feature64Bit,
                                                    FeatureCMPXCHG16B,
                                                    FeatureMOVBE,
                                                    FeatureSlowTwoMemOps,
                                                    FeatureLAHFSAHF,
                                                    FeatureInsertVZEROUPPER];
  list<SubtargetFeature> AtomSpecificFeatures = [ProcIntelAtom,
                                                 FeatureSlowUAMem16,
                                                 FeatureLEAForSP,
                                                 FeatureSlowDivide32,
                                                 FeatureSlowDivide64,
                                                 FeatureLEAUsesAG,
                                                 FeaturePadShortFunctions];
  list<SubtargetFeature> AtomFeatures =
    !listconcat(AtomInheritableFeatures, AtomSpecificFeatures);

  // Silvermont
  list<SubtargetFeature> SLMAdditionalFeatures = [FeatureSSE42,
                                                  FeaturePOPCNT,
                                                  FeaturePCLMUL,
                                                  FeaturePRFCHW,
                                                  FeatureSlowLEA,
                                                  FeatureSlowIncDec,
                                                  FeatureRDRAND];
  list<SubtargetFeature> SLMSpecificFeatures = [ProcIntelSLM,
                                                FeatureSlowDivide64,
                                                FeatureSlowPMULLD,
                                                FeatureFast7ByteNOP,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> SLMInheritableFeatures =
    !listconcat(AtomInheritableFeatures, SLMAdditionalFeatures);
  list<SubtargetFeature> SLMFeatures =
    !listconcat(SLMInheritableFeatures, SLMSpecificFeatures);

  // Goldmont
  list<SubtargetFeature> GLMAdditionalFeatures = [FeatureAES,
                                                  FeatureSHA,
                                                  FeatureRDSEED,
                                                  FeatureXSAVE,
                                                  FeatureXSAVEOPT,
                                                  FeatureXSAVEC,
                                                  FeatureXSAVES,
                                                  FeatureCLFLUSHOPT,
                                                  FeatureFSGSBase];
  list<SubtargetFeature> GLMSpecificFeatures = [FeatureUseGLMDivSqrtCosts,
                                                FeaturePOPCNTFalseDeps];
  list<SubtargetFeature> GLMInheritableFeatures =
    !listconcat(SLMInheritableFeatures, GLMAdditionalFeatures);
  list<SubtargetFeature> GLMFeatures =
    !listconcat(GLMInheritableFeatures, GLMSpecificFeatures);

  // Goldmont Plus
  list<SubtargetFeature> GLPAdditionalFeatures = [FeaturePTWRITE,
                                                  FeatureRDPID,
                                                  FeatureSGX];
  list<SubtargetFeature> GLPSpecificFeatures = [FeatureUseGLMDivSqrtCosts];
  list<SubtargetFeature> GLPInheritableFeatures =
    !listconcat(GLMInheritableFeatures, GLPAdditionalFeatures);
  list<SubtargetFeature> GLPFeatures =
    !listconcat(GLPInheritableFeatures, GLPSpecificFeatures);

  // Tremont
  list<SubtargetFeature> TRMAdditionalFeatures = [FeatureCLWB,
                                                  FeatureGFNI];
  list<SubtargetFeature> TRMSpecificFeatures = [FeatureUseGLMDivSqrtCosts];
  list<SubtargetFeature> TRMInheritableFeatures =
    !listconcat(GLPInheritableFeatures, TRMAdditionalFeatures);
  list<SubtargetFeature> TRMFeatures =
    !listconcat(TRMInheritableFeatures, TRMSpecificFeatures);

  // Knights Landing
  list<SubtargetFeature> KNLFeatures = [FeatureX87,
                                        FeatureCMPXCHG8B,
                                        FeatureCMOV,
                                        FeatureMMX,
                                        FeatureFXSR,
                                        FeatureNOPL,
                                        Feature64Bit,
                                        FeatureCMPXCHG16B,
                                        FeaturePOPCNT,
                                        FeatureSlowDivide64,
                                        FeaturePCLMUL,
                                        FeatureXSAVE,
                                        FeatureXSAVEOPT,
                                        FeatureLAHFSAHF,
                                        FeatureSlow3OpsLEA,
                                        FeatureSlowIncDec,
                                        FeatureAES,
                                        FeatureRDRAND,
                                        FeatureF16C,
                                        FeatureFSGSBase,
                                        FeatureAVX512,
                                        FeatureERI,
                                        FeatureCDI,
                                        FeaturePFI,
                                        FeaturePREFETCHWT1,
                                        FeatureADX,
                                        FeatureRDSEED,
                                        FeatureMOVBE,
                                        FeatureLZCNT,
                                        FeatureBMI,
                                        FeatureBMI2,
                                        FeatureFMA,
                                        FeaturePRFCHW,
                                        FeaturePreferMaskRegisters,
                                        FeatureSlowTwoMemOps,
                                        FeatureHasFastGather,
                                        FeatureSlowPMADDWD];
  // TODO Add AVX5124FMAPS/AVX5124VNNIW features
  list<SubtargetFeature> KNMFeatures =
    !listconcat(KNLFeatures, [FeatureVPOPCNTDQ]);

  // Barcelona
  list<SubtargetFeature> BarcelonaInheritableFeatures = [FeatureX87,
                                                         FeatureCMPXCHG8B,
                                                         FeatureSSE4A,
                                                         Feature3DNowA,
                                                         FeatureFXSR,
                                                         FeatureNOPL,
                                                         FeatureCMPXCHG16B,
                                                         FeaturePRFCHW,
                                                         FeatureLZCNT,
                                                         FeaturePOPCNT,
                                                         FeatureSlowSHLD,
                                                         FeatureLAHFSAHF,
                                                         FeatureCMOV,
                                                         Feature64Bit,
                                                         FeatureFastScalarShiftMasks,
                                                         FeatureInsertVZEROUPPER];
  list<SubtargetFeature> BarcelonaFeatures = BarcelonaInheritableFeatures;

  // Bobcat
  list<SubtargetFeature> BtVer1InheritableFeatures = [FeatureX87,
                                                      FeatureCMPXCHG8B,
                                                      FeatureCMOV,
                                                      FeatureMMX,
                                                      FeatureSSSE3,
                                                      FeatureSSE4A,
                                                      FeatureFXSR,
                                                      FeatureNOPL,
                                                      Feature64Bit,
                                                      FeatureCMPXCHG16B,
                                                      FeaturePRFCHW,
                                                      FeatureLZCNT,
                                                      FeaturePOPCNT,
                                                      FeatureSlowSHLD,
                                                      FeatureLAHFSAHF,
                                                      FeatureFast15ByteNOP,
                                                      FeatureFastScalarShiftMasks,
                                                      FeatureFastVectorShiftMasks];
  list<SubtargetFeature> BtVer1SpecificFeatures = [FeatureInsertVZEROUPPER];
  list<SubtargetFeature> BtVer1Features =
    !listconcat(BtVer1InheritableFeatures, BtVer1SpecificFeatures);

  // Jaguar
  list<SubtargetFeature> BtVer2AdditionalFeatures = [FeatureAVX,
                                                     FeatureAES,
                                                     FeaturePCLMUL,
                                                     FeatureBMI,
                                                     FeatureF16C,
                                                     FeatureMOVBE,
                                                     FeatureXSAVE,
                                                     FeatureXSAVEOPT];
  list<SubtargetFeature> BtVer2SpecificFeatures = [FeatureFastLZCNT,
                                                   FeatureFastBEXTR,
                                                   FeatureFastHorizontalOps];
  list<SubtargetFeature> BtVer2InheritableFeatures =
    !listconcat(BtVer1InheritableFeatures, BtVer2AdditionalFeatures);
  list<SubtargetFeature> BtVer2Features =
    !listconcat(BtVer2InheritableFeatures, BtVer2SpecificFeatures);

  // Bulldozer
  list<SubtargetFeature> BdVer1InheritableFeatures = [FeatureX87,
                                                      FeatureCMPXCHG8B,
                                                      FeatureCMOV,
                                                      FeatureXOP,
                                                      Feature64Bit,
                                                      FeatureCMPXCHG16B,
                                                      FeatureAES,
                                                      FeaturePRFCHW,
                                                      FeaturePCLMUL,
                                                      FeatureMMX,
                                                      FeatureFXSR,
                                                      FeatureNOPL,
                                                      FeatureLZCNT,
                                                      FeaturePOPCNT,
                                                      FeatureXSAVE,
                                                      FeatureLWP,
                                                      FeatureSlowSHLD,
                                                      FeatureLAHFSAHF,
                                                      FeatureFast11ByteNOP,
                                                      FeatureFastScalarShiftMasks,
                                                      FeatureBranchFusion,
                                                      FeatureInsertVZEROUPPER];
  list<SubtargetFeature> BdVer1Features = BdVer1InheritableFeatures;

  // PileDriver
  list<SubtargetFeature> BdVer2AdditionalFeatures = [FeatureF16C,
                                                     FeatureBMI,
                                                     FeatureTBM,
                                                     FeatureFMA,
                                                     FeatureFastBEXTR];
  list<SubtargetFeature> BdVer2InheritableFeatures =
    !listconcat(BdVer1InheritableFeatures, BdVer2AdditionalFeatures);
  list<SubtargetFeature> BdVer2Features = BdVer2InheritableFeatures;

  // Steamroller
  list<SubtargetFeature> BdVer3AdditionalFeatures = [FeatureXSAVEOPT,
                                                     FeatureFSGSBase];
  list<SubtargetFeature> BdVer3InheritableFeatures =
    !listconcat(BdVer2InheritableFeatures, BdVer3AdditionalFeatures);
  list<SubtargetFeature> BdVer3Features = BdVer3InheritableFeatures;

  // Excavator
  list<SubtargetFeature> BdVer4AdditionalFeatures = [FeatureAVX2,
                                                     FeatureBMI2,
                                                     FeatureMOVBE,
                                                     FeatureRDRAND,
                                                     FeatureMWAITX];
  list<SubtargetFeature> BdVer4InheritableFeatures =
    !listconcat(BdVer3InheritableFeatures, BdVer4AdditionalFeatures);
  list<SubtargetFeature> BdVer4Features = BdVer4InheritableFeatures;


  // AMD Zen Processors common ISAs
  list<SubtargetFeature> ZNFeatures = [FeatureADX,
                                       FeatureAES,
                                       FeatureAVX2,
                                       FeatureBMI,
                                       FeatureBMI2,
                                       FeatureCLFLUSHOPT,
                                       FeatureCLZERO,
                                       FeatureCMOV,
                                       Feature64Bit,
                                       FeatureCMPXCHG16B,
                                       FeatureF16C,
                                       FeatureFMA,
                                       FeatureFSGSBase,
                                       FeatureFXSR,
                                       FeatureNOPL,
                                       FeatureFastLZCNT,
                                       FeatureLAHFSAHF,
                                       FeatureLZCNT,
                                       FeatureFastBEXTR,
                                       FeatureFast15ByteNOP,
                                       FeatureBranchFusion,
                                       FeatureFastScalarShiftMasks,
                                       FeatureMMX,
                                       FeatureMOVBE,
                                       FeatureMWAITX,
                                       FeaturePCLMUL,
                                       FeaturePOPCNT,
                                       FeaturePRFCHW,
                                       FeatureRDRAND,
                                       FeatureRDSEED,
                                       FeatureSHA,
                                       FeatureSSE4A,
                                       FeatureSlowSHLD,
                                       FeatureInsertVZEROUPPER,
                                       FeatureX87,
                                       FeatureXSAVE,
                                       FeatureXSAVEC,
                                       FeatureXSAVEOPT,
                                       FeatureXSAVES];
  list<SubtargetFeature> ZN2AdditionalFeatures = [FeatureCLWB,
                                                  FeatureRDPID,
                                                  FeatureWBNOINVD];
  list<SubtargetFeature> ZN2Features =
    !listconcat(ZNFeatures, ZN2AdditionalFeatures);
}

//===----------------------------------------------------------------------===//
// X86 processors supported.
//===----------------------------------------------------------------------===//

class Proc<string Name, list<SubtargetFeature> Features>
 : ProcessorModel<Name, GenericModel, Features>;

// NOTE: CMPXCHG8B is here for legacy compatibility so that it is only disabled
// if i386/i486 is specifically requested.
def : Proc<"generic",         [FeatureX87, FeatureSlowUAMem16,
                               FeatureCMPXCHG8B, FeatureInsertVZEROUPPER]>;
def : Proc<"i386",            [FeatureX87, FeatureSlowUAMem16,
                               FeatureInsertVZEROUPPER]>;
def : Proc<"i486",            [FeatureX87, FeatureSlowUAMem16,
                               FeatureInsertVZEROUPPER]>;
def : Proc<"i586",            [FeatureX87, FeatureSlowUAMem16,
                               FeatureCMPXCHG8B, FeatureInsertVZEROUPPER]>;
def : Proc<"pentium",         [FeatureX87, FeatureSlowUAMem16,
                               FeatureCMPXCHG8B, FeatureInsertVZEROUPPER]>;
def : Proc<"pentium-mmx",     [FeatureX87, FeatureSlowUAMem16,
                               FeatureCMPXCHG8B, FeatureMMX,
                               FeatureInsertVZEROUPPER]>;

def : Proc<"i686", [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                    FeatureCMOV, FeatureInsertVZEROUPPER]>;
def : Proc<"pentiumpro", [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                          FeatureCMOV, FeatureNOPL, FeatureInsertVZEROUPPER]>;

def : Proc<"pentium2",        [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                               FeatureMMX, FeatureCMOV, FeatureFXSR,
                               FeatureNOPL, FeatureInsertVZEROUPPER]>;

foreach P = ["pentium3", "pentium3m"] in {
  def : Proc<P, [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,FeatureMMX,
                 FeatureSSE1, FeatureFXSR, FeatureNOPL, FeatureCMOV,
                 FeatureInsertVZEROUPPER]>;
}

// Enable the PostRAScheduler for SSE2 and SSE3 class cpus.
// The intent is to enable it for pentium4 which is the current default
// processor in a vanilla 32-bit clang compilation when no specific
// architecture is specified.  This generally gives a nice performance
// increase on silvermont, with largely neutral behavior on other
// contemporary large core processors.
// pentium-m, pentium4m, prescott and nocona are included as a preventative
// measure to avoid performance surprises, in case clang's default cpu
// changes slightly.

def : ProcessorModel<"pentium-m", GenericPostRAModel,
                     [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                      FeatureMMX, FeatureSSE2, FeatureFXSR, FeatureNOPL,
                      FeatureCMOV, FeatureInsertVZEROUPPER]>;

foreach P = ["pentium4", "pentium4m"] in {
//  def : ProcessorModel<P, GenericPostRAModel,
//                       [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
//                        FeatureMMX, FeatureSSE2, FeatureFXSR, FeatureNOPL,
//                        FeatureCMOV, FeatureInsertVZEROUPPER]>;

  // Since 'pentium4' is the default 32-bit CPU on Linux and Windows,
  // give it more modern tunings.
  // FIXME: This wouldn't be needed if we supported mtune.
  def : ProcessorModel<P, SandyBridgeModel,
                       [FeatureX87, FeatureCMPXCHG8B,
                        FeatureMMX, FeatureSSE2, FeatureFXSR, FeatureNOPL,
                        FeatureCMOV, FeatureInsertVZEROUPPER,
                        FeatureSlow3OpsLEA, FeatureSlowDivide64,
                        FeatureSlowIncDec, FeatureMacroFusion]>;
}

// Intel Quark.
def : Proc<"lakemont",        [FeatureInsertVZEROUPPER]>;

// Intel Core Duo.
def : ProcessorModel<"yonah", SandyBridgeModel,
                     [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                      FeatureMMX, FeatureSSE3, FeatureFXSR, FeatureNOPL,
                      FeatureCMOV, FeatureInsertVZEROUPPER]>;

// NetBurst.
def : ProcessorModel<"prescott", GenericPostRAModel,
                     [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                      FeatureMMX, FeatureSSE3, FeatureFXSR, FeatureNOPL,
                      FeatureCMOV, FeatureInsertVZEROUPPER]>;
def : ProcessorModel<"nocona", GenericPostRAModel, [
  FeatureX87,
  FeatureSlowUAMem16,
  FeatureCMPXCHG8B,
  FeatureCMOV,
  FeatureMMX,
  FeatureSSE3,
  FeatureFXSR,
  FeatureNOPL,
  Feature64Bit,
  FeatureCMPXCHG16B,
  FeatureInsertVZEROUPPER
]>;

// Intel Core 2 Solo/Duo.
def : ProcessorModel<"core2", SandyBridgeModel, [
  FeatureX87,
  FeatureSlowUAMem16,
  FeatureCMPXCHG8B,
  FeatureCMOV,
  FeatureMMX,
  FeatureSSSE3,
  FeatureFXSR,
  FeatureNOPL,
  Feature64Bit,
  FeatureCMPXCHG16B,
  FeatureLAHFSAHF,
  FeatureMacroFusion,
  FeatureInsertVZEROUPPER
]>;
def : ProcessorModel<"penryn", SandyBridgeModel, [
  FeatureX87,
  FeatureSlowUAMem16,
  FeatureCMPXCHG8B,
  FeatureCMOV,
  FeatureMMX,
  FeatureSSE41,
  FeatureFXSR,
  FeatureNOPL,
  Feature64Bit,
  FeatureCMPXCHG16B,
  FeatureLAHFSAHF,
  FeatureMacroFusion,
  FeatureInsertVZEROUPPER
]>;

// Atom CPUs.
foreach P = ["bonnell", "atom"] in {
  def : ProcessorModel<P, AtomModel, ProcessorFeatures.AtomFeatures>;
}

foreach P = ["silvermont", "slm"] in {
  def : ProcessorModel<P, SLMModel, ProcessorFeatures.SLMFeatures>;
}

def : ProcessorModel<"goldmont", SLMModel, ProcessorFeatures.GLMFeatures>;
def : ProcessorModel<"goldmont-plus", SLMModel, ProcessorFeatures.GLPFeatures>;
def : ProcessorModel<"tremont", SLMModel, ProcessorFeatures.TRMFeatures>;

// "Arrandale" along with corei3 and corei5
foreach P = ["nehalem", "corei7"] in {
  def : ProcessorModel<P, SandyBridgeModel, ProcessorFeatures.NHMFeatures>;
}

// Westmere is the corei3/i5/i7 path from nehalem to sandybridge
def : ProcessorModel<"westmere", SandyBridgeModel,
                     ProcessorFeatures.WSMFeatures>;

foreach P = ["sandybridge", "corei7-avx"] in {
  def : ProcessorModel<P, SandyBridgeModel, ProcessorFeatures.SNBFeatures>;
}

foreach P = ["ivybridge", "core-avx-i"] in {
  def : ProcessorModel<P, SandyBridgeModel, ProcessorFeatures.IVBFeatures>;
}

foreach P = ["haswell", "core-avx2"] in {
  def : ProcessorModel<P, HaswellModel, ProcessorFeatures.HSWFeatures>;
}

def : ProcessorModel<"broadwell", BroadwellModel,
                     ProcessorFeatures.BDWFeatures>;

def : ProcessorModel<"skylake", SkylakeClientModel,
                     ProcessorFeatures.SKLFeatures>;

// FIXME: define KNL scheduler model
def : ProcessorModel<"knl", HaswellModel, ProcessorFeatures.KNLFeatures>;
def : ProcessorModel<"knm", HaswellModel, ProcessorFeatures.KNMFeatures>;

foreach P = ["skylake-avx512", "skx"] in {
  def : ProcessorModel<P, SkylakeServerModel, ProcessorFeatures.SKXFeatures>;
}

def : ProcessorModel<"cascadelake", SkylakeServerModel,
                     ProcessorFeatures.CLXFeatures>;
def : ProcessorModel<"cooperlake", SkylakeServerModel,
                     ProcessorFeatures.CPXFeatures>;
def : ProcessorModel<"cannonlake", SkylakeServerModel,
                     ProcessorFeatures.CNLFeatures>;
def : ProcessorModel<"icelake-client", SkylakeServerModel,
                     ProcessorFeatures.ICLFeatures>;
def : ProcessorModel<"icelake-server", SkylakeServerModel,
                     ProcessorFeatures.ICXFeatures>;
def : ProcessorModel<"tigerlake", SkylakeServerModel,
                     ProcessorFeatures.TGLFeatures>;

// AMD CPUs.

def : Proc<"k6",   [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                    FeatureMMX, FeatureInsertVZEROUPPER]>;
def : Proc<"k6-2", [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                    Feature3DNow, FeatureInsertVZEROUPPER]>;
def : Proc<"k6-3", [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                    Feature3DNow, FeatureInsertVZEROUPPER]>;

foreach P = ["athlon", "athlon-tbird"] in {
  def : Proc<P, [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B, FeatureCMOV,
                 Feature3DNowA, FeatureNOPL, FeatureSlowSHLD,
                 FeatureInsertVZEROUPPER]>;
}

foreach P = ["athlon-4", "athlon-xp", "athlon-mp"] in {
  def : Proc<P, [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B, FeatureCMOV,
                 FeatureSSE1, Feature3DNowA, FeatureFXSR, FeatureNOPL,
                 FeatureSlowSHLD, FeatureInsertVZEROUPPER]>;
}

foreach P = ["k8", "opteron", "athlon64", "athlon-fx"] in {
  def : Proc<P, [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                 FeatureSSE2, Feature3DNowA, FeatureFXSR, FeatureNOPL,
                 Feature64Bit, FeatureSlowSHLD, FeatureCMOV,
                 FeatureFastScalarShiftMasks, FeatureInsertVZEROUPPER]>;
}

foreach P = ["k8-sse3", "opteron-sse3", "athlon64-sse3"] in {
  def : Proc<P, [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B, FeatureSSE3,
                 Feature3DNowA, FeatureFXSR, FeatureNOPL, FeatureCMPXCHG16B,
                 FeatureSlowSHLD, FeatureCMOV, Feature64Bit,
                 FeatureFastScalarShiftMasks, FeatureInsertVZEROUPPER]>;
}

foreach P = ["amdfam10", "barcelona"] in {
  def : Proc<P, ProcessorFeatures.BarcelonaFeatures>;
}

// Bobcat
def : Proc<"btver1", ProcessorFeatures.BtVer1Features>;
// Jaguar
def : ProcessorModel<"btver2", BtVer2Model, ProcessorFeatures.BtVer2Features>;

// Bulldozer
def : ProcessorModel<"bdver1", BdVer2Model, ProcessorFeatures.BdVer1Features>;
// Piledriver
def : ProcessorModel<"bdver2", BdVer2Model, ProcessorFeatures.BdVer2Features>;
// Steamroller
def : Proc<"bdver3", ProcessorFeatures.BdVer3Features>;
// Excavator
def : Proc<"bdver4", ProcessorFeatures.BdVer4Features>;

def : ProcessorModel<"znver1", Znver1Model, ProcessorFeatures.ZNFeatures>;
def : ProcessorModel<"znver2", Znver2Model, ProcessorFeatures.ZN2Features>;

def : Proc<"geode",           [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                               Feature3DNowA, FeatureInsertVZEROUPPER]>;

def : Proc<"winchip-c6",      [FeatureX87, FeatureSlowUAMem16, FeatureMMX,
                               FeatureInsertVZEROUPPER]>;
def : Proc<"winchip2",        [FeatureX87, FeatureSlowUAMem16, Feature3DNow,
                               FeatureInsertVZEROUPPER]>;
def : Proc<"c3",              [FeatureX87, FeatureSlowUAMem16, Feature3DNow,
                               FeatureInsertVZEROUPPER]>;
def : Proc<"c3-2",            [FeatureX87, FeatureSlowUAMem16, FeatureCMPXCHG8B,
                               FeatureMMX, FeatureSSE1, FeatureFXSR,
                               FeatureCMOV, FeatureInsertVZEROUPPER]>;

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
def : ProcessorModel<"x86-64", SandyBridgeModel, [
  FeatureX87,
  FeatureCMPXCHG8B,
  FeatureCMOV,
  FeatureMMX,
  FeatureSSE2,
  FeatureFXSR,
  FeatureNOPL,
  Feature64Bit,
  FeatureSlow3OpsLEA,
  FeatureSlowDivide64,
  FeatureSlowIncDec,
  FeatureMacroFusion,
  FeatureInsertVZEROUPPER
]>;

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
  let AllowRegisterRenaming = 1;
}

//===----------------------------------------------------------------------===//
// Pfm Counters
//===----------------------------------------------------------------------===//

include "X86PfmCounters.td"
