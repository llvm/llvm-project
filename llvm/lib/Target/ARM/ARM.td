//===-- ARM.td - Describe the ARM Target Machine -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// ARM Subtarget state.
//

// True if compiling for Thumb, false for ARM.
def ModeThumb             : SubtargetFeature<"thumb-mode", "IsThumb",
                                             "true", "Thumb mode">;

// True if we're using software floating point features.
def ModeSoftFloat         : SubtargetFeature<"soft-float","UseSoftFloat",
                                             "true", "Use software floating "
                                             "point features.">;


//===----------------------------------------------------------------------===//
// ARM Subtarget features.
//

// Floating Point, HW Division and Neon Support

// FP loads/stores/moves, shared between VFP and MVE (even in the integer-only
// version).
def FeatureFPRegs         : SubtargetFeature<"fpregs", "HasFPRegs", "true",
                                             "Enable FP registers">;

// 16-bit FP loads/stores/moves, shared between VFP (with the v8.2A FP16
// extension) and MVE (even in the integer-only version).
def FeatureFPRegs16       : SubtargetFeature<"fpregs16", "HasFPRegs16", "true",
                                             "Enable 16-bit FP registers",
                                             [FeatureFPRegs]>;

def FeatureFPRegs64       : SubtargetFeature<"fpregs64", "HasFPRegs64", "true",
                                             "Enable 64-bit FP registers",
                                             [FeatureFPRegs]>;

// True if the floating point unit supports double precision.
def FeatureFP64           : SubtargetFeature<"fp64", "HasFP64", "true",
                                             "Floating point unit supports "
                                             "double precision",
                                             [FeatureFPRegs64]>;

// True if subtarget has the full 32 double precision FP registers for VFPv3.
def FeatureD32            : SubtargetFeature<"d32", "HasD32", "true",
                                             "Extend FP to 32 double registers">;

/// Versions of the VFP flags restricted to single precision, or to
/// 16 d-registers, or both.
multiclass VFPver<string name, string query, string description,
                  list<SubtargetFeature> prev,
                  list<SubtargetFeature> otherimplies,
                  list<SubtargetFeature> vfp2prev = []> {
  def _D16_SP: SubtargetFeature<
    name#"d16sp", query#"D16SP", "true",
    description#" with only 16 d-registers and no double precision",
    !foreach(v, prev, !cast<SubtargetFeature>(v # "_D16_SP")) #
      !foreach(v, vfp2prev, !cast<SubtargetFeature>(v # "_SP")) #
      otherimplies>;
  def _SP: SubtargetFeature<
    name#"sp", query#"SP", "true",
    description#" with no double precision",
    !foreach(v, prev, !cast<SubtargetFeature>(v # "_SP")) #
      otherimplies # [FeatureD32, !cast<SubtargetFeature>(NAME # "_D16_SP")]>;
  def _D16: SubtargetFeature<
    name#"d16", query#"D16", "true",
    description#" with only 16 d-registers",
    !foreach(v, prev, !cast<SubtargetFeature>(v # "_D16")) #
      vfp2prev #
      otherimplies # [FeatureFP64, !cast<SubtargetFeature>(NAME # "_D16_SP")]>;
  def "": SubtargetFeature<
    name, query, "true", description,
    prev # otherimplies # [
        !cast<SubtargetFeature>(NAME # "_D16"),
        !cast<SubtargetFeature>(NAME # "_SP")]>;
}

def FeatureVFP2_SP        : SubtargetFeature<"vfp2sp", "HasVFPv2SP", "true",
                                             "Enable VFP2 instructions with "
                                             "no double precision",
                                             [FeatureFPRegs]>;

def FeatureVFP2           : SubtargetFeature<"vfp2", "HasVFPv2", "true",
                                             "Enable VFP2 instructions",
                                             [FeatureFP64, FeatureVFP2_SP]>;

defm FeatureVFP3: VFPver<"vfp3", "HasVFPv3", "Enable VFP3 instructions",
                         [], [], [FeatureVFP2]>;

def FeatureNEON           : SubtargetFeature<"neon", "HasNEON", "true",
                                             "Enable NEON instructions",
                                             [FeatureVFP3]>;

// True if subtarget supports half-precision FP conversions.
def FeatureFP16           : SubtargetFeature<"fp16", "HasFP16", "true",
                                             "Enable half-precision "
                                             "floating point">;

defm FeatureVFP4: VFPver<"vfp4", "HasVFPv4", "Enable VFP4 instructions",
                         [FeatureVFP3], [FeatureFP16]>;

defm FeatureFPARMv8: VFPver<"fp-armv8", "HasFPARMv8", "Enable ARMv8 FP",
                         [FeatureVFP4], []>;

// True if subtarget supports half-precision FP operations.
def FeatureFullFP16       : SubtargetFeature<"fullfp16", "HasFullFP16", "true",
                                             "Enable full half-precision "
                                             "floating point",
                                             [FeatureFPARMv8_D16_SP, FeatureFPRegs16]>;

// True if subtarget supports half-precision FP fml operations.
def FeatureFP16FML        : SubtargetFeature<"fp16fml", "HasFP16FML", "true",
                                             "Enable full half-precision "
                                             "floating point fml instructions",
                                             [FeatureFullFP16]>;

// True if subtarget supports [su]div in Thumb mode.
def FeatureHWDivThumb     : SubtargetFeature<"hwdiv",
                                             "HasDivideInThumbMode", "true",
                                             "Enable divide instructions in Thumb">;

// True if subtarget supports [su]div in ARM mode.
def FeatureHWDivARM       : SubtargetFeature<"hwdiv-arm",
                                             "HasDivideInARMMode", "true",
                                             "Enable divide instructions in ARM mode">;

// Atomic Support

// True if the subtarget supports DMB / DSB data barrier instructions.
def FeatureDB             : SubtargetFeature<"db", "HasDataBarrier", "true",
                                             "Has data barrier (dmb/dsb) instructions">;

// True if the subtarget supports CLREX instructions.
def FeatureV7Clrex        : SubtargetFeature<"v7clrex", "HasV7Clrex", "true",
                                             "Has v7 clrex instruction">;

// True if the subtarget supports DFB data barrier instruction.
def FeatureDFB  : SubtargetFeature<"dfb", "HasFullDataBarrier", "true",
                                   "Has full data barrier (dfb) instruction">;

// True if the subtarget supports v8 atomics (LDA/LDAEX etc) instructions.
def FeatureAcquireRelease : SubtargetFeature<"acquire-release",
                                             "HasAcquireRelease", "true",
                                             "Has v8 acquire/release (lda/ldaex "
                                             " etc) instructions">;


// True if floating point compare + branch is slow.
def FeatureSlowFPBrcc     : SubtargetFeature<"slow-fp-brcc", "IsFPBrccSlow", "true",
                                             "FP compare + branch is slow">;

// True if the processor supports the Performance Monitor Extensions. These
// include a generic cycle-counter as well as more fine-grained (often
// implementation-specific) events.
def FeaturePerfMon        : SubtargetFeature<"perfmon", "HasPerfMon", "true",
                                             "Enable support for Performance "
                                             "Monitor extensions">;


// TrustZone Security Extensions

// True if processor supports TrustZone security extensions.
def FeatureTrustZone      : SubtargetFeature<"trustzone", "HasTrustZone", "true",
                                             "Enable support for TrustZone "
                                             "security extensions">;

// True if processor supports ARMv8-M Security Extensions.
def Feature8MSecExt       : SubtargetFeature<"8msecext", "Has8MSecExt", "true",
                                             "Enable support for ARMv8-M "
                                             "Security Extensions">;

// True if processor supports SHA1 and SHA256.
def FeatureSHA2           : SubtargetFeature<"sha2", "HasSHA2", "true",
                                             "Enable SHA1 and SHA256 support", [FeatureNEON]>;

def FeatureAES            : SubtargetFeature<"aes", "HasAES", "true",
                                             "Enable AES support", [FeatureNEON]>;

// True if processor supports Cryptography extensions.
def FeatureCrypto         : SubtargetFeature<"crypto", "HasCrypto", "true",
                                             "Enable support for "
                                             "Cryptography extensions",
                                             [FeatureNEON, FeatureSHA2, FeatureAES]>;

// True if processor supports CRC instructions.
def FeatureCRC            : SubtargetFeature<"crc", "HasCRC", "true",
                                             "Enable support for CRC instructions">;

// True if the ARMv8.2A dot product instructions are supported.
def FeatureDotProd        : SubtargetFeature<"dotprod", "HasDotProd", "true",
                                             "Enable support for dot product instructions",
                                             [FeatureNEON]>;

// True if the processor supports RAS extensions.
// Not to be confused with FeatureHasRetAddrStack (return address stack).
def FeatureRAS            : SubtargetFeature<"ras", "HasRAS", "true",
                                             "Enable Reliability, Availability "
                                             "and Serviceability extensions">;

// Fast computation of non-negative address offsets.
// True if processor does positive address offset computation faster.
def FeatureFPAO           : SubtargetFeature<"fpao", "HasFPAO", "true",
                                             "Enable fast computation of "
                                             "positive address offsets">;

// Fast execution of AES crypto operations.
// True if processor executes back to back AES instruction pairs faster.
def FeatureFuseAES        : SubtargetFeature<"fuse-aes", "HasFuseAES", "true",
                                             "CPU fuses AES crypto operations">;

// Fast execution of bottom and top halves of literal generation.
// True if processor executes back to back bottom and top halves of literal generation faster.
def FeatureFuseLiterals   : SubtargetFeature<"fuse-literals", "HasFuseLiterals", "true",
                                             "CPU fuses literal generation operations">;

// The way of reading thread pointer.
// True if read thread pointer from coprocessor register.
def FeatureReadTp :  SubtargetFeature<"read-tp-hard", "IsReadTPHard", "true",
                                      "Reading thread pointer from register">;

// Cyclone can zero VFP registers in 0 cycles.
// True if the instructions "vmov.i32 d0, #0" and "vmov.i32 q0, #0" are
// particularly effective at zeroing a VFP register.
def FeatureZCZeroing      : SubtargetFeature<"zcz", "HasZeroCycleZeroing", "true",
                                             "Has zero-cycle zeroing instructions">;

// Whether it is profitable to unpredicate certain instructions during if-conversion.
// True if if conversion may decide to leave some instructions unpredicated.
def FeatureProfUnpredicate : SubtargetFeature<"prof-unpr",
                                              "IsProfitableToUnpredicate", "true",
                                              "Is profitable to unpredicate">;

// Some targets (e.g. Swift) have microcoded VGETLNi32.
// True if VMOV will be favored over VGETLNi32.
def FeatureSlowVGETLNi32  : SubtargetFeature<"slow-vgetlni32",
                                             "HasSlowVGETLNi32", "true",
                                             "Has slow VGETLNi32 - prefer VMOV">;

// Some targets (e.g. Swift) have microcoded VDUP32.
// True if VMOV will be favored over VDUP.
def FeatureSlowVDUP32     : SubtargetFeature<"slow-vdup32", "HasSlowVDUP32",
                                             "true",
                                             "Has slow VDUP32 - prefer VMOV">;

// Some targets (e.g. Cortex-A9) prefer VMOVSR to VMOVDRR even when using NEON
// for scalar FP, as this allows more effective execution domain optimization.
// True if VMOVSR will be favored over VMOVDRR.
def FeaturePreferVMOVSR   : SubtargetFeature<"prefer-vmovsr", "PreferVMOVSR",
                                             "true", "Prefer VMOVSR">;

// Swift has ISHST barriers compatible with Atomic Release semantics but weaker
// than ISH.
// True if ISHST barriers will be used for Release semantics.
def FeaturePrefISHSTBarrier : SubtargetFeature<"prefer-ishst", "PreferISHSTBarriers",
                                               "true", "Prefer ISHST barriers">;

// Some targets (e.g. Cortex-A9) have muxed AGU and NEON/FPU.
// True if the AGU and NEON/FPU units are multiplexed.
def FeatureMuxedUnits     : SubtargetFeature<"muxed-units", "HasMuxedUnits",
                                             "true",
                                             "Has muxed AGU and NEON/FPU">;

// Whether VLDM/VSTM starting with odd register number need more microops
// than single VLDRS.
// True if a VLDM/VSTM starting with an odd register number is considered to
// take more microops than single VLDRS/VSTRS.
def FeatureSlowOddRegister : SubtargetFeature<"slow-odd-reg", "HasSlowOddRegister",
                                              "true", "VLDM/VSTM starting "
                                              "with an odd register is slow">;

// Some targets have a renaming dependency when loading into D subregisters.
// True if loading into a D subregister will be penalized.
def FeatureSlowLoadDSubreg : SubtargetFeature<"slow-load-D-subreg",
                                              "HasSlowLoadDSubregister", "true",
                                              "Loading into D subregs is slow">;

// True if use a wider stride when allocating VFP registers.
def FeatureUseWideStrideVFP : SubtargetFeature<"wide-stride-vfp",
                                               "UseWideStrideVFP", "true",
                                               "Use a wide stride when allocating VFP registers">;

// Some targets (e.g. Cortex-A15) never want VMOVS to be widened to VMOVD.
// True if VMOVS will never be widened to VMOVD.
def FeatureDontWidenVMOVS : SubtargetFeature<"dont-widen-vmovs",
                                             "DontWidenVMOVS", "true",
                                             "Don't widen VMOVS to VMOVD">;

// Some targets (e.g. Cortex-A15) prefer to avoid mixing operations on different
// VFP register widths.
// True if splat a register between VFP and NEON instructions.
def FeatureSplatVFPToNeon : SubtargetFeature<"splat-vfp-neon",
                                             "UseSplatVFPToNeon", "true",
                                             "Splat register from VFP to NEON",
                                             [FeatureDontWidenVMOVS]>;

// Whether or not it is profitable to expand VFP/NEON MLA/MLS instructions.
// True if run the MLx expansion pass.
def FeatureExpandMLx      : SubtargetFeature<"expand-fp-mlx",
                                             "ExpandMLx", "true",
                                             "Expand VFP/NEON MLA/MLS instructions">;

// Some targets have special RAW hazards for VFP/NEON VMLA/VMLS.
// True if VFP/NEON VMLA/VMLS have special RAW hazards.
def FeatureHasVMLxHazards : SubtargetFeature<"vmlx-hazards", "HasVMLxHazards",
                                             "true", "Has VMLx hazards">;

// Some targets (e.g. Cortex-A9) want to convert VMOVRS, VMOVSR and VMOVS from
// VFP to NEON, as an execution domain optimization.
// True if VMOVRS, VMOVSR and VMOVS will be converted from VFP to NEON.
def FeatureNEONForFPMovs  : SubtargetFeature<"neon-fpmovs",
                                             "UseNEONForFPMovs", "true",
                                             "Convert VMOVSR, VMOVRS, "
                                             "VMOVS to NEON">;

// Some processors benefit from using NEON instructions for scalar
// single-precision FP operations. This affects instruction selection and should
// only be enabled if the handling of denormals is not important.
// Use the method useNEONForSinglePrecisionFP() to determine if NEON should actually be used.
def FeatureNEONForFP      : SubtargetFeature<"neonfp",
                                             "HasNEONForFP",
                                             "true",
                                             "Use NEON for single precision FP">;

// On some processors, VLDn instructions that access unaligned data take one
// extra cycle. Take that into account when computing operand latencies.
// True if VLDn instructions take an extra cycle for unaligned accesses.
def FeatureCheckVLDnAlign : SubtargetFeature<"vldn-align", "CheckVLDnAccessAlignment",
                                             "true",
                                             "Check for VLDn unaligned access">;

// Some processors have a nonpipelined VFP coprocessor.
// True if VFP instructions are not pipelined.
def FeatureNonpipelinedVFP : SubtargetFeature<"nonpipelined-vfp",
                                              "NonpipelinedVFP", "true",
                                              "VFP instructions are not pipelined">;

// Some processors have FP multiply-accumulate instructions that don't
// play nicely with other VFP / NEON instructions, and it's generally better
// to just not use them.
// If the VFP2 / NEON instructions are available, indicates
// whether the FP VML[AS] instructions are slow (if so, don't use them).
def FeatureHasSlowFPVMLx  : SubtargetFeature<"slowfpvmlx", "SlowFPVMLx", "true",
                                             "Disable VFP / NEON MAC instructions">;

// VFPv4 added VFMA instructions that can similarly be fast or slow.
// If the VFP4 / NEON instructions are available, indicates
// whether the FP VFM[AS] instructions are slow (if so, don't use them).
def FeatureHasSlowFPVFMx  : SubtargetFeature<"slowfpvfmx", "SlowFPVFMx", "true",
                                             "Disable VFP / NEON FMA instructions">;

// Cortex-A8 / A9 Advanced SIMD has multiplier accumulator forwarding.
/// True if NEON has special multiplier accumulator
/// forwarding to allow mul + mla being issued back to back.
def FeatureVMLxForwarding : SubtargetFeature<"vmlx-forwarding",
                                             "HasVMLxForwarding", "true",
                                             "Has multiplier accumulator forwarding">;

// Disable 32-bit to 16-bit narrowing for experimentation.
// True if codegen would prefer 32-bit Thumb instructions over 16-bit ones.
def FeaturePref32BitThumb : SubtargetFeature<"32bit", "Prefers32BitThumb", "true",
                                             "Prefer 32-bit Thumb instrs">;

def FeaturePrefLoopAlign32 : SubtargetFeature<"loop-align", "PrefLoopLogAlignment","2",
                                              "Prefer 32-bit alignment for loops">;

def FeatureMVEVectorCostFactor1 : SubtargetFeature<"mve1beat", "MVEVectorCostFactor", "1",
                        "Model MVE instructions as a 1 beat per tick architecture">;

def FeatureMVEVectorCostFactor2 : SubtargetFeature<"mve2beat", "MVEVectorCostFactor", "2",
                        "Model MVE instructions as a 2 beats per tick architecture">;

def FeatureMVEVectorCostFactor4 : SubtargetFeature<"mve4beat", "MVEVectorCostFactor", "4",
                        "Model MVE instructions as a 4 beats per tick architecture">;

/// Some instructions update CPSR partially, which can add false dependency for
/// out-of-order implementation, e.g. Cortex-A9, unless each individual bit is
/// mapped to a separate physical register. Avoid partial CPSR update for these
/// processors.
/// True if codegen would avoid using instructions
/// that partially update CPSR and add false dependency on the previous
/// CPSR setting instruction.
def FeatureAvoidPartialCPSR : SubtargetFeature<"avoid-partial-cpsr",
                                               "AvoidCPSRPartialUpdate", "true",
                                 "Avoid CPSR partial update for OOO execution">;

/// Disable +1 predication cost for instructions updating CPSR.
/// Enabled for Cortex-A57.
/// True if disable +1 predication cost for instructions updating CPSR. Enabled for Cortex-A57.
def FeatureCheapPredicableCPSR : SubtargetFeature<"cheap-predicable-cpsr",
                                                  "CheapPredicableCPSRDef",
                                                  "true",
                  "Disable +1 predication cost for instructions updating CPSR">;

// True if codegen should avoid using flag setting movs with shifter operand (i.e. asr, lsl, lsr).
def FeatureAvoidMOVsShOp  : SubtargetFeature<"avoid-movs-shop",
                                             "AvoidMOVsShifterOperand", "true",
                                             "Avoid movs instructions with "
                                             "shifter operand">;

// Some processors perform return stack prediction. CodeGen should avoid issue
// "normal" call instructions to callees which do not return.
def FeatureHasRetAddrStack : SubtargetFeature<"ret-addr-stack",
                                              "HasRetAddrStack", "true",
                                              "Has return address stack">;

// Some processors have no branch predictor, which changes the expected cost of
// taking a branch which affects the choice of whether to use predicated
// instructions.
// True if the subtarget has a branch predictor. Having
// a branch predictor or not changes the expected cost of taking a branch
// which affects the choice of whether to use predicated instructions.
def FeatureHasNoBranchPredictor : SubtargetFeature<"no-branch-predictor",
                                                   "HasBranchPredictor", "false",
                                                   "Has no branch predictor">;

/// DSP extension.
/// True if the subtarget supports the DSP (saturating arith and such) instructions.
def FeatureDSP            : SubtargetFeature<"dsp", "HasDSP", "true",
                                             "Supports DSP instructions in "
                                             "ARM and/or Thumb2">;

// True if the subtarget supports Multiprocessing extension (ARMv7 only).
def FeatureMP             : SubtargetFeature<"mp", "HasMPExtension", "true",
                                        "Supports Multiprocessing extension">;

// Virtualization extension - requires HW divide (ARMv7-AR ARMARM - 4.4.8).
def FeatureVirtualization : SubtargetFeature<"virtualization",
                                             "HasVirtualization", "true",
                                             "Supports Virtualization extension",
                                             [FeatureHWDivThumb, FeatureHWDivARM]>;

// Special TRAP encoding for NaCl, which looks like a TRAP in Thumb too.
// See ARMInstrInfo.td for details.
// True if NaCl TRAP instruction is generated instead of the regular TRAP.
def FeatureNaClTrap       : SubtargetFeature<"nacl-trap", "UseNaClTrap", "true",
                                             "NaCl trap">;

// True if the subtarget disallows unaligned memory
// accesses for some types.  For details, see
// ARMTargetLowering::allowsMisalignedMemoryAccesses().
def FeatureStrictAlign    : SubtargetFeature<"strict-align",
                                             "StrictAlign", "true",
                                             "Disallow all unaligned memory "
                                             "access">;

// Generate calls via indirect call instructions.
def FeatureLongCalls      : SubtargetFeature<"long-calls", "GenLongCalls", "true",
                                             "Generate calls via indirect call "
                                             "instructions">;

// Generate code that does not contain data access to code sections.
def FeatureExecuteOnly    : SubtargetFeature<"execute-only",
                                             "GenExecuteOnly", "true",
                                             "Enable the generation of "
                                             "execute only code.">;

// True if R9 is not available as a general purpose register.
def FeatureReserveR9      : SubtargetFeature<"reserve-r9", "ReserveR9", "true",
                                             "Reserve R9, making it unavailable"
                                             " as GPR">;

// True if MOVT / MOVW pairs are not used for materialization of
// 32-bit imms (including global addresses).
def FeatureNoMovt         : SubtargetFeature<"no-movt", "NoMovt", "true",
                                             "Don't use movt/movw pairs for "
                                             "32-bit imms">;

/// Implicitly convert an instruction to a different one if its immediates
/// cannot be encoded. For example, ADD r0, r1, #FFFFFFFF -> SUB r0, r1, #1.
def FeatureNoNegativeImmediates
                          : SubtargetFeature<"no-neg-immediates",
                                             "NegativeImmediates", "false",
                                             "Convert immediates and instructions "
                                             "to their negated or complemented "
                                             "equivalent when the immediate does "
                                             "not fit in the encoding.">;

// Use the MachineScheduler for instruction scheduling for the subtarget.
def FeatureUseMISched: SubtargetFeature<"use-misched", "UseMISched", "true",
                                        "Use the MachineScheduler">;

// Use the MachinePipeliner for instruction scheduling for the subtarget.
def FeatureUseMIPipeliner: SubtargetFeature<"use-mipipeliner", "UseMIPipeliner", "true",
                                            "Use the MachinePipeliner">;

// False if scheduling should happen again after register allocation.
def FeatureNoPostRASched : SubtargetFeature<"disable-postra-scheduler",
    "DisablePostRAScheduler", "true",
    "Don't schedule again after register allocation">;

// Armv8.5-A extensions

// Has speculation barrier.
def FeatureSB       : SubtargetFeature<"sb", "HasSB", "true",
  "Enable v8.5a Speculation Barrier" >;

// Armv8.6-A extensions

// True if subtarget supports BFloat16 floating point operations.
def FeatureBF16     : SubtargetFeature<"bf16", "HasBF16", "true",
  "Enable support for BFloat16 instructions",  [FeatureNEON]>;

// True if subtarget supports 8-bit integer matrix multiply.
def FeatureMatMulInt8 : SubtargetFeature<"i8mm", "HasMatMulInt8",
    "true", "Enable Matrix Multiply Int8 Extension", [FeatureNEON]>;

// Armv8.1-M extensions

// True if the processor supports the Low Overhead Branch extension.
def FeatureLOB            : SubtargetFeature<"lob", "HasLOB", "true",
                                             "Enable Low Overhead Branch "
                                             "extensions">;

// Mitigate against the cve-2021-35465 security vulnurability.
def FeatureFixCMSE_CVE_2021_35465 : SubtargetFeature<"fix-cmse-cve-2021-35465",
                                        "FixCMSE_CVE_2021_35465", "true",
                                        "Mitigate against the cve-2021-35465 "
                                        "security vulnurability">;

def FeaturePACBTI         : SubtargetFeature<"pacbti", "HasPACBTI", "true",
                                             "Enable Pointer Authentication and Branch "
                                             "Target Identification">;

/// Don't place a BTI instruction after return-twice constructs (setjmp).
def FeatureNoBTIAtReturnTwice : SubtargetFeature<"no-bti-at-return-twice",
                                                 "NoBTIAtReturnTwice", "true",
                                                 "Don't place a BTI instruction "
                                                 "after a return-twice">;

def FeatureFixCortexA57AES1742098 : SubtargetFeature<"fix-cortex-a57-aes-1742098",
  "FixCortexA57AES1742098", "true",
  "Work around Cortex-A57 Erratum 1742098 / Cortex-A72 Erratum 1655431 (AES)">;

def FeatureAAPCSFrameChain : SubtargetFeature<"aapcs-frame-chain",
                                              "CreateAAPCSFrameChain", "true",
                                              "Create an AAPCS compliant frame chain">;

def FeatureAAPCSFrameChainLeaf : SubtargetFeature<"aapcs-frame-chain-leaf",
                                                  "CreateAAPCSFrameChainLeaf", "true",
                                                  "Create an AAPCS compliant frame chain "
                                                  "for leaf functions",
                                                  [FeatureAAPCSFrameChain]>;

//===----------------------------------------------------------------------===//
// ARM architecture class
//

// A-series ISA
def FeatureAClass : SubtargetFeature<"aclass", "ARMProcClass", "AClass",
                                     "Is application profile ('A' series)">;

// R-series ISA
def FeatureRClass : SubtargetFeature<"rclass", "ARMProcClass", "RClass",
                                     "Is realtime profile ('R' series)">;

// M-series ISA
def FeatureMClass : SubtargetFeature<"mclass", "ARMProcClass", "MClass",
                                     "Is microcontroller profile ('M' series)">;

// True if Thumb2 instructions are supported.
def FeatureThumb2 : SubtargetFeature<"thumb2", "HasThumb2", "true",
                                     "Enable Thumb2 instructions">;

// True if subtarget does not support ARM mode execution.
def FeatureNoARM  : SubtargetFeature<"noarm", "NoARM", "true",
                                     "Does not support ARM mode execution">;

//===----------------------------------------------------------------------===//
// ARM ISAa.
//
// Specify whether target support specific ARM ISA variants.

def HasV4TOps   : SubtargetFeature<"v4t", "HasV4TOps", "true",
                                   "Support ARM v4T instructions">;

def HasV5TOps   : SubtargetFeature<"v5t", "HasV5TOps", "true",
                                   "Support ARM v5T instructions",
                                   [HasV4TOps]>;

def HasV5TEOps  : SubtargetFeature<"v5te", "HasV5TEOps", "true",
                                   "Support ARM v5TE, v5TEj, and "
                                   "v5TExp instructions",
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
                                   [HasV6T2Ops, FeatureV7Clrex]>;

def HasV8MMainlineOps :
                  SubtargetFeature<"v8m.main", "HasV8MMainlineOps", "true",
                                   "Support ARM v8M Mainline instructions",
                                   [HasV7Ops]>;

def HasV8Ops    : SubtargetFeature<"v8", "HasV8Ops", "true",
                                   "Support ARM v8 instructions",
                                   [HasV7Ops, FeaturePerfMon, FeatureAcquireRelease]>;

def HasV8_1aOps : SubtargetFeature<"v8.1a", "HasV8_1aOps", "true",
                                   "Support ARM v8.1a instructions",
                                   [HasV8Ops]>;

def HasV8_2aOps : SubtargetFeature<"v8.2a", "HasV8_2aOps", "true",
                                   "Support ARM v8.2a instructions",
                                   [HasV8_1aOps]>;

def HasV8_3aOps   : SubtargetFeature<"v8.3a", "HasV8_3aOps", "true",
                                   "Support ARM v8.3a instructions",
                                   [HasV8_2aOps]>;

def HasV8_4aOps   : SubtargetFeature<"v8.4a", "HasV8_4aOps", "true",
                                   "Support ARM v8.4a instructions",
                                   [HasV8_3aOps, FeatureDotProd]>;

def HasV8_5aOps   : SubtargetFeature<"v8.5a", "HasV8_5aOps", "true",
                                   "Support ARM v8.5a instructions",
                                   [HasV8_4aOps, FeatureSB]>;

def HasV8_6aOps   : SubtargetFeature<"v8.6a", "HasV8_6aOps", "true",
                                   "Support ARM v8.6a instructions",
                                   [HasV8_5aOps, FeatureBF16,
                                    FeatureMatMulInt8]>;

def HasV8_7aOps   : SubtargetFeature<"v8.7a", "HasV8_7aOps", "true",
                                   "Support ARM v8.7a instructions",
                                   [HasV8_6aOps]>;

def HasV8_8aOps   : SubtargetFeature<"v8.8a", "HasV8_8aOps", "true",
                                   "Support ARM v8.8a instructions",
                                   [HasV8_7aOps]>;

def HasV9_0aOps   : SubtargetFeature<"v9a", "HasV9_0aOps", "true",
                                   "Support ARM v9a instructions",
                                   [HasV8_5aOps]>;

def HasV9_1aOps   : SubtargetFeature<"v9.1a", "HasV9_1aOps", "true",
                                   "Support ARM v9.1a instructions",
                                   [HasV8_6aOps, HasV9_0aOps]>;

def HasV9_2aOps   : SubtargetFeature<"v9.2a", "HasV9_2aOps", "true",
                                   "Support ARM v9.2a instructions",
                                   [HasV8_7aOps, HasV9_1aOps]>;

def HasV9_3aOps   : SubtargetFeature<"v9.3a", "HasV9_3aOps", "true",
                                   "Support ARM v9.3a instructions",
                                   [HasV8_8aOps, HasV9_2aOps]>;

def HasV8_1MMainlineOps : SubtargetFeature<
               "v8.1m.main", "HasV8_1MMainlineOps", "true",
               "Support ARM v8-1M Mainline instructions",
               [HasV8MMainlineOps]>;
def HasMVEIntegerOps : SubtargetFeature<
               "mve", "HasMVEIntegerOps", "true",
               "Support M-Class Vector Extension with integer ops",
               [HasV8_1MMainlineOps, FeatureDSP, FeatureFPRegs16, FeatureFPRegs64]>;
def HasMVEFloatOps : SubtargetFeature<
               "mve.fp", "HasMVEFloatOps", "true",
               "Support M-Class Vector Extension with integer and floating ops",
               [HasMVEIntegerOps, FeatureFPARMv8_D16_SP, FeatureFullFP16]>;

def HasCDEOps : SubtargetFeature<"cde", "HasCDEOps", "true",
                                 "Support CDE instructions",
                                 [HasV8MMainlineOps]>;

foreach i = {0-7} in
    def FeatureCoprocCDE#i : SubtargetFeature<"cdecp"#i,
                                              "CoprocCDE["#i#"]", "true",
                                              "Coprocessor "#i#" ISA is CDEv1",
                                              [HasCDEOps]>;

//===----------------------------------------------------------------------===//
// Control codegen mitigation against Straight Line Speculation vulnerability.
//===----------------------------------------------------------------------===//

/// Harden against Straight Line Speculation for Returns and Indirect Branches.
def FeatureHardenSlsRetBr : SubtargetFeature<"harden-sls-retbr",
  "HardenSlsRetBr", "true",
  "Harden against straight line speculation across RETurn and BranchRegister "
  "instructions">;
/// Harden against Straight Line Speculation for indirect calls.
def FeatureHardenSlsBlr : SubtargetFeature<"harden-sls-blr",
  "HardenSlsBlr", "true",
  "Harden against straight line speculation across indirect calls">;
/// Generate thunk code for SLS mitigation in the normal text section.
def FeatureHardenSlsNoComdat : SubtargetFeature<"harden-sls-nocomdat",
  "HardenSlsNoComdat", "true",
  "Generate thunk code for SLS mitigation in the normal text section">;

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
def ProcA32     : SubtargetFeature<"a32", "ARMProcFamily", "CortexA32",
                                   "Cortex-A32 ARM processors", []>;
def ProcA35     : SubtargetFeature<"a35", "ARMProcFamily", "CortexA35",
                                   "Cortex-A35 ARM processors", []>;
def ProcA53     : SubtargetFeature<"a53", "ARMProcFamily", "CortexA53",
                                   "Cortex-A53 ARM processors", []>;
def ProcA55     : SubtargetFeature<"a55", "ARMProcFamily", "CortexA55",
                                   "Cortex-A55 ARM processors", []>;
def ProcA57     : SubtargetFeature<"a57", "ARMProcFamily", "CortexA57",
                                   "Cortex-A57 ARM processors", []>;
def ProcA72     : SubtargetFeature<"a72", "ARMProcFamily", "CortexA72",
                                   "Cortex-A72 ARM processors", []>;
def ProcA73     : SubtargetFeature<"a73", "ARMProcFamily", "CortexA73",
                                   "Cortex-A73 ARM processors", []>;
def ProcA75     : SubtargetFeature<"a75", "ARMProcFamily", "CortexA75",
                                   "Cortex-A75 ARM processors", []>;
def ProcA76     : SubtargetFeature<"a76", "ARMProcFamily", "CortexA76",
                                   "Cortex-A76 ARM processors", []>;
def ProcA77     : SubtargetFeature<"a77", "ARMProcFamily", "CortexA77",
                                   "Cortex-A77 ARM processors", []>;
def ProcA78     : SubtargetFeature<"cortex-a78", "ARMProcFamily", "CortexA78",
                                   "Cortex-A78 ARM processors", []>;
def ProcA78C    : SubtargetFeature<"a78c", "ARMProcFamily", "CortexA78C",
                                   "Cortex-A78C ARM processors", []>;
def ProcA710    : SubtargetFeature<"cortex-a710", "ARMProcFamily",
                                   "CortexA710", "Cortex-A710 ARM processors", []>;
def ProcX1      : SubtargetFeature<"cortex-x1", "ARMProcFamily", "CortexX1",
                                   "Cortex-X1 ARM processors", []>;
def ProcX1C     : SubtargetFeature<"cortex-x1c", "ARMProcFamily", "CortexX1C",
                                   "Cortex-X1C ARM processors", []>;

def ProcV1      : SubtargetFeature<"neoverse-v1", "ARMProcFamily",
                                   "NeoverseV1", "Neoverse-V1 ARM processors", []>;

def ProcKrait   : SubtargetFeature<"krait", "ARMProcFamily", "Krait",
                                   "Qualcomm Krait processors", []>;
def ProcKryo    : SubtargetFeature<"kryo", "ARMProcFamily", "Kryo",
                                   "Qualcomm Kryo processors", []>;
def ProcSwift   : SubtargetFeature<"swift", "ARMProcFamily", "Swift",
                                   "Swift ARM processors", []>;

def ProcExynos  : SubtargetFeature<"exynos", "ARMProcFamily", "Exynos",
                                   "Samsung Exynos processors",
                                   [FeatureZCZeroing,
                                    FeatureUseWideStrideVFP,
                                    FeatureSplatVFPToNeon,
                                    FeatureSlowVGETLNi32,
                                    FeatureSlowVDUP32,
                                    FeatureSlowFPBrcc,
                                    FeatureProfUnpredicate,
                                    FeatureHWDivThumb,
                                    FeatureHWDivARM,
                                    FeatureHasSlowFPVMLx,
                                    FeatureHasSlowFPVFMx,
                                    FeatureHasRetAddrStack,
                                    FeatureFuseLiterals,
                                    FeatureFuseAES,
                                    FeatureExpandMLx,
                                    FeatureCrypto,
                                    FeatureCRC]>;

def ProcR4      : SubtargetFeature<"r4", "ARMProcFamily", "CortexR4",
                                   "Cortex-R4 ARM processors", []>;
def ProcR5      : SubtargetFeature<"r5", "ARMProcFamily", "CortexR5",
                                   "Cortex-R5 ARM processors", []>;
def ProcR7      : SubtargetFeature<"r7", "ARMProcFamily", "CortexR7",
                                   "Cortex-R7 ARM processors", []>;
def ProcR52     : SubtargetFeature<"r52", "ARMProcFamily", "CortexR52",
                                   "Cortex-R52 ARM processors", []>;

def ProcM3      : SubtargetFeature<"m3", "ARMProcFamily", "CortexM3",
                                   "Cortex-M3 ARM processors", []>;
def ProcM7      : SubtargetFeature<"m7", "ARMProcFamily", "CortexM7",
                                   "Cortex-M7 ARM processors", []>;

//===----------------------------------------------------------------------===//
// ARM Helper classes.
//

class Architecture<string fname, string aname, list<SubtargetFeature> features>
  : SubtargetFeature<fname, "ARMArch", aname,
                     !strconcat(aname, " architecture"), features>;

class ProcNoItin<string Name, list<SubtargetFeature> Features>
  : Processor<Name, NoItineraries, Features>;


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

def ARMv6     : Architecture<"armv6",     "ARMv6",    [HasV6Ops,
                                                       FeatureDSP]>;

def ARMv6t2   : Architecture<"armv6t2",   "ARMv6t2",  [HasV6T2Ops,
                                                       FeatureDSP]>;

def ARMv6k    : Architecture<"armv6k",    "ARMv6k",   [HasV6KOps]>;

def ARMv6kz   : Architecture<"armv6kz",   "ARMv6kz",  [HasV6KOps,
                                                       FeatureTrustZone]>;

def ARMv6m    : Architecture<"armv6-m",   "ARMv6m",   [HasV6MOps,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureMClass,
                                                       FeatureStrictAlign]>;

def ARMv6sm   : Architecture<"armv6s-m",  "ARMv6sm",  [HasV6MOps,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureMClass,
                                                       FeatureStrictAlign]>;

def ARMv7a    : Architecture<"armv7-a",   "ARMv7a",   [HasV7Ops,
                                                       FeatureNEON,
                                                       FeatureDB,
                                                       FeatureDSP,
                                                       FeatureAClass,
                                                       FeaturePerfMon]>;

def ARMv7ve   : Architecture<"armv7ve",   "ARMv7ve",  [HasV7Ops,
                                                       FeatureNEON,
                                                       FeatureDB,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureAClass,
                                                       FeaturePerfMon]>;

def ARMv7r    : Architecture<"armv7-r",   "ARMv7r",   [HasV7Ops,
                                                       FeatureDB,
                                                       FeatureDSP,
                                                       FeatureHWDivThumb,
                                                       FeatureRClass,
                                                       FeaturePerfMon]>;

def ARMv7m    : Architecture<"armv7-m",   "ARMv7m",   [HasV7Ops,
                                                       FeatureThumb2,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureHWDivThumb,
                                                       FeatureMClass]>;

def ARMv7em   : Architecture<"armv7e-m",  "ARMv7em",  [HasV7Ops,
                                                       FeatureThumb2,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureHWDivThumb,
                                                       FeatureMClass,
                                                       FeatureDSP]>;

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
                                                       FeatureCRC,
                                                       FeatureRAS]>;

def ARMv83a   : Architecture<"armv8.3-a", "ARMv83a",  [HasV8_3aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS]>;

def ARMv84a   : Architecture<"armv8.4-a", "ARMv84a",  [HasV8_4aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;

def ARMv85a   : Architecture<"armv8.5-a", "ARMv85a",  [HasV8_5aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv86a   : Architecture<"armv8.6-a", "ARMv86a",  [HasV8_6aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv87a   : Architecture<"armv8.7-a", "ARMv87a",  [HasV8_7aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv88a   : Architecture<"armv8.8-a", "ARMv88a",  [HasV8_8aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;

def ARMv9a   : Architecture<"armv9-a", "ARMv9a",       [HasV9_0aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv91a   : Architecture<"armv9.1-a", "ARMv91a",   [HasV9_1aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv92a   : Architecture<"armv9.2-a", "ARMv92a",  [HasV9_2aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;
def ARMv93a   : Architecture<"armv9.3-a", "ARMv93a",  [HasV9_3aOps,
                                                       FeatureAClass,
                                                       FeatureDB,
                                                       FeatureFPARMv8,
                                                       FeatureNEON,
                                                       FeatureDSP,
                                                       FeatureTrustZone,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureCrypto,
                                                       FeatureCRC,
                                                       FeatureRAS,
                                                       FeatureDotProd]>;

def ARMv8r    : Architecture<"armv8-r",   "ARMv8r",   [HasV8Ops,
                                                       FeatureRClass,
                                                       FeatureDB,
                                                       FeatureDFB,
                                                       FeatureDSP,
                                                       FeatureCRC,
                                                       FeatureMP,
                                                       FeatureVirtualization,
                                                       FeatureFPARMv8,
                                                       FeatureNEON]>;

def ARMv8mBaseline : Architecture<"armv8-m.base", "ARMv8mBaseline",
                                                      [HasV8MBaselineOps,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureHWDivThumb,
                                                       FeatureV7Clrex,
                                                       Feature8MSecExt,
                                                       FeatureAcquireRelease,
                                                       FeatureMClass,
                                                       FeatureStrictAlign]>;

def ARMv8mMainline : Architecture<"armv8-m.main", "ARMv8mMainline",
                                                      [HasV8MMainlineOps,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureHWDivThumb,
                                                       Feature8MSecExt,
                                                       FeatureAcquireRelease,
                                                       FeatureMClass]>;

def ARMv81mMainline : Architecture<"armv8.1-m.main", "ARMv81mMainline",
                                                      [HasV8_1MMainlineOps,
                                                       FeatureNoARM,
                                                       ModeThumb,
                                                       FeatureDB,
                                                       FeatureHWDivThumb,
                                                       Feature8MSecExt,
                                                       FeatureAcquireRelease,
                                                       FeatureMClass,
                                                       FeatureRAS,
                                                       FeatureLOB]>;

// Aliases
def IWMMXT   : Architecture<"iwmmxt",      "ARMv5te",  [ARMv5te]>;
def IWMMXT2  : Architecture<"iwmmxt2",     "ARMv5te",  [ARMv5te]>;
def XScale   : Architecture<"xscale",      "ARMv5te",  [ARMv5te]>;
def ARMv6j   : Architecture<"armv6j",      "ARMv7a",   [ARMv6]>;
def ARMv7k   : Architecture<"armv7k",      "ARMv7a",   [ARMv7a]>;
def ARMv7s   : Architecture<"armv7s",      "ARMv7a",   [ARMv7a]>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "ARMRegisterInfo.td"
include "ARMRegisterBanks.td"
include "ARMCallingConv.td"

//===----------------------------------------------------------------------===//
// ARM schedules.
//===----------------------------------------------------------------------===//
//
include "ARMPredicates.td"
include "ARMSchedule.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "ARMInstrInfo.td"
def ARMInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// ARM schedules
//
include "ARMScheduleV6.td"
include "ARMScheduleA8.td"
include "ARMScheduleA9.td"
include "ARMScheduleSwift.td"
include "ARMScheduleR52.td"
include "ARMScheduleA57.td"
include "ARMScheduleM4.td"
include "ARMScheduleM7.td"

//===----------------------------------------------------------------------===//
// ARM processors
//
// Dummy CPU, used to target architectures
def : ProcessorModel<"generic",     CortexA8Model,      []>;

// FIXME: Several processors below are not using their own scheduler
// model, but one of similar/previous processor. These should be fixed.

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

def : Processor<"cortex-m0",        ARMV6Itineraries,   [ARMv6m,
                                                         FeatureHasNoBranchPredictor]>;
def : Processor<"cortex-m0plus",    ARMV6Itineraries,   [ARMv6m,
                                                         FeatureHasNoBranchPredictor]>;
def : Processor<"cortex-m1",        ARMV6Itineraries,   [ARMv6m,
                                                         FeatureHasNoBranchPredictor]>;
def : Processor<"sc000",            ARMV6Itineraries,   [ARMv6m,
                                                         FeatureHasNoBranchPredictor]>;

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

def : ProcessorModel<"cortex-a5",   CortexA8Model,      [ARMv7a, ProcA5,
                                                         FeatureHasRetAddrStack,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureVMLxForwarding,
                                                         FeatureMP,
                                                         FeatureVFP4]>;

def : ProcessorModel<"cortex-a7",   CortexA8Model,      [ARMv7a, ProcA7,
                                                         FeatureHasRetAddrStack,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasVMLxHazards,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureVMLxForwarding,
                                                         FeatureMP,
                                                         FeatureVFP4,
                                                         FeatureVirtualization]>;

def : ProcessorModel<"cortex-a8",   CortexA8Model,      [ARMv7a, ProcA8,
                                                         FeatureHasRetAddrStack,
                                                         FeatureNonpipelinedVFP,
                                                         FeatureTrustZone,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasVMLxHazards,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureVMLxForwarding]>;

def : ProcessorModel<"cortex-a9",   CortexA9Model,      [ARMv7a, ProcA9,
                                                         FeatureHasRetAddrStack,
                                                         FeatureTrustZone,
                                                         FeatureHasVMLxHazards,
                                                         FeatureVMLxForwarding,
                                                         FeatureFP16,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureExpandMLx,
                                                         FeaturePreferVMOVSR,
                                                         FeatureMuxedUnits,
                                                         FeatureNEONForFPMovs,
                                                         FeatureCheckVLDnAlign,
                                                         FeatureMP]>;

def : ProcessorModel<"cortex-a12",  CortexA9Model,      [ARMv7a, ProcA12,
                                                         FeatureHasRetAddrStack,
                                                         FeatureTrustZone,
                                                         FeatureVMLxForwarding,
                                                         FeatureVFP4,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization,
                                                         FeatureMP]>;

def : ProcessorModel<"cortex-a15",  CortexA9Model,      [ARMv7a, ProcA15,
                                                         FeatureDontWidenVMOVS,
                                                         FeatureSplatVFPToNeon,
                                                         FeatureHasRetAddrStack,
                                                         FeatureMuxedUnits,
                                                         FeatureTrustZone,
                                                         FeatureVFP4,
                                                         FeatureMP,
                                                         FeatureCheckVLDnAlign,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization]>;

def : ProcessorModel<"cortex-a17",  CortexA9Model,      [ARMv7a, ProcA17,
                                                         FeatureHasRetAddrStack,
                                                         FeatureTrustZone,
                                                         FeatureMP,
                                                         FeatureVMLxForwarding,
                                                         FeatureVFP4,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVirtualization]>;

// FIXME: krait has currently the same features as A9 plus VFP4 and  HWDiv
def : ProcessorModel<"krait",       CortexA9Model,      [ARMv7a, ProcKrait,
                                                         FeatureHasRetAddrStack,
                                                         FeatureMuxedUnits,
                                                         FeatureCheckVLDnAlign,
                                                         FeatureVMLxForwarding,
                                                         FeatureFP16,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureVFP4,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM]>;

def : ProcessorModel<"swift",       SwiftModel,         [ARMv7a, ProcSwift,
                                                         FeatureHasRetAddrStack,
                                                         FeatureNEONForFP,
                                                         FeatureVFP4,
                                                         FeatureUseWideStrideVFP,
                                                         FeatureMP,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureAvoidMOVsShOp,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureHasVMLxHazards,
                                                         FeatureProfUnpredicate,
                                                         FeaturePrefISHSTBarrier,
                                                         FeatureSlowOddRegister,
                                                         FeatureSlowLoadDSubreg,
                                                         FeatureSlowVGETLNi32,
                                                         FeatureSlowVDUP32,
                                                         FeatureUseMISched,
                                                         FeatureNoPostRASched]>;

def : ProcessorModel<"cortex-r4",   CortexA8Model,      [ARMv7r, ProcR4,
                                                         FeatureHasRetAddrStack,
                                                         FeatureAvoidPartialCPSR]>;

def : ProcessorModel<"cortex-r4f",  CortexA8Model,      [ARMv7r, ProcR4,
                                                         FeatureHasRetAddrStack,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureVFP3_D16,
                                                         FeatureAvoidPartialCPSR]>;

def : ProcessorModel<"cortex-r5",   CortexA8Model,      [ARMv7r, ProcR5,
                                                         FeatureHasRetAddrStack,
                                                         FeatureVFP3_D16,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHWDivARM,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureAvoidPartialCPSR]>;

def : ProcessorModel<"cortex-r7",   CortexA8Model,      [ARMv7r, ProcR7,
                                                         FeatureHasRetAddrStack,
                                                         FeatureVFP3_D16,
                                                         FeatureFP16,
                                                         FeatureMP,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHWDivARM,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureAvoidPartialCPSR]>;

def : ProcessorModel<"cortex-r8",   CortexA8Model,      [ARMv7r,
                                                         FeatureHasRetAddrStack,
                                                         FeatureVFP3_D16,
                                                         FeatureFP16,
                                                         FeatureMP,
                                                         FeatureSlowFPBrcc,
                                                         FeatureHWDivARM,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureAvoidPartialCPSR]>;

def : ProcessorModel<"cortex-m3",   CortexM4Model,      [ARMv7m,
                                                         ProcM3,
                                                         FeaturePrefLoopAlign32,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor]>;

def : ProcessorModel<"sc300",       CortexM4Model,      [ARMv7m,
                                                         ProcM3,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor]>;

def : ProcessorModel<"cortex-m4", CortexM4Model,        [ARMv7em,
                                                         FeatureVFP4_D16_SP,
                                                         FeaturePrefLoopAlign32,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor]>;

def : ProcessorModel<"cortex-m7", CortexM7Model,        [ARMv7em,
                                                         ProcM7,
                                                         FeatureFPARMv8_D16,
                                                         FeatureUseMIPipeliner,
                                                         FeatureUseMISched]>;

def : ProcNoItin<"cortex-m23",                          [ARMv8mBaseline,
                                                         FeatureNoMovt,
                                                         FeatureHasNoBranchPredictor]>;

def : ProcessorModel<"cortex-m33", CortexM4Model,       [ARMv8mMainline,
                                                         FeatureDSP,
                                                         FeatureFPARMv8_D16_SP,
                                                         FeaturePrefLoopAlign32,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor,
                                                         FeatureFixCMSE_CVE_2021_35465]>;

def : ProcessorModel<"cortex-m35p", CortexM4Model,      [ARMv8mMainline,
                                                         FeatureDSP,
                                                         FeatureFPARMv8_D16_SP,
                                                         FeaturePrefLoopAlign32,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor,
                                                         FeatureFixCMSE_CVE_2021_35465]>;

def : ProcessorModel<"cortex-m55", CortexM4Model,      [ARMv81mMainline,
                                                         FeatureDSP,
                                                         FeatureFPARMv8_D16,
                                                         FeatureUseMISched,
                                                         FeatureHasNoBranchPredictor,
                                                         FeaturePrefLoopAlign32,
                                                         FeatureHasSlowFPVMLx,
                                                         HasMVEFloatOps,
                                                         FeatureFixCMSE_CVE_2021_35465]>;

def : ProcessorModel<"cortex-m85", CortexM7Model,       [ARMv81mMainline,
                                                         FeatureDSP,
                                                         FeatureFPARMv8_D16,
                                                         FeaturePACBTI,
                                                         FeatureUseMISched,
                                                         HasMVEFloatOps]>;

def : ProcNoItin<"cortex-a32",                           [ARMv8a,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a35",                          [ARMv8a, ProcA35,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a53",                          [ARMv8a, ProcA53,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFPAO]>;

def : ProcNoItin<"cortex-a55",                          [ARMv82a, ProcA55,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureDotProd]>;

def : ProcessorModel<"cortex-a57",  CortexA57Model,     [ARMv8a, ProcA57,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFPAO,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureCheapPredicableCPSR,
                                                         FeatureFixCortexA57AES1742098]>;

def : ProcessorModel<"cortex-a72",  CortexA57Model,     [ARMv8a, ProcA72,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFixCortexA57AES1742098]>;

def : ProcNoItin<"cortex-a73",                          [ARMv8a, ProcA73,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcNoItin<"cortex-a75",                          [ARMv82a, ProcA75,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-a76",                          [ARMv82a, ProcA76,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-a76ae",                        [ARMv82a, ProcA76,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-a77",                          [ARMv82a, ProcA77,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-a78",                          [ARMv82a, ProcA78,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-a78c",                         [ARMv82a, ProcA78C,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureDotProd,
                                                         FeatureFullFP16]>;

def : ProcNoItin<"cortex-a710",                         [ARMv9a, ProcA710,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureFP16FML,
                                                         FeatureBF16,
                                                         FeatureMatMulInt8,
                                                         FeatureSB]>;

def : ProcNoItin<"cortex-x1",                           [ARMv82a, ProcX1,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"cortex-x1c",                          [ARMv82a, ProcX1C,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"neoverse-v1",                         [ARMv84a,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureFullFP16,
                                                         FeatureBF16,
                                                         FeatureMatMulInt8]>;

def : ProcNoItin<"neoverse-n1",                         [ARMv82a,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC,
                                                         FeatureDotProd]>;

def : ProcNoItin<"neoverse-n2",                         [ARMv85a,
                                                         FeatureBF16,
                                                         FeatureMatMulInt8]>;

def : ProcessorModel<"cyclone",     SwiftModel,         [ARMv8a, ProcSwift,
                                                         FeatureHasRetAddrStack,
                                                         FeatureNEONForFP,
                                                         FeatureVFP4,
                                                         FeatureMP,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureAvoidPartialCPSR,
                                                         FeatureAvoidMOVsShOp,
                                                         FeatureHasSlowFPVMLx,
                                                         FeatureHasSlowFPVFMx,
                                                         FeatureCrypto,
                                                         FeatureUseMISched,
                                                         FeatureZCZeroing,
                                                         FeatureNoPostRASched]>;

def : ProcNoItin<"exynos-m3",                           [ARMv8a, ProcExynos]>;
def : ProcNoItin<"exynos-m4",                           [ARMv82a, ProcExynos,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;
def : ProcNoItin<"exynos-m5",                           [ARMv82a, ProcExynos,
                                                         FeatureFullFP16,
                                                         FeatureDotProd]>;

def : ProcNoItin<"kryo",                                [ARMv8a, ProcKryo,
                                                         FeatureHWDivThumb,
                                                         FeatureHWDivARM,
                                                         FeatureCrypto,
                                                         FeatureCRC]>;

def : ProcessorModel<"cortex-r52", CortexR52Model,      [ARMv8r, ProcR52,
                                                         FeatureUseMISched,
                                                         FeatureFPAO]>;

//===----------------------------------------------------------------------===//
// Declare the target which we are implementing
//===----------------------------------------------------------------------===//

def ARMAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  int PassSubtarget = 1;
  int Variant = 0;
  bit isMCAsmWriter = 1;
}

def ARMAsmParser : AsmParser {
  bit ReportMultipleNearMisses = 1;
}

def ARMAsmParserVariant : AsmParserVariant {
  int Variant = 0;
  string Name = "ARM";
  string BreakCharacters = ".";
}

def ARM : Target {
  // Pull in Instruction Info.
  let InstructionSet = ARMInstrInfo;
  let AssemblyWriters = [ARMAsmWriter];
  let AssemblyParsers = [ARMAsmParser];
  let AssemblyParserVariants = [ARMAsmParserVariant];
  let AllowRegisterRenaming = 1;
}
