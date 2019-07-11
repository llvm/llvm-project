//===-- AMDGPU.td - AMDGPU Tablegen files --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"
include "AMDGPUFeatures.td"

class BoolToList<bit Value> {
  list<int> ret = !if(Value, [1]<int>, []<int>);
}

//===------------------------------------------------------------===//
// Subtarget Features (device properties)
//===------------------------------------------------------------===//

def FeatureFastFMAF32 : SubtargetFeature<"fast-fmaf",
  "FastFMAF32",
  "true",
  "Assuming f32 fma is at least as fast as mul + add"
>;

def FeatureMIMG_R128 : SubtargetFeature<"mimg-r128",
  "MIMG_R128",
  "true",
  "Support 128-bit texture resources"
>;

def HalfRate64Ops : SubtargetFeature<"half-rate-64-ops",
  "HalfRate64Ops",
  "true",
  "Most fp64 instructions are half rate instead of quarter"
>;

def FeatureFlatAddressSpace : SubtargetFeature<"flat-address-space",
  "FlatAddressSpace",
  "true",
  "Support flat address space"
>;

def FeatureFlatInstOffsets : SubtargetFeature<"flat-inst-offsets",
  "FlatInstOffsets",
  "true",
  "Flat instructions have immediate offset addressing mode"
>;

def FeatureFlatGlobalInsts : SubtargetFeature<"flat-global-insts",
  "FlatGlobalInsts",
  "true",
  "Have global_* flat memory instructions"
>;

def FeatureFlatScratchInsts : SubtargetFeature<"flat-scratch-insts",
  "FlatScratchInsts",
  "true",
  "Have scratch_* flat memory instructions"
>;

def FeatureScalarFlatScratchInsts : SubtargetFeature<"scalar-flat-scratch-insts",
  "ScalarFlatScratchInsts",
  "true",
  "Have s_scratch_* flat memory instructions"
>;

def FeatureAddNoCarryInsts : SubtargetFeature<"add-no-carry-insts",
  "AddNoCarryInsts",
  "true",
  "Have VALU add/sub instructions without carry out"
>;

def FeatureUnalignedBufferAccess : SubtargetFeature<"unaligned-buffer-access",
  "UnalignedBufferAccess",
  "true",
  "Support unaligned global loads and stores"
>;

def FeatureTrapHandler: SubtargetFeature<"trap-handler",
  "TrapHandler",
  "true",
  "Trap handler support"
>;

def FeatureUnalignedScratchAccess : SubtargetFeature<"unaligned-scratch-access",
  "UnalignedScratchAccess",
  "true",
  "Support unaligned scratch loads and stores"
>;

def FeatureApertureRegs : SubtargetFeature<"aperture-regs",
  "HasApertureRegs",
  "true",
  "Has Memory Aperture Base and Size Registers"
>;

def FeatureMadMixInsts : SubtargetFeature<"mad-mix-insts",
  "HasMadMixInsts",
  "true",
  "Has v_mad_mix_f32, v_mad_mixlo_f16, v_mad_mixhi_f16 instructions"
>;

def FeatureFmaMixInsts : SubtargetFeature<"fma-mix-insts",
  "HasFmaMixInsts",
  "true",
  "Has v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 instructions"
>;

def FeatureDoesNotSupportXNACK : SubtargetFeature<"no-xnack-support",
  "DoesNotSupportXNACK",
  "true",
  "Hardware does not support XNACK"
>;

// XNACK is disabled if SH_MEM_CONFIG.ADDRESS_MODE = GPUVM on chips that support
// XNACK. The current default kernel driver setting is:
// - graphics ring: XNACK disabled
// - compute ring: XNACK enabled
//
// If XNACK is enabled, the VMEM latency can be worse.
// If XNACK is disabled, the 2 SGPRs can be used for general purposes.
def FeatureXNACK : SubtargetFeature<"xnack",
  "EnableXNACK",
  "true",
  "Enable XNACK support"
>;

def FeatureCuMode : SubtargetFeature<"cumode",
  "EnableCuMode",
  "true",
  "Enable CU wavefront execution mode"
>;

def FeatureSGPRInitBug : SubtargetFeature<"sgpr-init-bug",
  "SGPRInitBug",
  "true",
  "VI SGPR initialization bug requiring a fixed SGPR allocation size"
>;

def FeatureLdsMisalignedBug : SubtargetFeature<"lds-misaligned-bug",
  "LDSMisalignedBug",
  "true",
  "Some GFX10 bug with misaligned multi-dword LDS access in WGP mode"
>;

def FeatureVcmpxPermlaneHazard : SubtargetFeature<"vcmpx-permlane-hazard",
  "HasVcmpxPermlaneHazard",
  "true",
  "TODO: describe me"
>;

def FeatureVMEMtoScalarWriteHazard : SubtargetFeature<"vmem-to-scalar-write-hazard",
  "HasVMEMtoScalarWriteHazard",
  "true",
  "VMEM instruction followed by scalar writing to EXEC mask, M0 or SGPR leads to incorrect execution."
>;

def FeatureSMEMtoVectorWriteHazard : SubtargetFeature<"smem-to-vector-write-hazard",
  "HasSMEMtoVectorWriteHazard",
  "true",
  "s_load_dword followed by v_cmp page faults"
>;

def FeatureInstFwdPrefetchBug : SubtargetFeature<"inst-fwd-prefetch-bug",
  "HasInstFwdPrefetchBug",
  "true",
  "S_INST_PREFETCH instruction causes shader to hang"
>;

def FeatureVcmpxExecWARHazard : SubtargetFeature<"vcmpx-exec-war-hazard",
  "HasVcmpxExecWARHazard",
  "true",
  "V_CMPX WAR hazard on EXEC (V_CMPX issue ONLY)"
>;

def FeatureLdsBranchVmemWARHazard : SubtargetFeature<"lds-branch-vmem-war-hazard",
  "HasLdsBranchVmemWARHazard",
  "true",
  "Switching between LDS and VMEM-tex not waiting VM_VSRC=0"
>;

def FeatureNSAtoVMEMBug : SubtargetFeature<"nsa-to-vmem-bug",
  "HasNSAtoVMEMBug",
  "true",
  "MIMG-NSA followed by VMEM fail if EXEC_LO or EXEC_HI equals zero"
>;

def FeatureFlatSegmentOffsetBug : SubtargetFeature<"flat-segment-offset-bug",
  "HasFlatSegmentOffsetBug",
  "true",
  "GFX10 bug, inst_offset ignored in flat segment"
>;

def FeatureOffset3fBug : SubtargetFeature<"offset-3f-bug",
  "HasOffset3fBug",
  "true",
  "Branch offset of 3f hardware bug"
>;

class SubtargetFeatureLDSBankCount <int Value> : SubtargetFeature <
  "ldsbankcount"#Value,
  "LDSBankCount",
  !cast<string>(Value),
  "The number of LDS banks per compute unit."
>;

def FeatureLDSBankCount16 : SubtargetFeatureLDSBankCount<16>;
def FeatureLDSBankCount32 : SubtargetFeatureLDSBankCount<32>;

def FeatureGCN3Encoding : SubtargetFeature<"gcn3-encoding",
  "GCN3Encoding",
  "true",
  "Encoding format for VI"
>;

def FeatureCIInsts : SubtargetFeature<"ci-insts",
  "CIInsts",
  "true",
  "Additional instructions for CI+"
>;

def FeatureGFX8Insts : SubtargetFeature<"gfx8-insts",
  "GFX8Insts",
  "true",
  "Additional instructions for GFX8+"
>;

def FeatureGFX9Insts : SubtargetFeature<"gfx9-insts",
  "GFX9Insts",
  "true",
  "Additional instructions for GFX9+"
>;

def FeatureGFX10Insts : SubtargetFeature<"gfx10-insts",
  "GFX10Insts",
  "true",
  "Additional instructions for GFX10+"
>;

def FeatureGFX7GFX8GFX9Insts : SubtargetFeature<"gfx7-gfx8-gfx9-insts",
  "GFX7GFX8GFX9Insts",
  "true",
  "Instructions shared in GFX7, GFX8, GFX9"
>;

def FeatureSMemRealTime : SubtargetFeature<"s-memrealtime",
  "HasSMemRealTime",
  "true",
  "Has s_memrealtime instruction"
>;

def FeatureInv2PiInlineImm : SubtargetFeature<"inv-2pi-inline-imm",
  "HasInv2PiInlineImm",
  "true",
  "Has 1 / (2 * pi) as inline immediate"
>;

def Feature16BitInsts : SubtargetFeature<"16-bit-insts",
  "Has16BitInsts",
  "true",
  "Has i16/f16 instructions"
>;

def FeatureVOP3P : SubtargetFeature<"vop3p",
  "HasVOP3PInsts",
  "true",
  "Has VOP3P packed instructions"
>;

def FeatureMovrel : SubtargetFeature<"movrel",
  "HasMovrel",
  "true",
  "Has v_movrel*_b32 instructions"
>;

def FeatureVGPRIndexMode : SubtargetFeature<"vgpr-index-mode",
  "HasVGPRIndexMode",
  "true",
  "Has VGPR mode register indexing"
>;

def FeatureScalarStores : SubtargetFeature<"scalar-stores",
  "HasScalarStores",
  "true",
  "Has store scalar memory instructions"
>;

def FeatureScalarAtomics : SubtargetFeature<"scalar-atomics",
  "HasScalarAtomics",
  "true",
  "Has atomic scalar memory instructions"
>;

def FeatureSDWA : SubtargetFeature<"sdwa",
  "HasSDWA",
  "true",
  "Support SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureSDWAOmod : SubtargetFeature<"sdwa-omod",
  "HasSDWAOmod",
  "true",
  "Support OMod with SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureSDWAScalar : SubtargetFeature<"sdwa-scalar",
  "HasSDWAScalar",
  "true",
  "Support scalar register with SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureSDWASdst : SubtargetFeature<"sdwa-sdst",
  "HasSDWASdst",
  "true",
  "Support scalar dst for VOPC with SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureSDWAMac : SubtargetFeature<"sdwa-mav",
  "HasSDWAMac",
  "true",
  "Support v_mac_f32/f16 with SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureSDWAOutModsVOPC : SubtargetFeature<"sdwa-out-mods-vopc",
  "HasSDWAOutModsVOPC",
  "true",
  "Support clamp for VOPC with SDWA (Sub-DWORD Addressing) extension"
>;

def FeatureDPP : SubtargetFeature<"dpp",
  "HasDPP",
  "true",
  "Support DPP (Data Parallel Primitives) extension"
>;

// DPP8 allows arbitrary cross-lane swizzling withing groups of 8 lanes.
def FeatureDPP8 : SubtargetFeature<"dpp8",
  "HasDPP8",
  "true",
  "Support DPP8 (Data Parallel Primitives) extension"
>;

def FeatureR128A16 : SubtargetFeature<"r128-a16",
  "HasR128A16",
  "true",
  "Support 16 bit coordindates/gradients/lod/clamp/mip types on gfx9"
>;

def FeatureNSAEncoding : SubtargetFeature<"nsa-encoding",
  "HasNSAEncoding",
  "true",
  "Support NSA encoding for image instructions"
>;

def FeatureIntClamp : SubtargetFeature<"int-clamp-insts",
  "HasIntClamp",
  "true",
  "Support clamp for integer destination"
>;

def FeatureUnpackedD16VMem : SubtargetFeature<"unpacked-d16-vmem",
  "HasUnpackedD16VMem",
  "true",
  "Has unpacked d16 vmem instructions"
>;

def FeatureDLInsts : SubtargetFeature<"dl-insts",
  "HasDLInsts",
  "true",
  "Has v_fmac_f32 and v_xnor_b32 instructions"
>;

def FeatureDot1Insts : SubtargetFeature<"dot1-insts",
  "HasDot1Insts",
  "true",
  "Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions"
>;

def FeatureDot2Insts : SubtargetFeature<"dot2-insts",
  "HasDot2Insts",
  "true",
  "Has v_dot2_f32_f16, v_dot2_i32_i16, v_dot2_u32_u16, v_dot4_u32_u8, v_dot8_u32_u4 instructions"
>;

def FeatureDot3Insts : SubtargetFeature<"dot3-insts",
  "HasDot3Insts",
  "true",
  "Has v_dot8c_i32_i4 instruction"
>;

def FeatureDot4Insts : SubtargetFeature<"dot4-insts",
  "HasDot4Insts",
  "true",
  "Has v_dot2c_i32_i16 instruction"
>;

def FeatureDot5Insts : SubtargetFeature<"dot5-insts",
  "HasDot5Insts",
  "true",
  "Has v_dot2c_f32_f16 instruction"
>;

def FeatureDot6Insts : SubtargetFeature<"dot6-insts",
  "HasDot6Insts",
  "true",
  "Has v_dot4c_i32_i8 instruction"
>;

def FeatureMAIInsts : SubtargetFeature<"mai-insts",
  "HasMAIInsts",
  "true",
  "Has mAI instructions"
>;

def FeaturePkFmacF16Inst : SubtargetFeature<"pk-fmac-f16-inst",
  "HasPkFmacF16Inst",
  "true",
  "Has v_pk_fmac_f16 instruction"
>;

def FeatureAtomicFaddInsts : SubtargetFeature<"atomic-fadd-insts",
  "HasAtomicFaddInsts",
  "true",
  "Has buffer_atomic_add_f32, buffer_atomic_pk_add_f16, global_atomic_add_f32, "
  "global_atomic_pk_add_f16 instructions"
>;

def FeatureDoesNotSupportSRAMECC : SubtargetFeature<"no-sram-ecc-support",
  "DoesNotSupportSRAMECC",
  "true",
  "Hardware does not support SRAM ECC"
>;

def FeatureSRAMECC : SubtargetFeature<"sram-ecc",
  "EnableSRAMECC",
  "true",
  "Enable SRAM ECC"
>;

def FeatureNoSdstCMPX : SubtargetFeature<"no-sdst-cmpx",
  "HasNoSdstCMPX",
  "true",
  "V_CMPX does not write VCC/SGPR in addition to EXEC"
>;

def FeatureVscnt : SubtargetFeature<"vscnt",
  "HasVscnt",
  "true",
  "Has separate store vscnt counter"
>;

def FeatureRegisterBanking : SubtargetFeature<"register-banking",
  "HasRegisterBanking",
  "true",
  "Has register banking"
>;

def FeatureVOP3Literal : SubtargetFeature<"vop3-literal",
  "HasVOP3Literal",
  "true",
  "Can use one literal in VOP3"
>;

def FeatureNoDataDepHazard : SubtargetFeature<"no-data-dep-hazard",
  "HasNoDataDepHazard",
  "true",
  "Does not need SW waitstates"
>;

//===------------------------------------------------------------===//
// Subtarget Features (options and debugging)
//===------------------------------------------------------------===//

// Denormal handling for fp64 and fp16 is controlled by the same
// config register when fp16 supported.
// TODO: Do we need a separate f16 setting when not legal?
def FeatureFP64FP16Denormals : SubtargetFeature<"fp64-fp16-denormals",
  "FP64FP16Denormals",
  "true",
  "Enable double and half precision denormal handling",
  [FeatureFP64]
>;

def FeatureFP64Denormals : SubtargetFeature<"fp64-denormals",
  "FP64FP16Denormals",
  "true",
  "Enable double and half precision denormal handling",
  [FeatureFP64, FeatureFP64FP16Denormals]
>;

def FeatureFP16Denormals : SubtargetFeature<"fp16-denormals",
  "FP64FP16Denormals",
  "true",
  "Enable half precision denormal handling",
  [FeatureFP64FP16Denormals]
>;

def FeatureFPExceptions : SubtargetFeature<"fp-exceptions",
  "FPExceptions",
  "true",
  "Enable floating point exceptions"
>;

class FeatureMaxPrivateElementSize<int size> : SubtargetFeature<
  "max-private-element-size-"#size,
  "MaxPrivateElementSize",
  !cast<string>(size),
  "Maximum private access size may be "#size
>;

def FeatureMaxPrivateElementSize4 : FeatureMaxPrivateElementSize<4>;
def FeatureMaxPrivateElementSize8 : FeatureMaxPrivateElementSize<8>;
def FeatureMaxPrivateElementSize16 : FeatureMaxPrivateElementSize<16>;

def FeatureDumpCode : SubtargetFeature <"DumpCode",
  "DumpCode",
  "true",
  "Dump MachineInstrs in the CodeEmitter"
>;

def FeatureDumpCodeLower : SubtargetFeature <"dumpcode",
  "DumpCode",
  "true",
  "Dump MachineInstrs in the CodeEmitter"
>;

// XXX - This should probably be removed once enabled by default
def FeatureEnableLoadStoreOpt : SubtargetFeature <"load-store-opt",
  "EnableLoadStoreOpt",
  "true",
  "Enable SI load/store optimizer pass"
>;

// Performance debugging feature. Allow using DS instruction immediate
// offsets even if the base pointer can't be proven to be base. On SI,
// base pointer values that won't give the same result as a 16-bit add
// are not safe to fold, but this will override the conservative test
// for the base pointer.
def FeatureEnableUnsafeDSOffsetFolding : SubtargetFeature <
  "unsafe-ds-offset-folding",
  "EnableUnsafeDSOffsetFolding",
  "true",
  "Force using DS instruction immediate offsets on SI"
>;

def FeatureEnableSIScheduler : SubtargetFeature<"si-scheduler",
  "EnableSIScheduler",
  "true",
  "Enable SI Machine Scheduler"
>;

def FeatureEnableDS128 : SubtargetFeature<"enable-ds128",
  "EnableDS128",
  "true",
  "Use ds_{read|write}_b128"
>;

// Sparse texture support requires that all result registers are zeroed when
// PRTStrictNull is set to true. This feature is turned on for all architectures
// but is enabled as a feature in case there are situations where PRTStrictNull
// is disabled by the driver.
def FeatureEnablePRTStrictNull : SubtargetFeature<"enable-prt-strict-null",
  "EnablePRTStrictNull",
  "true",
  "Enable zeroing of result registers for sparse texture fetches"
>;

// Unless +-flat-for-global is specified, turn on FlatForGlobal for
// all OS-es on VI and newer hardware to avoid assertion failures due
// to missing ADDR64 variants of MUBUF instructions.
// FIXME: moveToVALU should be able to handle converting addr64 MUBUF
// instructions.

def FeatureFlatForGlobal : SubtargetFeature<"flat-for-global",
  "FlatForGlobal",
  "true",
  "Force to generate flat instruction for global"
>;

def FeatureAutoWaitcntBeforeBarrier : SubtargetFeature <
  "auto-waitcnt-before-barrier",
  "AutoWaitcntBeforeBarrier",
  "true",
  "Hardware automatically inserts waitcnt before barrier"
>;

def FeatureCodeObjectV3 : SubtargetFeature <
  "code-object-v3",
  "CodeObjectV3",
  "true",
  "Generate code object version 3"
>;

def FeatureTrigReducedRange : SubtargetFeature<"trig-reduced-range",
  "HasTrigReducedRange",
  "true",
  "Requires use of fract on arguments to trig instructions"
>;

// Dummy feature used to disable assembler instructions.
def FeatureDisable : SubtargetFeature<"",
  "FeatureDisable","true",
  "Dummy feature to disable assembler instructions"
>;

class GCNSubtargetFeatureGeneration <string Value,
                                     string FeatureName,
                                     list<SubtargetFeature> Implies> :
        SubtargetFeatureGeneration <Value, FeatureName, "GCNSubtarget", Implies>;

def FeatureSouthernIslands : GCNSubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
    "southern-islands",
  [FeatureFP64, FeatureLocalMemorySize32768, FeatureMIMG_R128,
  FeatureWavefrontSize64,
  FeatureLDSBankCount32, FeatureMovrel, FeatureTrigReducedRange,
  FeatureDoesNotSupportSRAMECC, FeatureDoesNotSupportXNACK]
>;

def FeatureSeaIslands : GCNSubtargetFeatureGeneration<"SEA_ISLANDS",
    "sea-islands",
  [FeatureFP64, FeatureLocalMemorySize65536, FeatureMIMG_R128,
  FeatureWavefrontSize64, FeatureFlatAddressSpace,
  FeatureCIInsts, FeatureMovrel, FeatureTrigReducedRange,
  FeatureGFX7GFX8GFX9Insts, FeatureDoesNotSupportSRAMECC]
>;

def FeatureVolcanicIslands : GCNSubtargetFeatureGeneration<"VOLCANIC_ISLANDS",
  "volcanic-islands",
  [FeatureFP64, FeatureLocalMemorySize65536, FeatureMIMG_R128,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureVGPRIndexMode, FeatureMovrel,
   FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureSDWA, FeatureSDWAOutModsVOPC, FeatureSDWAMac, FeatureDPP,
   FeatureIntClamp, FeatureTrigReducedRange, FeatureDoesNotSupportSRAMECC,
   FeatureGFX8Insts, FeatureGFX7GFX8GFX9Insts
  ]
>;

def FeatureGFX9 : GCNSubtargetFeatureGeneration<"GFX9",
  "gfx9",
  [FeatureFP64, FeatureLocalMemorySize65536,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureApertureRegs, FeatureGFX9Insts, FeatureVOP3P, FeatureVGPRIndexMode,
   FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureSDWA, FeatureSDWAOmod, FeatureSDWAScalar, FeatureSDWASdst,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureGFX8Insts, FeatureGFX7GFX8GFX9Insts,
   FeatureScalarFlatScratchInsts, FeatureScalarAtomics, FeatureR128A16
  ]
>;

def FeatureGFX10 : GCNSubtargetFeatureGeneration<"GFX10",
  "gfx10",
  [FeatureFP64, FeatureLocalMemorySize65536, FeatureMIMG_R128,
   FeatureFlatAddressSpace,
   FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureInv2PiInlineImm,
   FeatureApertureRegs, FeatureGFX9Insts, FeatureGFX10Insts, FeatureVOP3P,
   FeatureMovrel, FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureSDWA, FeatureSDWAOmod, FeatureSDWAScalar, FeatureSDWASdst,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureFmaMixInsts, FeatureGFX8Insts,
   FeatureNoSdstCMPX, FeatureVscnt, FeatureRegisterBanking,
   FeatureVOP3Literal, FeatureDPP8,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst, FeatureDoesNotSupportSRAMECC
  ]
>;

class FeatureSet<list<SubtargetFeature> Features_> {
  list<SubtargetFeature> Features = Features_;
}

def FeatureISAVersion6_0_0 : FeatureSet<[FeatureSouthernIslands,
   FeatureFastFMAF32,
   HalfRate64Ops,
   FeatureLDSBankCount32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion6_0_1 : FeatureSet<
  [FeatureSouthernIslands,
   FeatureLDSBankCount32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion7_0_0 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion7_0_1 : FeatureSet<
  [FeatureSeaIslands,
   HalfRate64Ops,
   FeatureLDSBankCount32,
   FeatureFastFMAF32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion7_0_2 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount16,
   FeatureFastFMAF32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion7_0_3 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount16,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion7_0_4 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount32,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion8_0_1 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureFastFMAF32,
   HalfRate64Ops,
   FeatureLDSBankCount32,
   FeatureXNACK,
   FeatureUnpackedD16VMem,
   FeatureCodeObjectV3]>;

def FeatureISAVersion8_0_2 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureSGPRInitBug,
   FeatureUnpackedD16VMem,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion8_0_3 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureUnpackedD16VMem,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion8_1_0 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount16,
   FeatureXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion9_0_0 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureCodeObjectV3,
   FeatureDoesNotSupportXNACK,
   FeatureDoesNotSupportSRAMECC]>;

def FeatureISAVersion9_0_2 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureXNACK,
   FeatureDoesNotSupportSRAMECC,
   FeatureCodeObjectV3]>;

def FeatureISAVersion9_0_4 : FeatureSet<
  [FeatureGFX9,
   FeatureLDSBankCount32,
   FeatureFmaMixInsts,
   FeatureDoesNotSupportXNACK,
   FeatureDoesNotSupportSRAMECC,
   FeatureCodeObjectV3]>;

def FeatureISAVersion9_0_6 : FeatureSet<
  [FeatureGFX9,
   HalfRate64Ops,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureDoesNotSupportXNACK,
   FeatureCodeObjectV3]>;

def FeatureISAVersion9_0_8 : FeatureSet<
  [FeatureGFX9,
   HalfRate64Ops,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureDot3Insts,
   FeatureDot4Insts,
   FeatureDot5Insts,
   FeatureDot6Insts,
   FeatureMAIInsts,
   FeaturePkFmacF16Inst,
   FeatureAtomicFaddInsts,
   FeatureSRAMECC,
   FeatureCodeObjectV3]>;

def FeatureISAVersion9_0_9 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureXNACK,
   FeatureCodeObjectV3]>;

// TODO: Organize more features into groups.
def FeatureGroup {
  // Bugs present on gfx10.1.
  list<SubtargetFeature> GFX10_1_Bugs = [
    FeatureVcmpxPermlaneHazard,
    FeatureVMEMtoScalarWriteHazard,
    FeatureSMEMtoVectorWriteHazard,
    FeatureInstFwdPrefetchBug,
    FeatureVcmpxExecWARHazard,
    FeatureLdsBranchVmemWARHazard,
    FeatureNSAtoVMEMBug,
    FeatureOffset3fBug,
    FeatureFlatSegmentOffsetBug
   ];
}

def FeatureISAVersion10_1_0 : FeatureSet<
  !listconcat(FeatureGroup.GFX10_1_Bugs,
    [FeatureGFX10,
     FeatureLDSBankCount32,
     FeatureDLInsts,
     FeatureNSAEncoding,
     FeatureWavefrontSize32,
     FeatureScalarStores,
     FeatureScalarAtomics,
     FeatureScalarFlatScratchInsts,
     FeatureLdsMisalignedBug,
     FeatureDoesNotSupportXNACK,
     FeatureCodeObjectV3])>;

def FeatureISAVersion10_1_1 : FeatureSet<
  !listconcat(FeatureGroup.GFX10_1_Bugs,
    [FeatureGFX10,
     FeatureLDSBankCount32,
     FeatureDLInsts,
     FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureNSAEncoding,
     FeatureWavefrontSize32,
     FeatureScalarStores,
     FeatureScalarAtomics,
     FeatureScalarFlatScratchInsts,
     FeatureDoesNotSupportXNACK,
     FeatureCodeObjectV3])>;

def FeatureISAVersion10_1_2 : FeatureSet<
  !listconcat(FeatureGroup.GFX10_1_Bugs,
    [FeatureGFX10,
     FeatureLDSBankCount32,
     FeatureDLInsts,
     FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureNSAEncoding,
     FeatureWavefrontSize32,
     FeatureScalarStores,
     FeatureScalarAtomics,
     FeatureScalarFlatScratchInsts,
     FeatureLdsMisalignedBug,
     FeatureDoesNotSupportXNACK,
     FeatureCodeObjectV3])>;

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
  let noNamedPositionallyEncodedOperands = 1;
}

def AMDGPUAsmParser : AsmParser {
  // Some of the R600 registers have the same name, so this crashes.
  // For example T0_XYZW and T0_XY both have the asm name T0.
  let ShouldEmitMatchRegisterName = 0;
}

def AMDGPUAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def AMDGPUAsmVariants {
  string Default = "Default";
  int Default_ID = 0;
  string VOP3 = "VOP3";
  int VOP3_ID = 1;
  string SDWA = "SDWA";
  int SDWA_ID = 2;
  string SDWA9 = "SDWA9";
  int SDWA9_ID = 3;
  string DPP = "DPP";
  int DPP_ID = 4;
  string Disable = "Disable";
  int Disable_ID = 5;
}

def DefaultAMDGPUAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.Default_ID;
  let Name = AMDGPUAsmVariants.Default;
}

def VOP3AsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.VOP3_ID;
  let Name = AMDGPUAsmVariants.VOP3;
}

def SDWAAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.SDWA_ID;
  let Name = AMDGPUAsmVariants.SDWA;
}

def SDWA9AsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.SDWA9_ID;
  let Name = AMDGPUAsmVariants.SDWA9;
}


def DPPAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.DPP_ID;
  let Name = AMDGPUAsmVariants.DPP;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyParsers = [AMDGPUAsmParser];
  let AssemblyParserVariants = [DefaultAMDGPUAsmParserVariant,
                                VOP3AsmParserVariant,
                                SDWAAsmParserVariant,
                                SDWA9AsmParserVariant,
                                DPPAsmParserVariant];
  let AssemblyWriters = [AMDGPUAsmWriter];
  let AllowRegisterRenaming = 1;
}

// Dummy Instruction itineraries for pseudo instructions
def ALU_NULL : FuncUnit;
def NullALU : InstrItinClass;

//===----------------------------------------------------------------------===//
// Predicate helper class
//===----------------------------------------------------------------------===//

def isGFX6 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS">,
  AssemblerPredicate<"FeatureSouthernIslands">;

def isGFX6GFX7 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<"!FeatureGCN3Encoding,!FeatureGFX10Insts">;

def isGFX6GFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<"!FeatureGCN3Encoding">;

def isGFX7Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<"!FeatureGCN3Encoding,FeatureCIInsts,!FeatureGFX10Insts">;

def isGFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<"!FeatureGCN3Encoding,FeatureCIInsts">;

def isGFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"FeatureGFX7GFX8GFX9Insts">;

def isGFX6GFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"!FeatureGFX10Insts">;

def isGFX7Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<"FeatureCIInsts">;

def isGFX8Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate<"FeatureGFX8Insts">;

def isGFX8Only : Predicate<"Subtarget->getGeneration() =="
                           "AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate <"FeatureVolcanicIslands">;

def isGFX9Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"FeatureGFX9Insts">;

def isGFX9Only : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"FeatureGCN3Encoding,FeatureGFX9Insts">;

def isGFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"FeatureGFX8Insts,FeatureGCN3Encoding">;

def isGFX10Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<"FeatureGFX10Insts">;

def HasFlatAddressSpace : Predicate<"Subtarget->hasFlatAddressSpace()">,
  AssemblerPredicate<"FeatureFlatAddressSpace">;

def HasFlatGlobalInsts : Predicate<"Subtarget->hasFlatGlobalInsts()">,
  AssemblerPredicate<"FeatureFlatGlobalInsts">;
def HasFlatScratchInsts : Predicate<"Subtarget->hasFlatScratchInsts()">,
  AssemblerPredicate<"FeatureFlatScratchInsts">;
def HasScalarFlatScratchInsts : Predicate<"Subtarget->hasScalarFlatScratchInsts()">,
  AssemblerPredicate<"FeatureScalarFlatScratchInsts">;
def HasD16LoadStore : Predicate<"Subtarget->hasD16LoadStore()">,
  AssemblerPredicate<"FeatureGFX9Insts">;

def HasUnpackedD16VMem : Predicate<"Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<"FeatureUnpackedD16VMem">;
def HasPackedD16VMem : Predicate<"!Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<"!FeatureUnpackedD16VMem">;

def D16PreservesUnusedBits :
  Predicate<"Subtarget->d16PreservesUnusedBits()">,
  AssemblerPredicate<"FeatureGFX9Insts,!FeatureSRAMECC">;

def LDSRequiresM0Init : Predicate<"Subtarget->ldsRequiresM0Init()">;
def NotLDSRequiresM0Init : Predicate<"!Subtarget->ldsRequiresM0Init()">;

def HasDSAddTid : Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<"FeatureGFX9Insts">;

def HasAddNoCarryInsts : Predicate<"Subtarget->hasAddNoCarry()">,
  AssemblerPredicate<"FeatureAddNoCarryInsts">;

def NotHasAddNoCarryInsts : Predicate<"!Subtarget->hasAddNoCarry()">;

def Has16BitInsts : Predicate<"Subtarget->has16BitInsts()">,
  AssemblerPredicate<"Feature16BitInsts">;
def HasVOP3PInsts : Predicate<"Subtarget->hasVOP3PInsts()">,
  AssemblerPredicate<"FeatureVOP3P">;

def HasSDWA : Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<"FeatureSDWA,FeatureVolcanicIslands">;

def HasSDWA9 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<"FeatureGCN3Encoding,FeatureGFX9Insts,FeatureSDWA">;

def HasSDWA10 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<"!FeatureGCN3Encoding,FeatureGFX10Insts,FeatureSDWA">;

def HasDPP : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<"FeatureGCN3Encoding,FeatureDPP">;

def HasDPP8 : Predicate<"Subtarget->hasDPP8()">,
  AssemblerPredicate<"!FeatureGCN3Encoding,FeatureGFX10Insts,FeatureDPP8">;

def HasR128A16 : Predicate<"Subtarget->hasR128A16()">,
  AssemblerPredicate<"FeatureR128A16">;

def HasDPP16 : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<"!FeatureGCN3Encoding,FeatureGFX10Insts,FeatureDPP">;

def HasIntClamp : Predicate<"Subtarget->hasIntClamp()">,
  AssemblerPredicate<"FeatureIntClamp">;

def HasMadMixInsts : Predicate<"Subtarget->hasMadMixInsts()">,
  AssemblerPredicate<"FeatureMadMixInsts">;

def HasScalarStores : Predicate<"Subtarget->hasScalarStores()">,
  AssemblerPredicate<"FeatureScalarStores">;

def HasScalarAtomics : Predicate<"Subtarget->hasScalarAtomics()">,
  AssemblerPredicate<"FeatureScalarAtomics">;

def HasNoSdstCMPX : Predicate<"Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<"FeatureNoSdstCMPX">;

def HasSdstCMPX : Predicate<"!Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<"!FeatureNoSdstCMPX">;

def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;
def HasVGPRIndexMode : Predicate<"Subtarget->hasVGPRIndexMode()">,
                      AssemblerPredicate<"FeatureVGPRIndexMode">;
def HasMovrel : Predicate<"Subtarget->hasMovrel()">,
                AssemblerPredicate<"FeatureMovrel">;

def HasFmaMixInsts : Predicate<"Subtarget->hasFmaMixInsts()">,
  AssemblerPredicate<"FeatureFmaMixInsts">;

def HasDLInsts : Predicate<"Subtarget->hasDLInsts()">,
  AssemblerPredicate<"FeatureDLInsts">;

def HasDot1Insts : Predicate<"Subtarget->hasDot1Insts()">,
  AssemblerPredicate<"FeatureDot1Insts">;

def HasDot2Insts : Predicate<"Subtarget->hasDot2Insts()">,
  AssemblerPredicate<"FeatureDot2Insts">;

def HasDot3Insts : Predicate<"Subtarget->hasDot3Insts()">,
  AssemblerPredicate<"FeatureDot3Insts">;

def HasDot4Insts : Predicate<"Subtarget->hasDot4Insts()">,
  AssemblerPredicate<"FeatureDot4Insts">;

def HasDot5Insts : Predicate<"Subtarget->hasDot5Insts()">,
  AssemblerPredicate<"FeatureDot5Insts">;

def HasDot6Insts : Predicate<"Subtarget->hasDot6Insts()">,
  AssemblerPredicate<"FeatureDot6Insts">;

def HasMAIInsts : Predicate<"Subtarget->hasMAIInsts()">,
  AssemblerPredicate<"FeatureMAIInsts">;

def HasPkFmacF16Inst : Predicate<"Subtarget->hasPkFmacF16Inst()">,
  AssemblerPredicate<"FeaturePkFmacF16Inst">;

def HasAtomicFaddInsts : Predicate<"Subtarget->hasAtomicFaddInsts()">,
  AssemblerPredicate<"FeatureAtomicFaddInsts">;

def HasOffset3fBug : Predicate<"!Subtarget->hasOffset3fBug()">,
  AssemblerPredicate<"FeatureOffset3fBug">;

def EnableLateCFGStructurize : Predicate<
  "EnableLateStructurizeCFG">;

// Include AMDGPU TD files
include "SISchedule.td"
include "GCNProcessors.td"
include "AMDGPUInstrInfo.td"
include "AMDGPURegisterInfo.td"
include "AMDGPURegisterBanks.td"
include "AMDGPUInstructions.td"
include "SIInstrInfo.td"
include "AMDGPUCallingConv.td"
include "AMDGPUSearchableTables.td"
