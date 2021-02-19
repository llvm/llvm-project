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

def p0 : PtrValueType<i64, 0>;
def p1 : PtrValueType<i64, 1>;
def p2 : PtrValueType<i32, 2>;
def p3 : PtrValueType<i32, 3>;
def p4 : PtrValueType<i64, 4>;
def p5 : PtrValueType<i32, 5>;
def p6 : PtrValueType<i32, 6>;


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

def FeatureFastDenormalF32 : SubtargetFeature<"fast-denormal-f32",
  "FastDenormalF32",
  "true",
  "Enabling denormals does not cause f32 instructions to run at f64 rates"
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

def FullRate64Ops : SubtargetFeature<"full-rate-64-ops",
  "FullRate64Ops",
  "true",
  "Most fp64 instructions are full rate"
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
  "Hardware supports unaligned global loads and stores"
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

def FeatureUnalignedDSAccess : SubtargetFeature<"unaligned-ds-access",
  "UnalignedDSAccess",
  "true",
  "Hardware supports unaligned local and region loads and stores"
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

def FeatureSupportsXNACK : SubtargetFeature<"xnack-support",
  "SupportsXNACK",
  "true",
  "Hardware supports XNACK"
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

def FeatureTgSplit : SubtargetFeature<"tgsplit",
  "EnableTgSplit",
  "true",
  "Enable threadgroup split execution"
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
  "Some GFX10 bug with multi-dword LDS and flat access that is not naturally aligned in WGP mode"
>;

def FeatureMFMAInlineLiteralBug : SubtargetFeature<"mfma-inline-literal-bug",
  "HasMFMAInlineLiteralBug",
  "true",
  "MFMA cannot use inline literal as SrcC"
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

def FeatureImageStoreD16Bug : SubtargetFeature<"image-store-d16-bug",
  "HasImageStoreD16Bug",
  "true",
  "Image Store D16 hardware bug"
>;

def FeatureImageGather4D16Bug : SubtargetFeature<"image-gather4-d16-bug",
  "HasImageGather4D16Bug",
  "true",
  "Image Gather4 D16 hardware bug"
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

def FeatureGFX90AInsts : SubtargetFeature<"gfx90a-insts",
  "GFX90AInsts",
  "true",
  "Additional instructions for GFX90A+"
>;

def FeatureGFX10Insts : SubtargetFeature<"gfx10-insts",
  "GFX10Insts",
  "true",
  "Additional instructions for GFX10+"
>;

def FeatureGFX10_3Insts : SubtargetFeature<"gfx10-3-insts",
  "GFX10_3Insts",
  "true",
  "Additional instructions for GFX10.3"
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

def Feature64BitDPP : SubtargetFeature<"dpp-64bit",
  "Has64BitDPP",
  "true",
  "Support DPP (Data Parallel Primitives) extension"
>;

def FeaturePackedFP32Ops : SubtargetFeature<"packed-fp32-ops",
  "HasPackedFP32Ops",
  "true",
  "Support packed fp32 instructions"
>;

def FeatureR128A16 : SubtargetFeature<"r128-a16",
  "HasR128A16",
  "true",
  "Support gfx9-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands, where a16 is aliased with r128"
>;

def FeatureGFX10A16 : SubtargetFeature<"a16",
  "HasGFX10A16",
  "true",
  "Support gfx10-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands"
>;

def FeatureG16 : SubtargetFeature<"g16",
  "HasG16",
  "true",
  "Support G16 for 16-bit gradient image operands"
>;

def FeatureNSAEncoding : SubtargetFeature<"nsa-encoding",
  "HasNSAEncoding",
  "true",
  "Support NSA encoding for image instructions"
>;

def FeatureExtendedImageInsts : SubtargetFeature<"extended-image-insts",
  "HasExtendedImageInsts",
  "true",
  "Support mips != 0, lod != 0, gather4, and get_lod"
>;

def FeatureGFX10_BEncoding : SubtargetFeature<"gfx10_b-encoding",
  "GFX10_BEncoding",
  "true",
  "Encoding format GFX10_B"
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
  "global_atomic_pk_add_f16 instructions",
  [FeatureFlatGlobalInsts]
>;

def FeatureSupportsSRAMECC : SubtargetFeature<"sramecc-support",
  "SupportsSRAMECC",
  "true",
  "Hardware supports SRAMECC"
>;

def FeatureSRAMECC : SubtargetFeature<"sramecc",
  "EnableSRAMECC",
  "true",
  "Enable SRAMECC"
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

def FeatureGetWaveIdInst : SubtargetFeature<"get-wave-id-inst",
  "HasGetWaveIdInst",
  "true",
  "Has s_get_waveid_in_workgroup instruction"
>;

def FeatureSMemTimeInst : SubtargetFeature<"s-memtime-inst",
  "HasSMemTimeInst",
  "true",
  "Has s_memtime instruction"
>;

def FeatureMadMacF32Insts : SubtargetFeature<"mad-mac-f32-insts",
  "HasMadMacF32Insts",
  "true",
  "Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions"
>;

def FeatureDsSrc2Insts : SubtargetFeature<"ds-src2-insts",
  "HasDsSrc2Insts",
  "true",
  "Has ds_*_src2 instructions"
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

def FeatureTrigReducedRange : SubtargetFeature<"trig-reduced-range",
  "HasTrigReducedRange",
  "true",
  "Requires use of fract on arguments to trig instructions"
>;

// Alignment enforcement is controlled by a configuration register:
// SH_MEM_CONFIG.alignment_mode
def FeatureUnalignedAccessMode : SubtargetFeature<"unaligned-access-mode",
  "UnalignedAccessMode",
  "true",
  "Enable unaligned global, local and region loads and stores if the hardware"
  " supports it"
>;

def FeaturePackedTID : SubtargetFeature<"packed-tid",
  "HasPackedTID",
  "true",
  "Workitem IDs are packed into v0 at kernel launch"
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
  FeatureWavefrontSize64, FeatureSMemTimeInst, FeatureMadMacF32Insts,
  FeatureDsSrc2Insts, FeatureLDSBankCount32, FeatureMovrel,
  FeatureTrigReducedRange, FeatureExtendedImageInsts
  ]
>;

def FeatureSeaIslands : GCNSubtargetFeatureGeneration<"SEA_ISLANDS",
    "sea-islands",
  [FeatureFP64, FeatureLocalMemorySize65536, FeatureMIMG_R128,
  FeatureWavefrontSize64, FeatureFlatAddressSpace,
  FeatureCIInsts, FeatureMovrel, FeatureTrigReducedRange,
  FeatureGFX7GFX8GFX9Insts, FeatureSMemTimeInst, FeatureMadMacF32Insts,
  FeatureDsSrc2Insts, FeatureExtendedImageInsts, FeatureUnalignedBufferAccess
  ]
>;

def FeatureVolcanicIslands : GCNSubtargetFeatureGeneration<"VOLCANIC_ISLANDS",
  "volcanic-islands",
  [FeatureFP64, FeatureLocalMemorySize65536, FeatureMIMG_R128,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureVGPRIndexMode, FeatureMovrel,
   FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureSDWA, FeatureSDWAOutModsVOPC, FeatureSDWAMac, FeatureDPP,
   FeatureIntClamp, FeatureTrigReducedRange, FeatureGFX8Insts,
   FeatureGFX7GFX8GFX9Insts, FeatureSMemTimeInst, FeatureMadMacF32Insts,
   FeatureDsSrc2Insts, FeatureExtendedImageInsts, FeatureFastDenormalF32,
   FeatureUnalignedBufferAccess
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
   FeatureScalarFlatScratchInsts, FeatureScalarAtomics, FeatureR128A16,
   FeatureSMemTimeInst, FeatureFastDenormalF32, FeatureSupportsXNACK,
   FeatureUnalignedBufferAccess, FeatureUnalignedDSAccess
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
   FeatureVOP3Literal, FeatureDPP8, FeatureExtendedImageInsts,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst,
   FeatureGFX10A16, FeatureFastDenormalF32, FeatureG16,
   FeatureUnalignedBufferAccess, FeatureUnalignedDSAccess
  ]
>;

class FeatureSet<list<SubtargetFeature> Features_> {
  list<SubtargetFeature> Features = Features_;
}

def FeatureISAVersion6_0_0 : FeatureSet<[FeatureSouthernIslands,
   FeatureFastFMAF32,
   HalfRate64Ops,
   FeatureLDSBankCount32]>;

def FeatureISAVersion6_0_1 : FeatureSet<
  [FeatureSouthernIslands,
   FeatureLDSBankCount32]>;

def FeatureISAVersion6_0_2 : FeatureSet<
  [FeatureSouthernIslands,
   FeatureLDSBankCount32]>;

def FeatureISAVersion7_0_0 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount32]>;

def FeatureISAVersion7_0_1 : FeatureSet<
  [FeatureSeaIslands,
   HalfRate64Ops,
   FeatureLDSBankCount32,
   FeatureFastFMAF32]>;

def FeatureISAVersion7_0_2 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount16,
   FeatureFastFMAF32]>;

def FeatureISAVersion7_0_3 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount16]>;

def FeatureISAVersion7_0_4 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount32]>;

def FeatureISAVersion7_0_5 : FeatureSet<
  [FeatureSeaIslands,
   FeatureLDSBankCount16]>;

def FeatureISAVersion8_0_1 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureFastFMAF32,
   HalfRate64Ops,
   FeatureLDSBankCount32,
   FeatureSupportsXNACK,
   FeatureUnpackedD16VMem]>;

def FeatureISAVersion8_0_2 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureSGPRInitBug,
   FeatureUnpackedD16VMem]>;

def FeatureISAVersion8_0_3 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureUnpackedD16VMem]>;

def FeatureISAVersion8_0_5 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureSGPRInitBug,
   FeatureUnpackedD16VMem]>;

def FeatureISAVersion8_1_0 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount16,
   FeatureSupportsXNACK,
   FeatureImageStoreD16Bug,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_0 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_2 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_4 : FeatureSet<
  [FeatureGFX9,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
   FeatureFmaMixInsts,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_6 : FeatureSet<
  [FeatureGFX9,
   HalfRate64Ops,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
   FeatureDLInsts,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureSupportsSRAMECC,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_8 : FeatureSet<
  [FeatureGFX9,
   HalfRate64Ops,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
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
   FeatureSupportsSRAMECC,
   FeatureMFMAInlineLiteralBug,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_9 : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureMadMacF32Insts,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_A : FeatureSet<
  [FeatureGFX9,
   FeatureGFX90AInsts,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureDot3Insts,
   FeatureDot4Insts,
   FeatureDot5Insts,
   FeatureDot6Insts,
   Feature64BitDPP,
   FeaturePackedFP32Ops,
   FeatureMAIInsts,
   FeaturePkFmacF16Inst,
   FeatureAtomicFaddInsts,
   FeatureMadMacF32Insts,
   FeatureSupportsSRAMECC,
   FeaturePackedTID,
   FullRate64Ops]>;

def FeatureISAVersion9_0_C : FeatureSet<
  [FeatureGFX9,
   FeatureMadMixInsts,
   FeatureLDSBankCount32,
   FeatureXNACK,
   FeatureDsSrc2Insts,
   FeatureExtendedImageInsts,
   FeatureImageGather4D16Bug]>;

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
     FeatureGetWaveIdInst,
     FeatureSMemTimeInst,
     FeatureMadMacF32Insts,
     FeatureDsSrc2Insts,
     FeatureLdsMisalignedBug,
     FeatureSupportsXNACK])>;

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
     FeatureGetWaveIdInst,
     FeatureSMemTimeInst,
     FeatureMadMacF32Insts,
     FeatureDsSrc2Insts,
     FeatureLdsMisalignedBug,
     FeatureSupportsXNACK])>;

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
     FeatureGetWaveIdInst,
     FeatureSMemTimeInst,
     FeatureMadMacF32Insts,
     FeatureDsSrc2Insts,
     FeatureLdsMisalignedBug,
     FeatureSupportsXNACK])>;

def FeatureISAVersion10_3_0 : FeatureSet<
  [FeatureGFX10,
   FeatureGFX10_BEncoding,
   FeatureGFX10_3Insts,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureDot5Insts,
   FeatureDot6Insts,
   FeatureNSAEncoding,
   FeatureWavefrontSize32]>;

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
  AssemblerPredicate<(all_of FeatureSouthernIslands)>;

def isGFX6GFX7 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), (not FeatureGFX10Insts))>;

def isGFX6GFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding))>;

def isGFX7Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts, (not FeatureGFX10Insts))>;

def isGFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts)>;

def isGFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX7GFX8GFX9Insts)>;

def isGFX6GFX7GFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of (not FeatureGFX10Insts))>;

def isGFX6GFX7GFX8GFX9NotGFX90A :
  Predicate<"!Subtarget->hasGFX90AInsts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of (not FeatureGFX10Insts), (not FeatureGFX90AInsts))>;

def isGFX7Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of FeatureCIInsts)>;

def isGFX8Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate<(all_of FeatureGFX8Insts)>;

def isGFX8Only : Predicate<"Subtarget->getGeneration() =="
                           "AMDGPUSubtarget::VOLCANIC_ISLANDS">,
  AssemblerPredicate <(all_of FeatureVolcanicIslands)>;

def isGFX9Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def isGFX9Only : Predicate <
  "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureGFX9Insts)>;

def isGCN3ExcludingGFX90A :
  Predicate<"Subtarget->isGCN3Encoding() && !Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, (not FeatureGFX90AInsts))>;

def isGFX90APlus :
  Predicate<"Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts)>;

def isNotGFX90APlus :
  Predicate<"!Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of (not FeatureGFX90AInsts))>;

def isGFX8GFX9NotGFX90A :
  Predicate<"!Subtarget->hasGFX90AInsts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding, (not FeatureGFX90AInsts))>;

def isGFX90AOnly :
  Predicate<"Subtarget->hasGFX90AInsts()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts)>;

def isGFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding)>;

def isGFX10Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX10Insts)>;

def isGFX10Before1030 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 &&"
            "!Subtarget->hasGFX10_3Insts()">,
  AssemblerPredicate<(all_of FeatureGFX10Insts,(not FeatureGFX10_3Insts))>;

def HasFlatAddressSpace : Predicate<"Subtarget->hasFlatAddressSpace()">,
  AssemblerPredicate<(all_of FeatureFlatAddressSpace)>;

def HasFlatGlobalInsts : Predicate<"Subtarget->hasFlatGlobalInsts()">,
  AssemblerPredicate<(all_of FeatureFlatGlobalInsts)>;
def HasFlatScratchInsts : Predicate<"Subtarget->hasFlatScratchInsts()">,
  AssemblerPredicate<(all_of FeatureFlatScratchInsts)>;
def HasScalarFlatScratchInsts : Predicate<"Subtarget->hasScalarFlatScratchInsts()">,
  AssemblerPredicate<(all_of FeatureScalarFlatScratchInsts)>;
def HasD16LoadStore : Predicate<"Subtarget->hasD16LoadStore()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasFlatScratchSTMode : Predicate<"Subtarget->hasFlatScratchSTMode()">,
  AssemblerPredicate<(any_of FeatureGFX10_3Insts)>;

def HasGFX10_BEncoding : Predicate<"Subtarget->hasGFX10_BEncoding()">,
  AssemblerPredicate<(all_of FeatureGFX10_BEncoding)>;

def HasUnpackedD16VMem : Predicate<"Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<(all_of FeatureUnpackedD16VMem)>;
def HasPackedD16VMem : Predicate<"!Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<(all_of (not FeatureUnpackedD16VMem))>;

def D16PreservesUnusedBits :
  Predicate<"Subtarget->d16PreservesUnusedBits()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureSRAMECC))>;

def LDSRequiresM0Init : Predicate<"Subtarget->ldsRequiresM0Init()">;
def NotLDSRequiresM0Init : Predicate<"!Subtarget->ldsRequiresM0Init()">;

def HasDSAddTid : Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasLDSFPAtomics : Predicate<"Subtarget->hasLDSFPAtomics()">,
  AssemblerPredicate<(all_of FeatureGFX8Insts)>;

def HasAddNoCarryInsts : Predicate<"Subtarget->hasAddNoCarry()">,
  AssemblerPredicate<(all_of FeatureAddNoCarryInsts)>;

def NotHasAddNoCarryInsts : Predicate<"!Subtarget->hasAddNoCarry()">;

def Has16BitInsts : Predicate<"Subtarget->has16BitInsts()">,
  AssemblerPredicate<(all_of Feature16BitInsts)>;
def HasVOP3PInsts : Predicate<"Subtarget->hasVOP3PInsts()">,
  AssemblerPredicate<(all_of FeatureVOP3P)>;

def HasMinMaxDenormModes : Predicate<"Subtarget->supportsMinMaxDenormModes()">;
def NotHasMinMaxDenormModes : Predicate<"!Subtarget->supportsMinMaxDenormModes()">;

def HasSDWA : Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of FeatureSDWA, FeatureVolcanicIslands)>;

def HasSDWA9 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureGFX9Insts,FeatureSDWA)>;

def HasSDWA10 :
  Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureSDWA)>;

def HasDPP : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<(all_of FeatureGCN3Encoding, FeatureDPP)>;

def HasDPP8 : Predicate<"Subtarget->hasDPP8()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureDPP8)>;

def Has64BitDPP : Predicate<"Subtarget->has64BitDPP()">,
  AssemblerPredicate<(all_of Feature64BitDPP)>;

def HasPackedFP32Ops : Predicate<"Subtarget->hasPackedFP32Ops()">,
  AssemblerPredicate<(all_of FeaturePackedFP32Ops)>;

def HasExtendedImageInsts : Predicate<"Subtarget->hasExtendedImageInsts()">,
  AssemblerPredicate<(all_of FeatureExtendedImageInsts)>;

def HasR128A16 : Predicate<"Subtarget->hasR128A16()">,
  AssemblerPredicate<(all_of FeatureR128A16)>;

def HasGFX10A16 : Predicate<"Subtarget->hasGFX10A16()">,
  AssemblerPredicate<(all_of FeatureGFX10A16)>;

def HasG16 : Predicate<"Subtarget->hasG16()">,
  AssemblerPredicate<(all_of FeatureG16)>;

def HasDPP16 : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureDPP)>;

def HasIntClamp : Predicate<"Subtarget->hasIntClamp()">,
  AssemblerPredicate<(all_of FeatureIntClamp)>;

def HasMadMixInsts : Predicate<"Subtarget->hasMadMixInsts()">,
  AssemblerPredicate<(all_of FeatureMadMixInsts)>;

def HasScalarStores : Predicate<"Subtarget->hasScalarStores()">,
  AssemblerPredicate<(all_of FeatureScalarStores)>;

def HasScalarAtomics : Predicate<"Subtarget->hasScalarAtomics()">,
  AssemblerPredicate<(all_of FeatureScalarAtomics)>;

def HasNoSdstCMPX : Predicate<"Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<(all_of FeatureNoSdstCMPX)>;

def HasSdstCMPX : Predicate<"!Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<(all_of (not FeatureNoSdstCMPX))>;

def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;
def HasVGPRIndexMode : Predicate<"Subtarget->hasVGPRIndexMode()">,
                      AssemblerPredicate<(all_of FeatureVGPRIndexMode)>;
def HasMovrel : Predicate<"Subtarget->hasMovrel()">,
                AssemblerPredicate<(all_of FeatureMovrel)>;

def HasFmaMixInsts : Predicate<"Subtarget->hasFmaMixInsts()">,
  AssemblerPredicate<(all_of FeatureFmaMixInsts)>;

def HasDLInsts : Predicate<"Subtarget->hasDLInsts()">,
  AssemblerPredicate<(all_of FeatureDLInsts)>;

def HasDot1Insts : Predicate<"Subtarget->hasDot1Insts()">,
  AssemblerPredicate<(all_of FeatureDot1Insts)>;

def HasDot2Insts : Predicate<"Subtarget->hasDot2Insts()">,
  AssemblerPredicate<(all_of FeatureDot2Insts)>;

def HasDot3Insts : Predicate<"Subtarget->hasDot3Insts()">,
  AssemblerPredicate<(all_of FeatureDot3Insts)>;

def HasDot4Insts : Predicate<"Subtarget->hasDot4Insts()">,
  AssemblerPredicate<(all_of FeatureDot4Insts)>;

def HasDot5Insts : Predicate<"Subtarget->hasDot5Insts()">,
  AssemblerPredicate<(all_of FeatureDot5Insts)>;

def HasDot6Insts : Predicate<"Subtarget->hasDot6Insts()">,
  AssemblerPredicate<(all_of FeatureDot6Insts)>;

def HasGetWaveIdInst : Predicate<"Subtarget->hasGetWaveIdInst()">,
  AssemblerPredicate<(all_of FeatureGetWaveIdInst)>;

def HasMAIInsts : Predicate<"Subtarget->hasMAIInsts()">,
  AssemblerPredicate<(all_of FeatureMAIInsts)>;

def HasSMemRealTime : Predicate<"Subtarget->hasSMemRealTime()">,
  AssemblerPredicate<(all_of FeatureSMemRealTime)>;

def HasSMemTimeInst : Predicate<"Subtarget->hasSMemTimeInst()">,
  AssemblerPredicate<(all_of FeatureSMemTimeInst)>;

def HasNoSMemTimeInst : Predicate<"!Subtarget->hasSMemTimeInst()">;

def HasPkFmacF16Inst : Predicate<"Subtarget->hasPkFmacF16Inst()">,
  AssemblerPredicate<(all_of FeaturePkFmacF16Inst)>;

def HasMadMacF32Insts : Predicate<"Subtarget->hasMadMacF32Insts()">,
  AssemblerPredicate<(all_of FeatureMadMacF32Insts)>;

def HasFmaLegacy32 : Predicate<"Subtarget->hasGFX10_3Insts()">,
  AssemblerPredicate<(any_of FeatureGFX10_3Insts)>;

def HasAtomicFaddInsts : Predicate<"Subtarget->hasAtomicFaddInsts()">,
  AssemblerPredicate<(all_of FeatureAtomicFaddInsts)>;

def HasDsSrc2Insts : Predicate<"!Subtarget->hasDsSrc2Insts()">,
  AssemblerPredicate<(all_of FeatureDsSrc2Insts)>;

def HasOffset3fBug : Predicate<"!Subtarget->hasOffset3fBug()">,
  AssemblerPredicate<(all_of FeatureOffset3fBug)>;

def EnableLateCFGStructurize : Predicate<
  "EnableLateStructurizeCFG">;

def EnableFlatScratch : Predicate<"Subtarget->enableFlatScratch()">;

def DisableFlatScratch : Predicate<"!Subtarget->enableFlatScratch()">;

def HasUnalignedAccessMode : Predicate<"Subtarget->hasUnalignedAccessMode()">,
  AssemblerPredicate<(all_of FeatureUnalignedAccessMode)>;

// Include AMDGPU TD files
include "SISchedule.td"
include "GCNProcessors.td"
include "AMDGPUInstrInfo.td"
include "SIRegisterInfo.td"
include "AMDGPURegisterBanks.td"
include "AMDGPUInstructions.td"
include "SIInstrInfo.td"
include "AMDGPUCallingConv.td"
include "AMDGPUSearchableTables.td"
