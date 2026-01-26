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
include "AMDGPUPredicateControl.td"

def p0 : PtrValueType<i64, 0>;
def p1 : PtrValueType<i64, 1>;
def p2 : PtrValueType<i32, 2>;
def p3 : PtrValueType<i32, 3>;
def p4 : PtrValueType<i64, 4>;
def p5 : PtrValueType<i32, 5>;
def p6 : PtrValueType<i32, 6>;

//===-----------------------------------------------------------------------===//
// AMDGPU Subtarget Feature (device properties)
//===----------------------------------------------------------------------===//

// Multiclass to define a SubtargetFeature along with optional predicates.
// Parameters:
//   - FeatureString: The feature string used in the SubtargetFeature.
//   - Description: The description of the feature.
//   - GenPredicate: If 1 (default), generates a Has#NAME predicate.
//   - GenAssemblerPredicate: If 1 (default), the predicate includes AssemblerPredicate.
//   - Deps: List of dependent SubtargetFeatures (default empty).
//
// Usage:
//   defm MadMixInsts : AMDGPUSubtargetFeature<"mad-mix-insts", "description">;
// This generates:
//   - FeatureMadMixInsts : SubtargetFeature<"mad-mix-insts", "HasMadMixInsts", "true", "description">
//   - HasMadMixInsts : Predicate<"Subtarget->hasMadMixInsts()">,
//                      AssemblerPredicate<(any_of FeatureMadMixInsts)>
//
// With GenAssemblerPredicate=0:
//   defm Foo : AMDGPUSubtargetFeature<"foo", "desc", 1, 0>;
// This generates:
//   - FeatureFoo : SubtargetFeature<...>
//   - HasFoo : Predicate<"Subtarget->hasFoo()"> (no AssemblerPredicate)
//
// With dependencies:
//   defm Bar : AMDGPUSubtargetFeature<"bar", "desc", 1, 1, [FeatureFoo]>;
// This generates:
//   - FeatureBar : SubtargetFeature<"bar", "HasBar", "true", "desc", [FeatureFoo]>
//   - HasBar : Predicate + AssemblerPredicate
multiclass AMDGPUSubtargetFeature<string FeatureString,
                                  string Description,
                                  bit GenPredicate = 1,
                                  bit GenAssemblerPredicate = 1,
                                  list<SubtargetFeature> Deps = []> {
  def Feature#NAME : SubtargetFeature<FeatureString,
    "Has"#NAME,
    "true",
    Description,
    Deps
  >;

  if GenPredicate then
    if GenAssemblerPredicate then
      def Has#NAME
        : Predicate<"Subtarget->has"#NAME#"()">,
          AssemblerPredicate<(any_of !cast<SubtargetFeature>("Feature"#NAME))>;
    else
      def Has#NAME : Predicate<"Subtarget->has"#NAME#"()">;
}

defm FastFMAF32 : AMDGPUSubtargetFeature<"fast-fmaf",
  "Assuming f32 fma is at least as fast as mul + add",
  /*GenPredicate=*/0
>;

defm FastDenormalF32 : AMDGPUSubtargetFeature<"fast-denormal-f32",
  "Enabling denormals does not cause f32 instructions to run at f64 rates",
  /*GenPredicate=*/0
>;

defm MIMG_R128 : AMDGPUSubtargetFeature<"mimg-r128",
  "Support 128-bit texture resources",
  /*GenPredicate=*/0
>;

defm HalfRate64Ops : AMDGPUSubtargetFeature<"half-rate-64-ops",
  "Most fp64 instructions are half rate instead of quarter",
  /*GenPredicate=*/0
>;

defm FullRate64Ops : AMDGPUSubtargetFeature<"full-rate-64-ops",
  "Most fp64 instructions are full rate",
  /*GenPredicate=*/0
>;

defm FlatAddressSpace : AMDGPUSubtargetFeature<"flat-address-space",
  "Support flat address space"
>;

defm FlatInstOffsets : AMDGPUSubtargetFeature<"flat-inst-offsets",
  "Flat instructions have immediate offset addressing mode"
>;

defm FlatGlobalInsts : AMDGPUSubtargetFeature<"flat-global-insts",
  "Have global_* flat memory instructions",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm FlatScratchInsts : AMDGPUSubtargetFeature<"flat-scratch-insts",
  "Have scratch_* flat memory instructions",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm ScalarFlatScratchInsts : AMDGPUSubtargetFeature<"scalar-flat-scratch-insts",
  "Have s_scratch_* flat memory instructions"
>;

def FeatureEnableFlatScratch : SubtargetFeature<"enable-flat-scratch",
  "EnableFlatScratch",
  "true",
  "Use scratch_* flat memory instructions to access scratch"
>;

defm FlatGVSMode : AMDGPUSubtargetFeature<"flat-gvs-mode",
  "Have GVS addressing mode with flat_* instructions",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm AddNoCarryInsts : AMDGPUSubtargetFeature<"add-no-carry-insts",
  "Have VALU add/sub instructions without carry out"
>;

defm UnalignedBufferAccess : AMDGPUSubtargetFeature<"unaligned-buffer-access",
  "Hardware supports unaligned global loads and stores"
>;

defm TrapHandler: AMDGPUSubtargetFeature<"trap-handler",
  "Trap handler support",
  /*GenPredicate=*/0
>;

defm UnalignedScratchAccess : AMDGPUSubtargetFeature<"unaligned-scratch-access",
  "Support unaligned scratch loads and stores"
>;

defm UnalignedDSAccess : AMDGPUSubtargetFeature<"unaligned-ds-access",
  "Hardware supports unaligned local and region loads and stores"
>;

defm RelaxedBufferOOBMode : AMDGPUSubtargetFeature<"relaxed-buffer-oob-mode",
  "Disable strict out-of-bounds buffer guarantees. An OOB access may potentially"
  "cause an adjacent access to be treated as if it were also OOB"
>;

defm ApertureRegs : AMDGPUSubtargetFeature<"aperture-regs",
  "Has Memory Aperture Base and Size Registers",
  /*GenPredicate=*/0
>;

defm MadMixInsts : AMDGPUSubtargetFeature<"mad-mix-insts",
  "Has v_mad_mix_f32, v_mad_mixlo_f16, v_mad_mixhi_f16 instructions"
>;

defm FmaMixInsts : AMDGPUSubtargetFeature<"fma-mix-insts",
  "Has v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16 instructions"
>;

defm FmaMixBF16Insts : AMDGPUSubtargetFeature<"fma-mix-bf16-insts",
  "Has v_fma_mix_f32_bf16, v_fma_mixlo_bf16, v_fma_mixhi_bf16 instructions"
>;

defm IEEEMinimumMaximumInsts : AMDGPUSubtargetFeature<"ieee-minimum-maximum-insts",
  "Has v_minimum/maximum_f16/f32/f64, v_minimummaximum/maximumminimum_f16/f32 and"
  "v_pk_minimum/maximum_f16 instructions"
>;

defm Minimum3Maximum3F32 : AMDGPUSubtargetFeature<"minimum3-maximum3-f32",
  "Has v_minimum3_f32 and v_maximum3_f32 instructions"
>;

defm Minimum3Maximum3F16 : AMDGPUSubtargetFeature<"minimum3-maximum3-f16",
  "Has v_minimum3_f16 and v_maximum3_f16 instructions"
>;

defm Min3Max3PKF16 : AMDGPUSubtargetFeature<"min3-max3-pkf16",
  "Has v_pk_min3_num_f16 and v_pk_max3_num_f16 instructions"
>;

defm Minimum3Maximum3PKF16 : AMDGPUSubtargetFeature<"minimum3-maximum3-pkf16",
  "Has v_pk_minimum3_f16 and v_pk_maximum3_f16 instructions"
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

def FeaturePreciseMemory
    : SubtargetFeature<"precise-memory", "EnablePreciseMemory",
                       "true", "Enable precise memory mode">;

defm SGPRInitBug : AMDGPUSubtargetFeature<"sgpr-init-bug",
  "VI SGPR initialization bug requiring a fixed SGPR allocation size"
>;

defm UserSGPRInit16Bug : AMDGPUSubtargetFeature<"user-sgpr-init16-bug",
  "Bug requiring at least 16 user+system SGPRs to be enabled",
  /*GenPredicate=*/0
>;

defm LDSMisalignedBug : AMDGPUSubtargetFeature<"lds-misaligned-bug",
  "Some GFX10 bug with multi-dword LDS and flat access that is not naturally aligned in WGP mode",
  /*GenPredicate=*/0
>;

defm MFMAInlineLiteralBug : AMDGPUSubtargetFeature<"mfma-inline-literal-bug",
  "MFMA cannot use inline literal as SrcC",
  /*GenPredicate=*/0
>;

defm VcmpxPermlaneHazard : AMDGPUSubtargetFeature<"vcmpx-permlane-hazard",
  "TODO: describe me",
  /*GenPredicate=*/0
>;

defm VMEMtoScalarWriteHazard : AMDGPUSubtargetFeature<"vmem-to-scalar-write-hazard",
  "VMEM instruction followed by scalar writing to EXEC mask, M0 or SGPR leads to incorrect execution.",
  /*GenPredicate=*/0
>;

defm SMEMtoVectorWriteHazard : AMDGPUSubtargetFeature<"smem-to-vector-write-hazard",
  "s_load_dword followed by v_cmp page faults",
  /*GenPredicate=*/0
>;

defm InstFwdPrefetchBug : AMDGPUSubtargetFeature<"inst-fwd-prefetch-bug",
  "S_INST_PREFETCH instruction causes shader to hang",
  /*GenPredicate=*/0
>;

defm VmemPrefInsts : AMDGPUSubtargetFeature<"vmem-pref-insts",
  "Has flat_prefect_b8 and global_prefetch_b8 instructions"
>;

defm SafeSmemPrefetch : AMDGPUSubtargetFeature<"safe-smem-prefetch",
  "SMEM prefetches do not fail on illegal address",
  /*GenPredicate=*/0
>;

defm SafeCUPrefetch : AMDGPUSubtargetFeature<"safe-cu-prefetch",
  "VMEM CU scope prefetches do not fail on illegal address",
  /*GenPredicate=*/0
>;

defm VcmpxExecWARHazard : AMDGPUSubtargetFeature<"vcmpx-exec-war-hazard",
  "V_CMPX WAR hazard on EXEC (V_CMPX issue ONLY)",
  /*GenPredicate=*/0
>;

defm LdsBranchVmemWARHazard : AMDGPUSubtargetFeature<"lds-branch-vmem-war-hazard",
  "Switching between LDS and VMEM-tex not waiting VM_VSRC=0",
  /*GenPredicate=*/0
>;

class FeatureMaxHardClauseLength<int size> : SubtargetFeature<
  "max-hard-clause-length-"#size,
  "MaxHardClauseLength",
  !cast<string>(size),
  "Maximum number of instructions in an explicit S_CLAUSE is "#size
>;

/// Work around a hardware bug on some chips that can be triggered
/// under certain circumstances when clauses are longer than 32 operations.
def FeatureMaxHardClauseLength32 : FeatureMaxHardClauseLength<32>;
/// While the S_CLAUSE instruction permits encoding clause lengths up to 64,
/// hardware documentation for gfx10+ indicates that 63 is the maximum
/// permitted clause length.
def FeatureMaxHardClauseLength63 : FeatureMaxHardClauseLength<63>;

defm NSAtoVMEMBug : AMDGPUSubtargetFeature<"nsa-to-vmem-bug",
  "MIMG-NSA followed by VMEM fail if EXEC_LO or EXEC_HI equals zero",
  /*GenPredicate=*/0
>;

defm NSAClauseBug : AMDGPUSubtargetFeature<"nsa-clause-bug",
  "MIMG-NSA in a hard clause has unpredictable results on GFX10.1",
  /*GenPredicate=*/0
>;

defm FlatSegmentOffsetBug : AMDGPUSubtargetFeature<"flat-segment-offset-bug",
  "GFX10 bug where inst_offset is ignored when flat instructions access global memory",
  /*GenPredicate=*/0
>;

defm NegativeScratchOffsetBug : AMDGPUSubtargetFeature<"negative-scratch-offset-bug",
  "Negative immediate offsets in scratch instructions with an SGPR offset page fault on GFX9"
>;

defm NegativeUnalignedScratchOffsetBug : AMDGPUSubtargetFeature<"negative-unaligned-scratch-offset-bug",
  "Scratch instructions with a VGPR offset and a negative immediate offset that"
  "is not a multiple of 4 read wrong memory on GFX10",
  /*GenPredicate=*/0
>;

defm Offset3fBug : AMDGPUSubtargetFeature<"offset-3f-bug",
  "Branch offset of 3f hardware bug",
  /*GenPredicate=*/0
>;

defm ImageStoreD16Bug : AMDGPUSubtargetFeature<"image-store-d16-bug",
  "Image Store D16 hardware bug",
  /*GenPredicate=*/0
>;

defm ImageGather4D16Bug : AMDGPUSubtargetFeature<"image-gather4-d16-bug",
  "Image Gather4 D16 hardware bug",
  /*GenPredicate=*/0
>;

defm MADIntraFwdBug : AMDGPUSubtargetFeature<"mad-intra-fwd-bug",
  "MAD_U64/I64 intra instruction forwarding bug",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

defm MSAALoadDstSelBug : AMDGPUSubtargetFeature<"msaa-load-dst-sel-bug",
  "MSAA loads not honoring dst_sel bug",
  /*GenPredicate=*/0
>;

defm PrivEnabledTrap2NopBug : AMDGPUSubtargetFeature<"priv-enabled-trap2-nop-bug",
  "Hardware that runs with PRIV=1 interpreting 's_trap 2' as a nop bug",
  /*GenPredicate=*/0
>;

class SubtargetFeatureLDSBankCount <int Value> : SubtargetFeature <
  "ldsbankcount"#Value,
  "LDSBankCount",
  !cast<string>(Value),
  "The number of LDS banks per compute unit."
>;

def FeatureLDSBankCount16 : SubtargetFeatureLDSBankCount<16>;
def FeatureLDSBankCount32 : SubtargetFeatureLDSBankCount<32>;

defm GCN3Encoding : AMDGPUSubtargetFeature<"gcn3-encoding",
  "Encoding format for VI",
  /*GenPredicate=*/0
>;

defm CIInsts : AMDGPUSubtargetFeature<"ci-insts",
  "Additional instructions for CI+",
  /*GenPredicate=*/0
>;

defm GFX8Insts : AMDGPUSubtargetFeature<"gfx8-insts",
  "Additional instructions for GFX8+",
  /*GenPredicate=*/0
>;

defm GFX9Insts : AMDGPUSubtargetFeature<"gfx9-insts",
  "Additional instructions for GFX9+",
  /*GenPredicate=*/0
>;

def FeatureRequiresAlignedVGPRs : SubtargetFeature<"vgpr-align2",
  "RequiresAlignVGPR",
  "true",
  "VGPR and AGPR tuple operands require even alignment"
>;

defm GFX90AInsts : AMDGPUSubtargetFeature<"gfx90a-insts",
  "Additional instructions for GFX90A+",
  /*GenPredicate=*/0
>;

defm GFX940Insts : AMDGPUSubtargetFeature<"gfx940-insts",
  "Additional instructions for GFX940+",
  /*GenPredicate=*/0
>;

defm Permlane16Swap : AMDGPUSubtargetFeature<"permlane16-swap",
  "Has v_permlane16_swap_b32 instructions"
>;

defm Permlane32Swap : AMDGPUSubtargetFeature<"permlane32-swap",
  "Has v_permlane32_swap_b32 instructions"
>;

defm FP8ConversionScaleInsts : AMDGPUSubtargetFeature<"fp8-cvt-scale-insts",
  "Has fp8 conversion scale instructions"
>;

defm BF8ConversionScaleInsts : AMDGPUSubtargetFeature<"bf8-cvt-scale-insts",
  "Has bf8 conversion scale instructions"
>;

defm FP4ConversionScaleInsts : AMDGPUSubtargetFeature<"fp4-cvt-scale-insts",
  "Has fp4 conversion scale instructions"
>;

defm FP6BF6ConversionScaleInsts : AMDGPUSubtargetFeature<"fp6bf6-cvt-scale-insts",
  "Has fp6 and bf6 conversion scale instructions"
>;

defm F16BF16ToFP6BF6ConversionScaleInsts : AMDGPUSubtargetFeature<"f16bf16-to-fp6bf6-cvt-scale-insts",
  "Has f16bf16 to fp6bf6 conversion scale instructions"
>;

defm F32ToF16BF16ConversionSRInsts : AMDGPUSubtargetFeature<"f32-to-f16bf16-cvt-sr-insts",
  "Has f32 to f16bf16 conversion scale instructions"
>;

defm AshrPkInsts : AMDGPUSubtargetFeature<"ashr-pk-insts",
  "Has Arithmetic Shift Pack instructions"
>;

defm CvtPkF16F32Inst : AMDGPUSubtargetFeature<"cvt-pk-f16-f32-inst",
  "Has cvt_pk_f16_f32 instruction"
>;

defm McastLoadInsts : AMDGPUSubtargetFeature<"mcast-load-insts",
  "Has multicast load instructions"
>;

defm GFX950Insts : AMDGPUSubtargetFeature<"gfx950-insts",
  "Additional instructions for GFX950+",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeaturePermlane16Swap,
   FeaturePermlane32Swap,
   FeatureAshrPkInsts,
   FeatureFP8ConversionScaleInsts,
   FeatureBF8ConversionScaleInsts,
   FeatureFP4ConversionScaleInsts,
   FeatureFP6BF6ConversionScaleInsts,
   FeatureF16BF16ToFP6BF6ConversionScaleInsts,
   FeatureF32ToF16BF16ConversionSRInsts,
   FeatureCvtPkF16F32Inst,
   FeatureMinimum3Maximum3F32,
   FeatureMinimum3Maximum3PKF16,
   ]
>;

defm GFX10Insts : AMDGPUSubtargetFeature<"gfx10-insts",
  "Additional instructions for GFX10+",
  /*GenPredicate=*/0
>;

defm GFX11Insts : AMDGPUSubtargetFeature<"gfx11-insts",
  "Additional instructions for GFX11+",
  /*GenPredicate=*/0
>;

defm GFX12Insts : AMDGPUSubtargetFeature<"gfx12-insts",
  "Additional instructions for GFX12+",
  /*GenPredicate=*/0
>;

defm GFX1250Insts : AMDGPUSubtargetFeature<"gfx1250-insts",
  "Additional instructions for GFX1250+",
  /*GenPredicate=*/0
>;

defm GFX13Insts : AMDGPUSubtargetFeature<"gfx13-insts",
  "Additional instructions for GFX13+",
  /*GenPredicate=*/0
>;

defm GFX10_3Insts : AMDGPUSubtargetFeature<"gfx10-3-insts",
  "Additional instructions for GFX10.3",
  /*GenPredicate=*/0
>;

defm GFX7GFX8GFX9Insts : AMDGPUSubtargetFeature<"gfx7-gfx8-gfx9-insts",
  "Instructions shared in GFX7, GFX8, GFX9",
  /*GenPredicate=*/0
>;

defm SMemRealTime : AMDGPUSubtargetFeature<"s-memrealtime",
  "Has s_memrealtime instruction"
>;

defm Inv2PiInlineImm : AMDGPUSubtargetFeature<"inv-2pi-inline-imm",
  "Has 1 / (2 * pi) as inline immediate",
  /*GenPredicate=*/0
>;

defm 16BitInsts : AMDGPUSubtargetFeature<"16-bit-insts",
  "Has i16/f16 instructions"
>;

defm True16BitInsts : AMDGPUSubtargetFeature<"true16",
  "True 16-bit operand instructions"
>;

def FeatureRealTrue16Insts : SubtargetFeature<"real-true16",
  "EnableRealTrue16Insts",
  "true",
  "Use true 16-bit registers"
>;

defm D16Writes32BitVgpr : AMDGPUSubtargetFeature<"d16-write-vgpr32",
  "D16 instructions potentially have 32-bit data dependencies"
>;

defm BF16TransInsts : AMDGPUSubtargetFeature<"bf16-trans-insts",
  "Has bf16 transcendental instructions"
>;

defm BF16ConversionInsts : AMDGPUSubtargetFeature<"bf16-cvt-insts",
  "Has bf16 conversion instructions"
>;

defm BF16PackedInsts : AMDGPUSubtargetFeature<"bf16-pk-insts",
  "Has bf16 packed instructions (fma, add, mul, max, min)"
>;

defm VOP3PInsts : AMDGPUSubtargetFeature<"vop3p",
  "Has VOP3P packed instructions"
>;

defm Movrel : AMDGPUSubtargetFeature<"movrel",
  "Has v_movrel*_b32 instructions"
>;

defm VGPRIndexMode : AMDGPUSubtargetFeature<"vgpr-index-mode",
  "Has VGPR mode register indexing"
>;

defm ScalarDwordx3Loads : AMDGPUSubtargetFeature<"scalar-dwordx3-loads",
  "Has 96-bit scalar load instructions"
>;

defm ScalarStores : AMDGPUSubtargetFeature<"scalar-stores",
  "Has store scalar memory instructions"
>;

defm ScalarAtomics : AMDGPUSubtargetFeature<"scalar-atomics",
  "Has atomic scalar memory instructions"
>;

defm SDWA : AMDGPUSubtargetFeature<"sdwa",
  "Support SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

defm SDWAOmod : AMDGPUSubtargetFeature<"sdwa-omod",
  "Support OMod with SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/0
>;

defm SDWAScalar : AMDGPUSubtargetFeature<"sdwa-scalar",
  "Support scalar register with SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/0
>;

defm SDWASdst : AMDGPUSubtargetFeature<"sdwa-sdst",
  "Support scalar dst for VOPC with SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/0
>;

defm SDWAMac : AMDGPUSubtargetFeature<"sdwa-mav",
  "Support v_mac_f32/f16 with SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/0
>;

defm SDWAOutModsVOPC : AMDGPUSubtargetFeature<"sdwa-out-mods-vopc",
  "Support clamp for VOPC with SDWA (Sub-DWORD Addressing) extension",
  /*GenPredicate=*/0
>;

def FeatureDPP : SubtargetFeature<"dpp",
  "HasDPP",
  "true",
  "Support DPP (Data Parallel Primitives) extension"
>;

// DPP8 allows arbitrary cross-lane swizzling within groups of 8 lanes.
def FeatureDPP8 : SubtargetFeature<"dpp8",
  "HasDPP8",
  "true",
  "Support DPP8 (Data Parallel Primitives) extension"
>;

defm DPALU_DPP : AMDGPUSubtargetFeature<"dpp-64bit",
  "Support DPP (Data Parallel Primitives) extension in DP ALU"
>;

defm DPPSrc1SGPR : AMDGPUSubtargetFeature<"dpp-src1-sgpr",
  "Support SGPR for Src1 of DPP instructions",
  /*GenPredicate=*/0
>;

defm PackedFP32Ops : AMDGPUSubtargetFeature<"packed-fp32-ops",
  "Support packed fp32 instructions"
>;

defm R128A16 : AMDGPUSubtargetFeature<"r128-a16",
  "Support gfx9-style A16 for 16-bit coordinates/gradients/lod/clamp/mip image "
  "operands, where a16 is aliased with r128"
>;

defm A16 : AMDGPUSubtargetFeature<"a16",
  "Support A16 for 16-bit coordinates/gradients/lod/clamp/mip image operands"
>;

defm G16 : AMDGPUSubtargetFeature<"g16",
  "Support G16 for 16-bit gradient image operands"
>;

defm NSAEncoding : AMDGPUSubtargetFeature<"nsa-encoding",
  "Support NSA encoding for image instructions",
  /*GenPredicate=*/0
>;

defm PartialNSAEncoding : AMDGPUSubtargetFeature<"partial-nsa-encoding",
  "Support partial NSA encoding for image instructions",
  /*GenPredicate=*/0
>;

defm ImageInsts : AMDGPUSubtargetFeature<"image-insts",
  "Support image instructions"
>;

defm ExtendedImageInsts : AMDGPUSubtargetFeature<"extended-image-insts",
  "Support mips != 0, lod != 0, gather4, and get_lod"
>;

defm GFX10_AEncoding : AMDGPUSubtargetFeature<"gfx10_a-encoding",
  "Has BVH ray tracing instructions",
  /*GenPredicate=*/0
>;

defm GFX10_BEncoding : AMDGPUSubtargetFeature<"gfx10_b-encoding",
  "Encoding format GFX10_B",
  /*GenPredicate=*/0
>;

defm IntClamp : AMDGPUSubtargetFeature<"int-clamp-insts",
  "Support clamp for integer destination"
>;

defm UnpackedD16VMem : AMDGPUSubtargetFeature<"unpacked-d16-vmem",
  "Has unpacked d16 vmem instructions"
>;

defm DLInsts : AMDGPUSubtargetFeature<"dl-insts",
  "Has v_fmac_f32 and v_xnor_b32 instructions"
>;

defm FmacF64Inst : AMDGPUSubtargetFeature<"fmacf64-inst",
  "Has v_fmac_f64 instruction"
>;

defm Dot1Insts : AMDGPUSubtargetFeature<"dot1-insts",
  "Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions"
>;

defm Dot2Insts : AMDGPUSubtargetFeature<"dot2-insts",
  "Has v_dot2_i32_i16, v_dot2_u32_u16 instructions"
>;

defm Dot3Insts : AMDGPUSubtargetFeature<"dot3-insts",
  "Has v_dot8c_i32_i4 instruction"
>;

defm Dot4Insts : AMDGPUSubtargetFeature<"dot4-insts",
  "Has v_dot2c_i32_i16 instruction"
>;

defm Dot5Insts : AMDGPUSubtargetFeature<"dot5-insts",
  "Has v_dot2c_f32_f16 instruction"
>;

defm Dot6Insts : AMDGPUSubtargetFeature<"dot6-insts",
  "Has v_dot4c_i32_i8 instruction"
>;

defm Dot7Insts : AMDGPUSubtargetFeature<"dot7-insts",
  "Has v_dot4_u32_u8, v_dot8_u32_u4 instructions"
>;

defm Dot8Insts : AMDGPUSubtargetFeature<"dot8-insts",
  "Has v_dot4_i32_iu8, v_dot8_i32_iu4 instructions"
>;

defm Dot9Insts : AMDGPUSubtargetFeature<"dot9-insts",
  "Has v_dot2_f16_f16, v_dot2_bf16_bf16 instructions"
>;

defm Dot10Insts : AMDGPUSubtargetFeature<"dot10-insts",
  "Has v_dot2_f32_f16 instruction"
>;

defm Dot11Insts : AMDGPUSubtargetFeature<"dot11-insts",
  "Has v_dot4_f32_fp8_fp8, v_dot4_f32_fp8_bf8, v_dot4_f32_bf8_fp8, v_dot4_f32_bf8_bf8 instructions"
>;

defm Dot12Insts : AMDGPUSubtargetFeature<"dot12-insts",
  "Has v_dot2_f32_bf16 instructions"
>;

defm Dot13Insts : AMDGPUSubtargetFeature<"dot13-insts",
  "Has v_dot2c_f32_bf16 instructions"
>;

defm MAIInsts : AMDGPUSubtargetFeature<"mai-insts",
  "Has mAI instructions"
>;

defm FP8Insts : AMDGPUSubtargetFeature<"fp8-insts",
  "Has fp8 and bf8 instructions"
>;

defm FP8ConversionInsts : AMDGPUSubtargetFeature<"fp8-conversion-insts",
  "Has fp8 and bf8 conversion instructions"
>;

defm FP8E5M3Insts : AMDGPUSubtargetFeature<"fp8e5m3-insts",
  "Has fp8 e5m3 format support"
>;

defm CvtFP8VOP1Bug : AMDGPUSubtargetFeature<"cvt-fp8-vop1-bug",
  "FP8/BF8 VOP1 form of conversion to F32 is unreliable",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0,
  [FeatureFP8ConversionInsts]
>;

defm PkFmacF16Inst : AMDGPUSubtargetFeature<"pk-fmac-f16-inst",
  "Has v_pk_fmac_f16 instruction"
>;

defm CubeInsts : AMDGPUSubtargetFeature<"cube-insts",
  "Has v_cube* instructions"
>;

defm LerpInst : AMDGPUSubtargetFeature<"lerp-inst",
  "Has v_lerp_u8 instruction"
>;

defm SadInsts : AMDGPUSubtargetFeature<"sad-insts",
  "Has v_sad* instructions"
>;

defm QsadInsts : AMDGPUSubtargetFeature<"qsad-insts",
  "Has v_qsad* instructions"
>;

defm CvtNormInsts : AMDGPUSubtargetFeature<"cvt-norm-insts",
  "Has v_cvt_norm* instructions"
>;

defm CvtPkNormVOP2Insts : AMDGPUSubtargetFeature<"cvt-pknorm-vop2-insts",
  "Has v_cvt_pk_norm_*f32 instructions/Has v_cvt_pk_norm_*_f16 instructions"
>;

defm CvtPkNormVOP3Insts : AMDGPUSubtargetFeature<"cvt-pknorm-vop3-insts",
  "Has v_cvt_pk_norm_*f32 instructions/Has v_cvt_pk_norm_*_f16 instructions"
>;

defm AtomicDsPkAdd16Insts : AMDGPUSubtargetFeature<"atomic-ds-pk-add-16-insts",
  "Has ds_pk_add_bf16, ds_pk_add_f16, ds_pk_add_rtn_bf16, "
  "ds_pk_add_rtn_f16 instructions"
>;

defm AtomicFlatPkAdd16Insts : AMDGPUSubtargetFeature<"atomic-flat-pk-add-16-insts",
  "Has flat_atomic_pk_add_f16 and flat_atomic_pk_add_bf16 instructions"
>;

defm AtomicFaddRtnInsts : AMDGPUSubtargetFeature<"atomic-fadd-rtn-insts",
  "Has buffer_atomic_add_f32 and global_atomic_add_f32 instructions that "
  "return original value",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatGlobalInsts]
>;

defm AtomicFMinFMaxF32GlobalInsts : AMDGPUSubtargetFeature<"atomic-fmin-fmax-global-f32",
  "Has global/buffer instructions for atomicrmw fmin/fmax for float"
>;

defm AtomicFMinFMaxF64GlobalInsts : AMDGPUSubtargetFeature<"atomic-fmin-fmax-global-f64",
  "Has global/buffer instructions for atomicrmw fmin/fmax for float"
>;

defm AtomicFMinFMaxF32FlatInsts : AMDGPUSubtargetFeature<"atomic-fmin-fmax-flat-f32",
  "Has flat memory instructions for atomicrmw fmin/fmax for float",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm AtomicFMinFMaxF64FlatInsts : AMDGPUSubtargetFeature<"atomic-fmin-fmax-flat-f64",
  "Has flat memory instructions for atomicrmw fmin/fmax for double",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm AtomicFaddNoRtnInsts : AMDGPUSubtargetFeature<"atomic-fadd-no-rtn-insts",
  "Has buffer_atomic_add_f32 and global_atomic_add_f32 instructions that "
  "don't return original value",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatGlobalInsts]
>;

def FeatureAtomicBufferGlobalPkAddF16NoRtnInsts
  : SubtargetFeature<"atomic-buffer-global-pk-add-f16-no-rtn-insts",
  "HasAtomicBufferGlobalPkAddF16NoRtnInsts",
  "true",
  "Has buffer_atomic_pk_add_f16 and global_atomic_pk_add_f16 instructions that "
  "don't return original value",
  [FeatureFlatGlobalInsts]
>;

defm AtomicBufferGlobalPkAddF16Insts : AMDGPUSubtargetFeature<"atomic-buffer-global-pk-add-f16-insts",
  "Has buffer_atomic_pk_add_f16 and global_atomic_pk_add_f16 instructions that "
  "can return original value",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatGlobalInsts]
>;

defm AtomicGlobalPkAddBF16Inst : AMDGPUSubtargetFeature<"atomic-global-pk-add-bf16-inst",
  "Has global_atomic_pk_add_bf16 instruction",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatGlobalInsts]
>;

defm AtomicBufferPkAddBF16Inst : AMDGPUSubtargetFeature<"atomic-buffer-pk-add-bf16-inst",
  "Has buffer_atomic_pk_add_bf16 instruction"
>;

defm AtomicCSubNoRtnInsts : AMDGPUSubtargetFeature<"atomic-csub-no-rtn-insts",
  "Has buffer_atomic_csub and global_atomic_csub instructions that don't "
  "return original value",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

defm FlatAtomicFaddF32Inst : AMDGPUSubtargetFeature<"flat-atomic-fadd-f32-inst",
  "Has flat_atomic_add_f32 instruction",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/1,
  [FeatureFlatAddressSpace]
>;

defm FlatBufferGlobalAtomicFaddF64Inst : AMDGPUSubtargetFeature<"flat-buffer-global-fadd-f64-inst",
  "Has flat, buffer, and global instructions for f64 atomic fadd"
>;

def FeatureMemoryAtomicFAddF32DenormalSupport
  : SubtargetFeature<"memory-atomic-fadd-f32-denormal-support",
  "HasMemoryAtomicFaddF32DenormalSupport",
  "true",
  "global/flat/buffer atomic fadd for float supports denormal handling"
>;

defm AgentScopeFineGrainedRemoteMemoryAtomics : AMDGPUSubtargetFeature<
  "agent-scope-fine-grained-remote-memory-atomics",
  "Agent (device) scoped atomic operations, excluding those directly "
  "supported by PCIe (i.e. integer atomic add, exchange, and "
  "compare-and-swap), are functional for allocations in host or peer "
  "device memory.",
  /*GenPredicate=*/0
>;

defm EmulatedSystemScopeAtomics : AMDGPUSubtargetFeature<
  "emulated-system-scope-atomics",
  "System scope atomics unsupported by the PCI-e are emulated in HW via CAS "
  "loop and functional.",
  /*GenPredicate=*/0
>;

defm DefaultComponentZero : AMDGPUSubtargetFeature<"default-component-zero",
  "BUFFER/IMAGE store instructions set unspecified components to zero (before GFX12)"
>;

defm DefaultComponentBroadcast : AMDGPUSubtargetFeature<"default-component-broadcast",
  "BUFFER/IMAGE store instructions set unspecified components to x component (GFX12)"
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

defm NoSdstCMPX : AMDGPUSubtargetFeature<"no-sdst-cmpx",
  "V_CMPX does not write VCC/SGPR in addition to EXEC"
>;

defm Vscnt : AMDGPUSubtargetFeature<"vscnt",
  "Has separate store vscnt counter",
  /*GenPredicate=*/0
>;

defm GetWaveIdInst : AMDGPUSubtargetFeature<"get-wave-id-inst",
  "Has s_get_waveid_in_workgroup instruction"
>;

defm SMemTimeInst : AMDGPUSubtargetFeature<"s-memtime-inst",
  "Has s_memtime instruction"
>;

defm ShaderCyclesRegister : AMDGPUSubtargetFeature<"shader-cycles-register",
  "Has SHADER_CYCLES hardware register"
>;

defm ShaderCyclesHiLoRegisters : AMDGPUSubtargetFeature<"shader-cycles-hi-lo-registers",
  "Has SHADER_CYCLES_HI/LO hardware registers",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

defm MadMacF32Insts : AMDGPUSubtargetFeature<"mad-mac-f32-insts",
  "Has v_mad_f32/v_mac_f32/v_madak_f32/v_madmk_f32 instructions"
>;

defm DsSrc2Insts : AMDGPUSubtargetFeature<"ds-src2-insts",
  "Has ds_*_src2 instructions"
>;

defm VOP3Literal : AMDGPUSubtargetFeature<"vop3-literal",
  "Can use one literal in VOP3",
  /*GenPredicate=*/0
>;

defm NoDataDepHazard : AMDGPUSubtargetFeature<"no-data-dep-hazard",
  "Does not need SW waitstates",
  /*GenPredicate=*/0
>;

// Allocate 1536 VGPRs for wave32 and 768 VGPRs for wave64
// with allocation granularity 24 for wave32 and 12 for wave64
defm 1_5xVGPRs : AMDGPUSubtargetFeature<"allocate1_5xvgprs",
  "Has 50% more physical VGPRs and 50% larger allocation granule",
  /*GenPredicate=*/0
>;

defm VOPDInsts : AMDGPUSubtargetFeature<"vopd",
  "Has VOPD dual issue wave32 instructions",
  /*GenPredicate=*/0
>;

defm VALUTransUseHazard : AMDGPUSubtargetFeature<"valu-trans-use-hazard",
  "Hazard when TRANS instructions are closely followed by a use of the result",
  /*GenPredicate=*/0
>;

defm SALUFloatInsts : AMDGPUSubtargetFeature<"salu-float",
  "Has SALU floating point instructions"
>;

defm PseudoScalarTrans : AMDGPUSubtargetFeature<"pseudo-scalar-trans",
  "Has Pseudo Scalar Transcendental instructions"
>;

defm RestrictedSOffset : AMDGPUSubtargetFeature<"restricted-soffset",
  "Has restricted SOffset (immediate not supported)."
>;

defm RequiredExportPriority : AMDGPUSubtargetFeature<"required-export-priority",
  "Export priority must be explicitly manipulated on GFX11.5",
  /*GenPredicate=*/0
>;

defm VmemWriteVgprInOrder : AMDGPUSubtargetFeature<"vmem-write-vgpr-in-order",
  "VMEM instructions of the same type write VGPR results in order",
  /*GenPredicate=*/0
>;

defm BitOp3Insts : AMDGPUSubtargetFeature<"bitop3-insts",
  "Has v_bitop3_b32/v_bitop3_b16 instructions"
>;

defm TanhInsts : AMDGPUSubtargetFeature<"tanh-insts",
  "Has v_tanh_f32/f16 instructions"
>;

defm TensorCvtLutInsts : AMDGPUSubtargetFeature<"tensor-cvt-lut-insts",
  "Has v_perm_pk16* instructions"
>;

defm TransposeLoadF4F6Insts : AMDGPUSubtargetFeature<"transpose-load-f4f6-insts",
  "Has ds_load_tr4/tr6 and global_load_tr4/tr6 instructions"
>;

defm PrngInst : AMDGPUSubtargetFeature<"prng-inst",
  "Has v_prng_b32 instruction"
>;

defm BVHDualAndBVH8Insts : AMDGPUSubtargetFeature<"bvh-dual-bvh-8-insts",
  "Has image_bvh_dual_intersect_ray and image_bvh8_intersect_ray instructions"
>;

defm PointSampleAccel : AMDGPUSubtargetFeature<"point-sample-accel",
  "Has point sample acceleration feature",
  /*GenPredicate=*/0
>;

defm 64BitLiterals : AMDGPUSubtargetFeature<"64-bit-literals",
  "Can use 64-bit literals with single DWORD instructions"
>;

defm 1024AddressableVGPRs : AMDGPUSubtargetFeature<"1024-addressable-vgprs",
  "Has 1024 addressable VGPRs"
>;

defm SetregVGPRMSBFixup : AMDGPUSubtargetFeature<"setreg-vgpr-msb-fixup",
  "S_SETREG to MODE clobbers VGPR MSB bits, requires fixup",
  /*GenPredicate=*/0
>;

defm WaitXcnt : AMDGPUSubtargetFeature<"wait-xcnt",
  "Has s_wait_xcnt instruction"
>;

defm SetPrioIncWgInst : AMDGPUSubtargetFeature<"setprio-inc-wg-inst",
  "Has s_setprio_inc_wg instruction."
>;

defm SWakeupBarrier : AMDGPUSubtargetFeature<"s-wakeup-barrier-inst",
  "Has s_wakeup_barrier instruction."
>;

//===------------------------------------------------------------===//
// Subtarget Features (options and debugging)
//===------------------------------------------------------------===//

// Ugly hack to accomodate assembling modules with mixed
// wavesizes. Ideally we would have a mapping symbol in assembly which
// would keep track of which sections of code should be treated as
// wave32 and wave64. Instead what users do is assemble with both
// wavesizes enabled. We translate this into this special mode so this
// only influences assembler behavior and nothing else.
defm AssemblerPermissiveWavesize : AMDGPUSubtargetFeature<"assembler-permissive-wavesize",
  "Allow parsing wave32 and wave64 variants of instructions",
  /*GenPredicate=*/0
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

def FeatureUseFlatForGlobal : SubtargetFeature<"flat-for-global",
  "UseFlatForGlobal",
  "true",
  "Force to generate flat instruction for global"
>;

defm AutoWaitcntBeforeBarrier : AMDGPUSubtargetFeature <"auto-waitcnt-before-barrier",
  "Hardware automatically inserts waitcnt before barrier",
  /*GenPredicate=*/0
>;

defm BackOffBarrier : AMDGPUSubtargetFeature <"back-off-barrier",
  "Hardware supports backing off s_barrier if an exception occurs",
  /*GenPredicate=*/0
>;

defm TrigReducedRange : AMDGPUSubtargetFeature<"trig-reduced-range",
  "Requires use of fract on arguments to trig instructions",
  /*GenPredicate=*/0
>;

defm KernargPreload : AMDGPUSubtargetFeature <"kernarg-preload",
  "Hardware supports preloading of kernel arguments in user SGPRs.",
  /*GenPredicate=*/0
>;

// Alignment enforcement is controlled by a configuration register:
// SH_MEM_CONFIG.alignment_mode
defm UnalignedAccessMode : AMDGPUSubtargetFeature<"unaligned-access-mode",
  "Enable unaligned global, local and region loads and stores if the hardware"
  " supports it"
>;

defm PackedTID : AMDGPUSubtargetFeature<"packed-tid",
  "Workitem IDs are packed into v0 at kernel launch",
  /*GenPredicate=*/0
>;

defm ArchitectedFlatScratch : AMDGPUSubtargetFeature<"architected-flat-scratch",
  "Flat Scratch register is a readonly SPI initialized architected register",
  /*GenPredicate=*/0
>;

defm ArchitectedSGPRs : AMDGPUSubtargetFeature<"architected-sgprs",
  "Enable the architected SGPRs",
  /*GenPredicate=*/0
>;

defm GDS : AMDGPUSubtargetFeature<"gds",
  "Has Global Data Share",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

defm GWS : AMDGPUSubtargetFeature<"gws",
  "Has Global Wave Sync",
  /*GenPredicate=*/1,
  /*GenAssemblerPredicate=*/0
>;

def FeatureRequiresCOV6 : SubtargetFeature<"requires-cov6",
  "RequiresCOV6",
  "true",
  "Target Requires Code Object V6"
>;

defm XF32Insts : AMDGPUSubtargetFeature<"xf32-insts",
  "Has instructions that support xf32 format, such as "
  "v_mfma_f32_16x16x8_xf32 and v_mfma_f32_32x32x4_xf32"
>;

defm GloballyAddressableScratch : AMDGPUSubtargetFeature<"globally-addressable-scratch",
  "FLAT instructions can access scratch memory for any thread in any wave",
  /*GenPredicate=*/0
>;

// Enable the use of SCRATCH_STORE/LOAD_BLOCK instructions for saving and
// restoring the callee-saved registers.
def FeatureUseBlockVGPROpsForCSR : SubtargetFeature<"block-vgpr-csr",
  "UseBlockVGPROpsForCSR",
  "true",
  "Use block load/store for VGPR callee saved registers"
>;

defm LshlAddU64Inst : AMDGPUSubtargetFeature<"lshl-add-u64-inst",
  "Has v_lshl_add_u64 instruction"
>;

defm AddSubU64Insts : AMDGPUSubtargetFeature<"add-sub-u64-insts",
  "Has v_add_u64 and v_sub_u64 instructions"
>;

defm MadU32Inst : AMDGPUSubtargetFeature<"mad-u32-inst",
  "Has v_mad_u32 instruction"
>;

defm AddMinMaxInsts : AMDGPUSubtargetFeature<"add-min-max-insts",
  "Has v_add_{min|max}_{i|u}32 instructions"
>;

defm PkAddMinMaxInsts : AMDGPUSubtargetFeature<"pk-add-min-max-insts",
  "Has v_pk_add_{min|max}_{i|u}16 instructions"
>;

defm VMemToLDSLoad : AMDGPUSubtargetFeature<"vmem-to-lds-load-insts",
  "The platform has memory to lds instructions (global_load w/lds bit set, buffer_load"
  "w/lds bit set or global_load_lds. This does not include scratch_load_lds.",
  /*GenPredicate=*/0
>;

defm LdsBarrierArriveAtomic : AMDGPUSubtargetFeature<"lds-barrier-arrive-atomic",
  "Has LDS barrier-arrive atomic instructions"
>;

defm 45BitNumRecordsBufferResource : AMDGPUSubtargetFeature<"45-bit-num-records-buffer-resource",
  "The buffer resource (V#) supports 45-bit num_records",
  /*GenPredicate=*/0
>;

defm Clusters : AMDGPUSubtargetFeature<"clusters",
  "Has clusters of workgroups support",
  /*GenPredicate=*/0
>;

def FeatureWaitsBeforeSystemScopeStores : SubtargetFeature<
  "waits-before-system-scope-stores",
  "RequiresWaitsBeforeSystemScopeStores",
  "true",
  "Target requires waits for loads and atomics before system scope stores"
>;

def FeatureUseAddPC64Inst : SubtargetFeature<"use-add-pc64-inst",
  "UseAddPC64Inst",
  "true",
  "Use s_add_pc_i64 instruction."
>;

//===----------------------------------------------------------------------===//

class GCNSubtargetFeatureGeneration <string Value,
                                     string FeatureName,
                                     list<SubtargetFeature> Implies> :
        SubtargetFeatureGeneration <Value, FeatureName, "GCNSubtarget", Implies>;

def FeatureSouthernIslands : GCNSubtargetFeatureGeneration<"SOUTHERN_ISLANDS",
    "southern-islands",
  [FeatureFP64, FeatureAddressableLocalMemorySize32768, FeatureMIMG_R128,
  FeatureWavefrontSize64, FeatureSMemTimeInst, FeatureMadMacF32Insts,
  FeatureDsSrc2Insts, FeatureLDSBankCount32, FeatureMovrel,
  FeatureTrigReducedRange, FeatureExtendedImageInsts, FeatureImageInsts,
  FeatureGDS, FeatureGWS, FeatureDefaultComponentZero,
  FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF64GlobalInsts,
  FeatureVmemWriteVgprInOrder, FeatureCubeInsts, FeatureLerpInst,
  FeatureSadInsts, FeatureCvtPkNormVOP2Insts
  ]
>;

def FeatureSeaIslands : GCNSubtargetFeatureGeneration<"SEA_ISLANDS",
    "sea-islands",
  [FeatureFP64, FeatureAddressableLocalMemorySize65536, FeatureMIMG_R128,
  FeatureWavefrontSize64, FeatureFlatAddressSpace,
  FeatureCIInsts, FeatureMovrel, FeatureTrigReducedRange,
  FeatureGFX7GFX8GFX9Insts, FeatureSMemTimeInst, FeatureMadMacF32Insts,
  FeatureDsSrc2Insts, FeatureExtendedImageInsts, FeatureUnalignedBufferAccess,
  FeatureImageInsts, FeatureGDS, FeatureGWS, FeatureDefaultComponentZero,
  FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF64GlobalInsts,
  FeatureAtomicFMinFMaxF32FlatInsts, FeatureAtomicFMinFMaxF64FlatInsts,
  FeatureVmemWriteVgprInOrder, FeatureCubeInsts, FeatureLerpInst,
  FeatureSadInsts, FeatureQsadInsts, FeatureCvtPkNormVOP2Insts
  ]
>;

def FeatureVolcanicIslands : GCNSubtargetFeatureGeneration<"VOLCANIC_ISLANDS",
  "volcanic-islands",
  [FeatureFP64, FeatureAddressableLocalMemorySize65536, FeatureMIMG_R128,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureVGPRIndexMode, FeatureMovrel,
   FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureSDWA, FeatureSDWAOutModsVOPC, FeatureSDWAMac, FeatureDPP,
   FeatureIntClamp, FeatureTrigReducedRange, FeatureGFX8Insts,
   FeatureGFX7GFX8GFX9Insts, FeatureSMemTimeInst, FeatureMadMacF32Insts,
   FeatureDsSrc2Insts, FeatureExtendedImageInsts, FeatureFastDenormalF32,
   FeatureUnalignedBufferAccess, FeatureImageInsts, FeatureGDS, FeatureGWS,
   FeatureDefaultComponentZero, FeatureVmemWriteVgprInOrder, FeatureCubeInsts,
   FeatureLerpInst, FeatureSadInsts, FeatureQsadInsts,
   FeatureCvtPkNormVOP2Insts
  ]
>;

def FeatureGFX9 : GCNSubtargetFeatureGeneration<"GFX9",
  "gfx9",
  [FeatureFP64,
   FeatureWavefrontSize64, FeatureFlatAddressSpace,
   FeatureGCN3Encoding, FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureScalarStores, FeatureInv2PiInlineImm,
   FeatureApertureRegs, FeatureGFX9Insts, FeatureVOP3PInsts, FeatureVGPRIndexMode,
   FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureSDWA, FeatureSDWAOmod, FeatureSDWAScalar, FeatureSDWASdst,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureGFX8Insts, FeatureGFX7GFX8GFX9Insts,
   FeatureScalarFlatScratchInsts, FeatureScalarAtomics, FeatureR128A16,
   FeatureA16, FeatureSMemTimeInst, FeatureFastDenormalF32, FeatureSupportsXNACK,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureNegativeScratchOffsetBug, FeatureGWS,
   FeatureDefaultComponentZero,FeatureVmemWriteVgprInOrder, FeatureVMemToLDSLoad,
   FeatureCubeInsts, FeatureLerpInst, FeatureSadInsts, FeatureQsadInsts,
   FeatureCvtNormInsts, FeatureCvtPkNormVOP2Insts,
   FeatureCvtPkNormVOP3Insts
  ]
>;

def FeatureGFX10 : GCNSubtargetFeatureGeneration<"GFX10",
  "gfx10",
  [FeatureFP64, FeatureAddressableLocalMemorySize65536, FeatureMIMG_R128,
   FeatureFlatAddressSpace,
   FeatureCIInsts, Feature16BitInsts,
   FeatureSMemRealTime, FeatureInv2PiInlineImm,
   FeatureApertureRegs, FeatureGFX9Insts, FeatureGFX10Insts, FeatureVOP3PInsts,
   FeatureMovrel, FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureSDWA, FeatureSDWAOmod, FeatureSDWAScalar, FeatureSDWASdst,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureFmaMixInsts, FeatureGFX8Insts,
   FeatureNoSdstCMPX, FeatureVscnt,
   FeatureVOP3Literal, FeatureDPP8, FeatureExtendedImageInsts,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst,
   FeatureA16, FeatureSMemTimeInst, FeatureFastDenormalF32, FeatureG16,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureImageInsts, FeatureGDS, FeatureGWS,
   FeatureDefaultComponentZero, FeatureMaxHardClauseLength63,
   FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF64GlobalInsts,
   FeatureAtomicFMinFMaxF32FlatInsts, FeatureAtomicFMinFMaxF64FlatInsts,
   FeatureVmemWriteVgprInOrder, FeatureVMemToLDSLoad, FeatureCubeInsts,
   FeatureLerpInst, FeatureSadInsts, FeatureQsadInsts,
   FeatureCvtNormInsts, FeatureCvtPkNormVOP2Insts,
   FeatureCvtPkNormVOP3Insts
  ]
>;

def FeatureGFX11 : GCNSubtargetFeatureGeneration<"GFX11",
  "gfx11",
  [FeatureFP64, FeatureAddressableLocalMemorySize65536, FeatureMIMG_R128,
   FeatureFlatAddressSpace, Feature16BitInsts,
   FeatureInv2PiInlineImm, FeatureApertureRegs,
   FeatureCIInsts, FeatureGFX8Insts, FeatureGFX9Insts, FeatureGFX10Insts,
   FeatureGFX10_AEncoding, FeatureGFX10_BEncoding, FeatureGFX10_3Insts,
   FeatureGFX11Insts, FeatureVOP3PInsts, FeatureVOPDInsts, FeatureTrue16BitInsts,
   FeatureMovrel, FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureFmaMixInsts,
   FeatureNoSdstCMPX, FeatureVscnt,
   FeatureVOP3Literal, FeatureDPP8, FeatureExtendedImageInsts,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst,
   FeatureA16, FeatureFastDenormalF32, FeatureG16,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureGDS, FeatureGWS,
   FeatureDefaultComponentZero, FeatureMaxHardClauseLength32,
   FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF32FlatInsts,
   FeatureVmemWriteVgprInOrder, FeatureCubeInsts, FeatureLerpInst,
   FeatureSadInsts, FeatureQsadInsts, FeatureCvtNormInsts,
   FeatureCvtPkNormVOP2Insts, FeatureCvtPkNormVOP3Insts
  ]
>;

def FeatureGFX12 : GCNSubtargetFeatureGeneration<"GFX12",
  "gfx12",
  [FeatureFP64, FeatureMIMG_R128,
   FeatureFlatAddressSpace, Feature16BitInsts,
   FeatureInv2PiInlineImm, FeatureApertureRegs,
   FeatureCIInsts, FeatureGFX8Insts, FeatureGFX9Insts, FeatureGFX10Insts,
   FeatureGFX10_AEncoding, FeatureGFX10_BEncoding, FeatureGFX10_3Insts,
   FeatureGFX11Insts, FeatureGFX12Insts, FeatureVOP3PInsts, FeatureVOPDInsts,
   FeatureMovrel, FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureFmaMixInsts,
   FeatureNoSdstCMPX, FeatureVscnt,
   FeatureVOP3Literal, FeatureDPP8,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst,
   FeatureA16, FeatureFastDenormalF32, FeatureG16,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureTrue16BitInsts,
   FeatureDefaultComponentBroadcast, FeatureMaxHardClauseLength32,
   FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF32FlatInsts,
   FeatureIEEEMinimumMaximumInsts, FeatureMinimum3Maximum3F32,
   FeatureMinimum3Maximum3F16, FeatureAgentScopeFineGrainedRemoteMemoryAtomics
  ]
>;

def FeatureGFX13 : GCNSubtargetFeatureGeneration<"GFX13",
  "gfx13",
  [FeatureFP64, FeatureMIMG_R128,
   FeatureFlatAddressSpace, Feature16BitInsts,
   FeatureInv2PiInlineImm, FeatureApertureRegs,
   FeatureCIInsts, FeatureGFX8Insts, FeatureGFX9Insts, FeatureGFX10Insts,
   FeatureGFX10_AEncoding, FeatureGFX10_BEncoding, FeatureGFX10_3Insts,
   FeatureGFX11Insts, FeatureGFX12Insts, FeatureGFX13Insts, FeatureVOP3PInsts,
   FeatureVOPDInsts, FeatureMovrel, FeatureFastFMAF32, FeatureDPP, FeatureIntClamp,
   FeatureFlatInstOffsets, FeatureFlatGlobalInsts, FeatureFlatScratchInsts,
   FeatureAddNoCarryInsts, FeatureFmaMixInsts,
   FeatureNoSdstCMPX, FeatureVscnt,
   FeatureVOP3Literal, FeatureDPP8,
   FeatureNoDataDepHazard, FeaturePkFmacF16Inst,
   FeatureA16, FeatureFastDenormalF32, FeatureG16,
   FeatureUnalignedBufferAccess, FeatureUnalignedScratchAccess,
   FeatureUnalignedDSAccess, FeatureTrue16BitInsts,
   FeatureDefaultComponentBroadcast, FeatureMaxHardClauseLength32,
   FeatureAtomicFMinFMaxF32GlobalInsts, FeatureAtomicFMinFMaxF32FlatInsts,
   FeatureIEEEMinimumMaximumInsts, FeatureMinimum3Maximum3F32,
   FeatureMinimum3Maximum3F16, FeatureAgentScopeFineGrainedRemoteMemoryAtomics
  ]
>;
//===----------------------------------------------------------------------===//

class FeatureSet<list<SubtargetFeature> Features_> {
  list<SubtargetFeature> Features = Features_;
}

def FeatureISAVersion6_0_0 : FeatureSet<[FeatureSouthernIslands,
   FeatureFastFMAF32,
   FeatureHalfRate64Ops,
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
   FeatureHalfRate64Ops,
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

def FeatureISAVersion8_0_Common : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount32,
   FeatureUnpackedD16VMem]>;

def FeatureISAVersion8_0_1 : FeatureSet<
  !listconcat(FeatureISAVersion8_0_Common.Features,
    [FeatureFastFMAF32,
     FeatureHalfRate64Ops,
     FeatureSupportsXNACK])>;

def FeatureISAVersion8_0_2 : FeatureSet<
  !listconcat(FeatureISAVersion8_0_Common.Features,
    [FeatureSGPRInitBug])>;

def FeatureISAVersion8_0_3 : FeatureSet<
  !listconcat(FeatureISAVersion8_0_Common.Features,
    [])>;

def FeatureISAVersion8_0_5 : FeatureSet<
  !listconcat(FeatureISAVersion8_0_Common.Features,
    [FeatureSGPRInitBug])>;

def FeatureISAVersion8_1_0 : FeatureSet<
  [FeatureVolcanicIslands,
   FeatureLDSBankCount16,
   FeatureSupportsXNACK,
   FeatureImageStoreD16Bug,
   FeatureImageGather4D16Bug]>;

def FeatureISAVersion9_0_Common : FeatureSet<
  [FeatureGFX9,
   FeatureAddressableLocalMemorySize65536,
   FeatureLDSBankCount32,
   FeatureImageInsts,
   FeatureMadMacF32Insts]>;

def FeatureISAVersion9_0_Consumer_Common : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Common.Features,
    [FeatureImageGather4D16Bug,
     FeatureDsSrc2Insts,
     FeatureExtendedImageInsts,
     FeatureGDS])>;

def FeatureISAVersion9_Generic : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureRequiresCOV6])>;

def FeatureISAVersion9_0_MI_Common : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Common.Features,
    [FeatureFmaMixInsts,
     FeatureDLInsts,
     FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot3Insts,
     FeatureDot4Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureDot7Insts,
     FeatureDot10Insts,
     FeatureMAIInsts,
     FeaturePkFmacF16Inst,
     FeatureAtomicFaddNoRtnInsts,
     FeatureSupportsSRAMECC])>;

def FeatureISAVersion9_0_0 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureMadMixInsts])>;

def FeatureISAVersion9_0_2 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureMadMixInsts])>;

def FeatureISAVersion9_0_4 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureFmaMixInsts])>;

def FeatureISAVersion9_0_6 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureHalfRate64Ops,
     FeatureFmaMixInsts,
     FeatureDLInsts,
     FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot7Insts,
     FeatureDot10Insts,
     FeatureSupportsSRAMECC])>;

def FeatureISAVersion9_0_8 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_MI_Common.Features,
    [FeatureGDS,
     FeatureHalfRate64Ops,
     FeatureDsSrc2Insts,
     FeatureExtendedImageInsts,
     FeatureAtomicBufferGlobalPkAddF16NoRtnInsts,
     FeatureMFMAInlineLiteralBug,
     FeatureImageGather4D16Bug])>;

def FeatureISAVersion9_0_9 : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureMadMixInsts])>;

def FeatureISAVersion9_0_A : FeatureSet<
  !listconcat(FeatureISAVersion9_0_MI_Common.Features,
    [FeatureGFX90AInsts,
     FeatureRequiresAlignedVGPRs,
     FeatureFmacF64Inst,
     FeatureDPALU_DPP,
     FeaturePackedFP32Ops,
     FeatureAtomicFaddRtnInsts,
     FeatureAtomicBufferGlobalPkAddF16Insts,
     FeaturePackedTID,
     FeatureFullRate64Ops,
     FeatureBackOffBarrier,
     FeatureKernargPreload,
     FeatureAtomicFMinFMaxF64GlobalInsts,
     FeatureAtomicFMinFMaxF64FlatInsts,
     FeatureFlatBufferGlobalAtomicFaddF64Inst
     ])>;

def FeatureISAVersion9_0_C : FeatureSet<
  !listconcat(FeatureISAVersion9_0_Consumer_Common.Features,
    [FeatureMadMixInsts])>;

def FeatureISAVersion9_4_Common : FeatureSet<
  [FeatureGFX9,
   FeatureGFX90AInsts,
   FeatureGFX940Insts,
   FeatureRequiresAlignedVGPRs,
   FeatureFmaMixInsts,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureFmacF64Inst,
   FeatureDot1Insts,
   FeatureDot2Insts,
   FeatureDot3Insts,
   FeatureDot4Insts,
   FeatureDot5Insts,
   FeatureDot6Insts,
   FeatureDot7Insts,
   FeatureDot10Insts,
   FeatureAtomicDsPkAdd16Insts,
   FeatureAtomicFlatPkAdd16Insts,
   FeatureDPALU_DPP,
   FeaturePackedFP32Ops,
   FeatureMAIInsts,
   FeaturePkFmacF16Inst,
   FeatureAtomicFaddRtnInsts,
   FeatureAtomicFaddNoRtnInsts,
   FeatureAtomicBufferGlobalPkAddF16Insts,
   FeatureAtomicGlobalPkAddBF16Inst,
   FeatureFlatAtomicFaddF32Inst,
   FeatureSupportsSRAMECC,
   FeaturePackedTID,
   FeatureArchitectedFlatScratch,
   FeatureFullRate64Ops,
   FeatureBackOffBarrier,
   FeatureKernargPreload,
   FeatureAtomicFMinFMaxF64GlobalInsts,
   FeatureAtomicFMinFMaxF64FlatInsts,
   FeatureAgentScopeFineGrainedRemoteMemoryAtomics,
   FeatureMemoryAtomicFAddF32DenormalSupport,
   FeatureFlatBufferGlobalAtomicFaddF64Inst,
   FeatureLshlAddU64Inst,
   ]>;

def FeatureISAVersion9_5_Common : FeatureSet<
  !listconcat(FeatureISAVersion9_4_Common.Features,
  [FeatureAddressableLocalMemorySize163840,
   FeatureFP8Insts,
   FeatureFP8ConversionInsts,
   FeatureGFX950Insts,
   FeaturePrngInst,
   FeatureBF16ConversionInsts,
   FeatureBitOp3Insts,
   FeatureFP8ConversionScaleInsts,
   FeatureBF8ConversionScaleInsts,
   FeatureFP4ConversionScaleInsts,
   FeatureFP6BF6ConversionScaleInsts,
   FeatureDot12Insts,
   FeatureDot13Insts,
   FeatureAtomicBufferPkAddBF16Inst
   ])>;

def FeatureISAVersion9_4_2 : FeatureSet<
  !listconcat(FeatureISAVersion9_4_Common.Features,
    [
      FeatureAddressableLocalMemorySize65536,
      FeatureFP8Insts,
      FeatureFP8ConversionInsts,
      FeatureCvtFP8VOP1Bug,
      FeatureXF32Insts
    ])>;

def FeatureISAVersion9_4_Generic : FeatureSet<
  !listconcat(FeatureISAVersion9_4_Common.Features,
    [FeatureAddressableLocalMemorySize65536,
     FeatureRequiresCOV6])>;

def FeatureISAVersion9_5_0 : FeatureSet<FeatureISAVersion9_5_Common.Features>;

def FeatureISAVersion10_Common : FeatureSet<
  [FeatureGFX10,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureNSAEncoding,
   FeatureBackOffBarrier]>;

def FeatureISAVersion10_1_Common : FeatureSet<
  !listconcat(FeatureISAVersion10_Common.Features,
    [FeatureScalarStores,
     FeatureScalarAtomics,
     FeatureScalarFlatScratchInsts,
     FeatureGetWaveIdInst,
     FeatureMadMacF32Insts,
     FeatureDsSrc2Insts,
     FeatureLDSMisalignedBug,
     FeatureSupportsXNACK,
     // gfx101x bugs
     FeatureVcmpxPermlaneHazard,
     FeatureVMEMtoScalarWriteHazard,
     FeatureSMEMtoVectorWriteHazard,
     FeatureInstFwdPrefetchBug,
     FeatureVcmpxExecWARHazard,
     FeatureLdsBranchVmemWARHazard,
     FeatureNSAtoVMEMBug,
     FeatureNSAClauseBug,
     FeatureOffset3fBug,
     FeatureFlatSegmentOffsetBug,
     FeatureNegativeUnalignedScratchOffsetBug])>;

def FeatureISAVersion10_1_Generic : FeatureSet<
  !listconcat(FeatureISAVersion10_1_Common.Features,
    [FeatureRequiresCOV6])>;

def FeatureISAVersion10_1_0 : FeatureSet<
  !listconcat(FeatureISAVersion10_1_Common.Features,
    [])>;

def FeatureISAVersion10_1_1 : FeatureSet<
  !listconcat(FeatureISAVersion10_1_Common.Features,
    [FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureDot7Insts,
     FeatureDot10Insts])>;

def FeatureISAVersion10_1_2 : FeatureSet<
  !listconcat(FeatureISAVersion10_1_Common.Features,
    [FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureDot7Insts,
     FeatureDot10Insts])>;

def FeatureISAVersion10_1_3 : FeatureSet<
  !listconcat(FeatureISAVersion10_1_Common.Features,
    [FeatureGFX10_AEncoding])>;

def FeatureISAVersion10_3_0 : FeatureSet<
  !listconcat(FeatureISAVersion10_Common.Features,
    [FeatureGFX10_AEncoding,
     FeatureGFX10_BEncoding,
     FeatureGFX10_3Insts,
     FeatureDot1Insts,
     FeatureDot2Insts,
     FeatureDot5Insts,
     FeatureDot6Insts,
     FeatureDot7Insts,
     FeatureDot10Insts,
     FeatureShaderCyclesRegister])>;

def FeatureISAVersion10_3_Generic: FeatureSet<
  !listconcat(FeatureISAVersion10_3_0.Features,
    [FeatureRequiresCOV6])>;

def FeatureISAVersion11_Common : FeatureSet<
  [FeatureGFX11,
   FeatureBackOffBarrier,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot5Insts,
   FeatureDot7Insts,
   FeatureDot8Insts,
   FeatureDot9Insts,
   FeatureDot10Insts,
   FeatureDot12Insts,
   FeatureNSAEncoding,
   FeaturePartialNSAEncoding,
   FeatureShaderCyclesRegister,
   FeatureArchitectedFlatScratch,
   FeatureAtomicFaddRtnInsts,
   FeatureAtomicFaddNoRtnInsts,
   FeatureFlatAtomicFaddF32Inst,
   FeatureImageInsts,
   FeaturePackedTID,
   FeatureVcmpxPermlaneHazard,
   FeatureMemoryAtomicFAddF32DenormalSupport,
   FeatureRealTrue16Insts,
   FeatureD16Writes32BitVgpr,
]>;

// There are few workarounds that need to be
// added to all targets. This pessimizes codegen
// a bit on the generic GFX11 target.
def FeatureISAVersion11_Generic: FeatureSet<
  !listconcat(FeatureISAVersion11_Common.Features,
    [FeatureMSAALoadDstSelBug,
     FeatureVALUTransUseHazard,
     FeatureUserSGPRInit16Bug,
     FeatureMADIntraFwdBug,
     FeaturePrivEnabledTrap2NopBug,
     FeatureRequiresCOV6,
     FeatureRequiredExportPriority])>;

def FeatureISAVersion11_0_Common : FeatureSet<
  !listconcat(FeatureISAVersion11_Common.Features,
    [FeatureMSAALoadDstSelBug,
     FeatureVALUTransUseHazard,
     FeatureMADIntraFwdBug,
     FeaturePrivEnabledTrap2NopBug])>;

def FeatureISAVersion11_0_0 : FeatureSet<
  !listconcat(FeatureISAVersion11_0_Common.Features,
    [Feature1_5xVGPRs,
     FeatureUserSGPRInit16Bug])>;

def FeatureISAVersion11_0_1 : FeatureSet<
  !listconcat(FeatureISAVersion11_0_Common.Features,
    [Feature1_5xVGPRs])>;

def FeatureISAVersion11_0_2 : FeatureSet<
  !listconcat(FeatureISAVersion11_0_Common.Features,
    [FeatureUserSGPRInit16Bug])>;

def FeatureISAVersion11_0_3 : FeatureSet<
  !listconcat(FeatureISAVersion11_0_Common.Features,
    [])>;

def FeatureISAVersion11_5_Common : FeatureSet<
  !listconcat(FeatureISAVersion11_Common.Features,
    [FeatureSALUFloatInsts,
     FeatureDPPSrc1SGPR,
     FeatureRequiredExportPriority])>;

def FeatureISAVersion11_5_0 : FeatureSet<
  !listconcat(FeatureISAVersion11_5_Common.Features,
    [FeaturePointSampleAccel])>;

def FeatureISAVersion11_5_1 : FeatureSet<
  !listconcat(FeatureISAVersion11_5_Common.Features,
    [Feature1_5xVGPRs,
     FeaturePointSampleAccel])>;

def FeatureISAVersion11_5_2 : FeatureSet<
  !listconcat(FeatureISAVersion11_5_Common.Features,
    [FeaturePointSampleAccel])>;

def FeatureISAVersion11_5_3 : FeatureSet<
  !listconcat(FeatureISAVersion11_5_Common.Features,
    [])>;

def FeatureISAVersion12 : FeatureSet<
  [FeatureGFX12,
   FeatureBackOffBarrier,
   FeatureAddressableLocalMemorySize65536,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureDot7Insts,
   FeatureDot8Insts,
   FeatureDot9Insts,
   FeatureDot10Insts,
   FeatureDot11Insts,
   FeatureDot12Insts,
   FeatureNSAEncoding,
   FeaturePartialNSAEncoding,
   FeatureShaderCyclesHiLoRegisters,
   FeatureArchitectedFlatScratch,
   FeatureArchitectedSGPRs,
   FeatureAtomicFaddRtnInsts,
   FeatureAtomicFaddNoRtnInsts,
   FeatureAtomicDsPkAdd16Insts,
   FeatureAtomicFlatPkAdd16Insts,
   FeatureAtomicBufferGlobalPkAddF16Insts,
   FeatureAtomicGlobalPkAddBF16Inst,
   FeatureAtomicBufferPkAddBF16Inst,
   FeatureFlatAtomicFaddF32Inst,
   FeatureImageInsts,
   FeatureExtendedImageInsts,
   FeatureFP8ConversionInsts,
   FeatureIEEEMinimumMaximumInsts,
   FeaturePackedTID,
   FeatureVcmpxPermlaneHazard,
   FeatureSALUFloatInsts,
   FeaturePseudoScalarTrans,
   FeatureRestrictedSOffset,
   FeatureScalarDwordx3Loads,
   FeatureDPPSrc1SGPR,
   FeatureMaxHardClauseLength32,
   Feature1_5xVGPRs,
   FeatureMemoryAtomicFAddF32DenormalSupport,
   FeatureBVHDualAndBVH8Insts,
   FeatureWaitsBeforeSystemScopeStores,
   FeatureD16Writes32BitVgpr,
   FeatureCubeInsts,
   FeatureLerpInst,
   FeatureSadInsts,
   FeatureQsadInsts,
   FeatureCvtNormInsts,
   FeatureCvtPkNormVOP2Insts,
   FeatureCvtPkNormVOP3Insts
   ]>;

def FeatureISAVersion12_50_Common : FeatureSet<
  [FeatureGFX12,
   FeatureGFX1250Insts,
   FeatureRequiresAlignedVGPRs,
   FeatureCuMode,
   Feature1024AddressableVGPRs,
   Feature64BitLiterals,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureFmacF64Inst,
   FeaturePackedFP32Ops,
   FeatureDot7Insts,
   FeatureDot8Insts,
   FeatureWavefrontSize32,
   FeatureShaderCyclesHiLoRegisters,
   FeatureArchitectedFlatScratch,
   FeatureArchitectedSGPRs,
   FeatureFlatGVSMode,
   FeatureAtomicFaddRtnInsts,
   FeatureAtomicFaddNoRtnInsts,
   FeatureAtomicDsPkAdd16Insts,
   FeatureAtomicFlatPkAdd16Insts,
   FeatureAtomicBufferGlobalPkAddF16Insts,
   FeatureAtomicGlobalPkAddBF16Inst,
   FeatureAtomicBufferPkAddBF16Inst,
   FeatureFlatAtomicFaddF32Inst,
   FeatureFP8ConversionInsts,
   FeatureFP8E5M3Insts,
   FeaturePackedTID,
   FeatureVcmpxPermlaneHazard,
   FeatureSALUFloatInsts,
   FeaturePseudoScalarTrans,
   FeatureRestrictedSOffset,
   FeatureScalarDwordx3Loads,
   FeatureDPPSrc1SGPR,
   FeatureBitOp3Insts,
   FeatureTanhInsts,
   FeatureTensorCvtLutInsts,
   FeatureTransposeLoadF4F6Insts,
   FeatureBF16TransInsts,
   FeatureBF16ConversionInsts,
   FeatureBF16PackedInsts,
   FeatureCvtPkF16F32Inst,
   FeatureFmaMixBF16Insts,
   FeatureMin3Max3PKF16,
   FeatureMinimum3Maximum3PKF16,
   FeaturePrngInst,
   FeaturePermlane16Swap,
   FeatureAshrPkInsts,
   FeatureSupportsSRAMECC,
   FeatureMaxHardClauseLength63,
   FeatureWaitXcnt,
   FeatureAtomicFMinFMaxF64GlobalInsts,
   FeatureAtomicFMinFMaxF64FlatInsts,
   FeatureFlatBufferGlobalAtomicFaddF64Inst,
   FeatureMemoryAtomicFAddF32DenormalSupport,
   FeatureEmulatedSystemScopeAtomics,
   FeatureGloballyAddressableScratch,
   FeatureKernargPreload,
   FeatureVmemPrefInsts,
   FeatureLshlAddU64Inst,
   FeatureAddSubU64Insts,
   FeatureMadU32Inst,
   FeatureAddMinMaxInsts,
   FeaturePkAddMinMaxInsts,
   FeatureLdsBarrierArriveAtomic,
   FeatureSetPrioIncWgInst,
   FeatureSWakeupBarrier,
   Feature45BitNumRecordsBufferResource,
   FeatureSupportsXNACK,
   FeatureXNACK,
   FeatureClusters,
   FeatureD16Writes32BitVgpr,
   FeatureCubeInsts,
   FeatureLerpInst,
   FeatureSadInsts,
   FeatureQsadInsts,
   FeatureCvtNormInsts,
   FeatureCvtPkNormVOP2Insts,
   FeatureCvtPkNormVOP3Insts,
   FeatureMcastLoadInsts
]>;

def FeatureISAVersion12_50 : FeatureSet<
  !listconcat(FeatureISAVersion12_50_Common.Features,
  [FeatureAddressableLocalMemorySize327680,
   FeatureSetregVGPRMSBFixup])>;

def FeatureISAVersion12_51 : FeatureSet<
  !listconcat(FeatureISAVersion12_50_Common.Features,
  [FeatureAddressableLocalMemorySize327680,
   FeatureDPALU_DPP])>;

def FeatureISAVersion12_Generic: FeatureSet<
  !listconcat(FeatureISAVersion12.Features,
    [FeatureRequiresCOV6])>;

def FeatureISAVersion13 : FeatureSet<
  [FeatureGFX13,
   FeatureGFX1250Insts,
   FeatureAddressableLocalMemorySize65536,
   Feature64BitLiterals,
   FeatureLDSBankCount32,
   FeatureDLInsts,
   FeatureFmacF64Inst,
   FeatureDot7Insts,
   FeatureDot8Insts,
   FeatureNSAEncoding,
   FeaturePartialNSAEncoding,
   FeatureShaderCyclesRegister,
   FeatureArchitectedFlatScratch,
   FeatureArchitectedSGPRs,
   FeatureAtomicFaddRtnInsts,
   FeatureAtomicFaddNoRtnInsts,
   FeatureAtomicDsPkAdd16Insts,
   FeatureAtomicFlatPkAdd16Insts,
   FeatureAtomicBufferGlobalPkAddF16Insts,
   FeatureAtomicGlobalPkAddBF16Inst,
   FeatureAtomicBufferPkAddBF16Inst,
   FeatureFlatAtomicFaddF32Inst,
   FeatureFP8ConversionInsts,
   FeaturePackedTID,
   FeatureVcmpxPermlaneHazard,
   FeatureSALUFloatInsts,
   FeaturePseudoScalarTrans,
   FeatureRestrictedSOffset,
   FeatureScalarDwordx3Loads,
   FeatureDPPSrc1SGPR,
   FeatureBitOp3Insts,
   FeatureTanhInsts,
   FeatureTensorCvtLutInsts,
   FeatureTransposeLoadF4F6Insts,
   Feature1_5xVGPRs,
   FeatureBF16TransInsts,
   FeatureBF16ConversionInsts,
   FeatureBF16PackedInsts,
   FeaturePrngInst,
   FeaturePermlane16Swap,
   FeatureAshrPkInsts,
   FeatureAtomicFMinFMaxF64GlobalInsts,
   FeatureAtomicFMinFMaxF64FlatInsts,
   FeatureFmaMixBF16Insts,
   FeatureGloballyAddressableScratch,
   FeatureCvtPkF16F32Inst,
   FeatureF16BF16ToFP6BF6ConversionScaleInsts,
   FeatureIEEEMinimumMaximumInsts,
   FeatureClusters,
   FeatureCubeInsts,
   FeatureLerpInst,
   FeatureSadInsts,
   FeatureQsadInsts,
   FeatureCvtNormInsts,
   FeatureCvtPkNormVOP2Insts,
   FeatureCvtPkNormVOP3Insts,
]>;

//===----------------------------------------------------------------------===//

def AMDGPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
}

def AMDGPUAsmParser : AsmParser {
  // Some of the R600 registers have the same name, so this crashes.
  // For example T0_XYZW and T0_XY both have the asm name T0.
  let ShouldEmitMatchRegisterName = 0;

  // Call the custom operand parser for all operands.
  let OperandParserMethod = "parseCustomOperand";
  let CallCustomParserForAllOperands = true;
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
  string VOP3_DPP = "VOP3_DPP";
  int VOP3_DPP_ID = 5;
  string Disable = "Disable";
  int Disable_ID = 6;
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

def VOP3_DPPAsmParserVariant : AsmParserVariant {
  let Variant = AMDGPUAsmVariants.VOP3_DPP_ID;
  let Name = AMDGPUAsmVariants.VOP3_DPP;
}

def AMDGPU : Target {
  // Pull in Instruction Info:
  let InstructionSet = AMDGPUInstrInfo;
  let AssemblyParsers = [AMDGPUAsmParser];
  let AssemblyParserVariants = [DefaultAMDGPUAsmParserVariant,
                                VOP3AsmParserVariant,
                                SDWAAsmParserVariant,
                                SDWA9AsmParserVariant,
                                DPPAsmParserVariant,
                                VOP3_DPPAsmParserVariant];
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
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), (not FeatureGFX11Insts))>;

def isGFX6GFX7GFX10Plus :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding))>;

def isGFX7Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts, (not FeatureGFX10Insts))>;

def isGFX7GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureCIInsts, (not FeatureGFX11Insts))>;

def isGFX7GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
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

def isGFX6GFX7GFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SOUTHERN_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of (not FeatureGFX11Insts))>;

def isNotGFX12Plus :
  Predicate<"Subtarget->getGeneration() <= AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of (not FeatureGFX12Insts))>;

def isGFX7GFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::SEA_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureCIInsts, (not FeatureGFX11Insts))>;

def isGFX8GFX9GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, (not FeatureGFX12Insts))>;

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

def isNotGFX9Plus :
  Predicate<"Subtarget->getGeneration() < AMDGPUSubtarget::GFX9">;

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
  Predicate<"Subtarget->hasGFX90AInsts() && !Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts, (not FeatureGFX940Insts))>;

def isGFX908orGFX90A :
  Predicate<"Subtarget->hasMAIInsts() && !Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureMAIInsts, (not FeatureGFX940Insts))>;

def isGFX940Plus :
  Predicate<"Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of FeatureGFX940Insts)>;

def isNotGFX940Plus :
  Predicate<"!Subtarget->hasGFX940Insts()">,
  AssemblerPredicate<(all_of (not FeatureGFX940Insts))>;

def isGFX8GFX9NotGFX940 :
  Predicate<"!Subtarget->hasGFX940Insts() &&"
            "(Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            " Subtarget->getGeneration() == AMDGPUSubtarget::GFX9)">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding, (not FeatureGFX940Insts))>;

def isGFX8GFX9 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, FeatureGCN3Encoding)>;

def isGFX10Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX10Insts, (not FeatureGFX11Insts))>;

def isGFX10Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX10Insts)>;

def isGFX10GFX11 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX10Insts, (not FeatureGFX12Insts))>;

def isGFX10Before1030 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX10 &&"
            "!Subtarget->hasGFX10_3Insts()">,
  AssemblerPredicate<(all_of FeatureGFX10Insts,(not FeatureGFX10_3Insts))>;

def isGFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureGFX11Insts))>;

def isGFX9GFX10GFX11 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9 &&"
            "Subtarget->getGeneration() < AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureGFX12Insts))>;

def isGFX8GFX9GFX10 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::VOLCANIC_ISLANDS ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX9 ||"
            "Subtarget->getGeneration() == AMDGPUSubtarget::GFX10">,
  AssemblerPredicate<(all_of FeatureGFX8Insts, (not FeatureGFX11Insts))>;

def isGFX11Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (not FeatureGFX12Insts))>;

def isGFX11Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX11">,
  AssemblerPredicate<(all_of FeatureGFX11Insts)>;

def isGFX12Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (not FeatureGFX13Insts))>;

def isGFX12Not12_50 :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12 && !Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (not FeatureGFX1250Insts))>;

def isGFX12Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts)>;

def isGFX12PlusNot12_50 :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12 &&"
            "(Subtarget->getGeneration() >= AMDGPUSubtarget::GFX13 || !Subtarget->hasGFX1250Insts())">,
  AssemblerPredicate<(all_of FeatureGFX12Insts, (any_of FeatureGFX13Insts, (not FeatureGFX1250Insts)))>;

def isGFX125xOnly :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX12 && Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX1250Insts, (not FeatureGFX13Insts))>;

def isGFX1250Plus :
  Predicate<"Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of FeatureGFX1250Insts)>;

def isNotGFX1250Plus :
  Predicate<"!Subtarget->hasGFX1250Insts()">,
  AssemblerPredicate<(all_of (not FeatureGFX1250Insts))>;

def isGFX940orGFX1250 :
  Predicate<"Subtarget->hasGFX940Insts() ||"
            "(Subtarget->hasGFX1250Insts() && !Subtarget->hasGFX13Insts())">,
  AssemblerPredicate<(any_of FeatureGFX940Insts,
                     (all_of FeatureGFX1250Insts, (not FeatureGFX13Insts)))>;

def isGFX13Only :
  Predicate<"Subtarget->getGeneration() == AMDGPUSubtarget::GFX13">,
  AssemblerPredicate<(all_of FeatureGFX13Insts)>;

def isGFX13Plus :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX13">,
  AssemblerPredicate<(all_of FeatureGFX13Insts)>;

def HasAtomicCondSubClampFlatInsts :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts)>;

def HasLdsAtomicAddF64 :
  Predicate<"Subtarget->hasLdsAtomicAddF64()">,
  AssemblerPredicate<(any_of FeatureGFX90AInsts, FeatureGFX1250Insts)>;

def HasD16LoadStore : Predicate<"Subtarget->hasD16LoadStore()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasFlatScratchSTMode : Predicate<"Subtarget->hasFlatScratchSTMode()">,
  AssemblerPredicate<(any_of FeatureGFX10_3Insts, FeatureGFX940Insts)>;
def HasFlatScratchSVSMode : Predicate<"Subtarget->hasFlatScratchSVSMode()">,
  AssemblerPredicate<(any_of FeatureGFX940Insts, FeatureGFX11Insts)>;

def HasGFX10_AEncoding : Predicate<"Subtarget->hasGFX10_AEncoding()">,
  AssemblerPredicate<(all_of FeatureGFX10_AEncoding)>;

def HasGFX10_BEncoding : Predicate<"Subtarget->hasGFX10_BEncoding()">,
  AssemblerPredicate<(all_of FeatureGFX10_BEncoding)>;

def HasPackedD16VMem : Predicate<"!Subtarget->hasUnpackedD16VMem()">,
  AssemblerPredicate<(all_of (not FeatureUnpackedD16VMem))>;

def HasUnrestrictedSOffset : Predicate<"!Subtarget->hasRestrictedSOffset()">,
  AssemblerPredicate<(all_of (not FeatureRestrictedSOffset))>;

def D16PreservesUnusedBits :
  Predicate<"Subtarget->d16PreservesUnusedBits()">,
  AssemblerPredicate<(all_of FeatureGFX9Insts, (not FeatureSRAMECC))>;

def LDSRequiresM0Init : Predicate<"Subtarget->ldsRequiresM0Init()">;
def NotLDSRequiresM0Init : Predicate<"!Subtarget->ldsRequiresM0Init()">;

def HasMTBUFInsts : Predicate<"Subtarget->hasMTBUFInsts()">,
  AssemblerPredicate<(all_of (not FeatureGFX1250Insts))>;

def HasFormattedMUBUFInsts : Predicate<"Subtarget->hasFormattedMUBUFInsts()">,
  AssemblerPredicate<(all_of (not FeatureGFX1250Insts))>;

def HasExportInsts : Predicate<"Subtarget->hasExportInsts()">,
  AssemblerPredicate<(all_of (not FeatureGFX90AInsts), (not FeatureGFX1250Insts))>;

def HasVINTERPEncoding : Predicate<"Subtarget->hasVINTERPEncoding()">,
  AssemblerPredicate<(all_of FeatureGFX11Insts, (not FeatureGFX1250Insts))>;

def HasDSAddTid : Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX9">,
  AssemblerPredicate<(all_of FeatureGFX9Insts)>;

def HasLDSFPAtomicAddF32 : Predicate<"Subtarget->hasLDSFPAtomicAddF32()">,
  AssemblerPredicate<(all_of FeatureGFX8Insts)>;

def NotHasAddNoCarryInsts : Predicate<"!Subtarget->hasAddNoCarryInsts()">;

def HasXNACKEnabled : Predicate<"Subtarget->isXNACKEnabled()">;

def NotHasTrue16BitInsts : True16PredicateClass<"!Subtarget->hasTrue16BitInsts()">,
  AssemblerPredicate<(all_of (not FeatureTrue16BitInsts))>;

// Control use of True16 instructions. The real True16 instructions are
// True16 instructions as they are defined in the ISA. Fake True16
// instructions have the same encoding as real ones but syntactically
// only allow 32-bit registers in operands and use low halves thereof.
def UseRealTrue16Insts : True16PredicateClass<"Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts)>;
def NotUseRealTrue16Insts : True16PredicateClass<"!Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(not (all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts))>;
def UseFakeTrue16Insts : True16PredicateClass<"Subtarget->hasTrue16BitInsts() && "
                                              "!Subtarget->useRealTrue16Insts()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, (not FeatureRealTrue16Insts))>;

def UseTrue16WithSramECC : True16PredicateClass<"Subtarget->useRealTrue16Insts() && "
                                                "!Subtarget->d16PreservesUnusedBits()">;

def NotHasD16Writes32BitVgpr: Predicate<"!Subtarget->hasD16Writes32BitVgpr()">,
  AssemblerPredicate<(all_of FeatureTrue16BitInsts, FeatureRealTrue16Insts, (not FeatureD16Writes32BitVgpr))>;

def NotHasMed3_16 : Predicate<"!Subtarget->hasMed3_16()">;
def HasMed3_16 : Predicate<"Subtarget->hasMed3_16()">;

def HasMinMaxDenormModes : Predicate<"Subtarget->supportsMinMaxDenormModes()">;
def NotHasMinMaxDenormModes : Predicate<"!Subtarget->supportsMinMaxDenormModes()">;

def HasFminFmaxLegacy : Predicate<"Subtarget->hasFminFmaxLegacy()">;

def HasSDWA8 : Predicate<"Subtarget->hasSDWA()">,
  AssemblerPredicate<(all_of (not FeatureGFX9Insts), FeatureSDWA)>;

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

def HasPkMovB32 : Predicate<"Subtarget->hasPkMovB32()">,
  AssemblerPredicate<(all_of FeatureGFX90AInsts)>;

def HasFmaakFmamkF32Insts :
  Predicate<"Subtarget->hasFmaakFmamkF32Insts()">,
  AssemblerPredicate<(any_of FeatureGFX10Insts, FeatureGFX940Insts)>;

def HasFmaakFmamkF64Insts :
  Predicate<"Subtarget->hasFmaakFmamkF64Insts()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasPkMinMax3Insts :
  Predicate<"Subtarget->hasPkMinMax3Insts()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasSGetShaderCyclesInst :
  Predicate<"Subtarget->hasSGetShaderCyclesInst()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasDPP16 : Predicate<"Subtarget->hasDPP()">,
  AssemblerPredicate<(all_of (not FeatureGCN3Encoding), FeatureGFX10Insts, FeatureDPP)>;

def HasSdstCMPX : Predicate<"!Subtarget->hasNoSdstCMPX()">,
  AssemblerPredicate<(all_of (not FeatureNoSdstCMPX))>;

def has16BankLDS : Predicate<"Subtarget->getLDSBankCount() == 16">;
def has32BankLDS : Predicate<"Subtarget->getLDSBankCount() == 32">;

def NotHasMAIInsts : Predicate<"!Subtarget->hasMAIInsts()">,
  AssemblerPredicate<(all_of (not FeatureMAIInsts))>;

def NotHasFP8E5M3Insts : Predicate<"!Subtarget->hasFP8E5M3Insts()">,
  AssemblerPredicate<(all_of (not FeatureFP8E5M3Insts))>;

def HasFmaLegacy32 : Predicate<"Subtarget->hasGFX10_3Insts()">,
  AssemblerPredicate<(any_of FeatureGFX10_3Insts)>;

def HasFmacLegacy32 : Predicate<"Subtarget->hasGFX10_3Insts() && Subtarget->getGeneration() < AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX10_3Insts, (not FeatureGFX12Insts))>;

def HasAtomicDsCondSubClampInsts :
  Predicate<"Subtarget->getGeneration() >= AMDGPUSubtarget::GFX12">,
  AssemblerPredicate<(all_of FeatureGFX12Insts)>;

def HasAtomicBufferGlobalPkAddF16NoRtnInsts
  : Predicate<"Subtarget->hasAtomicBufferGlobalPkAddF16NoRtnInsts() || Subtarget->hasAtomicBufferGlobalPkAddF16Insts()">,
  AssemblerPredicate<(any_of FeatureAtomicBufferGlobalPkAddF16NoRtnInsts, FeatureAtomicBufferGlobalPkAddF16Insts)>;

def HasAddPC64Inst : Predicate<"Subtarget->hasAddPC64Inst()">,
  AssemblerPredicate<(any_of FeatureGFX1250Insts)>;

def HasFlatScratchEnabled : Predicate<"Subtarget->hasFlatScratchEnabled()">;

def NotHasFlatScratchEnabled : Predicate<"!Subtarget->hasFlatScratchEnabled()">;

def NotHasMADIntraFwdBug : Predicate<"!Subtarget->hasMADIntraFwdBug()">;

def NotHasSALUFloatInsts : Predicate<"!Subtarget->hasSALUFloatInsts()">,
  AssemblerPredicate<(all_of (not FeatureSALUFloatInsts))>;

def NotHasCvtFP8VOP1Bug : Predicate<"!Subtarget->hasCvtFP8VOP1Bug()">;

def NeedsAlignedVGPRs : Predicate<"Subtarget->needsAlignedVGPRs()">,
                      AssemblerPredicate<(all_of FeatureRequiresAlignedVGPRs)>;

def NotNeedsAlignedVGPRs : Predicate<"!Subtarget->needsAlignedVGPRs()">,
                      AssemblerPredicate<(all_of (not FeatureRequiresAlignedVGPRs))>;

def isWave32 : Predicate<"Subtarget->isWave32()">,
  AssemblerPredicate <(any_of FeatureWavefrontSize32,
                              FeatureAssemblerPermissiveWavesize)>;
def isWave64 : Predicate<"Subtarget->isWave64()">,
  AssemblerPredicate <(any_of FeatureWavefrontSize64,
                              FeatureAssemblerPermissiveWavesize)>;

def isWave32Strict : Predicate<"Subtarget->isWave32()">,
  AssemblerPredicate <(all_of FeatureWavefrontSize32)>;
def isWave64Strict : Predicate<"Subtarget->isWave64()">,
  AssemblerPredicate <(all_of FeatureWavefrontSize64)>;

//===----------------------------------------------------------------------===//
// HwModes
//===----------------------------------------------------------------------===//

defvar DefaultMode_Wave64 = DefaultMode;
defvar DefaultMode_Wave32 = HwMode<[isWave32, NotNeedsAlignedVGPRs]>;

// gfx90a-gfx950. Has AGPRs, and also the align2 VGPR/AGPR requirement. Implied
// wave64.
def AVAlign2LoadStoreMode : HwMode<[HasMAIInsts, NeedsAlignedVGPRs]>;

// gfx1250, has alignment requirement but no AGPRs.
def AlignedVGPRNoAGPRMode_Wave32 : HwMode<[NotHasMAIInsts, NeedsAlignedVGPRs, isWave32Strict]>;
def AlignedVGPRNoAGPRMode_Wave64 : HwMode<[NotHasMAIInsts, NeedsAlignedVGPRs, isWave64Strict]>;

// FIXME: This should be able to only define a separate hwmode that
// only depends on wavesize for just ValueTypes. These use different
// HwMode namespaces. If we don't define the full set of modes used
// for RegClassByHwMode, tablegen crashes for some reason
def WaveSizeVT : ValueTypeByHwMode<[
  DefaultMode_Wave64,
  AVAlign2LoadStoreMode,
  AlignedVGPRNoAGPRMode_Wave64,
  DefaultMode_Wave32,
  AlignedVGPRNoAGPRMode_Wave32], [i64, i64, i64, i32, i32]>;


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
