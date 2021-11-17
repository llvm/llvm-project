//=- AMDGPUCombine.td - Define AMDGPU Combine Rules ----------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

// TODO: This really belongs after legalization after scalarization.
// TODO: GICombineRules should accept subtarget predicates

def fmin_fmax_legacy_matchdata : GIDefMatchData<"AMDGPUPostLegalizerCombinerHelper::FMinFMaxLegacyInfo">;

def fcmp_select_to_fmin_fmax_legacy : GICombineRule<
  (defs root:$select, fmin_fmax_legacy_matchdata:$matchinfo),
  (match (wip_match_opcode G_SELECT):$select,
         [{ return PostLegalizerHelper.matchFMinFMaxLegacy(*${select}, ${matchinfo}); }]),
  (apply [{ PostLegalizerHelper.applySelectFCmpToFMinToFMaxLegacy(*${select}, ${matchinfo}); }])>;


def uchar_to_float : GICombineRule<
  (defs root:$itofp),
  (match (wip_match_opcode G_UITOFP, G_SITOFP):$itofp,
         [{ return PostLegalizerHelper.matchUCharToFloat(*${itofp}); }]),
  (apply [{ PostLegalizerHelper.applyUCharToFloat(*${itofp}); }])>;

def cvt_f32_ubyteN_matchdata : GIDefMatchData<"AMDGPUPostLegalizerCombinerHelper::CvtF32UByteMatchInfo">;

def cvt_f32_ubyteN : GICombineRule<
  (defs root:$cvt_f32_ubyteN, cvt_f32_ubyteN_matchdata:$matchinfo),
  (match (wip_match_opcode G_AMDGPU_CVT_F32_UBYTE0,
                           G_AMDGPU_CVT_F32_UBYTE1,
                           G_AMDGPU_CVT_F32_UBYTE2,
                           G_AMDGPU_CVT_F32_UBYTE3):$cvt_f32_ubyteN,
         [{ return PostLegalizerHelper.matchCvtF32UByteN(*${cvt_f32_ubyteN}, ${matchinfo}); }]),
  (apply [{ PostLegalizerHelper.applyCvtF32UByteN(*${cvt_f32_ubyteN}, ${matchinfo}); }])>;

def clamp_i64_to_i16_matchdata : GIDefMatchData<"AMDGPUPreLegalizerCombinerHelper::ClampI64ToI16MatchInfo">;

def clamp_i64_to_i16 : GICombineRule<
  (defs root:$clamp_i64_to_i16, clamp_i64_to_i16_matchdata:$matchinfo),
  (match (wip_match_opcode G_TRUNC):$clamp_i64_to_i16,
      [{ return PreLegalizerHelper.matchClampI64ToI16(*${clamp_i64_to_i16}, MRI, *MF, ${matchinfo}); }]),
  (apply [{ PreLegalizerHelper.applyClampI64ToI16(*${clamp_i64_to_i16}, ${matchinfo}); }])>;

def med3_matchdata : GIDefMatchData<"AMDGPURegBankCombinerHelper::Med3MatchInfo">;

def int_minmax_to_med3 : GICombineRule<
  (defs root:$min_or_max, med3_matchdata:$matchinfo),
  (match (wip_match_opcode G_SMAX,
                           G_SMIN,
                           G_UMAX,
                           G_UMIN):$min_or_max,
         [{ return RegBankHelper.matchIntMinMaxToMed3(*${min_or_max}, ${matchinfo}); }]),
  (apply [{ RegBankHelper.applyMed3(*${min_or_max}, ${matchinfo}); }])>;

def remove_fcanonicalize_matchinfo : GIDefMatchData<"Register">;

def remove_fcanonicalize : GICombineRule<
  (defs root:$fcanonicalize, remove_fcanonicalize_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FCANONICALIZE):$fcanonicalize,
         [{ return PostLegalizerHelper.matchRemoveFcanonicalize(*${fcanonicalize}, ${matchinfo}); }]),
  (apply [{ Helper.replaceSingleDefInstWithReg(*${fcanonicalize}, ${matchinfo}); }])>;

def foldable_fneg_matchdata : GIDefMatchData<"MachineInstr *">;

def foldable_fneg : GICombineRule<
  (defs root:$ffn, foldable_fneg_matchdata:$matchinfo),
  (match (wip_match_opcode G_FNEG):$ffn,
         [{ return Helper.matchFoldableFneg(*${ffn}, ${matchinfo}); }]),
  (apply [{ Helper.applyFoldableFneg(*${ffn}, ${matchinfo}); }])>;

// Combines which should only apply on SI/VI
def gfx6gfx7_combines : GICombineGroup<[fcmp_select_to_fmin_fmax_legacy]>;

def AMDGPUPreLegalizerCombinerHelper: GICombinerHelper<
  "AMDGPUGenPreLegalizerCombinerHelper",
  [all_combines, clamp_i64_to_i16, foldable_fneg]> {
  let DisableRuleOption = "amdgpuprelegalizercombiner-disable-rule";
  let StateClass = "AMDGPUPreLegalizerCombinerHelperState";
  let AdditionalArguments = [];
}

def AMDGPUPostLegalizerCombinerHelper: GICombinerHelper<
  "AMDGPUGenPostLegalizerCombinerHelper",
  [all_combines, gfx6gfx7_combines,
   uchar_to_float, cvt_f32_ubyteN, remove_fcanonicalize, foldable_fneg]> {
  let DisableRuleOption = "amdgpupostlegalizercombiner-disable-rule";
  let StateClass = "AMDGPUPostLegalizerCombinerHelperState";
  let AdditionalArguments = [];
}

def AMDGPURegBankCombinerHelper : GICombinerHelper<
  "AMDGPUGenRegBankCombinerHelper", [zext_trunc_fold, int_minmax_to_med3, ptr_add_immed_chain]> {
  let DisableRuleOption = "amdgpuregbankcombiner-disable-rule";
  let StateClass = "AMDGPURegBankCombinerHelperState";
  let AdditionalArguments = [];
}
