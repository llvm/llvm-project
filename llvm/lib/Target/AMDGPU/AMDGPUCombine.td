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

def fmin_fmax_legacy_matchdata : GIDefMatchData<"FMinFMaxLegacyInfo">;

def fcmp_select_to_fmin_fmax_legacy : GICombineRule<
  (defs root:$select, fmin_fmax_legacy_matchdata:$matchinfo),
  (match (wip_match_opcode G_SELECT):$select,
         [{ return matchFMinFMaxLegacy(*${select}, MRI, *MF, ${matchinfo}); }]),
  (apply [{ applySelectFCmpToFMinToFMaxLegacy(*${select}, ${matchinfo}); }])>;


// Combines which should only apply on SI/VI
def gfx6gfx7_combines : GICombineGroup<[fcmp_select_to_fmin_fmax_legacy]>;


def AMDGPUPreLegalizerCombinerHelper: GICombinerHelper<
  "AMDGPUGenPreLegalizerCombinerHelper", [all_combines,
                                          elide_br_by_inverting_cond]> {
  let DisableRuleOption = "amdgpuprelegalizercombiner-disable-rule";
}

def AMDGPUPostLegalizerCombinerHelper: GICombinerHelper<
  "AMDGPUGenPostLegalizerCombinerHelper", [all_combines,
                                           gfx6gfx7_combines]> {
  let DisableRuleOption = "amdgpupostlegalizercombiner-disable-rule";
}
