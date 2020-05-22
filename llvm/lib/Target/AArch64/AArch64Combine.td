//=- AArch64.td - Define AArch64 Combine Rules ---------------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def fconstant_to_constant : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_FCONSTANT):$root,
         [{ return matchFConstantToConstant(*${root}, MRI); }]),
  (apply [{ applyFConstantToConstant(*${root}); }])>;

def AArch64PreLegalizerCombinerHelper: GICombinerHelper<
  "AArch64GenPreLegalizerCombinerHelper", [all_combines,
                                           elide_br_by_inverting_cond,
                                           fconstant_to_constant]> {
  let DisableRuleOption = "aarch64prelegalizercombiner-disable-rule";
}

def AArch64PostLegalizerCombinerHelper
    : GICombinerHelper<"AArch64GenPostLegalizerCombinerHelper",
                       [erase_undef_store, combines_for_extload]> {
  let DisableRuleOption = "aarch64postlegalizercombiner-disable-rule";
}
