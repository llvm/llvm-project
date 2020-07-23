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
  let StateClass = "AArch64PreLegalizerCombinerHelperState";
  let AdditionalArguments = [];
}

// Matchdata for combines which replace a G_SHUFFLE_VECTOR with a
// target-specific opcode.
def shuffle_matchdata : GIDefMatchData<"ShuffleVectorPseudo">;

def rev : GICombineRule<
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchREV(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShuffleVectorPseudo(*${root}, ${matchinfo}); }])
>;

def zip : GICombineRule<
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchZip(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShuffleVectorPseudo(*${root}, ${matchinfo}); }])
>;

def uzp : GICombineRule<
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchUZP(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShuffleVectorPseudo(*${root}, ${matchinfo}); }])
>;

def dup: GICombineRule <
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchDup(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShuffleVectorPseudo(*${root}, ${matchinfo}); }])
>;

def trn : GICombineRule<
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchTRN(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShuffleVectorPseudo(*${root}, ${matchinfo}); }])
>;

def ext: GICombineRule <
  (defs root:$root, shuffle_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchEXT(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyEXT(*${root}, ${matchinfo}); }])
>;

// Combines which replace a G_SHUFFLE_VECTOR with a target-specific pseudo
// instruction.
def shuffle_vector_pseudos : GICombineGroup<[dup, rev, ext, zip, uzp, trn]>;

def AArch64PostLegalizerCombinerHelper
    : GICombinerHelper<"AArch64GenPostLegalizerCombinerHelper",
                       [erase_undef_store, combines_for_extload,
                        sext_already_extended, shuffle_vector_pseudos]> {
  let DisableRuleOption = "aarch64postlegalizercombiner-disable-rule";
}
