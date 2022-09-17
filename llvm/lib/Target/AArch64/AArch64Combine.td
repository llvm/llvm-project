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

def icmp_redundant_trunc_matchdata : GIDefMatchData<"Register">;
def icmp_redundant_trunc : GICombineRule<
  (defs root:$root, icmp_redundant_trunc_matchdata:$matchinfo),
  (match (wip_match_opcode G_ICMP):$root,
         [{ return matchICmpRedundantTrunc(*${root}, MRI, Helper.getKnownBits(), ${matchinfo}); }]),
  (apply [{ applyICmpRedundantTrunc(*${root}, MRI, B, Observer, ${matchinfo}); }])>;

// AArch64-specific offset folding for G_GLOBAL_VALUE.
def fold_global_offset_matchdata : GIDefMatchData<"std::pair<uint64_t, uint64_t>">;
def fold_global_offset : GICombineRule<
  (defs root:$root, fold_global_offset_matchdata:$matchinfo),
  (match (wip_match_opcode G_GLOBAL_VALUE):$root,
          [{ return matchFoldGlobalOffset(*${root}, MRI, ${matchinfo}); }]),
  (apply [{  return applyFoldGlobalOffset(*${root}, MRI, B, Observer, ${matchinfo});}])
>;

def AArch64PreLegalizerCombinerHelper: GICombinerHelper<
  "AArch64GenPreLegalizerCombinerHelper", [all_combines,
                                           fconstant_to_constant,
                                           icmp_redundant_trunc,
                                           fold_global_offset]> {
  let DisableRuleOption = "aarch64prelegalizercombiner-disable-rule";
  let StateClass = "AArch64PreLegalizerCombinerHelperState";
  let AdditionalArguments = [];
}

def AArch64O0PreLegalizerCombinerHelper: GICombinerHelper<
  "AArch64GenO0PreLegalizerCombinerHelper", [optnone_combines]> {
  let DisableRuleOption = "aarch64O0prelegalizercombiner-disable-rule";
  let StateClass = "AArch64O0PreLegalizerCombinerHelperState";
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

def shuf_to_ins_matchdata : GIDefMatchData<"std::tuple<Register, int, Register, int>">;
def shuf_to_ins: GICombineRule <
  (defs root:$root, shuf_to_ins_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return matchINS(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ return applyINS(*${root}, MRI, B, ${matchinfo}); }])
>;

def vashr_vlshr_imm_matchdata : GIDefMatchData<"int64_t">;
def vashr_vlshr_imm : GICombineRule<
  (defs root:$root, vashr_vlshr_imm_matchdata:$matchinfo),
  (match (wip_match_opcode G_ASHR, G_LSHR):$root,
          [{ return matchVAshrLshrImm(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyVAshrLshrImm(*${root}, MRI, ${matchinfo}); }])
>;

def form_duplane_matchdata :
  GIDefMatchData<"std::pair<unsigned, int>">;
def form_duplane : GICombineRule <
  (defs root:$root, form_duplane_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
          [{ return matchDupLane(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyDupLane(*${root}, MRI, B, ${matchinfo}); }])
>;

def shuffle_vector_lowering : GICombineGroup<[dup, rev, ext, zip, uzp, trn,
                                              form_duplane,
                                              shuf_to_ins]>;

def adjust_icmp_imm_matchdata :
  GIDefMatchData<"std::pair<uint64_t, CmpInst::Predicate>">;
def adjust_icmp_imm : GICombineRule <
  (defs root:$root, adjust_icmp_imm_matchdata:$matchinfo),
  (match (wip_match_opcode G_ICMP):$root,
          [{ return matchAdjustICmpImmAndPred(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyAdjustICmpImmAndPred(*${root}, ${matchinfo}, B, Observer); }])
>;

def swap_icmp_operands : GICombineRule <
  (defs root:$root),
  (match (wip_match_opcode G_ICMP):$root,
          [{ return trySwapICmpOperands(*${root}, MRI); }]),
  (apply [{ applySwapICmpOperands(*${root}, Observer); }])
>;

def icmp_lowering : GICombineGroup<[adjust_icmp_imm, swap_icmp_operands]>;

def extractvecelt_pairwise_add_matchdata : GIDefMatchData<"std::tuple<unsigned, LLT, Register>">;
def extractvecelt_pairwise_add : GICombineRule<
  (defs root:$root, extractvecelt_pairwise_add_matchdata:$matchinfo),
  (match (wip_match_opcode G_EXTRACT_VECTOR_ELT):$root,
          [{ return matchExtractVecEltPairwiseAdd(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyExtractVecEltPairwiseAdd(*${root}, MRI, B, ${matchinfo}); }])
>;

def mul_const_matchdata : GIDefMatchData<"std::function<void(MachineIRBuilder&, Register)>">;
def mul_const : GICombineRule<
  (defs root:$root, mul_const_matchdata:$matchinfo),
  (match (wip_match_opcode G_MUL):$root,
          [{ return matchAArch64MulConstCombine(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyAArch64MulConstCombine(*${root}, MRI, B, ${matchinfo}); }])
>;

def build_vector_to_dup : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
          [{ return matchBuildVectorToDup(*${root}, MRI); }]),
  (apply [{ return applyBuildVectorToDup(*${root}, MRI, B); }])
>;

def build_vector_lowering : GICombineGroup<[build_vector_to_dup]>;

def lower_vector_fcmp : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_FCMP):$root,
    [{ return lowerVectorFCMP(*${root}, MRI, B); }]),
  (apply [{}])>;

def form_truncstore_matchdata : GIDefMatchData<"Register">;
def form_truncstore : GICombineRule<
  (defs root:$root, form_truncstore_matchdata:$matchinfo),
  (match (wip_match_opcode G_STORE):$root,
          [{ return matchFormTruncstore(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyFormTruncstore(*${root}, MRI, B, Observer, ${matchinfo}); }])
>;

def fold_merge_to_zext : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_MERGE_VALUES):$d,
          [{ return matchFoldMergeToZext(*${d}, MRI); }]),
  (apply [{ applyFoldMergeToZext(*${d}, MRI, B, Observer); }])
>;

def mutate_anyext_to_zext : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_ANYEXT):$d,
          [{ return matchMutateAnyExtToZExt(*${d}, MRI); }]),
  (apply [{ applyMutateAnyExtToZExt(*${d}, MRI, B, Observer); }])
>;

def split_store_zero_128 : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_STORE):$d,
          [{ return matchSplitStoreZero128(*${d}, MRI); }]),
  (apply [{ applySplitStoreZero128(*${d}, MRI, B, Observer); }])
>;

// Post-legalization combines which should happen at all optimization levels.
// (E.g. ones that facilitate matching for the selector) For example, matching
// pseudos.
def AArch64PostLegalizerLoweringHelper
    : GICombinerHelper<"AArch64GenPostLegalizerLoweringHelper",
                       [shuffle_vector_lowering, vashr_vlshr_imm,
                        icmp_lowering, build_vector_lowering,
                        lower_vector_fcmp, form_truncstore]> {
  let DisableRuleOption = "aarch64postlegalizerlowering-disable-rule";
}

// Post-legalization combines which are primarily optimizations.
def AArch64PostLegalizerCombinerHelper
    : GICombinerHelper<"AArch64GenPostLegalizerCombinerHelper",
                       [copy_prop, combines_for_extload,
                        sext_trunc_sextload, mutate_anyext_to_zext,
                        hoist_logic_op_with_same_opcode_hands,
                        redundant_and, xor_of_and_with_same_reg,
                        extractvecelt_pairwise_add, redundant_or,
                        mul_const, redundant_sext_inreg,
                        form_bitfield_extract, rotate_out_of_range,
                        icmp_to_true_false_known_bits, merge_unmerge,
                        select_combines, fold_merge_to_zext,
                        constant_fold, identity_combines,
                        ptr_add_immed_chain, overlapping_and,
                        split_store_zero_128, undef_combines,
                        select_to_minmax]> {
  let DisableRuleOption = "aarch64postlegalizercombiner-disable-rule";
}
