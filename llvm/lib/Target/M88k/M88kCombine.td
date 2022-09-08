//=- M88kCombine.td - Define M88k Combine Rules --------------*- tablegen -*-=//
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

// Combine
//   G_AND $dst, (G_ASHR $src, offset), (2**width - 1)
// into
//   G_UBFX $dst, $src, offset, width
// under certain restrictions.
def bitfield_extract_from_and_ashr : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_AND):$root,
         [{ return matchBitfieldExtractFromAndAShr(*${root}, MRI,
                                                   ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// Like all_combines, but removed vector and fma combines.
def m88k_combines : GICombineGroup<[trivial_combines, combines_for_extload,
    combine_indexed_load_store, undef_combines, identity_combines, phi_combines,
    simplify_add_to_sub, hoist_logic_op_with_same_opcode_hands,
    reassocs, ptr_add_immed_chain,
    shl_ashr_to_sext_inreg, sext_inreg_of_load,
    width_reduction_combines, select_combines,
    known_bits_simplifications, ext_ext_fold,
    not_cmp_fold, opt_brcond_by_inverting_cond,
    unmerge_merge, fabs_fabs_fold, unmerge_cst, unmerge_dead_to_trunc,
    unmerge_zext_to_zext, merge_unmerge, trunc_ext_fold, trunc_shl,
    const_combines, xor_of_and_with_same_reg, ptr_add_with_zero,
    shift_immed_chain, shift_of_shifted_logic_chain, load_or_combine,
    truncstore_merge, div_rem_to_divrem, funnel_shift_to_rotate, //funnel_shift_combines,
    form_bitfield_extract, constant_fold, fabs_fneg_fold,
    intdiv_combines, mulh_combines, redundant_neg_operands,
    and_or_disjoint_mask, bitfield_extract_from_and_ashr]>;

// Like all_combines, but removed vector and fma combines.
def m88k_required_combines : GICombineGroup<[trivial_combines,
    shl_ashr_to_sext_inreg, sext_inreg_of_load,
    funnel_shift_to_rotate, form_bitfield_extract,
    and_or_disjoint_mask, bitfield_extract_from_and_ashr]>;

def M88kPreLegalizerCombinerHelper: GICombinerHelper<
  "M88kGenPreLegalizerCombinerHelper", [m88k_combines]> {
  let DisableRuleOption = "m88kprelegalizercombiner-disable-rule";
  let StateClass = "M88kPreLegalizerCombinerHelperState";
  let AdditionalArguments = [];
}

// Combine
//   G_ADD $dst, (G_ZEXT (G_ICMP eq, $src, 0))
// into
//   G_UADDE $dst, (G_USUBO 0, %src)
// under certain restrictions.
def subadd_from_icmpadd_matchdata : GIDefMatchData<"std::tuple<Register, Register, Register>">;
def subadd_from_icmpadd1 : GICombineRule<
  (defs root:$root, subadd_from_icmpadd_matchdata:$matchinfo),
  (match (wip_match_opcode G_ADD):$root,
         [{ return matchAddCmpToSubAdd(*${root}, MRI,
                                                   ${matchinfo}); }]),
  (apply [{ applyAddCmpToSubAdd(*${root}, MRI, ${matchinfo}); }])>;

def subadd_from_icmpadd2 : GICombineRule<
  (defs root:$dst, build_fn_matchinfo:$matchinfo),
  (match (G_ICMP $icmp, $cc, $src2, $src3),
         (G_ZEXT $zext, $icmp),
         (G_ADD $dst, $zext, $src1):$mi,
         [{ return matchAddCmpToSubAdd2(*${mi}, ${src1}, ${src2}, ${src3}, ${cc}, MRI,
                                                   ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${mi}, ${matchinfo}); }])>;

// Combine
//  Dst = G_ADD SrcA, (G_ZEXT (G_ICMP Pred, SrcB, SrcC)
// with:
//  - Pred = unsigned >=
//  - Pred = unsigned <=
//  - Pred = equal and either SrcB or SrcC being zero
// into the instruction sequence
//  Unused, Carry = G_USUBU SrcB', SrcC'
//  Dst, UnusedCarry = G_UADDE SrcA, Zero, Carry
// with SrcB' and SrcC' derived from SrcB and SrcC, inserting a zero constant
// value if necessary.
def addsub_from_addicmp : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ADD):$root,
         [{ return matchAddSubFromAddICmp(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// Combine
//  Unused, CarryBit = G_USUBU SrcB, SrcC
//  Carry = G_ZEXT CarryBit
//  Dst = G_UADD SrcA, Carry
// into the instruction sequence
//  Unused, Carry = G_USUBU SrcB, SrcC
//  Dst, UnusedCarry = G_UADDE SrcA, Zero, Carry
// inserting a zero constant value if necessary.
def addsub_from_addsub : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ADD):$root,
         [{ return matchAddSubFromAddSub(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// Post-legalization combines which are primarily optimizations.
def M88kPostLegalizerCombinerHelper: GICombinerHelper<
  "M88kGenPostLegalizerCombinerHelper", [
  //subadd_from_icmpadd1,
  //subadd_from_icmpadd2,
  addsub_from_addicmp,
  addsub_from_addsub,
  form_bitfield_extract, // Some rules require legalizer
  rotate_out_of_range // Required for left rotates, which produce a negative constant
  ]> {
  let DisableRuleOption = "m88kpostlegalizercombiner-disable-rule";
//  let StateClass = "M88kPostLegalizerCombinerHelperState";
//  let AdditionalArguments = [];
}

// Lower
//   G_SHL $dst, (G_AND $src, (2**width - 1)), offset
// to
//   MAKrwo $dst, $src, width<offset>
def shiftand_to_mak_matchdata : GIDefMatchData<"std::tuple<Register, uint32_t, uint32_t>">;
def shiftand_to_mak : GICombineRule<
  (defs root:$root, shiftand_to_mak_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHL):$root,
         [{ return matchShiftAndToMak(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyShiftAndToMak(*${root}, MRI, ${matchinfo}); }])>;


// Lower G_SDIV into G_UDIV on M88100.
def sdiv_to_udiv : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SDIV):$root,
         [{ return ReplaceSignedDiv; }]),
  (apply [{ applySDivtoUDiv(Observer, *${root}, MRI, B, KB); }])>;

def insert_div_by_zero_trap : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_UDIV, G_SDIV):$root,
         [{ return AddZeroDivCheck
                   && (!ReplaceSignedDiv || MIs[0]->getOpcode() == TargetOpcode::G_UDIV)
                   && matchInsertDivByZeroTrap(*${root}, MRI, KB); }]),
  (apply [{ applyInsertDivByZeroTrap(Observer, *${root}, MRI, B); }])>;

// Post-legalization combines which should happen at all optimization levels.
// (E.g. ones that facilitate matching for the selector) For example, matching
// pseudos.
def M88kPostLegalizerLoweringHelper
    : GICombinerHelper<"M88kGenPostLegalizerLoweringHelper",
                       [shiftand_to_mak, sdiv_to_udiv, insert_div_by_zero_trap]> {
  let DisableRuleOption = "m88kpostlegalizerlowering-disable-rule";
  let AdditionalArguments = [GICombinerHelperArg<"GISelKnownBits *", "KB">,
                             GICombinerHelperArg<"bool", "ReplaceSignedDiv">,
                             GICombinerHelperArg<"bool", "AddZeroDivCheck">];
}
