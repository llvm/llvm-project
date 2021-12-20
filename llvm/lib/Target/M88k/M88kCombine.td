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

// Lower
//   G_AND $dst, (G_LSHR/G_ASHR $src, offset), (2**width - 1)
// to
//   EXTUrwo $dst, $src, width<offset>
def andshift_to_extu_matchdata : GIDefMatchData<"std::tuple<Register, uint32_t, uint32_t>">;
def andshift_to_extu : GICombineRule<
  (defs root:$root, andshift_to_extu_matchdata:$matchinfo),
  (match (wip_match_opcode G_AND):$root,
         [{ return matchAndShiftToExtU(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyAndShiftToExtU(*${root}, MRI, ${matchinfo}); }])>;

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

// Lower
//   G_AND $dst, $src, ~((2**width - 1) << offset)
// to
//   CLRrwo $dst, $src, width<offset>
def and_to_clr_matchdata : GIDefMatchData<"std::tuple<Register, uint32_t, uint32_t>">;
def and_to_clr : GICombineRule<
  (defs root:$root, and_to_clr_matchdata:$matchinfo),
  (match (wip_match_opcode G_AND):$root,
         [{ return matchAndToClr(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyAndToClr(*${root}, MRI, ${matchinfo}); }])>;

// Lower
//   G_OR $dst, $src, ((2**width - 1) << offset
// to
//   SETrwo $dst, $src, width<offset>
def or_to_set_matchdata : GIDefMatchData<"std::tuple<Register, uint32_t, uint32_t>">;
def or_to_set : GICombineRule<
  (defs root:$root, or_to_set_matchdata:$matchinfo),
  (match (wip_match_opcode G_OR):$root,
         [{ return matchOrToSet(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyOrToSet(*${root}, MRI, ${matchinfo}); }])>;


// Post-legalization combines which should happen at all optimization levels.
// (E.g. ones that facilitate matching for the selector) For example, matching
// pseudos.
def M88kPostLegalizerLoweringHelper
    : GICombinerHelper<"M88kGenPostLegalizerLoweringHelper",
                       [andshift_to_extu, shiftand_to_mak, and_to_clr,
                        or_to_set]> {
  let DisableRuleOption = "m88kpostlegalizerlowering-disable-rule";
}

// Post-legalization combines which are primarily optimizations.
def M88kPostLegalizerCombinerHelper
    : GICombinerHelper<"M88kGenPostLegalizerCombinerHelper",
                       []> {
  let DisableRuleOption = "m88kpostlegalizercombiner-disable-rule";
}
