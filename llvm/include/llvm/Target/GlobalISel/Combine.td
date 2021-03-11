//===- Combine.td - Combine rule definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declare GlobalISel combine rules and provide mechanisms to opt-out.
//
//===----------------------------------------------------------------------===//

// Common base class for GICombineRule and GICombineGroup.
class GICombine {
  // See GICombineGroup. We only declare it here to make the tablegen pass
  // simpler.
  list<GICombine> Rules = ?;
}

// A group of combine rules that can be added to a GICombiner or another group.
class GICombineGroup<list<GICombine> rules> : GICombine {
  // The rules contained in this group. The rules in a group are flattened into
  // a single list and sorted into whatever order is most efficient. However,
  // they will never be re-ordered such that behaviour differs from the
  // specified order. It is therefore possible to use the order of rules in this
  // list to describe priorities.
  let Rules = rules;
}

class GICombinerHelperArg<string type, string name> {
  string Type = type;
  string Name = name;
}

// Declares a combiner helper class
class GICombinerHelper<string classname, list<GICombine> rules>
    : GICombineGroup<rules> {
  // The class name to use in the generated output.
  string Classname = classname;
  // The name of a run-time compiler option that will be generated to disable
  // specific rules within this combiner.
  string DisableRuleOption = ?;
  // The state class to inherit from (if any). The generated helper will inherit
  // from this class and will forward arguments to its constructors.
  string StateClass = "";
  // Any additional arguments that should be appended to the tryCombine*().
  list<GICombinerHelperArg> AdditionalArguments =
      [GICombinerHelperArg<"CombinerHelper &", "Helper">];
}
class GICombineRule<dag defs, dag match, dag apply> : GICombine {
  /// Defines the external interface of the match rule. This includes:
  /// * The names of the root nodes (requires at least one)
  /// See GIDefKind for details.
  dag Defs = defs;

  /// Defines the things which must be true for the pattern to match
  /// See GIMatchKind for details.
  dag Match = match;

  /// Defines the things which happen after the decision is made to apply a
  /// combine rule.
  /// See GIApplyKind for details.
  dag Apply = apply;
}

/// The operator at the root of a GICombineRule.Defs dag.
def defs;

/// All arguments of the defs operator must be subclasses of GIDefKind or
/// sub-dags whose operator is GIDefKindWithArgs.
class GIDefKind;
class GIDefKindWithArgs;
/// Declare a root node. There must be at least one of these in every combine
/// rule.
/// TODO: The plan is to elide `root` definitions and determine it from the DAG
///       itself with an overide for situations where the usual determination
///       is incorrect.
def root : GIDefKind;

/// Declares data that is passed from the match stage to the apply stage.
class GIDefMatchData<string type> : GIDefKind {
  /// A C++ type name indicating the storage type.
  string Type = type;
}

def extending_load_matchdata : GIDefMatchData<"PreferredTuple">;
def indexed_load_store_matchdata : GIDefMatchData<"IndexedLoadStoreMatchInfo">;
def instruction_steps_matchdata: GIDefMatchData<"InstructionStepsMatchInfo">;

/// The operator at the root of a GICombineRule.Match dag.
def match;
/// All arguments of the match operator must be either:
/// * A subclass of GIMatchKind
/// * A subclass of GIMatchKindWithArgs
/// * A subclass of Instruction
/// * A MIR code block (deprecated)
/// The GIMatchKind and GIMatchKindWithArgs cases are described in more detail
/// in their definitions below.
/// For the Instruction case, these are collected into a DAG where operand names
/// that occur multiple times introduce edges.
class GIMatchKind;
class GIMatchKindWithArgs;

/// In lieu of having proper macro support. Trivial one-off opcode checks can be
/// performed with this.
def wip_match_opcode : GIMatchKindWithArgs;

/// The operator at the root of a GICombineRule.Apply dag.
def apply;
/// All arguments of the apply operator must be subclasses of GIApplyKind, or
/// sub-dags whose operator is GIApplyKindWithArgs, or an MIR block
/// (deprecated).
class GIApplyKind;
class GIApplyKindWithArgs;

def register_matchinfo: GIDefMatchData<"Register">;

def copy_prop : GICombineRule<
  (defs root:$d),
  (match (COPY $d, $s):$mi,
         [{ return Helper.matchCombineCopy(*${mi}); }]),
  (apply [{ Helper.applyCombineCopy(*${mi}); }])>;

def extending_loads : GICombineRule<
  (defs root:$root, extending_load_matchdata:$matchinfo),
  (match (wip_match_opcode G_LOAD, G_SEXTLOAD, G_ZEXTLOAD):$root,
         [{ return Helper.matchCombineExtendingLoads(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyCombineExtendingLoads(*${root}, ${matchinfo}); }])>;
def combines_for_extload: GICombineGroup<[extending_loads]>;

def sext_trunc_sextload : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_SEXT_INREG):$d,
         [{ return Helper.matchSextTruncSextLoad(*${d}); }]),
  (apply [{ Helper.applySextTruncSextLoad(*${d}); }])>;

def sext_inreg_of_load_matchdata : GIDefMatchData<"std::tuple<Register, unsigned>">;
def sext_inreg_of_load : GICombineRule<
  (defs root:$root, sext_inreg_of_load_matchdata:$matchinfo),
  (match (wip_match_opcode G_SEXT_INREG):$root,
         [{ return Helper.matchSextInRegOfLoad(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applySextInRegOfLoad(*${root}, ${matchinfo}); }])>;

def combine_indexed_load_store : GICombineRule<
  (defs root:$root, indexed_load_store_matchdata:$matchinfo),
  (match (wip_match_opcode G_LOAD, G_SEXTLOAD, G_ZEXTLOAD, G_STORE):$root,
         [{ return Helper.matchCombineIndexedLoadStore(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyCombineIndexedLoadStore(*${root}, ${matchinfo}); }])>;

def opt_brcond_by_inverting_cond : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BR):$root,
         [{ return Helper.matchOptBrCondByInvertingCond(*${root}); }]),
  (apply [{ Helper.applyOptBrCondByInvertingCond(*${root}); }])>;

def ptr_add_immed_matchdata : GIDefMatchData<"PtrAddChain">;
def ptr_add_immed_chain : GICombineRule<
  (defs root:$d, ptr_add_immed_matchdata:$matchinfo),
  (match (wip_match_opcode G_PTR_ADD):$d,
         [{ return Helper.matchPtrAddImmedChain(*${d}, ${matchinfo}); }]),
  (apply [{ Helper.applyPtrAddImmedChain(*${d}, ${matchinfo}); }])>;

// Fold shift (shift base x), y -> shift base, (x+y), if shifts are same
def shift_immed_matchdata : GIDefMatchData<"RegisterImmPair">;
def shift_immed_chain : GICombineRule<
  (defs root:$d, shift_immed_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHL, G_ASHR, G_LSHR, G_SSHLSAT, G_USHLSAT):$d,
         [{ return Helper.matchShiftImmedChain(*${d}, ${matchinfo}); }]),
  (apply [{ Helper.applyShiftImmedChain(*${d}, ${matchinfo}); }])>;

// Transform shift (logic (shift X, C0), Y), C1
//        -> logic (shift X, (C0+C1)), (shift Y, C1), if shifts are same
def shift_of_shifted_logic_matchdata : GIDefMatchData<"ShiftOfShiftedLogic">;
def shift_of_shifted_logic_chain : GICombineRule<
  (defs root:$d, shift_of_shifted_logic_matchdata:$matchinfo),
  (match (wip_match_opcode G_SHL, G_ASHR, G_LSHR, G_USHLSAT, G_SSHLSAT):$d,
         [{ return Helper.matchShiftOfShiftedLogic(*${d}, ${matchinfo}); }]),
  (apply [{ Helper.applyShiftOfShiftedLogic(*${d}, ${matchinfo}); }])>;

def mul_to_shl_matchdata : GIDefMatchData<"unsigned">;
def mul_to_shl : GICombineRule<
  (defs root:$d, mul_to_shl_matchdata:$matchinfo),
  (match (G_MUL $d, $op1, $op2):$mi,
         [{ return Helper.matchCombineMulToShl(*${mi}, ${matchinfo}); }]),
  (apply [{ Helper.applyCombineMulToShl(*${mi}, ${matchinfo}); }])>;

// shl ([asz]ext x), y => zext (shl x, y), if shift does not overflow int
def reduce_shl_of_extend_matchdata : GIDefMatchData<"RegisterImmPair">;
def reduce_shl_of_extend : GICombineRule<
  (defs root:$dst, reduce_shl_of_extend_matchdata:$matchinfo),
  (match (G_SHL $dst, $src0, $src1):$mi,
         [{ return Helper.matchCombineShlOfExtend(*${mi}, ${matchinfo}); }]),
  (apply [{ Helper.applyCombineShlOfExtend(*${mi}, ${matchinfo}); }])>;

// [us]itofp(undef) = 0, because the result value is bounded.
def undef_to_fp_zero : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_UITOFP, G_SITOFP):$root,
         [{ return Helper.matchAnyExplicitUseIsUndef(*${root}); }]),
  (apply [{ Helper.replaceInstWithFConstant(*${root}, 0.0); }])>;

def undef_to_int_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_AND, G_MUL):$root,
         [{ return Helper.matchAnyExplicitUseIsUndef(*${root}); }]),
  (apply [{ Helper.replaceInstWithConstant(*${root}, 0); }])>;

def undef_to_negative_one: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_OR):$root,
         [{ return Helper.matchAnyExplicitUseIsUndef(*${root}); }]),
  (apply [{ Helper.replaceInstWithConstant(*${root}, -1); }])>;

def binop_left_undef_to_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SHL):$root,
         [{ return Helper.matchOperandIsUndef(*${root}, 1); }]),
  (apply [{ Helper.replaceInstWithConstant(*${root}, 0); }])>;

// Instructions where if any source operand is undef, the instruction can be
// replaced with undef.
def propagate_undef_any_op: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_ADD, G_FPTOSI, G_FPTOUI, G_SUB, G_XOR, G_TRUNC):$root,
         [{ return Helper.matchAnyExplicitUseIsUndef(*${root}); }]),
  (apply [{ Helper.replaceInstWithUndef(*${root}); }])>;

// Instructions where if all source operands are undef, the instruction can be
// replaced with undef.
def propagate_undef_all_ops: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
          [{ return Helper.matchAllExplicitUsesAreUndef(*${root}); }]),
  (apply [{ Helper.replaceInstWithUndef(*${root}); }])>;

// Replace a G_SHUFFLE_VECTOR with an undef mask with a G_IMPLICIT_DEF.
def propagate_undef_shuffle_mask: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SHUFFLE_VECTOR):$root,
         [{ return Helper.matchUndefShuffleVectorMask(*${root}); }]),
  (apply [{ Helper.replaceInstWithUndef(*${root}); }])>;

// Fold (cond ? x : x) -> x
def select_same_val: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SELECT):$root,
    [{ return Helper.matchSelectSameVal(*${root}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 2); }])
>;

// Fold (undef ? x : y) -> y
def select_undef_cmp: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SELECT):$root,
    [{ return Helper.matchUndefSelectCmp(*${root}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 2); }])
>;

// Fold (true ? x : y) -> x
// Fold (false ? x : y) -> y
def select_constant_cmp_matchdata : GIDefMatchData<"unsigned">;
def select_constant_cmp: GICombineRule<
  (defs root:$root, select_constant_cmp_matchdata:$matchinfo),
  (match (wip_match_opcode G_SELECT):$root,
    [{ return Helper.matchConstantSelectCmp(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, ${matchinfo}); }])
>;

// Fold x op 0 -> x
def right_identity_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SUB, G_ADD, G_OR, G_XOR, G_SHL, G_ASHR, G_LSHR,
                           G_PTR_ADD):$root,
    [{ return Helper.matchConstantOp(${root}->getOperand(2), 0); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 1); }])
>;

// Fold x op 1 -> x
def right_identity_one: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_MUL):$root,
    [{ return Helper.matchConstantOp(${root}->getOperand(2), 1); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 1); }])
>;

// Fold (x op x) - > x
def binop_same_val: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_AND, G_OR):$root,
    [{ return Helper.matchBinOpSameVal(*${root}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 1); }])
>;

// Fold (0 op x) - > 0
def binop_left_to_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SDIV, G_UDIV, G_SREM, G_UREM):$root,
    [{ return Helper.matchOperandIsZero(*${root}, 1); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 1); }])
>;

def urem_pow2_to_mask : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_UREM):$root,
    [{ return Helper.matchOperandIsKnownToBeAPowerOfTwo(*${root}, 2); }]),
  (apply [{ return Helper.applySimplifyURemByPow2(*${root}); }])
>;

// Fold (x op 0) - > 0
def binop_right_to_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_MUL):$root,
    [{ return Helper.matchOperandIsZero(*${root}, 2); }]),
  (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 2); }])
>;

// Erase stores of undef values.
def erase_undef_store : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_STORE):$root,
    [{ return Helper.matchUndefStore(*${root}); }]),
  (apply [{ return Helper.eraseInst(*${root}); }])
>;

def simplify_add_to_sub_matchinfo: GIDefMatchData<"std::tuple<Register, Register>">;
def simplify_add_to_sub: GICombineRule <
  (defs root:$root, simplify_add_to_sub_matchinfo:$info),
  (match (wip_match_opcode G_ADD):$root,
    [{ return Helper.matchSimplifyAddToSub(*${root}, ${info}); }]),
  (apply [{ return Helper.applySimplifyAddToSub(*${root}, ${info});}])
>;

// Fold fp_op(cst) to the constant result of the floating point operation.
def constant_fp_op_matchinfo: GIDefMatchData<"Optional<APFloat>">;
def constant_fp_op: GICombineRule <
  (defs root:$root, constant_fp_op_matchinfo:$info),
  (match (wip_match_opcode G_FNEG, G_FABS, G_FPTRUNC, G_FSQRT, G_FLOG2):$root,
    [{ return Helper.matchCombineConstantFoldFpUnary(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineConstantFoldFpUnary(*${root}, ${info}); }])
>;

// Fold int2ptr(ptr2int(x)) -> x
def p2i_to_i2p: GICombineRule<
  (defs root:$root, register_matchinfo:$info),
  (match (wip_match_opcode G_INTTOPTR):$root,
    [{ return Helper.matchCombineI2PToP2I(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineI2PToP2I(*${root}, ${info}); }])
>;

// Fold ptr2int(int2ptr(x)) -> x
def i2p_to_p2i: GICombineRule<
  (defs root:$root, register_matchinfo:$info),
  (match (wip_match_opcode G_PTRTOINT):$root,
    [{ return Helper.matchCombineP2IToI2P(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineP2IToI2P(*${root}, ${info}); }])
>;

// Fold add ptrtoint(x), y -> ptrtoint (ptr_add x), y
def add_p2i_to_ptradd_matchinfo : GIDefMatchData<"std::pair<Register, bool>">;
def add_p2i_to_ptradd : GICombineRule<
  (defs root:$root, add_p2i_to_ptradd_matchinfo:$info),
  (match (wip_match_opcode G_ADD):$root,
    [{ return Helper.matchCombineAddP2IToPtrAdd(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineAddP2IToPtrAdd(*${root}, ${info}); }])
>;

// Fold (ptr_add (int2ptr C1), C2) -> C1 + C2
def const_ptradd_to_i2p_matchinfo : GIDefMatchData<"int64_t">;
def const_ptradd_to_i2p: GICombineRule<
  (defs root:$root, const_ptradd_to_i2p_matchinfo:$info),
  (match (wip_match_opcode G_PTR_ADD):$root,
    [{ return Helper.matchCombineConstPtrAddToI2P(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineConstPtrAddToI2P(*${root}, ${info}); }])
>;

// Simplify: (logic_op (op x...), (op y...)) -> (op (logic_op x, y))
def hoist_logic_op_with_same_opcode_hands: GICombineRule <
  (defs root:$root, instruction_steps_matchdata:$info),
  (match (wip_match_opcode G_AND, G_OR, G_XOR):$root,
    [{ return Helper.matchHoistLogicOpWithSameOpcodeHands(*${root}, ${info}); }]),
  (apply [{ return Helper.applyBuildInstructionSteps(*${root}, ${info});}])
>;

// Fold ashr (shl x, C), C -> sext_inreg (C)
def shl_ashr_to_sext_inreg_matchinfo : GIDefMatchData<"std::tuple<Register, int64_t>">;
def shl_ashr_to_sext_inreg : GICombineRule<
  (defs root:$root, shl_ashr_to_sext_inreg_matchinfo:$info),
  (match (wip_match_opcode G_ASHR): $root,
    [{ return Helper.matchAshrShlToSextInreg(*${root}, ${info}); }]),
  (apply [{ return Helper.applyAshShlToSextInreg(*${root}, ${info});}])
>;
// Fold (x & y) -> x or (x & y) -> y when (x & y) is known to equal x or equal y.
def redundant_and: GICombineRule <
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_AND):$root,
         [{ return Helper.matchRedundantAnd(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fold (x | y) -> x or (x | y) -> y when (x | y) is known to equal x or equal y.
def redundant_or: GICombineRule <
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_OR):$root,
         [{ return Helper.matchRedundantOr(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// If the input is already sign extended, just drop the extension.
// sext_inreg x, K ->
//   if computeNumSignBits(x) >= (x.getScalarSizeInBits() - K + 1)
def redundant_sext_inreg: GICombineRule <
  (defs root:$root),
  (match (wip_match_opcode G_SEXT_INREG):$root,
         [{ return Helper.matchRedundantSExtInReg(*${root}); }]),
     (apply [{ return Helper.replaceSingleDefInstWithOperand(*${root}, 1); }])
>;

// Fold (anyext (trunc x)) -> x if the source type is same as
// the destination type.
def anyext_trunc_fold: GICombineRule <
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ANYEXT):$root,
         [{ return Helper.matchCombineAnyExtTrunc(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fold (zext (trunc x)) -> x if the source type is same as the destination type
// and truncated bits are known to be zero.
def zext_trunc_fold_matchinfo : GIDefMatchData<"Register">;
def zext_trunc_fold: GICombineRule <
  (defs root:$root, zext_trunc_fold_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ZEXT):$root,
         [{ return Helper.matchCombineZextTrunc(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fold ([asz]ext ([asz]ext x)) -> ([asz]ext x).
def ext_ext_fold_matchinfo : GIDefMatchData<"std::tuple<Register, unsigned>">;
def ext_ext_fold: GICombineRule <
  (defs root:$root, ext_ext_fold_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ANYEXT, G_SEXT, G_ZEXT):$root,
         [{ return Helper.matchCombineExtOfExt(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applyCombineExtOfExt(*${root}, ${matchinfo}); }])
>;

def not_cmp_fold_matchinfo : GIDefMatchData<"SmallVector<Register, 4>">;
def not_cmp_fold : GICombineRule<
  (defs root:$d, not_cmp_fold_matchinfo:$info),
  (match (wip_match_opcode G_XOR): $d,
  [{ return Helper.matchNotCmp(*${d}, ${info}); }]),
  (apply [{ return Helper.applyNotCmp(*${d}, ${info}); }])
>;

// Fold (fneg (fneg x)) -> x.
def fneg_fneg_fold: GICombineRule <
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FNEG):$root,
         [{ return Helper.matchCombineFNegOfFNeg(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fold (unmerge(merge x, y, z)) -> z, y, z.
def unmerge_merge_matchinfo : GIDefMatchData<"SmallVector<Register, 8>">;
def unmerge_merge : GICombineRule<
  (defs root:$d, unmerge_merge_matchinfo:$info),
  (match (wip_match_opcode G_UNMERGE_VALUES): $d,
  [{ return Helper.matchCombineUnmergeMergeToPlainValues(*${d}, ${info}); }]),
  (apply [{ return Helper.applyCombineUnmergeMergeToPlainValues(*${d}, ${info}); }])
>;

// Fold (fabs (fabs x)) -> (fabs x).
def fabs_fabs_fold: GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FABS):$root,
         [{ return Helper.matchCombineFAbsOfFAbs(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fold (unmerge cst) -> cst1, cst2, ...
def unmerge_cst_matchinfo : GIDefMatchData<"SmallVector<APInt, 8>">;
def unmerge_cst : GICombineRule<
  (defs root:$d, unmerge_cst_matchinfo:$info),
  (match (wip_match_opcode G_UNMERGE_VALUES): $d,
  [{ return Helper.matchCombineUnmergeConstant(*${d}, ${info}); }]),
  (apply [{ return Helper.applyCombineUnmergeConstant(*${d}, ${info}); }])
>;

// Transform x,y<dead> = unmerge z -> x = trunc z.
def unmerge_dead_to_trunc : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_UNMERGE_VALUES): $d,
  [{ return Helper.matchCombineUnmergeWithDeadLanesToTrunc(*${d}); }]),
  (apply [{ return Helper.applyCombineUnmergeWithDeadLanesToTrunc(*${d}); }])
>;

// Transform x,y = unmerge(zext(z)) -> x = zext z; y = 0.
def unmerge_zext_to_zext : GICombineRule<
  (defs root:$d),
  (match (wip_match_opcode G_UNMERGE_VALUES): $d,
  [{ return Helper.matchCombineUnmergeZExtToZExt(*${d}); }]),
  (apply [{ return Helper.applyCombineUnmergeZExtToZExt(*${d}); }])
>;

// Fold trunc ([asz]ext x) -> x or ([asz]ext x) or (trunc x).
def trunc_ext_fold_matchinfo : GIDefMatchData<"std::pair<Register, unsigned>">;
def trunc_ext_fold: GICombineRule <
  (defs root:$root, trunc_ext_fold_matchinfo:$matchinfo),
  (match (wip_match_opcode G_TRUNC):$root,
         [{ return Helper.matchCombineTruncOfExt(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applyCombineTruncOfExt(*${root}, ${matchinfo}); }])
>;

// Fold trunc (shl x, K) -> shl (trunc x), K => K < VT.getScalarSizeInBits().
def trunc_shl_matchinfo : GIDefMatchData<"std::pair<Register, Register>">;
def trunc_shl: GICombineRule <
  (defs root:$root, trunc_shl_matchinfo:$matchinfo),
  (match (wip_match_opcode G_TRUNC):$root,
         [{ return Helper.matchCombineTruncOfShl(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applyCombineTruncOfShl(*${root}, ${matchinfo}); }])
>;

// Transform (mul x, -1) -> (sub 0, x)
def mul_by_neg_one: GICombineRule <
  (defs root:$root),
  (match (wip_match_opcode G_MUL):$root,
         [{ return Helper.matchConstantOp(${root}->getOperand(2), -1); }]),
  (apply [{ return Helper.applyCombineMulByNegativeOne(*${root}); }])
>;

// Fold (xor (and x, y), y) -> (and (not x), y)
def xor_of_and_with_same_reg_matchinfo :
    GIDefMatchData<"std::pair<Register, Register>">;
def xor_of_and_with_same_reg: GICombineRule <
  (defs root:$root, xor_of_and_with_same_reg_matchinfo:$matchinfo),
  (match (wip_match_opcode G_XOR):$root,
         [{ return Helper.matchXorOfAndWithSameReg(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applyXorOfAndWithSameReg(*${root}, ${matchinfo}); }])
>;

// Transform (ptr_add 0, x) -> (int_to_ptr x)
def ptr_add_with_zero: GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_PTR_ADD):$root,
         [{ return Helper.matchPtrAddZero(*${root}); }]),
  (apply [{ return Helper.applyPtrAddZero(*${root}); }])>;

def regs_small_vec : GIDefMatchData<"SmallVector<Register, 4>">;
def combine_insert_vec_elts_build_vector : GICombineRule<
  (defs root:$root, regs_small_vec:$info),
  (match (wip_match_opcode G_INSERT_VECTOR_ELT):$root,
    [{ return Helper.matchCombineInsertVecElts(*${root}, ${info}); }]),
  (apply [{ return Helper.applyCombineInsertVecElts(*${root}, ${info}); }])>;

def load_or_combine_matchdata :
GIDefMatchData<"std::function<void(MachineIRBuilder &)>">;
def load_or_combine : GICombineRule<
  (defs root:$root, load_or_combine_matchdata:$info),
  (match (wip_match_opcode G_OR):$root,
    [{ return Helper.matchLoadOrCombine(*${root}, ${info}); }]),
  (apply [{ return Helper.applyLoadOrCombine(*${root}, ${info}); }])>;

def extend_through_phis_matchdata: GIDefMatchData<"MachineInstr*">;
def extend_through_phis : GICombineRule<
  (defs root:$root, extend_through_phis_matchdata:$matchinfo),
  (match (wip_match_opcode G_PHI):$root,
    [{ return Helper.matchExtendThroughPhis(*${root}, ${matchinfo}); }]),
  (apply [{ return Helper.applyExtendThroughPhis(*${root}, ${matchinfo}); }])>;

// Currently only the one combine above.
def insert_vec_elt_combines : GICombineGroup<
                            [combine_insert_vec_elts_build_vector]>;

def extract_vec_elt_build_vec : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_EXTRACT_VECTOR_ELT):$root,
    [{ return Helper.matchExtractVecEltBuildVec(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyExtractVecEltBuildVec(*${root}, ${matchinfo}); }])>;

def extract_vec_elt_combines : GICombineGroup<[extract_vec_elt_build_vec]>;

// FIXME: These should use the custom predicate feature once it lands.
def undef_combines : GICombineGroup<[undef_to_fp_zero, undef_to_int_zero,
                                     undef_to_negative_one,
                                     binop_left_undef_to_zero,
                                     propagate_undef_any_op,
                                     propagate_undef_all_ops,
                                     propagate_undef_shuffle_mask,
                                     erase_undef_store]>;

def identity_combines : GICombineGroup<[select_same_val, right_identity_zero,
                                        binop_same_val, binop_left_to_zero,
                                        binop_right_to_zero, p2i_to_i2p,
                                        i2p_to_p2i, anyext_trunc_fold,
                                        fneg_fneg_fold, right_identity_one]>;

def const_combines : GICombineGroup<[constant_fp_op, const_ptradd_to_i2p]>;

def known_bits_simplifications : GICombineGroup<[
  redundant_and, redundant_sext_inreg, redundant_or, urem_pow2_to_mask,
  zext_trunc_fold]>;

def width_reduction_combines : GICombineGroup<[reduce_shl_of_extend]>;

def phi_combines : GICombineGroup<[extend_through_phis]>;

def select_combines : GICombineGroup<[select_undef_cmp, select_constant_cmp]>;

def trivial_combines : GICombineGroup<[copy_prop, mul_to_shl, add_p2i_to_ptradd,
                                       mul_by_neg_one]>;

def all_combines : GICombineGroup<[trivial_combines, insert_vec_elt_combines,
    extract_vec_elt_combines, ptr_add_immed_chain, combines_for_extload,
    combine_indexed_load_store, undef_combines, identity_combines, phi_combines,
    simplify_add_to_sub, hoist_logic_op_with_same_opcode_hands,
    shl_ashr_to_sext_inreg, sext_inreg_of_load,
    width_reduction_combines, select_combines,
    known_bits_simplifications, ext_ext_fold,
    not_cmp_fold, opt_brcond_by_inverting_cond,
    unmerge_merge, fabs_fabs_fold, unmerge_cst, unmerge_dead_to_trunc,
    unmerge_zext_to_zext, trunc_ext_fold, trunc_shl,
    const_combines, xor_of_and_with_same_reg, ptr_add_with_zero,
    shift_immed_chain, shift_of_shifted_logic_chain, load_or_combine]>;
