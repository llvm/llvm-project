//===-- SOPInstructions.td - SOP Instruction Definitions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def GPRIdxMode : CustomOperand<i32>;

class SOP_Pseudo<string opName, dag outs, dag ins, string asmOps,
                  list<dag> pattern=[]> :
    InstSI<outs, ins, "", pattern>,
    SIMCInstr<opName, SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_sdst = 0;
}

//===----------------------------------------------------------------------===//
// SOP1 Instructions
//===----------------------------------------------------------------------===//

class SOP1_Pseudo <string opName, dag outs, dag ins,
                   string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, " " # asmOps, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOP1 = 1;
  let SchedRW = [WriteSALU];
  let Size = 4;
  let UseNamedOperandTable = 1;

  bits<1> has_src0 = 1;
  let has_sdst = 1;
}

class SOP1_Real<bits<8> op, SOP1_Pseudo ps, string real_name = ps.Mnemonic> :
  InstSI <ps.OutOperandList, ps.InOperandList,
          real_name # ps.AsmOperands>,
  Enc32 {

  let SALU = 1;
  let SOP1 = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let Size = 4;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let SchedRW            = ps.SchedRW;
  let mayLoad            = ps.mayLoad;
  let mayStore           = ps.mayStore;

  // encoding
  bits<7> sdst;
  bits<8> src0;

  let Inst{7-0} = !if(ps.has_src0, src0, ?);
  let Inst{15-8} = op;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{31-23} = 0x17d; //encoding;
}

class SOP1_32 <string opName, list<dag> pattern=[], bit tied_in = 0> : SOP1_Pseudo <
  opName, (outs SReg_32:$sdst),
  !if(tied_in, (ins SSrc_b32:$src0, SReg_32:$sdst_in),
               (ins SSrc_b32:$src0)),
  "$sdst, $src0", pattern> {
  let Constraints = !if(tied_in, "$sdst = $sdst_in", "");
}

// Only register input allowed.
class SOP1_32R <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SReg_32:$sdst), (ins SReg_32:$src0),
  "$sdst, $src0", pattern>;

// 32-bit input, no output.
class SOP1_0_32 <string opName, list<dag> pattern = []> : SOP1_Pseudo <
  opName, (outs), (ins SSrc_b32:$src0),
  "$src0", pattern> {
  let has_sdst = 0;
}

// Special case for movreld where sdst is treated as a use operand.
class SOP1_32_movreld <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs), (ins SReg_32:$sdst, SSrc_b32:$src0),
  "$sdst, $src0", pattern>;

// Special case for movreld where sdst is treated as a use operand.
class SOP1_64_movreld <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs), (ins SReg_64:$sdst, SSrc_b64:$src0),
  "$sdst, $src0", pattern
>;

class SOP1_0_32R <string opName, list<dag> pattern = []> : SOP1_Pseudo <
  opName, (outs), (ins SReg_32:$src0),
  "$src0", pattern> {
  let has_sdst = 0;
}

class SOP1_64 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SReg_64:$sdst), (ins SSrc_b64:$src0),
  "$sdst, $src0", pattern
>;

// Only register input allowed.
class SOP1_64R <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SReg_64:$sdst), (ins SReg_64:$src0),
  "$sdst, $src0", pattern
>;

// 64-bit input, 32-bit output.
class SOP1_32_64 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SReg_32:$sdst), (ins SSrc_b64:$src0),
  "$sdst, $src0", pattern
>;

// 32-bit input, 64-bit output.
class SOP1_64_32 <string opName, list<dag> pattern=[], bit tied_in = 0> : SOP1_Pseudo <
  opName, (outs SReg_64:$sdst),
  !if(tied_in, (ins SSrc_b32:$src0, SReg_64:$sdst_in),
               (ins SSrc_b32:$src0)),
  "$sdst, $src0", pattern> {
  let Constraints = !if(tied_in, "$sdst = $sdst_in", "");
}

// no input, 64-bit output.
class SOP1_64_0 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SReg_64:$sdst), (ins), "$sdst", pattern> {
  let has_src0 = 0;
}

// 64-bit input, no output
class SOP1_1 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs), (ins SReg_64:$src0), "$src0", pattern> {
  let has_sdst = 0;
}


class UniformUnaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0),
  (Op $src0),
  [{ return !N->isDivergent(); }]> {
  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}

class UniformBinFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1),
  (Op $src0, $src1),
  [{ return !N->isDivergent(); }]> {
  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}

class UniformTernaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1, node:$src2),
  (Op $src0, $src1, $src2),
  [{ return !N->isDivergent(); }]> {
  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}

class DivergentBinFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1),
  (Op $src0, $src1),
  [{ return N->isDivergent(); }]> {
  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}


let isMoveImm = 1 in {
  let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
    def S_MOV_B32 : SOP1_32 <"s_mov_b32">;
    def S_MOV_B64 : SOP1_64 <"s_mov_b64">;
  } // End isReMaterializable = 1

  let Uses = [SCC] in {
    def S_CMOV_B32 : SOP1_32 <"s_cmov_b32">;
    def S_CMOV_B64 : SOP1_64 <"s_cmov_b64">;
  } // End Uses = [SCC]
} // End isMoveImm = 1

let Defs = [SCC] in {
  def S_NOT_B32 : SOP1_32 <"s_not_b32",
    [(set i32:$sdst, (UniformUnaryFrag<not> i32:$src0))]
  >;

  def S_NOT_B64 : SOP1_64 <"s_not_b64",
    [(set i64:$sdst, (UniformUnaryFrag<not> i64:$src0))]
  >;
  def S_WQM_B32 : SOP1_32 <"s_wqm_b32">;
  def S_WQM_B64 : SOP1_64 <"s_wqm_b64">;
} // End Defs = [SCC]


let WaveSizePredicate = isWave32 in {
def : GCNPat <
  (int_amdgcn_wqm_vote i1:$src0),
  (S_WQM_B32 SSrc_b32:$src0)
>;
}

let WaveSizePredicate = isWave64 in {
def : GCNPat <
  (int_amdgcn_wqm_vote i1:$src0),
  (S_WQM_B64 SSrc_b64:$src0)
>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def S_BREV_B32 : SOP1_32 <"s_brev_b32",
  [(set i32:$sdst, (UniformUnaryFrag<bitreverse> i32:$src0))]
>;
def S_BREV_B64 : SOP1_64 <"s_brev_b64",
  [(set i64:$sdst, (UniformUnaryFrag<bitreverse> i64:$src0))]
>;
} // End isReMaterializable = 1, isAsCheapAsAMove = 1

let Defs = [SCC] in {
def S_BCNT0_I32_B32 : SOP1_32 <"s_bcnt0_i32_b32">;
def S_BCNT0_I32_B64 : SOP1_32_64 <"s_bcnt0_i32_b64">;
def S_BCNT1_I32_B32 : SOP1_32 <"s_bcnt1_i32_b32",
  [(set i32:$sdst, (UniformUnaryFrag<ctpop> i32:$src0))]
>;
def S_BCNT1_I32_B64 : SOP1_32_64 <"s_bcnt1_i32_b64",
  [(set i32:$sdst, (UniformUnaryFrag<ctpop> i64:$src0))]
>;
} // End Defs = [SCC]

let isReMaterializable = 1 in {
def S_FF0_I32_B32 : SOP1_32 <"s_ff0_i32_b32">;
def S_FF0_I32_B64 : SOP1_32_64 <"s_ff0_i32_b64">;
def S_FF1_I32_B64 : SOP1_32_64 <"s_ff1_i32_b64",
  [(set i32:$sdst, (UniformUnaryFrag<AMDGPUffbl_b32> i64:$src0))]
>;

def S_FF1_I32_B32 : SOP1_32 <"s_ff1_i32_b32",
  [(set i32:$sdst, (UniformUnaryFrag<AMDGPUffbl_b32> i32:$src0))]
>;

def S_FLBIT_I32_B32 : SOP1_32 <"s_flbit_i32_b32",
  [(set i32:$sdst, (UniformUnaryFrag<AMDGPUffbh_u32> i32:$src0))]
>;

def S_FLBIT_I32_B64 : SOP1_32_64 <"s_flbit_i32_b64",
  [(set i32:$sdst, (UniformUnaryFrag<AMDGPUffbh_u32> i64:$src0))]
>;
def S_FLBIT_I32 : SOP1_32 <"s_flbit_i32",
  [(set i32:$sdst, (UniformUnaryFrag<AMDGPUffbh_i32> i32:$src0))]
>;
def S_FLBIT_I32_I64 : SOP1_32_64 <"s_flbit_i32_i64">;
def S_SEXT_I32_I8 : SOP1_32 <"s_sext_i32_i8",
  [(set i32:$sdst, (UniformSextInreg<i8> i32:$src0))]
>;
def S_SEXT_I32_I16 : SOP1_32 <"s_sext_i32_i16",
  [(set i32:$sdst, (UniformSextInreg<i16> i32:$src0))]
>;
} // End isReMaterializable = 1

def S_BITSET0_B32 : SOP1_32    <"s_bitset0_b32", [], 1>;
def S_BITSET0_B64 : SOP1_64_32 <"s_bitset0_b64", [], 1>;
def S_BITSET1_B32 : SOP1_32    <"s_bitset1_b32", [], 1>;
def S_BITSET1_B64 : SOP1_64_32 <"s_bitset1_b64", [], 1>;

def S_GETPC_B64 : SOP1_64_0  <"s_getpc_b64",
  [(set i64:$sdst, (int_amdgcn_s_getpc))]
>;

let isTerminator = 1, isBarrier = 1, SchedRW = [WriteBranch] in {

let isBranch = 1, isIndirectBranch = 1 in {
def S_SETPC_B64 : SOP1_1  <"s_setpc_b64">;
} // End isBranch = 1, isIndirectBranch = 1

let isReturn = 1 in {
// Define variant marked as return rather than branch.
def S_SETPC_B64_return : SOP1_1<"">;
}
} // End isTerminator = 1, isBarrier = 1

let isCall = 1 in {
def S_SWAPPC_B64 : SOP1_64 <"s_swappc_b64"
>;
}

def S_RFE_B64 : SOP1_1  <"s_rfe_b64">;

let hasSideEffects = 1, Uses = [EXEC], Defs = [EXEC, SCC] in {

def S_AND_SAVEEXEC_B64 : SOP1_64 <"s_and_saveexec_b64">;
def S_OR_SAVEEXEC_B64 : SOP1_64 <"s_or_saveexec_b64">;
def S_XOR_SAVEEXEC_B64 : SOP1_64 <"s_xor_saveexec_b64">;
def S_ANDN2_SAVEEXEC_B64 : SOP1_64 <"s_andn2_saveexec_b64">;
def S_ORN2_SAVEEXEC_B64 : SOP1_64 <"s_orn2_saveexec_b64">;
def S_NAND_SAVEEXEC_B64 : SOP1_64 <"s_nand_saveexec_b64">;
def S_NOR_SAVEEXEC_B64 : SOP1_64 <"s_nor_saveexec_b64">;
def S_XNOR_SAVEEXEC_B64 : SOP1_64 <"s_xnor_saveexec_b64">;

} // End hasSideEffects = 1, Uses = [EXEC], Defs = [EXEC, SCC]

def S_QUADMASK_B32 : SOP1_32 <"s_quadmask_b32">;
def S_QUADMASK_B64 : SOP1_64 <"s_quadmask_b64">;

let Uses = [M0] in {
def S_MOVRELS_B32 : SOP1_32R <"s_movrels_b32">;
def S_MOVRELS_B64 : SOP1_64R <"s_movrels_b64">;
def S_MOVRELD_B32 : SOP1_32_movreld <"s_movreld_b32">;
def S_MOVRELD_B64 : SOP1_64_movreld <"s_movreld_b64">;
} // End Uses = [M0]

let SubtargetPredicate = isGFX6GFX7GFX8GFX9 in {
def S_CBRANCH_JOIN : SOP1_0_32R <"s_cbranch_join">;
} // End SubtargetPredicate = isGFX6GFX7GFX8GFX9

let Defs = [SCC] in {
def S_ABS_I32 : SOP1_32 <"s_abs_i32",
    [(set i32:$sdst, (UniformUnaryFrag<abs> i32:$src0))]
  >;
} // End Defs = [SCC]

let SubtargetPredicate = HasVGPRIndexMode in {
def S_SET_GPR_IDX_IDX : SOP1_0_32<"s_set_gpr_idx_idx"> {
  let Uses = [M0, MODE];
  let Defs = [M0, MODE];
}
}

let SubtargetPredicate = isGFX9Plus in {
  let hasSideEffects = 1, Defs = [EXEC, SCC], Uses = [EXEC] in {
    def S_ANDN1_SAVEEXEC_B64 : SOP1_64<"s_andn1_saveexec_b64">;
    def S_ORN1_SAVEEXEC_B64  : SOP1_64<"s_orn1_saveexec_b64">;
    def S_ANDN1_WREXEC_B64   : SOP1_64<"s_andn1_wrexec_b64">;
    def S_ANDN2_WREXEC_B64   : SOP1_64<"s_andn2_wrexec_b64">;
  } // End hasSideEffects = 1, Defs = [EXEC, SCC], Uses = [EXEC]

  let isReMaterializable = 1 in
  def S_BITREPLICATE_B64_B32 : SOP1_64_32<"s_bitreplicate_b64_b32">;
} // End SubtargetPredicate = isGFX9Plus

let SubtargetPredicate = isGFX10Plus in {
  let hasSideEffects = 1, Defs = [EXEC, SCC], Uses = [EXEC] in {
    def S_AND_SAVEEXEC_B32   : SOP1_32<"s_and_saveexec_b32">;
    def S_OR_SAVEEXEC_B32    : SOP1_32<"s_or_saveexec_b32">;
    def S_XOR_SAVEEXEC_B32   : SOP1_32<"s_xor_saveexec_b32">;
    def S_ANDN2_SAVEEXEC_B32 : SOP1_32<"s_andn2_saveexec_b32">;
    def S_ORN2_SAVEEXEC_B32  : SOP1_32<"s_orn2_saveexec_b32">;
    def S_NAND_SAVEEXEC_B32  : SOP1_32<"s_nand_saveexec_b32">;
    def S_NOR_SAVEEXEC_B32   : SOP1_32<"s_nor_saveexec_b32">;
    def S_XNOR_SAVEEXEC_B32  : SOP1_32<"s_xnor_saveexec_b32">;
    def S_ANDN1_SAVEEXEC_B32 : SOP1_32<"s_andn1_saveexec_b32">;
    def S_ORN1_SAVEEXEC_B32  : SOP1_32<"s_orn1_saveexec_b32">;
    def S_ANDN1_WREXEC_B32   : SOP1_32<"s_andn1_wrexec_b32">;
    def S_ANDN2_WREXEC_B32   : SOP1_32<"s_andn2_wrexec_b32">;
  } // End hasSideEffects = 1, Defs = [EXEC, SCC], Uses = [EXEC]

  let Uses = [M0] in {
    def S_MOVRELSD_2_B32 : SOP1_32<"s_movrelsd_2_b32">;
  } // End Uses = [M0]
} // End SubtargetPredicate = isGFX10Plus

let SubtargetPredicate = isGFX11Plus in {
  let hasSideEffects = 1 in {
    // For s_sendmsg_rtn_* the src0 field encodes the message type directly; it
    // is not an SGPR number.
    def S_SENDMSG_RTN_B32 : SOP1_Pseudo<
      "s_sendmsg_rtn_b32", (outs SReg_32:$sdst), (ins SendMsg:$src0),
      "$sdst, $src0", [(set i32:$sdst, (int_amdgcn_s_sendmsg_rtn timm:$src0))]
    >;
    def S_SENDMSG_RTN_B64 : SOP1_Pseudo<
      "s_sendmsg_rtn_b64", (outs SReg_64:$sdst), (ins SendMsg:$src0),
      "$sdst, $src0", [(set i64:$sdst, (int_amdgcn_s_sendmsg_rtn timm:$src0))]
    >;
  }
} // End SubtargetPredicate = isGFX11Plus

//===----------------------------------------------------------------------===//
// SOP2 Instructions
//===----------------------------------------------------------------------===//

class SOP2_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, " " # asmOps, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOP2 = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;

  let has_sdst = 1;

  // Pseudo instructions have no encodings, but adding this field here allows
  // us to do:
  // let sdst = xxx in {
  // for multiclasses that include both real and pseudo instructions.
  // field bits<7> sdst = 0;
  // let Size = 4; // Do we need size here?
}

class SOP2_Real<bits<7> op, SOP_Pseudo ps, string real_name = ps.Mnemonic> :
  InstSI <ps.OutOperandList, ps.InOperandList,
          real_name # ps.AsmOperands>,
  Enc32 {
  let SALU = 1;
  let SOP2 = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let TSFlags              = ps.TSFlags;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;

  // encoding
  bits<7> sdst;
  bits<8> src0;
  bits<8> src1;

  let Inst{7-0}   = src0;
  let Inst{15-8}  = src1;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{29-23} = op;
  let Inst{31-30} = 0x2; // encoding
}


class SOP2_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SReg_32:$sdst), (ins SSrc_b32:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern
>;

class SOP2_64 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SReg_64:$sdst), (ins SSrc_b64:$src0, SSrc_b64:$src1),
  "$sdst, $src0, $src1", pattern
>;

class SOP2_64_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SReg_64:$sdst), (ins SSrc_b64:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern
>;

class SOP2_64_32_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SReg_64:$sdst), (ins SSrc_b32:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern
>;


let Defs = [SCC] in { // Carry out goes to SCC
let isCommutable = 1 in {
def S_ADD_U32 : SOP2_32 <"s_add_u32">;
def S_ADD_I32 : SOP2_32 <"s_add_i32",
  [(set i32:$sdst, (UniformBinFrag<add> SSrc_b32:$src0, SSrc_b32:$src1))]
>;
} // End isCommutable = 1

def S_SUB_U32 : SOP2_32 <"s_sub_u32">;
def S_SUB_I32 : SOP2_32 <"s_sub_i32",
  [(set i32:$sdst, (UniformBinFrag<sub> SSrc_b32:$src0, SSrc_b32:$src1))]
>;

let Uses = [SCC] in { // Carry in comes from SCC
let isCommutable = 1 in {
def S_ADDC_U32 : SOP2_32 <"s_addc_u32",
  [(set i32:$sdst, (UniformBinFrag<adde> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]>;
} // End isCommutable = 1

def S_SUBB_U32 : SOP2_32 <"s_subb_u32",
  [(set i32:$sdst, (UniformBinFrag<sube> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]>;
} // End Uses = [SCC]

let isCommutable = 1 in {
def S_MIN_I32 : SOP2_32 <"s_min_i32",
  [(set i32:$sdst, (UniformBinFrag<smin> i32:$src0, i32:$src1))]
>;
def S_MIN_U32 : SOP2_32 <"s_min_u32",
  [(set i32:$sdst, (UniformBinFrag<umin> i32:$src0, i32:$src1))]
>;
def S_MAX_I32 : SOP2_32 <"s_max_i32",
  [(set i32:$sdst, (UniformBinFrag<smax> i32:$src0, i32:$src1))]
>;
def S_MAX_U32 : SOP2_32 <"s_max_u32",
  [(set i32:$sdst, (UniformBinFrag<umax> i32:$src0, i32:$src1))]
>;
} // End isCommutable = 1
} // End Defs = [SCC]

def SelectPat : PatFrag <
  (ops node:$src1, node:$src2),
  (select SCC, $src1, $src2),
  [{ return !N->isDivergent(); }]
>;

let Uses = [SCC] in {
  let AddedComplexity = 20 in {
    def S_CSELECT_B32 : SOP2_32 <"s_cselect_b32",
      [(set i32:$sdst, (SelectPat i32:$src0, i32:$src1))]
    >;
  }

  def S_CSELECT_B64 : SOP2_64 <"s_cselect_b64">;
} // End Uses = [SCC]

let Defs = [SCC] in {
let isCommutable = 1 in {
def S_AND_B32 : SOP2_32 <"s_and_b32",
  [(set i32:$sdst, (UniformBinFrag<and> i32:$src0, i32:$src1))]
>;

def S_AND_B64 : SOP2_64 <"s_and_b64",
  [(set i64:$sdst, (UniformBinFrag<and> i64:$src0, i64:$src1))]
>;

def S_OR_B32 : SOP2_32 <"s_or_b32",
  [(set i32:$sdst, (UniformBinFrag<or> i32:$src0, i32:$src1))]
>;

def S_OR_B64 : SOP2_64 <"s_or_b64",
  [(set i64:$sdst, (UniformBinFrag<or> i64:$src0, i64:$src1))]
>;

def S_XOR_B32 : SOP2_32 <"s_xor_b32",
  [(set i32:$sdst, (UniformBinFrag<xor> i32:$src0, i32:$src1))]
>;

def S_XOR_B64 : SOP2_64 <"s_xor_b64",
  [(set i64:$sdst, (UniformBinFrag<xor> i64:$src0, i64:$src1))]
>;

def S_XNOR_B32 : SOP2_32 <"s_xnor_b32",
  [(set i32:$sdst, (UniformUnaryFrag<not> (xor_oneuse i32:$src0, i32:$src1)))]
>;

def S_XNOR_B64 : SOP2_64 <"s_xnor_b64",
  [(set i64:$sdst, (UniformUnaryFrag<not> (xor_oneuse i64:$src0, i64:$src1)))]
>;

def S_NAND_B32 : SOP2_32 <"s_nand_b32",
  [(set i32:$sdst, (UniformUnaryFrag<not> (and_oneuse i32:$src0, i32:$src1)))]
>;

def S_NAND_B64 : SOP2_64 <"s_nand_b64",
  [(set i64:$sdst, (UniformUnaryFrag<not> (and_oneuse i64:$src0, i64:$src1)))]
>;

def S_NOR_B32 : SOP2_32 <"s_nor_b32",
  [(set i32:$sdst, (UniformUnaryFrag<not> (or_oneuse i32:$src0, i32:$src1)))]
>;

def S_NOR_B64 : SOP2_64 <"s_nor_b64",
  [(set i64:$sdst, (UniformUnaryFrag<not> (or_oneuse i64:$src0, i64:$src1)))]
>;
} // End isCommutable = 1

// There are also separate patterns for types other than i32
def S_ANDN2_B32 : SOP2_32 <"s_andn2_b32",
  [(set i32:$sdst, (UniformBinFrag<and> i32:$src0, (UniformUnaryFrag<not> i32:$src1)))]
>;

def S_ANDN2_B64 : SOP2_64 <"s_andn2_b64",
  [(set i64:$sdst, (UniformBinFrag<and> i64:$src0, (UniformUnaryFrag<not> i64:$src1)))]
>;

def S_ORN2_B32 : SOP2_32 <"s_orn2_b32",
  [(set i32:$sdst, (UniformBinFrag<or> i32:$src0, (UniformUnaryFrag<not> i32:$src1)))]
>;

def S_ORN2_B64 : SOP2_64 <"s_orn2_b64",
  [(set i64:$sdst, (UniformBinFrag<or> i64:$src0, (UniformUnaryFrag<not> i64:$src1)))]
>;
} // End Defs = [SCC]

// Use added complexity so these patterns are preferred to the VALU patterns.
let AddedComplexity = 1 in {

let Defs = [SCC] in {
// TODO: b64 versions require VOP3 change since v_lshlrev_b64 is VOP3
def S_LSHL_B32 : SOP2_32 <"s_lshl_b32",
  [(set SReg_32:$sdst, (UniformBinFrag<cshl_32> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]
>;
def S_LSHL_B64 : SOP2_64_32 <"s_lshl_b64",
  [(set SReg_64:$sdst, (UniformBinFrag<cshl_64> (i64 SSrc_b64:$src0), (i32 SSrc_b32:$src1)))]
>;
def S_LSHR_B32 : SOP2_32 <"s_lshr_b32",
  [(set SReg_32:$sdst, (UniformBinFrag<csrl_32> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]
>;
def S_LSHR_B64 : SOP2_64_32 <"s_lshr_b64",
  [(set SReg_64:$sdst, (UniformBinFrag<csrl_64> (i64 SSrc_b64:$src0), (i32 SSrc_b32:$src1)))]
>;
def S_ASHR_I32 : SOP2_32 <"s_ashr_i32",
  [(set SReg_32:$sdst, (UniformBinFrag<csra_32> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]
>;
def S_ASHR_I64 : SOP2_64_32 <"s_ashr_i64",
  [(set SReg_64:$sdst, (UniformBinFrag<csra_64> (i64 SSrc_b64:$src0), (i32 SSrc_b32:$src1)))]
>;
} // End Defs = [SCC]

let isReMaterializable = 1 in {
def S_BFM_B32 : SOP2_32 <"s_bfm_b32",
  [(set i32:$sdst, (UniformBinFrag<AMDGPUbfm> i32:$src0, i32:$src1))]>;
def S_BFM_B64 : SOP2_64_32_32 <"s_bfm_b64">;

def S_MUL_I32 : SOP2_32 <"s_mul_i32",
  [(set i32:$sdst, (UniformBinFrag<mul> i32:$src0, i32:$src1))]> {
  let isCommutable = 1;
}
} // End isReMaterializable = 1
} // End AddedComplexity = 1

let Defs = [SCC] in {
def S_BFE_U32 : SOP2_32 <"s_bfe_u32">;
def S_BFE_I32 : SOP2_32 <"s_bfe_i32">;
def S_BFE_U64 : SOP2_64_32 <"s_bfe_u64">;
def S_BFE_I64 : SOP2_64_32 <"s_bfe_i64">;
} // End Defs = [SCC]

def S_CBRANCH_G_FORK : SOP2_Pseudo <
  "s_cbranch_g_fork", (outs),
  (ins SCSrc_b64:$src0, SCSrc_b64:$src1),
  "$src0, $src1"
> {
  let has_sdst = 0;
  let SubtargetPredicate = isGFX6GFX7GFX8GFX9;
}

let Defs = [SCC] in {
def S_ABSDIFF_I32 : SOP2_32 <"s_absdiff_i32">;
} // End Defs = [SCC]

let SubtargetPredicate = isGFX8GFX9 in {
  def S_RFE_RESTORE_B64 : SOP2_Pseudo <
    "s_rfe_restore_b64", (outs),
    (ins SSrc_b64:$src0, SSrc_b32:$src1),
    "$src0, $src1"
  > {
    let hasSideEffects = 1;
    let has_sdst = 0;
  }
}

let SubtargetPredicate = isGFX9Plus in {
  let isReMaterializable = 1 in {
    def S_PACK_LL_B32_B16 : SOP2_32<"s_pack_ll_b32_b16">;
    def S_PACK_LH_B32_B16 : SOP2_32<"s_pack_lh_b32_b16">;
    def S_PACK_HH_B32_B16 : SOP2_32<"s_pack_hh_b32_b16">;
  } // End isReMaterializable = 1

  let Defs = [SCC] in {
    def S_LSHL1_ADD_U32 : SOP2_32<"s_lshl1_add_u32",
      [(set i32:$sdst, (shl1_add SSrc_b32:$src0, SSrc_b32:$src1))]
    >;
    def S_LSHL2_ADD_U32 : SOP2_32<"s_lshl2_add_u32",
      [(set i32:$sdst, (shl2_add SSrc_b32:$src0, SSrc_b32:$src1))]
    >;
    def S_LSHL3_ADD_U32 : SOP2_32<"s_lshl3_add_u32",
      [(set i32:$sdst, (shl3_add SSrc_b32:$src0, SSrc_b32:$src1))]
    >;
    def S_LSHL4_ADD_U32 : SOP2_32<"s_lshl4_add_u32",
      [(set i32:$sdst, (shl4_add SSrc_b32:$src0, SSrc_b32:$src1))]
    >;
  } // End Defs = [SCC]

  let isCommutable = 1, isReMaterializable = 1 in {
    def S_MUL_HI_U32 : SOP2_32<"s_mul_hi_u32",
      [(set i32:$sdst, (UniformBinFrag<mulhu> SSrc_b32:$src0, SSrc_b32:$src1))]>;
    def S_MUL_HI_I32 : SOP2_32<"s_mul_hi_i32",
      [(set i32:$sdst, (UniformBinFrag<mulhs> SSrc_b32:$src0, SSrc_b32:$src1))]>;
  } // End isCommutable = 1, isReMaterializable = 1
} // End SubtargetPredicate = isGFX9Plus

let SubtargetPredicate = isGFX11Plus in {
  def S_PACK_HL_B32_B16 : SOP2_32<"s_pack_hl_b32_b16">;
} // End SubtargetPredicate = isGFX11Plus

//===----------------------------------------------------------------------===//
// SOPK Instructions
//===----------------------------------------------------------------------===//

class SOPK_Pseudo <string opName, dag outs, dag ins,
                   string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, " " # asmOps, pattern> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPK = 1;
  let FixedSize = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;

  let has_sdst = 1;
}

class SOPK_Real<SOPK_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList,
          ps.Mnemonic # ps.AsmOperands> {
  let SALU = 1;
  let SOPK = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let UseNamedOperandTable = 1;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let DisableEncoding    = ps.DisableEncoding;
  let Constraints        = ps.Constraints;
  let SchedRW            = ps.SchedRW;
  let mayLoad            = ps.mayLoad;
  let mayStore           = ps.mayStore;
  let isBranch           = ps.isBranch;
  let isCall             = ps.isCall;

  // encoding
  bits<7>  sdst;
  bits<16> simm16;
  bits<32> imm;
}

class SOPK_Real32<bits<5> op, SOPK_Pseudo ps> :
  SOPK_Real <ps>,
  Enc32 {
  let Inst{15-0}  = simm16;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
}

class SOPK_Real64<bits<5> op, SOPK_Pseudo ps> :
  SOPK_Real<ps>,
  Enc64 {
  let Inst{15-0}  = simm16;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
  let Inst{63-32} = imm;
}

class SOPKInstTable <bit is_sopk, string cmpOp = ""> {
  bit IsSOPK = is_sopk;
  string BaseCmpOp = cmpOp;
}

class SOPK_32 <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs SReg_32:$sdst),
  (ins s16imm:$simm16),
  "$sdst, $simm16",
  pattern>;

class SOPK_32_BR <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs),
  (ins SOPPBrTarget:$simm16, SReg_32:$sdst),
  "$sdst, $simm16",
  pattern> {
  let Defs = [EXEC];
  let Uses = [EXEC];
  let isBranch = 1;
  let isTerminator = 1;
  let SchedRW = [WriteBranch];
}

class SOPK_SCC <string opName, string base_op, bit isSignExt> : SOPK_Pseudo <
  opName,
  (outs),
  !if(isSignExt,
      (ins SReg_32:$sdst, s16imm:$simm16),
      (ins SReg_32:$sdst, u16imm:$simm16)),
  "$sdst, $simm16">,
  SOPKInstTable<1, base_op>{
  let Defs = [SCC];
}

class SOPK_32TIE <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs SReg_32:$sdst),
  (ins SReg_32:$src0, s16imm:$simm16),
  "$sdst, $simm16",
  pattern
>;

let isReMaterializable = 1, isMoveImm = 1 in {
def S_MOVK_I32 : SOPK_32 <"s_movk_i32">;
} // End isReMaterializable = 1
let Uses = [SCC] in {
def S_CMOVK_I32 : SOPK_32 <"s_cmovk_i32">;
}

let isCompare = 1 in {

// This instruction is disabled for now until we can figure out how to teach
// the instruction selector to correctly use the  S_CMP* vs V_CMP*
// instructions.
//
// When this instruction is enabled the code generator sometimes produces this
// invalid sequence:
//
// SCC = S_CMPK_EQ_I32 SGPR0, imm
// VCC = COPY SCC
// VGPR0 = V_CNDMASK VCC, VGPR0, VGPR1
//
// def S_CMPK_EQ_I32 : SOPK_SCC <"s_cmpk_eq_i32",
//   [(set i1:$dst, (setcc i32:$src0, imm:$src1, SETEQ))]
// >;

def S_CMPK_EQ_I32 : SOPK_SCC <"s_cmpk_eq_i32", "s_cmp_eq_i32", 1>;
def S_CMPK_LG_I32 : SOPK_SCC <"s_cmpk_lg_i32", "s_cmp_lg_i32", 1>;
def S_CMPK_GT_I32 : SOPK_SCC <"s_cmpk_gt_i32", "s_cmp_gt_i32", 1>;
def S_CMPK_GE_I32 : SOPK_SCC <"s_cmpk_ge_i32", "s_cmp_ge_i32", 1>;
def S_CMPK_LT_I32 : SOPK_SCC <"s_cmpk_lt_i32", "s_cmp_lt_i32", 1>;
def S_CMPK_LE_I32 : SOPK_SCC <"s_cmpk_le_i32", "s_cmp_le_i32", 1>;

let SOPKZext = 1 in {
def S_CMPK_EQ_U32 : SOPK_SCC <"s_cmpk_eq_u32", "s_cmp_eq_u32", 0>;
def S_CMPK_LG_U32 : SOPK_SCC <"s_cmpk_lg_u32", "s_cmp_lg_u32", 0>;
def S_CMPK_GT_U32 : SOPK_SCC <"s_cmpk_gt_u32", "s_cmp_gt_u32", 0>;
def S_CMPK_GE_U32 : SOPK_SCC <"s_cmpk_ge_u32", "s_cmp_ge_u32", 0>;
def S_CMPK_LT_U32 : SOPK_SCC <"s_cmpk_lt_u32", "s_cmp_lt_u32", 0>;
def S_CMPK_LE_U32 : SOPK_SCC <"s_cmpk_le_u32", "s_cmp_le_u32", 0>;
} // End SOPKZext = 1
} // End isCompare = 1

let isCommutable = 1, DisableEncoding = "$src0",
    Constraints = "$sdst = $src0" in {
  let Defs = [SCC] in
    def S_ADDK_I32 : SOPK_32TIE <"s_addk_i32">;
  def S_MULK_I32 : SOPK_32TIE <"s_mulk_i32">;
}

let SubtargetPredicate = isGFX6GFX7GFX8GFX9 in
def S_CBRANCH_I_FORK : SOPK_Pseudo <
  "s_cbranch_i_fork",
  (outs), (ins SReg_64:$sdst, SOPPBrTarget:$simm16),
  "$sdst, $simm16"
>;

// This is hasSideEffects to allow its use in readcyclecounter selection.
// FIXME: Need to truncate immediate to 16-bits.
def S_GETREG_B32 : SOPK_Pseudo <
  "s_getreg_b32",
  (outs SReg_32:$sdst), (ins hwreg:$simm16),
  "$sdst, $simm16",
  [(set i32:$sdst, (int_amdgcn_s_getreg (i32 timm:$simm16)))]> {
  let SOPKZext = 1;
  let hasSideEffects = 1;
}

let Defs = [MODE], Uses = [MODE] in {

// FIXME: Need to truncate immediate to 16-bits.
class S_SETREG_B32_Pseudo <list<dag> pattern=[]> : SOPK_Pseudo <
  "s_setreg_b32",
  (outs), (ins SReg_32:$sdst, hwreg:$simm16),
  "$simm16, $sdst",
  pattern>;

def S_SETREG_B32 : S_SETREG_B32_Pseudo <
  [(int_amdgcn_s_setreg (i32 timm:$simm16), i32:$sdst)]> {
  // Use custom inserter to optimize some cases to
  // S_DENORM_MODE/S_ROUND_MODE/S_SETREG_B32_mode.
  let usesCustomInserter = 1;
  let hasSideEffects = 1;
}

// Variant of SETREG that is guaranteed to only touch FP bits in the MODE
// register, so doesn't have unmodeled side effects.
def S_SETREG_B32_mode : S_SETREG_B32_Pseudo {
  let hasSideEffects = 0;
}

// FIXME: Not on SI?
//def S_GETREG_REGRD_B32 : SOPK_32 <sopk<0x14, 0x13>, "s_getreg_regrd_b32">;

class S_SETREG_IMM32_B32_Pseudo : SOPK_Pseudo <
  "s_setreg_imm32_b32",
  (outs), (ins i32imm:$imm, hwreg:$simm16),
  "$simm16, $imm"> {
  let Size = 8; // Unlike every other SOPK instruction.
  let has_sdst = 0;
}

def S_SETREG_IMM32_B32 : S_SETREG_IMM32_B32_Pseudo {
  let hasSideEffects = 1;
}

// Variant of SETREG_IMM32 that is guaranteed to only touch FP bits in the MODE
// register, so doesn't have unmodeled side effects.
def S_SETREG_IMM32_B32_mode : S_SETREG_IMM32_B32_Pseudo {
  let hasSideEffects = 0;
}

} // End Defs = [MODE], Uses = [MODE]

class SOPK_WAITCNT<string opName, list<dag> pat=[]> :
    SOPK_Pseudo<
        opName,
        (outs),
        (ins SReg_32:$sdst, s16imm:$simm16),
        "$sdst, $simm16",
        pat> {
  let hasSideEffects = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let has_sdst = 1; // First source takes place of sdst in encoding
}

let SubtargetPredicate = isGFX9Plus in {
  def S_CALL_B64 : SOPK_Pseudo<
      "s_call_b64",
      (outs SReg_64:$sdst),
      (ins SOPPBrTarget:$simm16),
      "$sdst, $simm16"> {
    let isCall = 1;
  }
} // End SubtargetPredicate = isGFX9Plus

let SubtargetPredicate = isGFX10Plus in {
  def S_VERSION : SOPK_Pseudo<
      "s_version",
      (outs),
      (ins s16imm:$simm16),
      "$simm16"> {
    let has_sdst = 0;
  }

  def S_SUBVECTOR_LOOP_BEGIN : SOPK_32_BR<"s_subvector_loop_begin">;
  def S_SUBVECTOR_LOOP_END   : SOPK_32_BR<"s_subvector_loop_end">;

  def S_WAITCNT_VSCNT   : SOPK_WAITCNT<"s_waitcnt_vscnt">;
  def S_WAITCNT_VMCNT   : SOPK_WAITCNT<"s_waitcnt_vmcnt">;
  def S_WAITCNT_EXPCNT  : SOPK_WAITCNT<"s_waitcnt_expcnt">;
  def S_WAITCNT_LGKMCNT : SOPK_WAITCNT<"s_waitcnt_lgkmcnt">;
} // End SubtargetPredicate = isGFX10Plus

//===----------------------------------------------------------------------===//
// SOPC Instructions
//===----------------------------------------------------------------------===//

class SOPC_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, " " # asmOps, pattern> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPC = 1;
  let Defs = [SCC];
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
}

class SOPC_Real<bits<7> op, SOPC_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList,
          ps.Mnemonic # ps.AsmOperands>,
  Enc32 {
  let SALU = 1;
  let SOPC = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let OtherPredicates      = ps.OtherPredicates;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let TSFlags              = ps.TSFlags;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;

  // encoding
  bits<8> src0;
  bits<8> src1;

  let Inst{7-0} = src0;
  let Inst{15-8} = src1;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17e;
}

class SOPC_Base <RegisterOperand rc0, RegisterOperand rc1,
                 string opName, list<dag> pattern = []> : SOPC_Pseudo <
  opName, (outs), (ins rc0:$src0, rc1:$src1),
  "$src0, $src1", pattern > {
}

class SOPC_Helper <RegisterOperand rc, ValueType vt,
                    string opName, SDPatternOperator cond> : SOPC_Base <
  rc, rc, opName,
  [(set SCC, (UniformTernaryFrag<setcc> vt:$src0, vt:$src1, cond))] > {
}

class SOPC_CMP_32<string opName,
                  SDPatternOperator cond = COND_NULL, string revOp = opName>
  : SOPC_Helper<SSrc_b32, i32, opName, cond>,
    Commutable_REV<revOp, !eq(revOp, opName)>,
    SOPKInstTable<0, opName> {
  let isCompare = 1;
  let isCommutable = 1;
}

class SOPC_CMP_64<string opName,
                  SDPatternOperator cond = COND_NULL, string revOp = opName>
  : SOPC_Helper<SSrc_b64, i64, opName, cond>,
    Commutable_REV<revOp, !eq(revOp, opName)> {
  let isCompare = 1;
  let isCommutable = 1;
}

class SOPC_32<string opName, list<dag> pattern = []>
  : SOPC_Base<SSrc_b32, SSrc_b32, opName, pattern>;

class SOPC_64_32<string opName, list<dag> pattern = []>
  : SOPC_Base<SSrc_b64, SSrc_b32, opName, pattern>;

def S_CMP_EQ_I32 : SOPC_CMP_32 <"s_cmp_eq_i32">;
def S_CMP_LG_I32 : SOPC_CMP_32 <"s_cmp_lg_i32">;
def S_CMP_GT_I32 : SOPC_CMP_32 <"s_cmp_gt_i32", COND_SGT>;
def S_CMP_GE_I32 : SOPC_CMP_32 <"s_cmp_ge_i32", COND_SGE>;
def S_CMP_LT_I32 : SOPC_CMP_32 <"s_cmp_lt_i32", COND_SLT, "s_cmp_gt_i32">;
def S_CMP_LE_I32 : SOPC_CMP_32 <"s_cmp_le_i32", COND_SLE, "s_cmp_ge_i32">;
def S_CMP_EQ_U32 : SOPC_CMP_32 <"s_cmp_eq_u32", COND_EQ>;
def S_CMP_LG_U32 : SOPC_CMP_32 <"s_cmp_lg_u32", COND_NE>;
def S_CMP_GT_U32 : SOPC_CMP_32 <"s_cmp_gt_u32", COND_UGT>;
def S_CMP_GE_U32 : SOPC_CMP_32 <"s_cmp_ge_u32", COND_UGE>;
def S_CMP_LT_U32 : SOPC_CMP_32 <"s_cmp_lt_u32", COND_ULT, "s_cmp_gt_u32">;
def S_CMP_LE_U32 : SOPC_CMP_32 <"s_cmp_le_u32", COND_ULE, "s_cmp_ge_u32">;

def S_BITCMP0_B32 : SOPC_32 <"s_bitcmp0_b32">;
def S_BITCMP1_B32 : SOPC_32 <"s_bitcmp1_b32">;
def S_BITCMP0_B64 : SOPC_64_32 <"s_bitcmp0_b64">;
def S_BITCMP1_B64 : SOPC_64_32 <"s_bitcmp1_b64">;
let SubtargetPredicate = isGFX6GFX7GFX8GFX9 in
def S_SETVSKIP : SOPC_32 <"s_setvskip">;

let SubtargetPredicate = isGFX8Plus in {
def S_CMP_EQ_U64 : SOPC_CMP_64 <"s_cmp_eq_u64", COND_EQ>;
def S_CMP_LG_U64 : SOPC_CMP_64 <"s_cmp_lg_u64", COND_NE>;
} // End SubtargetPredicate = isGFX8Plus

let SubtargetPredicate = HasVGPRIndexMode in {
// Setting the GPR index mode is really writing the fields in the mode
// register. We don't want to add mode register uses to every
// instruction, and it's too complicated to deal with anyway. This is
// modeled just as a side effect.
def S_SET_GPR_IDX_ON : SOPC_Pseudo <
  "s_set_gpr_idx_on" ,
  (outs),
  (ins SSrc_b32:$src0, GPRIdxMode:$src1),
  "$src0, $src1"> {
  let Defs = [M0, MODE]; // No scc def
  let Uses = [M0, MODE]; // Other bits of mode, m0 unmodified.
  let hasSideEffects = 1; // Sets mode.gpr_idx_en
  let FixedSize = 1;
}
}

//===----------------------------------------------------------------------===//
// SOPP Instructions
//===----------------------------------------------------------------------===//

class SOPP_Pseudo<string opName, dag ins,
                  string asmOps = "", list<dag> pattern=[],
                  string sep = !if(!empty(asmOps), "", " "),
                  string keyName = opName> :
  SOP_Pseudo<opName, (outs), ins, sep # asmOps, pattern> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPP = 1;
  let FixedSize = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
  bits <16> simm16;
  bits <1> fixed_imm = 0;
  string KeyName = keyName;
}

class SOPPRelaxTable <bit isRelaxed, string keyName, string gfxip> {
  bit IsRelaxed = isRelaxed;
  string KeyName = keyName # gfxip;
}

class SOPP_Real<SOPP_Pseudo ps, string real_name = ps.Mnemonic> :
  InstSI <ps.OutOperandList, ps.InOperandList,
          real_name # ps.AsmOperands> {
  let SALU = 1;
  let SOPP = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let OtherPredicates      = ps.OtherPredicates;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let TSFlags              = ps.TSFlags;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  bits <16> simm16;
}

class SOPP_Real_32 <bits<7> op, SOPP_Pseudo ps, string real_name = ps.Mnemonic> : SOPP_Real<ps, real_name>,
Enc32 {
  let Inst{15-0} = !if(ps.fixed_imm, ps.simm16, simm16);
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f;
}

class SOPP_Real_64 <bits<7> op, SOPP_Pseudo ps, string real_name = ps.Mnemonic> : SOPP_Real<ps, real_name>,
Enc64 {
  // encoding
  let Inst{15-0} = !if(ps.fixed_imm, ps.simm16, simm16);
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f;
  //effectively a nop
  let Inst{47-32} = 0x0;
  let Inst{54-48} = 0x0;
  let Inst{63-55} = 0x17f;
}

multiclass SOPP_With_Relaxation <string opName, dag ins,
                  string asmOps, list<dag> pattern=[]> {
  def "" : SOPP_Pseudo <opName, ins, asmOps, pattern>;
  def _pad_s_nop : SOPP_Pseudo <opName # "_pad_s_nop", ins, asmOps, pattern, " ", opName>;
}

def S_NOP : SOPP_Pseudo<"s_nop" , (ins i16imm:$simm16), "$simm16",
  [(int_amdgcn_s_nop timm:$simm16)]> {
  let hasSideEffects = 1;
}

let isTerminator = 1 in {
def S_ENDPGM : SOPP_Pseudo<"s_endpgm", (ins Endpgm:$simm16), "$simm16", [], ""> {
  let isBarrier = 1;
  let isReturn = 1;
  let hasSideEffects = 1;
}

def S_ENDPGM_SAVED : SOPP_Pseudo<"s_endpgm_saved", (ins)> {
  let SubtargetPredicate = isGFX8Plus;
  let simm16 = 0;
  let fixed_imm = 1;
  let isBarrier = 1;
  let isReturn = 1;
}

let SubtargetPredicate = isGFX9GFX10 in {
  let isBarrier = 1, isReturn = 1, simm16 = 0, fixed_imm = 1 in {
    def S_ENDPGM_ORDERED_PS_DONE :
      SOPP_Pseudo<"s_endpgm_ordered_ps_done", (ins)>;
  } // End isBarrier = 1, isReturn = 1, simm16 = 0, fixed_imm = 1
} // End SubtargetPredicate = isGFX9GFX10

let SubtargetPredicate = isGFX10Plus in {
  let isBarrier = 1, isReturn = 1, simm16 = 0, fixed_imm = 1 in {
    def S_CODE_END :
      SOPP_Pseudo<"s_code_end", (ins)>;
  } // End isBarrier = 1, isReturn = 1, simm16 = 0, fixed_imm = 1
} // End SubtargetPredicate = isGFX10Plus

let isBranch = 1, SchedRW = [WriteBranch] in {
let isBarrier = 1 in {
defm S_BRANCH : SOPP_With_Relaxation<
  "s_branch" , (ins SOPPBrTarget:$simm16), "$simm16",
  [(br bb:$simm16)]>;
}

let Uses = [SCC] in {
defm S_CBRANCH_SCC0 : SOPP_With_Relaxation<
  "s_cbranch_scc0" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
defm S_CBRANCH_SCC1 : SOPP_With_Relaxation <
  "s_cbranch_scc1" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
} // End Uses = [SCC]

let Uses = [VCC] in {
defm S_CBRANCH_VCCZ : SOPP_With_Relaxation <
  "s_cbranch_vccz" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
defm S_CBRANCH_VCCNZ : SOPP_With_Relaxation <
  "s_cbranch_vccnz" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
} // End Uses = [VCC]

let Uses = [EXEC] in {
defm S_CBRANCH_EXECZ : SOPP_With_Relaxation <
  "s_cbranch_execz" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
defm S_CBRANCH_EXECNZ : SOPP_With_Relaxation <
  "s_cbranch_execnz" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;
} // End Uses = [EXEC]

defm S_CBRANCH_CDBGSYS : SOPP_With_Relaxation <
  "s_cbranch_cdbgsys" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;

defm S_CBRANCH_CDBGSYS_AND_USER : SOPP_With_Relaxation <
  "s_cbranch_cdbgsys_and_user" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;

defm S_CBRANCH_CDBGSYS_OR_USER : SOPP_With_Relaxation <
  "s_cbranch_cdbgsys_or_user" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;

defm S_CBRANCH_CDBGUSER : SOPP_With_Relaxation <
  "s_cbranch_cdbguser" , (ins SOPPBrTarget:$simm16),
  "$simm16"
>;

} // End isBranch = 1
} // End isTerminator = 1

let hasSideEffects = 1 in {
def S_BARRIER : SOPP_Pseudo <"s_barrier", (ins), "",
  [(int_amdgcn_s_barrier)]> {
  let SchedRW = [WriteBarrier];
  let simm16 = 0;
  let fixed_imm = 1;
  let isConvergent = 1;
}

def S_WAKEUP : SOPP_Pseudo <"s_wakeup", (ins) > {
  let SubtargetPredicate = isGFX8Plus;
  let simm16 = 0;
  let fixed_imm = 1;
  let mayLoad = 1;
  let mayStore = 1;
}

let hasSideEffects = 1 in
def S_WAITCNT : SOPP_Pseudo <"s_waitcnt" , (ins SWaitCnt:$simm16), "$simm16",
    [(int_amdgcn_s_waitcnt timm:$simm16)]>;
def S_SETHALT : SOPP_Pseudo <"s_sethalt" , (ins i32imm:$simm16), "$simm16",
    [(int_amdgcn_s_sethalt timm:$simm16)]>;
def S_SETKILL : SOPP_Pseudo <"s_setkill" , (ins i16imm:$simm16), "$simm16">;

// On SI the documentation says sleep for approximately 64 * low 2
// bits, consistent with the reported maximum of 448. On VI the
// maximum reported is 960 cycles, so 960 / 64 = 15 max, so is the
// maximum really 15 on VI?
def S_SLEEP : SOPP_Pseudo <"s_sleep", (ins i32imm:$simm16),
  "$simm16", [(int_amdgcn_s_sleep timm:$simm16)]> {
  let hasSideEffects = 1;
}

def S_SETPRIO : SOPP_Pseudo <"s_setprio", (ins i16imm:$simm16), "$simm16",
  [(int_amdgcn_s_setprio timm:$simm16)]> {
  let hasSideEffects = 1;
}

let Uses = [EXEC, M0] in {
def S_SENDMSG : SOPP_Pseudo <"s_sendmsg" , (ins SendMsg:$simm16), "$simm16",
  [(int_amdgcn_s_sendmsg (i32 timm:$simm16), M0)]> {
  let hasSideEffects = 1;
}

def S_SENDMSGHALT : SOPP_Pseudo <"s_sendmsghalt" , (ins SendMsg:$simm16), "$simm16",
  [(int_amdgcn_s_sendmsghalt (i32 timm:$simm16), M0)]> {
  let hasSideEffects = 1;
}

} // End Uses = [EXEC, M0]

def S_TRAP : SOPP_Pseudo <"s_trap" , (ins i16imm:$simm16), "$simm16"> {
  let isTrap = 1;
}

def S_ICACHE_INV : SOPP_Pseudo <"s_icache_inv", (ins)> {
  let simm16 = 0;
  let fixed_imm = 1;
}
def S_INCPERFLEVEL : SOPP_Pseudo <"s_incperflevel", (ins i32imm:$simm16), "$simm16",
  [(int_amdgcn_s_incperflevel timm:$simm16)]> {
  let hasSideEffects = 1;
}
def S_DECPERFLEVEL : SOPP_Pseudo <"s_decperflevel", (ins i32imm:$simm16), "$simm16",
  [(int_amdgcn_s_decperflevel timm:$simm16)]> {
  let hasSideEffects = 1;
}
def S_TTRACEDATA : SOPP_Pseudo <"s_ttracedata", (ins)> {
  let simm16 = 0;
  let fixed_imm = 1;
}

let SubtargetPredicate = HasVGPRIndexMode in {
def S_SET_GPR_IDX_OFF : SOPP_Pseudo<"s_set_gpr_idx_off", (ins) > {
  let simm16 = 0;
  let fixed_imm = 1;
  let Defs = [MODE];
  let Uses = [MODE];
}
}
} // End hasSideEffects

let SubtargetPredicate = HasVGPRIndexMode in {
def S_SET_GPR_IDX_MODE : SOPP_Pseudo<"s_set_gpr_idx_mode", (ins GPRIdxMode:$simm16),
  "$simm16"> {
  let Defs = [M0, MODE];
  let Uses = [MODE];
}
}

let SubtargetPredicate = isGFX10Plus in {
  def S_INST_PREFETCH :
    SOPP_Pseudo<"s_inst_prefetch", (ins s16imm:$simm16), "$simm16">;
  def S_CLAUSE :
    SOPP_Pseudo<"s_clause", (ins s16imm:$simm16), "$simm16">;
  def S_WAIT_IDLE :
    SOPP_Pseudo <"s_wait_idle", (ins)> {
      let simm16 = 0;
      let fixed_imm = 1;
    }
  def S_WAITCNT_DEPCTR :
    SOPP_Pseudo <"s_waitcnt_depctr" , (ins DepCtr:$simm16), "$simm16">;

  let hasSideEffects = 0, Uses = [MODE], Defs = [MODE] in {
    def S_ROUND_MODE :
      SOPP_Pseudo<"s_round_mode", (ins s16imm:$simm16), "$simm16">;
    def S_DENORM_MODE :
      SOPP_Pseudo<"s_denorm_mode", (ins i32imm:$simm16), "$simm16",
      [(SIdenorm_mode (i32 timm:$simm16))]>;
  }

  def S_TTRACEDATA_IMM :
    SOPP_Pseudo<"s_ttracedata_imm", (ins s16imm:$simm16), "$simm16">;
} // End SubtargetPredicate = isGFX10Plus

let SubtargetPredicate = isGFX11Plus in {
  def S_WAIT_EVENT : SOPP_Pseudo<"s_wait_event", (ins s16imm:$simm16),
                                 "$simm16"> {
                                   let hasSideEffects = 1;
                                 }
  def S_DELAY_ALU : SOPP_Pseudo<"s_delay_alu", (ins SDelayALU:$simm16),
                                "$simm16">;
} // End SubtargetPredicate = isGFX11Plus

//===----------------------------------------------------------------------===//
// SOP1 Patterns
//===----------------------------------------------------------------------===//

def : GCNPat <
  (AMDGPUendpgm),
    (S_ENDPGM (i16 0))
>;

def : GCNPat <
  (int_amdgcn_endpgm),
    (S_ENDPGM (i16 0))
>;

def : GCNPat <
  (i64 (UniformUnaryFrag<ctpop> i64:$src)),
    (i64 (REG_SEQUENCE SReg_64,
     (i32 (COPY_TO_REGCLASS (S_BCNT1_I32_B64 $src), SReg_32)), sub0,
     (S_MOV_B32 (i32 0)), sub1))
>;

def : GCNPat <
  (i32 (UniformBinFrag<smax> i32:$x, (i32 (ineg i32:$x)))),
  (S_ABS_I32 SReg_32:$x)
>;

def : GCNPat <
  (i16 imm:$imm),
  (S_MOV_B32 imm:$imm)
>;

// Same as a 32-bit inreg
def : GCNPat<
  (i32 (UniformUnaryFrag<sext> i16:$src)),
  (S_SEXT_I32_I16 $src)
>;

def : GCNPat <
  (int_amdgcn_s_wait_event_export_ready),
    (S_WAIT_EVENT (i16 0))
>;

//===----------------------------------------------------------------------===//
// SOP2 Patterns
//===----------------------------------------------------------------------===//

// V_ADD_I32_e32/S_ADD_U32 produces carry in VCC/SCC. For the vector
// case, the sgpr-copies pass will fix this to use the vector version.
def : GCNPat <
  (i32 (addc i32:$src0, i32:$src1)),
  (S_ADD_U32 $src0, $src1)
>;

// FIXME: We need to use COPY_TO_REGCLASS to work-around the fact that
// REG_SEQUENCE patterns don't support instructions with multiple
// outputs.
def : GCNPat<
  (i64 (UniformUnaryFrag<zext> i16:$src)),
    (REG_SEQUENCE SReg_64,
      (i32 (COPY_TO_REGCLASS (S_AND_B32 $src, (S_MOV_B32 (i32 0xffff))), SGPR_32)), sub0,
      (S_MOV_B32 (i32 0)), sub1)
>;

def : GCNPat <
  (i64 (UniformUnaryFrag<sext> i16:$src)),
    (REG_SEQUENCE SReg_64, (i32 (S_SEXT_I32_I16 $src)), sub0,
    (i32 (COPY_TO_REGCLASS (S_ASHR_I32 (i32 (S_SEXT_I32_I16 $src)), (S_MOV_B32 (i32 31))), SGPR_32)), sub1)
>;

def : GCNPat<
  (i32 (UniformUnaryFrag<zext> i16:$src)),
  (S_AND_B32 (S_MOV_B32 (i32 0xffff)), $src)
>;

// FIXME: ValueType should have isVector field
class ScalarNot2Pat<Instruction inst, SDPatternOperator op, ValueType vt,
                    bit isVector = 1> : GCNPat<
  (UniformBinFrag<op> vt:$src0, (UniformUnaryFrag<!if(isVector, vnot, not)> vt:$src1)),
  (inst getSOPSrcForVT<vt>.ret:$src0, getSOPSrcForVT<vt>.ret:$src1)
>;

// Match these for some more types
// TODO: i1
def : ScalarNot2Pat<S_ANDN2_B32, and, i16, 0>;
def : ScalarNot2Pat<S_ANDN2_B32, and, v2i16>;
def : ScalarNot2Pat<S_ANDN2_B64, and, v4i16>;
def : ScalarNot2Pat<S_ANDN2_B64, and, v2i32>;

def : ScalarNot2Pat<S_ORN2_B32, or, i16, 0>;
def : ScalarNot2Pat<S_ORN2_B32, or, v2i16>;
def : ScalarNot2Pat<S_ORN2_B64, or, v4i16>;
def : ScalarNot2Pat<S_ORN2_B64, or, v2i32>;

//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

class Select_gfx11<string opName> : SIMCInstr<opName, SIEncodingFamily.GFX11> {
  Predicate AssemblerPredicate = isGFX11Only;
  string DecoderNamespace      = "GFX11";
}

class Select_gfx10<string opName> : SIMCInstr<opName, SIEncodingFamily.GFX10> {
  Predicate AssemblerPredicate = isGFX10Only;
  string DecoderNamespace      = "GFX10";
}

class Select_vi<string opName> : SIMCInstr<opName, SIEncodingFamily.VI> {
  Predicate AssemblerPredicate = isGFX8GFX9;
  string DecoderNamespace = "GFX8";
}

class Select_gfx6_gfx7<string opName> : SIMCInstr<opName, SIEncodingFamily.SI> {
  Predicate AssemblerPredicate = isGFX6GFX7;
  string DecoderNamespace      = "GFX6GFX7";
}

//===----------------------------------------------------------------------===//
//  GFX11.
//===----------------------------------------------------------------------===//

multiclass SOP1_Real_gfx11<bits<8> op> {
  def _gfx11 : SOP1_Real<op, !cast<SOP1_Pseudo>(NAME)>,
               Select_gfx11<!cast<SOP1_Pseudo>(NAME).Mnemonic>;
}

multiclass SOP1_Real_Renamed_gfx11<bits<8> op, SOP1_Pseudo backing_pseudo, string real_name> {
  def _gfx11 : SOP1_Real<op, backing_pseudo, real_name>,
               Select_gfx11<backing_pseudo.Mnemonic>,
               MnemonicAlias<backing_pseudo.Mnemonic, real_name>, Requires<[isGFX11Plus]>;
}

defm S_MOV_B32               : SOP1_Real_gfx11<0x000>;
defm S_MOV_B64               : SOP1_Real_gfx11<0x001>;
defm S_CMOV_B32              : SOP1_Real_gfx11<0x002>;
defm S_CMOV_B64              : SOP1_Real_gfx11<0x003>;
defm S_BREV_B32              : SOP1_Real_gfx11<0x004>;
defm S_BREV_B64              : SOP1_Real_gfx11<0x005>;
defm S_CTZ_I32_B32           : SOP1_Real_Renamed_gfx11<0x008, S_FF1_I32_B32, "s_ctz_i32_b32">;
defm S_CTZ_I32_B64           : SOP1_Real_Renamed_gfx11<0x009, S_FF1_I32_B64, "s_ctz_i32_b64">;
defm S_CLZ_I32_U32           : SOP1_Real_Renamed_gfx11<0x00a, S_FLBIT_I32_B32, "s_clz_i32_u32">;
defm S_CLZ_I32_U64           : SOP1_Real_Renamed_gfx11<0x00b, S_FLBIT_I32_B64, "s_clz_i32_u64">;
defm S_CLS_I32               : SOP1_Real_Renamed_gfx11<0x00c, S_FLBIT_I32, "s_cls_i32">;
defm S_CLS_I32_I64           : SOP1_Real_Renamed_gfx11<0x00d, S_FLBIT_I32_I64, "s_cls_i32_i64">;
defm S_SEXT_I32_I8           : SOP1_Real_gfx11<0x00e>;
defm S_SEXT_I32_I16          : SOP1_Real_gfx11<0x00f>;
defm S_BITSET0_B32           : SOP1_Real_gfx11<0x010>;
defm S_BITSET0_B64           : SOP1_Real_gfx11<0x011>;
defm S_BITSET1_B32           : SOP1_Real_gfx11<0x012>;
defm S_BITSET1_B64           : SOP1_Real_gfx11<0x013>;
defm S_BITREPLICATE_B64_B32  : SOP1_Real_gfx11<0x014>;
defm S_ABS_I32               : SOP1_Real_gfx11<0x015>;
defm S_BCNT0_I32_B32         : SOP1_Real_gfx11<0x016>;
defm S_BCNT0_I32_B64         : SOP1_Real_gfx11<0x017>;
defm S_BCNT1_I32_B32         : SOP1_Real_gfx11<0x018>;
defm S_BCNT1_I32_B64         : SOP1_Real_gfx11<0x019>;
defm S_QUADMASK_B32          : SOP1_Real_gfx11<0x01a>;
defm S_QUADMASK_B64          : SOP1_Real_gfx11<0x01b>;
defm S_WQM_B32               : SOP1_Real_gfx11<0x01c>;
defm S_WQM_B64               : SOP1_Real_gfx11<0x01d>;
defm S_NOT_B32               : SOP1_Real_gfx11<0x01e>;
defm S_NOT_B64               : SOP1_Real_gfx11<0x01f>;
defm S_AND_SAVEEXEC_B32      : SOP1_Real_gfx11<0x020>;
defm S_AND_SAVEEXEC_B64      : SOP1_Real_gfx11<0x021>;
defm S_OR_SAVEEXEC_B32       : SOP1_Real_gfx11<0x022>;
defm S_OR_SAVEEXEC_B64       : SOP1_Real_gfx11<0x023>;
defm S_XOR_SAVEEXEC_B32      : SOP1_Real_gfx11<0x024>;
defm S_XOR_SAVEEXEC_B64      : SOP1_Real_gfx11<0x025>;
defm S_NAND_SAVEEXEC_B32     : SOP1_Real_gfx11<0x026>;
defm S_NAND_SAVEEXEC_B64     : SOP1_Real_gfx11<0x027>;
defm S_NOR_SAVEEXEC_B32      : SOP1_Real_gfx11<0x028>;
defm S_NOR_SAVEEXEC_B64      : SOP1_Real_gfx11<0x029>;
defm S_XNOR_SAVEEXEC_B32     : SOP1_Real_gfx11<0x02a>;
/*defm S_XNOR_SAVEEXEC_B64   : SOP1_Real_gfx11<0x02b>; //same as older arch, handled there*/
defm S_AND_NOT0_SAVEEXEC_B32 : SOP1_Real_Renamed_gfx11<0x02c, S_ANDN1_SAVEEXEC_B32, "s_and_not0_saveexec_b32">;
defm S_AND_NOT0_SAVEEXEC_B64 : SOP1_Real_Renamed_gfx11<0x02d, S_ANDN1_SAVEEXEC_B64, "s_and_not0_saveexec_b64">;
defm S_OR_NOT0_SAVEEXEC_B32  : SOP1_Real_Renamed_gfx11<0x02e, S_ORN1_SAVEEXEC_B32, "s_or_not0_saveexec_b32">;
defm S_OR_NOT0_SAVEEXEC_B64  : SOP1_Real_Renamed_gfx11<0x02f, S_ORN1_SAVEEXEC_B64, "s_or_not0_saveexec_b64">;
defm S_AND_NOT1_SAVEEXEC_B32 : SOP1_Real_Renamed_gfx11<0x030, S_ANDN2_SAVEEXEC_B32, "s_and_not1_saveexec_b32">;
defm S_AND_NOT1_SAVEEXEC_B64 : SOP1_Real_Renamed_gfx11<0x031, S_ANDN2_SAVEEXEC_B64, "s_and_not1_saveexec_b64">;
defm S_OR_NOT1_SAVEEXEC_B32  : SOP1_Real_Renamed_gfx11<0x032, S_ORN2_SAVEEXEC_B32, "s_or_not1_saveexec_b32">;
defm S_OR_NOT1_SAVEEXEC_B64  : SOP1_Real_Renamed_gfx11<0x033, S_ORN2_SAVEEXEC_B64, "s_or_not1_saveexec_b64">;
defm S_AND_NOT0_WREXEC_B32   : SOP1_Real_Renamed_gfx11<0x034, S_ANDN1_WREXEC_B32, "s_and_not0_wrexec_b32">;
defm S_AND_NOT0_WREXEC_B64   : SOP1_Real_Renamed_gfx11<0x035, S_ANDN1_WREXEC_B64, "s_and_not0_wrexec_b64">;
defm S_AND_NOT1_WREXEC_B32   : SOP1_Real_Renamed_gfx11<0x036, S_ANDN2_WREXEC_B32, "s_and_not1_wrexec_b32">;
defm S_AND_NOT1_WREXEC_B64   : SOP1_Real_Renamed_gfx11<0x037, S_ANDN2_WREXEC_B64, "s_and_not1_wrexec_b64">;
defm S_MOVRELS_B32           : SOP1_Real_gfx11<0x040>;
defm S_MOVRELS_B64           : SOP1_Real_gfx11<0x041>;
defm S_MOVRELD_B32           : SOP1_Real_gfx11<0x042>;
defm S_MOVRELD_B64           : SOP1_Real_gfx11<0x043>;
defm S_MOVRELSD_2_B32        : SOP1_Real_gfx11<0x044>;
defm S_GETPC_B64             : SOP1_Real_gfx11<0x047>;
defm S_SETPC_B64             : SOP1_Real_gfx11<0x048>;
defm S_SWAPPC_B64            : SOP1_Real_gfx11<0x049>;
defm S_RFE_B64               : SOP1_Real_gfx11<0x04a>;
defm S_SENDMSG_RTN_B32       : SOP1_Real_gfx11<0x04c>;
defm S_SENDMSG_RTN_B64       : SOP1_Real_gfx11<0x04d>;

//===----------------------------------------------------------------------===//
// SOP1 - GFX10.
//===----------------------------------------------------------------------===//

multiclass SOP1_Real_gfx10<bits<8> op> {
  defvar ps = !cast<SOP1_Pseudo>(NAME);
  def _gfx10 : SOP1_Real<op, ps>,
               Select_gfx10<ps.Mnemonic>;
}

multiclass SOP1_Real_gfx10_gfx11<bits<8> op> :
  SOP1_Real_gfx10<op>, SOP1_Real_gfx11<op>;

defm S_ANDN1_SAVEEXEC_B64   : SOP1_Real_gfx10<0x037>;
defm S_ORN1_SAVEEXEC_B64    : SOP1_Real_gfx10<0x038>;
defm S_ANDN1_WREXEC_B64     : SOP1_Real_gfx10<0x039>;
defm S_ANDN2_WREXEC_B64     : SOP1_Real_gfx10<0x03a>;
defm S_BITREPLICATE_B64_B32 : SOP1_Real_gfx10<0x03b>;
defm S_AND_SAVEEXEC_B32     : SOP1_Real_gfx10<0x03c>;
defm S_OR_SAVEEXEC_B32      : SOP1_Real_gfx10<0x03d>;
defm S_XOR_SAVEEXEC_B32     : SOP1_Real_gfx10<0x03e>;
defm S_ANDN2_SAVEEXEC_B32   : SOP1_Real_gfx10<0x03f>;
defm S_ORN2_SAVEEXEC_B32    : SOP1_Real_gfx10<0x040>;
defm S_NAND_SAVEEXEC_B32    : SOP1_Real_gfx10<0x041>;
defm S_NOR_SAVEEXEC_B32     : SOP1_Real_gfx10<0x042>;
defm S_XNOR_SAVEEXEC_B32    : SOP1_Real_gfx10<0x043>;
defm S_ANDN1_SAVEEXEC_B32   : SOP1_Real_gfx10<0x044>;
defm S_ORN1_SAVEEXEC_B32    : SOP1_Real_gfx10<0x045>;
defm S_ANDN1_WREXEC_B32     : SOP1_Real_gfx10<0x046>;
defm S_ANDN2_WREXEC_B32     : SOP1_Real_gfx10<0x047>;
defm S_MOVRELSD_2_B32       : SOP1_Real_gfx10<0x049>;

//===----------------------------------------------------------------------===//
// SOP1 - GFX6, GFX7, GFX10, GFX11.
//===----------------------------------------------------------------------===//


multiclass SOP1_Real_gfx6_gfx7<bits<8> op> {
  defvar ps = !cast<SOP1_Pseudo>(NAME);
  def _gfx6_gfx7 : SOP1_Real<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>;
}

multiclass SOP1_Real_gfx6_gfx7_gfx10<bits<8> op> :
  SOP1_Real_gfx6_gfx7<op>, SOP1_Real_gfx10<op>;

multiclass SOP1_Real_gfx6_gfx7_gfx10_gfx11<bits<8> op> :
  SOP1_Real_gfx6_gfx7<op>, SOP1_Real_gfx10_gfx11<op>;

defm S_CBRANCH_JOIN  : SOP1_Real_gfx6_gfx7<0x032>;

defm S_MOV_B32            : SOP1_Real_gfx6_gfx7_gfx10<0x003>;
defm S_MOV_B64            : SOP1_Real_gfx6_gfx7_gfx10<0x004>;
defm S_CMOV_B32           : SOP1_Real_gfx6_gfx7_gfx10<0x005>;
defm S_CMOV_B64           : SOP1_Real_gfx6_gfx7_gfx10<0x006>;
defm S_NOT_B32            : SOP1_Real_gfx6_gfx7_gfx10<0x007>;
defm S_NOT_B64            : SOP1_Real_gfx6_gfx7_gfx10<0x008>;
defm S_WQM_B32            : SOP1_Real_gfx6_gfx7_gfx10<0x009>;
defm S_WQM_B64            : SOP1_Real_gfx6_gfx7_gfx10<0x00a>;
defm S_BREV_B32           : SOP1_Real_gfx6_gfx7_gfx10<0x00b>;
defm S_BREV_B64           : SOP1_Real_gfx6_gfx7_gfx10<0x00c>;
defm S_BCNT0_I32_B32      : SOP1_Real_gfx6_gfx7_gfx10<0x00d>;
defm S_BCNT0_I32_B64      : SOP1_Real_gfx6_gfx7_gfx10<0x00e>;
defm S_BCNT1_I32_B32      : SOP1_Real_gfx6_gfx7_gfx10<0x00f>;
defm S_BCNT1_I32_B64      : SOP1_Real_gfx6_gfx7_gfx10<0x010>;
defm S_FF0_I32_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x011>;
defm S_FF0_I32_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x012>;
defm S_FF1_I32_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x013>;
defm S_FF1_I32_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x014>;
defm S_FLBIT_I32_B32      : SOP1_Real_gfx6_gfx7_gfx10<0x015>;
defm S_FLBIT_I32_B64      : SOP1_Real_gfx6_gfx7_gfx10<0x016>;
defm S_FLBIT_I32          : SOP1_Real_gfx6_gfx7_gfx10<0x017>;
defm S_FLBIT_I32_I64      : SOP1_Real_gfx6_gfx7_gfx10<0x018>;
defm S_SEXT_I32_I8        : SOP1_Real_gfx6_gfx7_gfx10<0x019>;
defm S_SEXT_I32_I16       : SOP1_Real_gfx6_gfx7_gfx10<0x01a>;
defm S_BITSET0_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x01b>;
defm S_BITSET0_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x01c>;
defm S_BITSET1_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x01d>;
defm S_BITSET1_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x01e>;
defm S_GETPC_B64          : SOP1_Real_gfx6_gfx7_gfx10<0x01f>;
defm S_SETPC_B64          : SOP1_Real_gfx6_gfx7_gfx10<0x020>;
defm S_SWAPPC_B64         : SOP1_Real_gfx6_gfx7_gfx10<0x021>;
defm S_RFE_B64            : SOP1_Real_gfx6_gfx7_gfx10<0x022>;
defm S_AND_SAVEEXEC_B64   : SOP1_Real_gfx6_gfx7_gfx10<0x024>;
defm S_OR_SAVEEXEC_B64    : SOP1_Real_gfx6_gfx7_gfx10<0x025>;
defm S_XOR_SAVEEXEC_B64   : SOP1_Real_gfx6_gfx7_gfx10<0x026>;
defm S_ANDN2_SAVEEXEC_B64 : SOP1_Real_gfx6_gfx7_gfx10<0x027>;
defm S_ORN2_SAVEEXEC_B64  : SOP1_Real_gfx6_gfx7_gfx10<0x028>;
defm S_NAND_SAVEEXEC_B64  : SOP1_Real_gfx6_gfx7_gfx10<0x029>;
defm S_NOR_SAVEEXEC_B64   : SOP1_Real_gfx6_gfx7_gfx10<0x02a>;
defm S_XNOR_SAVEEXEC_B64  : SOP1_Real_gfx6_gfx7_gfx10_gfx11<0x02b>;
defm S_QUADMASK_B32       : SOP1_Real_gfx6_gfx7_gfx10<0x02c>;
defm S_QUADMASK_B64       : SOP1_Real_gfx6_gfx7_gfx10<0x02d>;
defm S_MOVRELS_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x02e>;
defm S_MOVRELS_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x02f>;
defm S_MOVRELD_B32        : SOP1_Real_gfx6_gfx7_gfx10<0x030>;
defm S_MOVRELD_B64        : SOP1_Real_gfx6_gfx7_gfx10<0x031>;
defm S_ABS_I32            : SOP1_Real_gfx6_gfx7_gfx10<0x034>;

//===----------------------------------------------------------------------===//
// SOP2 - GFX11.
//===----------------------------------------------------------------------===//

multiclass SOP2_Real_gfx11<bits<7> op> {
  def _gfx11 : SOP2_Real<op, !cast<SOP2_Pseudo>(NAME)>,
               Select_gfx11<!cast<SOP2_Pseudo>(NAME).Mnemonic>;
}

multiclass SOP2_Real_Renamed_gfx11<bits<7> op, SOP2_Pseudo backing_pseudo, string real_name> {
  def _gfx11 : SOP2_Real<op, backing_pseudo, real_name>,
               Select_gfx11<backing_pseudo.Mnemonic>,
               MnemonicAlias<backing_pseudo.Mnemonic, real_name>, Requires<[isGFX11Plus]>;
}

defm S_ABSDIFF_I32     : SOP2_Real_gfx11<0x006>;
defm S_LSHL_B32        : SOP2_Real_gfx11<0x008>;
defm S_LSHL_B64        : SOP2_Real_gfx11<0x009>;
defm S_LSHR_B32        : SOP2_Real_gfx11<0x00a>;
defm S_LSHR_B64        : SOP2_Real_gfx11<0x00b>;
defm S_ASHR_I32        : SOP2_Real_gfx11<0x00c>;
defm S_ASHR_I64        : SOP2_Real_gfx11<0x00d>;
defm S_LSHL1_ADD_U32   : SOP2_Real_gfx11<0x00e>;
defm S_LSHL2_ADD_U32   : SOP2_Real_gfx11<0x00f>;
defm S_LSHL3_ADD_U32   : SOP2_Real_gfx11<0x010>;
defm S_LSHL4_ADD_U32   : SOP2_Real_gfx11<0x011>;
defm S_MIN_I32         : SOP2_Real_gfx11<0x012>;
defm S_MIN_U32         : SOP2_Real_gfx11<0x013>;
defm S_MAX_I32         : SOP2_Real_gfx11<0x014>;
defm S_MAX_U32         : SOP2_Real_gfx11<0x015>;
defm S_AND_B32         : SOP2_Real_gfx11<0x016>;
defm S_AND_B64         : SOP2_Real_gfx11<0x017>;
defm S_OR_B32          : SOP2_Real_gfx11<0x018>;
defm S_OR_B64          : SOP2_Real_gfx11<0x019>;
defm S_XOR_B32         : SOP2_Real_gfx11<0x01a>;
defm S_XOR_B64         : SOP2_Real_gfx11<0x01b>;
defm S_NAND_B32        : SOP2_Real_gfx11<0x01c>;
defm S_NAND_B64        : SOP2_Real_gfx11<0x01d>;
defm S_NOR_B32         : SOP2_Real_gfx11<0x01e>;
defm S_NOR_B64         : SOP2_Real_gfx11<0x01f>;
defm S_XNOR_B32        : SOP2_Real_gfx11<0x020>;
defm S_XNOR_B64        : SOP2_Real_gfx11<0x021>;
defm S_AND_NOT1_B32    : SOP2_Real_Renamed_gfx11<0x022, S_ANDN2_B32, "s_and_not1_b32">;
defm S_AND_NOT1_B64    : SOP2_Real_Renamed_gfx11<0x023, S_ANDN2_B64, "s_and_not1_b64">;
defm S_OR_NOT1_B32     : SOP2_Real_Renamed_gfx11<0x024, S_ORN2_B32, "s_or_not1_b32">;
defm S_OR_NOT1_B64     : SOP2_Real_Renamed_gfx11<0x025, S_ORN2_B64, "s_or_not1_b64">;
defm S_BFE_U32         : SOP2_Real_gfx11<0x026>;
defm S_BFE_I32         : SOP2_Real_gfx11<0x027>;
defm S_BFE_U64         : SOP2_Real_gfx11<0x028>;
defm S_BFE_I64         : SOP2_Real_gfx11<0x029>;
defm S_BFM_B32         : SOP2_Real_gfx11<0x02a>;
defm S_BFM_B64         : SOP2_Real_gfx11<0x02b>;
defm S_MUL_I32         : SOP2_Real_gfx11<0x02c>;
defm S_MUL_HI_U32      : SOP2_Real_gfx11<0x02d>;
defm S_MUL_HI_I32      : SOP2_Real_gfx11<0x02e>;
defm S_CSELECT_B32     : SOP2_Real_gfx11<0x030>;
defm S_CSELECT_B64     : SOP2_Real_gfx11<0x031>;
defm S_PACK_HL_B32_B16 : SOP2_Real_gfx11<0x035>;

//===----------------------------------------------------------------------===//
// SOP2 - GFX10.
//===----------------------------------------------------------------------===//

multiclass SOP2_Real_gfx10<bits<7> op> {
  defvar ps = !cast<SOP2_Pseudo>(NAME);
  def _gfx10 : SOP2_Real<op, ps>,
               Select_gfx10<ps.Mnemonic>;
}

multiclass SOP2_Real_gfx10_gfx11<bits<7> op> :
  SOP2_Real_gfx10<op>, SOP2_Real_gfx11<op>;

defm S_LSHL1_ADD_U32   : SOP2_Real_gfx10<0x02e>;
defm S_LSHL2_ADD_U32   : SOP2_Real_gfx10<0x02f>;
defm S_LSHL3_ADD_U32   : SOP2_Real_gfx10<0x030>;
defm S_LSHL4_ADD_U32   : SOP2_Real_gfx10<0x031>;
defm S_PACK_LL_B32_B16 : SOP2_Real_gfx10_gfx11<0x032>;
defm S_PACK_LH_B32_B16 : SOP2_Real_gfx10_gfx11<0x033>;
defm S_PACK_HH_B32_B16 : SOP2_Real_gfx10_gfx11<0x034>;
defm S_MUL_HI_U32      : SOP2_Real_gfx10<0x035>;
defm S_MUL_HI_I32      : SOP2_Real_gfx10<0x036>;

//===----------------------------------------------------------------------===//
// SOP2 - GFX6, GFX7.
//===----------------------------------------------------------------------===//

multiclass SOP2_Real_gfx6_gfx7<bits<7> op> {
  defvar ps = !cast<SOP_Pseudo>(NAME);
  def _gfx6_gfx7 : SOP2_Real<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>;
}

multiclass SOP2_Real_gfx6_gfx7_gfx10<bits<7> op> :
  SOP2_Real_gfx6_gfx7<op>, SOP2_Real_gfx10<op>;

multiclass SOP2_Real_gfx6_gfx7_gfx10_gfx11<bits<7> op> :
  SOP2_Real_gfx6_gfx7<op>, SOP2_Real_gfx10_gfx11<op>;

defm S_CBRANCH_G_FORK : SOP2_Real_gfx6_gfx7<0x02b>;

defm S_ADD_U32     : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x000>;
defm S_SUB_U32     : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x001>;
defm S_ADD_I32     : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x002>;
defm S_SUB_I32     : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x003>;
defm S_ADDC_U32    : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x004>;
defm S_SUBB_U32    : SOP2_Real_gfx6_gfx7_gfx10_gfx11<0x005>;
defm S_MIN_I32     : SOP2_Real_gfx6_gfx7_gfx10<0x006>;
defm S_MIN_U32     : SOP2_Real_gfx6_gfx7_gfx10<0x007>;
defm S_MAX_I32     : SOP2_Real_gfx6_gfx7_gfx10<0x008>;
defm S_MAX_U32     : SOP2_Real_gfx6_gfx7_gfx10<0x009>;
defm S_CSELECT_B32 : SOP2_Real_gfx6_gfx7_gfx10<0x00a>;
defm S_CSELECT_B64 : SOP2_Real_gfx6_gfx7_gfx10<0x00b>;
defm S_AND_B32     : SOP2_Real_gfx6_gfx7_gfx10<0x00e>;
defm S_AND_B64     : SOP2_Real_gfx6_gfx7_gfx10<0x00f>;
defm S_OR_B32      : SOP2_Real_gfx6_gfx7_gfx10<0x010>;
defm S_OR_B64      : SOP2_Real_gfx6_gfx7_gfx10<0x011>;
defm S_XOR_B32     : SOP2_Real_gfx6_gfx7_gfx10<0x012>;
defm S_XOR_B64     : SOP2_Real_gfx6_gfx7_gfx10<0x013>;
defm S_ANDN2_B32   : SOP2_Real_gfx6_gfx7_gfx10<0x014>;
defm S_ANDN2_B64   : SOP2_Real_gfx6_gfx7_gfx10<0x015>;
defm S_ORN2_B32    : SOP2_Real_gfx6_gfx7_gfx10<0x016>;
defm S_ORN2_B64    : SOP2_Real_gfx6_gfx7_gfx10<0x017>;
defm S_NAND_B32    : SOP2_Real_gfx6_gfx7_gfx10<0x018>;
defm S_NAND_B64    : SOP2_Real_gfx6_gfx7_gfx10<0x019>;
defm S_NOR_B32     : SOP2_Real_gfx6_gfx7_gfx10<0x01a>;
defm S_NOR_B64     : SOP2_Real_gfx6_gfx7_gfx10<0x01b>;
defm S_XNOR_B32    : SOP2_Real_gfx6_gfx7_gfx10<0x01c>;
defm S_XNOR_B64    : SOP2_Real_gfx6_gfx7_gfx10<0x01d>;
defm S_LSHL_B32    : SOP2_Real_gfx6_gfx7_gfx10<0x01e>;
defm S_LSHL_B64    : SOP2_Real_gfx6_gfx7_gfx10<0x01f>;
defm S_LSHR_B32    : SOP2_Real_gfx6_gfx7_gfx10<0x020>;
defm S_LSHR_B64    : SOP2_Real_gfx6_gfx7_gfx10<0x021>;
defm S_ASHR_I32    : SOP2_Real_gfx6_gfx7_gfx10<0x022>;
defm S_ASHR_I64    : SOP2_Real_gfx6_gfx7_gfx10<0x023>;
defm S_BFM_B32     : SOP2_Real_gfx6_gfx7_gfx10<0x024>;
defm S_BFM_B64     : SOP2_Real_gfx6_gfx7_gfx10<0x025>;
defm S_MUL_I32     : SOP2_Real_gfx6_gfx7_gfx10<0x026>;
defm S_BFE_U32     : SOP2_Real_gfx6_gfx7_gfx10<0x027>;
defm S_BFE_I32     : SOP2_Real_gfx6_gfx7_gfx10<0x028>;
defm S_BFE_U64     : SOP2_Real_gfx6_gfx7_gfx10<0x029>;
defm S_BFE_I64     : SOP2_Real_gfx6_gfx7_gfx10<0x02a>;
defm S_ABSDIFF_I32 : SOP2_Real_gfx6_gfx7_gfx10<0x02c>;

//===----------------------------------------------------------------------===//
// SOPK - GFX11.
//===----------------------------------------------------------------------===//

multiclass SOPK_Real32_gfx11<bits<5> op> {
  def _gfx11 : SOPK_Real32<op, !cast<SOPK_Pseudo>(NAME)>,
               Select_gfx11<!cast<SOPK_Pseudo>(NAME).Mnemonic>;
}

multiclass SOPK_Real64_gfx11<bits<5> op> {
  def _gfx11 : SOPK_Real64<op, !cast<SOPK_Pseudo>(NAME)>,
               Select_gfx11<!cast<SOPK_Pseudo>(NAME).Mnemonic>;
}

defm S_GETREG_B32           : SOPK_Real32_gfx11<0x011>;
defm S_SETREG_B32           : SOPK_Real32_gfx11<0x012>;
defm S_SETREG_IMM32_B32     : SOPK_Real64_gfx11<0x013>;
defm S_CALL_B64             : SOPK_Real32_gfx11<0x014>;
defm S_SUBVECTOR_LOOP_BEGIN : SOPK_Real32_gfx11<0x016>;
defm S_SUBVECTOR_LOOP_END   : SOPK_Real32_gfx11<0x017>;
defm S_WAITCNT_VSCNT        : SOPK_Real32_gfx11<0x018>;
defm S_WAITCNT_VMCNT        : SOPK_Real32_gfx11<0x019>;
defm S_WAITCNT_EXPCNT       : SOPK_Real32_gfx11<0x01a>;
defm S_WAITCNT_LGKMCNT      : SOPK_Real32_gfx11<0x01b>;

//===----------------------------------------------------------------------===//
// SOPK - GFX10.
//===----------------------------------------------------------------------===//

multiclass SOPK_Real32_gfx10<bits<5> op> {
  defvar ps = !cast<SOPK_Pseudo>(NAME);
  def _gfx10 : SOPK_Real32<op, ps>,
               Select_gfx10<ps.Mnemonic>;
}

multiclass SOPK_Real64_gfx10<bits<5> op> {
  defvar ps = !cast<SOPK_Pseudo>(NAME);
  def _gfx10 : SOPK_Real64<op, ps>,
               Select_gfx10<ps.Mnemonic>;
}

multiclass SOPK_Real32_gfx10_gfx11<bits<5> op> :
  SOPK_Real32_gfx10<op>, SOPK_Real32_gfx11<op>;

defm S_VERSION              : SOPK_Real32_gfx10_gfx11<0x001>;
defm S_CALL_B64             : SOPK_Real32_gfx10<0x016>;
defm S_WAITCNT_VSCNT        : SOPK_Real32_gfx10<0x017>;
defm S_WAITCNT_VMCNT        : SOPK_Real32_gfx10<0x018>;
defm S_WAITCNT_EXPCNT       : SOPK_Real32_gfx10<0x019>;
defm S_WAITCNT_LGKMCNT      : SOPK_Real32_gfx10<0x01a>;
defm S_SUBVECTOR_LOOP_BEGIN : SOPK_Real32_gfx10<0x01b>;
defm S_SUBVECTOR_LOOP_END   : SOPK_Real32_gfx10<0x01c>;

//===----------------------------------------------------------------------===//
// SOPK - GFX6, GFX7.
//===----------------------------------------------------------------------===//

multiclass SOPK_Real32_gfx6_gfx7<bits<5> op> {
  defvar ps = !cast<SOPK_Pseudo>(NAME);
  def _gfx6_gfx7 : SOPK_Real32<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>;
}

multiclass SOPK_Real64_gfx6_gfx7<bits<5> op> {
  defvar ps = !cast<SOPK_Pseudo>(NAME);
  def _gfx6_gfx7 : SOPK_Real64<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>;
}

multiclass SOPK_Real32_gfx6_gfx7_gfx10<bits<5> op> :
  SOPK_Real32_gfx6_gfx7<op>, SOPK_Real32_gfx10<op>;

multiclass SOPK_Real64_gfx6_gfx7_gfx10<bits<5> op> :
  SOPK_Real64_gfx6_gfx7<op>, SOPK_Real64_gfx10<op>;

multiclass SOPK_Real32_gfx6_gfx7_gfx10_gfx11<bits<5> op> :
  SOPK_Real32_gfx6_gfx7<op>, SOPK_Real32_gfx10_gfx11<op>;

defm S_CBRANCH_I_FORK : SOPK_Real32_gfx6_gfx7<0x011>;

defm S_MOVK_I32         : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x000>;
defm S_CMOVK_I32        : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x002>;
defm S_CMPK_EQ_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x003>;
defm S_CMPK_LG_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x004>;
defm S_CMPK_GT_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x005>;
defm S_CMPK_GE_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x006>;
defm S_CMPK_LT_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x007>;
defm S_CMPK_LE_I32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x008>;
defm S_CMPK_EQ_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x009>;
defm S_CMPK_LG_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00a>;
defm S_CMPK_GT_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00b>;
defm S_CMPK_GE_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00c>;
defm S_CMPK_LT_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00d>;
defm S_CMPK_LE_U32      : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00e>;
defm S_ADDK_I32         : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x00f>;
defm S_MULK_I32         : SOPK_Real32_gfx6_gfx7_gfx10_gfx11<0x010>;
defm S_GETREG_B32       : SOPK_Real32_gfx6_gfx7_gfx10<0x012>;
defm S_SETREG_B32       : SOPK_Real32_gfx6_gfx7_gfx10<0x013>;
defm S_SETREG_IMM32_B32 : SOPK_Real64_gfx6_gfx7_gfx10<0x015>;

//===----------------------------------------------------------------------===//
// SOPP - GFX11
//===----------------------------------------------------------------------===//

multiclass SOPP_Real_32_gfx11<bits<7> op> {
  def _gfx11 : SOPP_Real_32<op, !cast<SOPP_Pseudo>(NAME), !cast<SOPP_Pseudo>(NAME).Mnemonic>,
               Select_gfx11<!cast<SOPP_Pseudo>(NAME).Mnemonic>,
               SOPPRelaxTable<0, !cast<SOPP_Pseudo>(NAME).KeyName, "_gfx11">;
}

multiclass SOPP_Real_64_gfx11<bits<7> op> {
  def _gfx11 : SOPP_Real_64<op, !cast<SOPP_Pseudo>(NAME), !cast<SOPP_Pseudo>(NAME).Mnemonic>,
               Select_gfx11<!cast<SOPP_Pseudo>(NAME).Mnemonic>,
               SOPPRelaxTable<1, !cast<SOPP_Pseudo>(NAME).KeyName, "_gfx11">;
}

multiclass SOPP_Real_32_Renamed_gfx11<bits<7> op, SOPP_Pseudo backing_pseudo, string real_name> {
  def _gfx11 : SOPP_Real_32<op, backing_pseudo, real_name>,
               Select_gfx11<backing_pseudo.Mnemonic>,
               MnemonicAlias<backing_pseudo.Mnemonic, real_name>, Requires<[isGFX11Plus]>;
}

multiclass SOPP_Real_With_Relaxation_gfx11<bits<7> op> {
  defm "" : SOPP_Real_32_gfx11<op>;
  defm _pad_s_nop : SOPP_Real_64_gfx11<op>;
}

defm S_SETKILL                    : SOPP_Real_32_gfx11<0x001>;
defm S_SETHALT                    : SOPP_Real_32_gfx11<0x002>;
defm S_SLEEP                      : SOPP_Real_32_gfx11<0x003>;
defm S_SET_INST_PREFETCH_DISTANCE : SOPP_Real_32_Renamed_gfx11<0x004, S_INST_PREFETCH, "s_set_inst_prefetch_distance">;
defm S_CLAUSE                     : SOPP_Real_32_gfx11<0x005>;
defm S_DELAY_ALU                  : SOPP_Real_32_gfx11<0x007>;
defm S_WAITCNT_DEPCTR             : SOPP_Real_32_gfx11<0x008>;
defm S_WAITCNT                    : SOPP_Real_32_gfx11<0x009>;
defm S_WAIT_IDLE                  : SOPP_Real_32_gfx11<0x00a>;
defm S_WAIT_EVENT                 : SOPP_Real_32_gfx11<0x00b>;
defm S_TRAP                       : SOPP_Real_32_gfx11<0x010>;
defm S_ROUND_MODE                 : SOPP_Real_32_gfx11<0x011>;
defm S_DENORM_MODE                : SOPP_Real_32_gfx11<0x012>;
defm S_BRANCH                     : SOPP_Real_With_Relaxation_gfx11<0x020>;
defm S_CBRANCH_SCC0               : SOPP_Real_With_Relaxation_gfx11<0x021>;
defm S_CBRANCH_SCC1               : SOPP_Real_With_Relaxation_gfx11<0x022>;
defm S_CBRANCH_VCCZ               : SOPP_Real_With_Relaxation_gfx11<0x023>;
defm S_CBRANCH_VCCNZ              : SOPP_Real_With_Relaxation_gfx11<0x024>;
defm S_CBRANCH_EXECZ              : SOPP_Real_With_Relaxation_gfx11<0x025>;
defm S_CBRANCH_EXECNZ             : SOPP_Real_With_Relaxation_gfx11<0x026>;
defm S_CBRANCH_CDBGSYS            : SOPP_Real_With_Relaxation_gfx11<0x027>;
defm S_CBRANCH_CDBGUSER           : SOPP_Real_With_Relaxation_gfx11<0x028>;
defm S_CBRANCH_CDBGSYS_OR_USER    : SOPP_Real_With_Relaxation_gfx11<0x029>;
defm S_CBRANCH_CDBGSYS_AND_USER   : SOPP_Real_With_Relaxation_gfx11<0x02a>;
defm S_ENDPGM                     : SOPP_Real_32_gfx11<0x030>;
defm S_ENDPGM_SAVED               : SOPP_Real_32_gfx11<0x031>;
defm S_WAKEUP                     : SOPP_Real_32_gfx11<0x034>;
defm S_SETPRIO                    : SOPP_Real_32_gfx11<0x035>;
defm S_SENDMSG                    : SOPP_Real_32_gfx11<0x036>;
defm S_SENDMSGHALT                : SOPP_Real_32_gfx11<0x037>;
defm S_INCPERFLEVEL               : SOPP_Real_32_gfx11<0x038>;
defm S_DECPERFLEVEL               : SOPP_Real_32_gfx11<0x039>;
defm S_TTRACEDATA                 : SOPP_Real_32_gfx11<0x03a>;
defm S_TTRACEDATA_IMM             : SOPP_Real_32_gfx11<0x03b>;
defm S_ICACHE_INV                 : SOPP_Real_32_gfx11<0x03c>;
defm S_BARRIER                    : SOPP_Real_32_gfx11<0x03d>;

//===----------------------------------------------------------------------===//
// SOPP - GFX6, GFX7, GFX8, GFX9, GFX10
//===----------------------------------------------------------------------===//

multiclass SOPP_Real_32_gfx6_gfx7<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _gfx6_gfx7 : SOPP_Real_32<op, ps, !cast<SOPP_Pseudo>(NAME).Mnemonic>,
                   Select_gfx6_gfx7<ps.Mnemonic>,
                   SOPPRelaxTable<0, ps.KeyName, "_gfx6_gfx7">;
}

multiclass SOPP_Real_32_gfx8_gfx9<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _vi : SOPP_Real_32<op, ps>,
            Select_vi<ps.Mnemonic>,
            SOPPRelaxTable<0, ps.KeyName, "_vi">;
}

multiclass SOPP_Real_32_gfx10<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _gfx10 : SOPP_Real_32<op, ps>,
               Select_gfx10<ps.Mnemonic>,
               SOPPRelaxTable<0, ps.KeyName, "_gfx10">;
}

multiclass SOPP_Real_32_gfx8_gfx9_gfx10<bits<7> op> :
  SOPP_Real_32_gfx8_gfx9<op>, SOPP_Real_32_gfx10<op>;

multiclass SOPP_Real_32_gfx6_gfx7_gfx8_gfx9<bits<7> op> :
  SOPP_Real_32_gfx6_gfx7<op>, SOPP_Real_32_gfx8_gfx9<op>;

multiclass SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<bits<7> op> :
  SOPP_Real_32_gfx6_gfx7_gfx8_gfx9<op>, SOPP_Real_32_gfx10<op>;

multiclass SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10_gfx11<bits<7> op> :
  SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<op>, SOPP_Real_32_gfx11<op>;

multiclass SOPP_Real_32_gfx10_gfx11<bits<7> op> :
  SOPP_Real_32_gfx10<op>, SOPP_Real_32_gfx11<op>;

//64 bit encodings, for Relaxation
multiclass SOPP_Real_64_gfx6_gfx7<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _gfx6_gfx7 : SOPP_Real_64<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>,
                   SOPPRelaxTable<1, ps.KeyName, "_gfx6_gfx7">;
}

multiclass SOPP_Real_64_gfx8_gfx9<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _vi : SOPP_Real_64<op, ps>,
            Select_vi<ps.Mnemonic>,
            SOPPRelaxTable<1, ps.KeyName, "_vi">;
}

multiclass SOPP_Real_64_gfx10<bits<7> op> {
  defvar ps = !cast<SOPP_Pseudo>(NAME);
  def _gfx10 : SOPP_Real_64<op, ps>,
               Select_gfx10<ps.Mnemonic>,
               SOPPRelaxTable<1, ps.KeyName, "_gfx10">;
}

multiclass SOPP_Real_64_gfx8_gfx9_gfx10<bits<7> op> :
  SOPP_Real_64_gfx8_gfx9<op>, SOPP_Real_64_gfx10<op>;

multiclass SOPP_Real_64_gfx6_gfx7_gfx8_gfx9<bits<7> op> :
  SOPP_Real_64_gfx6_gfx7<op>, SOPP_Real_64_gfx8_gfx9<op>;

multiclass SOPP_Real_64_gfx6_gfx7_gfx8_gfx9_gfx10<bits<7> op> :
  SOPP_Real_64_gfx6_gfx7_gfx8_gfx9<op>, SOPP_Real_64_gfx10<op>;

multiclass SOPP_Real_64_gfx6_gfx7_gfx8_gfx9_gfx10_gfx11<bits<7> op> :
  SOPP_Real_64_gfx6_gfx7_gfx8_gfx9_gfx10<op>, SOPP_Real_64_gfx11<op>;

//relaxation for insts with no operands not implemented
multiclass SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<bits<7> op> {
  defm "" : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<op>;
  defm _pad_s_nop : SOPP_Real_64_gfx6_gfx7_gfx8_gfx9_gfx10<op>;
}

defm S_NOP                      : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10_gfx11<0x000>;
defm S_ENDPGM                   : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x001>;
defm S_WAKEUP                   : SOPP_Real_32_gfx8_gfx9_gfx10<0x003>;
defm S_BARRIER                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00a>;
defm S_WAITCNT                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00c>;
defm S_SETHALT                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00d>;
defm S_SETKILL                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00b>;
defm S_SLEEP                    : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00e>;
defm S_SETPRIO                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x00f>;
defm S_SENDMSG                  : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x010>;
defm S_SENDMSGHALT              : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x011>;
defm S_TRAP                     : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x012>;
defm S_ICACHE_INV               : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x013>;
defm S_INCPERFLEVEL             : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x014>;
defm S_DECPERFLEVEL             : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x015>;
defm S_TTRACEDATA               : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x016>;
defm S_ENDPGM_SAVED             : SOPP_Real_32_gfx6_gfx7_gfx8_gfx9_gfx10<0x01B>;
defm S_SET_GPR_IDX_OFF          : SOPP_Real_32_gfx8_gfx9<0x01c>;
defm S_SET_GPR_IDX_MODE         : SOPP_Real_32_gfx8_gfx9<0x01d>;
defm S_ENDPGM_ORDERED_PS_DONE   : SOPP_Real_32_gfx8_gfx9_gfx10<0x01e>;
defm S_CODE_END                 : SOPP_Real_32_gfx10_gfx11<0x01f>;
defm S_INST_PREFETCH            : SOPP_Real_32_gfx10<0x020>;
defm S_CLAUSE                   : SOPP_Real_32_gfx10<0x021>;
defm S_WAIT_IDLE                : SOPP_Real_32_gfx10<0x022>;
defm S_WAITCNT_DEPCTR           : SOPP_Real_32_gfx10<0x023>;
defm S_ROUND_MODE               : SOPP_Real_32_gfx10<0x024>;
defm S_DENORM_MODE              : SOPP_Real_32_gfx10<0x025>;
defm S_TTRACEDATA_IMM           : SOPP_Real_32_gfx10<0x028>;

let isBranch = 1 in {
defm S_BRANCH                   : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x002>;
defm S_CBRANCH_SCC0             : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x004>;
defm S_CBRANCH_SCC1             : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x005>;
defm S_CBRANCH_VCCZ             : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x006>;
defm S_CBRANCH_VCCNZ            : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x007>;
defm S_CBRANCH_EXECZ            : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x008>;
defm S_CBRANCH_EXECNZ           : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x009>;
defm S_CBRANCH_CDBGSYS          : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x017>;
defm S_CBRANCH_CDBGUSER         : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x018>;
defm S_CBRANCH_CDBGSYS_OR_USER  : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x019>;
defm S_CBRANCH_CDBGSYS_AND_USER : SOPP_Real_With_Relaxation_gfx6_gfx7_gfx8_gfx9_gfx10<0x01A>;
}

//===----------------------------------------------------------------------===//
// SOPC - GFX11
//===----------------------------------------------------------------------===//

multiclass SOPC_Real_gfx11<bits<7> op> {
  def _gfx11 : SOPC_Real<op, !cast<SOPC_Pseudo>(NAME)>,
               Select_gfx11<!cast<SOPC_Pseudo>(NAME).Mnemonic>;
}

defm S_CMP_EQ_I32     : SOPC_Real_gfx11<0x00>;
defm S_CMP_LG_I32     : SOPC_Real_gfx11<0x01>;
defm S_CMP_GT_I32     : SOPC_Real_gfx11<0x02>;
defm S_CMP_GE_I32     : SOPC_Real_gfx11<0x03>;
defm S_CMP_LT_I32     : SOPC_Real_gfx11<0x04>;
defm S_CMP_LE_I32     : SOPC_Real_gfx11<0x05>;
defm S_CMP_EQ_U32     : SOPC_Real_gfx11<0x06>;
defm S_CMP_LG_U32     : SOPC_Real_gfx11<0x07>;
defm S_CMP_GT_U32     : SOPC_Real_gfx11<0x08>;
defm S_CMP_GE_U32     : SOPC_Real_gfx11<0x09>;
defm S_CMP_LT_U32     : SOPC_Real_gfx11<0x0a>;
defm S_CMP_LE_U32     : SOPC_Real_gfx11<0x0b>;
defm S_BITCMP0_B32    : SOPC_Real_gfx11<0x0c>;
defm S_BITCMP1_B32    : SOPC_Real_gfx11<0x0d>;
defm S_BITCMP0_B64    : SOPC_Real_gfx11<0x0e>;
defm S_BITCMP1_B64    : SOPC_Real_gfx11<0x0f>;
defm S_CMP_EQ_U64     : SOPC_Real_gfx11<0x10>;
defm S_CMP_LG_U64     : SOPC_Real_gfx11<0x11>;

//===----------------------------------------------------------------------===//
// SOPC - GFX6, GFX7, GFX8, GFX9, GFX10
//===----------------------------------------------------------------------===//

multiclass SOPC_Real_gfx6_gfx7<bits<7> op> {
  defvar ps = !cast<SOPC_Pseudo>(NAME);
  def _gfx6_gfx7 : SOPC_Real<op, ps>,
                   Select_gfx6_gfx7<ps.Mnemonic>;
}

multiclass SOPC_Real_gfx8_gfx9<bits<7> op> {
  defvar ps = !cast<SOPC_Pseudo>(NAME);
  def _vi : SOPC_Real<op, ps>,
            Select_vi<ps.Mnemonic>;
}

multiclass SOPC_Real_gfx10<bits<7> op> {
  defvar ps = !cast<SOPC_Pseudo>(NAME);
  def _gfx10 : SOPC_Real<op, ps>,
               Select_gfx10<ps.Mnemonic>;
}

multiclass SOPC_Real_gfx8_gfx9_gfx10<bits<7> op> :
  SOPC_Real_gfx8_gfx9<op>, SOPC_Real_gfx10<op>;

multiclass SOPC_Real_gfx6_gfx7_gfx8_gfx9<bits<7> op> :
  SOPC_Real_gfx6_gfx7<op>, SOPC_Real_gfx8_gfx9<op>;

multiclass SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<bits<7> op> :
  SOPC_Real_gfx6_gfx7_gfx8_gfx9<op>, SOPC_Real_gfx10<op>;

defm S_CMP_EQ_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x00>;
defm S_CMP_LG_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x01>;
defm S_CMP_GT_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x02>;
defm S_CMP_GE_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x03>;
defm S_CMP_LT_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x04>;
defm S_CMP_LE_I32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x05>;
defm S_CMP_EQ_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x06>;
defm S_CMP_LG_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x07>;
defm S_CMP_GT_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x08>;
defm S_CMP_GE_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x09>;
defm S_CMP_LT_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0a>;
defm S_CMP_LE_U32     : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0b>;
defm S_BITCMP0_B32    : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0c>;
defm S_BITCMP1_B32    : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0d>;
defm S_BITCMP0_B64    : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0e>;
defm S_BITCMP1_B64    : SOPC_Real_gfx6_gfx7_gfx8_gfx9_gfx10<0x0f>;
defm S_SETVSKIP       : SOPC_Real_gfx6_gfx7_gfx8_gfx9<0x10>;
defm S_SET_GPR_IDX_ON : SOPC_Real_gfx8_gfx9<0x11>;
defm S_CMP_EQ_U64     : SOPC_Real_gfx8_gfx9_gfx10<0x12>;
defm S_CMP_LG_U64     : SOPC_Real_gfx8_gfx9_gfx10<0x13>;

//===----------------------------------------------------------------------===//
// GFX8 (VI), GFX9.
//===----------------------------------------------------------------------===//

class SOP1_Real_vi<bits<8> op, SOP1_Pseudo ps> :
  SOP1_Real<op, ps>,
  Select_vi<ps.Mnemonic>;


class SOP2_Real_vi<bits<7> op, SOP2_Pseudo ps> :
  SOP2_Real<op, ps>,
  Select_vi<ps.Mnemonic>;

class SOPK_Real_vi<bits<5> op, SOPK_Pseudo ps> :
  SOPK_Real32<op, ps>,
  Select_vi<ps.Mnemonic>;

def S_MOV_B32_vi           : SOP1_Real_vi <0x00, S_MOV_B32>;
def S_MOV_B64_vi           : SOP1_Real_vi <0x01, S_MOV_B64>;
def S_CMOV_B32_vi          : SOP1_Real_vi <0x02, S_CMOV_B32>;
def S_CMOV_B64_vi          : SOP1_Real_vi <0x03, S_CMOV_B64>;
def S_NOT_B32_vi           : SOP1_Real_vi <0x04, S_NOT_B32>;
def S_NOT_B64_vi           : SOP1_Real_vi <0x05, S_NOT_B64>;
def S_WQM_B32_vi           : SOP1_Real_vi <0x06, S_WQM_B32>;
def S_WQM_B64_vi           : SOP1_Real_vi <0x07, S_WQM_B64>;
def S_BREV_B32_vi          : SOP1_Real_vi <0x08, S_BREV_B32>;
def S_BREV_B64_vi          : SOP1_Real_vi <0x09, S_BREV_B64>;
def S_BCNT0_I32_B32_vi     : SOP1_Real_vi <0x0a, S_BCNT0_I32_B32>;
def S_BCNT0_I32_B64_vi     : SOP1_Real_vi <0x0b, S_BCNT0_I32_B64>;
def S_BCNT1_I32_B32_vi     : SOP1_Real_vi <0x0c, S_BCNT1_I32_B32>;
def S_BCNT1_I32_B64_vi     : SOP1_Real_vi <0x0d, S_BCNT1_I32_B64>;
def S_FF0_I32_B32_vi       : SOP1_Real_vi <0x0e, S_FF0_I32_B32>;
def S_FF0_I32_B64_vi       : SOP1_Real_vi <0x0f, S_FF0_I32_B64>;
def S_FF1_I32_B32_vi       : SOP1_Real_vi <0x10, S_FF1_I32_B32>;
def S_FF1_I32_B64_vi       : SOP1_Real_vi <0x11, S_FF1_I32_B64>;
def S_FLBIT_I32_B32_vi     : SOP1_Real_vi <0x12, S_FLBIT_I32_B32>;
def S_FLBIT_I32_B64_vi     : SOP1_Real_vi <0x13, S_FLBIT_I32_B64>;
def S_FLBIT_I32_vi         : SOP1_Real_vi <0x14, S_FLBIT_I32>;
def S_FLBIT_I32_I64_vi     : SOP1_Real_vi <0x15, S_FLBIT_I32_I64>;
def S_SEXT_I32_I8_vi       : SOP1_Real_vi <0x16, S_SEXT_I32_I8>;
def S_SEXT_I32_I16_vi      : SOP1_Real_vi <0x17, S_SEXT_I32_I16>;
def S_BITSET0_B32_vi       : SOP1_Real_vi <0x18, S_BITSET0_B32>;
def S_BITSET0_B64_vi       : SOP1_Real_vi <0x19, S_BITSET0_B64>;
def S_BITSET1_B32_vi       : SOP1_Real_vi <0x1a, S_BITSET1_B32>;
def S_BITSET1_B64_vi       : SOP1_Real_vi <0x1b, S_BITSET1_B64>;
def S_GETPC_B64_vi         : SOP1_Real_vi <0x1c, S_GETPC_B64>;
def S_SETPC_B64_vi         : SOP1_Real_vi <0x1d, S_SETPC_B64>;
def S_SWAPPC_B64_vi        : SOP1_Real_vi <0x1e, S_SWAPPC_B64>;
def S_RFE_B64_vi           : SOP1_Real_vi <0x1f, S_RFE_B64>;
def S_AND_SAVEEXEC_B64_vi  : SOP1_Real_vi <0x20, S_AND_SAVEEXEC_B64>;
def S_OR_SAVEEXEC_B64_vi   : SOP1_Real_vi <0x21, S_OR_SAVEEXEC_B64>;
def S_XOR_SAVEEXEC_B64_vi  : SOP1_Real_vi <0x22, S_XOR_SAVEEXEC_B64>;
def S_ANDN2_SAVEEXEC_B64_vi: SOP1_Real_vi <0x23, S_ANDN2_SAVEEXEC_B64>;
def S_ORN2_SAVEEXEC_B64_vi : SOP1_Real_vi <0x24, S_ORN2_SAVEEXEC_B64>;
def S_NAND_SAVEEXEC_B64_vi : SOP1_Real_vi <0x25, S_NAND_SAVEEXEC_B64>;
def S_NOR_SAVEEXEC_B64_vi  : SOP1_Real_vi <0x26, S_NOR_SAVEEXEC_B64>;
def S_XNOR_SAVEEXEC_B64_vi : SOP1_Real_vi <0x27, S_XNOR_SAVEEXEC_B64>;
def S_QUADMASK_B32_vi      : SOP1_Real_vi <0x28, S_QUADMASK_B32>;
def S_QUADMASK_B64_vi      : SOP1_Real_vi <0x29, S_QUADMASK_B64>;
def S_MOVRELS_B32_vi       : SOP1_Real_vi <0x2a, S_MOVRELS_B32>;
def S_MOVRELS_B64_vi       : SOP1_Real_vi <0x2b, S_MOVRELS_B64>;
def S_MOVRELD_B32_vi       : SOP1_Real_vi <0x2c, S_MOVRELD_B32>;
def S_MOVRELD_B64_vi       : SOP1_Real_vi <0x2d, S_MOVRELD_B64>;
def S_CBRANCH_JOIN_vi      : SOP1_Real_vi <0x2e, S_CBRANCH_JOIN>;
def S_ABS_I32_vi           : SOP1_Real_vi <0x30, S_ABS_I32>;
def S_SET_GPR_IDX_IDX_vi   : SOP1_Real_vi <0x32, S_SET_GPR_IDX_IDX>;

def S_ADD_U32_vi           : SOP2_Real_vi <0x00, S_ADD_U32>;
def S_ADD_I32_vi           : SOP2_Real_vi <0x02, S_ADD_I32>;
def S_SUB_U32_vi           : SOP2_Real_vi <0x01, S_SUB_U32>;
def S_SUB_I32_vi           : SOP2_Real_vi <0x03, S_SUB_I32>;
def S_ADDC_U32_vi          : SOP2_Real_vi <0x04, S_ADDC_U32>;
def S_SUBB_U32_vi          : SOP2_Real_vi <0x05, S_SUBB_U32>;
def S_MIN_I32_vi           : SOP2_Real_vi <0x06, S_MIN_I32>;
def S_MIN_U32_vi           : SOP2_Real_vi <0x07, S_MIN_U32>;
def S_MAX_I32_vi           : SOP2_Real_vi <0x08, S_MAX_I32>;
def S_MAX_U32_vi           : SOP2_Real_vi <0x09, S_MAX_U32>;
def S_CSELECT_B32_vi       : SOP2_Real_vi <0x0a, S_CSELECT_B32>;
def S_CSELECT_B64_vi       : SOP2_Real_vi <0x0b, S_CSELECT_B64>;
def S_AND_B32_vi           : SOP2_Real_vi <0x0c, S_AND_B32>;
def S_AND_B64_vi           : SOP2_Real_vi <0x0d, S_AND_B64>;
def S_OR_B32_vi            : SOP2_Real_vi <0x0e, S_OR_B32>;
def S_OR_B64_vi            : SOP2_Real_vi <0x0f, S_OR_B64>;
def S_XOR_B32_vi           : SOP2_Real_vi <0x10, S_XOR_B32>;
def S_XOR_B64_vi           : SOP2_Real_vi <0x11, S_XOR_B64>;
def S_ANDN2_B32_vi         : SOP2_Real_vi <0x12, S_ANDN2_B32>;
def S_ANDN2_B64_vi         : SOP2_Real_vi <0x13, S_ANDN2_B64>;
def S_ORN2_B32_vi          : SOP2_Real_vi <0x14, S_ORN2_B32>;
def S_ORN2_B64_vi          : SOP2_Real_vi <0x15, S_ORN2_B64>;
def S_NAND_B32_vi          : SOP2_Real_vi <0x16, S_NAND_B32>;
def S_NAND_B64_vi          : SOP2_Real_vi <0x17, S_NAND_B64>;
def S_NOR_B32_vi           : SOP2_Real_vi <0x18, S_NOR_B32>;
def S_NOR_B64_vi           : SOP2_Real_vi <0x19, S_NOR_B64>;
def S_XNOR_B32_vi          : SOP2_Real_vi <0x1a, S_XNOR_B32>;
def S_XNOR_B64_vi          : SOP2_Real_vi <0x1b, S_XNOR_B64>;
def S_LSHL_B32_vi          : SOP2_Real_vi <0x1c, S_LSHL_B32>;
def S_LSHL_B64_vi          : SOP2_Real_vi <0x1d, S_LSHL_B64>;
def S_LSHR_B32_vi          : SOP2_Real_vi <0x1e, S_LSHR_B32>;
def S_LSHR_B64_vi          : SOP2_Real_vi <0x1f, S_LSHR_B64>;
def S_ASHR_I32_vi          : SOP2_Real_vi <0x20, S_ASHR_I32>;
def S_ASHR_I64_vi          : SOP2_Real_vi <0x21, S_ASHR_I64>;
def S_BFM_B32_vi           : SOP2_Real_vi <0x22, S_BFM_B32>;
def S_BFM_B64_vi           : SOP2_Real_vi <0x23, S_BFM_B64>;
def S_MUL_I32_vi           : SOP2_Real_vi <0x24, S_MUL_I32>;
def S_BFE_U32_vi           : SOP2_Real_vi <0x25, S_BFE_U32>;
def S_BFE_I32_vi           : SOP2_Real_vi <0x26, S_BFE_I32>;
def S_BFE_U64_vi           : SOP2_Real_vi <0x27, S_BFE_U64>;
def S_BFE_I64_vi           : SOP2_Real_vi <0x28, S_BFE_I64>;
def S_CBRANCH_G_FORK_vi    : SOP2_Real_vi <0x29, S_CBRANCH_G_FORK>;
def S_ABSDIFF_I32_vi       : SOP2_Real_vi <0x2a, S_ABSDIFF_I32>;
def S_PACK_LL_B32_B16_vi   : SOP2_Real_vi <0x32, S_PACK_LL_B32_B16>;
def S_PACK_LH_B32_B16_vi   : SOP2_Real_vi <0x33, S_PACK_LH_B32_B16>;
def S_PACK_HH_B32_B16_vi   : SOP2_Real_vi <0x34, S_PACK_HH_B32_B16>;
def S_RFE_RESTORE_B64_vi   : SOP2_Real_vi <0x2b, S_RFE_RESTORE_B64>;

def S_MOVK_I32_vi          : SOPK_Real_vi <0x00, S_MOVK_I32>;
def S_CMOVK_I32_vi         : SOPK_Real_vi <0x01, S_CMOVK_I32>;
def S_CMPK_EQ_I32_vi       : SOPK_Real_vi <0x02, S_CMPK_EQ_I32>;
def S_CMPK_LG_I32_vi       : SOPK_Real_vi <0x03, S_CMPK_LG_I32>;
def S_CMPK_GT_I32_vi       : SOPK_Real_vi <0x04, S_CMPK_GT_I32>;
def S_CMPK_GE_I32_vi       : SOPK_Real_vi <0x05, S_CMPK_GE_I32>;
def S_CMPK_LT_I32_vi       : SOPK_Real_vi <0x06, S_CMPK_LT_I32>;
def S_CMPK_LE_I32_vi       : SOPK_Real_vi <0x07, S_CMPK_LE_I32>;
def S_CMPK_EQ_U32_vi       : SOPK_Real_vi <0x08, S_CMPK_EQ_U32>;
def S_CMPK_LG_U32_vi       : SOPK_Real_vi <0x09, S_CMPK_LG_U32>;
def S_CMPK_GT_U32_vi       : SOPK_Real_vi <0x0A, S_CMPK_GT_U32>;
def S_CMPK_GE_U32_vi       : SOPK_Real_vi <0x0B, S_CMPK_GE_U32>;
def S_CMPK_LT_U32_vi       : SOPK_Real_vi <0x0C, S_CMPK_LT_U32>;
def S_CMPK_LE_U32_vi       : SOPK_Real_vi <0x0D, S_CMPK_LE_U32>;
def S_ADDK_I32_vi          : SOPK_Real_vi <0x0E, S_ADDK_I32>;
def S_MULK_I32_vi          : SOPK_Real_vi <0x0F, S_MULK_I32>;
def S_CBRANCH_I_FORK_vi    : SOPK_Real_vi <0x10, S_CBRANCH_I_FORK>;
def S_GETREG_B32_vi        : SOPK_Real_vi <0x11, S_GETREG_B32>;
def S_SETREG_B32_vi        : SOPK_Real_vi <0x12, S_SETREG_B32>;
//def S_GETREG_REGRD_B32_vi  : SOPK_Real_vi <0x13, S_GETREG_REGRD_B32>; // see pseudo for comments
def S_SETREG_IMM32_B32_vi  : SOPK_Real64<0x14, S_SETREG_IMM32_B32>,
                             Select_vi<S_SETREG_IMM32_B32.Mnemonic>;

def S_CALL_B64_vi          : SOPK_Real_vi <0x15, S_CALL_B64>;

//===----------------------------------------------------------------------===//
// SOP1 - GFX9.
//===----------------------------------------------------------------------===//

def S_ANDN1_SAVEEXEC_B64_vi   : SOP1_Real_vi<0x33, S_ANDN1_SAVEEXEC_B64>;
def S_ORN1_SAVEEXEC_B64_vi    : SOP1_Real_vi<0x34, S_ORN1_SAVEEXEC_B64>;
def S_ANDN1_WREXEC_B64_vi     : SOP1_Real_vi<0x35, S_ANDN1_WREXEC_B64>;
def S_ANDN2_WREXEC_B64_vi     : SOP1_Real_vi<0x36, S_ANDN2_WREXEC_B64>;
def S_BITREPLICATE_B64_B32_vi : SOP1_Real_vi<0x37, S_BITREPLICATE_B64_B32>;

//===----------------------------------------------------------------------===//
// SOP2 - GFX9.
//===----------------------------------------------------------------------===//

def S_LSHL1_ADD_U32_vi   : SOP2_Real_vi<0x2e, S_LSHL1_ADD_U32>;
def S_LSHL2_ADD_U32_vi   : SOP2_Real_vi<0x2f, S_LSHL2_ADD_U32>;
def S_LSHL3_ADD_U32_vi   : SOP2_Real_vi<0x30, S_LSHL3_ADD_U32>;
def S_LSHL4_ADD_U32_vi   : SOP2_Real_vi<0x31, S_LSHL4_ADD_U32>;
def S_MUL_HI_U32_vi      : SOP2_Real_vi<0x2c, S_MUL_HI_U32>;
def S_MUL_HI_I32_vi      : SOP2_Real_vi<0x2d, S_MUL_HI_I32>;
