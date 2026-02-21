//==- ConnexInstrInfoVec.td -  Scalar Target Description for Connex Target -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Connex-S vector instructions in TableGen format.
//    The ISA is described in ConnexISA.pdf.
//
//===----------------------------------------------------------------------===//

// From include/llvm/IR/IntrinsicsConnex.td:
//    def int_connex_reduce : Intrinsic<[], [llvm_anyvector_ty], []>;
class RED_1R_DESC_BASE<string instr_asm,
                       RegisterOperand ROWS,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins ROWS:$wl);
  string AsmString = !strconcat(instr_asm,
                                " $wl ; // MSA_1R generic instruction");
  list<dag> Pattern = [(int_connex_reduce ROWS:$wl)];
  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
class RED_H_DESC : RED_1R_DESC_BASE<"RED", VectorHOpnd>;
class RED_H_ENC : MSA_1R_FMT<0b101010110>;
def RED_H: RED_H_ENC, RED_H_DESC;

// NOTE: RED_U, unsigned reduction, is only used for manual/custom ISel
class RED_U_1R_DESC_BASE<string instr_asm,
                       RegisterOperand ROWS,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins ROWS:$wl);
  string AsmString = !strconcat(instr_asm,
                                " $wl ; // MSA_1R generic instruction");
  list<dag> Pattern = [];
  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
class RED_U_H_DESC : RED_U_1R_DESC_BASE<"RED_U", VectorHOpnd>;
class RED_U_H_ENC : MSA_1R_FMT<0b100000000>;
// Note: does NOT affect any flags
def RED_U_H: RED_U_H_ENC, RED_U_H_DESC;

let hasSideEffects = 1 in // Inspired from MSP430InstrInfo.td
def NOP : NonImmediateInstruction<0b000000000, (outs), (ins),
                                  "NOP; // NOP : NonImmediateInstruction", []>;
// Note: does NOT affect any flags

// NEW32
class NOP_BITCONVERT_DESC_BASE<string AsmStrInfo,
                               RegisterOperand ROWS,
                               RegisterOperand ROWD,
                               InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$wl);
  string AsmString = !strconcat(
                       "// NOP used for vector bitconvert $wl -> $wd . ",
                       AsmStrInfo);
  list<dag> Pattern = [];

  // Inspired from include/llvm/Target/Target.td:
  //   <<OperandConstraint, e.g. $src = $dst.>>
  string Constraints = "$wl = $wd";

  /*
  bit hasSideEffects = 1;
  // We need to put this since we don't specify a DAG pattern in Pattern
  */
  InstrItinClass Itinerary = itin;
}
class NOP_BITCONVERT_ENC : MSA_2R_FMT<0b000000000>; // MSAInst
class NOP_BITCONVERT_HW_DESC : NOP_BITCONVERT_DESC_BASE<"v8i16 to v4i32",
                                                        VectorHOpnd,
                                                        VectorHOpnd>;
def NOP_BITCONVERT_HW: NOP_BITCONVERT_ENC, NOP_BITCONVERT_HW_DESC;
//
class NOP_BITCONVERT_WH_DESC : NOP_BITCONVERT_DESC_BASE<"v4i32 to v8i16",
                                                        VectorHOpnd,
                                                        VectorHOpnd>;
def NOP_BITCONVERT_WH: NOP_BITCONVERT_ENC, NOP_BITCONVERT_WH_DESC;
//
class NOP_BITCONVERT_HH_DESC : NOP_BITCONVERT_DESC_BASE<"v8i16 to v8i16",
                                                        VectorHOpnd,
                                                        VectorHOpnd>;
def NOP_BITCONVERT_HH: NOP_BITCONVERT_ENC, NOP_BITCONVERT_HH_DESC;


class NOP_BOGUS_DESC_BASE<InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  string AsmString = "// NOP used for ConnexInstrInfo::copyPhysReg()";
  list<dag> Pattern = [];

  // We put this normally since we don't specify a DAG pattern in Pattern
  //bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}
class NOP_BOGUS_ENC : MSAInst; // TODO: add encoding;
def NOP_BOGUS: NOP_BOGUS_ENC, NOP_BOGUS_DESC_BASE;

class NOP_SPECIAL_DESC_BASE<RegisterOperand ROWS,
                            RegisterOperand ROWD_TIED = ROWS,
                            InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD_TIED:$wdTied);
  dag InOperandList = (ins ROWS:$ws);
  string AsmString =
                 "NOP ; // NOP_SPECIAL_DESC_BASE (ws = $ws, wdTied = $wdTied )";
  list<dag> Pattern = [];

  string Constraints = "$ws = $wdTied";

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}
class NOP_SPECIAL_DESC : NOP_SPECIAL_DESC_BASE<VectorHOpnd>;
class NOP_SPECIAL_ENC : MSAInst; // TODO: add encoding;
def NOP_SPECIAL: NOP_SPECIAL_ENC, NOP_SPECIAL_DESC;








// From MipsMSAInstrInfo.td
class IsCommutable {
  bit isCommutable = 1;
}



class MSA_2R_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       RegisterOperand ROWD, RegisterOperand ROWS = ROWD,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$wl);
  // INFIX_NOTATION:
  string AsmString = !strconcat("$wd = ",
                            !strconcat(instr_asm,
                                       " $wl ; // MSA_2R generic instruction"));
  list<dag> Pattern = [(set ROWD:$wd, (OpNode ROWS:$wl))];
  InstrItinClass Itinerary = itin;
}

class MSA_3R_DESC_BASE<string instr_asm,
                       SDPatternOperator OpNode,
                       RegisterOperand ROWD,
                       RegisterOperand ROWR = ROWD,
                       RegisterOperand ROWL = ROWD,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl);
  // INFIX_NOTATION:
  string AsmString = !strconcat("$wd = $wr ",
                            !strconcat(instr_asm,
                                       " $wl ; // MSA_3R generic instruction"));
  list<dag> Pattern = [(set ROWD:$wd, (OpNode ROWR:$wr, ROWL:$wl))];
  InstrItinClass Itinerary = itin;
}

class MSA_3R_PREFIX_DESC_BASE<string instr_asm,
                              SDPatternOperator OpNode,
                              RegisterOperand ROWD,
                              RegisterOperand ROWR = ROWD,
                              RegisterOperand ROWL = ROWD,
                              InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl);
  string AsmString = !strconcat(!strconcat("$wd = ", instr_asm),
                        "( $wr , $wl ) ; // MSA_3R_PREFIX generic instruction");
  list<dag> Pattern = [(set ROWD:$wd, (OpNode ROWR:$wr, ROWL:$wl))];
  //bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}


///////////////////////////////////////////////////////////////////////////////
//////////////////////BEGIN (i)read/(i)write //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//include "ConnexInstrInfoVec_READ_WRITE.td"
class LD_H_DESC : LD_DESC_BASE<load, v8i16, VectorHOpnd>; // ldvd OR iread
// NEW32
class LD_W_ENC : MSA_3R_FMT<0b000000000>;
                               // NOTE: this is not a real Connex-S instruction
class LD_H_ENC : MSA_1R_FMT_dest_imm<0b110100>;
// Note: does NOT affect any flags
def   LD_H: LD_H_ENC, LD_H_DESC;

class ST_H_DESC : ST_DESC_BASE<store, v8i16, VectorHOpnd>; // stvd OR iwrite
class ST_H_ENC : MSA_1R_FMT_left_imm<0b110010>;
// Note: does NOT affect any flags
def   ST_H: ST_H_ENC, ST_H_DESC;

class LD_INDIRECT_MASKED_H_DESC : LD_INDIRECT_MASKED_DESC_BASE<VectorHOpnd>;
class LD_INDIRECT_MASKED_H_ENC : MSA_3R_FMT2<0b100100100>;
// Note: does NOT affect any flags
def LD_INDIRECT_MASKED_H: LD_INDIRECT_MASKED_H_ENC, LD_INDIRECT_MASKED_H_DESC;

class LD_INDIRECT_W_DESC : LD_INDIRECT_DESC_BASE<VectorHOpnd>;
class LD_INDIRECT_W_ENC : MSA_3R_FMT2<0b100100100>;
// Note: does NOT affect any flags
def LD_INDIRECT_W: LD_INDIRECT_W_ENC, LD_INDIRECT_W_DESC;
//
class LD_INDIRECT_H_DESC : LD_INDIRECT_DESC_BASE<VectorHOpnd>;
class LD_INDIRECT_H_ENC : MSA_3R_FMT2<0b100100100>;
// Note: does NOT affect any flags
def LD_INDIRECT_H: LD_INDIRECT_H_ENC, LD_INDIRECT_H_DESC;

class ST_INDIRECT_W_DESC : ST_INDIRECT_DESC_BASE<VectorHOpnd>;
class ST_INDIRECT_W_ENC : MSA_2R_FMT2<0b101001011>;
// Note: affects flags: like Sub, Lt, Eq
def ST_INDIRECT_W: ST_INDIRECT_W_ENC, ST_INDIRECT_W_DESC;
//
class ST_INDIRECT_H_DESC : ST_INDIRECT_DESC_BASE<VectorHOpnd>;
class ST_INDIRECT_H_ENC : MSA_2R_FMT2<0b101001011>;
// Note: affects flags: like Sub, Lt, Eq
def ST_INDIRECT_H: ST_INDIRECT_H_ENC, ST_INDIRECT_H_DESC;

class ST_INDIRECT_MASKED_H_DESC : ST_INDIRECT_MASKED_DESC_BASE<VectorHOpnd>;
class ST_INDIRECT_MASKED_H_ENC : MSA_3R_FMT<0b100010100>;
// Note: affects flags: like Sub, Lt, Eq
def ST_INDIRECT_MASKED_H: ST_INDIRECT_MASKED_H_ENC, ST_INDIRECT_MASKED_H_DESC;


// Fill (load of spilling) - to recognize it's a fill in ConnexTargetMachine.cpp
def LD_FILL_H: LD_H_ENC, LD_H_DESC;
//
// For the spill - to recognize it is a spill in ConnexTargetMachine.cpp
def ST_SPILL_H: ST_H_ENC, ST_H_DESC;

///////////////////////////////////////////////////////////////////////////////
///////////////////////END (i)read/(i)write ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
///////////////BEGIN REPEAT////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// VERY IMPORTANT: we use the OPINCAA name instructions REPEAT and END_REPEAT
//     because REPEAT is transformed by the OPINCAA assembler in
//     2 Connex SETLC assembly instructions due to a hardware workaround.
//   Also, END_REPEAT is transformed by OPINCAA in IJMPNZDEC and NOP.


class REPEAT_DESC_BASE<InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins i64imm:$imm);

  string AsmString = "REPEAT_TIMES($imm ); // ; REP_1R";

  list<dag> Pattern = [(int_connex_repeat_x_times imm:$imm)];
  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
class REPEAT_DESC : REPEAT_DESC_BASE;
class REPEAT_ENC :  Connex_IMM16_FMT2<0b010101>;
def REPEAT: REPEAT_ENC, REPEAT_DESC;


class REPEAT_DESC_BASE_SYM_IMM<InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);

  string AsmString =
            "REPEAT_TIMES( // (fake but necessary ; ) REPEAT_DESC_BASE_SYM_IMM";
  /* IMPORTANT: this instruction does NOT have a pattern here, but is matched
     using custom matching code in ConnexISelDAGToDAG.cpp,
     void ConnexDAGToDAGISel::Select(SDNode *Node) */
  list<dag> Pattern = [];
  bit hasSideEffects = 1;
         // We need to put this since we don't specify a DAG pattern in Pattern
  InstrItinClass Itinerary = itin;
}
class REPEAT_DESC_SYM_IMM : REPEAT_DESC_BASE_SYM_IMM;
class REPEAT_ENC_SYM_IMM :  MSAInst; // small-MEGA TODO: add encoding;
def REPEAT_SYM_IMM: REPEAT_ENC_SYM_IMM, REPEAT_DESC_SYM_IMM;


class END_REPEAT_DESC_BASE<InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);

  string AsmString = "END_REPEAT; // END_REPEAT_DESC_BASE";
  list<dag> Pattern = [(int_connex_end_repeat)]; // small-TODO: maybe I should put brcond targetAddress
  bit hasSideEffects = 1;
  bit isBranch       = 1; // small-TODO: Is this instruction a branch instruction?
  InstrItinClass Itinerary = itin;
}
class END_REPEAT_DESC : END_REPEAT_DESC_BASE;
class END_REPEAT_ENC :  MSAInst;
def END_REPEAT: END_REPEAT_ENC, END_REPEAT_DESC;

///////////////////////////////////////////////////////////////////////////////
//////////////END REPEAT //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// These are taken from Mips MipsMSAInstrInfo.td.
// See also http://llvm.org/docs/LangRef.html#shl-instruction
class SHLV_H_DESC : MSA_3R_DESC_BASE<"<<", shl, VectorHOpnd>;
class SHLV_H_ENC : MSA_3R_FMT<0b101000000>;
// Note: affects flags: like Add, Lt, Eq
def SHLV_H : SHLV_H_ENC, SHLV_H_DESC;

// srl is defined in [LLVM]/llvm/include/llvm/Target/TargetSelectionDAG.td
class SHRV_H_DESC : MSA_3R_DESC_BASE<">>", srl, VectorHOpnd>;
class SHRV_H_ENC : MSA_3R_FMT<0b101010000>;
// Note: affects flags: like Sub, Lt, Eq
def SHRV_H : SHRV_H_ENC, SHRV_H_DESC;

// sra is defined in [LLVM]/llvm/include/llvm/Target/TargetSelectionDAG.td
class SHRAV_H_DESC : MSA_3R_PREFIX_DESC_BASE<"SHRA", sra, VectorHOpnd>;
class SHRAV_H_ENC : MSA_3R_FMT<0b101100000>;
// Note: affects flags: like Addc, Ult, Eq
def SHRAV_H : SHRAV_H_ENC, SHRAV_H_DESC;


class NOT_H_DESC : MSA_2R_DESC_BASE<"~", not, VectorHOpnd>;
class NOT_H_ENC : MSA_2R_FMT<0b101001100>;
// Note: affects in an undefined way the flags
def NOT_H : NOT_H_ENC, NOT_H_DESC;

class ORV_H_DESC : MSA_3R_DESC_BASE<"|", or, VectorHOpnd>, IsCommutable;
class ORV_H_ENC : MSA_3R_FMT<0b101011100>;
// Note: affects flags: like Sub, Lt, Eq
def ORV_H : ORV_H_ENC, ORV_H_DESC;

class ANDV_H_DESC : MSA_3R_DESC_BASE<"&", and, VectorHOpnd>, IsCommutable;
class ANDV_H_ENC : MSA_3R_FMT<0b101101100>;
// Note: affects flags: like Addc, Ult, Eq
def ANDV_H : ANDV_H_ENC, ANDV_H_DESC;

class XORV_H_DESC : MSA_3R_DESC_BASE<"^", xor, VectorHOpnd>, IsCommutable;
class XORV_H_ENC : MSA_3R_FMT<0b101111100>;
// Note: affects flags: like Subc, Ult, Eq
def XORV_H : XORV_H_ENC, XORV_H_DESC;




// Using ctpop intrinsic - just like llvm/lib/Target/AArch64/AArch64InstrInfo.td
class POPCNT_H_DESC : MSA_2R_DESC_BASE<"POPCNT", ctpop, VectorHOpnd>;
class POPCNT_H_ENC : MSA_2R_FMT<0b101110000>;
// Note: does NOT affect any flags
def POPCNT_H : POPCNT_H_ENC, POPCNT_H_DESC;




class MSA_2R_SPECIAL_DESC_BASE<string instr_asm,
                               SDPatternOperator OpNode,
                               RegisterOperand ROWD,
                               RegisterOperand ROWS = ROWD,
                               RegisterOperand ROWS_TIED = ROWD,
                               InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$wl, ROWS_TIED:$wlTied);
  // INFIX_NOTATION:
  string AsmString = !strconcat("$wd = ",
                       !strconcat(instr_asm,
             " $wl ; // MSA_2R_SPECIAL generic instruction (wlTied = $wlTied )")
                       );
  list<dag> Pattern = [];

  // Inspired from include/llvm/Target/Target.td:
  //   <<OperandConstraint, e.g. $src = $dst.>>
  string Constraints = "$wd = $wlTied";

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}

class MSA_3R_SPECIAL_PREFIX_DESC_BASE<string instr_asm,
                                      SDPatternOperator OpNode,
                                      RegisterOperand ROWD,
                                      RegisterOperand ROWR = ROWD,
                                      RegisterOperand ROWL = ROWD,
                                      RegisterOperand ROWR_TIED = ROWD,
                                      InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl, ROWR_TIED:$wrTied);
  string AsmString = !strconcat("$wd = ",
                       !strconcat(instr_asm,
       "($wr, $wl ) ; // MSA_3R_SPECIAL generic instruction (wrTied = $wrTied )"
                       ) );
  list<dag> Pattern = [];

  // Inspired from include/llvm/Target/Target.td:
  //   <<OperandConstraint, e.g. $src = $dst.>>
  string Constraints = "$wd = $wrTied";

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}

class MSA_3R_SPECIAL_DESC_BASE<string instr_asm,
                               SDPatternOperator OpNode,
                               RegisterOperand ROWD,
                               RegisterOperand ROWR = ROWD,
                               RegisterOperand ROWL = ROWD,
                               RegisterOperand ROWR_TIED = ROWD,
                               InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl, ROWR_TIED:$wrTied);
  // INFIX_NOTATION:
  string AsmString = !strconcat("$wd = $wr ",
                       !strconcat(instr_asm,
              " $wl ; // MSA_3R_SPECIAL generic instruction (wrTied = $wrTied )"
                       ) );
  list<dag> Pattern = [];

  // Inspired from llvm/include/llvm/Target/Target.td:
  //   <<OperandConstraint, e.g. $src = $dst.>>
  string Constraints = "$wd = $wrTied";

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}






include "ConnexInstrInfoVecVsplat.td"


///////////////////////////////////////////////////////////////////////////////
////////////////////////BEGIN ISHR/ISHR(A)/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//include "ConnexInstrInfoVec_bit.td"
/* VERY IMPORTANT: even if the _ISHL/_ISHR(A) instructions are
   immediate, they have the 2nd opcode bit (also called the IMM bit)
   set to 0, as we can see in the Instruction format diagram above.
  So they are of type INSTRUCTION_TYPE_NO_IMMEDIATE
    - the immediate value is stored in the 5 bits of the
    right register, since it is enough for the delta operand
    of SHIFT operations (normally values 0-16 are enough for
    16-bit registers).

IMPORTANT NOTE: these instructions take an immediate value which is to be
     applied as an operand on a vector binary operand.
    Hence, we must make this immediate operand a vector immediate operand
       (and not a scalar immediate operand as we might be tempted to put it,
       imm/i64imm),
      otherwise TableGen will complain "Could not infer all types in pattern!"
        at the Pattern(s) below.
*/
class Connex_IMM_SHIFT_DESC_BASE<string instr_asm,
                                 SDPatternOperator OpNode,
                                 SplatComplexPattern SplatImm,
                                 RegisterOperand ROWD,
                                 RegisterOperand ROWS = ROWD,
                                 InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$wl, SplatImm.OpClass:$wr);
  // Note: For ISHL, ISHR(A) the immediate is actually the wr (Connex right)
  //       register

  string AsmString = !strconcat("$wd = ",
                       !strconcat(instr_asm,
                         "($wl, $wr ); // Connex_IMM_SHIFT generic instruction"
                       ));

  list<dag> Pattern = [(set ROWD:$wd, (OpNode ROWS:$wl, SplatImm:$wr))];

  InstrItinClass Itinerary = itin;
}


class ISHLV_H_DESC : Connex_IMM_SHIFT_DESC_BASE<"ISHL", shl, vsplati16_uimm5,
                                                                   VectorHOpnd>;
class ISHLV_H_ENC : Connex_NI_FMT_ISHV<0b101000001>;
def ISHLV_H : ISHLV_H_ENC, ISHLV_H_DESC;

class ISHRAV_H_DESC : Connex_IMM_SHIFT_DESC_BASE<"ISHRA", sra, vsplati16_uimm5,
                                                                   VectorHOpnd>;
class ISHRAV_H_ENC : Connex_NI_FMT_ISHV<0b101100001>;
def ISHRAV_H : ISHRAV_H_ENC, ISHRAV_H_DESC;

class ISHRV_H_DESC : Connex_IMM_SHIFT_DESC_BASE<"ISHR", srl, vsplati16_uimm5,
                                                                   VectorHOpnd>;
class ISHRV_H_ENC : Connex_NI_FMT_ISHV<0b101010001>;
def ISHRV_H : ISHRV_H_ENC, ISHRV_H_DESC;

class MSA_3R_PREFIX_SPECIAL_DESC_BASE<string instr_asm,
                                      SDPatternOperator OpNode,
                                      RegisterOperand ROWD,
                                      RegisterOperand ROWR = ROWD,
                                      RegisterOperand ROWL = ROWD,
                                      RegisterOperand ROWR_TIED = ROWD,
                                      InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl, ROWR_TIED:$wrTied);
  string AsmString = !strconcat(
                       !strconcat("$wd = ", instr_asm),
     "($wr , $wl ) ; // MSA_3R_PREFIX_SPECIAL generic instr (wrTied = $wrTied )"
                       );
  list<dag> Pattern = [];

  string Constraints = "$wd = $wrTied";

  // We need to put this to prevent llc (SelectionDAG, DAGCombiner, MachineCSE)
  //   to apply CSE and other optimizations on the predicated instructions
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}

class SHLV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"<<", shl, VectorHOpnd>;
class SHLV_SPECIAL_H_ENC : MSA_3R_FMT<0b101000000>;
// Note: affects flags: like Add, Lt, Eq
def SHLV_SPECIAL_H : SHLV_SPECIAL_H_ENC, SHLV_SPECIAL_H_DESC;

class SHRV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<">>", srl, VectorHOpnd>;
class SHRV_SPECIAL_H_ENC : MSA_3R_FMT<0b101010000>;
// Note: affects flags: like Sub, Lt, Eq
def SHRV_SPECIAL_H : SHRV_SPECIAL_H_ENC, SHRV_SPECIAL_H_DESC;

class SHRAV_SPECIAL_H_DESC : MSA_3R_PREFIX_SPECIAL_DESC_BASE<"SHRA", sra,
                                                             VectorHOpnd>;
class SHRAV_SPECIAL_H_ENC : MSA_3R_FMT<0b101100000>;
// Note: affects flags: like Addc, Ult, Eq
def SHRAV_SPECIAL_H : SHRAV_SPECIAL_H_ENC, SHRAV_SPECIAL_H_DESC;

class NOT_SPECIAL_H_DESC : MSA_2R_SPECIAL_DESC_BASE<"~", not, VectorHOpnd>;
class NOT_SPECIAL_H_ENC : MSA_2R_FMT<0b101001100>;
// Note: affects in an undefined way the flags
def NOT_SPECIAL_H : NOT_SPECIAL_H_ENC, NOT_SPECIAL_H_DESC;

// We use ORV_SPECIAL_H for codegen of VSELECT
class ORV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"|", or, VectorHOpnd>,
                                                                IsCommutable;
class ORV_SPECIAL_H_ENC : MSA_3R_FMT<0b101011100>;
// Note: affects flags: like Addc, Ult, Eq
def ORV_SPECIAL_H : ORV_SPECIAL_H_ENC, ORV_SPECIAL_H_DESC;

class ANDV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"&", and, VectorHOpnd>,
                                                                IsCommutable;
class ANDV_SPECIAL_H_ENC : MSA_3R_FMT<0b111111101>;
// Note: affects flags: like Addc, Ult, Eq
def ANDV_SPECIAL_H : ANDV_SPECIAL_H_ENC, ANDV_SPECIAL_H_DESC;

class XORV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"^", xor, VectorHOpnd>,
                                                                IsCommutable;
class XORV_SPECIAL_H_ENC : MSA_3R_FMT<0b101111100>;
// Note: affects flags: like Subc, Ult, Eq
def XORV_SPECIAL_H : XORV_SPECIAL_H_ENC, XORV_SPECIAL_H_DESC;



// Using ctpop intrinsic - just like llvm/lib/Target/AArch64/AArch64InstrInfo.td
class POPCNT_SPECIAL_H_DESC : MSA_2R_SPECIAL_DESC_BASE<"POPCNT", ctpop,
                                                                VectorHOpnd>;
class POPCNT_SPECIAL_H_ENC : MSA_2R_FMT<0b101110000>;
// Note: does NOT affect any flags
def POPCNT_SPECIAL_H : POPCNT_SPECIAL_H_ENC, POPCNT_SPECIAL_H_DESC;



class POWER_CELL_H_DESC_BASE<string instr_asm,
                             InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins);
  string AsmString = !strconcat(instr_asm, "; // POWER_CELL_H_DESC ");
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
class DISABLE_CELL_H_DESC : POWER_CELL_H_DESC_BASE<"DISABLE_CELL">;
class DISABLE_CELL_H_ENC : MSAInst; // small-MEGA TODO: add encoding
def DISABLE_CELL_H: DISABLE_CELL_H_ENC, DISABLE_CELL_H_DESC;
//
class ENABLE_ALL_CELLS_H_DESC : POWER_CELL_H_DESC_BASE<"ENABLE_ALL_CELLS">;
class ENABLE_ALL_CELLS_H_ENC : MSAInst; // small-MEGA TODO: add encoding
def ENABLE_ALL_CELLS_H: ENABLE_ALL_CELLS_H_ENC, ENABLE_ALL_CELLS_H_DESC;




///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////








//include "ConnexInstrInfoVec_vsetcc_vselect.td"
// Inspired from [LLVM]/llvm/lib/Target/Mips/MipsMSAInstrInfo.td, from:
// From MipsMSAInstrInfo.td
// 1 result, 3 input operands
def SDT_VSetCC : SDTypeProfile<1, 3, [
                                      SDTCisVec<0>,
                                      SDTCisVec<1>,
                                      SDTCisSameAs<0, 1>,
                                      SDTCisVT<3, OtherVT>
                                    ]>;
def vsetcc : SDNode<"ISD::SETCC", SDT_VSetCC>;

class vsetcc_type<ValueType ResTy, CondCode CC> :
  PatFrag<(ops node:$lhs, node:$rhs),
          (ResTy (vsetcc node:$lhs, node:$rhs, CC))>;

def vseteq_v128i16  : vsetcc_type<v8i16, SETEQ>;
def vsetle_v128i16  : vsetcc_type<v8i16, SETLE>;
def vsetlt_v128i16  : vsetcc_type<v8i16, SETLT>;
def vsetule_v128i16 : vsetcc_type<v8i16, SETULE>;
def vsetult_v128i16 : vsetcc_type<v8i16, SETULT>;









class Connex_IMM_SHIFT_SPECIAL_DESC_BASE<string instr_asm,
                                         SDPatternOperator OpNode,
                                         SplatComplexPattern SplatImm,
                                         RegisterOperand ROWD,
                                         RegisterOperand ROWS = ROWD,
                                         RegisterOperand ROWS_TIED = ROWD,
                                         InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$wl, SplatImm.OpClass:$wr, ROWS_TIED:$wsTied);
  string AsmString = !strconcat("$wd = ",
                       !strconcat(instr_asm,
   "($wl, $wr ); // Connex_IMM_SHIFT_SPECIAL generic instr (wsTied = $wsTied )"
                       ));

  list<dag> Pattern = [];

  string Constraints = "$wd = $wsTied";
            // Inspired from Target.td: <<OperandConstraint, e.g. $src = $dst.>>
  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern

  InstrItinClass Itinerary = itin;
}

class ISHLV_SPECIAL_H_DESC : Connex_IMM_SHIFT_SPECIAL_DESC_BASE<"ISHL", shl,
                                                                vsplati16_uimm5,
                                                                VectorHOpnd>;
def ISHLV_SPECIAL_H : ISHLV_H_ENC, ISHLV_SPECIAL_H_DESC;

class ISHRV_SPECIAL_H_DESC : Connex_IMM_SHIFT_SPECIAL_DESC_BASE<"ISHR", srl,
                                                                vsplati16_uimm5,
                                                                VectorHOpnd>;
def ISHRV_SPECIAL_H : ISHRV_H_ENC, ISHRV_SPECIAL_H_DESC;
///////////////////////////////////////////////////////////////////////////////
//////////////////////////END ISHR/ISHR(A)/////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////






//include "ConnexInstrInfoVec_vinsert.td"
// These are for ConnexTargetLowering::LowerEXTRACT_VECTOR_ELT and for
//   BUILD_VECTOR
// Inspired from llvm/lib/Target/Mips/MipsMSAInstrInfo.td
def MipsVExtractSExt : SDNode<"ConnexISD::VEXTRACT_SEXT_ELT",
    SDTypeProfile<1, 3, [SDTCisPtrTy<2>]>, []>;
def MipsVExtractZExt : SDNode<"ConnexISD::VEXTRACT_ZEXT_ELT",
    SDTypeProfile<1, 3, [SDTCisPtrTy<2>]>, []>;


// Inspired from MipsMSAInstrInfo.td
def immZExt1Ptr : ImmLeaf<iPTR, [{return isUInt<1>(Imm);}]>;
def immZExt2Ptr : ImmLeaf<iPTR, [{return isUInt<2>(Imm);}]>;
def immZExt3Ptr : ImmLeaf<iPTR, [{return isUInt<3>(Imm);}]>;
def immZExt4Ptr : ImmLeaf<iPTR, [{return isUInt<4>(Imm);}]>;


// Pattern fragments
def vextract_sext_i8  : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractSExt node:$vec, node:$idx, i8)>;
def vextract_sext_i16 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractSExt node:$vec, node:$idx, i16)>;
def vextract_sext_i32 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractSExt node:$vec, node:$idx, i32)>;
def vextract_sext_i64 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractSExt node:$vec, node:$idx, i64)>;

def vextract_zext_i8  : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractZExt node:$vec, node:$idx, i8)>;
def vextract_zext_i16 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractZExt node:$vec, node:$idx, i16)>;
def vextract_zext_i32 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractZExt node:$vec, node:$idx, i32)>;
def vextract_zext_i64 : PatFrag<(ops node:$vec, node:$idx),
                                (MipsVExtractZExt node:$vec, node:$idx, i64)>;

def vinsert_v8i16 : PatFrag<(ops node:$vec, node:$val, node:$idx),
    (v8i16 (vector_insert node:$vec, node:$val, node:$idx))>;












// From MipsDSPInstrInfo.td
def immZExt1 : ImmLeaf<i32, [{return isUInt<1>(Imm);}]>;




class EQ_H_DESC : MSA_3R_DESC_BASE<"==", vseteq_v128i16, VectorHOpnd>,
                                   IsCommutable;
class EQ_H_ENC : MSA_3R_FMT<0b101001000>;
let isCompare       = 1 in // (Is this instruction a comparison instruction?)
// We model the fact it changes the flags of Connex
//   (hasSideEffects - The instruction has side effects that are not captured
//    by any operands of the instruction or other flags.)
let  hasSideEffects = 1 in
def EQ_H : EQ_H_ENC, EQ_H_DESC;

class LT_H_DESC : MSA_3R_DESC_BASE<"<", setlt, VectorHOpnd>;
class LT_H_ENC : MSA_3R_FMT<0b101011000>;
let isCompare       = 1 in // (Is this instruction a comparison instruction?)
let  hasSideEffects = 1 in
def LT_H : LT_H_ENC, LT_H_DESC;

class ULT_H_DESC : MSA_3R_PREFIX_DESC_BASE<"ULT", setult, VectorHOpnd>;
class ULT_H_ENC : MSA_3R_FMT<0b101101000>;
let isCompare       = 1 in
let  hasSideEffects = 1 in
def ULT_H : ULT_H_ENC, ULT_H_DESC;

class EQ_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"==", vseteq_v128i16,
                                                   VectorHOpnd>, IsCommutable;
class EQ_SPECIAL_H_ENC : MSA_3R_FMT<0b101001000>;
let isCompare       = 1 in
let  hasSideEffects = 1 in
def EQ_SPECIAL_H : EQ_SPECIAL_H_ENC, EQ_SPECIAL_H_DESC;

class LT_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"<", vsetlt_v128i16,
                                                   VectorHOpnd>;
class LT_SPECIAL_H_ENC : MSA_3R_FMT<0b101011000>;
let isCompare       = 1 in
let  hasSideEffects = 1 in
def LT_SPECIAL_H : LT_SPECIAL_H_ENC, LT_SPECIAL_H_DESC;

class ULT_SPECIAL_H_DESC : MSA_3R_SPECIAL_PREFIX_DESC_BASE<"ULT",
                                                           vsetult_v128i16,
                                                           VectorHOpnd>;
class ULT_SPECIAL_H_ENC : MSA_3R_FMT<0b101101000>;
let isCompare       = 1 in
let  hasSideEffects = 1 in
def ULT_SPECIAL_H : ULT_SPECIAL_H_ENC, ULT_SPECIAL_H_DESC;

///////////////////////////////////////////////////////////////////////////////
//////////BEGIN VLOAD, LDIX, LDSH (load INDEX or from SHIFT_REG)///////////////
///////////////////////////////////////////////////////////////////////////////
//include "ConnexInstrInfoVec_VLOAD_LDIX_LDSH.td"

/* IMPORTANT NOTE: these instructions take an immediate value which is to be
     applied as an operand on a vector binary operand.
    Hence, we must make this immediate operand a vector immediate operand
       (and not a scalar immediate operand as we might be tempted to put it,
       imm/i64imm),
      otherwise TableGen will complain "Could not infer all types in pattern!"
        at the Pattern(s) below.
*/

class MSA_I16_LDI_DESC_BASE<string instr_asm,
                            SplatComplexPattern SplatImm,
                            RegisterOperand ROWD,
                            InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins SplatImm.OpClass:$imm);
  string AsmString = "$wd = $imm ; // MSA_I16";

  /* IMPORTANT: this instruction does NOT have a pattern here, but is matched
     using custom matching code in ConnexISelDAGToDAG.cpp,
     void ConnexDAGToDAGISel::Select(SDNode *Node) */
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern
  // bit isPredicable = 1;
          // Is this instruction predicable?
  InstrItinClass Itinerary = itin;
}

class VLOAD_H_DESC: MSA_I16_LDI_DESC_BASE<"VLOAD", vsplati16_simm16,
                                          VectorHOpnd>;
class VLOAD_H_ENC : Connex_IMM16_FMT3<0b110101>;
// Note: does NOT affect any flags
// let isPredicable = 1 in
let hasSideEffects = 0 in
def VLOAD_H : VLOAD_H_ENC, VLOAD_H_DESC;


class VLOAD_W_DESC : MSA_I16_LDI_DESC_BASE<"VLOAD_W_TODO",
                                           vsplati16_simm16, // TODO: Should be bigger
                                           VectorHOpnd>;
class VLOAD_W_ENC : Connex_IMM16_FMT3<0b110101>;
    // IMPORTANT Note: the opcode is NOT correct since this is NOT an i16 native
    //   Connex-S instruction
//class VLOAD_W_ENC  : MSA_2R_FMT<0b110110001>;
def VLOAD_W : VLOAD_W_ENC, VLOAD_W_DESC;



/* This is a bogus VLOAD to avoid initializing registers in
     Select*_OpincaaCodeGen.cpp.
   This allows to have predicated instructions refer to tied-to
     constraints to the nodes using this class (destination registers of
     predicated instr)
   without initializing the respective dest registers, since it's not necessary.
*/
class MSA_I16_LDI_BOGUS_DESC_BASE<string instr_asm,
                                  SplatComplexPattern SplatImm,
                                  RegisterOperand ROWD,
                                  InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins SplatImm.OpClass:$imm);
  string AsmString = "// BOGUS: $wd = $imm ; // MSA_I16";
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
    // We need to put this since we don't specify a DAG pattern in Pattern
  // bit isPredicable = 1;
    // Is this instruction predicable?
  InstrItinClass Itinerary = itin;
}
//
class VLOAD_BOGUS_H_DESC : MSA_I16_LDI_BOGUS_DESC_BASE<"VLOAD_BOGUS",
                                                       vsplati16_simm16,
                                                       VectorHOpnd>;
// let isPredicable = 1 in
def VLOAD_BOGUS_H : VLOAD_H_ENC, VLOAD_BOGUS_H_DESC;

// Used for special case of REDUCE_i32, etc
class MSA_I16_LDI_SPECIAL_DESC_BASE<string instr_asm,
                                    SplatComplexPattern SplatImm,
                                    RegisterOperand ROWD,
                                    RegisterOperand ROWR=ROWD,
                                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins SplatImm.OpClass:$imm, ROWR:$wrTied);
  string AsmString = "$wd = $imm ; // MSA_I16 (special input wd = $wrTied )";

  /* IMPORTANT: this instruction does NOT have a pattern here, but is matched
     using custom matching code in ConnexISelDAGToDAG.cpp,
     void ConnexDAGToDAGISel::Select(SDNode *Node) */
  list<dag> Pattern = [];

  string Constraints = "$wrTied = $wd";
            // Inspired from Target.td: <<OperandConstraint, e.g. $src = $dst.>>

  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern
  //bit isPredicable = 1; // Is this instruction predicable?
  InstrItinClass Itinerary = itin;
}
class VLOAD_SPECIAL_H_DESC : MSA_I16_LDI_SPECIAL_DESC_BASE<"VLOAD_SPECIAL",
                                                           vsplati16_simm16,
                                                           VectorHOpnd>;
def VLOAD_SPECIAL_H : VLOAD_H_ENC, VLOAD_SPECIAL_H_DESC;

class MSA_I16_LDI_SYM_IMM_DESC_BASE<string instr_asm,
                                    RegisterOperand ROWD,
                                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins);
  string AsmString = "$wd = // (fake but necessary ; ) VLOAD_H_SYM_IMM MSA_I16";

  /* IMPORTANT: this instruction does NOT have a pattern here, but is matched
     using custom matching code in ConnexISelDAGToDAG.cpp,
     void ConnexDAGToDAGISel::Select(SDNode *Node) */
  list<dag> Pattern = [];

  /* This prevents that the pre-RA LICM no longer move the instructions,
     in principle. We need to put this since we don't specify a
     DAG pattern in Pattern. */
  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
class VLOAD_H_DESC_SYM_IMM : MSA_I16_LDI_SYM_IMM_DESC_BASE<"VLOAD",
                                                           VectorHOpnd>;
class VLOAD_H_ENC_SYM_IMM  : Connex_IMM16_SYM_FMT<0b110101>;
def VLOAD_H_SYM_IMM : VLOAD_H_ENC_SYM_IMM, VLOAD_H_DESC_SYM_IMM;

class LDIX_H_DESC : MSA_LDIX_LDSH_MULT_H_DESC_BASE<"INDEX", VectorHOpnd>;
class LDIX_H_ENC  : MSA_1R_FMT_dest<0b100100000>;
// Note: does NOT affect any flags
def LDIX_H : LDIX_H_ENC, LDIX_H_DESC;

class MSA_LDIX_LDSH_MULT_SPECIAL_H_DESC_BASE<string instr_asm,
                                             RegisterOperand ROWD,
                                             RegisterOperand ROWS = ROWD,
                                             InstrItinClass itin = NoItinerary>
                                                                              {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWS:$ws);
  string AsmString = !strconcat(!strconcat("$wd = ", instr_asm),
                                " ; // MSA_LDIX_LDSH_MULT (ws = $ws )"
                               );

  // LDI is matched using custom matching code in MipsSEISelDAGToDAG.cpp
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern

  string Constraints = "$ws = $wd";
            // Inspired from Target.td: <<OperandConstraint, e.g. $src = $dst.>>

  InstrItinClass Itinerary = itin;
}
class LDIX_SPECIAL_H_DESC : MSA_LDIX_LDSH_MULT_SPECIAL_H_DESC_BASE<"INDEX",
                                                                   VectorHOpnd>;
class LDIX_SPECIAL_H_ENC  : MSA_1R_FMT_dest<0b100100000>;
// Note: does NOT affect any flags
def LDIX_SPECIAL_H : LDIX_SPECIAL_H_ENC, LDIX_SPECIAL_H_DESC;

class LDSH_H_DESC : MSA_LDIX_LDSH_MULT_H_DESC_BASE<"SHIFT_REG", VectorHOpnd>;
class LDSH_H_ENC  : MSA_1R_FMT_dest<0b100110000>;
// Note: does NOT affect any flags
def LDSH_H : LDSH_H_ENC, LDSH_H_DESC;

///////////////////////////////////////////////////////////////////////////////
/////////////////END VLOAD, LDIX and LDSH//////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////








// Note: add is defined in [LLVM]/llvm/include/llvm/Target/TargetSelectionDAG.td
class ADDV_H_DESC : MSA_3R_DESC_BASE<"+", add, VectorHOpnd>, IsCommutable;
class ADDV_H_ENC : MSA_3R_FMT<0b101000100>;
// Note: affects flags: like Add, Lt, Eq
def ADDV_H : ADDV_H_ENC, ADDV_H_DESC;

class ADDV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"+", add, VectorHOpnd>,
                                                               IsCommutable;
class ADDV_SPECIAL_H_ENC : MSA_3R_FMT<0b101000101>;
// Note: affects flags: like Add, Lt, Eq
def ADDV_SPECIAL_H : ADDV_SPECIAL_H_ENC, ADDV_SPECIAL_H_DESC;

class SUBV_H_DESC : MSA_3R_DESC_BASE<"-", sub, VectorHOpnd>;
class SUBV_H_ENC : MSA_3R_FMT<0b101010100>;
// Note: affects flags: like Sub, Lt, Eq
def SUBV_H : SUBV_H_ENC, SUBV_H_DESC;

class SUBV_SPECIAL_H_DESC : MSA_3R_SPECIAL_DESC_BASE<"-", sub, VectorHOpnd>;
// Note: affects flags: like Sub, Lt, Eq
def SUBV_SPECIAL_H : SUBV_H_ENC, SUBV_SPECIAL_H_DESC;




/*
The ADDC, SUBC can experience NOT coming immediately after ADD and SUB,
  respectively, even if we make ADDC take a MVT::Glue result from ADD
  (this is not guaranteeing us to have ADDC coming immediately after ADD).
   In fact ADD can be scheduled after ADDC by the post-RA scheduler
      ("********** List Scheduling **********") - see
        DawnCC/35_MatMul_i32/2/STDerr_llc_01;
    (a solution would be to disable running the post-RA scheduler pass,
     but this is not desirable).
Therefore, we add for ADDC and SUBC one more input, coming from ADD and SUB,
  respectively - in order to make (at least) ADDC come after ADD. However, not
  even this should not guarantee that ADDC comes immediately after ADD as it
  should to preserve the Carry flags.
*/
class MSA_3R_PREFIX_CARRY_DESC_BASE<string instr_asm,
                                    RegisterOperand ROWD,
                                    RegisterOperand ROWR = ROWD,
                                    RegisterOperand ROWL = ROWD,
                                    RegisterOperand ROWR_CARRY = ROWD,
                                    InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl, ROWR_CARRY:$wrCarry);
  string AsmString = !strconcat(!strconcat("$wd = ", instr_asm),
     "( $wl , $wr ) ; // MSA_3R prefix Carry instruction (wrCarry = $wrCarry ) "
                     );
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern
  InstrItinClass Itinerary = itin;
}
class ADDCV_H_DESC : MSA_3R_PREFIX_CARRY_DESC_BASE<"ADDC", VectorHOpnd>,
                                                           IsCommutable;
class ADDCV_H_ENC : MSA_3R_FMT<0b101100100>;
// Note: affects flags: like Addc, Ult, Eq
def ADDCV_H : ADDCV_H_ENC, ADDCV_H_DESC;

class SUBCV_H_DESC : MSA_3R_PREFIX_CARRY_DESC_BASE<"SUBC", VectorHOpnd>;
class SUBCV_H_ENC : MSA_3R_FMT<0b101110100>;
// Note: affects flags: like Subc, Ult, Eq
def SUBCV_H : SUBCV_H_ENC, SUBCV_H_DESC;

// Similarly as for MSA_3R_PREFIX_CARRY_DESC_BASE we also
// take the result of the instruction before that sets the Carry flags.
class MSA_3R_PREFIX_CARRY_SPECIAL_DESC_BASE<string instr_asm,
                                            RegisterOperand ROWD,
                                            RegisterOperand ROWR = ROWD,
                                            RegisterOperand ROWL = ROWD,
                                            RegisterOperand ROWR_CARRY = ROWD,
                                            RegisterOperand ROWD_TIED = ROWD,
                                            InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl, ROWR_CARRY:$wrCarry,
                           ROWD_TIED:$wdTied);
  string AsmString = !strconcat(!strconcat("$wd = ", instr_asm),
                                "( $wl , $wr ) ; // MSA_3R prefix Carry special instruction (wrCarry = $wrCarry and wdTied = $wdTied ) "
                     );
  list<dag> Pattern = [];

  string Constraints = "$wd = $wdTied";
    // Inspired from llvm/include/llvm/Target/Target.td:
    //    <<OperandConstraint, e.g. $src = $dst.>>

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;
  InstrItinClass Itinerary = itin;
}
//
class ADDCV_SPECIAL_H_DESC : MSA_3R_PREFIX_CARRY_SPECIAL_DESC_BASE<"ADDC",
                                                                   VectorHOpnd>,
                                                                   IsCommutable;
class ADDCV_SPECIAL_H_ENC : MSA_3R_FMT<0b101110101>;
// Note: affects flags: like Addc, Ult, Eq
def ADDCV_SPECIAL_H : ADDCV_SPECIAL_H_ENC, ADDCV_SPECIAL_H_DESC;
//
class SUBCV_SPECIAL_H_DESC : MSA_3R_PREFIX_CARRY_SPECIAL_DESC_BASE<"SUBC",
                                                                   VectorHOpnd>;
class SUBCV_SPECIAL_H_ENC : MSA_3R_FMT<0b101110101>;
// Note: affects flags: like Subc, Ult, Eq
def SUBCV_SPECIAL_H : SUBCV_SPECIAL_H_ENC, SUBCV_SPECIAL_H_DESC;

class MUL_3R_DESC_BASE<string instr_asm,
                       SDPatternOperator OpNode,
                       RegisterOperand ROWD,
                       RegisterOperand ROWR = ROWD,
                       RegisterOperand ROWL = ROWD,
                       InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWR:$wr, ROWL:$wl);
  // Important Note: we write in string "( $wl )" to be parsed properly
  string AsmString = !strconcat("$wr * ( $wl ); ", "$wd = MULTLO(); // MUL_3R");
  // TODO: Maybe return also the high 32 bits
  list<dag> Pattern = [(set ROWD:$wd, (OpNode ROWR:$wr, ROWL:$wl))];
  InstrItinClass Itinerary = itin;
}

/* Important:
   This generic i16 multiplication with destination is actually NOT a Connex-S
   instruction, but we use it to match the ISD::MUL automatically with
     TableGen, since it is simpler like this.
    MULV is multiplication with src1, src2 and dst
    (actually it has 2 Connex instructions)

  Note that we specify below also the Connex assembly instructions for
    multiplication.
*/
class MUL3RV_H_DESC : MUL_3R_DESC_BASE<"*", mul, VectorHOpnd>, IsCommutable;
class MUL3RV_H_ENC : MSA_3R_FMT<0b111111111>;

def MUL3RV_H : MUL3RV_H_ENC, MUL3RV_H_DESC;

// Now we specify all actual Connex-S instructions for multiplication
class MULT_H_DESC : MSA_RR_INFIX_DESC_BASE<"*", VectorHOpnd>, IsCommutable;
class MULT_H_ENC  : MSA_2R_FMT2<0b100001000>;
// Note: affects flags: like Add, Lt, Eq
def MULT_H : MULT_H_ENC, MULT_H_DESC;

// IMPORTANT NOTE: MULT_U_H, unsigned multiplication, is only used for manual
//     ISel
// TODO: maybe do also MULT_SPECIAL_H_DESC, MULT_U_SPECIAL_H_DESC
class MULT_U_H_DESC : MSA_RR_PREFIX_DESC_BASE<"MULT_U", VectorHOpnd>,
                                                        IsCommutable;
class MULT_U_H_ENC  : MSA_2R_FMT2<0b010110111>;
// Note: affects flags: like Add, Lt, Eq
def MULT_U_H : MULT_U_H_ENC, MULT_U_H_DESC;

class MULTLO_H_DESC : MSA_LDIX_LDSH_MULT_H_DESC_BASE<"MULTLO()", VectorHOpnd>;
class MULTLO_H_ENC  : MSA_1R_FMT_dest<0b100101000>;
// Note: does NOT affect any flags
def MULTLO_H : MULTLO_H_ENC, MULTLO_H_DESC;

class MULTHI_H_DESC : MSA_LDIX_LDSH_MULT_H_DESC_BASE<"MULTHI()", VectorHOpnd>;
class MULTHI_H_ENC  : MSA_1R_FMT_dest<0b100111000>;
// Note: does NOT affect any flags
def MULTHI_H : MULTHI_H_ENC, MULTHI_H_DESC;

class MULTLO_SPECIAL_H_DESC : MSA_LDIX_LDSH_MULT_SPECIAL_H_DESC_BASE<"MULTLO()",
                                                                   VectorHOpnd>;
def MULTLO_SPECIAL_H : MULTLO_H_ENC, MULTLO_SPECIAL_H_DESC;

class MULTHI_SPECIAL_H_DESC : MSA_LDIX_LDSH_MULT_SPECIAL_H_DESC_BASE<"MULTHI()",
                                                                   VectorHOpnd>;
def MULTHI_SPECIAL_H : MULTHI_H_ENC, MULTHI_SPECIAL_H_DESC;


let Constraints = "$specialOperandOut = $specialOperandIn" in
let isCodeGenOnly = 1 in
// let hasSideEffects = 1 in
// let isBarrier = 1 in // Can control flow fall through this instruction?
// bit isSelect     = 1;     // Is this instruction a select instruction?
def END_WHERE_2OPNDS : NonImmediateInstruction<0b100011111,
                                        (outs VectorHOpnd:$specialOperandOut),
                                        (ins VectorHOpnd:$specialOperandIn),
                                        "\n); // END_WHERE (NII) \n EXECUTE_IN_ALL( // (specialOperandIn = $specialOperandIn, specialOperandOut = $specialOperandOut) ;",
                                        []>;
//
let isCodeGenOnly    = 1 in
let hasSideEffects   = 1 in
// let isBarrier      = 1 in // Can control flow fall through this instruction?
// bit isSelect       = 1;     // Is this instruction a select instruction?
def END_WHERE : NonImmediateInstruction<0b100011111,
                              (outs),
                              (ins),
                              "\n); // END_WHERE (NII) \n EXECUTE_IN_ALL( // ;",
                              []>;
// Note: does NOT affect any flags

let isCodeGenOnly = 1 in
    // To avoid: <<error: multiline instruction is not valid for the asmparser,
    //   mark it isCodeGenOnly>>
let hasSideEffects = 1 in
let isBarrier = 1 in
def WHERECRY : NonImmediateInstruction<0b100011100,
                    (outs),
                    (ins),
                    "\n); // END_EXECUTE_IN_ALL\n  EXECUTE_WHERE_CRY(  // NII;",
                    []>;
// Note: does NOT affect any flags


//let Constraints = "$specialOperandOut = $specialOperandIn" in
// To avoid: <<error: multiline instruction is not valid for the asmparser>>
let isCodeGenOnly    = 1 in
let hasSideEffects = 1 in
//let isBarrier      = 1 in
def WHEREEQ : NonImmediateInstruction<0b100011101,
                       (outs),
                       (ins),
                       //(outs VectorHOpnd:$specialOperandOut),
                       //(ins VectorHOpnd:$specialOperandIn),
                       /*"); // END_EXECUTE_IN_ALL\n  EXECUTE_WHERE_EQ( // NII "
                              "(specialOperandIn = $specialOperandIn,
                              "specialOperandOut = $specialOperandOut) ;",*/
                       "); // END_EXECUTE_IN_ALL\n  EXECUTE_WHERE_EQ( // NII ;",
                       []>;
// Note: does NOT affect any flags




/*
let Constraints = "$specialOperandOut = $specialOperandIn" in
  // Inspired from llvm/include/llvm/Target/Target.td:
  // <<OperandConstraint, e.g. $src = $dst.>>
*/
let isCodeGenOnly    = 1 in // To avoid: <<error: multiline instruction is not
                            //  valid for the asmparser, mark it isCodeGenOnly>>
let hasSideEffects = 1 in
//let isBarrier      = 1 in
def WHERELT : NonImmediateInstruction<0b100011110,
                      (outs),
                      (ins),
                      //(outs VectorHOpnd:$specialOperandOut),
                      //(ins VectorHOpnd:$specialOperandIn),
                      /*
                         "); // END_EXECUTE_IN_ALL\n  EXECUTE_WHERE_LT( "
                         "// NII (specialOperandIn = $specialOperandIn, "
                         "specialOperandOut = $specialOperandOut) ;",
                      */
                      "); // END_EXECUTE_IN_ALL\n  EXECUTE_WHERE_LT( // NII ;",
                      []>;
// Note: does NOT affect any flags


// Inspired from lib/Target/WebAssembly/WebAssemblyInstrInfo.td
def bb_op : Operand<OtherVT>;









// Inspired from [LLVM]/llvm/lib/Target/Mips/MipsMSAInstrInfo.td

class MSA_3R_DESC_BASE_2STR<string instr_asm,
                            SDPatternOperator OpNode,
                            RegisterOperand ROWD,
                            RegisterOperand ROWR = ROWD,
                            RegisterOperand ROWL = ROWD,
                            RegisterOperand ROWP = ROWD,
                            InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWL:$wl, ROWR:$wr, ROWP:$pred);
  // INFIX_NOTATION:
  string AsmString = instr_asm;
  list<dag> Pattern = [];
  InstrItinClass Itinerary = itin;
}

/* OLD: Important: WHERE_EQ/LT/CRY has 2 inputs and 1 output because I
  lower VSELECT to WHEREEQ_H, which I later use to create a bundle with real
  machine instructions to "resist" the post-RA scheduler without changing the
  order of the nodes (in ConnexTargetMachine.cpp, passes PassCreateWhereBlocks
  and PassFinalizeBundles). */
class WHEREEQ_BUNDLE_H_ENC : MSA_3R_FMT<0b000110010>;
class WHEREEQ_BUNDLE_H_DESC : MSA_3R_DESC_BASE_2STR<
                                                ");\n EXECUTE_WHERE_EQ( //;",
                                                vseteq_v128i16, VectorHOpnd>;
                                                //,  IsCommutable;
//
let isCodeGenOnly = 1 in
let hasSideEffects = 1 in
// let isBarrier = 1 in
// let isPseudo = 1 in
def WHEREEQ_BUNDLE_H : WHEREEQ_BUNDLE_H_ENC, WHEREEQ_BUNDLE_H_DESC;

class WHERELT_BUNDLE_H_ENC : MSA_3R_FMT<0b000110010>;
class WHERELT_BUNDLE_H_DESC : MSA_3R_DESC_BASE_2STR<
                                ");\n EXECUTE_WHERE_LT( //;",
                                vsetlt_v128i16, VectorHOpnd>;
                                //,  IsCommutable;
//
let isCodeGenOnly = 1 in
let hasSideEffects = 1 in
// let isBarrier = 1 in
// let isPseudo = 1 in
def WHERELT_BUNDLE_H : WHERELT_BUNDLE_H_ENC, WHERELT_BUNDLE_H_DESC;

class WHEREULT_BUNDLE_H_ENC : MSA_3R_FMT<0b000110010>;
class WHEREULT_BUNDLE_H_DESC : MSA_3R_DESC_BASE_2STR<
                                 ");\n EXECUTE_WHERE_ULT( //;",
                                 vsetult_v128i16, VectorHOpnd>;
                                 //,  IsCommutable;
//
let isCodeGenOnly = 1 in
let hasSideEffects = 1 in
// let isBarrier = 1 in
// let isPseudo = 1 in
def WHEREULT_BUNDLE_H : WHEREULT_BUNDLE_H_ENC, WHEREULT_BUNDLE_H_DESC;

// NOTE: It modifies Carry (Sub), Lt, Eq

class CELLSHR_H_DESC : MSA_RR_PREFIX_DESC_BASE<"CELLSHR", VectorHOpnd>;
class CELLSHR_H_ENC : MSA_RR_FMT<0b100010010>;
// Note: affects flags: like Sub, Lt, Eq
def CELLSHR_H : CELLSHR_H_ENC, CELLSHR_H_DESC;

class CELLSHL_H_DESC : MSA_RR_PREFIX_DESC_BASE<"CELLSHL", VectorHOpnd>;
class CELLSHL_H_ENC : MSA_RR_FMT<0b100010001>;
// Note: affects flags: like Sub, Lt, Eq
def CELLSHL_H : CELLSHL_H_ENC, CELLSHL_H_DESC;

