//===-- VOPInstructions.td - Vector Instruction Definitions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// dummies for outer let
class LetDummies {
  bit TRANS;
  bit ReadsModeReg;
  bit mayRaiseFPException;
  bit isCommutable;
  bit isConvertibleToThreeAddress;
  bit isMoveImm;
  bit isReMaterializable;
  bit isConvergent;
  bit isAsCheapAsAMove;
  bit FPDPRounding;
  Predicate SubtargetPredicate;
  string Constraints;
  string DisableEncoding;
  list<SchedReadWrite> SchedRW;
  list<Register> Uses;
  list<Register> Defs;
  list<Predicate> OtherPredicates;
  Predicate AssemblerPredicate;
  string DecoderNamespace;
}

class VOP <string opName> {
  string OpName = opName;
}

// First 13 insts from VOPDY are also VOPDX. DOT2ACC_F32_BF16 is omitted
defvar VOPDX_Max_Index = 12;
defvar VOPD3X_Max_Index = 36;

class VOPD_Component<bits<6> OpIn, string vOPDName> {
  Instruction BaseVOP = !cast<Instruction>(NAME);
  string VOPDName = "v_dual_" # !substr(vOPDName, 2);
  bits<6> VOPDOp = OpIn;
  bit CanBeVOPDX = !le(VOPDOp, VOPDX_Max_Index);
  bit CanBeVOPD3X = !and(!le(VOPDOp, VOPD3X_Max_Index),
                         !and(!ne(vOPDName, "v_bitop2_b32"),
                              !and(!ne(vOPDName, "v_max_i32"),
                                   !ne(vOPDName, "v_min_i32"))));
}

class VOPAnyCommon <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VALU = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);
}

class VOP_Pseudo <string opName, string suffix, VOPProfile P, dag outs, dag ins,
                  string asm, list<dag> pattern> :
  InstSI <outs, ins, asm, pattern>,
  VOP <opName>,
  SIMCInstr <opName#suffix, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  Instruction Opcode = !cast<Instruction>(NAME);
  bit IsTrue16 = P.IsTrue16;
  VOPProfile Pfl = P;

  string AsmOperands;
}

class VOP3Common <dag outs, dag ins, string asm = "",
                  list<dag> pattern = [], bit HasMods = 0> :
  VOPAnyCommon <outs, ins, asm, pattern> {

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always preferred, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;

  let AsmVariantName = AMDGPUAsmVariants.VOP3;
  let AsmMatchConverter = !if(HasMods, "cvtVOP3", "");

  let isCodeGenOnly = 0;

  int Size = 8;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;
}

class VOP3_Pseudo <string opName, VOPProfile P, list<dag> pattern = [],
                   bit isVOP3P = 0, bit isVop3OpSel = 0> :
  VOP_Pseudo <opName, "_e64", P, P.Outs64,
              !if(isVop3OpSel,
                  P.InsVOP3OpSel,
                  !if(!and(isVOP3P, P.IsPacked), P.InsVOP3P, P.Ins64)),
              "", pattern> {

  let VOP3_OPSEL = isVop3OpSel;
  let IsPacked = P.IsPacked;
  let IsMAI = P.IsMAI;
  let IsWMMA = P.IsWMMA;
  let IsSWMMAC = P.IsSWMMAC;

  bit HasFP8DstByteSel = P.HasFP8DstByteSel;
  bit HasFP4DstByteSel = P.HasFP4DstByteSel;

  let AsmOperands = !if(!and(isVOP3P, P.IsPacked), P.AsmVOP3P, P.Asm64);

  let Size = 8;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always preferred, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;
  let VALU = 1;
  let FPClamp = P.HasFPClamp;
  let IntClamp = P.HasIntClamp;
  let ClampLo = P.HasClampLo;
  let ClampHi = P.HasClampHi;

  let ReadsModeReg = !or(P.DstVT.isFP, P.Src0VT.isFP);

  let mayRaiseFPException = ReadsModeReg;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);

  let AsmVariantName = AMDGPUAsmVariants.VOP3;
  let AsmMatchConverter =
    !if(isVOP3P,
        "cvtVOP3P",
        !if(!or(P.HasModifiers, P.HasOMod, P.HasClamp),
            "cvtVOP3",
            ""));
}

class VOP3P_Pseudo <string opName, VOPProfile P, list<dag> pattern = []> :
  VOP3_Pseudo<opName, P, pattern, 1> {
  let VOP3P = 1;
  let IsDOT = P.IsDOT;
}

class VOP_Real<VOP_Pseudo ps> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bit IsSingle = ps.Pfl.IsSingle;
}

class VOP3_Real <VOP_Pseudo ps, int EncodingFamily, string asm_name = ps.Mnemonic> :
  VOP_Real <ps>,
  InstSI <ps.OutOperandList, ps.InOperandList, asm_name # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let VALU = 1;
  let VOP3 = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let UseNamedOperandTable = 1;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let WaveSizePredicate  = ps.WaveSizePredicate;
  let OtherPredicates    = ps.OtherPredicates;
  let True16Predicate    = ps.True16Predicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let Constraints        = ps.Constraints;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let isConvergent         = ps.isConvergent;

  VOPProfile Pfl = ps.Pfl;
}

class VOP3_Real_Gen <VOP_Pseudo ps, GFXGen Gen, string asm_name = ps.Mnemonic> :
  VOP3_Real <ps, Gen.Subtarget, asm_name> {
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let True16Predicate = !if(ps.Pfl.IsRealTrue16, UseRealTrue16Insts, NoTrue16Predicate);
  let DecoderNamespace = Gen.DecoderNamespace#
                         !if(ps.Pfl.IsRealTrue16, "", "_FAKE16");
}

// XXX - Is there any reason to distinguish this from regular VOP3
// here?
class VOP3P_Real<VOP_Pseudo ps, int EncodingFamily, string asm_name = ps.Mnemonic> :
  VOP3_Real<ps, EncodingFamily, asm_name> {

  // The v_wmma pseudos have extra constraints that we do not want to impose on the real instruction.
  let Constraints        = !if(!eq(!substr(ps.Mnemonic,0,6), "v_wmma"), "", ps.Constraints);
}

class VOP3P_Real_Gen<VOP_Pseudo ps, GFXGen Gen, string asm_name = ps.Mnemonic> :
  VOP3P_Real<ps, Gen.Subtarget, asm_name> {
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let DecoderNamespace = Gen.DecoderNamespace;
}

class VOP3a<VOPProfile P> : Enc64 {
  bits<4> src0_modifiers;
  bits<9> src0;
  bits<3> src1_modifiers;
  bits<9> src1;
  bits<3> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{8}     = !if(P.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{9}     = !if(P.HasSrc1Mods, src1_modifiers{1}, 0);
  let Inst{10}    = !if(P.HasSrc2Mods, src2_modifiers{1}, 0);

  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

// To avoid having different version of every type of operand depending on if
// they are part of a True16 instruction or not, the operand encoding should be
// the same for SGPR, imm, and VGPR_32 whether the instruction is True16 or not.
class VOP3a_t16<VOPProfile P> : Enc64 {
  bits<11> vdst;
  bits<4> src0_modifiers;
  bits<11> src0;
  bits<3> src1_modifiers;
  bits<11> src1;
  bits<3> src2_modifiers;
  bits<11> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{7-0} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{8}     = !if(P.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{9}     = !if(P.HasSrc1Mods, src1_modifiers{1}, 0);
  let Inst{10}    = !if(P.HasSrc2Mods, src2_modifiers{1}, 0);
  // 16-bit select fields which can be interpreted as OpSel or hi/lo suffix
  let Inst{11} = !if(P.HasSrc0Mods, src0_modifiers{2}, 0);
  let Inst{12} = !if(P.HasSrc1Mods, src1_modifiers{2}, 0);
  let Inst{13} = !if(P.HasSrc2Mods, src2_modifiers{2}, 0);
  let Inst{14} = !if(!and(P.HasDst, P.HasSrc0Mods), src0_modifiers{3}, 0);
  let Inst{15} = !if(P.HasClamp, clamp{0}, 0);

  let Inst{31-26} = 0x35;
  let Inst{40-32} = !if(P.HasSrc0, src0{8-0}, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1{8-0}, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2{8-0}, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3a_gfx6_gfx7<bits<9> op, VOPProfile p> : VOP3a<p> {
  let Inst{11}    = !if(p.HasClamp, clamp{0}, 0);
  let Inst{25-17} = op;
}

class VOP3a_gfx10<bits<10> op, VOPProfile p> : VOP3a<p> {
  let Inst{15}    = !if(p.HasClamp, clamp{0}, 0);
  let Inst{25-16} = op;
  let Inst{31-26} = 0x35;
}

class VOP3a_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3a_gfx10<op, p>;

class VOP3a_vi <bits<10> op, VOPProfile P> : VOP3a<P> {
  let Inst{25-16} = op;
  let Inst{15}    = !if(P.HasClamp, clamp{0}, 0);
}

class VOP3e_gfx6_gfx7<bits<9> op, VOPProfile p> : VOP3a_gfx6_gfx7<op, p> {
  bits<8> vdst;
  let Inst{7-0} = !if(p.EmitDst, vdst{7-0}, 0);
}

class VOP3e_gfx10<bits<10> op, VOPProfile p> : VOP3a_gfx10<op, p> {
  bits<8> vdst;
  let Inst{7-0} = !if(p.EmitDst, vdst{7-0}, 0);
}

class VOP3e_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3e_gfx10<op, p>;

class VOP3e_t16_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3a_t16<p> {
  let Inst{25-16} = op;
}

class VOP3e_vi <bits<10> op, VOPProfile P> : VOP3a_vi <op, P> {
  bits<8> vdst;
  let Inst{7-0} = !if(P.EmitDst, vdst{7-0}, 0);
}

class VOP3OpSel_gfx9 <bits<10> op, VOPProfile P> : VOP3e_vi <op, P> {
  let Inst{11} = !if(P.HasSrc0, src0_modifiers{2}, 0);
  let Inst{12} = !if(P.HasSrc1, src1_modifiers{2}, 0);
  let Inst{13} = !if(P.HasSrc2, src2_modifiers{2}, 0);
  let Inst{14} = !if(P.HasDst,  src0_modifiers{3}, 0);
}

// Special case for v_permlane16_swap_b32/v_permlane32_swap_b32
// op_sel[0]/op_sel[1] are treated as bound_ctrl and fi dpp operands.
class VOP3OpSelIsDPP_base  {
  bits<1> fi;
  bits<1> bound_ctrl;
}

class VOP3OpSelIsDPP_gfx9 <bits<10> op, VOPProfile P> : VOP3OpSelIsDPP_base, VOP3e_vi <op, P> {
  // OPSEL[0] specifies FI
  let Inst{11} = fi;
  // OPSEL[1] specifies BOUND_CTRL
  let Inst{12} = bound_ctrl;
}

class VOP3OpSelIsDPP_gfx12 <bits<10> op, VOPProfile P> : VOP3OpSelIsDPP_base, VOP3e_gfx11_gfx12 <op, P> {
  // OPSEL[0] specifies FI
  let Inst{11} = fi;
  // OPSEL[1] specifies BOUND_CTRL
  let Inst{12} = bound_ctrl;
}

class VOP3OpSel_gfx10<bits<10> op, VOPProfile p> : VOP3e_gfx10<op, p> {
  let Inst{11} = !if(p.HasSrc0, src0_modifiers{2}, 0);
  let Inst{12} = !if(p.HasSrc1, src1_modifiers{2}, 0);
  let Inst{13} = !if(p.HasSrc2, src2_modifiers{2}, 0);
  let Inst{14} = !if(p.HasDst,  src0_modifiers{3}, 0);
}

class VOP3OpSel_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3OpSel_gfx10<op, p>;

class VOP3FP8OpSel_src_bytesel_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3e_gfx10<op, p> {
  bits<2> byte_sel;
  let Inst{11-12} = byte_sel; // NB: bit order is intentionally reversed!
  let Inst{13} = !if(!and(p.HasOpSel, p.HasSrc2), src2_modifiers{2}, 0);
  let Inst{14} = !if(!and(p.HasOpSel, p.HasDst), src0_modifiers{3}, 0);
}

class VOP3FP8OpSel_dst_bytesel_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3e_gfx10<op, p> {
  bits<2> byte_sel;

  let Inst{11} = !if(!and(p.HasOpSel, p.HasSrc0Mods), src0_modifiers{2}, 0); // op_sel0
  let Inst{12} = !if(!and(p.HasOpSel, p.HasSrc1Mods), src1_modifiers{2}, 0); // op_sel1
  let Inst{14-13} = byte_sel;  // op_sel2/3
}

class VOP3DotOpSel_gfx11_gfx12<bits<10> op, VOPProfile p> :
    VOP3e_t16_gfx11_gfx12<op, p>{
  let Inst{11} = ?;
  let Inst{12} = ?;
  let Inst{13} = !if(p.HasSrc2Mods, src2_modifiers{2}, 0);
  let Inst{14} = !if(!and(p.HasDst, p.HasSrc0Mods),  src0_modifiers{3}, 0);
}

// NB: For V_INTERP* opcodes, src0 is encoded as src1 and vice versa
class VOP3Interp_vi <bits<10> op, VOPProfile P> : VOP3e_vi <op, P> {
  bits<2> attrchan;
  bits<6> attr;
  bits<1> high;

  let Inst{8}     = 0; // No modifiers for src0
  let Inst{61}    = 0;

  let Inst{9}     = !if(P.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{62}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);

  let Inst{37-32} = attr;
  let Inst{39-38} = attrchan;
  let Inst{40}    = !if(P.HasHigh, high, 0);

  let Inst{49-41} = src0;
}

class VOP3a_BITOP3_gfx12<bits<10> op, VOPProfile p> : VOP3e_gfx11_gfx12<op, p> {
  bits<8> bitop3;

  let Inst{60-59} = bitop3{7-6};
  let Inst{10-8}  = bitop3{5-3};
  let Inst{63-61} = bitop3{2-0};

  let Inst{11} = !if(p.HasOpSel, src0_modifiers{2}, 0);
  let Inst{12} = !if(p.HasOpSel, src1_modifiers{2}, 0);
  let Inst{13} = !if(p.HasOpSel, src2_modifiers{2}, 0);
  let Inst{14} = !if(p.HasOpSel, src0_modifiers{3}, 0);
}

class VOP3a_ScaleSel_gfx1250<bits<10> op, VOPProfile p> : VOP3e_gfx11_gfx12<op, p> {
  bits<4> scale_sel;

  let Inst{14-11} = scale_sel;
}

class VOP3Interp_OpSel_gfx9<bits<10> op, VOPProfile p> : VOP3Interp_vi<op, p> {
  let Inst{11} = src0_modifiers{2};
  // There's no src1
  let Inst{13} = src2_modifiers{2};
  let Inst{14} = !if(p.HasDst,  src0_modifiers{3}, 0);
}

class VOP3Interp_gfx10<bits<10> op, VOPProfile p> : VOP3e_gfx10<op, p> {
  bits<6> attr;
  bits<2> attrchan;
  bits<1> high;

  let Inst{8}     = 0;
  let Inst{9}     = !if(p.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{37-32} = attr;
  let Inst{39-38} = attrchan;
  let Inst{40}    = !if(p.HasHigh, high, 0);
  let Inst{49-41} = src0;
  let Inst{61}    = 0;
  let Inst{62}    = !if(p.HasSrc0Mods, src0_modifiers{0}, 0);
}

class VOP3Interp_gfx11<bits<10> op, VOPProfile p> : VOP3Interp_gfx10<op, p>;

class VOP3be <VOPProfile P> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<7> sdst;
  bits<2> omod;

  let Inst{7-0}   = vdst;
  let Inst{14-8}  = sdst;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3Pe_Base {
  bits<8> vdst;
  bits<4> src0_modifiers;
  bits<9> src0;
  bits<4> src1_modifiers;
  bits<9> src1;
  bits<4> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> index_key_8bit;
  bits<1> index_key_16bit;
  bits<1> index_key_32bit;
  bits<3> matrix_a_fmt;
  bits<3> matrix_b_fmt;
  bits<1> matrix_a_scale;
  bits<1> matrix_b_scale;
  bits<2> matrix_a_scale_fmt;
  bits<2> matrix_b_scale_fmt;
  bits<1> matrix_a_reuse;
  bits<1> matrix_b_reuse;
}

class VOP3Pe <VOPProfile P> : Enc64, VOP3Pe_Base {
  let Inst{7-0} = !if(P.HasDst, vdst, 0);
  let Inst{8} = !if(P.HasSrc0Mods, src0_modifiers{1},
                                   !if(P.HasMatrixScale, matrix_b_scale_fmt{0}, 0)); // neg_hi src0
  let Inst{9} = !if(P.HasSrc1Mods, src1_modifiers{1},
                                   !if(P.HasMatrixScale, matrix_b_scale_fmt{1}, 0)); // neg_hi src1
  let Inst{10} = !if(P.HasSrc2Mods, src2_modifiers{1}, 0); // neg_hi src2

  let Inst{11} = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{2},
                     !if(P.HasMatrixScale, matrix_a_scale{0}, 0)); // op_sel(0)
  let Inst{12} = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{2}, 0); // op_sel(1)
  let Inst{13} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{2},
                     !if(P.HasMatrixReuse, matrix_a_reuse, 0));    // op_sel(2)

  let Inst{14} = !cond(!and(P.HasSrc2, P.HasOpSel) : src2_modifiers{3},
                       P.IsDOT : 1,
                       P.HasMatrixReuse : matrix_b_reuse,
                       1: ?); // op_sel_hi(2)

  let Inst{15} = !if(P.HasClamp, clamp{0}, 0);

  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{59}    = !cond(!and(P.HasSrc0, P.HasOpSel) : src0_modifiers{3},
                          P.IsDOT : 1,
                          P.HasMatrixScale : matrix_b_scale{0},
                          1: ?); // op_sel_hi(0)
  let Inst{60}    = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{3},
                        !if(P.HasMatrixScale, 0,
                            !if(P.IsDOT, 1, ?))); // op_sel_hi(1)
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0},
                                       !if(P.HasMatrixScale, matrix_a_scale_fmt{0}, 0)); // neg (lo)
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0},
                                       !if(P.HasMatrixScale, matrix_a_scale_fmt{1}, 0)); // neg (lo)
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0); // neg (lo)
}

class VOP3Pe_MAI_Base {
  bits<8> vdst;
  bits<10> src0;
  bits<10> src1;
  bits<9> src2;
  bits<3> blgp;
  bits<3> cbsz;
  bits<4> abid;
}

class VOP3Pe_MAI <bits<7> op, VOPProfile P, bit acc_cd = 0> : Enc64, VOP3Pe_MAI_Base {
  let Inst{7-0} = vdst;

  let Inst{10-8}  = !if(P.HasSrc1, cbsz, 0);
  let Inst{14-11} = !if(P.HasAbid, abid, 0);

  let Inst{15} = acc_cd;

  let Inst{22-16} = op;
  let Inst{31-23} = 0x1a7; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0{8-0}, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1{8-0}, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);

  let Inst{59}    = !if(P.HasSrc0, src0{9}, 0); // acc(0)
  let Inst{60}    = !if(P.HasSrc1, src1{9}, 0); // acc(1)

  let Inst{63-61} = !if(P.HasSrc1, blgp, 0);
}

class VOP3Pe_SMFMAC <bits<7> op> : Enc64 {
  bits<10> vdst; // VGPR or AGPR, but not SGPR. vdst{8} is not encoded in the instruction.
  bits<10> src0;
  bits<10> src1;
  bits<9> idx;
  bits<3> blgp;
  bits<3> cbsz;
  bits<4> abid;

  let blgp = 0;

  let Inst{7-0} = vdst{7-0};

  let Inst{10-8}  = cbsz;
  let Inst{14-11} = abid;

  let Inst{15} = vdst{9}; // acc(vdst)

  let Inst{22-16} = op;
  let Inst{31-23} = 0x1a7; // encoding
  let Inst{40-32} = src0{8-0};
  let Inst{49-41} = src1{8-0};
  let Inst{58-50} = idx;

  let Inst{59}    = src0{9}; // acc(0)
  let Inst{60}    = src1{9}; // acc(1)

  let Inst{63-61} = blgp;
}

class VOP3PXe <bits<7> op, VOPProfile MFMAPfl, bit acc_cd = 0> : Enc128, VOP3Pe_MAI_Base {
  bits<9> scale_src0;
  bits<9> scale_src1;

  //MFMALdScaleModifierOp transforms 2 bit opsel input to 4 bit value
  //where opsel and opselHi are in 3rd and 4th bit.
  bits<4> src0_modifiers;
  bits<4> src1_modifiers;

  // Inst{7-0} = unused
  // Inst{10-8} = neg_hi;
  // Inst{13-11} = op_sel
  let Inst{11} = src0_modifiers{2}; //opsel[0]
  let Inst{12} = src1_modifiers{2}; //opsel[1]
  // Inst{13} = unused op_sel
  // Inst{14} = unused op_sel_hi2

  let Inst{31-16} = 0b1101001110101100;
  let Inst{40-32} = scale_src0;
  let Inst{49-41} = scale_src1;
  // Inst{50-58} = unused
  // Inst{60-59} = op_sel_hi;
  let Inst{59} = src0_modifiers{3}; //opsel_hi[0]
  let Inst{60} = src1_modifiers{3}; //opsel_hi[1]
  // Inst{63-61} = neg;

  // The high half of the encoding is the unscaled mfma op.
  //
  // FIXME: Defining the encoding in terms of the base instruction
  // seems to not work, results in all 0 encoding, so replicate all
  // the fields from VOP3Pe_MAI, shifted up by 64
  //
  // defvar Hi = VOP3Pe_MAI<op, MFMAPfl, acc_cd>;
  // let Inst{127-64} = Hi.Inst;

  let Inst{71-64} = vdst;
  let Inst{74-72} = !if(MFMAPfl.HasSrc1, cbsz, 0);

  // abid must be 1 to use a scale.
  let Inst{78-75} = 0b0001; // abid

  let Inst{79} = acc_cd;

  let Inst{86-80} = op;
  let Inst{95-87} = 0x1a7; //encoding
  let Inst{104-96} = !if(MFMAPfl.HasSrc0, src0{8-0}, 0);
  let Inst{113-105} = !if(MFMAPfl.HasSrc1, src1{8-0}, 0);
  let Inst{122-114} = !if(MFMAPfl.HasSrc2, src2, 0);

  let Inst{123}    = !if(MFMAPfl.HasSrc0, src0{9}, 0); // acc(0)
  let Inst{124}    = !if(MFMAPfl.HasSrc1, src1{9}, 0); // acc(1)

  let Inst{127-125} = !if(MFMAPfl.HasSrc1, blgp, 0);
}

class VOP3Pe_vi <bits<7> op, VOPProfile P> : VOP3Pe<P> {
  let Inst{22-16} = op;
  let Inst{31-23} = 0x1a7; //encoding
}

class VOP3Pe_gfx10 <bits<8> op, VOPProfile P> : VOP3Pe<P> {
  let Inst{23-16} = op;
  let Inst{31-24} = 0xcc; //encoding
}

class VOP3Pe_gfx11_gfx12<bits<8> op, VOPProfile P> : VOP3Pe_gfx10<op, P>;

class VOP3be_gfx6_gfx7<bits<9> op, VOPProfile p> : VOP3be<p> {
  let Inst{25-17} = op;
}

class VOP3be_gfx10<bits<10> op, VOPProfile p> : VOP3be<p> {
  bits<1> clamp;
  let Inst{15}    = !if(p.HasClamp, clamp{0}, 0);
  let Inst{25-16} = op;
  let Inst{31-26} = 0x35;
}

class VOP3be_gfx11_gfx12<bits<10> op, VOPProfile p> : VOP3be_gfx10<op, p>;

class VOP3be_vi <bits<10> op, VOPProfile P> : VOP3be<P> {
  bits<1> clamp;
  let Inst{25-16} = op;
  let Inst{15}    = !if(P.HasClamp, clamp{0}, 0);
}

def SDWA {
  // sdwa_sel
  int BYTE_0 = 0;
  int BYTE_1 = 1;
  int BYTE_2 = 2;
  int BYTE_3 = 3;
  int WORD_0 = 4;
  int WORD_1 = 5;
  int DWORD = 6;

  // dst_unused
  int UNUSED_PAD = 0;
  int UNUSED_SEXT = 1;
  int UNUSED_PRESERVE = 2;
}

class VOP_SDWAe<VOPProfile P> : Enc64 {
  bits<8> src0;
  bits<3> src0_sel;
  bits<5> src0_modifiers; // float: {abs,neg}, int {sext}
  bits<3> src1_sel;
  bits<5> src1_modifiers;
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{42-40} = !if(P.EmitDstSel, dst_sel{2-0}, ?);
  let Inst{44-43} = !if(P.EmitDstSel, dst_unused{1-0}, ?);
  let Inst{45}    = !if(P.HasSDWAClamp, clamp{0}, 0);
  let Inst{50-48} = !if(P.HasSrc0, src0_sel{2-0}, 0);
  let Inst{51}    = !if(P.HasSrc0IntMods, src0_modifiers{4}, 0);
  let Inst{53-52} = !if(P.HasSrc0FloatMods, src0_modifiers{1-0}, 0);
  let Inst{58-56} = !if(P.HasSrc1, src1_sel{2-0}, 0);
  let Inst{59}    = !if(P.HasSrc1IntMods, src1_modifiers{4}, 0);
  let Inst{61-60} = !if(P.HasSrc1FloatMods, src1_modifiers{1-0}, 0);
}

// GFX9 adds two features to SDWA:
// 1.	Add 3 fields to the SDWA microcode word: S0, S1 and OMOD.
//    a. S0 and S1 indicate that source 0 and 1 respectively are SGPRs rather
//       than VGPRs (at most 1 can be an SGPR);
//    b. OMOD is the standard output modifier (result *2, *4, /2)
// 2.	Add a new version of the SDWA microcode word for VOPC: SDWAB. This
//    replaces OMOD and the dest fields with SD and SDST (SGPR destination)
//    field.
//    a. When SD=1, the SDST is used as the destination for the compare result;
//    b. When SD=0, VCC is used.
//
// In GFX9, V_MAC_F16, V_MAC_F32 opcodes cannot be used with SDWA

// gfx9 SDWA basic encoding
class VOP_SDWA9e<VOPProfile P> : Enc64 {
  bits<9> src0; // {src0_sgpr{0}, src0{7-0}}
  bits<3> src0_sel;
  bits<5> src0_modifiers; // float: {abs,neg}, int {sext}
  bits<3> src1_sel;
  bits<5> src1_modifiers;
  bits<1> src1_sgpr;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{50-48} = !if(P.HasSrc0, src0_sel{2-0}, 0);
  let Inst{51}    = !if(P.HasSrc0IntMods, src0_modifiers{4}, 0);
  let Inst{53-52} = !if(P.HasSrc0FloatMods, src0_modifiers{1-0}, 0);
  let Inst{55}    = !if(P.HasSrc0, src0{8}, 0);
  let Inst{58-56} = !if(P.HasSrc1, src1_sel{2-0}, 0);
  let Inst{59}    = !if(P.HasSrc1IntMods, src1_modifiers{4}, 0);
  let Inst{61-60} = !if(P.HasSrc1FloatMods, src1_modifiers{1-0}, 0);
  let Inst{63}    = 0; // src1_sgpr - should be specified in subclass
}

// gfx9 SDWA-A
class VOP_SDWA9Ae<VOPProfile P> : VOP_SDWA9e<P> {
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;
  bits<2> omod;

  let Inst{42-40} = !if(P.EmitDstSel, dst_sel{2-0}, ?);
  let Inst{44-43} = !if(P.EmitDstSel, dst_unused{1-0}, ?);
  let Inst{45}    = !if(P.HasSDWAClamp, clamp{0}, 0);
  let Inst{47-46} = !if(P.HasSDWAOMod, omod{1-0}, 0);
}

// gfx9 SDWA-B
class VOP_SDWA9Be<VOPProfile P> : VOP_SDWA9e<P> {
  bits<8> sdst; // {vcc_sdst{0}, sdst{6-0}}

  let Inst{46-40} = !if(P.EmitDst, sdst{6-0}, ?);
  let Inst{47} = !if(P.EmitDst, sdst{7}, 0);
}

class VOP_SDWA_Pseudo <string opName, VOPProfile P, list<dag> pattern=[]> :
  InstSI <P.OutsSDWA, P.InsSDWA, "", pattern>,
  VOP <opName>,
  SIMCInstr <opName#"_sdwa", SIEncodingFamily.NONE> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = P.AsmSDWA;
  string AsmOperands9 = P.AsmSDWA9;

  let Size = 8;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let VALU = 1;
  let SDWA = 1;

  let ReadsModeReg = !or(P.DstVT.isFP, P.Src0VT.isFP);

  let mayRaiseFPException = ReadsModeReg;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);

  let SubtargetPredicate = HasSDWA;
  let AsmVariantName = !if(P.HasExtSDWA, AMDGPUAsmVariants.SDWA,
                                         AMDGPUAsmVariants.Disable);
  let DecoderNamespace = "GFX8";

  VOPProfile Pfl = P;
}

class VOP_SDWA8_Real <VOP_SDWA_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SDWA> {

  let VALU = 1;
  let SDWA = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let hasSideEffects = ps.hasSideEffects;

  let Constraints     = ps.Constraints;

  // Copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AssemblerPredicate   = HasSDWA8;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let AsmVariantName       = ps.AsmVariantName;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let DecoderNamespace     = ps.DecoderNamespace;
  let Constraints          = ps.Constraints;
  let TSFlags              = ps.TSFlags;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let isConvergent         = ps.isConvergent;
}

class Base_VOP_SDWA9_Real <VOP_SDWA_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands9, []> {

  let VALU = 1;
  let SDWA = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let hasSideEffects = ps.hasSideEffects;

  let Constraints     = ps.Constraints;

  let SubtargetPredicate = ps.SubtargetPredicate;
  let AssemblerPredicate = HasSDWA9;
  let OtherPredicates    = ps.OtherPredicates;
  let AsmVariantName = !if(ps.Pfl.HasExtSDWA9, AMDGPUAsmVariants.SDWA9,
                                               AMDGPUAsmVariants.Disable);
  let DecoderNamespace = "GFX9";

  // Copy relevant pseudo op flags
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Constraints          = ps.Constraints;
  let TSFlags              = ps.TSFlags;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let isConvergent         = ps.isConvergent;
}

class VOP_SDWA9_Real <VOP_SDWA_Pseudo ps> :
  Base_VOP_SDWA9_Real <ps >,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SDWA9>;

class Base_VOP_SDWA10_Real<VOP_SDWA_Pseudo ps> : Base_VOP_SDWA9_Real<ps> {
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AssemblerPredicate = HasSDWA10;
  let DecoderNamespace = "GFX10";
}

class VOP_SDWA10_Real<VOP_SDWA_Pseudo ps> :
  Base_VOP_SDWA10_Real<ps>, SIMCInstr<ps.PseudoInstr, SIEncodingFamily.SDWA10>;

class VOP_DPPe<VOPProfile P, bit IsDPP16=0> : Enc64 {
  bits<2> src0_modifiers;
  bits<8> src0;
  bits<2> src1_modifiers;
  bits<9> dpp_ctrl;
  bits<1> bound_ctrl;
  bits<4> bank_mask;
  bits<4> row_mask;
  bit     fi;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{48-40} = dpp_ctrl;
  let Inst{50}    = !if(IsDPP16, fi, ?);
  let Inst{51}    = bound_ctrl;
  let Inst{52}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0); // src0_neg
  let Inst{53}    = !if(P.HasSrc0Mods, src0_modifiers{1}, 0); // src0_abs
  let Inst{54}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0); // src1_neg
  let Inst{55}    = !if(P.HasSrc1Mods, src1_modifiers{1}, 0); // src1_abs
  let Inst{59-56} = bank_mask;
  let Inst{63-60} = row_mask;
}

class VOP3_DPPe_Fields_Base {
  bits<9> dpp_ctrl;
  bits<1> bound_ctrl;
  bits<4> bank_mask;
  bits<4> row_mask;
  bit     fi;
}
class VOP3_DPPe_Fields : VOP3_DPPe_Fields_Base {
  bits<8> src0;
}

class VOP3_DPPe_Fields_t16 : VOP3_DPPe_Fields_Base {
  bits<11> src0;
}

// Common refers to common between DPP and DPP8
// Base refers to a shared base between T16 and regular instructions
class VOP3_DPPe_Common_Base<bits<10> op, VOPProfile P> : Enc96 {
  bits<4> src0_modifiers;
  bits<3> src1_modifiers;
  bits<3> src2_modifiers;
  bits<1> clamp;
  bits<2> omod;
  bits<2> byte_sel;

  let Inst{8}     = !if(P.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{9}     = !if(P.HasSrc1Mods, src1_modifiers{1}, 0);
  let Inst{10}    = !if(P.HasSrc2Mods, src2_modifiers{1}, 0);
  // 16-bit select fields which can be interpreted as OpSel or hi/lo suffix
  let Inst{11} = !if(P.HasFP8SrcByteSel, byte_sel{1},
                 !if(P.HasOpSel, !if(P.HasSrc0Mods, src0_modifiers{2}, 0), ?));
  let Inst{12} = !if(P.HasFP8SrcByteSel, byte_sel{0},
                 !if(P.HasOpSel, !if(P.HasSrc1Mods, src1_modifiers{2}, 0), ?));
  let Inst{13} = !if(P.HasFP8DstByteSel, byte_sel{0},
                 !if(P.HasOpSel, !if(P.HasSrc2Mods, src2_modifiers{2}, 0), ?));
  let Inst{14} = !if(P.HasFP8DstByteSel, byte_sel{1},
                 !if(P.HasOpSel, !if(P.HasSrc0Mods, src0_modifiers{3}, 0), ?));
  let Inst{15}    = !if(P.HasClamp, clamp, 0);
  let Inst{25-16} = op;
  let Inst{31-26} = 0x35;

  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3_DPPe_Common<bits<10> op, VOPProfile P> : VOP3_DPPe_Common_Base<op, P> {
  bits<8> vdst;
  bits<9> src1;
  bits<9> src2;

  let Inst{7-0}   = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
}

class VOP3_DPPe_Common_t16<bits<10> op, VOPProfile P> : VOP3_DPPe_Common_Base<op, P> {
  bits<11> vdst;
  bits<11> src1;
  bits<11> src2;

  let Inst{7-0}   = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1{8-0}, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2{8-0}, 0);
}

class VOP3P_DPPe_Common_Base<bits<8> op, VOPProfile P> : Enc96 {
  bits<4> src0_modifiers;
  bits<4> src1_modifiers;
  bits<4> src2_modifiers;
  bits<1> clamp;

  let Inst{8} = !if(P.HasSrc0Mods, src0_modifiers{1}, 0); // neg_hi src0
  let Inst{9} = !if(P.HasSrc1Mods, src1_modifiers{1}, 0); // neg_hi src1
  let Inst{10} = !if(P.HasSrc2Mods, src2_modifiers{1}, 0); // neg_hi src2
  // OPSEL must be set such that the low result only uses low inputs, and the high result only uses high inputs.
  let Inst{11} = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{2}, 0); // op_sel(0)
  let Inst{12} = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{2}, 0); // op_sel(1)
  let Inst{13} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{2}, 0); // op_sel(2)
  let Inst{14} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{3}, !if(P.IsDOT, 1, ?)); // op_sel_hi(2)
  let Inst{15} = !if(P.HasClamp, clamp{0}, 0);
  let Inst{23-16} = op;
  let Inst{31-24} = 0xcc; // encoding
  let Inst{59}    = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{3}, !if(P.IsDOT, 1, ?)); // op_sel_hi(0)
  let Inst{60}    = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{3}, !if(P.IsDOT, 1, ?)); // op_sel_hi(1)
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0); // neg (lo)
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0); // neg (lo)
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0); // neg (lo)
}

class VOP3P_DPPe_Common<bits<8> op, VOPProfile P> : VOP3P_DPPe_Common_Base<op, P> {
  bits<8> vdst;
  bits<9> src1;
  bits<9> src2;

  let Inst{7-0} = vdst;
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
}

class VOP3P_DPPe_Common_t16<bits<8> op, VOPProfile P> : VOP3P_DPPe_Common_Base<op, P> {
  bits<11> vdst;
  bits<11> src1;
  bits<11> src2;

  let Inst{7-0} = vdst{7-0};
  let Inst{49-41} = !if(P.HasSrc1, src1{8-0}, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2{8-0}, 0);
}

class VOP_DPP_Pseudo <string OpName, VOPProfile P, list<dag> pattern=[],
  dag Ins = P.InsDPP, string asmOps = P.AsmDPP> :
  VOP_Pseudo<OpName, "_dpp", P, P.OutsDPP, Ins, asmOps, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let VALU = 1;
  let DPP = 1;
  let Size = 8;
  let IsPacked = P.IsPacked;

  let ReadsModeReg = !or(P.DstVT.isFP, P.Src0VT.isFP);

  let mayRaiseFPException = ReadsModeReg;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);
  let isConvergent = 1;

  string AsmOperands = asmOps;

  let AsmMatchConverter = !if(P.HasModifiers, "cvtDPP", "");
  let AssemblerPredicate = !if(P.HasExt64BitDPP, HasDPALU_DPP, HasDPP);
  let AsmVariantName = !if(P.HasExtDPP, AMDGPUAsmVariants.DPP,
                                        AMDGPUAsmVariants.Disable);
  let Constraints = !if(P.NumSrcArgs, P.TieRegDPP # " = $vdst", "");
  let DecoderNamespace = "GFX8";
}

class VOP3_DPP_Pseudo <string OpName, VOPProfile P> :
  VOP_DPP_Pseudo <OpName, P, [], P.InsVOP3DPP, P.AsmVOP3DPP> {
  let PseudoInstr = OpName#"_e64"#"_dpp";
  let OutOperandList = P.OutsVOP3DPP;
  let Size = 12;
  let VOP3 = 1;
  let AsmMatchConverter = "cvtVOP3DPP";
  let AsmVariantName = !if(!or(P.HasExtVOP3DPP, P.HasExt64BitDPP),
                           AMDGPUAsmVariants.VOP3_DPP,
                           AMDGPUAsmVariants.Disable);
}

class VOP_DPP_Real <VOP_DPP_Pseudo ps, int EncodingFamily> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let VALU = 1;
  let DPP = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let hasSideEffects = ps.hasSideEffects;

  let Constraints     = ps.Constraints;

  // Copy relevant pseudo op flags
  let isConvergent         = ps.isConvergent;
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let True16Predicate      = ps.True16Predicate;
  let AssemblerPredicate   = ps.AssemblerPredicate;
  let OtherPredicates      = ps.OtherPredicates;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let AsmVariantName       = ps.AsmVariantName;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let DecoderNamespace     = ps.DecoderNamespace;
  let Constraints          = ps.Constraints;
  let TSFlags              = ps.TSFlags;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
  let TRANS                = ps.TRANS;
  let isConvergent         = ps.isConvergent;
}

class VOP_DPP_Base <string OpName, VOPProfile P,
               dag InsDPP,
               string AsmDPP > :
  InstSI <P.OutsDPP, InsDPP, OpName#AsmDPP, []> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;

  let VALU = 1;
  let DPP = 1;
  let Size = 8;

  let AsmMatchConverter = !if(P.HasModifiers, "cvtDPP", "");
  let AssemblerPredicate = !if(P.HasExt64BitDPP, HasDPALU_DPP, HasDPP);
  let AsmVariantName = !if(P.HasExtDPP, AMDGPUAsmVariants.DPP,
                                        AMDGPUAsmVariants.Disable);
  let Constraints = !if(P.NumSrcArgs, P.TieRegDPP # " = $vdst", "");
  let DecoderNamespace = "GFX8";
}

class VOP_DPP <string OpName, VOPProfile P, bit IsDPP16,
               dag InsDPP = !if(IsDPP16, P.InsDPP16, P.InsDPP),
               string AsmDPP = !if(IsDPP16, P.AsmDPP16, P.AsmDPP)> :
  VOP_DPP_Base<OpName, P, InsDPP, AsmDPP>, VOP_DPPe<P, IsDPP16>;

class VOP3_DPP_Base <string OpName, VOPProfile P, bit IsDPP16,
               dag InsDPP = !if(IsDPP16, P.InsVOP3DPP16, P.InsVOP3DPP),
               string AsmDPP = !if(IsDPP16, P.AsmVOP3DPP16, P.AsmVOP3DPP)> :
  VOP_DPP_Base<OpName, P, InsDPP, AsmDPP> {
  let OutOperandList = P.OutsVOP3DPP;
  let AsmMatchConverter = "cvtVOP3DPP";
  let VOP3 = 1;
  let AsmVariantName = !if(!or(P.HasExtVOP3DPP, P.HasExt64BitDPP),
                           AMDGPUAsmVariants.VOP3_DPP,
                           AMDGPUAsmVariants.Disable);
  let Size = 12;
}

class VOP3_DPP_Enc <bits<10> op, VOPProfile P, bit IsDPP16> :
  VOP3_DPPe_Common<op, P>,
  VOP3_DPPe_Fields {

  let Inst{40-32} = 0xfa;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{80-72} = dpp_ctrl;
  let Inst{82}    = !if(IsDPP16, fi, ?);
  let Inst{83}    = bound_ctrl;

  // Inst{87-84} ignored by hw
  let Inst{91-88} = bank_mask;
  let Inst{95-92} = row_mask;
}

class VOP3_DPP <bits<10> op, string OpName, VOPProfile P, bit IsDPP16,
               dag InsDPP = !if(IsDPP16, P.InsVOP3DPP16, P.InsVOP3DPP),
               string AsmDPP = !if(IsDPP16, P.AsmVOP3DPP16, P.AsmVOP3DPP)> :
  VOP3_DPP_Base<OpName, P, IsDPP16, InsDPP, AsmDPP>, VOP3_DPP_Enc<op, P, IsDPP16>;

class VOP3_DPP_Enc_t16<bits<10> op, VOPProfile P, bit IsDPP16 >
    : VOP3_DPPe_Common_t16<op, P>,
      VOP3_DPPe_Fields_t16 {

  let Inst{40-32} = 0xfa;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{80-72} = dpp_ctrl;
  let Inst{82}    = !if(IsDPP16, fi, ?);
  let Inst{83}    = bound_ctrl;

  // Inst{87-84} ignored by hw
  let Inst{91-88} = bank_mask;
  let Inst{95-92} = row_mask;
}

class VOP3_DPP_t16<bits<10> op, string OpName, VOPProfile P, bit IsDPP16,
                   dag InsDPP = !if (IsDPP16, P.InsVOP3DPP16, P.InsVOP3DPP),
                   string AsmDPP = !if (IsDPP16, P.AsmVOP3DPP16, P.AsmVOP3DPP)>
    : VOP3_DPP_Base<OpName, P, IsDPP16, InsDPP, AsmDPP>,
      VOP3_DPP_Enc_t16<op, P, IsDPP16> {
}

class VOP3P_DPP <bits<8> op, string OpName, VOPProfile P, bit IsDPP16,
               dag InsDPP = !if(IsDPP16, P.InsVOP3DPP16, P.InsVOP3DPP),
               string AsmDPP = !if(IsDPP16, P.AsmVOP3DPP16, P.AsmVOP3DPP)> :
  VOP3_DPP_Base<OpName, P, IsDPP16, InsDPP, AsmDPP>, VOP3P_DPPe_Common<op, P>,
  VOP3_DPPe_Fields {

  let VOP3P = 1;

  let Inst{40-32} = 0xfa;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{80-72} = dpp_ctrl;
  let Inst{82}    = !if(IsDPP16, fi, ?);
  let Inst{83}    = bound_ctrl;

  // Inst{87-84} ignored by hw
  let Inst{91-88} = bank_mask;
  let Inst{95-92} = row_mask;
}

class VOP_DPP8e<VOPProfile P> : Enc64 {
  bits<8> src0;
  bits<24> dpp8;
  bits<9> fi;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{63-40} = dpp8{23-0};
}

class VOP3_DPP8e_Fields {
  bits<8> src0;
  bits<24> dpp8;
  bits<9> fi;
}

class VOP3_DPP8e_Fields_t16 {
  bits<11> src0;
  bits<24> dpp8;
  bits<9> fi;
}

class VOP_DPP8_Base<string OpName, VOPProfile P, dag InsDPP8 = P.InsDPP8, string AsmDPP8 = P.AsmDPP8> :
  InstSI<P.OutsDPP8, InsDPP8, OpName#AsmDPP8, []> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;

  let VALU = 1;
  let DPP = 1;
  let Size = 8;

  let AsmMatchConverter = "cvtDPP8";
  let AssemblerPredicate = HasDPP8;
  let AsmVariantName = AMDGPUAsmVariants.DPP;
  let Constraints = !if(P.NumSrcArgs, P.TieRegDPP # " = $vdst", "");
}

class VOP_DPP8<string OpName, VOPProfile P> :
  VOP_DPP8_Base<OpName, P>, VOP_DPP8e<P>;

class VOP3_DPP8_Base<string OpName, VOPProfile P> :
  VOP_DPP8_Base<OpName, P, P.InsVOP3DPP8, P.AsmVOP3DPP8> {
  let OutOperandList = P.OutsVOP3DPP8;
  let AsmMatchConverter = "cvtVOP3DPP8";
  let AsmVariantName = !if(P.HasExtVOP3DPP, AMDGPUAsmVariants.VOP3_DPP,
                                            AMDGPUAsmVariants.Disable);
  let VOP3 = 1;
  let Size = 12;
}

class VOP3_DPP8_Enc <bits<10> op, VOPProfile P> :
  VOP3_DPPe_Common<op, P>,
  VOP3_DPP8e_Fields {
  let Inst{40-32} = fi;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{95-72} = dpp8{23-0};
}

class VOP3_DPP8<bits<10> op, string OpName, VOPProfile P> :
  VOP3_DPP8_Base<OpName, P>, VOP3_DPP8_Enc<op, P>;

class VOP3_DPP8_Enc_t16 <bits<10> op, VOPProfile P> :
  VOP3_DPPe_Common_t16<op, P>,
  VOP3_DPP8e_Fields_t16 {
  let Inst{40-32} = fi;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{95-72} = dpp8{23-0};
}

class VOP3_DPP8_t16<bits<10> op, string OpName, VOPProfile P> :
  VOP3_DPP8_Base<OpName, P>, VOP3_DPP8_Enc_t16<op, P>;

class VOP3P_DPP8<bits<8> op, string OpName, VOPProfile P> :
  VOP3_DPP8_Base<OpName, P>, VOP3P_DPPe_Common<op, P>,
  VOP3_DPP8e_Fields {

  let VOP3P = 1;
  let Inst{40-32} = fi;
  let Inst{71-64} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{95-72} = dpp8{23-0};
}

def DPP8Mode {
  int FI_0 = 0xE9;
  int FI_1 = 0xEA;
}

class getNumNodeArgs<SDPatternOperator Op> {
  SDNode N = !cast<SDNode>(Op);
  SDTypeProfile TP = N.TypeProfile;
  int ret = TP.NumOperands;
}

class getDivergentFrag<SDPatternOperator Op> {
  assert !or(!isa<SDNode>(Op), !isa<PatFrags>(Op)), "Expected SDNode or PatFrags";

  int NumSrcArgs = !if(!isa<SDNode>(Op), getNumNodeArgs<Op>.ret,
    !size(!cast<PatFrags>(Op).Operands));
  PatFrag ret = PatFrag <
    !if(!eq(NumSrcArgs, 1),
             (ops node:$src0),
             !if(!eq(NumSrcArgs, 2),
               (ops node:$src0, node:$src1),
               (ops node:$src0, node:$src1, node:$src2))),
    !if(!eq(NumSrcArgs, 1),
             (Op $src0),
             !if(!eq(NumSrcArgs, 2),
               (Op $src0, $src1),
               (Op $src0, $src1, $src2))),
    [{ return N->isDivergent(); }]
  >;
}

class VOPPatGen<SDPatternOperator Op, VOPProfile P> {
  PatFrag Operator = getDivergentFrag < Op >.ret;

  dag Ins = !foreach(tmp, P.Ins32, !subst(ins, Operator,
                                         !subst(P.Src0RC32, P.Src0VT,
                                               !subst(P.Src1RC32, P.Src1VT, tmp))));

  dag Outs = !foreach(tmp, P.Outs32, !subst(outs, set,
                                           !subst(P.DstRC, P.DstVT, tmp)));

  list<dag> ret =  [!con(Outs, (set Ins))];
}

class DivergentUnaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0),
  (Op $src0),
  [{ return N->isDivergent(); }]> {
  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}

class VOPPatOrNull<SDPatternOperator Op, VOPProfile P> {
  list<dag> ret = !if(!ne(P.NeedPatGen,PatGenMode.NoPattern), VOPPatGen<Op, P>.ret, []);
}

class DivergentFragOrOp<SDPatternOperator Op, VOPProfile P> {
  SDPatternOperator ret = !if(!eq(P.NeedPatGen,PatGenMode.Pattern),
   !if(!isa<SDNode>(Op), getDivergentFrag<Op>.ret, Op), Op);
}

class getVSrcOp<ValueType vt> {
  RegisterOperand ret = !if(!eq(vt.Size, 32), VSrc_b32, VSrc_b16);
}

// Class for binary integer operations with the clamp bit set for saturation
// TODO: Add sub with negated inline constant pattern.
class VOPBinOpClampPat<SDPatternOperator node, Instruction inst, ValueType vt> :
  GCNPat<(node vt:$src0, vt:$src1),
         (inst getVSrcOp<vt>.ret:$src0, getVSrcOp<vt>.ret:$src1,
               DSTCLAMP.ENABLE)
>;

//===----------------------------------------------------------------------===//
// VOP3 Classes
//===----------------------------------------------------------------------===//

class getVOP3ModPat<VOPProfile P, SDPatternOperator node> {
  dag src0 = !if(P.HasOMod,
    (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
    (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp));

  list<dag> ret3 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0)))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3PModPat<VOPProfile P, SDPatternOperator node, bit HasExplicitClamp,
                     bit IsDOT = 0,
                     ComplexPattern SrcPat = !if(IsDOT, VOP3PModsDOT, VOP3PMods)> {
  dag src0_dag = (P.Src0VT (SrcPat P.Src0VT:$src0, i32:$src0_modifiers));
  dag src1_dag = (P.Src1VT (SrcPat P.Src1VT:$src1, i32:$src1_modifiers));
  dag src2_dag = (P.Src2VT (SrcPat P.Src2VT:$src2, i32:$src2_modifiers));
  dag clamp_dag = (i1 timm:$clamp);

  list<dag> ret3 = [(set P.DstVT:$vdst,
    !if(HasExplicitClamp,
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag, src2_dag, clamp_dag),
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag, src2_dag)))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    !if(HasExplicitClamp,
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag, clamp_dag),
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag)))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    !if(HasExplicitClamp,
        (DivergentFragOrOp<node, P>.ret src0_dag, clamp_dag),
        (DivergentFragOrOp<node, P>.ret src0_dag)))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3OpSelPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
        (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers)),
          (P.Src1VT (VOP3OpSel P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3OpSel P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers)),
                                    (P.Src1VT (VOP3OpSel P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3OpSelModPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT !if(P.HasClamp, (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers),
                                    (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers))),
          (P.Src1VT (VOP3OpSelMods P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3OpSelMods P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret !if(P.HasClamp, (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers)),
                          (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers))),
          (P.Src1VT (VOP3OpSelMods P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3FromVOP2Pat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))];
}
// In VOP1, we can have clamp and omod even if !HasModifiers
class getVOP3Pat<VOPProfile P, SDPatternOperator node> {
  dag src0 =
    !if(P.HasOMod,
      !if(P.HasClamp,
          (VOP3Mods0 P.Src0VT:$src0, i1:$clamp, i32:$omod),
          (VOP3Mods0 P.Src0VT:$src0, i32:$omod)), // impossible?
      !if(P.HasClamp,
          (VOP3Mods0 P.Src0VT:$src0, i1:$clamp),
          (VOP3Mods0 P.Src0VT:$src0))
    );
  list<dag> ret3 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret (P.Src0VT src0), P.Src1VT:$src1, P.Src2VT:$src2))];

  list<dag> ret2 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret (P.Src0VT src0), P.Src1VT:$src1))];

  list<dag> ret1 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret (P.Src0VT src0)))];
  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3ClampPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2, i1:$clamp))];
  list<dag> ret2 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, i1:$clamp))];
  list<dag> ret1 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, i1:$clamp))];
  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3MAIPat<VOPProfile P, SDPatternOperator node> {
  list<dag> mfma_with_abid = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2,
                                   timm:$cbsz, timm:$abid, timm:$blgp))];
  list<dag> mfma_no_abid = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2,
                            timm:$cbsz, timm:$blgp))];

  list<dag> ret = !if(!not(P.IsSMFMAC),
                      // mfma
                      !if(P.HasAbid, mfma_with_abid, mfma_no_abid),

                      // smfmac
                      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2, i32:$idx,
                                            timm:$cbsz, timm:$abid))]);
}

class getVOP3MAIScaledPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret = !if(!not(P.IsSMFMAC),
                      // mfma
                      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2,
                                            timm:$cbsz, timm:$blgp,
                                            MFMALdScaleModifierOp:$src0_modifiers,
                                            i32:$scale_src0,
                                            MFMALdScaleModifierOp:$src1_modifiers,
                                            i32:$scale_src1
                                            ))],
                      // smfmac
                      [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2, i32:$idx,
                                            timm:$cbsz, timm:$abid,
                                            MFMALdScaleModifierOp:$src0_modifiers,
                                            i32:$scale_src0,
                                            MFMALdScaleModifierOp:$src1_modifiers,
                                            i32:$scale_src1))]);
}

class VOP3Features<bit Clamp, bit OpSel, bit Packed, bit MAI> {
  bit HasClamp = Clamp;
  bit HasOpSel = OpSel;
  bit IsPacked = Packed;
  bit IsMAI = MAI;
}

def VOP3_REGULAR         : VOP3Features<0, 0, 0, 0>;
def VOP3_CLAMP           : VOP3Features<1, 0, 0, 0>;
def VOP3_OPSEL           : VOP3Features<1, 1, 0, 0>;
def VOP3_PACKED          : VOP3Features<1, 1, 1, 0>;
def VOP3_PACKED_NO_OPSEL : VOP3Features<1, 0, 1, 0>;
def VOP3_MAI             : VOP3Features<0, 0, 0, 1>;
def VOP3_OPSEL_ONLY      : VOP3Features<0, 1, 0, 0>;

// Packed is misleading, but it enables the appropriate op_sel
// modifiers.
def VOP3P_LD_SCALE : VOP3Features<0, 1, 1, 0>;

class VOP3_Profile_Base<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOPProfile<P.ArgVT> {

  let HasClamp = !if(Features.HasClamp, 1, P.HasClamp);
  let HasOpSel = !if(Features.HasOpSel, 1, P.HasOpSel);
  let IsMAI    = !if(Features.IsMAI,    1, P.IsMAI);
  let IsPacked = !if(Features.IsPacked, 1, P.IsPacked);
  let HasFP8SrcByteSel = P.HasFP8SrcByteSel;
  let HasFP8DstByteSel = P.HasFP8DstByteSel;
  let HasOMod = P.HasOMod;
  let HasBitOp3 = P.HasBitOp3;

  let HasModifiers =
      !if (Features.IsMAI, 0,
           !or(Features.IsPacked, P.HasModifiers));
}

class VOP3_Profile<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOP3_Profile_Base<P, Features> {
  let IsSingle = 1;

}

class VOP3_Profile_True16<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOPProfile_True16<P> {
  let HasClamp = !if(Features.HasClamp, 1, P.HasClamp);
  let HasOpSel = !if(Features.HasOpSel, 1, P.HasOpSel);
  let IsMAI    = !if(Features.IsMAI,    1, P.IsMAI);
  let IsPacked = !if(Features.IsPacked, 1, P.IsPacked);
  let HasFP8SrcByteSel = P.HasFP8SrcByteSel;
  let HasFP8DstByteSel = P.HasFP8DstByteSel;
  let HasOMod = P.HasOMod;
  let HasBitOp3 = P.HasBitOp3;

  let HasModifiers =
      !if (Features.IsMAI, 0,
           !or(Features.IsPacked, Features.HasOpSel, P.HasModifiers));
  let IsSingle = 1;
}

class VOP3_Profile_Fake16<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOPProfile_Fake16<P> {
  let HasClamp = !if(Features.HasClamp, 1, P.HasClamp);
  let HasOpSel = !if(Features.HasOpSel, 1, P.HasOpSel);
  let IsMAI    = !if(Features.IsMAI,    1, P.IsMAI);
  let IsPacked = !if(Features.IsPacked, 1, P.IsPacked);
  let HasFP8SrcByteSel = P.HasFP8SrcByteSel;
  let HasFP8DstByteSel = P.HasFP8DstByteSel;
  let HasOMod = P.HasOMod;
  let HasBitOp3 = P.HasBitOp3;

  let HasModifiers =
      !if (Features.IsMAI, 0,
           !or(Features.IsPacked, Features.HasOpSel, P.HasModifiers));
  let IsSingle = 1;
}

// consistently gives instructions a _e64 suffix
multiclass VOP3Inst_Pseudo_Wrapper<string opName, VOPProfile P, list<dag> pattern = [], bit VOP3Only = 0> {
    def _e64 : VOP3_Pseudo<opName, P, pattern, VOP3Only>;
}

class VOP3InstBase<string OpName, VOPProfile P, SDPatternOperator node = null_frag, bit IsVOP2 = 0, bit MAIScaled = false> :
  VOP3_Pseudo<OpName, P,
    !if(P.HasOpSel,
        !if(P.HasModifiers,
            getVOP3OpSelModPat<P, node>.ret,
            getVOP3OpSelPat<P, node>.ret),
        !if(P.HasModifiers,
            getVOP3ModPat<P, node>.ret,
            !if(IsVOP2,
              getVOP3FromVOP2Pat<P, node>.ret,
              !if(P.HasIntClamp,
                  getVOP3ClampPat<P, node>.ret,
                  !if (P.IsMAI,
                      !if(MAIScaled, getVOP3MAIScaledPat<P, node>.ret, getVOP3MAIPat<P, node>.ret),
                      getVOP3Pat<P, node>.ret))))),
    0, P.HasOpSel> {

  let IntClamp = P.HasIntClamp;
  let AsmMatchConverter =
    !if(P.HasOpSel,
        "cvtVOP3OpSel",
        !if(!or(P.HasModifiers, P.HasOMod, P.HasIntClamp),
            "cvtVOP3",
            ""));
}

multiclass VOP3Inst<string OpName, VOPProfile P, SDPatternOperator node = null_frag,
                    list<Predicate> predicates = []> {
  def _e64 : VOP3InstBase<OpName, P, node>;
  if P.HasExtVOP3DPP then
    def _e64_dpp  : VOP3_DPP_Pseudo <OpName, P> {
      let SubtargetPredicate = isGFX11Plus;
    }
  else if P.HasExt64BitDPP then
    def _e64_dpp  : VOP3_DPP_Pseudo <OpName, P> {
      let OtherPredicates = !listconcat(predicates, [HasDPALU_DPP]);
    }
}

class UniformUnaryFragOrOp<SDPatternOperator Op> {
  SDPatternOperator ret = !if(!or(!isa<SDNode>(Op), !isa<PatFrags>(Op)),
                              UniformUnaryFrag<Op>, Op);
}

multiclass VOP3PseudoScalarInst<string OpName, VOPProfile P,
                                SDPatternOperator node = null_frag> {
  def _e64 : VOP3_Pseudo<OpName, P, [(set P.DstVT:$vdst,
      (UniformUnaryFragOrOp<node>.ret
          (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp,
                               i32:$omod))))]>;
}

multiclass VOP3Inst_t16_with_profiles<string OpName, VOPProfile P, VOPProfile P_t16,
                        VOPProfile P_fake16,
                        SDPatternOperator node = null_frag,
                        SDPatternOperator node_t16 = node> {
  let True16Predicate = NotHasTrue16BitInsts  in {
    defm NAME : VOP3Inst<OpName, P, node>;
  }
  let True16Predicate = UseRealTrue16Insts in {
    defm _t16 : VOP3Inst<OpName # "_t16", P_t16, node_t16>;
  }
  let True16Predicate = UseFakeTrue16Insts in {
    defm _fake16 : VOP3Inst<OpName # "_fake16", P_fake16, node>;
  }
}

multiclass VOP3Inst_t16<string OpName, VOPProfile P,
                        SDPatternOperator node = null_frag,
                        SDPatternOperator node_t16 = node>
 : VOP3Inst_t16_with_profiles<OpName, VOP3_Profile<P, VOP3_OPSEL>,
                VOP3_Profile_True16<P, VOP3_OPSEL>, VOP3_Profile_Fake16<P, VOP3_OPSEL>,
                node, node_t16>;

//===----------------------------------------------------------------------===//
// VOP3 DPP
//===----------------------------------------------------------------------===//

class VOP3_DPP16_Helper<bits<10> op, VOP_DPP_Pseudo ps, string opName = ps.OpName>
    : VOP3_DPP<op, opName, ps.Pfl, 1> {
  let VOP3_OPSEL = ps.Pfl.HasOpSel;
  let IsDOT = ps.IsDOT;
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;
  let AssemblerPredicate = HasDPP16;
  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates = ps.OtherPredicates;
}

class VOP3_DPP16_t16_Helper<bits<10> op, VOP_DPP_Pseudo ps,
                          string opName = ps.OpName>
    : VOP3_DPP_t16<op, opName, ps.Pfl, 1> {
  let VOP3_OPSEL = ps.Pfl.HasOpSel;
  let IsDOT = ps.IsDOT;
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;
  let AssemblerPredicate = HasDPP16;
  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates = ps.OtherPredicates;
}

class VOP3_DPP16<bits<10> op, VOP_DPP_Pseudo ps, int subtarget,
                 string opName = ps.OpName>
    : VOP3_DPP16_Helper<op, ps, opName>, SIMCInstr<ps.PseudoInstr, subtarget>;

class VOP3_DPP16_t16<bits<10> op, VOP_DPP_Pseudo ps, int subtarget,
                     string opName = ps.OpName>
    : VOP3_DPP16_t16_Helper<op, ps, opName>, SIMCInstr<ps.PseudoInstr, subtarget>;

class VOP3_DPP16_Gen<bits<10> op, VOP_DPP_Pseudo ps, GFXGen Gen,
                     string opName = ps.OpName>
    : VOP3_DPP16<op, ps, Gen.Subtarget, opName> {
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let DecoderNamespace = Gen.DecoderNamespace;
}

class VOP3_DPP16_Gen_t16<bits<10> op, VOP_DPP_Pseudo ps, GFXGen Gen,
                         string opName = ps.OpName>
    : VOP3_DPP16_t16<op, ps, Gen.Subtarget, opName> {
  let True16Predicate =
      !if (ps.Pfl.IsRealTrue16, UseRealTrue16Insts, NoTrue16Predicate);
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let DecoderNamespace =
      Gen.DecoderNamespace #!if (ps.Pfl.IsRealTrue16, "", "_FAKE16");
}

class Base_VOP3_DPP8<bits<10> op, VOP_Pseudo ps, string opName = ps.OpName>
    : VOP3_DPP8<op, opName, ps.Pfl> {
  let VOP3_OPSEL = ps.Pfl.HasOpSel;
  let IsDOT = ps.IsDOT;
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates = ps.OtherPredicates;
  let True16Predicate = ps.True16Predicate;
}

class Base_VOP3_DPP8_t16<bits<10> op, VOP_Pseudo ps, string opName = ps.OpName>
    : VOP3_DPP8_t16<op, opName, ps.Pfl> {
  let VOP3_OPSEL = ps.Pfl.HasOpSel;
  let IsDOT = ps.IsDOT;
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates = ps.OtherPredicates;
  let True16Predicate = ps.True16Predicate;
}

class Base_VOP3b_DPP16<bits<10> op, VOP_DPP_Pseudo ps,
                       string opName = ps.OpName>
    : VOP3_DPP16_Helper<op, ps, opName> {
  bits<7> sdst;
  let Inst{14 - 8} = sdst;
}

class VOP3b_DPP8_Base<bits<10> op, VOP_Pseudo ps, string opName = ps.OpName>
    : Base_VOP3_DPP8<op, ps, opName> {
  bits<7> sdst;
  let Inst{14 - 8} = sdst;
}

class VOP3_BITOP3_DPP16_Gen<bits<10> op, VOP_DPP_Pseudo p, GFXGen Gen, string asmName>
    : VOP3_DPP16_Gen_t16<op, p, Gen, asmName> {
  bits<8> bitop3;

  let Inst{60-59} = bitop3{7-6};
  let Inst{10-8}  = bitop3{5-3};
  let Inst{63-61} = bitop3{2-0};

  let Inst{11} = !if(p.Pfl.HasOpSel, src0_modifiers{2}, 0);
  let Inst{12} = !if(p.Pfl.HasOpSel, src1_modifiers{2}, 0);
  let Inst{13} = !if(p.Pfl.HasOpSel, src2_modifiers{2}, 0);
  let Inst{14} = !if(p.Pfl.HasOpSel, src0_modifiers{3}, 0);
}

class VOP3_BITOP3_DPP8<bits<10> op, VOP_Pseudo p, string asmName>
    : Base_VOP3_DPP8_t16<op, p, asmName> {
  bits<8> bitop3;

  let Inst{60-59} = bitop3{7-6};
  let Inst{10-8}  = bitop3{5-3};
  let Inst{63-61} = bitop3{2-0};

  let Inst{11} = !if(p.Pfl.HasOpSel, src0_modifiers{2}, 0);
  let Inst{12} = !if(p.Pfl.HasOpSel, src1_modifiers{2}, 0);
  let Inst{13} = !if(p.Pfl.HasOpSel, src2_modifiers{2}, 0);
  let Inst{14} = !if(p.Pfl.HasOpSel, src0_modifiers{3}, 0);
}

class VOP3b_DPP8_Base_t16<bits<10> op, VOP_Pseudo ps, string opName = ps.OpName>
    : Base_VOP3_DPP8<op, ps, opName> {
  bits<8> sdst;
  let Inst{14 - 8} = sdst{7-1};
}

//===----------------------------------------------------------------------===//
// VOP3 GFX11, GFX12
//===----------------------------------------------------------------------===//

multiclass VOP3_Real_Base<GFXGen Gen, bits<10> op, string opName = NAME,
                          bit isSingle = 0> {
  defvar ps = !cast<VOP_Pseudo>(opName#"_e64");
  let IsSingle = !or(isSingle, ps.Pfl.IsSingle) in {
    if ps.Pfl.HasFP8SrcByteSel then {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3FP8OpSel_src_bytesel_gfx11_gfx12<op, ps.Pfl>;
    } else if ps.Pfl.HasFP8DstByteSel then {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3FP8OpSel_dst_bytesel_gfx11_gfx12<op, ps.Pfl>;
    } else if ps.Pfl.HasOpSel then {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3OpSel_gfx11_gfx12<op, ps.Pfl>;
    } else {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3e_gfx11_gfx12<op, ps.Pfl>;
    }
  }
}

multiclass VOP3Dot_Real_Base<GFXGen Gen, bits<10> op, string asmName, string opName = NAME,
                             bit isSingle = 0> {
  defvar ps = !cast<VOP_Pseudo>(opName#"_e64");
  let AsmString = asmName # ps.AsmOperands,
      DecoderNamespace = Gen.DecoderNamespace # !if(ps.Pfl.IsRealTrue16, "", "_FAKE16"),
      IsSingle = !or(isSingle, ps.Pfl.IsSingle) in {
    def _e64#Gen.Suffix :
      VOP3_Real_Gen<ps, Gen>,
      VOP3DotOpSel_gfx11_gfx12<op, ps.Pfl>;
  }
}

multiclass VOP3_Real_with_name<GFXGen Gen, bits<10> op, string opName,
                               string asmName, string pseudo_mnemonic = "", bit isSingle = 0> {
  defvar ps = !cast<VOP_Pseudo>(opName#"_e64");
  let AsmString = asmName # ps.AsmOperands,
      IsSingle = !or(isSingle, ps.Pfl.IsSingle) in {
    // FIXME-TRUE16 support FP8 instructions properly
    if ps.Pfl.HasFP8SrcByteSel then {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3FP8OpSel_src_bytesel_gfx11_gfx12<op, ps.Pfl>;
    } else if ps.Pfl.HasFP8DstByteSel then {
      def _e64#Gen.Suffix :
        VOP3_Real_Gen<ps, Gen>,
        VOP3FP8OpSel_dst_bytesel_gfx11_gfx12<op, ps.Pfl>;
    } else {
      if ps.Pfl.IsRealTrue16 then {
        def _e64#Gen.Suffix :
          VOP3_Real_Gen<ps, Gen>,
          VOP3e_t16_gfx11_gfx12<op, ps.Pfl>;
      } else {
        if ps.Pfl.HasOpSel then {
          def _e64#Gen.Suffix :
            VOP3_Real_Gen<ps, Gen>,
            VOP3OpSel_gfx11_gfx12<op, ps.Pfl>;
        } else {
          def _e64#Gen.Suffix :
            VOP3_Real_Gen<ps, Gen>,
            VOP3e_gfx11_gfx12<op, ps.Pfl>;
        }
      }
    }
  }
  if !ne(ps.Mnemonic, asmName) then {
    def Gen.Suffix#"_VOP3_alias" : LetDummies,
                                   AMDGPUMnemonicAlias<!if(!empty(pseudo_mnemonic),
                                                       ps.Mnemonic, pseudo_mnemonic), asmName, ""> {
      let AssemblerPredicate = Gen.AssemblerPredicate;
    }
  }
}

// for READLANE/WRITELANE
multiclass VOP3_Real_No_Suffix<GFXGen Gen, bits<10> op, string opName = NAME> {
  defvar ps = !cast<VOP_Pseudo>(opName);
    def _e64#Gen.Suffix :
      VOP3_Real_Gen<ps, Gen>,
      VOP3e_gfx11_gfx12<op, ps.Pfl>;
}

multiclass VOP3_Real_dpp_Base<GFXGen Gen, bits<10> op, string opName = NAME> {
  defvar ps = !cast<VOP_DPP_Pseudo>(opName#"_e64"#"_dpp");
  if ps.Pfl.IsTrue16 then
    def _e64_dpp#Gen.Suffix :
      VOP3_DPP16_Gen_t16<op, ps, Gen>;
  else
    def _e64_dpp#Gen.Suffix :
      VOP3_DPP16_Gen<op, ps, Gen>;
}

multiclass VOP3Dot_Real_dpp_Base<GFXGen Gen, bits<10> op, string asmName, string opName = NAME> {
  defvar ps = !cast<VOP_DPP_Pseudo>(opName#"_e64"#"_dpp");
  def _e64_dpp#Gen.Suffix :
    VOP3_DPP16_Gen_t16<op, ps, Gen> {
      let AsmString = asmName # ps.Pfl.AsmVOP3DPP16;
      let DecoderNamespace = Gen.DecoderNamespace
                             # !if(ps.Pfl.IsRealTrue16, "", "_FAKE16");
      let Inst{11} = ?;
      let Inst{12} = ?;
    }
}

multiclass VOP3_Real_dpp_with_name<GFXGen Gen, bits<10> op, string opName,
                                   string asmName> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  let AsmString = asmName # ps.Pfl.AsmVOP3DPP16 in {
    defm NAME : VOP3_Real_dpp_Base<Gen, op, opName>;
  }
}

multiclass VOP3_Real_dpp8_Base<GFXGen Gen, bits<10> op, string opName = NAME> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  if !not(ps.Pfl.HasExt64BitDPP) then
    def _e64_dpp8#Gen.Suffix : Base_VOP3_DPP8<op, ps> {
      let DecoderNamespace = Gen.DecoderNamespace;
      let AssemblerPredicate = Gen.AssemblerPredicate;
    }
}

multiclass VOP3Dot_Real_dpp8_Base<GFXGen Gen, bits<10> op, string asmName, string opName = NAME> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  if !not(ps.Pfl.HasExt64BitDPP) then
    def _e64_dpp8#Gen.Suffix : Base_VOP3_DPP8<op, ps> {
      let Inst{11} = ?;
      let Inst{12} = ?;
      let AsmString = asmName # ps.Pfl.AsmVOP3DPP8;
      let DecoderNamespace = Gen.DecoderNamespace
                             # !if(ps.Pfl.IsRealTrue16, "", "_FAKE16");
      let AssemblerPredicate = Gen.AssemblerPredicate;
    }
}

multiclass VOP3_Real_dpp8_with_name<GFXGen Gen, bits<10> op, string opName,
                                    string asmName> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  if !not(ps.Pfl.HasExt64BitDPP) then
    let AsmString = asmName # ps.Pfl.AsmVOP3DPP8,
        DecoderNamespace = Gen.DecoderNamespace#
                           !if(ps.Pfl.IsRealTrue16, "", "_FAKE16"),
        True16Predicate = !if(ps.Pfl.IsRealTrue16, UseRealTrue16Insts,
                              NoTrue16Predicate) in {
      defm NAME : VOP3_Real_dpp8_Base<Gen, op, opName>;
  }
}

multiclass VOP3be_Real<GFXGen Gen, bits<10> op, string opName, string asmName,
                      bit isSingle = 0> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  let IsSingle = !or(isSingle, ps.Pfl.IsSingle) in
    def _e64#Gen.Suffix :
      VOP3_Real_Gen<ps, Gen, asmName>,
      VOP3be_gfx11_gfx12<op, ps.Pfl> ;
}

multiclass VOP3be_Real_dpp<GFXGen Gen, bits<10> op, string opName,
                           string asmName> {
  defvar ps = !cast<VOP3_Pseudo>(opName #"_e64");
  defvar dpp_ps = !cast<VOP_DPP_Pseudo>(opName #"_e64" #"_dpp");
  def _e64_dpp#Gen.Suffix : Base_VOP3b_DPP16<op, dpp_ps, asmName>,
                            SIMCInstr<dpp_ps.PseudoInstr, Gen.Subtarget> {
    let DecoderNamespace = Gen.DecoderNamespace;
    let AssemblerPredicate = Gen.AssemblerPredicate;
  }
}

multiclass VOP3be_Real_dpp8<GFXGen Gen, bits<10> op, string opName,
                            string asmName> {
  defvar ps = !cast<VOP3_Pseudo>(opName #"_e64");
  if !not(ps.Pfl.HasExt64BitDPP) then
    def _e64_dpp8#Gen.Suffix : VOP3b_DPP8_Base<op, ps, asmName> {
      let DecoderNamespace = Gen.DecoderNamespace;
      let AssemblerPredicate = Gen.AssemblerPredicate;
    }
}

// VOP1 and VOP2 depend on these triple defs
multiclass VOP3_Realtriple<GFXGen Gen, bits<10> op, bit isSingle = 0,
                           string opName = NAME> :
  VOP3_Real_Base<Gen, op, opName, isSingle>,
  VOP3_Real_dpp_Base<Gen, op, opName>,
  VOP3_Real_dpp8_Base<Gen, op, opName>;

multiclass VOP3Dot_Realtriple<GFXGen Gen, bits<10> op, string asmName, bit isSingle = 0,
                              string opName = NAME> :
  VOP3Dot_Real_Base<Gen, op, asmName, opName, isSingle>,
  VOP3Dot_Real_dpp_Base<Gen, op, asmName, opName>,
  VOP3Dot_Real_dpp8_Base<Gen, op, asmName, opName>;

multiclass VOP3Only_Realtriple<GFXGen Gen, bits<10> op> :
  VOP3_Realtriple<Gen, op, 1>;

multiclass VOP3_Realtriple_with_name<GFXGen Gen, bits<10> op, string opName,
                                     string asmName, string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3_Real_with_name<Gen, op, opName, asmName, pseudo_mnemonic, isSingle>,
  VOP3_Real_dpp_with_name<Gen, op, opName, asmName>,
  VOP3_Real_dpp8_with_name<Gen, op, opName, asmName>;

multiclass VOP3Only_Realtriple_with_name<GFXGen Gen, bits<10> op, string opName,
                                         string asmName> :
  VOP3_Realtriple_with_name<Gen, op, opName, asmName, "", 1>;

multiclass VOP3be_Realtriple<
    GFXGen Gen, bits<10> op, bit isSingle = 0, string opName = NAME,
    string asmName = !cast<VOP_Pseudo>(opName#"_e64").Mnemonic> :
  VOP3be_Real<Gen, op, opName, asmName, isSingle>,
  VOP3be_Real_dpp<Gen, op, opName, asmName>,
  VOP3be_Real_dpp8<Gen, op, opName, asmName>;

multiclass VOP3beOnly_Realtriple<GFXGen Gen, bits<10> op> :
  VOP3be_Realtriple<Gen, op, 1>;

multiclass VOP3_BITOP3_Real_dpp_Base<GFXGen Gen, bits<10> op, string asmName> {
  def _e64_dpp#Gen.Suffix :
    VOP3_BITOP3_DPP16_Gen<op, !cast<VOP_DPP_Pseudo>(NAME#"_e64"#"_dpp"), Gen, asmName>;
}

multiclass VOP3_BITOP3_Real_dpp8_Base<GFXGen Gen, bits<10> op, string asmName> {
  defvar ps = !cast<VOP3_Pseudo>(NAME#"_e64");
  def _e64_dpp8#Gen.Suffix : VOP3_BITOP3_DPP8<op, ps, asmName> {
    let DecoderNamespace =
      Gen.DecoderNamespace #!if (ps.Pfl.IsRealTrue16, "", "_FAKE16");
    let AssemblerPredicate = Gen.AssemblerPredicate;
  }
}

multiclass VOP3_BITOP3_Real_Base<GFXGen Gen, bits<10> op, string asmName> {
  defvar ps = !cast<VOP_Pseudo>(NAME#"_e64");
  let IsSingle = ps.Pfl.IsSingle, AsmString = asmName # ps.AsmOperands in {
    def _e64#Gen.Suffix :
      VOP3_Real_Gen<ps, Gen>,
      VOP3a_BITOP3_gfx12<op, ps.Pfl>;
  }
}

multiclass VOP3Only_ScaleSel_Real_gfx1250<bits<10> op> {
  defvar ps = !cast<VOP_Pseudo>(NAME#"_e64");
  def _e64_gfx1250 :
    VOP3_Real_Gen<ps, GFX1250Gen>,
    VOP3a_ScaleSel_gfx1250<op, ps.Pfl>;
}

multiclass VOP3Only_Realtriple_t16_gfx11_gfx12_not_gfx1250<bits<10> op, string asmName, string opName = NAME,
                                                       string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX11Gen, op, opName, asmName, pseudo_mnemonic, isSingle>,
  VOP3_Realtriple_with_name<GFX12Not12_50Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3Only_Realtriple_t16_and_fake16_gfx11_gfx12_not_gfx1250<bits<10> op, string asmName,
                                                          string opName = NAME, string pseudo_mnemonic = ""> {
  defm _t16 : VOP3Only_Realtriple_t16_gfx11_gfx12_not_gfx1250<op, asmName, opName#"_t16", pseudo_mnemonic, 1>;
  defm _fake16 : VOP3Only_Realtriple_t16_gfx11_gfx12_not_gfx1250<op, asmName, opName#"_fake16", pseudo_mnemonic, 1>;
}

multiclass VOP3_Realtriple_with_name_gfx11_gfx12_not_gfx1250<bits<10> op, string opName,
                                                           string asmName, string pseudo_mnemonic = "",
                                                           bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX11Gen, op, opName, asmName, pseudo_mnemonic, isSingle>,
  VOP3_Realtriple_with_name<GFX12Not12_50Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

//===----------------------------------------------------------------------===//
// VOP3 GFX11
//===----------------------------------------------------------------------===//

// VOP1 and VOP2 depend on these triple defs

multiclass VOP3_Realtriple_t16_gfx11<bits<10> op, string asmName, string opName = NAME,
                                     string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX11Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3_Realtriple_t16_and_fake16_gfx11<bits<10> op, string asmName, string opName = NAME,
                                                string pseudo_mnemonic = "", bit isSingle = 0> {
  defm _t16: VOP3_Realtriple_t16_gfx11<op, asmName, opName#"_t16", pseudo_mnemonic, isSingle>;
  defm _fake16: VOP3_Realtriple_t16_gfx11<op, asmName, opName#"_fake16", pseudo_mnemonic, isSingle>;
}

multiclass VOP3Only_Realtriple_t16_gfx11<bits<10> op, string asmName,
                                     string opName = NAME, string pseudo_mnemonic = "">
  : VOP3_Realtriple_t16_gfx11<op, asmName, opName, pseudo_mnemonic, 1>;

multiclass VOP3Only_Realtriple_t16_and_fake16_gfx11<bits<10> op, string asmName,
                                                    string opName = NAME, string pseudo_mnemonic = ""> {
  defm _t16: VOP3_Realtriple_t16_gfx11<op, asmName, opName#"_t16", pseudo_mnemonic, 1>;
  defm _fake16: VOP3_Realtriple_t16_gfx11<op, asmName, opName#"_fake16", pseudo_mnemonic, 1>;
}

multiclass VOP3be_Real_gfx11<bits<10> op, string opName, string asmName,
                             bit isSingle = 0> :
  VOP3be_Real<GFX11Gen, op, opName, asmName, isSingle>;

multiclass VOP3_Real_Base_gfx11<bits<10> op, string opName = NAME,
                                bit isSingle = 0> :
  VOP3_Real_Base<GFX11Gen, op, opName, isSingle>;

multiclass VOP3_Realtriple_gfx11<bits<10> op, bit isSingle = 0,
                                 string opName = NAME> :
  VOP3_Realtriple<GFX11Gen, op, isSingle, opName>;

//===----------------------------------------------------------------------===//
// VOP3 GFX12
//===----------------------------------------------------------------------===//

multiclass VOP3Only_Realtriple_gfx12<bits<10> op, bit isSingle = 0> :
  VOP3_Realtriple<GFX12Gen, op, isSingle>;

// IsSingle is captured from the vopprofile for these instructions, but the
// following alternative is more explicit
multiclass VOP3Only_Real_Base_gfx12<bits<10> op> :
  VOP3_Real_Base<GFX12Gen, op, NAME, 1/*IsSingle*/>;

multiclass VOP3Only_Realtriple_with_name_gfx12_not_gfx1250<bits<10> op, string opName,
                                                           string asmName, string pseudo_mnemonic = "",
                                                           bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX12Not12_50Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3Only_Real_Base_gfx1250<bits<10> op> :
  VOP3_Real_Base<GFX1250Gen, op, NAME, 1/*IsSingle*/>;

multiclass VOP3Only_Realtriple_gfx1250<bits<10> op, bit isSingle = 0> :
  VOP3_Realtriple<GFX1250Gen, op, isSingle>;

multiclass VOP3Only_Realtriple_gfx12_not_gfx1250<bits<10> op, bit isSingle = 0> :
  VOP3_Realtriple<GFX12Not12_50Gen, op, isSingle>;

multiclass VOP3Only_Realtriple_with_name_gfx1250<bits<10> op, string opName,
                                                 string asmName, string pseudo_mnemonic = "",
                                                 bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX1250Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3Only_Realtriple_t16_gfx1250<bits<10> op, string asmName = !cast<VOP3_Pseudo>(NAME#"_e64").Mnemonic,
                                           string opName = NAME, string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3Only_Realtriple_with_name_gfx1250<op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3_Realtriple_t16_gfx12<bits<10> op, string asmName, string opName = NAME,
                                     string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX12Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3_Realtriple_t16_and_fake16_gfx12<bits<10> op, string asmName, string opName = NAME,
                                                string pseudo_mnemonic = "", bit isSingle = 0> {
  defm _t16:VOP3_Realtriple_t16_gfx12<op, asmName, opName#"_t16", pseudo_mnemonic, isSingle>;
  defm _fake16:VOP3_Realtriple_t16_gfx12<op, asmName, opName#"_fake16", pseudo_mnemonic, isSingle>;
}

multiclass VOP3Only_Realtriple_t16_gfx12<bits<10> op, string asmName,
                                     string opName = NAME, string pseudo_mnemonic = "">
  : VOP3_Realtriple_t16_gfx12<op, asmName, opName, pseudo_mnemonic, 1>;

multiclass VOP3Only_Realtriple_t16_and_fake16_gfx12<bits<10> op, string asmName,
                                     string opName = NAME, string pseudo_mnemonic = ""> {
  defm _t16 : VOP3Only_Realtriple_t16_gfx12<op, asmName, opName#"_t16", pseudo_mnemonic>;
  defm _fake16 : VOP3Only_Realtriple_t16_gfx12<op, asmName, opName#"_fake16", pseudo_mnemonic>;
}

multiclass VOP3Only_Realtriple_t16_and_fake16_gfx1250<bits<10> op,
                                                      string asmName = !cast<VOP3_Pseudo>(NAME#"_e64").Mnemonic,
                                                      string opName = NAME, string pseudo_mnemonic = ""> {
  defm _t16 : VOP3Only_Realtriple_t16_gfx1250<op, asmName, opName#"_t16", pseudo_mnemonic>;
  defm _fake16 : VOP3Only_Realtriple_t16_gfx1250<op, asmName, opName#"_fake16", pseudo_mnemonic>;
}

multiclass VOP3be_Real_with_name_gfx12<bits<10> op, string opName,
                                       string asmName, bit isSingle = 0> {
  defvar ps = !cast<VOP3_Pseudo>(opName#"_e64");
  defm NAME : VOP3be_Realtriple<GFX12Gen, op, !or(isSingle, ps.Pfl.IsSingle),
                                opName, asmName>;
  def : AMDGPUMnemonicAlias<ps.Mnemonic, asmName> {
    let AssemblerPredicate = GFX12Gen.AssemblerPredicate;
  }
}

multiclass VOP3_Realtriple_with_name_gfx12<bits<10> op, string opName,
                                           string asmName, string pseudo_mnemonic = "", bit isSingle = 0> :
  VOP3_Realtriple_with_name<GFX12Gen, op, opName, asmName, pseudo_mnemonic, isSingle>;

multiclass VOP3Only_Realtriple_with_name_gfx11_gfx12<bits<10> op, string opName,
                                                     string asmName> :
  VOP3Only_Realtriple_with_name<GFX11Gen, op, opName, asmName>,
  VOP3Only_Realtriple_with_name<GFX12Gen, op, opName, asmName>;

multiclass VOP3_Real_BITOP3_gfx1250<bits<10> op, string asmName = !cast<VOP3_Pseudo>(NAME#"_e64").Mnemonic> :
  VOP3_BITOP3_Real_Base<GFX1250Gen, op, asmName>,
  VOP3_BITOP3_Real_dpp_Base<GFX1250Gen, op, asmName>,
  VOP3_BITOP3_Real_dpp8_Base<GFX1250Gen, op, asmName>;

multiclass VOP3_Real_BITOP3_t16_and_fake16_gfx1250<bits<10> op, string asmName = !cast<VOP3_Pseudo>(NAME#"_e64").Mnemonic> {
  defm _t16 : VOP3_Real_BITOP3_gfx1250<op, asmName>;
  defm _fake16: VOP3_Real_BITOP3_gfx1250<op, asmName>;
}

multiclass VOP3Dot_Realtriple_gfx11_gfx12<bits<10> op, string asmName, bit isSingle = 0,
                                          string opName = NAME> :
  VOP3Dot_Realtriple<GFX11Gen, op, asmName, isSingle, opName>,
  VOP3Dot_Realtriple<GFX12Gen, op, asmName, isSingle, opName>;


//===----------------------------------------------------------------------===//

include "VOPCInstructions.td"
include "VOP1Instructions.td"
include "VOP2Instructions.td"
include "VOP3Instructions.td"
include "VOP3PInstructions.td"
include "VOPDInstructions.td"

class ClassPat<Instruction inst, ValueType vt> : GCNPat <
  (is_fpclass (vt (VOP3ModsNonCanonicalizing vt:$src0, i32:$src0_mods)), (i32 timm:$mask)),
  (inst i32:$src0_mods, vt:$src0, (V_MOV_B32_e32 timm:$mask))
>;

class ClassPat_t16<Instruction inst, ValueType vt> : GCNPat <
  (is_fpclass (vt (VOP3ModsNonCanonicalizing vt:$src0, i32:$src0_mods)), (i32 timm:$mask)),
  (inst i32:$src0_mods, vt:$src0, SRCMODS.NONE, (V_MOV_B32_e32 timm:$mask))
>;

def : ClassPat<V_CMP_CLASS_F16_e64, f16> {
  let OtherPredicates = [Has16BitInsts];
  let True16Predicate = NotHasTrue16BitInsts;
}

def : ClassPat_t16<V_CMP_CLASS_F16_t16_e64, f16> {
  let True16Predicate = UseRealTrue16Insts;
}

def : ClassPat_t16<V_CMP_CLASS_F16_fake16_e64, f16> {
  let True16Predicate = UseFakeTrue16Insts;
}

def : ClassPat<V_CMP_CLASS_F32_e64, f32>;
def : ClassPat<V_CMP_CLASS_F64_e64, f64>;

class VOPInfoTable <string Format> : GenericTable {
  let FilterClass = Format # "_Real";
  let CppTypeName = "VOPInfo";
  let Fields = ["Opcode", "IsSingle"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "get" # Format # "OpcodeHelper";
}

def VOP1InfoTable : VOPInfoTable<"VOP1">;
def VOP2InfoTable : VOPInfoTable<"VOP2">;
def VOP3InfoTable : VOPInfoTable<"VOP3">;

class VOPC64Table <string Format> : GenericTable {
  let FilterClass = "VOPC64_" # Format;
  let CppTypeName = "VOPC64DPPInfo";
  let Fields = ["Opcode"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "isVOPC64" # Format # "OpcodeHelper";
}

def VOPC64DPPTable : VOPC64Table<"DPP">;
def VOPC64DPP8Table : VOPC64Table<"DPP8">;

class AsmOnlyInfoTable <string Format, string Class>: GenericTable {
  let FilterClass = Class;
  let FilterClassField = "isAsmParserOnly";
  let CppTypeName = Format # "DPPAsmOnlyInfo";
  let Fields = ["Opcode"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "is" # Format # "AsmOnlyOpcodeHelper";
}

def VOPCAsmOnlyInfoTable : AsmOnlyInfoTable <"VOPC", "VOPC_DPPe_Common">;

def VOPTrue16Table : GenericTable {
  let FilterClass = "VOP_Pseudo";
  let CppTypeName = "VOPTrue16Info";
  let Fields = ["Opcode", "IsTrue16"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTrue16OpcodeHelper";
}
