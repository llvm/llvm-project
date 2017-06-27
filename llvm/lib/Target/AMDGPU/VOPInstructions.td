//===-- VOPInstructions.td - Vector Instruction Defintions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// dummies for outer let
class LetDummies {
  bit isCommutable;
  bit isConvertibleToThreeAddress;
  bit isMoveImm;
  bit isReMaterializable;
  bit isAsCheapAsAMove;
  bit VOPAsmPrefer32Bit;
  Predicate SubtargetPredicate;
  string Constraints;
  string DisableEncoding;
  list<SchedReadWrite> SchedRW;
  list<Register> Uses;
  list<Register> Defs;
}

class VOP <string opName> {
  string OpName = opName;
}

class VOPAnyCommon <dag outs, dag ins, string asm, list<dag> pattern> :
    InstSI <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VALU = 1;
  let Uses = [EXEC];
}

class VOP3Common <dag outs, dag ins, string asm = "",
                  list<dag> pattern = [], bit HasMods = 0,
                  bit VOP3Only = 0> :
  VOPAnyCommon <outs, ins, asm, pattern> {

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always preferred, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;

  let AsmMatchConverter =
    !if(!eq(VOP3Only,1),
        "cvtVOP3",
        !if(!eq(HasMods,1), "cvtVOP3_2_mod", ""));

  let AsmVariantName = AMDGPUAsmVariants.VOP3;

  let isCodeGenOnly = 0;

  int Size = 8;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;
}

class VOP3_Pseudo <string opName, VOPProfile P, list<dag> pattern = [],
                   bit VOP3Only = 0, bit isVOP3P = 0> :
  InstSI <P.Outs64, !if(!and(isVOP3P, P.IsPacked), P.InsVOP3P, P.Ins64), "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.NONE>,
  MnemonicAlias<opName#"_e64", opName> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = !if(!and(isVOP3P, P.IsPacked), P.AsmVOP3P, P.Asm64);

  let Size = 8;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SubtargetPredicate = isGCN;

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
  let Uses = [EXEC];

  let AsmVariantName = AMDGPUAsmVariants.VOP3;
  let AsmMatchConverter =
    !if(!eq(VOP3Only,1),
        !if(!and(P.IsPacked, isVOP3P), "cvtVOP3P", "cvtVOP3"),
        !if(!eq(P.HasModifiers, 1),
            "cvtVOP3_2_mod",
            !if(!eq(P.HasOMod, 1), "cvtVOP3OMod", "")
        )
    );

  VOPProfile Pfl = P;
}

class VOP3P_Pseudo <string opName, VOPProfile P, list<dag> pattern = []> :
  VOP3_Pseudo<opName, P, pattern, 1, 1> {
  let VOP3P = 1;
}

class VOP3_Real <VOP3_Pseudo ps, int EncodingFamily> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let UseNamedOperandTable = 1;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
}

// XXX - Is there any reason to distingusih this from regular VOP3
// here?
class VOP3P_Real<VOP3P_Pseudo ps, int EncodingFamily> :
  VOP3_Real<ps, EncodingFamily>;

class VOP3a<VOPProfile P> : Enc64 {
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
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

class VOP3a_si <bits<9> op, VOPProfile P> : VOP3a<P> {
  let Inst{25-17} = op;
  let Inst{11}    = !if(P.HasClamp, clamp{0}, 0);
}

class VOP3a_vi <bits<10> op, VOPProfile P> : VOP3a<P> {
  let Inst{25-16} = op;
  let Inst{15}    = !if(P.HasClamp, clamp{0}, 0);
}

class VOP3e_si <bits<9> op, VOPProfile P> : VOP3a_si <op, P> {
  bits<8> vdst;
  let Inst{7-0} = !if(P.EmitDst, vdst{7-0}, 0);
}

class VOP3e_vi <bits<10> op, VOPProfile P> : VOP3a_vi <op, P> {
  bits<8> vdst;
  let Inst{7-0} = !if(P.EmitDst, vdst{7-0}, 0);
}

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

class VOP3Pe <bits<10> op, VOPProfile P> : Enc64 {
  bits<8> vdst;
  // neg, neg_hi, op_sel put in srcN_modifiers
  bits<4> src0_modifiers;
  bits<9> src0;
  bits<4> src1_modifiers;
  bits<9> src1;
  bits<4> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;

  let Inst{7-0} = vdst;
  let Inst{8} = !if(P.HasSrc0Mods, src0_modifiers{1}, 0); // neg_hi src0
  let Inst{9} = !if(P.HasSrc1Mods, src1_modifiers{1}, 0); // neg_hi src1
  let Inst{10} = !if(P.HasSrc2Mods, src2_modifiers{1}, 0); // neg_hi src2

  let Inst{11} = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{2}, 0); // op_sel(0)
  let Inst{12} = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{2}, 0); // op_sel(1)
  let Inst{13} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{2}, 0); // op_sel(2)

  let Inst{14} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{3}, 0); // op_sel_hi(2)

  let Inst{15} = !if(P.HasClamp, clamp{0}, 0);

  let Inst{25-16} = op;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{59}    = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{3}, 0); // op_sel_hi(0)
  let Inst{60}    = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{3}, 0); // op_sel_hi(1)
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0); // neg (lo)
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0); // neg (lo)
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0); // neg (lo)
}

class VOP3be_si <bits<9> op, VOPProfile P> : VOP3be<P> {
  let Inst{25-17} = op;
}

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
  bits<2> src0_modifiers; // float: {abs,neg}, int {sext}
  bits<3> src1_sel;
  bits<2> src1_modifiers;
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{42-40} = !if(P.EmitDst, dst_sel{2-0}, SDWA.DWORD);
  let Inst{44-43} = !if(P.EmitDst, dst_unused{1-0}, SDWA.UNUSED_PRESERVE);
  let Inst{45}    = !if(P.HasSDWAClamp, clamp{0}, 0);
  let Inst{50-48} = !if(P.HasSrc0, src0_sel{2-0}, SDWA.DWORD);
  let Inst{51}    = !if(P.HasSrc0IntMods, src0_modifiers{0}, 0);
  let Inst{53-52} = !if(P.HasSrc0FloatMods, src0_modifiers{1-0}, 0);
  let Inst{58-56} = !if(P.HasSrc1, src1_sel{2-0}, SDWA.DWORD);
  let Inst{59}    = !if(P.HasSrc1IntMods, src1_modifiers{0}, 0);
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
  bits<2> src0_modifiers; // float: {abs,neg}, int {sext}
  bits<3> src1_sel;
  bits<2> src1_modifiers;
  bits<1> src1_sgpr;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{50-48} = !if(P.HasSrc0, src0_sel{2-0}, SDWA.DWORD);
  let Inst{51}    = !if(P.HasSrc0IntMods, src0_modifiers{0}, 0);
  let Inst{53-52} = !if(P.HasSrc0FloatMods, src0_modifiers{1-0}, 0);
  let Inst{55}    = !if(P.HasSrc0, src0{8}, 0);
  let Inst{58-56} = !if(P.HasSrc1, src1_sel{2-0}, SDWA.DWORD);
  let Inst{59}    = !if(P.HasSrc1IntMods, src1_modifiers{0}, 0);
  let Inst{61-60} = !if(P.HasSrc1FloatMods, src1_modifiers{1-0}, 0);
  let Inst{63}    = 0; // src1_sgpr - should be specified in subclass
}

// gfx9 SDWA-A
class VOP_SDWA9Ae<VOPProfile P> : VOP_SDWA9e<P> {
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;
  bits<2> omod;

  let Inst{42-40} = !if(P.EmitDst, dst_sel{2-0}, SDWA.DWORD);
  let Inst{44-43} = !if(P.EmitDst, dst_unused{1-0}, SDWA.UNUSED_PRESERVE);
  let Inst{45}    = !if(P.HasSDWAClamp, clamp{0}, 0);
  let Inst{47-46} = !if(P.HasSDWAOMod, omod{1-0}, 0);
}

// gfx9 SDWA-B
class VOP_SDWA9Be<VOPProfile P> : VOP_SDWA9e<P> {
  bits<8> sdst; // {vcc_sdst{0}, sdst{6-0}}

  let Inst{46-40} = !if(P.EmitDst, sdst{6-0}, 0);
  let Inst{47} = !if(P.EmitDst, sdst{7}, 0);
}

class VOP_SDWA_Pseudo <string opName, VOPProfile P, list<dag> pattern=[]> :
  InstSI <P.OutsSDWA, P.InsSDWA, "", pattern>,
  VOP <opName>,
  SIMCInstr <opName#"_sdwa", SIEncodingFamily.NONE>,
  MnemonicAlias <opName#"_sdwa", opName> {

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
  let Uses = [EXEC];

  let SubtargetPredicate = !if(P.HasExt, HasSDWA, DisableInst);
  let AssemblerPredicate = !if(P.HasExt, HasSDWA, DisableInst);
  let AsmVariantName = !if(P.HasExt, AMDGPUAsmVariants.SDWA,
                                     AMDGPUAsmVariants.Disable);
  let DecoderNamespace = "SDWA";

  VOPProfile Pfl = P;
}

class VOP_SDWA_Real <VOP_SDWA_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SDWA> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let SchedRW = ps.SchedRW;
  let hasSideEffects = ps.hasSideEffects;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  // Copy relevant pseudo op flags
  let SubtargetPredicate   = ps.SubtargetPredicate;
  let AssemblerPredicate   = ps.AssemblerPredicate;
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let AsmVariantName       = ps.AsmVariantName;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let DecoderNamespace     = ps.DecoderNamespace;
  let Constraints          = ps.Constraints;
  let DisableEncoding      = ps.DisableEncoding;
  let TSFlags              = ps.TSFlags;
}

class VOP_SDWA9_Real <VOP_SDWA_Pseudo ps> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands9, []>,
  SIMCInstr <ps.PseudoInstr, SIEncodingFamily.SDWA9> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Defs = ps.Defs;
  let Uses = ps.Uses;
  let SchedRW = ps.SchedRW;
  let hasSideEffects = ps.hasSideEffects;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  let SubtargetPredicate = !if(ps.Pfl.HasSDWA9, HasSDWA9, DisableInst);
  let AssemblerPredicate = !if(ps.Pfl.HasSDWA9, HasSDWA9, DisableInst);
  let AsmVariantName = !if(ps.Pfl.HasSDWA9, AMDGPUAsmVariants.SDWA9,
                                            AMDGPUAsmVariants.Disable);
  let DecoderNamespace = "SDWA9";

  // Copy relevant pseudo op flags
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Constraints          = ps.Constraints;
  let DisableEncoding      = ps.DisableEncoding;
  let TSFlags              = ps.TSFlags;
}

class VOP_DPPe<VOPProfile P> : Enc64 {
  bits<2> src0_modifiers;
  bits<8> src0;
  bits<2> src1_modifiers;
  bits<9> dpp_ctrl;
  bits<1> bound_ctrl;
  bits<4> bank_mask;
  bits<4> row_mask;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{48-40} = dpp_ctrl;
  let Inst{51}    = bound_ctrl;
  let Inst{52}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0); // src0_neg
  let Inst{53}    = !if(P.HasSrc0Mods, src0_modifiers{1}, 0); // src0_abs
  let Inst{54}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0); // src1_neg
  let Inst{55}    = !if(P.HasSrc1Mods, src1_modifiers{1}, 0); // src1_abs
  let Inst{59-56} = bank_mask;
  let Inst{63-60} = row_mask;
}

class VOP_DPP <string OpName, VOPProfile P> :
  InstSI <P.OutsDPP, P.InsDPP, OpName#P.AsmDPP, []>,
  VOP_DPPe<P> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;

  let VALU = 1;
  let DPP = 1;
  let Size = 8;

  let AsmMatchConverter = !if(!eq(P.HasModifiers,1), "cvtDPP", "");
  let SubtargetPredicate = HasDPP;
  let AssemblerPredicate = !if(P.HasExt, HasDPP, DisableInst);
  let AsmVariantName = !if(P.HasExt, AMDGPUAsmVariants.DPP,
                                     AMDGPUAsmVariants.Disable);
  let DecoderNamespace = "DPP";
}

include "VOPCInstructions.td"
include "VOP1Instructions.td"
include "VOP2Instructions.td"
include "VOP3Instructions.td"
include "VOP3PInstructions.td"
