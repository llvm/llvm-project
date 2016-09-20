//===-- VOPInstructions.td - Vector Instruction Defintions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class VOP3_PseudoNew <string opName, VOPProfile P, list<dag> pattern, bit VOP3Only = 0> :
  InstSI <P.Outs64, P.Ins64, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e64", SIEncodingFamily.NONE>,
  MnemonicAlias<opName#"_e64", opName> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = P.Asm64;

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
  // but standalone patterns are almost always prefered, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;
  let VALU = 1;
  let Uses = [EXEC];

  let AsmVariantName = AMDGPUAsmVariants.VOP3;
  let AsmMatchConverter =
    !if(!eq(VOP3Only,1),
        "cvtVOP3",
        !if(!eq(P.HasModifiers, 1), "cvtVOP3_2_mod", ""));

  VOPProfile Pfl = P;
}

class VOP3_Real <VOP3_PseudoNew ps, int EncodingFamily> :
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
}

class VOP3aNew<VOPProfile P> : Enc64 {
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
  let Inst{40-32} = src0;
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3a_siNew <bits<9> op, VOPProfile P> : VOP3aNew<P> {
  let Inst{25-17} = op;
  let Inst{11}    = !if(P.HasClamp, clamp, 0);
}

class VOP3a_viNew <bits<10> op, VOPProfile P> : VOP3aNew<P> {
  let Inst{25-16} = op;
  let Inst{15}    = !if(P.HasClamp, clamp, 0);
}

class VOP3e_siNew <bits<9> op, VOPProfile P> : VOP3a_siNew <op, P> {
  bits<8> vdst;
  let Inst{7-0} = vdst;
}

class VOP3e_viNew <bits<10> op, VOPProfile P> : VOP3a_viNew <op, P> {
  bits<8> vdst;
  let Inst{7-0} = vdst;
}

class VOP3beNew <VOPProfile P> : Enc64 {
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

class VOP3be_siNew <bits<9> op, VOPProfile P> : VOP3beNew<P> {
  let Inst{25-17} = op;
}

class VOP3be_viNew <bits<10> op, VOPProfile P> : VOP3beNew<P> {
  bits<1> clamp;
  let Inst{25-16} = op;
  let Inst{15}    = !if(P.HasClamp, clamp, 0);
}

class VOP_SDWAeNew<VOPProfile P> : Enc64 {
  bits<8> src0;
  bits<3> src0_sel;
  bits<2> src0_fmodifiers; // {abs,neg}
  bits<1> src0_imodifiers; // sext
  bits<3> src1_sel;
  bits<2> src1_fmodifiers;
  bits<1> src1_imodifiers;
  bits<3> dst_sel;
  bits<2> dst_unused;
  bits<1> clamp;

  bits<3> SDWA_DWORD = 6;
  bits<2> SDWA_UNUSED_PRESERVE = 2;

  let Inst{39-32} = !if(P.HasSrc0, src0{7-0}, 0);
  let Inst{42-40} = !if(P.HasDst, dst_sel{2-0}, SDWA_DWORD{2-0});
  let Inst{44-43} = !if(P.HasDst, dst_unused{1-0}, SDWA_UNUSED_PRESERVE{1-0});
  let Inst{45}    = !if(P.HasSDWAClamp, clamp{0}, 0);
  let Inst{50-48} = !if(P.HasSrc0, src0_sel{2-0}, SDWA_DWORD{2-0});
  let Inst{53-52} = !if(P.HasSrc0Mods, src0_fmodifiers{1-0}, 0);
  let Inst{51}    = !if(P.HasSrc0IntMods, src0_imodifiers{0}, 0);
  let Inst{58-56} = !if(P.HasSrc1, src1_sel{2-0}, SDWA_DWORD{2-0});
  let Inst{61-60} = !if(P.HasSrc1Mods, src1_fmodifiers{1-0}, 0);
  let Inst{59}    = !if(P.HasSrc1IntMods, src1_imodifiers{0}, 0);
}

include "VOPCInstructions.td"
include "VOP3Instructions.td"
