//===-- VOP2Instructions.td - Vector Instruction Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VOP2 Classes
//===----------------------------------------------------------------------===//

class VOP2e <bits<6> op, VOPProfile P> : Enc32 {
  bits<8> vdst;
  bits<9> src0;
  bits<8> src1;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; //encoding
}

class VOP2_MADKe <bits<6> op, VOPProfile P> : Enc64 {
  bits<8>  vdst;
  bits<9>  src0;
  bits<8>  src1;
  bits<32> imm;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
  let Inst{63-32} = imm;
}

class VOP2_SDWAe <bits<6> op, VOPProfile P> : VOP_SDWAe <P> {
  bits<8> vdst;
  bits<8> src1;

  let Inst{8-0}   = 0xf9; // sdwa
  let Inst{16-9}  = !if(P.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
}

class VOP2_SDWA9Ae <bits<6> op, VOPProfile P> : VOP_SDWA9Ae <P> {
  bits<8> vdst;
  bits<9> src1;

  let Inst{8-0}   = 0xf9; // sdwa
  let Inst{16-9}  = !if(P.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
  let Inst{63}    = !if(P.HasSrc1, src1{8}, 0); // src1_sgpr
}

class VOP2_Pseudo <string opName, VOPProfile P, list<dag> pattern=[], string suffix = "_e32"> :
  VOP_Pseudo <opName, suffix, P, P.Outs32, P.Ins32, "", pattern> {

  let AsmOperands = P.Asm32;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let ReadsModeReg = !or(isFloatType<P.DstVT>.ret, isFloatType<P.Src0VT>.ret);

  let mayRaiseFPException = ReadsModeReg;

  let VOP2 = 1;
  let VALU = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);

  let AsmVariantName = AMDGPUAsmVariants.Default;
}

class VOP2_Real <VOP2_Pseudo ps, int EncodingFamily> :
  VOP_Real <ps>,
  InstSI <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  SIMCInstr <ps.PseudoInstr, EncodingFamily> {

  let VALU = 1;
  let VOP2 = 1;
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  // copy relevant pseudo op flags
  let SubtargetPredicate = ps.SubtargetPredicate;
  let OtherPredicates    = ps.OtherPredicates;
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let AsmVariantName     = ps.AsmVariantName;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
  let mayLoad              = ps.mayLoad;
  let mayStore             = ps.mayStore;
}

class VOP2_SDWA_Pseudo <string OpName, VOPProfile P, list<dag> pattern=[]> :
  VOP_SDWA_Pseudo <OpName, P, pattern> {
  let AsmMatchConverter = "cvtSdwaVOP2";
}

class VOP2_DPP_Pseudo <string OpName, VOPProfile P, list<dag> pattern=[]> :
  VOP_DPP_Pseudo <OpName, P, pattern> {
}


class getVOP2Pat64 <SDPatternOperator node, VOPProfile P> : LetDummies {
  list<dag> ret = !if(P.HasModifiers,
    [(set P.DstVT:$vdst,
      (node (P.Src0VT
              !if(P.HasOMod,
                  (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
                  (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp))),
            (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
    [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]);
}

multiclass VOP2Inst_e32<string opName,
                        VOPProfile P,
                        SDPatternOperator node = null_frag,
                        string revOp = opName,
                        bit GFX9Renamed = 0> {
  let renamedInGFX9 = GFX9Renamed in {
    def _e32 : VOP2_Pseudo <opName, P, VOPPatOrNull<node,P>.ret>,
               Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;
  } // End renamedInGFX9 = GFX9Renamed
}

multiclass VOP2Inst_e64<string opName,
                        VOPProfile P,
                        SDPatternOperator node = null_frag,
                        string revOp = opName,
                        bit GFX9Renamed = 0> {
  let renamedInGFX9 = GFX9Renamed in {
    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
  } // End renamedInGFX9 = GFX9Renamed
}

multiclass VOP2Inst_sdwa<string opName,
                         VOPProfile P,
                         bit GFX9Renamed = 0> {
  let renamedInGFX9 = GFX9Renamed in {
    foreach _ = BoolToList<P.HasExtSDWA>.ret in
      def _sdwa : VOP2_SDWA_Pseudo <opName, P>;
  } // End renamedInGFX9 = GFX9Renamed
}

multiclass VOP2Inst<string opName,
                    VOPProfile P,
                    SDPatternOperator node = null_frag,
                    string revOp = opName,
                    bit GFX9Renamed = 0> :
    VOP2Inst_e32<opName, P, node, revOp, GFX9Renamed>,
    VOP2Inst_e64<opName, P, node, revOp, GFX9Renamed>,
    VOP2Inst_sdwa<opName, P, GFX9Renamed> {
  let renamedInGFX9 = GFX9Renamed in {
    foreach _ = BoolToList<P.HasExtDPP>.ret in
      def _dpp  : VOP2_DPP_Pseudo <opName, P>;
  }
}

multiclass VOP2bInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit GFX9Renamed = 0,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {
  let renamedInGFX9 = GFX9Renamed in {
    let SchedRW = [Write32Bit, WriteSALU] in {
      let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]), Defs = [VCC] in {
        def _e32 : VOP2_Pseudo <opName, P, VOPPatOrNull<node,P>.ret>,
                   Commutable_REV<revOp#"_e32", !eq(revOp, opName)> {
          let usesCustomInserter = true;
        }

        foreach _ = BoolToList<P.HasExtSDWA>.ret in
          def _sdwa  : VOP2_SDWA_Pseudo <opName, P> {
            let AsmMatchConverter = "cvtSdwaVOP2b";
          }
        foreach _ = BoolToList<P.HasExtDPP>.ret in
          def _dpp  : VOP2_DPP_Pseudo <opName, P>;
      }

      def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
                 Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
    }
  }
}

class VOP2bInstAlias <VOP2_Pseudo ps, Instruction inst,
                      string OpName, string opnd> :
  InstAlias <OpName#" "#!subst("vcc", opnd, ps.Pfl.Asm32),
             (inst ps.Pfl.DstRC:$vdst, ps.Pfl.Src0RC32:$src0,
                   ps.Pfl.Src1RC32:$src1)>,
  PredicateControl {
}

multiclass VOP2bInstAliases<VOP2_Pseudo ps, VOP2_Real inst, string OpName> {
  let WaveSizePredicate = isWave32 in {
    def : VOP2bInstAlias<ps, inst, OpName, "vcc_lo">;
  }
  let WaveSizePredicate = isWave64 in {
    def : VOP2bInstAlias<ps, inst, OpName, "vcc">;
  }
}

multiclass VOP2eInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {

  let SchedRW = [Write32Bit] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]) in {
      def _e32 : VOP2_Pseudo <opName, P>,
                 Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;

      foreach _ = BoolToList<P.HasExtSDWA>.ret in
        def _sdwa : VOP2_SDWA_Pseudo <opName, P> {
          let AsmMatchConverter = "cvtSdwaVOP2e";
        }

      foreach _ = BoolToList<P.HasExtDPP>.ret in
        def _dpp  : VOP2_DPP_Pseudo <opName, P>;
    }

    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)> {
      let isReMaterializable = 1;
    }
  }
}

class VOP2eInstAlias <VOP2_Pseudo ps, Instruction inst, string opnd = ""> :
  InstAlias <ps.OpName#" "#ps.Pfl.Asm32#", "#opnd,
             (inst ps.Pfl.DstRC:$vdst, ps.Pfl.Src0RC32:$src0,
                   ps.Pfl.Src1RC32:$src1)>, PredicateControl;

class VOP2e64InstAlias <VOP3_Pseudo ps, Instruction inst> :
  InstAlias <ps.OpName#" "#ps.Pfl.Asm64,
             (inst ps.Pfl.DstRC:$vdst, VOPDstS64orS32:$sdst,
                   ps.Pfl.Src0RC32:$src0, ps.Pfl.Src1RC32:$src1, clampmod:$clamp)>,
  PredicateControl;

multiclass VOP2eInstAliases<VOP2_Pseudo ps, VOP2_Real inst> {
  let WaveSizePredicate = isWave32 in {
    def : VOP2eInstAlias<ps, inst, "vcc_lo">;
  }
  let WaveSizePredicate = isWave64 in {
    def : VOP2eInstAlias<ps, inst, "vcc">;
  }
}

class VOP_MADAK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = !if(!eq(vt.Size, 32),
                        (ins VSrc_f32_Deferred:$src0, VGPR_32:$src1, ImmOpType:$imm),
                        (ins VSrc_f16_Deferred:$src0, VGPR_32:$src1, ImmOpType:$imm));
  field string Asm32 = "$vdst, $src0, $src1, $imm";
  field bit HasExt = 0;
  let IsSingle = 1;
}

def VOP_MADAK_F16 : VOP_MADAK <f16>;
def VOP_MADAK_F32 : VOP_MADAK <f32>;

class VOP_MADMK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = (ins VSrc_f32_Deferred:$src0, ImmOpType:$imm, VGPR_32:$src1);
  field string Asm32 = "$vdst, $src0, $imm, $src1";
  field bit HasExt = 0;
  let IsSingle = 1;
}

def VOP_MADMK_F16 : VOP_MADMK <f16>;
def VOP_MADMK_F32 : VOP_MADMK <f32>;

class getRegisterOperandForVT<ValueType VT> {
  RegisterOperand ret = RegisterOperand<getVregSrcForVT<VT>.ret>;
}

// FIXME: Remove src2_modifiers. It isn't used, so is wasting memory
// and processing time but it makes it easier to convert to mad.
class VOP_MAC <ValueType vt0, ValueType vt1=vt0> : VOPProfile <[vt0, vt1, vt1, vt0]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1, getVregSrcForVT<Src2VT>.ret:$src2);
  let Ins64 = getIns64<Src0RC64, Src1RC64, getRegisterOperandForVT<Src2VT>.ret, 3,
                       0, HasModifiers, HasModifiers, HasOMod,
                       Src0Mod, Src1Mod, Src2Mod>.ret;
  let InsDPP = (ins Src0ModDPP:$src0_modifiers, Src0DPP:$src0,
                    Src1ModDPP:$src1_modifiers, Src1DPP:$src1,
                    getVregSrcForVT<Src2VT>.ret:$src2, // stub argument
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsDPP16 = !con(InsDPP, (ins FI:$fi));
  let InsDPP8 = (ins Src0ModDPP:$src0_modifiers, Src0DPP:$src0,
                     Src1ModDPP:$src1_modifiers, Src1DPP:$src1,
                     getVregSrcForVT<Src2VT>.ret:$src2, // stub argument
                     dpp8:$dpp8, FI:$fi);
  let InsSDWA = (ins Src0ModSDWA:$src0_modifiers, Src0SDWA:$src0,
                     Src1ModSDWA:$src1_modifiers, Src1SDWA:$src1,
                     getVregSrcForVT<Src2VT>.ret:$src2, // stub argument
                     clampmod:$clamp, omod:$omod,
                     dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);
  let Asm32 = getAsm32<1, 2, vt0>.ret;
  let Asm64 = getAsm64<1, 2, 0, HasModifiers, HasOMod, vt0>.ret;
  let AsmDPP = getAsmDPP<1, 2, HasModifiers, vt0>.ret;
  let AsmDPP16 = getAsmDPP16<1, 2, HasModifiers, vt0>.ret;
  let AsmDPP8 = getAsmDPP8<1, 2, 0, vt0>.ret;
  let AsmSDWA = getAsmSDWA<1, 2, vt0>.ret;
  let AsmSDWA9 = getAsmSDWA9<1, 1, 2, vt0>.ret;
  let HasSrc2 = 0;
  let HasSrc2Mods = 0;

  let HasExt = 1;
  let HasExtDPP = 1;
  let HasExtSDWA = 1;
  let HasExtSDWA9 = 0;
  let TieRegDPP = "$src2";
}

def VOP_MAC_F16 : VOP_MAC <f16>;
def VOP_MAC_F32 : VOP_MAC <f32>;
let HasExtDPP = 0 in
def VOP_MAC_LEGACY_F32 : VOP_MAC <f32>;
let HasExtSDWA = 0, HasExt64BitDPP = 1 in
def VOP_MAC_F64 : VOP_MAC <f64>;

class VOP_DOT_ACC<ValueType vt0, ValueType vt1> : VOP_MAC<vt0, vt1> {
  let HasClamp = 0;
  let HasExtSDWA = 0;
  let HasOpSel = 0;
  let IsPacked = 0;
}

def VOP_DOT_ACC_F32_V2F16 : VOP_DOT_ACC<f32, v2f16> {
  let Src0ModDPP = FPVRegInputMods;
  let Src1ModDPP = FPVRegInputMods;
}

def VOP_DOT_ACC_I32_I32   : VOP_DOT_ACC<i32, i32> {
  let HasSrc0Mods = 1;
  let HasSrc1Mods = 1;
}

// Write out to vcc or arbitrary SGPR.
def VOP2b_I32_I1_I32_I32 : VOPProfile<[i32, i32, i32, untyped], 0, /*EnableClamp=*/1> {
  let Asm32 = "$vdst, vcc, $src0, $src1";
  let Asm64 = "$vdst, $sdst, $src0, $src1$clamp";
  let AsmSDWA = "$vdst, vcc, $src0_modifiers, $src1_modifiers$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmSDWA9 = "$vdst, vcc, $src0_modifiers, $src1_modifiers$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmDPP = "$vdst, vcc, $src0, $src1 $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
  let AsmDPP8 = "$vdst, vcc, $src0, $src1 $dpp8$fi";
  let AsmDPP16 = AsmDPP#"$fi";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
}

// Write out to vcc or arbitrary SGPR and read in from vcc or
// arbitrary SGPR.
def VOP2b_I32_I1_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1], 0, /*EnableClamp=*/1> {
  let Asm32 = "$vdst, vcc, $src0, $src1, vcc";
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2$clamp";
  let AsmSDWA = "$vdst, vcc, $src0_modifiers, $src1_modifiers, vcc$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmSDWA9 = "$vdst, vcc, $src0_modifiers, $src1_modifiers, vcc$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmDPP = "$vdst, vcc, $src0, $src1, vcc $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
  let AsmDPP8 = "$vdst, vcc, $src0, $src1, vcc $dpp8$fi";
  let AsmDPP16 = AsmDPP#"$fi";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);

  let InsSDWA = (ins Src0ModSDWA:$src0_modifiers, Src0SDWA:$src0,
                     Src1ModSDWA:$src1_modifiers, Src1SDWA:$src1,
                     clampmod:$clamp,
                     dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);

  let InsDPP = (ins DstRCDPP:$old,
                    Src0DPP:$src0,
                    Src1DPP:$src1,
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsDPP16 = !con(InsDPP, (ins FI:$fi));

  let HasExt = 1;
  let HasExtDPP = 1;
  let HasExtSDWA = 1;
  let HasExtSDWA9 = 1;
}

// Read in from vcc or arbitrary SGPR.
def VOP2e_I32_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1], /*EnableF32SrcMods=*/1> {
  let Asm32 = "$vdst, $src0, $src1";
  let Asm64 = "$vdst, $src0_modifiers, $src1_modifiers, $src2";
  let AsmSDWA = "$vdst, $src0_modifiers, $src1_modifiers, vcc$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmSDWA9 = "$vdst, $src0_modifiers, $src1_modifiers, vcc$clamp $dst_sel $dst_unused $src0_sel $src1_sel";
  let AsmDPP = "$vdst, $src0, $src1, vcc $dpp_ctrl$row_mask$bank_mask$bound_ctrl";
  let AsmDPP8 = "$vdst, $src0, $src1, vcc $dpp8$fi";
  let AsmDPP16 = AsmDPP#"$fi";

  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);

  let InsSDWA = (ins Src0ModSDWA:$src0_modifiers, Src0SDWA:$src0,
                     Src1ModSDWA:$src1_modifiers, Src1SDWA:$src1,
                     clampmod:$clamp,
                     dst_sel:$dst_sel, dst_unused:$dst_unused,
                     src0_sel:$src0_sel, src1_sel:$src1_sel);

  let InsDPP = (ins DstRCDPP:$old,
                    Src0ModDPP:$src0_modifiers, Src0DPP:$src0,
                    Src1ModDPP:$src1_modifiers, Src1DPP:$src1,
                    dpp_ctrl:$dpp_ctrl, row_mask:$row_mask,
                    bank_mask:$bank_mask, bound_ctrl:$bound_ctrl);
  let InsDPP16 = !con(InsDPP, (ins FI:$fi));

  let HasExt = 1;
  let HasExtDPP = 1;
  let HasExtSDWA = 1;
  let HasExtSDWA9 = 1;
}

def VOP_READLANE : VOPProfile<[i32, i32, i32]> {
  let Outs32 = (outs SReg_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins VRegOrLds_32:$src0, SCSrc_b32:$src1);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;

  let HasExt = 0;
  let HasExtDPP = 0;
  let HasExt64BitDPP = 0;
  let HasExtSDWA = 0;
  let HasExtSDWA9 = 0;
}

def VOP_WRITELANE : VOPProfile<[i32, i32, i32, i32]> {
  let Outs32 = (outs VGPR_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins SCSrc_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;
  let HasSrc2 = 0;
  let HasSrc2Mods = 0;

  let HasExt = 0;
  let HasExtDPP = 0;
  let HasExt64BitDPP = 0;
  let HasExtSDWA = 0;
  let HasExtSDWA9 = 0;
}

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

defm V_CNDMASK_B32 : VOP2eInst <"v_cndmask_b32", VOP2e_I32_I32_I32_I1>;
let SubtargetPredicate = HasMadMacF32Insts, isReMaterializable = 1 in
def V_MADMK_F32 : VOP2_Pseudo <"v_madmk_f32", VOP_MADMK_F32, []>;

let isCommutable = 1 in {
let isReMaterializable = 1 in {
defm V_ADD_F32 : VOP2Inst <"v_add_f32", VOP_F32_F32_F32, any_fadd>;
defm V_SUB_F32 : VOP2Inst <"v_sub_f32", VOP_F32_F32_F32, any_fsub>;
defm V_SUBREV_F32 : VOP2Inst <"v_subrev_f32", VOP_F32_F32_F32, null_frag, "v_sub_f32">;
defm V_MUL_LEGACY_F32 : VOP2Inst <"v_mul_legacy_f32", VOP_F32_F32_F32, AMDGPUfmul_legacy>;
defm V_MUL_F32 : VOP2Inst <"v_mul_f32", VOP_F32_F32_F32, any_fmul>;
defm V_MUL_I32_I24 : VOP2Inst <"v_mul_i32_i24", VOP_I32_I32_I32_ARITH, AMDGPUmul_i24>;
defm V_MUL_HI_I32_I24 : VOP2Inst <"v_mul_hi_i32_i24", VOP_PAT_GEN<VOP_I32_I32_I32, 2>, AMDGPUmulhi_i24>;
defm V_MUL_U32_U24 : VOP2Inst <"v_mul_u32_u24", VOP_I32_I32_I32_ARITH, AMDGPUmul_u24>;
defm V_MUL_HI_U32_U24 : VOP2Inst <"v_mul_hi_u32_u24", VOP_PAT_GEN<VOP_I32_I32_I32, 2>, AMDGPUmulhi_u24>;
defm V_MIN_F32 : VOP2Inst <"v_min_f32", VOP_F32_F32_F32, fminnum_like>;
defm V_MAX_F32 : VOP2Inst <"v_max_f32", VOP_F32_F32_F32, fmaxnum_like>;
defm V_MIN_I32 : VOP2Inst <"v_min_i32", VOP_PAT_GEN<VOP_I32_I32_I32>, smin>;
defm V_MAX_I32 : VOP2Inst <"v_max_i32", VOP_PAT_GEN<VOP_I32_I32_I32>, smax>;
defm V_MIN_U32 : VOP2Inst <"v_min_u32", VOP_PAT_GEN<VOP_I32_I32_I32>, umin>;
defm V_MAX_U32 : VOP2Inst <"v_max_u32", VOP_PAT_GEN<VOP_I32_I32_I32>, umax>;
defm V_LSHRREV_B32 : VOP2Inst <"v_lshrrev_b32", VOP_I32_I32_I32, lshr_rev, "v_lshr_b32">;
defm V_ASHRREV_I32 : VOP2Inst <"v_ashrrev_i32", VOP_I32_I32_I32, ashr_rev, "v_ashr_i32">;
defm V_LSHLREV_B32 : VOP2Inst <"v_lshlrev_b32", VOP_I32_I32_I32, lshl_rev, "v_lshl_b32">;
defm V_AND_B32 : VOP2Inst <"v_and_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, and>;
defm V_OR_B32 : VOP2Inst <"v_or_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, or>;
defm V_XOR_B32 : VOP2Inst <"v_xor_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, xor>;
} // End isReMaterializable = 1

let mayRaiseFPException = 0 in {
let OtherPredicates = [HasMadMacF32Insts] in {
let Constraints = "$vdst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F32 : VOP2Inst <"v_mac_f32", VOP_MAC_F32>;

let SubtargetPredicate = isGFX6GFX7GFX10 in
defm V_MAC_LEGACY_F32 : VOP2Inst <"v_mac_legacy_f32", VOP_MAC_LEGACY_F32>;
} // End Constraints = "$vdst = $src2", DisableEncoding="$src2",
  //     isConvertibleToThreeAddress = 1

let isReMaterializable = 1 in
def V_MADAK_F32 : VOP2_Pseudo <"v_madak_f32", VOP_MADAK_F32, []>;
} // End OtherPredicates = [HasMadMacF32Insts]
} // End mayRaiseFPException = 0

// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.
defm V_ADD_CO_U32 : VOP2bInst <"v_add_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_add_co_u32", 1>;
defm V_SUB_CO_U32 : VOP2bInst <"v_sub_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_sub_co_u32", 1>;
defm V_SUBREV_CO_U32 : VOP2bInst <"v_subrev_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_sub_co_u32", 1>;
defm V_ADDC_U32 : VOP2bInst <"v_addc_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_addc_u32", 1>;
defm V_SUBB_U32 : VOP2bInst <"v_subb_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32", 1>;
defm V_SUBBREV_U32 : VOP2bInst <"v_subbrev_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32", 1>;


let SubtargetPredicate = HasAddNoCarryInsts, isReMaterializable = 1 in {
defm V_ADD_U32 : VOP2Inst <"v_add_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_add_u32", 1>;
defm V_SUB_U32 : VOP2Inst <"v_sub_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_sub_u32", 1>;
defm V_SUBREV_U32 : VOP2Inst <"v_subrev_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_sub_u32", 1>;
}

} // End isCommutable = 1

// These are special and do not read the exec mask.
let isConvergent = 1, Uses = []<Register> in {
def V_READLANE_B32 : VOP2_Pseudo<"v_readlane_b32", VOP_READLANE,
  [(set i32:$vdst, (int_amdgcn_readlane i32:$src0, i32:$src1))]>;

let Constraints = "$vdst = $vdst_in", DisableEncoding="$vdst_in" in {
def V_WRITELANE_B32 : VOP2_Pseudo<"v_writelane_b32", VOP_WRITELANE,
  [(set i32:$vdst, (int_amdgcn_writelane i32:$src0, i32:$src1, i32:$vdst_in))]>;
} // End $vdst = $vdst_in, DisableEncoding $vdst_in
} // End isConvergent = 1

let isReMaterializable = 1 in {
defm V_BFM_B32 : VOP2Inst <"v_bfm_b32", VOP_NO_EXT<VOP_I32_I32_I32>>;
defm V_BCNT_U32_B32 : VOP2Inst <"v_bcnt_u32_b32", VOP_NO_EXT<VOP_I32_I32_I32>, add_ctpop>;
defm V_MBCNT_LO_U32_B32 : VOP2Inst <"v_mbcnt_lo_u32_b32", VOP_NO_EXT<VOP_I32_I32_I32>, int_amdgcn_mbcnt_lo>;
defm V_MBCNT_HI_U32_B32 : VOP2Inst <"v_mbcnt_hi_u32_b32", VOP_NO_EXT<VOP_I32_I32_I32>, int_amdgcn_mbcnt_hi>;
defm V_LDEXP_F32 : VOP2Inst <"v_ldexp_f32", VOP_NO_EXT<VOP_F32_F32_I32>, AMDGPUldexp>;

let ReadsModeReg = 0, mayRaiseFPException = 0 in {
defm V_CVT_PKNORM_I16_F32 : VOP2Inst <"v_cvt_pknorm_i16_f32", VOP_NO_EXT<VOP_V2I16_F32_F32>, AMDGPUpknorm_i16_f32>;
defm V_CVT_PKNORM_U16_F32 : VOP2Inst <"v_cvt_pknorm_u16_f32", VOP_NO_EXT<VOP_V2I16_F32_F32>, AMDGPUpknorm_u16_f32>;
}

defm V_CVT_PKRTZ_F16_F32 : VOP2Inst <"v_cvt_pkrtz_f16_f32", VOP_NO_EXT<VOP_V2F16_F32_F32>, AMDGPUpkrtz_f16_f32>;
defm V_CVT_PK_U16_U32 : VOP2Inst <"v_cvt_pk_u16_u32", VOP_NO_EXT<VOP_V2I16_I32_I32>, AMDGPUpk_u16_u32>;
defm V_CVT_PK_I16_I32 : VOP2Inst <"v_cvt_pk_i16_i32", VOP_NO_EXT<VOP_V2I16_I32_I32>, AMDGPUpk_i16_i32>;


let SubtargetPredicate = isGFX6GFX7 in {
defm V_MIN_LEGACY_F32 : VOP2Inst <"v_min_legacy_f32", VOP_F32_F32_F32, AMDGPUfmin_legacy>;
defm V_MAX_LEGACY_F32 : VOP2Inst <"v_max_legacy_f32", VOP_F32_F32_F32, AMDGPUfmax_legacy>;
} // End SubtargetPredicate = isGFX6GFX7

let isCommutable = 1 in {
let SubtargetPredicate = isGFX6GFX7 in {
defm V_LSHR_B32 : VOP2Inst <"v_lshr_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, srl>;
defm V_ASHR_I32 : VOP2Inst <"v_ashr_i32", VOP_PAT_GEN<VOP_I32_I32_I32>, sra>;
defm V_LSHL_B32 : VOP2Inst <"v_lshl_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, shl>;
} // End SubtargetPredicate = isGFX6GFX7
} // End isCommutable = 1
} // End isReMaterializable = 1

defm V_CVT_PKACCUM_U8_F32 : VOP2Inst <"v_cvt_pkaccum_u8_f32", VOP_NO_EXT<VOP_I32_F32_I32>>; // TODO: set "Uses = dst"

class DivergentBinOp<SDPatternOperator Op, VOP_Pseudo Inst> :
  GCNPat<
      (getDivergentFrag<Op>.ret Inst.Pfl.Src0VT:$src0, Inst.Pfl.Src1VT:$src1),
      !if(!cast<Commutable_REV>(Inst).IsOrig,
        (Inst $src0, $src1),
        (Inst $src1, $src0)
      )
  >;

class DivergentClampingBinOp<SDPatternOperator Op, VOP_Pseudo Inst> :
  GCNPat<
      (getDivergentFrag<Op>.ret Inst.Pfl.Src0VT:$src0, Inst.Pfl.Src1VT:$src1),
      !if(!cast<Commutable_REV>(Inst).IsOrig,
        (Inst $src0, $src1, 0),
        (Inst $src1, $src0, 0)
      )
  >;

def : DivergentBinOp<srl, V_LSHRREV_B32_e64>;
def : DivergentBinOp<sra, V_ASHRREV_I32_e64>;
def : DivergentBinOp<shl, V_LSHLREV_B32_e64>;

let SubtargetPredicate = HasAddNoCarryInsts in {
  def : DivergentClampingBinOp<add, V_ADD_U32_e64>;
  def : DivergentClampingBinOp<sub, V_SUB_U32_e64>;
}

let SubtargetPredicate = isGFX6GFX7GFX8GFX9, Predicates = [isGFX6GFX7GFX8GFX9] in {
def : DivergentClampingBinOp<add, V_ADD_CO_U32_e64>;
def : DivergentClampingBinOp<sub, V_SUB_CO_U32_e64>;
}

def : DivergentBinOp<adde, V_ADDC_U32_e32>;
def : DivergentBinOp<sube, V_SUBB_U32_e32>;

class divergent_i64_BinOp <SDPatternOperator Op, Instruction Inst> :
  GCNPat<
      (getDivergentFrag<Op>.ret i64:$src0, i64:$src1),
      (REG_SEQUENCE VReg_64,
        (Inst
          (i32 (EXTRACT_SUBREG $src0, sub0)),
          (i32 (EXTRACT_SUBREG $src1, sub0))
        ), sub0,
        (Inst
          (i32 (EXTRACT_SUBREG $src0, sub1)),
          (i32 (EXTRACT_SUBREG $src1, sub1))
        ), sub1
      )
  >;

def :  divergent_i64_BinOp <and, V_AND_B32_e32>;
def :  divergent_i64_BinOp <or,  V_OR_B32_e32>;
def :  divergent_i64_BinOp <xor, V_XOR_B32_e32>;

let SubtargetPredicate = Has16BitInsts in {

let FPDPRounding = 1 in {
def V_MADMK_F16 : VOP2_Pseudo <"v_madmk_f16", VOP_MADMK_F16, [], "">;
defm V_LDEXP_F16 : VOP2Inst <"v_ldexp_f16", VOP_F16_F16_I32, AMDGPUldexp>;
} // End FPDPRounding = 1

defm V_LSHLREV_B16 : VOP2Inst <"v_lshlrev_b16", VOP_I16_I16_I16, lshl_rev>;
defm V_LSHRREV_B16 : VOP2Inst <"v_lshrrev_b16", VOP_I16_I16_I16, lshr_rev>;
defm V_ASHRREV_I16 : VOP2Inst <"v_ashrrev_i16", VOP_I16_I16_I16, ashr_rev>;

let isCommutable = 1 in {
let FPDPRounding = 1 in {
defm V_ADD_F16 : VOP2Inst <"v_add_f16", VOP_F16_F16_F16, any_fadd>;
defm V_SUB_F16 : VOP2Inst <"v_sub_f16", VOP_F16_F16_F16, any_fsub>;
defm V_SUBREV_F16 : VOP2Inst <"v_subrev_f16", VOP_F16_F16_F16, null_frag, "v_sub_f16">;
defm V_MUL_F16 : VOP2Inst <"v_mul_f16", VOP_F16_F16_F16, any_fmul>;

let mayRaiseFPException = 0 in {
def V_MADAK_F16 : VOP2_Pseudo <"v_madak_f16", VOP_MADAK_F16, [], "">;
}

} // End FPDPRounding = 1
defm V_ADD_U16 : VOP2Inst <"v_add_u16", VOP_I16_I16_I16_ARITH, add>;
defm V_SUB_U16 : VOP2Inst <"v_sub_u16" , VOP_I16_I16_I16_ARITH, sub>;
defm V_SUBREV_U16 : VOP2Inst <"v_subrev_u16", VOP_I16_I16_I16_ARITH, null_frag, "v_sub_u16">;
defm V_MUL_LO_U16 : VOP2Inst <"v_mul_lo_u16", VOP_I16_I16_I16, mul>;
defm V_MAX_F16 : VOP2Inst <"v_max_f16", VOP_F16_F16_F16, fmaxnum_like>;
defm V_MIN_F16 : VOP2Inst <"v_min_f16", VOP_F16_F16_F16, fminnum_like>;
defm V_MAX_U16 : VOP2Inst <"v_max_u16", VOP_I16_I16_I16, umax>;
defm V_MAX_I16 : VOP2Inst <"v_max_i16", VOP_I16_I16_I16, smax>;
defm V_MIN_U16 : VOP2Inst <"v_min_u16", VOP_I16_I16_I16, umin>;
defm V_MIN_I16 : VOP2Inst <"v_min_i16", VOP_I16_I16_I16, smin>;

let Constraints = "$vdst = $src2", DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1 in {
defm V_MAC_F16 : VOP2Inst <"v_mac_f16", VOP_MAC_F16>;
}
} // End isCommutable = 1

} // End SubtargetPredicate = Has16BitInsts

let SubtargetPredicate = HasDLInsts in {

let isReMaterializable = 1 in
defm V_XNOR_B32 : VOP2Inst <"v_xnor_b32", VOP_I32_I32_I32, xnor>;

let Constraints = "$vdst = $src2",
    DisableEncoding = "$src2",
    isConvertibleToThreeAddress = 1,
    isCommutable = 1 in
defm V_FMAC_F32 : VOP2Inst <"v_fmac_f32", VOP_MAC_F32>;

} // End SubtargetPredicate = HasDLInsts

let SubtargetPredicate = HasFmaLegacy32 in {

let Constraints = "$vdst = $src2",
    DisableEncoding = "$src2",
    isConvertibleToThreeAddress = 1,
    isCommutable = 1 in
defm V_FMAC_LEGACY_F32 : VOP2Inst <"v_fmac_legacy_f32", VOP_MAC_LEGACY_F32>;

} // End SubtargetPredicate = HasFmaLegacy32

let SubtargetPredicate = isGFX90APlus,
    Constraints = "$vdst = $src2",
    DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1,
    isCommutable = 1,
    SchedRW = [WriteDoubleAdd] in
defm V_FMAC_F64 : VOP2Inst <"v_fmac_f64", VOP_MAC_F64>;

let Constraints = "$vdst = $src2",
      DisableEncoding="$src2",
      isConvertibleToThreeAddress = 1,
      isCommutable = 1,
      IsDOT = 1 in {
  let SubtargetPredicate = HasDot5Insts in
    defm V_DOT2C_F32_F16 : VOP2Inst<"v_dot2c_f32_f16", VOP_DOT_ACC_F32_V2F16>;
  let SubtargetPredicate = HasDot6Insts in
    defm V_DOT4C_I32_I8  : VOP2Inst<"v_dot4c_i32_i8",  VOP_DOT_ACC_I32_I32>;

  let SubtargetPredicate = HasDot4Insts in
    defm V_DOT2C_I32_I16 : VOP2Inst<"v_dot2c_i32_i16", VOP_DOT_ACC_I32_I32>;
  let SubtargetPredicate = HasDot3Insts in
    defm V_DOT8C_I32_I4  : VOP2Inst<"v_dot8c_i32_i4",  VOP_DOT_ACC_I32_I32>;
}

let AddedComplexity = 30 in {
  def : GCNPat<
    (f32 (AMDGPUfdot2 v2f16:$src0, v2f16:$src1, f32:$src2, (i1 DSTCLAMP.NONE))),
    (f32 (V_DOT2C_F32_F16_e32 $src0, $src1, $src2))
  > {
    let SubtargetPredicate = HasDot5Insts;
  }
  def : GCNPat<
    (i32 (int_amdgcn_sdot4 i32:$src0, i32:$src1, i32:$src2, (i1 DSTCLAMP.NONE))),
    (i32 (V_DOT4C_I32_I8_e32 $src0, $src1, $src2))
  > {
    let SubtargetPredicate = HasDot6Insts;
  }
  def : GCNPat<
    (i32 (int_amdgcn_sdot2 v2i16:$src0, v2i16:$src1, i32:$src2, (i1 DSTCLAMP.NONE))),
    (i32 (V_DOT2C_I32_I16_e32 $src0, $src1, $src2))
  > {
    let SubtargetPredicate = HasDot4Insts;
  }
  def : GCNPat<
    (i32 (int_amdgcn_sdot8 i32:$src0, i32:$src1, i32:$src2, (i1 DSTCLAMP.NONE))),
    (i32 (V_DOT8C_I32_I4_e32 $src0, $src1, $src2))
  > {
    let SubtargetPredicate = HasDot3Insts;
  }
} // End AddedComplexity = 30

let SubtargetPredicate = HasFmaakFmamkF32Insts, isReMaterializable = 1 in {
def V_FMAMK_F32 : VOP2_Pseudo<"v_fmamk_f32", VOP_MADMK_F32, [], "">;

let isCommutable = 1 in
def V_FMAAK_F32 : VOP2_Pseudo<"v_fmaak_f32", VOP_MADAK_F32, [], "">;
}

let SubtargetPredicate = isGFX10Plus in {

let FPDPRounding = 1 in {
def V_FMAMK_F16 : VOP2_Pseudo <"v_fmamk_f16", VOP_MADMK_F16, [], "">;

let isCommutable = 1 in
def V_FMAAK_F16 : VOP2_Pseudo <"v_fmaak_f16", VOP_MADAK_F16, [], "">;
} // End FPDPRounding  = 1

let Constraints = "$vdst = $src2",
    DisableEncoding="$src2",
    isConvertibleToThreeAddress = 1,
    isCommutable = 1 in {
defm V_FMAC_F16 : VOP2Inst <"v_fmac_f16", VOP_MAC_F16>;
}

} // End SubtargetPredicate = isGFX10Plus

let SubtargetPredicate = HasPkFmacF16Inst in {
defm V_PK_FMAC_F16 : VOP2Inst<"v_pk_fmac_f16", VOP_V2F16_V2F16_V2F16>;
} // End SubtargetPredicate = HasPkFmacF16Inst

// Note: 16-bit instructions produce a 0 result in the high 16-bits
// on GFX8 and GFX9 and preserve high 16 bits on GFX10+
multiclass Arithmetic_i16_0Hi_Pats <SDPatternOperator op, Instruction inst> {

def : GCNPat<
  (i32 (zext (op i16:$src0, i16:$src1))),
  (inst VSrc_b16:$src0, VSrc_b16:$src1)
>;

def : GCNPat<
  (i64 (zext (op i16:$src0, i16:$src1))),
   (REG_SEQUENCE VReg_64,
     (inst $src0, $src1), sub0,
     (V_MOV_B32_e32 (i32 0)), sub1)
>;
}

class ZExt_i16_i1_Pat <SDNode ext> : GCNPat <
  (i16 (ext i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0/*src0mod*/), (i32 0/*src0*/),
                     (i32 0/*src1mod*/), (i32 1/*src1*/),
                     $src)
>;

foreach vt = [i16, v2i16] in {
def : GCNPat <
  (and vt:$src0, vt:$src1),
  (V_AND_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;

def : GCNPat <
  (or vt:$src0, vt:$src1),
  (V_OR_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;

def : GCNPat <
  (xor vt:$src0, vt:$src1),
  (V_XOR_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;
}

let Predicates = [Has16BitInsts] in {

// Undo sub x, c -> add x, -c canonicalization since c is more likely
// an inline immediate than -c.
// TODO: Also do for 64-bit.
def : GCNPat<
  (add i16:$src0, (i16 NegSubInlineIntConst16:$src1)),
  (V_SUB_U16_e64 VSrc_b16:$src0, NegSubInlineIntConst16:$src1)
>;


let Predicates = [Has16BitInsts, isGFX7GFX8GFX9] in {

def : GCNPat<
  (i32 (zext (add i16:$src0, (i16 NegSubInlineIntConst16:$src1)))),
  (V_SUB_U16_e64 VSrc_b16:$src0, NegSubInlineIntConst16:$src1)
>;

defm : Arithmetic_i16_0Hi_Pats<add, V_ADD_U16_e64>;
defm : Arithmetic_i16_0Hi_Pats<mul, V_MUL_LO_U16_e64>;
defm : Arithmetic_i16_0Hi_Pats<sub, V_SUB_U16_e64>;
defm : Arithmetic_i16_0Hi_Pats<smin, V_MIN_I16_e64>;
defm : Arithmetic_i16_0Hi_Pats<smax, V_MAX_I16_e64>;
defm : Arithmetic_i16_0Hi_Pats<umin, V_MIN_U16_e64>;
defm : Arithmetic_i16_0Hi_Pats<umax, V_MAX_U16_e64>;
defm : Arithmetic_i16_0Hi_Pats<lshl_rev, V_LSHLREV_B16_e64>;
defm : Arithmetic_i16_0Hi_Pats<lshr_rev, V_LSHRREV_B16_e64>;
defm : Arithmetic_i16_0Hi_Pats<ashr_rev, V_ASHRREV_I16_e64>;
}  // End Predicates = [Has16BitInsts, isGFX7GFX8GFX9]

def : ZExt_i16_i1_Pat<zext>;
def : ZExt_i16_i1_Pat<anyext>;

def : GCNPat <
  (i16 (sext i1:$src)),
  (V_CNDMASK_B32_e64 /*src0mod*/(i32 0), /*src0*/(i32 0),
                     /*src1mod*/(i32 0), /*src1*/(i32 -1), $src)
>;

} // End Predicates = [Has16BitInsts]


let SubtargetPredicate = HasIntClamp in {
// Set clamp bit for saturation.
def : VOPBinOpClampPat<uaddsat, V_ADD_CO_U32_e64, i32>;
def : VOPBinOpClampPat<usubsat, V_SUB_CO_U32_e64, i32>;
}

let SubtargetPredicate = HasAddNoCarryInsts, OtherPredicates = [HasIntClamp] in {
let AddedComplexity = 1 in { // Prefer over form with carry-out.
def : VOPBinOpClampPat<uaddsat, V_ADD_U32_e64, i32>;
def : VOPBinOpClampPat<usubsat, V_SUB_U32_e64, i32>;
}
}

let SubtargetPredicate = Has16BitInsts, OtherPredicates = [HasIntClamp] in {
def : VOPBinOpClampPat<uaddsat, V_ADD_U16_e64, i16>;
def : VOPBinOpClampPat<usubsat, V_SUB_U16_e64, i16>;
}

//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

class VOP2_DPP<bits<6> op, VOP2_DPP_Pseudo ps,
               string opName = ps.OpName, VOPProfile p = ps.Pfl,
               bit IsDPP16 = 0> :
    VOP_DPP<opName, p, IsDPP16> {
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  bits<8> vdst;
  bits<8> src1;
  let Inst{8-0}   = 0xfa;
  let Inst{16-9}  = !if(p.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(p.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0;
}

class Base_VOP2_DPP16<bits<6> op, VOP2_DPP_Pseudo ps,
                 string opName = ps.OpName, VOPProfile p = ps.Pfl> :
    VOP2_DPP<op, ps, opName, p, 1> {
  let AssemblerPredicate = HasDPP16;
  let SubtargetPredicate = HasDPP16;
  let OtherPredicates = ps.OtherPredicates;
}

class VOP2_DPP16<bits<6> op, VOP2_DPP_Pseudo ps,
                 string opName = ps.OpName, VOPProfile p = ps.Pfl> :
    Base_VOP2_DPP16<op, ps, opName, p>,
    SIMCInstr <ps.PseudoInstr, SIEncodingFamily.GFX10>;

class VOP2_DPP8<bits<6> op, VOP2_Pseudo ps,
                VOPProfile p = ps.Pfl> :
    VOP_DPP8<ps.OpName, p> {
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  bits<8> vdst;
  bits<8> src1;

  let Inst{8-0}   = fi;
  let Inst{16-9}  = !if(p.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(p.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0;

  let OtherPredicates = ps.OtherPredicates;
}

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10" in {
  //===------------------------------- VOP2 -------------------------------===//
  multiclass VOP2Only_Real_MADK_gfx10<bits<6> op> {
    def _gfx10 :
      VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.GFX10>,
      VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP2Only_Real_MADK_gfx10_with_name<bits<6> op, string opName,
                                                string asmName> {
    def _gfx10 :
        VOP2_Real<!cast<VOP2_Pseudo>(opName), SIEncodingFamily.GFX10>,
        VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(opName).Pfl> {
      VOP2_Pseudo ps = !cast<VOP2_Pseudo>(opName);
      let AsmString = asmName # ps.AsmOperands;
    }
  }
  multiclass VOP2_Real_e32_gfx10<bits<6> op> {
    def _e32_gfx10 :
      VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.GFX10>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>;
  }
  multiclass VOP2_Real_e64_gfx10<bits<6> op> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<{0, 1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP2_Real_sdwa_gfx10<bits<6> op> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx10 :
      VOP_SDWA10_Real<!cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa")>,
      VOP2_SDWA9Ae<op{5-0}, !cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa").Pfl> {
      let DecoderNamespace = "SDWA10";
    }
  }
  multiclass VOP2_Real_dpp_gfx10<bits<6> op> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_gfx10 : VOP2_DPP16<op, !cast<VOP2_DPP_Pseudo>(NAME#"_dpp")> {
      let DecoderNamespace = "SDWA10";
    }
  }
  multiclass VOP2_Real_dpp8_gfx10<bits<6> op> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp8_gfx10 : VOP2_DPP8<op, !cast<VOP2_Pseudo>(NAME#"_e32")> {
      let DecoderNamespace = "DPP8";
    }
  }

  //===------------------------- VOP2 (with name) -------------------------===//
  multiclass VOP2_Real_e32_gfx10_with_name<bits<6> op, string opName,
                                           string asmName> {
    def _e32_gfx10 :
      VOP2_Real<!cast<VOP2_Pseudo>(opName#"_e32"), SIEncodingFamily.GFX10>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(opName#"_e32").Pfl> {
        VOP2_Pseudo ps = !cast<VOP2_Pseudo>(opName#"_e32");
        let AsmString = asmName # ps.AsmOperands;
      }
  }
  multiclass VOP2_Real_e64_gfx10_with_name<bits<6> op, string opName,
                                           string asmName> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<{0, 1, 0, 0, op{5-0}},
                  !cast<VOP3_Pseudo>(opName#"_e64").Pfl> {
        VOP3_Pseudo ps = !cast<VOP3_Pseudo>(opName#"_e64");
        let AsmString = asmName # ps.AsmOperands;
      }
  }
  let DecoderNamespace = "SDWA10" in {
    multiclass VOP2_Real_sdwa_gfx10_with_name<bits<6> op, string opName,
                                              string asmName> {
      foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtSDWA9>.ret in
      def _sdwa_gfx10 :
        VOP_SDWA10_Real<!cast<VOP2_SDWA_Pseudo>(opName#"_sdwa")>,
        VOP2_SDWA9Ae<op{5-0}, !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa").Pfl> {
          VOP2_SDWA_Pseudo ps = !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa");
          let AsmString = asmName # ps.AsmOperands;
        }
    }
    multiclass VOP2_Real_dpp_gfx10_with_name<bits<6> op, string opName,
                                             string asmName> {
      foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
      def _dpp_gfx10 : VOP2_DPP16<op, !cast<VOP2_DPP_Pseudo>(opName#"_dpp")> {
        VOP2_Pseudo ps = !cast<VOP2_Pseudo>(opName#"_e32");
        let AsmString = asmName # ps.Pfl.AsmDPP16;
      }
    }
    multiclass VOP2_Real_dpp8_gfx10_with_name<bits<6> op, string opName,
                                              string asmName> {
      foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
      def _dpp8_gfx10 : VOP2_DPP8<op, !cast<VOP2_Pseudo>(opName#"_e32")> {
        VOP2_Pseudo ps = !cast<VOP2_Pseudo>(opName#"_e32");
        let AsmString = asmName # ps.Pfl.AsmDPP8;
        let DecoderNamespace = "DPP8";
      }
    }
  } // End DecoderNamespace = "SDWA10"

  //===------------------------------ VOP2be ------------------------------===//
  multiclass VOP2be_Real_e32_gfx10<bits<6> op, string opName, string asmName> {
    def _e32_gfx10 :
      VOP2_Real<!cast<VOP2_Pseudo>(opName#"_e32"), SIEncodingFamily.GFX10>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(opName#"_e32").Pfl> {
        VOP2_Pseudo Ps = !cast<VOP2_Pseudo>(opName#"_e32");
        let AsmString = asmName # !subst(", vcc", "", Ps.AsmOperands);
      }
  }
  multiclass VOP2be_Real_e64_gfx10<bits<6> op, string opName, string asmName> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.GFX10>,
      VOP3be_gfx10<{0, 1, 0, 0, op{5-0}},
                   !cast<VOP3_Pseudo>(opName#"_e64").Pfl> {
        VOP3_Pseudo Ps = !cast<VOP3_Pseudo>(opName#"_e64");
        let AsmString = asmName # Ps.AsmOperands;
      }
  }
  multiclass VOP2be_Real_sdwa_gfx10<bits<6> op, string opName, string asmName> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx10 :
      VOP_SDWA10_Real<!cast<VOP2_SDWA_Pseudo>(opName#"_sdwa")>,
      VOP2_SDWA9Ae<op{5-0}, !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa").Pfl> {
        VOP2_SDWA_Pseudo Ps = !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa");
        let AsmString = asmName # !subst(", vcc", "", Ps.AsmOperands);
        let DecoderNamespace = "SDWA10";
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_w32_gfx10 :
      Base_VOP_SDWA10_Real<!cast<VOP2_SDWA_Pseudo>(opName#"_sdwa")>,
      VOP2_SDWA9Ae<op{5-0}, !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa").Pfl> {
        VOP2_SDWA_Pseudo Ps = !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa");
        let AsmString = asmName # !subst("vcc", "vcc_lo", Ps.AsmOperands);
        let isAsmParserOnly = 1;
        let DecoderNamespace = "SDWA10";
        let WaveSizePredicate = isWave32;
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_w64_gfx10 :
      Base_VOP_SDWA10_Real<!cast<VOP2_SDWA_Pseudo>(opName#"_sdwa")>,
      VOP2_SDWA9Ae<op{5-0}, !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa").Pfl> {
        VOP2_SDWA_Pseudo Ps = !cast<VOP2_SDWA_Pseudo>(opName#"_sdwa");
        let AsmString = asmName # Ps.AsmOperands;
        let isAsmParserOnly = 1;
        let DecoderNamespace = "SDWA10";
        let WaveSizePredicate = isWave64;
      }
  }
  multiclass VOP2be_Real_dpp_gfx10<bits<6> op, string opName, string asmName> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_gfx10 :
      VOP2_DPP16<op, !cast<VOP2_DPP_Pseudo>(opName#"_dpp"), asmName> {
        string AsmDPP = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP16;
        let AsmString = asmName # !subst(", vcc", "", AsmDPP);
        let DecoderNamespace = "SDWA10";
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_w32_gfx10 :
      Base_VOP2_DPP16<op, !cast<VOP2_DPP_Pseudo>(opName#"_dpp"), asmName> {
        string AsmDPP = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP16;
        let AsmString = asmName # !subst("vcc", "vcc_lo", AsmDPP);
        let isAsmParserOnly = 1;
        let WaveSizePredicate = isWave32;
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_w64_gfx10 :
      Base_VOP2_DPP16<op, !cast<VOP2_DPP_Pseudo>(opName#"_dpp"), asmName> {
        string AsmDPP = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP16;
        let AsmString = asmName # AsmDPP;
        let isAsmParserOnly = 1;
        let WaveSizePredicate = isWave64;
      }
  }
  multiclass VOP2be_Real_dpp8_gfx10<bits<6> op, string opName, string asmName> {
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp8_gfx10 :
      VOP2_DPP8<op, !cast<VOP2_Pseudo>(opName#"_e32")> {
        string AsmDPP8 = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP8;
        let AsmString = asmName # !subst(", vcc", "", AsmDPP8);
        let DecoderNamespace = "DPP8";
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp8_w32_gfx10 :
      VOP2_DPP8<op, !cast<VOP2_Pseudo>(opName#"_e32")> {
        string AsmDPP8 = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP8;
        let AsmString = asmName # !subst("vcc", "vcc_lo", AsmDPP8);
        let isAsmParserOnly = 1;
        let WaveSizePredicate = isWave32;
      }
    foreach _ = BoolToList<!cast<VOP2_Pseudo>(opName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp8_w64_gfx10 :
      VOP2_DPP8<op, !cast<VOP2_Pseudo>(opName#"_e32")> {
        string AsmDPP8 = !cast<VOP2_Pseudo>(opName#"_e32").Pfl.AsmDPP8;
        let AsmString = asmName # AsmDPP8;
        let isAsmParserOnly = 1;
        let WaveSizePredicate = isWave64;
      }
  }

  //===----------------------------- VOP3Only -----------------------------===//
  multiclass VOP3Only_Real_gfx10<bits<10> op> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
        let IsSingle = 1;
      }
  }

  //===---------------------------- VOP3beOnly ----------------------------===//
  multiclass VOP3beOnly_Real_gfx10<bits<10> op> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3be_gfx10<op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
        let IsSingle = 1;
      }
  }
} // End AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10"

multiclass VOP2be_Real_gfx10<bits<6> op, string opName, string asmName> :
  VOP2be_Real_e32_gfx10<op, opName, asmName>,
  VOP2be_Real_e64_gfx10<op, opName, asmName>,
  VOP2be_Real_sdwa_gfx10<op, opName, asmName>,
  VOP2be_Real_dpp_gfx10<op, opName, asmName>,
  VOP2be_Real_dpp8_gfx10<op, opName, asmName>;

multiclass VOP2e_Real_gfx10<bits<6> op, string opName, string asmName> :
  VOP2_Real_e32_gfx10<op>,
  VOP2_Real_e64_gfx10<op>,
  VOP2be_Real_sdwa_gfx10<op, opName, asmName>,
  VOP2be_Real_dpp_gfx10<op, opName, asmName>,
  VOP2be_Real_dpp8_gfx10<op, opName, asmName>;

multiclass VOP2_Real_gfx10<bits<6> op> :
  VOP2_Real_e32_gfx10<op>, VOP2_Real_e64_gfx10<op>,
  VOP2_Real_sdwa_gfx10<op>, VOP2_Real_dpp_gfx10<op>, VOP2_Real_dpp8_gfx10<op>;

multiclass VOP2_Real_gfx10_with_name<bits<6> op, string opName,
                                     string asmName> :
  VOP2_Real_e32_gfx10_with_name<op, opName, asmName>,
  VOP2_Real_e64_gfx10_with_name<op, opName, asmName>,
  VOP2_Real_sdwa_gfx10_with_name<op, opName, asmName>,
  VOP2_Real_dpp_gfx10_with_name<op, opName, asmName>,
  VOP2_Real_dpp8_gfx10_with_name<op, opName, asmName>;

// NB: Same opcode as v_mac_legacy_f32
let DecoderNamespace = "GFX10_B" in
defm V_FMAC_LEGACY_F32 : VOP2_Real_gfx10<0x006>;

defm V_XNOR_B32        : VOP2_Real_gfx10<0x01e>;
defm V_FMAC_F32        : VOP2_Real_gfx10<0x02b>;
defm V_FMAMK_F32       : VOP2Only_Real_MADK_gfx10<0x02c>;
defm V_FMAAK_F32       : VOP2Only_Real_MADK_gfx10<0x02d>;
defm V_ADD_F16         : VOP2_Real_gfx10<0x032>;
defm V_SUB_F16         : VOP2_Real_gfx10<0x033>;
defm V_SUBREV_F16      : VOP2_Real_gfx10<0x034>;
defm V_MUL_F16         : VOP2_Real_gfx10<0x035>;
defm V_FMAC_F16        : VOP2_Real_gfx10<0x036>;
defm V_FMAMK_F16       : VOP2Only_Real_MADK_gfx10<0x037>;
defm V_FMAAK_F16       : VOP2Only_Real_MADK_gfx10<0x038>;
defm V_MAX_F16         : VOP2_Real_gfx10<0x039>;
defm V_MIN_F16         : VOP2_Real_gfx10<0x03a>;
defm V_LDEXP_F16       : VOP2_Real_gfx10<0x03b>;

let IsSingle = 1 in {
defm V_PK_FMAC_F16     : VOP2_Real_e32_gfx10<0x03c>;
}

// VOP2 no carry-in, carry-out.
defm V_ADD_NC_U32 :
  VOP2_Real_gfx10_with_name<0x025, "V_ADD_U32", "v_add_nc_u32">;
defm V_SUB_NC_U32 :
  VOP2_Real_gfx10_with_name<0x026, "V_SUB_U32", "v_sub_nc_u32">;
defm V_SUBREV_NC_U32 :
  VOP2_Real_gfx10_with_name<0x027, "V_SUBREV_U32", "v_subrev_nc_u32">;

// VOP2 carry-in, carry-out.
defm V_ADD_CO_CI_U32 :
  VOP2be_Real_gfx10<0x028, "V_ADDC_U32", "v_add_co_ci_u32">;
defm V_SUB_CO_CI_U32 :
  VOP2be_Real_gfx10<0x029, "V_SUBB_U32", "v_sub_co_ci_u32">;
defm V_SUBREV_CO_CI_U32 :
  VOP2be_Real_gfx10<0x02a, "V_SUBBREV_U32", "v_subrev_co_ci_u32">;

defm V_CNDMASK_B32 :
  VOP2e_Real_gfx10<0x001, "V_CNDMASK_B32", "v_cndmask_b32">;

// VOP3 only.
defm V_BFM_B32            : VOP3Only_Real_gfx10<0x363>;
defm V_BCNT_U32_B32       : VOP3Only_Real_gfx10<0x364>;
defm V_MBCNT_LO_U32_B32   : VOP3Only_Real_gfx10<0x365>;
defm V_MBCNT_HI_U32_B32   : VOP3Only_Real_gfx10<0x366>;
defm V_LDEXP_F32          : VOP3Only_Real_gfx10<0x362>;
defm V_CVT_PKNORM_I16_F32 : VOP3Only_Real_gfx10<0x368>;
defm V_CVT_PKNORM_U16_F32 : VOP3Only_Real_gfx10<0x369>;
defm V_CVT_PK_U16_U32     : VOP3Only_Real_gfx10<0x36a>;
defm V_CVT_PK_I16_I32     : VOP3Only_Real_gfx10<0x36b>;

// VOP3 carry-out.
defm V_ADD_CO_U32 : VOP3beOnly_Real_gfx10<0x30f>;
defm V_SUB_CO_U32 : VOP3beOnly_Real_gfx10<0x310>;
defm V_SUBREV_CO_U32 : VOP3beOnly_Real_gfx10<0x319>;

let SubtargetPredicate = isGFX10Plus in {
  defm : VOP2eInstAliases<V_CNDMASK_B32_e32, V_CNDMASK_B32_e32_gfx10>;

  defm : VOP2bInstAliases<
    V_ADDC_U32_e32, V_ADD_CO_CI_U32_e32_gfx10, "v_add_co_ci_u32">;
  defm : VOP2bInstAliases<
    V_SUBB_U32_e32, V_SUB_CO_CI_U32_e32_gfx10, "v_sub_co_ci_u32">;
  defm : VOP2bInstAliases<
    V_SUBBREV_U32_e32, V_SUBREV_CO_CI_U32_e32_gfx10, "v_subrev_co_ci_u32">;
} // End SubtargetPredicate = isGFX10Plus

//===----------------------------------------------------------------------===//
// GFX6, GFX7, GFX10.
//===----------------------------------------------------------------------===//

class VOP2_DPPe <bits<6> op, VOP2_DPP_Pseudo ps, VOPProfile P = ps.Pfl> :
  VOP_DPPe <P> {
  bits<8> vdst;
  bits<8> src1;
  let Inst{8-0}   = 0xfa; //dpp
  let Inst{16-9}  = !if(P.HasSrc1, src1{7-0}, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; //encoding
}

let AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7" in {
  multiclass VOP2_Lane_Real_gfx6_gfx7<bits<6> op> {
    def _gfx6_gfx7 :
      VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.SI>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP2Only_Real_MADK_gfx6_gfx7<bits<6> op> {
    def _gfx6_gfx7 :
      VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.SI>,
      VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP2_Real_e32_gfx6_gfx7<bits<6> op, string opName = NAME> {
    def _e32_gfx6_gfx7 :
      VOP2_Real<!cast<VOP2_Pseudo>(opName#"_e32"), SIEncodingFamily.SI>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(opName#"_e32").Pfl>;
  }
  multiclass VOP2_Real_e64_gfx6_gfx7<bits<6> op, string opName = NAME> {
    def _e64_gfx6_gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.SI>,
      VOP3e_gfx6_gfx7<{1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(opName#"_e64").Pfl>;
  }
  multiclass VOP2be_Real_e64_gfx6_gfx7<bits<6> op, string opName = NAME> {
    def _e64_gfx6_gfx7 :
      VOP3_Real<!cast<VOP3_Pseudo>(opName#"_e64"), SIEncodingFamily.SI>,
      VOP3be_gfx6_gfx7<{1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(opName#"_e64").Pfl>;
  }
} // End AssemblerPredicate = isGFX6GFX7, DecoderNamespace = "GFX6GFX7"

multiclass VOP2Only_Real_MADK_gfx6_gfx7_gfx10<bits<6> op> :
  VOP2Only_Real_MADK_gfx6_gfx7<op>, VOP2Only_Real_MADK_gfx10<op>;

multiclass VOP2_Real_gfx6_gfx7<bits<6> op> :
  VOP2_Real_e32_gfx6_gfx7<op>, VOP2_Real_e64_gfx6_gfx7<op>;

multiclass VOP2_Real_gfx6_gfx7_gfx10<bits<6> op> :
  VOP2_Real_gfx6_gfx7<op>, VOP2_Real_gfx10<op>;

multiclass VOP2be_Real_gfx6_gfx7<bits<6> op> :
  VOP2_Real_e32_gfx6_gfx7<op>, VOP2be_Real_e64_gfx6_gfx7<op>;

multiclass VOP2be_Real_gfx6_gfx7_with_name<bits<6> op,
  string opName, string asmName>  {
  defvar ps32 = !cast<VOP2_Pseudo>(opName#"_e32");
  defvar ps64 = !cast<VOP3_Pseudo>(opName#"_e64");

  let AsmString = asmName # ps32.AsmOperands in {
    defm "" : VOP2_Real_e32_gfx6_gfx7<op, opName>;
  }

   let AsmString = asmName # ps64.AsmOperands in {
    defm "" : VOP2be_Real_e64_gfx6_gfx7<op, opName>;
  }
}

defm V_CNDMASK_B32        : VOP2_Real_gfx6_gfx7<0x000>;
defm V_MIN_LEGACY_F32     : VOP2_Real_gfx6_gfx7<0x00d>;
defm V_MAX_LEGACY_F32     : VOP2_Real_gfx6_gfx7<0x00e>;
defm V_LSHR_B32           : VOP2_Real_gfx6_gfx7<0x015>;
defm V_ASHR_I32           : VOP2_Real_gfx6_gfx7<0x017>;
defm V_LSHL_B32           : VOP2_Real_gfx6_gfx7<0x019>;
defm V_BFM_B32            : VOP2_Real_gfx6_gfx7<0x01e>;
defm V_BCNT_U32_B32       : VOP2_Real_gfx6_gfx7<0x022>;
defm V_MBCNT_LO_U32_B32   : VOP2_Real_gfx6_gfx7<0x023>;
defm V_MBCNT_HI_U32_B32   : VOP2_Real_gfx6_gfx7<0x024>;
defm V_LDEXP_F32          : VOP2_Real_gfx6_gfx7<0x02b>;
defm V_CVT_PKACCUM_U8_F32 : VOP2_Real_gfx6_gfx7<0x02c>;
defm V_CVT_PKNORM_I16_F32 : VOP2_Real_gfx6_gfx7<0x02d>;
defm V_CVT_PKNORM_U16_F32 : VOP2_Real_gfx6_gfx7<0x02e>;
defm V_CVT_PK_U16_U32     : VOP2_Real_gfx6_gfx7<0x030>;
defm V_CVT_PK_I16_I32     : VOP2_Real_gfx6_gfx7<0x031>;

// V_ADD_I32, V_SUB_I32, and V_SUBREV_I32 where renamed to *_U32 in
// VI, but the VI instructions behave the same as the SI versions.
defm V_ADD_I32            : VOP2be_Real_gfx6_gfx7_with_name<0x025, "V_ADD_CO_U32", "v_add_i32">;
defm V_SUB_I32            : VOP2be_Real_gfx6_gfx7_with_name<0x026, "V_SUB_CO_U32", "v_sub_i32">;
defm V_SUBREV_I32         : VOP2be_Real_gfx6_gfx7_with_name<0x027, "V_SUBREV_CO_U32", "v_subrev_i32">;
defm V_ADDC_U32           : VOP2be_Real_gfx6_gfx7<0x028>;
defm V_SUBB_U32           : VOP2be_Real_gfx6_gfx7<0x029>;
defm V_SUBBREV_U32        : VOP2be_Real_gfx6_gfx7<0x02a>;

defm V_READLANE_B32 : VOP2_Lane_Real_gfx6_gfx7<0x001>;

let InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in) in {
  defm V_WRITELANE_B32 : VOP2_Lane_Real_gfx6_gfx7<0x002>;
} // End InOperandList = (ins SSrcOrLds_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in)

let SubtargetPredicate = isGFX6GFX7 in {
  defm : VOP2eInstAliases<V_CNDMASK_B32_e32, V_CNDMASK_B32_e32_gfx6_gfx7>;
  defm : VOP2eInstAliases<V_ADD_CO_U32_e32, V_ADD_I32_e32_gfx6_gfx7>;
  defm : VOP2eInstAliases<V_SUB_CO_U32_e32, V_SUB_I32_e32_gfx6_gfx7>;
  defm : VOP2eInstAliases<V_SUBREV_CO_U32_e32, V_SUBREV_I32_e32_gfx6_gfx7>;

  def : VOP2e64InstAlias<V_ADD_CO_U32_e64, V_ADD_I32_e64_gfx6_gfx7>;
  def : VOP2e64InstAlias<V_SUB_CO_U32_e64, V_SUB_I32_e64_gfx6_gfx7>;
  def : VOP2e64InstAlias<V_SUBREV_CO_U32_e64, V_SUBREV_I32_e64_gfx6_gfx7>;
} // End SubtargetPredicate = isGFX6GFX7

defm V_ADD_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x003>;
defm V_SUB_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x004>;
defm V_SUBREV_F32         : VOP2_Real_gfx6_gfx7_gfx10<0x005>;
defm V_MAC_LEGACY_F32     : VOP2_Real_gfx6_gfx7_gfx10<0x006>;
defm V_MUL_LEGACY_F32     : VOP2_Real_gfx6_gfx7_gfx10<0x007>;
defm V_MUL_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x008>;
defm V_MUL_I32_I24        : VOP2_Real_gfx6_gfx7_gfx10<0x009>;
defm V_MUL_HI_I32_I24     : VOP2_Real_gfx6_gfx7_gfx10<0x00a>;
defm V_MUL_U32_U24        : VOP2_Real_gfx6_gfx7_gfx10<0x00b>;
defm V_MUL_HI_U32_U24     : VOP2_Real_gfx6_gfx7_gfx10<0x00c>;
defm V_MIN_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x00f>;
defm V_MAX_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x010>;
defm V_MIN_I32            : VOP2_Real_gfx6_gfx7_gfx10<0x011>;
defm V_MAX_I32            : VOP2_Real_gfx6_gfx7_gfx10<0x012>;
defm V_MIN_U32            : VOP2_Real_gfx6_gfx7_gfx10<0x013>;
defm V_MAX_U32            : VOP2_Real_gfx6_gfx7_gfx10<0x014>;
defm V_LSHRREV_B32        : VOP2_Real_gfx6_gfx7_gfx10<0x016>;
defm V_ASHRREV_I32        : VOP2_Real_gfx6_gfx7_gfx10<0x018>;
defm V_LSHLREV_B32        : VOP2_Real_gfx6_gfx7_gfx10<0x01a>;
defm V_AND_B32            : VOP2_Real_gfx6_gfx7_gfx10<0x01b>;
defm V_OR_B32             : VOP2_Real_gfx6_gfx7_gfx10<0x01c>;
defm V_XOR_B32            : VOP2_Real_gfx6_gfx7_gfx10<0x01d>;
defm V_MAC_F32            : VOP2_Real_gfx6_gfx7_gfx10<0x01f>;
defm V_CVT_PKRTZ_F16_F32  : VOP2_Real_gfx6_gfx7_gfx10<0x02f>;
defm V_MADMK_F32          : VOP2Only_Real_MADK_gfx6_gfx7_gfx10<0x020>;
defm V_MADAK_F32          : VOP2Only_Real_MADK_gfx6_gfx7_gfx10<0x021>;

//===----------------------------------------------------------------------===//
// GFX8, GFX9 (VI).
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX8GFX9, DecoderNamespace = "GFX8" in {

multiclass VOP2_Real_MADK_vi <bits<6> op> {
  def _vi : VOP2_Real<!cast<VOP2_Pseudo>(NAME), SIEncodingFamily.VI>,
            VOP2_MADKe<op{5-0}, !cast<VOP2_Pseudo>(NAME).Pfl>;
}

multiclass VOP2_Real_e32_vi <bits<6> op> {
  def _e32_vi :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.VI>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>;
}

multiclass VOP2_Real_e64_vi <bits<10> op> {
  def _e64_vi :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
    VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
}

multiclass VOP2_Real_e64only_vi <bits<10> op> {
  def _e64_vi :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.VI>,
    VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
      let IsSingle = 1;
    }
}

multiclass Base_VOP2_Real_e32e64_vi <bits<6> op> :
  VOP2_Real_e32_vi<op>,
  VOP2_Real_e64_vi<{0, 1, 0, 0, op{5-0}}>;

} // End AssemblerPredicate = isGFX8GFX9, DecoderNamespace = "GFX8"

multiclass VOP2_SDWA_Real <bits<6> op> {
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA>.ret in
  def _sdwa_vi :
    VOP_SDWA_Real <!cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa")>,
    VOP2_SDWAe <op{5-0}, !cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa").Pfl>;
}

multiclass VOP2_SDWA9_Real <bits<6> op> {
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA9>.ret in
  def _sdwa_gfx9 :
    VOP_SDWA9_Real <!cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa")>,
    VOP2_SDWA9Ae <op{5-0}, !cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa").Pfl>;
}

let AssemblerPredicate = isGFX8Only in {

multiclass VOP2be_Real_e32e64_vi_only <bits<6> op, string OpName, string AsmName> {
  def _e32_vi :
    VOP2_Real<!cast<VOP2_Pseudo>(OpName#"_e32"), SIEncodingFamily.VI>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(OpName#"_e32").Pfl> {
      VOP2_Pseudo ps = !cast<VOP2_Pseudo>(OpName#"_e32");
      let AsmString = AsmName # ps.AsmOperands;
      let DecoderNamespace = "GFX8";
    }
  def _e64_vi :
    VOP3_Real<!cast<VOP3_Pseudo>(OpName#"_e64"), SIEncodingFamily.VI>,
    VOP3be_vi <{0, 1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(OpName#"_e64").Pfl> {
      VOP3_Pseudo ps = !cast<VOP3_Pseudo>(OpName#"_e64");
      let AsmString = AsmName # ps.AsmOperands;
      let DecoderNamespace = "GFX8";
    }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(OpName#"_e32").Pfl.HasExtSDWA>.ret in
    def _sdwa_vi :
      VOP_SDWA_Real <!cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa")>,
      VOP2_SDWAe <op{5-0}, !cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa").Pfl> {
        VOP2_SDWA_Pseudo ps = !cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa");
        let AsmString = AsmName # ps.AsmOperands;
      }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(OpName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_vi :
      VOP_DPP_Real<!cast<VOP2_DPP_Pseudo>(OpName#"_dpp"), SIEncodingFamily.VI>,
      VOP2_DPPe<op, !cast<VOP2_DPP_Pseudo>(OpName#"_dpp")> {
        VOP2_DPP_Pseudo ps = !cast<VOP2_DPP_Pseudo>(OpName#"_dpp");
        let AsmString = AsmName # ps.AsmOperands;
      }
}
}

let AssemblerPredicate = isGFX9Only in {

multiclass VOP2be_Real_e32e64_gfx9 <bits<6> op, string OpName, string AsmName> {
  def _e32_gfx9 :
    VOP2_Real<!cast<VOP2_Pseudo>(OpName#"_e32"), SIEncodingFamily.GFX9>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(OpName#"_e32").Pfl> {
      VOP2_Pseudo ps = !cast<VOP2_Pseudo>(OpName#"_e32");
      let AsmString = AsmName # ps.AsmOperands;
      let DecoderNamespace = "GFX9";
    }
  def _e64_gfx9 :
    VOP3_Real<!cast<VOP3_Pseudo>(OpName#"_e64"), SIEncodingFamily.GFX9>,
    VOP3be_vi <{0, 1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(OpName#"_e64").Pfl> {
      VOP3_Pseudo ps = !cast<VOP3_Pseudo>(OpName#"_e64");
      let AsmString = AsmName # ps.AsmOperands;
      let DecoderNamespace = "GFX9";
    }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(OpName#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx9 :
      VOP_SDWA9_Real <!cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa")>,
      VOP2_SDWA9Ae <op{5-0}, !cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa").Pfl> {
        VOP2_SDWA_Pseudo ps = !cast<VOP2_SDWA_Pseudo>(OpName#"_sdwa");
        let AsmString = AsmName # ps.AsmOperands;
      }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(OpName#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_gfx9 :
      VOP_DPP_Real<!cast<VOP2_DPP_Pseudo>(OpName#"_dpp"), SIEncodingFamily.GFX9>,
      VOP2_DPPe<op, !cast<VOP2_DPP_Pseudo>(OpName#"_dpp")> {
        VOP2_DPP_Pseudo ps = !cast<VOP2_DPP_Pseudo>(OpName#"_dpp");
        let AsmString = AsmName # ps.AsmOperands;
        let DecoderNamespace = "SDWA9";
      }
}

multiclass VOP2_Real_e32e64_gfx9 <bits<6> op> {
  def _e32_gfx9 :
    VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.GFX9>,
    VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>{
      let DecoderNamespace = "GFX9";
    }
  def _e64_gfx9 :
    VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX9>,
    VOP3e_vi <{0, 1, 0, 0, op{5-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
      let DecoderNamespace = "GFX9";
    }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx9 :
      VOP_SDWA9_Real <!cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa")>,
      VOP2_SDWA9Ae <op{5-0}, !cast<VOP2_SDWA_Pseudo>(NAME#"_sdwa").Pfl> {
      }
  foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_gfx9 :
      VOP_DPP_Real<!cast<VOP2_DPP_Pseudo>(NAME#"_dpp"), SIEncodingFamily.GFX9>,
      VOP2_DPPe<op, !cast<VOP2_DPP_Pseudo>(NAME#"_dpp")> {
        let DecoderNamespace = "SDWA9";
      }
}

} // AssemblerPredicate = isGFX9Only

multiclass VOP2_Real_e32e64_vi <bits<6> op> :
  Base_VOP2_Real_e32e64_vi<op>, VOP2_SDWA_Real<op>, VOP2_SDWA9_Real<op> {

  foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_vi :
      VOP_DPP_Real<!cast<VOP2_DPP_Pseudo>(NAME#"_dpp"), SIEncodingFamily.VI>,
      VOP2_DPPe<op, !cast<VOP2_DPP_Pseudo>(NAME#"_dpp")>;
}

defm V_CNDMASK_B32        : VOP2_Real_e32e64_vi <0x0>;
defm V_ADD_F32            : VOP2_Real_e32e64_vi <0x1>;
defm V_SUB_F32            : VOP2_Real_e32e64_vi <0x2>;
defm V_SUBREV_F32         : VOP2_Real_e32e64_vi <0x3>;
let AssemblerPredicate = isGCN3ExcludingGFX90A in
defm V_MUL_LEGACY_F32     : VOP2_Real_e32e64_vi <0x4>;
defm V_MUL_F32            : VOP2_Real_e32e64_vi <0x5>;
defm V_MUL_I32_I24        : VOP2_Real_e32e64_vi <0x6>;
defm V_MUL_HI_I32_I24     : VOP2_Real_e32e64_vi <0x7>;
defm V_MUL_U32_U24        : VOP2_Real_e32e64_vi <0x8>;
defm V_MUL_HI_U32_U24     : VOP2_Real_e32e64_vi <0x9>;
defm V_MIN_F32            : VOP2_Real_e32e64_vi <0xa>;
defm V_MAX_F32            : VOP2_Real_e32e64_vi <0xb>;
defm V_MIN_I32            : VOP2_Real_e32e64_vi <0xc>;
defm V_MAX_I32            : VOP2_Real_e32e64_vi <0xd>;
defm V_MIN_U32            : VOP2_Real_e32e64_vi <0xe>;
defm V_MAX_U32            : VOP2_Real_e32e64_vi <0xf>;
defm V_LSHRREV_B32        : VOP2_Real_e32e64_vi <0x10>;
defm V_ASHRREV_I32        : VOP2_Real_e32e64_vi <0x11>;
defm V_LSHLREV_B32        : VOP2_Real_e32e64_vi <0x12>;
defm V_AND_B32            : VOP2_Real_e32e64_vi <0x13>;
defm V_OR_B32             : VOP2_Real_e32e64_vi <0x14>;
defm V_XOR_B32            : VOP2_Real_e32e64_vi <0x15>;
defm V_MAC_F32            : VOP2_Real_e32e64_vi <0x16>;
defm V_MADMK_F32          : VOP2_Real_MADK_vi <0x17>;
defm V_MADAK_F32          : VOP2_Real_MADK_vi <0x18>;

defm V_ADD_U32            : VOP2be_Real_e32e64_vi_only <0x19, "V_ADD_CO_U32",     "v_add_u32">;
defm V_SUB_U32            : VOP2be_Real_e32e64_vi_only <0x1a, "V_SUB_CO_U32",     "v_sub_u32">;
defm V_SUBREV_U32         : VOP2be_Real_e32e64_vi_only <0x1b, "V_SUBREV_CO_U32",  "v_subrev_u32">;
defm V_ADDC_U32           : VOP2be_Real_e32e64_vi_only <0x1c, "V_ADDC_U32",    "v_addc_u32">;
defm V_SUBB_U32           : VOP2be_Real_e32e64_vi_only <0x1d, "V_SUBB_U32",    "v_subb_u32">;
defm V_SUBBREV_U32        : VOP2be_Real_e32e64_vi_only <0x1e, "V_SUBBREV_U32", "v_subbrev_u32">;

defm V_ADD_CO_U32         : VOP2be_Real_e32e64_gfx9 <0x19, "V_ADD_CO_U32",     "v_add_co_u32">;
defm V_SUB_CO_U32         : VOP2be_Real_e32e64_gfx9 <0x1a, "V_SUB_CO_U32",     "v_sub_co_u32">;
defm V_SUBREV_CO_U32      : VOP2be_Real_e32e64_gfx9 <0x1b, "V_SUBREV_CO_U32",  "v_subrev_co_u32">;
defm V_ADDC_CO_U32        : VOP2be_Real_e32e64_gfx9 <0x1c, "V_ADDC_U32",    "v_addc_co_u32">;
defm V_SUBB_CO_U32        : VOP2be_Real_e32e64_gfx9 <0x1d, "V_SUBB_U32",    "v_subb_co_u32">;
defm V_SUBBREV_CO_U32     : VOP2be_Real_e32e64_gfx9 <0x1e, "V_SUBBREV_U32", "v_subbrev_co_u32">;

defm V_ADD_U32            : VOP2_Real_e32e64_gfx9 <0x34>;
defm V_SUB_U32            : VOP2_Real_e32e64_gfx9 <0x35>;
defm V_SUBREV_U32         : VOP2_Real_e32e64_gfx9 <0x36>;

defm V_BFM_B32            : VOP2_Real_e64only_vi <0x293>;
defm V_BCNT_U32_B32       : VOP2_Real_e64only_vi <0x28b>;
defm V_MBCNT_LO_U32_B32   : VOP2_Real_e64only_vi <0x28c>;
defm V_MBCNT_HI_U32_B32   : VOP2_Real_e64only_vi <0x28d>;
defm V_LDEXP_F32          : VOP2_Real_e64only_vi <0x288>;
defm V_CVT_PKACCUM_U8_F32 : VOP2_Real_e64only_vi <0x1f0>;
defm V_CVT_PKNORM_I16_F32 : VOP2_Real_e64only_vi <0x294>;
defm V_CVT_PKNORM_U16_F32 : VOP2_Real_e64only_vi <0x295>;
defm V_CVT_PKRTZ_F16_F32  : VOP2_Real_e64only_vi <0x296>;
defm V_CVT_PK_U16_U32     : VOP2_Real_e64only_vi <0x297>;
defm V_CVT_PK_I16_I32     : VOP2_Real_e64only_vi <0x298>;

defm V_ADD_F16            : VOP2_Real_e32e64_vi <0x1f>;
defm V_SUB_F16            : VOP2_Real_e32e64_vi <0x20>;
defm V_SUBREV_F16         : VOP2_Real_e32e64_vi <0x21>;
defm V_MUL_F16            : VOP2_Real_e32e64_vi <0x22>;
defm V_MAC_F16            : VOP2_Real_e32e64_vi <0x23>;
defm V_MADMK_F16          : VOP2_Real_MADK_vi <0x24>;
defm V_MADAK_F16          : VOP2_Real_MADK_vi <0x25>;
defm V_ADD_U16            : VOP2_Real_e32e64_vi <0x26>;
defm V_SUB_U16            : VOP2_Real_e32e64_vi <0x27>;
defm V_SUBREV_U16         : VOP2_Real_e32e64_vi <0x28>;
defm V_MUL_LO_U16         : VOP2_Real_e32e64_vi <0x29>;
defm V_LSHLREV_B16        : VOP2_Real_e32e64_vi <0x2a>;
defm V_LSHRREV_B16        : VOP2_Real_e32e64_vi <0x2b>;
defm V_ASHRREV_I16        : VOP2_Real_e32e64_vi <0x2c>;
defm V_MAX_F16            : VOP2_Real_e32e64_vi <0x2d>;
defm V_MIN_F16            : VOP2_Real_e32e64_vi <0x2e>;
defm V_MAX_U16            : VOP2_Real_e32e64_vi <0x2f>;
defm V_MAX_I16            : VOP2_Real_e32e64_vi <0x30>;
defm V_MIN_U16            : VOP2_Real_e32e64_vi <0x31>;
defm V_MIN_I16            : VOP2_Real_e32e64_vi <0x32>;
defm V_LDEXP_F16          : VOP2_Real_e32e64_vi <0x33>;

let SubtargetPredicate = isGFX8GFX9 in {

// Aliases to simplify matching of floating-point instructions that
// are VOP2 on SI and VOP3 on VI.
class SI2_VI3Alias <string name, VOP3_Real inst> : InstAlias <
  name#" $dst, $src0, $src1",
  !if(inst.Pfl.HasOMod,
      (inst VGPR_32:$dst, 0, VCSrc_f32:$src0, 0, VCSrc_f32:$src1, 0, 0),
      (inst VGPR_32:$dst, 0, VCSrc_f32:$src0, 0, VCSrc_f32:$src1, 0))
>, PredicateControl {
  let UseInstAsmMatchConverter = 0;
  let AsmVariantName = AMDGPUAsmVariants.VOP3;
}

def : SI2_VI3Alias <"v_ldexp_f32", V_LDEXP_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkaccum_u8_f32", V_CVT_PKACCUM_U8_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_i16_f32", V_CVT_PKNORM_I16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pknorm_u16_f32", V_CVT_PKNORM_U16_F32_e64_vi>;
def : SI2_VI3Alias <"v_cvt_pkrtz_f16_f32", V_CVT_PKRTZ_F16_F32_e64_vi>;

defm : VOP2eInstAliases<V_CNDMASK_B32_e32, V_CNDMASK_B32_e32_vi>;

} // End SubtargetPredicate = isGFX8GFX9

let SubtargetPredicate = isGFX9Only in {

defm : VOP2bInstAliases<V_ADD_U32_e32,     V_ADD_CO_U32_e32_gfx9,     "v_add_co_u32">;
defm : VOP2bInstAliases<V_ADDC_U32_e32,    V_ADDC_CO_U32_e32_gfx9,    "v_addc_co_u32">;
defm : VOP2bInstAliases<V_SUB_U32_e32,     V_SUB_CO_U32_e32_gfx9,     "v_sub_co_u32">;
defm : VOP2bInstAliases<V_SUBB_U32_e32,    V_SUBB_CO_U32_e32_gfx9,    "v_subb_co_u32">;
defm : VOP2bInstAliases<V_SUBREV_U32_e32,  V_SUBREV_CO_U32_e32_gfx9,  "v_subrev_co_u32">;
defm : VOP2bInstAliases<V_SUBBREV_U32_e32, V_SUBBREV_CO_U32_e32_gfx9, "v_subbrev_co_u32">;

} // End SubtargetPredicate = isGFX9Only

let SubtargetPredicate = HasDLInsts in {

defm V_FMAC_F32 : VOP2_Real_e32e64_vi <0x3b>;
defm V_XNOR_B32 : VOP2_Real_e32e64_vi <0x3d>;

} // End SubtargetPredicate = HasDLInsts

let AssemblerPredicate = isGFX90APlus, DecoderNamespace = "GFX90A" in {
  multiclass VOP2_Real_e32_gfx90a <bits<6> op> {
    def _e32_gfx90a :
      VOP2_Real<!cast<VOP2_Pseudo>(NAME#"_e32"), SIEncodingFamily.GFX90A>,
      VOP2e<op{5-0}, !cast<VOP2_Pseudo>(NAME#"_e32").Pfl>;
  }

  multiclass VOP2_Real_e64_gfx90a <bits<10> op> {
    def _e64_gfx90a :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX90A>,
      VOP3e_vi <op, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }

  multiclass Base_VOP2_Real_e32e64_gfx90a <bits<6> op> :
    VOP2_Real_e32_gfx90a<op>,
    VOP2_Real_e64_gfx90a<{0, 1, 0, 0, op{5-0}}>;

  multiclass VOP2_Real_e32e64_gfx90a <bits<6> op> :
    Base_VOP2_Real_e32e64_gfx90a<op> {

    foreach _ = BoolToList<!cast<VOP2_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
      def _dpp_gfx90a :
        VOP_DPP_Real<!cast<VOP2_DPP_Pseudo>(NAME#"_dpp"), SIEncodingFamily.GFX90A>,
        VOP2_DPPe<op, !cast<VOP2_DPP_Pseudo>(NAME#"_dpp")> {
          let DecoderNamespace = "SDWA9";
        }
  }
} // End AssemblerPredicate = isGFX90APlus, DecoderNamespace = "GFX90A"

let SubtargetPredicate = isGFX90APlus in {
  defm V_FMAC_F64       : VOP2_Real_e32e64_gfx90a <0x4>;
  let IsSingle = 1 in {
    defm V_MUL_LEGACY_F32 : VOP2_Real_e64_gfx90a <0x2a1>;
  }
} // End SubtargetPredicate = isGFX90APlus

multiclass VOP2_Real_DOT_ACC_gfx9<bits<6> op> : VOP2_Real_e32_vi<op> {
  def _dpp_vi : VOP2_DPP<op, !cast<VOP2_DPP_Pseudo>(NAME#"_dpp")>;
}

multiclass VOP2_Real_DOT_ACC_gfx10<bits<6> op> :
  VOP2_Real_e32_gfx10<op>,
  VOP2_Real_dpp_gfx10<op>,
  VOP2_Real_dpp8_gfx10<op>;

let SubtargetPredicate = HasDot5Insts in {
  defm V_DOT2C_F32_F16 : VOP2_Real_DOT_ACC_gfx9<0x37>;
  // NB: Opcode conflicts with V_DOT8C_I32_I4
  // This opcode exists in gfx 10.1* only
  defm V_DOT2C_F32_F16 : VOP2_Real_DOT_ACC_gfx10<0x02>;
}

let SubtargetPredicate = HasDot6Insts in {
  defm V_DOT4C_I32_I8  : VOP2_Real_DOT_ACC_gfx9<0x39>;
  defm V_DOT4C_I32_I8  : VOP2_Real_DOT_ACC_gfx10<0x0d>;
}

let SubtargetPredicate = HasDot4Insts in {
  defm V_DOT2C_I32_I16 : VOP2_Real_DOT_ACC_gfx9<0x38>;
}
let SubtargetPredicate = HasDot3Insts in {
  defm V_DOT8C_I32_I4  : VOP2_Real_DOT_ACC_gfx9<0x3a>;
}

let SubtargetPredicate = HasPkFmacF16Inst in {
defm V_PK_FMAC_F16 : VOP2_Real_e32_vi<0x3c>;
} // End SubtargetPredicate = HasPkFmacF16Inst

let SubtargetPredicate = HasDot3Insts in {
  // NB: Opcode conflicts with V_DOT2C_F32_F16
  let DecoderNamespace = "GFX10_B" in
  defm V_DOT8C_I32_I4 : VOP2_Real_DOT_ACC_gfx10<0x02>;
}
