//===-- VOPDInstructions.td - Vector Instruction Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Encodings
//===----------------------------------------------------------------------===//

class VOPDe<bits<4> opX, bits<5> opY> : Enc64 {
  bits<9> src0X;
  bits<8> vsrc1X;
  bits<8> vdstX;
  bits<9> src0Y;
  bits<8> vsrc1Y;
  bits<8> vdstY;

  let Inst{8-0} = src0X;
  let Inst{16-9} = vsrc1X;
  let Inst{21-17} = opY;
  let Inst{25-22} = opX;
  let Inst{31-26} = 0x32; // encoding
  let Inst{40-32} = src0Y;
  let Inst{48-41} = vsrc1Y;
  let Inst{55-49} = vdstY{7-1};
  let Inst{63-56} = vdstX;
}

class VOPD_MADKe<bits<4> opX, bits<5> opY> : Enc96 {
  bits<9> src0X;
  bits<8> vsrc1X;
  bits<8> vdstX;
  bits<9> src0Y;
  bits<8> vsrc1Y;
  bits<8> vdstY;
  bits<32> imm;

  let Inst{8-0} = src0X;
  let Inst{16-9} = vsrc1X;
  let Inst{21-17} = opY;
  let Inst{25-22} = opX;
  let Inst{31-26} = 0x32; // encoding
  let Inst{40-32} = src0Y;
  let Inst{48-41} = vsrc1Y;
  let Inst{55-49} = vdstY{7-1};
  let Inst{63-56} = vdstX;
  let Inst{95-64} = imm;
}

class VOPD3e<bits<6> opX, bits<6> opY, VOP_Pseudo VDX, VOP_Pseudo VDY> : Enc96 {
  bits<9> src0X;
  bits<8> vsrc1X;
  bits<8> vsrc2X;
  bits<8> vdstX;
  bits<9> src0Y;
  bits<8> vsrc1Y;
  bits<8> vsrc2Y;
  bits<8> vdstY;
  // neg modifiers
  bit src0X_modifiers;
  bit src0Y_modifiers;
  bit vsrc1X_modifiers;
  bit vsrc1Y_modifiers;
  bit vsrc2X_modifiers;
  bit vsrc2Y_modifiers;
  bits<8> bitop3;

  let Inst{8-0} = src0X;
  let Inst{17-12} = opY;
  let Inst{23-18} = opX;
  let Inst{31-24} = 0xcf; // encoding
  let Inst{40-32} = src0Y;
  let Inst{41} = !if(VDX.Pfl.HasModifiers, src0X_modifiers, 0);
  let Inst{42} = !if(!and(VDX.Pfl.HasSrc1, VDX.Pfl.HasModifiers), vsrc1X_modifiers, 0);
  let Inst{43} = !if(!and(VDX.Pfl.HasVOPD3Src2, VDX.Pfl.HasModifiers), vsrc2X_modifiers, 0);
  let Inst{44} = !if(VDY.Pfl.HasModifiers, src0Y_modifiers, 0);
  let Inst{45} = !if(!and(VDY.Pfl.HasSrc1, VDY.Pfl.HasModifiers), vsrc1Y_modifiers, 0);
  let Inst{46} = !if(!and(VDY.Pfl.HasVOPD3Src2, VDY.Pfl.HasModifiers), vsrc2Y_modifiers, 0);
  let Inst{55-48} = !if(!eq(!find(VDX.Pfl.AsmVOPD3X, "$vsrc1X"), -1), 0, vsrc1X);

  // Despite the vsrc operand name, SGPRs can be used for vsrc2X for
  // V_DUAL_CNDMASK_B32
  let Inst{63-56} = !if(!eq(!find(VDX.Pfl.AsmVOPD3X, "$vsrc2X"), -1), 0, vsrc2X);
  let Inst{71-64} = vdstX;
  let Inst{79-72} = !if(!eq(!find(VDY.Pfl.AsmVOPD3Y, "$vsrc1Y"), -1), 0, vsrc1Y);
  let Inst{87-80} = !if(!ne(!find(VDY.Pfl.AsmVOPD3Y, "bitop"), -1), bitop3,
                        !if(!eq(!find(VDY.Pfl.AsmVOPD3Y, "$vsrc2Y"), -1), 0, vsrc2Y));
  let Inst{95-88} = vdstY;
}

//===----------------------------------------------------------------------===//
// VOPD classes
//===----------------------------------------------------------------------===//


class GFXGenD<GFXGen Gen, list<string> DXPseudos, list<string> DYPseudos,
              Predicate subtargetPred = Gen.AssemblerPredicate> :
    GFXGen<Gen.AssemblerPredicate, Gen.DecoderNamespace, Gen.Suffix,
           Gen.Subtarget> {
  list<string> VOPDXPseudos = DXPseudos;
  list<string> VOPDYPseudos = DYPseudos;
  Predicate SubtargetPredicate = subtargetPred;
}

class VOPD_Base<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
                VOPD_Component XasVC, VOPD_Component YasVC, GFXGenD Gen>
    : VOPAnyCommon<outs, ins, asm, []>,
      VOP<NAME>,
      SIMCInstr<NAME, Gen.Subtarget> {
  // Fields for table indexing
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<6> OpX = XasVC.VOPDOp;
  bits<6> OpY = YasVC.VOPDOp;
  bits<4> SubTgt = Gen.Subtarget;

  let VALU = 1;

  let DecoderNamespace = Gen.DecoderNamespace;
  let AssemblerPredicate = Gen.AssemblerPredicate;
  let WaveSizePredicate = isWave32;
  let isCodeGenOnly = 0;
  let SubtargetPredicate = Gen.SubtargetPredicate;
  let AsmMatchConverter  = "cvtVOPD";
  let Size = 8;
  let ReadsModeReg = !or(VDX.ReadsModeReg, VDY.ReadsModeReg);
  let mayRaiseFPException = ReadsModeReg;

  // V_DUAL_FMAC and V_DUAL_DOT2ACC_F32_F16 and V_DUAL_DOT2ACC_F32_BF16 need a
  // dummy src2 tied to dst for passes to track its uses. Its presence does not
  // affect VOPD formation rules because the rules for src2 and dst are the
  // same. src2X and src2Y should not be encoded.
  bit hasSrc2AccX = !or(!eq(VDX.Mnemonic, "v_fmac_f32"), !eq(VDX.Mnemonic, "v_dot2c_f32_f16"), !eq(VDX.Mnemonic, "v_dot2c_f32_bf16"));
  bit hasSrc2AccY = !or(!eq(VDY.Mnemonic, "v_fmac_f32"), !eq(VDY.Mnemonic, "v_dot2c_f32_f16"), !eq(VDY.Mnemonic, "v_dot2c_f32_bf16"));
  string ConstraintsX = !if(hasSrc2AccX, "$src2X = $vdstX", "");
  string ConstraintsY = !if(hasSrc2AccY, "$src2Y = $vdstY", "");
  let Constraints =
      ConstraintsX # !if(!and(hasSrc2AccX, hasSrc2AccY), ", ", "") # ConstraintsY;

  let Uses = RegListUnion<VDX.Uses, VDY.Uses>.ret;
  let Defs = RegListUnion<VDX.Defs, VDY.Defs>.ret;
  let SchedRW = !listconcat(VDX.SchedRW, VDY.SchedRW);
}

class VOPD<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
           VOPD_Component XasVC, VOPD_Component YasVC, GFXGenD Gen>
    : VOPD_Base<outs, ins, asm, VDX, VDY, XasVC, YasVC, Gen>,
      VOPDe<XasVC.VOPDOp{3-0}, YasVC.VOPDOp{4-0}> {
  let Inst{16-9} = !if (!eq(VDX.Mnemonic, "v_mov_b32"), 0x0, vsrc1X);
  let Inst{48-41} = !if (!eq(VDY.Mnemonic, "v_mov_b32"), 0x0, vsrc1Y);
}

class VOPD_MADK<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
                VOPD_Component XasVC, VOPD_Component YasVC, GFXGenD Gen>
    : VOPD_Base<outs, ins, asm, VDX, VDY, XasVC, YasVC, Gen>,
      VOPD_MADKe<XasVC.VOPDOp{3-0}, YasVC.VOPDOp{4-0}> {
  let Inst{16-9} = !if (!eq(VDX.Mnemonic, "v_mov_b32"), 0x0, vsrc1X);
  let Inst{48-41} = !if (!eq(VDY.Mnemonic, "v_mov_b32"), 0x0, vsrc1Y);
  let Size = 12;
  let FixedSize = 1;
}

class VOPD3<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
            VOPD_Component XasVC, VOPD_Component YasVC, GFXGenD Gen>
    : VOPD_Base<outs, ins, asm, VDX, VDY, XasVC, YasVC, Gen>,
      VOPD3e<XasVC.VOPDOp, YasVC.VOPDOp, VDX, VDY> {
  let VOPD3 = 1;
  let Size = 12;
  // VOPD3 uses promoted form of VOP2 instructions, so V_CNDMASK_B32 is not
  // limited to VCC src2 only, and a real SGPR will be used as an operand
  // instead.
  defvar UsesX = !if(!eq(VDX, V_CNDMASK_B32_e32), !filter(x, VDX.Uses, !ne(x, VCC)), VDX.Uses);
  defvar UsesY = !if(!eq(VDY, V_CNDMASK_B32_e32), !filter(x, VDY.Uses, !ne(x, VCC)), VDY.Uses);
  let Uses = RegListUnion<UsesX, UsesY>.ret;
}

defvar VOPDPseudosCommon = [
  "V_FMAC_F32_e32", "V_FMAAK_F32", "V_FMAMK_F32", "V_MUL_F32_e32",
  "V_ADD_F32_e32", "V_SUB_F32_e32", "V_SUBREV_F32_e32", "V_MUL_LEGACY_F32_e32",
  "V_MOV_B32_e32", "V_CNDMASK_B32_e32", "V_MAX_F32_e32", "V_MIN_F32_e32",
  "V_DOT2C_F32_F16_e32", "V_DOT2C_F32_BF16_e32"
];
defvar VOPDYOnlyPseudosCommon = ["V_ADD_U32_e32", "V_LSHLREV_B32_e32"];
defvar VOPDYOnlyPseudosGFX11_12 = ["V_AND_B32_e32"];
defvar VOPDYOnlyPseudosGFX1250 = ["V_MAX_I32_e32", "V_MIN_I32_e32",
                                  "V_SUB_U32_e32", "V_LSHRREV_B32_e32",
                                  "V_ASHRREV_I32_e32"];

defvar VOPDXPseudosGFX11 = VOPDPseudosCommon;
defvar VOPDXPseudosGFX12 = VOPDPseudosCommon;
defvar VOPDYPseudosGFX11 = !listconcat(VOPDXPseudosGFX11, VOPDYOnlyPseudosCommon, VOPDYOnlyPseudosGFX11_12);
defvar VOPDYPseudosGFX12 = !listconcat(VOPDXPseudosGFX12, VOPDYOnlyPseudosCommon, VOPDYOnlyPseudosGFX11_12);
defvar VOPDYPseudosGFX1250 = !listconcat(VOPDXPseudosGFX12, VOPDYOnlyPseudosCommon, VOPDYOnlyPseudosGFX1250);

def GFX11GenD : GFXGenD<GFX11Gen, VOPDXPseudosGFX11, VOPDYPseudosGFX11>;
def GFX12GenD : GFXGenD<GFX12Not12_50Gen, VOPDXPseudosGFX12, VOPDYPseudosGFX12>;
def GFX1250GenD : GFXGenD<GFX1250Gen, VOPDXPseudosGFX12, VOPDYPseudosGFX1250>;


def VOPDDstYOperand : RegisterOperand<VGPR_32, "printRegularOperand"> {
  let DecoderMethod = "decodeOperandVOPDDstY";
}

class getRenamed<string VOPDName, GFXGen Gen> {
  string ret = !cond(!eq(Gen.Subtarget, GFX11Gen.Subtarget) : VOPDName,
                     !eq(VOPDName, "v_dual_max_f32")        : "v_dual_max_num_f32",
                     !eq(VOPDName, "v_dual_min_f32")        : "v_dual_min_num_f32",
                     true                                   : VOPDName);
}

foreach Gen = [GFX11GenD, GFX12GenD, GFX1250GenD] in {
  foreach x = Gen.VOPDXPseudos in {
    foreach y = Gen.VOPDYPseudos in {
      defvar xInst = !cast<VOP_Pseudo>(x);
      defvar yInst = !cast<VOP_Pseudo>(y);
      defvar XasVC = !cast<VOPD_Component>(x);
      defvar YasVC = !cast<VOPD_Component>(y);
      defvar xAsmName = getRenamed<XasVC.VOPDName, Gen>.ret;
      defvar yAsmName = getRenamed<YasVC.VOPDName, Gen>.ret;
      defvar isMADK = !or(!eq(x, "V_FMAAK_F32"), !eq(x, "V_FMAMK_F32"),
                          !eq(y, "V_FMAAK_F32"), !eq(y, "V_FMAMK_F32"));
      defvar isOpXMADK = !or(!eq(x, "V_FMAAK_F32"), !eq(x, "V_FMAMK_F32"));
      defvar isOpYMADK = !or(!eq(y, "V_FMAAK_F32"), !eq(y, "V_FMAMK_F32"));
      defvar OpName = "V_DUAL_" # !substr(x,2) # "_X_" # !substr(y,2) # Gen.Suffix;
      defvar outs = (outs VGPROp_32:$vdstX, VOPDDstYOperand:$vdstY);
      if !or(isOpXMADK, isOpYMADK) then {
        // If Both X and Y are MADK, the mandatory literal of X additionally must
        // use an alternate operand format which defers to the 'real' Y literal.
        defvar isOpXYMADK = !and(isOpXMADK, isOpYMADK);
        defvar X_MADK_Pfl = !cast<VOP_MADK_Base>(xInst.Pfl);
        defvar asm = xAsmName #" "#
                     !if(isOpXYMADK, X_MADK_Pfl.AsmVOPDX_immX, xInst.Pfl.AsmVOPDX)#
                     " :: "# yAsmName #" "# yInst.Pfl.AsmVOPDY;
        defvar ins = !con(!if(isOpXYMADK, xInst.Pfl.InsVOPDX_immX, xInst.Pfl.InsVOPDX),
                          yInst.Pfl.InsVOPDY);
        def OpName : VOPD_MADK<outs, ins, asm, xInst, yInst, XasVC, YasVC, Gen>;
      } else {
        defvar ins = !con(xInst.Pfl.InsVOPDX, yInst.Pfl.InsVOPDY);
        defvar asm = xAsmName #" "# xInst.Pfl.AsmVOPDX #" :: "# yAsmName #" "# yInst.Pfl.AsmVOPDY;
        def OpName : VOPD<outs, ins, asm, xInst, yInst, XasVC, YasVC, Gen>;
      }
    }
  }
}

defvar VOPD3XPseudosExtra = ["V_ADD_U32_e32", "V_LSHLREV_B32_e32", "V_FMA_F32_e64", "V_SUB_U32_e32",
                             "V_LSHRREV_B32_e32", "V_ASHRREV_I32_e32", "V_FMA_F64_e64", "V_ADD_F64_pseudo_e32",
                             "V_MUL_F64_pseudo_e32", "V_MAX_NUM_F64_e32", "V_MIN_NUM_F64_e32"];
defvar VOPD3XPseudosGFX1250 = !listconcat(
                                !filter(x, VOPDXPseudosGFX12, !and(!eq(!find(x, "FMAAK"), -1),
                                                                   !eq(!find(x, "FMAMK"), -1))),
                                VOPD3XPseudosExtra);
defvar VOPD3YPseudosExtra = ["V_BITOP3_B32_e64", "V_FMA_F32_e64"];
defvar VOPD3YPseudosGFX1250 = !listconcat(
                                !filter(x, VOPDYPseudosGFX1250, !and(!eq(!find(x, "FMAAK"), -1),
                                                                     !eq(!find(x, "FMAMK"), -1))),
                                VOPD3YPseudosExtra);

def GFX1250GenD3 : GFXGenD<GFX1250Gen, VOPD3XPseudosGFX1250, VOPD3YPseudosGFX1250>;

class getOpcMap<string OPName> {
  defvar BaseName = !substr(OPName,2);
  string ret = !cond(!eq(BaseName, "BITOP3_B32_e64")   : "BITOP2_B32_e64",
                     1 : BaseName);
}

foreach Gen = [GFX1250GenD3] in {
  foreach x = Gen.VOPDXPseudos in {
    foreach y = Gen.VOPDYPseudos in {
      defvar xInst = !cast<VOP_Pseudo>(x);
      defvar yInst = !cast<VOP_Pseudo>(y);
      defvar XasVC = !cast<VOPD_Component>(x);
      defvar YasVC = !cast<VOPD_Component>(y);
      defvar xAsmName = getRenamed<XasVC.VOPDName, Gen>.ret;
      defvar yAsmName = getRenamed<YasVC.VOPDName, Gen>.ret;
      defvar OpName = "V_DUAL_" # getOpcMap<x>.ret # "_X_" # getOpcMap<y>.ret # "_e96" # Gen.Suffix;
      defvar asm = xAsmName # xInst.Pfl.AsmVOPD3X #" :: "# yAsmName #" "# yInst.Pfl.AsmVOPD3Y;
      defvar ins = !con(xInst.Pfl.InsVOPD3X, yInst.Pfl.InsVOPD3Y);
      defvar outs = (outs xInst.Pfl.DstRC:$vdstX, yInst.Pfl.DstRC:$vdstY);
      def OpName : VOPD3<outs, ins, asm, xInst, yInst, XasVC, YasVC, Gen>;
    }
  }
}
