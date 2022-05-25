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

//===----------------------------------------------------------------------===//
// VOPD classes
//===----------------------------------------------------------------------===//

class VOPD_Base<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
                VOPD_Component XasVC, VOPD_Component YasVC>
    : VOPAnyCommon<outs, ins, asm, []>,
      VOP<NAME>,
      SIMCInstr<NAME, SIEncodingFamily.GFX11> {
  // Fields for table indexing
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<5> OpX = XasVC.VOPDOp;
  bits<5> OpY = YasVC.VOPDOp;

  let VALU = 1;

  let DecoderNamespace = "GFX11";
  let AssemblerPredicate = isGFX11Plus;
  let WaveSizePredicate = isWave32;
  let isCodeGenOnly = 0;
  let SubtargetPredicate = isGFX11Plus;
  let AsmMatchConverter  = "cvtVOPD";
  let Size = 8;
  let ReadsModeReg = !or(VDX.ReadsModeReg, VDY.ReadsModeReg);
  let mayRaiseFPException = ReadsModeReg;

  let Uses = RegListUnion<VDX.Uses, VDY.Uses>.ret;
  let Defs = RegListUnion<VDX.Defs, VDY.Defs>.ret;
  let SchedRW = !listconcat(VDX.SchedRW, VDY.SchedRW);
}

class VOPD<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
           VOPD_Component XasVC, VOPD_Component YasVC>
    : VOPD_Base<outs, ins, asm, VDX, VDY, XasVC, YasVC>,
      VOPDe<XasVC.VOPDOp{3-0}, YasVC.VOPDOp> {
  let Inst{16-9} = !if (!eq(VDX.Mnemonic, "v_mov_b32"), 0x0, vsrc1X);
  let Inst{48-41} = !if (!eq(VDY.Mnemonic, "v_mov_b32"), 0x0, vsrc1Y);
}

class VOPD_MADK<dag outs, dag ins, string asm, VOP_Pseudo VDX, VOP_Pseudo VDY,
                VOPD_Component XasVC, VOPD_Component YasVC>
    : VOPD_Base<outs, ins, asm, VDX, VDY, XasVC, YasVC>,
      VOPD_MADKe<XasVC.VOPDOp{3-0}, YasVC.VOPDOp> {
  let Inst{16-9} = !if (!eq(VDX.Mnemonic, "v_mov_b32"), 0x0, vsrc1X);
  let Inst{48-41} = !if (!eq(VDY.Mnemonic, "v_mov_b32"), 0x0, vsrc1Y);
  let Size = 12;
}

// V_DUAL_DOT2ACC_F32_BF16 is a legal instruction, but V_DOT2ACC_F32_BF16 is
// not. Since we generate the DUAL form by converting from the normal form we
// will never generate it.
defvar VOPDYPseudos = [
  "V_FMAC_F32_e32", "V_FMAAK_F32", "V_FMAMK_F32", "V_MUL_F32_e32",
  "V_ADD_F32_e32", "V_SUB_F32_e32", "V_SUBREV_F32_e32", "V_MUL_LEGACY_F32_e32",
  "V_MOV_B32_e32", "V_CNDMASK_B32_e32", "V_MAX_F32_e32", "V_MIN_F32_e32",
  "V_DOT2C_F32_F16_e32", "V_ADD_U32_e32", "V_LSHLREV_B32_e32", "V_AND_B32_e32"
];
defvar VOPDXPseudos = VOPDYPseudos[0...VOPDX_Max_Index];

def VOPDDstYOperand : RegisterOperand<VGPR_32, "printRegularOperand"> {
  let DecoderMethod = "decodeOperandVOPDDstY";
}

foreach x = VOPDXPseudos in {
  foreach y = VOPDYPseudos in {
    defvar xInst = !cast<VOP_Pseudo>(x);
    defvar yInst = !cast<VOP_Pseudo>(y);
    defvar XasVC = !cast<VOPD_Component>(x);
    defvar YasVC = !cast<VOPD_Component>(y);
    defvar isMADK = !or(!eq(x, "V_FMAAK_F32"), !eq(x, "V_FMAMK_F32"),
                        !eq(y, "V_FMAAK_F32"), !eq(y, "V_FMAMK_F32"));
    // If X or Y is MADK (have a mandatory immediate), all src operands which
    // may contain an optional literal must use the VSrc_*_Deferred operand
    // type. Optional literal operands in MADK VOPD components always use this
    // operand form. If Both X and Y are MADK, the mandatory literal of X
    // additionally must use an alternate operand format which defers to the
    // 'real' Y literal
    defvar isOpXMADK = !or(!eq(x, "V_FMAAK_F32"), !eq(x, "V_FMAMK_F32"));
    defvar isOpYMADK = !or(!eq(y, "V_FMAAK_F32"), !eq(y, "V_FMAMK_F32"));
    defvar OpName = "V_DUAL_" # !substr(x,2) # "_X_" # !substr(y,2);
    defvar outs = (outs VGPRSrc_32:$vdstX, VOPDDstYOperand:$vdstY);
    if !or(isOpXMADK, isOpYMADK) then {
      if !and(isOpXMADK, isOpYMADK) then {
        defvar X_MADK_Pfl = !cast<VOP_MADK_Base>(xInst.Pfl);
        defvar ins = !con(xInst.Pfl.InsVOPDXDeferred, yInst.Pfl.InsVOPDY);
        defvar asm = XasVC.VOPDName #" "# X_MADK_Pfl.AsmVOPDXDeferred #" :: "# YasVC.VOPDName #" "# yInst.Pfl.AsmVOPDY;
        def OpName : VOPD_MADK<outs, ins, asm, xInst, yInst, XasVC, YasVC>;
      } else {
        defvar asm = XasVC.VOPDName #" "# xInst.Pfl.AsmVOPDX #" :: "# YasVC.VOPDName #" "# yInst.Pfl.AsmVOPDY;
        if isOpXMADK then {
          assert !not(isOpYMADK), "Expected only OpX as MADK";
          defvar ins = !con(xInst.Pfl.InsVOPDX, yInst.Pfl.InsVOPDYDeferred);
          def OpName : VOPD_MADK<outs, ins, asm, xInst, yInst, XasVC, YasVC>;
        } else {
          assert !not(isOpXMADK), "Expected only OpY as MADK";
          defvar ins = !con(xInst.Pfl.InsVOPDXDeferred, yInst.Pfl.InsVOPDY);
          def OpName : VOPD_MADK<outs, ins, asm, xInst, yInst, XasVC, YasVC>;
        }
      }
    } else {
      defvar ins = !con(xInst.Pfl.InsVOPDX, yInst.Pfl.InsVOPDY);
      defvar asm = XasVC.VOPDName #" "# xInst.Pfl.AsmVOPDX #" :: "# YasVC.VOPDName #" "# yInst.Pfl.AsmVOPDY;
      def OpName : VOPD<outs, ins, asm, xInst, yInst, XasVC, YasVC>;
    }
  }
}

