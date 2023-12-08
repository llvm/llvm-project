//===-- LDSDIRInstructions.td - LDS Direct Instruction Definitions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LDSDIR encoding
//===----------------------------------------------------------------------===//

class LDSDIRe<bits<2> op, bit is_direct> : Enc32 {
  // encoding fields
  bits<2> attrchan;
  bits<6> attr;
  bits<4> waitvdst;
  bits<8> vdst;

  // encoding
  let Inst{31-24} = 0xce; // encoding
  let Inst{23-22} = 0x0; // reserved
  let Inst{21-20} = op;
  let Inst{19-16} = waitvdst;
  let Inst{15-10} = !if(is_direct, ?, attr);
  let Inst{9-8} = !if(is_direct, ?, attrchan);
  let Inst{7-0} = vdst;
}

//===----------------------------------------------------------------------===//
// LDSDIR Classes
//===----------------------------------------------------------------------===//

class LDSDIR_getIns<bit direct> {
  dag ret = !if(direct,
    (ins wait_vdst:$waitvdst),
    (ins InterpAttr:$attr, InterpAttrChan:$attrchan, wait_vdst:$waitvdst)
  );
}

class LDSDIR_Common<string opName, string asm = "", bit direct> : InstSI<
    (outs VGPR_32:$vdst),
    LDSDIR_getIns<direct>.ret,
    asm> {
  let LDSDIR = 1;
  let EXP_CNT = 1;

  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;

  string Mnemonic = opName;
  let UseNamedOperandTable = 1;

  let Uses = [M0, EXEC];
  let DisableWQM = 0;
  let SchedRW = [WriteLDS];

  bit is_direct;
  let is_direct = direct;
}

class LDSDIR_Pseudo<string opName, bit direct> :
  LDSDIR_Common<opName, "", direct>,
  SIMCInstr<opName, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class LDSDIR_getAsm<bit direct> {
  string ret = !if(direct,
    " $vdst$waitvdst",
    " $vdst, $attr$attrchan$waitvdst"
  );
}

class LDSDIR_Real<bits<2> op, LDSDIR_Pseudo lds, int subtarget> :
  LDSDIR_Common<lds.Mnemonic,
                lds.Mnemonic # LDSDIR_getAsm<lds.is_direct>.ret,
                lds.is_direct>,
  SIMCInstr <lds.Mnemonic, subtarget>,
  LDSDIRe<op, lds.is_direct> {
  let isPseudo = 0;
  let isCodeGenOnly = 0;
}

//===----------------------------------------------------------------------===//
// LDS Direct Instructions
//===----------------------------------------------------------------------===//

def LDS_DIRECT_LOAD : LDSDIR_Pseudo<"lds_direct_load", 1>;
def LDS_PARAM_LOAD : LDSDIR_Pseudo<"lds_param_load", 0>;

def : GCNPat <
  (f32 (int_amdgcn_lds_direct_load M0)),
  (LDS_DIRECT_LOAD 0)
>;

def : GCNPat <
  (f32 (int_amdgcn_lds_param_load timm:$attrchan, timm:$attr, M0)),
  (LDS_PARAM_LOAD timm:$attr, timm:$attrchan, 0)
>;

//===----------------------------------------------------------------------===//
// GFX11+
//===----------------------------------------------------------------------===//

multiclass LDSDIR_Real_gfx11<bits<2> op, LDSDIR_Pseudo lds = !cast<LDSDIR_Pseudo>(NAME)> {
  def _gfx11 : LDSDIR_Real<op, lds, SIEncodingFamily.GFX11> {
    let AssemblerPredicate = isGFX11Plus;
    let DecoderNamespace = "GFX11";
  }
}

defm LDS_PARAM_LOAD : LDSDIR_Real_gfx11<0x0>;
defm LDS_DIRECT_LOAD : LDSDIR_Real_gfx11<0x1>;
