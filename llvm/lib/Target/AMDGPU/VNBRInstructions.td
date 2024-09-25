//===-- VNBRInstructions.td - VNBR Instruction Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VNBR Encoding
//===----------------------------------------------------------------------===//

class VNBRe_Base<bits<4> op> : Enc64 {
  bit en;
  bits<3> sema_id;
  bits<3> sema_wave_id;
  bits<3> sema_id_refl;
  bits<3> sema_wave_id_refl;
  bits<4> wait_va_dst;

  let Inst{31-25} = 0x7d;

  let Inst{0}     = en;
  let Inst{4-2}   = sema_id;
  let Inst{7-5}   = sema_id_refl;
  let Inst{10-8}  = sema_wave_id;
  let Inst{14-12} = sema_wave_id_refl;
  let Inst{19-16} = wait_va_dst;
  let Inst{24-21} = op;
}

class VNBRe<bits<4> op> : VNBRe_Base<op> {
  bits<10> vdst;
  bits<10> vdst_refl;
  bits<10> vsrc;

  let Inst{41-32} = vsrc;
  let Inst{51-42} = vdst;
  let Inst{61-52} = vdst_refl;

  let en = 1;
}

class VNBRe_no_write<bits<4> op> : VNBRe_Base<op> {
  let en = 0;
}

//===----------------------------------------------------------------------===//
// VNBR Pseudo Instructions
//===----------------------------------------------------------------------===//

class SemaOperand<string prefix, bit alwaysPrint = 0>
    : NamedIntOperand<prefix> {
  let Validator = "[](int64_t V) { return -4 <= V && V <= 7; }";
  let PrintMethod = "[this](const MCInst *MI, unsigned OpNo, "#
    "const MCSubtargetInfo &STI, raw_ostream &O) { "#
    "printSema(MI, OpNo, O, \""#prefix#"\", "#
    !if(alwaysPrint, "true", "false")#"); }";
}

def SemaId : SemaOperand<"sema_id">;
def SemaWaveId : SemaOperand<"sema_wave_id", 1>;
def SemaIdRefl : SemaOperand<"sema_id_refl">;
def SemaWaveIdRefl : SemaOperand<"sema_wave_id_refl", 1>;

class VNBR_Pseudo<string opName, dag outs, dag regIns, string asmRegOps> :
    InstSI<outs, !con(regIns, (ins SemaId:$sema_id, SemaWaveId:$sema_wave_id,
                                   SemaIdRefl:$sema_id_refl,
                                   SemaWaveIdRefl:$sema_wave_id_refl,
                                   WaitVAVDst:$wait_va_dst))>,
    SIMCInstr<NAME, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  string Mnemonic = !tolower(opName);
  string AsmOperands = asmRegOps # "$sema_id$sema_wave_id$"
                       "sema_id_refl$sema_wave_id_refl$wait_va_dst";

  let SubtargetPredicate = HasVNBREncoding;
  let WaveSizePredicate = isWave32;
}

multiclass VNBR_Pseudopair {
  def NAME : VNBR_Pseudo<NAME, (outs GVGPR_32:$vdst, GVGPR_32:$vdst_refl),
                         (ins GVGPR_32:$vsrc), "$vdst, $vdst_refl, $vsrc">;
  def _no_write : VNBR_Pseudo<NAME, (outs), (ins), " off, off, off">;
}

defm V_SEND_VGPR_NEXT_B32 : VNBR_Pseudopair;
defm V_SEND_VGPR_PREV_B32 : VNBR_Pseudopair;

//===----------------------------------------------------------------------===//
// VNBR Real Instructions
//===----------------------------------------------------------------------===//

multiclass VNBR_Real_Base<GFXGen gen, VNBR_Pseudo ps> {
  def gen.Suffix : InstSI<ps.OutOperandList, ps.InOperandList,
                          ps.Mnemonic#ps.AsmOperands> {
    let SubtargetPredicate = ps.SubtargetPredicate;
    let WaveSizePredicate = ps.WaveSizePredicate;

    let DecoderNamespace = gen.DecoderNamespace;
  }
}

multiclass VNBR_Realpair<GFXGen gen, bits<4> op> {
  defvar ps = !cast<VNBR_Pseudo>(NAME);
  defm NAME : VNBR_Real_Base<gen, ps>, VNBRe<op>;

  defvar noWritePs = !cast<VNBR_Pseudo>(NAME#"_no_write");
  defm _no_write : VNBR_Real_Base<gen, noWritePs>, VNBRe_no_write<op>;
}

defm V_SEND_VGPR_NEXT_B32 : VNBR_Realpair<GFX13Gen, 0x0>;
defm V_SEND_VGPR_PREV_B32 : VNBR_Realpair<GFX13Gen, 0x1>;
