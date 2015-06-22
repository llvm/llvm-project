//===-- SIInstrInfo.td - SI Instruction Infos -------------*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class vop {
  field bits<9> SI3;
  field bits<10> VI3;
}

class vopc <bits<8> si, bits<8> vi = !add(0x40, si)> : vop {
  field bits<8> SI = si;
  field bits<8> VI = vi;

  field bits<9>  SI3 = {0, si{7-0}};
  field bits<10> VI3 = {0, 0, vi{7-0}};
}

class vop1 <bits<8> si, bits<8> vi = si> : vop {
  field bits<8> SI = si;
  field bits<8> VI = vi;

  field bits<9>  SI3 = {1, 1, si{6-0}};
  field bits<10> VI3 = !add(0x140, vi);
}

class vop2 <bits<6> si, bits<6> vi = si> : vop {
  field bits<6> SI = si;
  field bits<6> VI = vi;

  field bits<9>  SI3 = {1, 0, 0, si{5-0}};
  field bits<10> VI3 = {0, 1, 0, 0, vi{5-0}};
}

// Specify a VOP2 opcode for SI and VOP3 opcode for VI
// that doesn't have VOP2 encoding on VI
class vop23 <bits<6> si, bits<10> vi> : vop2 <si> {
  let VI3 = vi;
}

class vop3 <bits<9> si, bits<10> vi = {0, si}> : vop {
  let SI3 = si;
  let VI3 = vi;
}

class sop1 <bits<8> si, bits<8> vi = si> {
  field bits<8> SI = si;
  field bits<8> VI = vi;
}

class sop2 <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class sopk <bits<5> si, bits<5> vi = si> {
  field bits<5> SI = si;
  field bits<5> VI = vi;
}

// Execpt for the NONE field, this must be kept in sync with the SISubtarget enum
// in AMDGPUInstrInfo.cpp
def SISubtarget {
  int NONE = -1;
  int SI = 0;
  int VI = 1;
}

//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

def SIload_constant : SDNode<"AMDGPUISD::LOAD_CONSTANT",
  SDTypeProfile<1, 2, [SDTCisVT<0, f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i32>]>,
                      [SDNPMayLoad, SDNPMemOperand]
>;

def SItbuffer_store : SDNode<"AMDGPUISD::TBUFFER_STORE_FORMAT",
  SDTypeProfile<0, 13,
    [SDTCisVT<0, v4i32>,   // rsrc(SGPR)
     SDTCisVT<1, iAny>,   // vdata(VGPR)
     SDTCisVT<2, i32>,    // num_channels(imm)
     SDTCisVT<3, i32>,    // vaddr(VGPR)
     SDTCisVT<4, i32>,    // soffset(SGPR)
     SDTCisVT<5, i32>,    // inst_offset(imm)
     SDTCisVT<6, i32>,    // dfmt(imm)
     SDTCisVT<7, i32>,    // nfmt(imm)
     SDTCisVT<8, i32>,    // offen(imm)
     SDTCisVT<9, i32>,    // idxen(imm)
     SDTCisVT<10, i32>,   // glc(imm)
     SDTCisVT<11, i32>,   // slc(imm)
     SDTCisVT<12, i32>    // tfe(imm)
    ]>,
  [SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SIload_input : SDNode<"AMDGPUISD::LOAD_INPUT",
  SDTypeProfile<1, 3, [SDTCisVT<0, v4f32>, SDTCisVT<1, v4i32>, SDTCisVT<2, i16>,
                       SDTCisVT<3, i32>]>
>;

class SDSample<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 4, [SDTCisVT<0, v4f32>, SDTCisVT<2, v32i8>,
                       SDTCisVT<3, v4i32>, SDTCisVT<4, i32>]>
>;

def SIsample : SDSample<"AMDGPUISD::SAMPLE">;
def SIsampleb : SDSample<"AMDGPUISD::SAMPLEB">;
def SIsampled : SDSample<"AMDGPUISD::SAMPLED">;
def SIsamplel : SDSample<"AMDGPUISD::SAMPLEL">;

def SIconstdata_ptr : SDNode<
  "AMDGPUISD::CONST_DATA_PTR", SDTypeProfile <1, 0, [SDTCisVT<0, i64>]>
>;

// Transformation function, extract the lower 32bit of a 64bit immediate
def LO32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & 0xffffffff, MVT::i32);
}]>;

def LO32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), MVT::f32);
}]>;

// Transformation function, extract the upper 32bit of a 64bit immediate
def HI32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 32, MVT::i32);
}]>;

def HI32f : SDNodeXForm<fpimm, [{
  APInt V = N->getValueAPF().bitcastToAPInt().lshr(32).trunc(32);
  return CurDAG->getTargetConstantFP(APFloat(APFloat::IEEEsingle, V), MVT::f32);
}]>;

def IMM8bitDWORD : PatLeaf <(imm),
  [{return (N->getZExtValue() & ~0x3FC) == 0;}]
>;

def as_dword_i32imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() >> 2, MVT::i32);
}]>;

def as_i1imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), MVT::i1);
}]>;

def as_i8imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), MVT::i8);
}]>;

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i16);
}]>;

def as_i32imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i32);
}]>;

def as_i64imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), MVT::i64);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i32 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i64 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), MVT::i64);
}]>;

def IMM8bit : PatLeaf <(imm),
  [{return isUInt<8>(N->getZExtValue());}]
>;

def IMM12bit : PatLeaf <(imm),
  [{return isUInt<12>(N->getZExtValue());}]
>;

def IMM16bit : PatLeaf <(imm),
  [{return isUInt<16>(N->getZExtValue());}]
>;

def IMM20bit : PatLeaf <(imm),
  [{return isUInt<20>(N->getZExtValue());}]
>;

def IMM32bit : PatLeaf <(imm),
  [{return isUInt<32>(N->getZExtValue());}]
>;

def mubuf_vaddr_offset : PatFrag<
  (ops node:$ptr, node:$offset, node:$imm_offset),
  (add (add node:$ptr, node:$offset), node:$imm_offset)
>;

class InlineImm <ValueType vt> : PatLeaf <(vt imm), [{
  return isInlineImmediate(N);
}]>;

class InlineFPImm <ValueType vt> : PatLeaf <(vt fpimm), [{
  return isInlineImmediate(N);
}]>;

class SGPRImm <dag frag> : PatLeaf<frag, [{
  if (TM.getSubtarget<AMDGPUSubtarget>().getGeneration() <
      AMDGPUSubtarget::SOUTHERN_ISLANDS) {
    return false;
  }
  const SIRegisterInfo *SIRI =
                       static_cast<const SIRegisterInfo*>(TM.getSubtargetImpl()->getRegisterInfo());
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
                                                U != E; ++U) {
    if (SIRI->isSGPRClass(getOperandRegClass(*U, U.getOperandNo()))) {
      return true;
    }
  }
  return false;
}]>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def FRAMEri32 : Operand<iPTR> {
  let MIOperandInfo = (ops i32:$ptr, i32imm:$index);
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let OperandType = "OPERAND_PCREL";
}

include "SIInstrFormats.td"
include "VIInstrFormats.td"

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : Operand<i1> {
  let PrintMethod = "printOffen";
}
def idxen : Operand<i1> {
  let PrintMethod = "printIdxen";
}
def addr64 : Operand<i1> {
  let PrintMethod = "printAddr64";
}
def mbuf_offset : Operand<i16> {
  let PrintMethod = "printMBUFOffset";
}
def ds_offset : Operand<i16> {
  let PrintMethod = "printDSOffset";
}
def ds_offset0 : Operand<i8> {
  let PrintMethod = "printDSOffset0";
}
def ds_offset1 : Operand<i8> {
  let PrintMethod = "printDSOffset1";
}
def glc : Operand <i1> {
  let PrintMethod = "printGLC";
}
def slc : Operand <i1> {
  let PrintMethod = "printSLC";
}
def tfe : Operand <i1> {
  let PrintMethod = "printTFE";
}

def omod : Operand <i32> {
  let PrintMethod = "printOModSI";
}

def ClampMod : Operand <i1> {
  let PrintMethod = "printClampSI";
}

} // End OperandType = "OPERAND_IMMEDIATE"

//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;

def MUBUFAddr32 : ComplexPattern<i64, 9, "SelectMUBUFAddr32">;
def MUBUFAddr64 : ComplexPattern<i64, 3, "SelectMUBUFAddr64">;
def MUBUFAddr64Atomic : ComplexPattern<i64, 4, "SelectMUBUFAddr64">;
def MUBUFScratch : ComplexPattern<i64, 4, "SelectMUBUFScratch">;
def MUBUFOffset : ComplexPattern<i64, 6, "SelectMUBUFOffset">;
def MUBUFOffsetAtomic : ComplexPattern<i64, 4, "SelectMUBUFOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3Mods0Clamp : ComplexPattern<untyped, 3, "SelectVOP3Mods0Clamp">;
def VOP3Mods0Clamp0OMod : ComplexPattern<untyped, 4, "SelectVOP3Mods0Clamp0OMod">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;

//===----------------------------------------------------------------------===//
// SI assembler operands
//===----------------------------------------------------------------------===//

def SIOperand {
  int ZERO = 0x80;
  int VCC = 0x6A;
  int FLAT_SCR = 0x68;
}

def SRCMODS {
  int NONE = 0;
}

def DSTCLAMP {
  int NONE = 0;
}

def DSTOMOD {
  int NONE = 0;
}

//===----------------------------------------------------------------------===//
//
// SI Instruction multiclass helpers.
//
// Instructions with _32 take 32-bit operands.
// Instructions with _64 take 64-bit operands.
//
// VOP_* instructions can use either a 32-bit or 64-bit encoding.  The 32-bit
// encoding is the standard encoding, but instruction that make use of
// any of the instruction modifiers must use the 64-bit encoding.
//
// Instructions with _e32 use the 32-bit encoding.
// Instructions with _e64 use the 64-bit encoding.
//
//===----------------------------------------------------------------------===//

class SIMCInstr <string pseudo, int subtarget> {
  string PseudoInstr = pseudo;
  int Subtarget = subtarget;
}

//===----------------------------------------------------------------------===//
// EXP classes
//===----------------------------------------------------------------------===//

class EXPCommon : InstSI<
  (outs),
  (ins i32imm:$en, i32imm:$tgt, i32imm:$compr, i32imm:$done, i32imm:$vm,
       VGPR_32:$src0, VGPR_32:$src1, VGPR_32:$src2, VGPR_32:$src3),
  "exp $en, $tgt, $compr, $done, $vm, $src0, $src1, $src2, $src3",
  [] > {

  let EXP_CNT = 1;
  let Uses = [EXEC];
}

multiclass EXP_m {

  let isPseudo = 1 in {
    def "" : EXPCommon, SIMCInstr <"exp", SISubtarget.NONE> ;
  }

  def _si : EXPCommon, SIMCInstr <"exp", SISubtarget.SI>, EXPe;

  def _vi : EXPCommon, SIMCInstr <"exp", SISubtarget.VI>, EXPe_vi;
}

//===----------------------------------------------------------------------===//
// Scalar classes
//===----------------------------------------------------------------------===//

class SOP1_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOP1 <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class SOP1_Real_si <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.SI>,
  SIMCInstr<opName, SISubtarget.SI>;

class SOP1_Real_vi <sop1 op, string opName, dag outs, dag ins, string asm> :
  SOP1 <outs, ins, asm, []>,
  SOP1e <op.VI>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass SOP1_32 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_32:$dst), (ins SSrc_32:$src0),
    pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_32:$dst), (ins SSrc_32:$src0),
    opName#" $dst, $src0">;

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_32:$dst), (ins SSrc_32:$src0),
    opName#" $dst, $src0">;
}

multiclass SOP1_64 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_64:$dst), (ins SSrc_64:$src0),
    pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_64:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0">;

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_64:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0">;
}

// no input, 64-bit output.
multiclass SOP1_64_0 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_64:$dst), (ins), pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_64:$dst), (ins),
    opName#" $dst"> {
    let SSRC0 = 0;
  }

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_64:$dst), (ins),
    opName#" $dst"> {
    let SSRC0 = 0;
  }
}

// 64-bit input, 32-bit output.
multiclass SOP1_32_64 <sop1 op, string opName, list<dag> pattern> {
  def "" : SOP1_Pseudo <opName, (outs SReg_32:$dst), (ins SSrc_64:$src0),
    pattern>;

  def _si : SOP1_Real_si <op, opName, (outs SReg_32:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0">;

  def _vi : SOP1_Real_vi <op, opName, (outs SReg_32:$dst), (ins SSrc_64:$src0),
    opName#" $dst, $src0">;
}

class SOP2_Pseudo<string opName, dag outs, dag ins, list<dag> pattern> :
  SOP2<outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
  let Size = 4;
}

class SOP2_Real_si<sop2 op, string opName, dag outs, dag ins, string asm> :
  SOP2<outs, ins, asm, []>,
  SOP2e<op.SI>,
  SIMCInstr<opName, SISubtarget.SI>;

class SOP2_Real_vi<sop2 op, string opName, dag outs, dag ins, string asm> :
  SOP2<outs, ins, asm, []>,
  SOP2e<op.VI>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass SOP2_SELECT_32 <sop2 op, string opName, list<dag> pattern> {
  def "" : SOP2_Pseudo <opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1, SCCReg:$scc), pattern>;

  def _si : SOP2_Real_si <op, opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1, SCCReg:$scc),
    opName#" $dst, $src0, $src1 [$scc]">;

  def _vi : SOP2_Real_vi <op, opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1, SCCReg:$scc),
    opName#" $dst, $src0, $src1 [$scc]">;
}

multiclass SOP2_32 <sop2 op, string opName, list<dag> pattern> {
  def "" : SOP2_Pseudo <opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1), pattern>;

  def _si : SOP2_Real_si <op, opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1), opName#" $dst, $src0, $src1">;

  def _vi : SOP2_Real_vi <op, opName, (outs SReg_32:$dst),
    (ins SSrc_32:$src0, SSrc_32:$src1), opName#" $dst, $src0, $src1">;
}

multiclass SOP2_64 <sop2 op, string opName, list<dag> pattern> {
  def "" : SOP2_Pseudo <opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_64:$src1), pattern>;

  def _si : SOP2_Real_si <op, opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_64:$src1), opName#" $dst, $src0, $src1">;

  def _vi : SOP2_Real_vi <op, opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_64:$src1), opName#" $dst, $src0, $src1">;
}

multiclass SOP2_64_32 <sop2 op, string opName, list<dag> pattern> {
  def "" : SOP2_Pseudo <opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_32:$src1), pattern>;

  def _si : SOP2_Real_si <op, opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_32:$src1), opName#" $dst, $src0, $src1">;

  def _vi : SOP2_Real_vi <op, opName, (outs SReg_64:$dst),
    (ins SSrc_64:$src0, SSrc_32:$src1), opName#" $dst, $src0, $src1">;
}


class SOPC_Helper <bits<7> op, RegisterOperand rc, ValueType vt,
                    string opName, PatLeaf cond> : SOPC <
  op, (outs SCCReg:$dst), (ins rc:$src0, rc:$src1),
  opName#" $dst, $src0, $src1", []>;

class SOPC_32<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_32, i32, opName, cond>;

class SOPC_64<bits<7> op, string opName, PatLeaf cond = COND_NULL>
  : SOPC_Helper<op, SSrc_64, i64, opName, cond>;

class SOPK_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SOPK <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class SOPK_Real_si <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.SI>,
  SIMCInstr<opName, SISubtarget.SI>;

class SOPK_Real_vi <sopk op, string opName, dag outs, dag ins, string asm> :
  SOPK <outs, ins, asm, []>,
  SOPKe <op.VI>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass SOPK_32 <sopk op, string opName, list<dag> pattern> {
  def "" : SOPK_Pseudo <opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    pattern>;

  def _si : SOPK_Real_si <op, opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    opName#" $dst, $src0">;

  def _vi : SOPK_Real_vi <op, opName, (outs SReg_32:$dst), (ins u16imm:$src0),
    opName#" $dst, $src0">;
}

multiclass SOPK_SCC <sopk op, string opName, list<dag> pattern> {
  def "" : SOPK_Pseudo <opName, (outs SCCReg:$dst),
    (ins SReg_32:$src0, u16imm:$src1), pattern>;

  def _si : SOPK_Real_si <op, opName, (outs SCCReg:$dst),
    (ins SReg_32:$src0, u16imm:$src1), opName#" $dst, $src0">;

  def _vi : SOPK_Real_vi <op, opName, (outs SCCReg:$dst),
    (ins SReg_32:$src0, u16imm:$src1), opName#" $dst, $src0">;
}

//===----------------------------------------------------------------------===//
// SMRD classes
//===----------------------------------------------------------------------===//

class SMRD_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  SMRD <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class SMRD_Real_si <bits<5> op, string opName, bit imm, dag outs, dag ins,
                    string asm> :
  SMRD <outs, ins, asm, []>,
  SMRDe <op, imm>,
  SIMCInstr<opName, SISubtarget.SI>;

class SMRD_Real_vi <bits<8> op, string opName, bit imm, dag outs, dag ins,
                    string asm> :
  SMRD <outs, ins, asm, []>,
  SMEMe_vi <op, imm>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass SMRD_m <bits<5> op, string opName, bit imm, dag outs, dag ins,
                   string asm, list<dag> pattern> {

  def "" : SMRD_Pseudo <opName, outs, ins, pattern>;

  def _si : SMRD_Real_si <op, opName, imm, outs, ins, asm>;

  def _vi : SMRD_Real_vi <{0, 0, 0, op}, opName, imm, outs, ins, asm>;
}

multiclass SMRD_Helper <bits<5> op, string opName, RegisterClass baseClass,
                        RegisterClass dstClass> {
  defm _IMM : SMRD_m <
    op, opName#"_IMM", 1, (outs dstClass:$dst),
    (ins baseClass:$sbase, u32imm:$offset),
    opName#" $dst, $sbase, $offset", []
  >;

  defm _SGPR : SMRD_m <
    op, opName#"_SGPR", 0, (outs dstClass:$dst),
    (ins baseClass:$sbase, SReg_32:$soff),
    opName#" $dst, $sbase, $soff", []
  >;
}

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

// This must always be right before the operand being input modified.
def InputMods : OperandWithDefaultOps <i32, (ops (i32 0))> {
  let PrintMethod = "printOperandAndMods";
}
def InputModsNoDefault : Operand <i32> {
  let PrintMethod = "printOperandAndMods";
}

class getNumSrcArgs<ValueType Src1, ValueType Src2> {
  int ret =
    !if (!eq(Src1.Value, untyped.Value),      1,   // VOP1
         !if (!eq(Src2.Value, untyped.Value), 2,   // VOP2
                                              3)); // VOP3
}

// Returns the register class to use for the destination of VOP[123C]
// instructions for the given VT.
class getVALUDstForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VGPR_32,
                          !if(!eq(VT.Size, 64), VReg_64,
                            SReg_64)); // else VT == i1
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 32), VSrc_32, VSrc_64);
}

// Returns the register class to use for source 1 of VOP[12C] for the
// given VT.
class getVOPSrc1ForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 32), VGPR_32, VReg_64);
}

// Returns the register classes for the source arguments of a VOP[12C]
// instruction for the given SrcVTs.
class getInRC32 <list<ValueType> SrcVT> {
  list<DAGOperand> ret = [
    getVOPSrc0ForVT<SrcVT[0]>.ret,
    getVOPSrc1ForVT<SrcVT[1]>.ret
  ];
}

// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 32), VCSrc_32, VCSrc_64);
}

// Returns the register classes for the source arguments of a VOP3
// instruction for the given SrcVTs.
class getInRC64 <list<ValueType> SrcVT> {
  list<DAGOperand> ret = [
    getVOP3SrcForVT<SrcVT[0]>.ret,
    getVOP3SrcForVT<SrcVT[1]>.ret,
    getVOP3SrcForVT<SrcVT[2]>.ret
  ];
}

// Returns 1 if the source arguments have modifiers, 0 if they do not.
class hasModifiers<ValueType SrcVT> {
  bit ret = !if(!eq(SrcVT.Value, f32.Value), 1,
            !if(!eq(SrcVT.Value, f64.Value), 1, 0));
}

// Returns the input arguments for VOP[12C] instructions for the given SrcVT.
class getIns32 <RegisterOperand Src0RC, RegisterClass Src1RC, int NumSrcArgs> {
  dag ret = !if(!eq(NumSrcArgs, 1), (ins Src0RC:$src0),               // VOP1
            !if(!eq(NumSrcArgs, 2), (ins Src0RC:$src0, Src1RC:$src1), // VOP2
                                    (ins)));
}

// Returns the input arguments for VOP3 instructions for the given SrcVT.
class getIns64 <RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasModifiers> {

  dag ret =
    !if (!eq(NumSrcArgs, 1),
      !if (!eq(HasModifiers, 1),
        // VOP1 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP1 without modifiers
        (ins Src0RC:$src0)
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (!eq(HasModifiers, 1),
        // VOP 2 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP2 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1)
      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (!eq(HasModifiers, 1),
        // VOP3 with modifiers
        (ins InputModsNoDefault:$src0_modifiers, Src0RC:$src0,
             InputModsNoDefault:$src1_modifiers, Src1RC:$src1,
             InputModsNoDefault:$src2_modifiers, Src2RC:$src2,
             ClampMod:$clamp, omod:$omod)
      /* else */,
        // VOP3 without modifiers
        (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2)
      /* endif */ )));
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <int NumSrcArgs> {
  string src1 = ", $src1";
  string src2 = ", $src2";
  string ret = " $dst, $src0"#
               !if(!eq(NumSrcArgs, 1), "", src1)#
               !if(!eq(NumSrcArgs, 3), src2, "");
}

// Returns the assembly string for the inputs and outputs of a VOP3
// instruction.
class getAsm64 <int NumSrcArgs, bit HasModifiers> {
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string ret =
  !if(!eq(HasModifiers, 0),
      getAsm32<NumSrcArgs>.ret,
      " $dst, "#src0#src1#src2#"$clamp"#"$omod");
}


class VOPProfile <list<ValueType> _ArgVT> {

  field list<ValueType> ArgVT = _ArgVT;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterClass DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVOPSrc1ForVT<Src1VT>.ret;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;

  field int NumSrcArgs = getNumSrcArgs<Src1VT, Src2VT>.ret;
  field bit HasModifiers = hasModifiers<Src0VT>.ret;

  field dag Outs = (outs DstRC:$dst);

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasModifiers>.ret;

  field string Asm32 = "_e32"#getAsm32<NumSrcArgs>.ret;
  field string Asm64 = getAsm64<NumSrcArgs, HasModifiers>.ret;
}

def VOP_F32_F32 : VOPProfile <[f32, f32, untyped, untyped]>;
def VOP_F32_F64 : VOPProfile <[f32, f64, untyped, untyped]>;
def VOP_F32_I32 : VOPProfile <[f32, i32, untyped, untyped]>;
def VOP_F64_F32 : VOPProfile <[f64, f32, untyped, untyped]>;
def VOP_F64_F64 : VOPProfile <[f64, f64, untyped, untyped]>;
def VOP_F64_I32 : VOPProfile <[f64, i32, untyped, untyped]>;
def VOP_I32_F32 : VOPProfile <[i32, f32, untyped, untyped]>;
def VOP_I32_F64 : VOPProfile <[i32, f64, untyped, untyped]>;
def VOP_I32_I32 : VOPProfile <[i32, i32, untyped, untyped]>;

def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_F32_I32 : VOPProfile <[i32, f32, i32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;
def VOP_I32_I32_I32_VCC : VOPProfile <[i32, i32, i32, untyped]> {
  let Src0RC32 = VCSrc_32;
}

def VOP_I1_F32_I32 : VOPProfile <[i1, f32, i32, untyped]> {
  let Ins64 = (ins InputModsNoDefault:$src0_modifiers, Src0RC64:$src0, Src1RC64:$src1);
  let Asm64 = " $dst, $src0_modifiers, $src1";
}

def VOP_I1_F64_I32 : VOPProfile <[i1, f64, i32, untyped]> {
  let Ins64 = (ins InputModsNoDefault:$src0_modifiers, Src0RC64:$src0, Src1RC64:$src1);
  let Asm64 = " $dst, $src0_modifiers, $src1";
}

def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I32_I64 : VOPProfile <[i64, i32, i64, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_MADK : VOPProfile <[f32, f32, f32, f32]> {
  field dag Ins = (ins VCSrc_32:$src0, VGPR_32:$vsrc1, u32imm:$src2);
  field string Asm = " $dst, $src0, $vsrc1, $src2";
}
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;


class VOP <string opName> {
  string OpName = opName;
}

class VOP2_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}

class VOP1_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP1Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr <opName#"_e32", SISubtarget.NONE> {
  let isPseudo = 1;
}

multiclass VOP1_m <vop1 op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName> {
  def "" : VOP1_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP1<op.SI, outs, ins, asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>;
  def _vi : VOP1<op.VI, outs, ins, asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.VI>;
}

multiclass VOP1SI_m <vop1 op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName> {
  def "" : VOP1_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP1<op.SI, outs, ins, asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>;
  // No VI instruction. This class is for SI only.
}

class VOP2_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP2Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SISubtarget.NONE> {
  let isPseudo = 1;
}

multiclass VOP2SI_m <vop2 op, dag outs, dag ins, string asm, list<dag> pattern,
                     string opName, string revOp> {
  def "" : VOP2_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _si : VOP2 <op.SI, outs, ins, opName#asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>;
}

multiclass VOP2_m <vop2 op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, string revOp> {
  def "" : VOP2_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e32", !eq(revOp, opName)>;

  def _si : VOP2 <op.SI, outs, ins, opName#asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>;
  def _vi : VOP2 <op.VI, outs, ins, opName#asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.VI>;
}

class VOP3DisableFields <bit HasSrc1, bit HasSrc2, bit HasModifiers> {

  bits<2> src0_modifiers = !if(HasModifiers, ?, 0);
  bits<2> src1_modifiers = !if(HasModifiers, !if(HasSrc1, ?, 0), 0);
  bits<2> src2_modifiers = !if(HasModifiers, !if(HasSrc2, ? ,0) ,0);
  bits<2> omod = !if(HasModifiers, ?, 0);
  bits<1> clamp = !if(HasModifiers, ?, 0);
  bits<9> src1 = !if(HasSrc1, ?, 0);
  bits<9> src2 = !if(HasSrc2, ?, 0);
}

class VOP3_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOP3Common <outs, ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e64", SISubtarget.NONE> {
  let isPseudo = 1;
}

class VOP3_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3e <op>,
  SIMCInstr<opName#"_e64", SISubtarget.SI>;

class VOP3_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3e_vi <op>,
  SIMCInstr <opName#"_e64", SISubtarget.VI>;

class VOP3b_Real_si <bits<9> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3be <op>,
  SIMCInstr<opName#"_e64", SISubtarget.SI>;

class VOP3b_Real_vi <bits<10> op, dag outs, dag ins, string asm, string opName> :
  VOP3Common <outs, ins, asm, []>,
  VOP3be_vi <op>,
  SIMCInstr <opName#"_e64", SISubtarget.VI>;

multiclass VOP3_m <vop op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, int NumSrcArgs, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<!if(!eq(NumSrcArgs, 1), 0, 1),
                              !if(!eq(NumSrcArgs, 2), 0, 1),
                              HasMods>;
}

// VOP3_m without source modifiers
multiclass VOP3_m_nosrcmod <vop op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, int NumSrcArgs, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  let src0_modifiers = 0,
      src1_modifiers = 0,
      src2_modifiers = 0 in {
    def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>;
    def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>;
  }
}

multiclass VOP3_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;
}

multiclass VOP3SI_1_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, bit HasMods = 1> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<0, 0, HasMods>;
  // No VI instruction. This class is for SI only.
}

multiclass VOP3_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1, bit UseFullOp = 0> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;
}

multiclass VOP3SI_2_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName, string revOp,
                     bit HasMods = 1, bit UseFullOp = 0> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods>;

  // No VI instruction. This class is for SI only.
}

// XXX - Is v_div_scale_{f32|f64} only available in vop3b without
// option of implicit vcc use?
multiclass VOP3b_2_m <vop op, dag outs, dag ins, string asm,
                      list<dag> pattern, string opName, string revOp,
                      bit HasMods = 1, bit UseFullOp = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName>,
           VOP2_REV<revOp#"_e64", !eq(revOp, opName)>;

  // The VOP2 variant puts the carry out into VCC, the VOP3 variant
  // can write it into any SGPR. We currently don't use the carry out,
  // so for now hardcode it to VCC as well.
  let sdst = SIOperand.VCC, Defs = [VCC] in {
    def _si : VOP3b_Real_si <op.SI3, outs, ins, asm, opName>,
              VOP3DisableFields<1, 0, HasMods>;

    def _vi : VOP3b_Real_vi <op.VI3, outs, ins, asm, opName>,
              VOP3DisableFields<1, 0, HasMods>;
  } // End sdst = SIOperand.VCC, Defs = [VCC]
}

multiclass VOP3b_3_m <vop op, dag outs, dag ins, string asm,
                      list<dag> pattern, string opName, string revOp,
                      bit HasMods = 1, bit UseFullOp = 0> {
  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;


  def _si : VOP3b_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 1, HasMods>;

  def _vi : VOP3b_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 1, HasMods>;
}

multiclass VOP3_C_m <vop op, dag outs, dag ins, string asm,
                     list<dag> pattern, string opName,
                     bit HasMods, bit defExec> {

  def "" : VOP3_Pseudo <outs, ins, pattern, opName>;

  def _si : VOP3_Real_si <op.SI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
  }

  def _vi : VOP3_Real_vi <op.VI3, outs, ins, asm, opName>,
            VOP3DisableFields<1, 0, HasMods> {
    let Defs = !if(defExec, [EXEC], []);
  }
}

// An instruction that is VOP2 on SI and VOP3 on VI, no modifiers.
multiclass VOP2SI_3VI_m <vop3 op, string opName, dag outs, dag ins,
                         string asm, list<dag> pattern = []> {
  let isPseudo = 1 in {
    def "" : VOPAnyCommon <outs, ins, "", pattern>,
             SIMCInstr<opName, SISubtarget.NONE>;
  }

  def _si : VOP2 <op.SI3{5-0}, outs, ins, asm, []>,
            SIMCInstr <opName, SISubtarget.SI>;

  def _vi : VOP3Common <outs, ins, asm, []>,
            VOP3e_vi <op.VI3>,
            VOP3DisableFields <1, 0, 0>,
            SIMCInstr <opName, SISubtarget.VI>;
}

multiclass VOP1_Helper <vop1 op, string opName, dag outs,
                        dag ins32, string asm32, list<dag> pat32,
                        dag ins64, string asm64, list<dag> pat64,
                        bit HasMods> {

  defm _e32 : VOP1_m <op, outs, ins32, opName#asm32, pat32, opName>;

  defm _e64 : VOP3_1_m <op, outs, ins64, opName#"_e64"#asm64, pat64, opName, HasMods>;
}

multiclass VOP1Inst <vop1 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP1_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0))]),
  P.HasModifiers
>;

multiclass VOP1InstSI <vop1 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag> {

  defm _e32 : VOP1SI_m <op, P.Outs, P.Ins32, opName#P.Asm32, [], opName>;

  defm _e64 : VOP3SI_1_m <op, P.Outs, P.Ins64, opName#P.Asm64,
    !if(P.HasModifiers,
      [(set P.DstVT:$dst, (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0,
                                i32:$src0_modifiers, i1:$clamp, i32:$omod))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0))]),
    opName, P.HasModifiers>;
}

multiclass VOP2_Helper <vop2 op, string opName, dag outs,
                        dag ins32, string asm32, list<dag> pat32,
                        dag ins64, string asm64, list<dag> pat64,
                        string revOp, bit HasMods> {
  defm _e32 : VOP2_m <op, outs, ins32, asm32, pat32, opName, revOp>;

  defm _e64 : VOP3_2_m <op,
    outs, ins64, opName#"_e64"#asm64, pat64, opName, revOp, HasMods
  >;
}

multiclass VOP2Inst <vop2 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag,
                     string revOp = opName> : VOP2_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, P.HasModifiers
>;

multiclass VOP2InstSI <vop2 op, string opName, VOPProfile P,
                       SDPatternOperator node = null_frag,
                       string revOp = opName> {
  defm _e32 : VOP2SI_m <op, P.Outs, P.Ins32, P.Asm32, [], opName, revOp>;

  defm _e64 : VOP3SI_2_m <op, P.Outs, P.Ins64, opName#"_e64"#P.Asm64,
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
             (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                        i1:$clamp, i32:$omod)),
                   (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
    opName, revOp, P.HasModifiers>;
}

multiclass VOP2b_Helper <vop2 op, string opName, dag outs,
                         dag ins32, string asm32, list<dag> pat32,
                         dag ins64, string asm64, list<dag> pat64,
                         string revOp, bit HasMods> {

  defm _e32 : VOP2_m <op, outs, ins32, asm32, pat32, opName, revOp>;

  defm _e64 : VOP3b_2_m <op,
    outs, ins64, opName#"_e64"#asm64, pat64, opName, revOp, HasMods
  >;
}

multiclass VOP2bInst <vop2 op, string opName, VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName> : VOP2b_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, P.HasModifiers
>;

// A VOP2 instruction that is VOP3-only on VI.
multiclass VOP2_VI3_Helper <vop23 op, string opName, dag outs,
                            dag ins32, string asm32, list<dag> pat32,
                            dag ins64, string asm64, list<dag> pat64,
                            string revOp, bit HasMods> {
  defm _e32 : VOP2SI_m <op, outs, ins32, asm32, pat32, opName, revOp>;

  defm _e64 : VOP3_2_m <op, outs, ins64, opName#"_e64"#asm64, pat64, opName,
                        revOp, HasMods>;
}

multiclass VOP2_VI3_Inst <vop23 op, string opName, VOPProfile P,
                          SDPatternOperator node = null_frag,
                          string revOp = opName>
                          : VOP2_VI3_Helper <
  op, opName, P.Outs,
  P.Ins32, P.Asm32, [],
  P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set P.DstVT:$dst,
           (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
      [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))]),
  revOp, P.HasModifiers
>;

multiclass VOP2MADK <vop2 op, string opName, list<dag> pattern = []> {

  def "" : VOP2_Pseudo <VOP_MADK.Outs, VOP_MADK.Ins, pattern, opName>;

let isCodeGenOnly = 0 in {
  def _si : VOP2Common <VOP_MADK.Outs, VOP_MADK.Ins,
                        !strconcat(opName, VOP_MADK.Asm), []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI>,
            VOP2_MADKe <op.SI>;

  def _vi : VOP2Common <VOP_MADK.Outs, VOP_MADK.Ins,
                        !strconcat(opName, VOP_MADK.Asm), []>,
            SIMCInstr <opName#"_e32", SISubtarget.VI>,
            VOP2_MADKe <op.VI>;
} // End isCodeGenOnly = 0
}

class VOPC_Pseudo <dag outs, dag ins, list<dag> pattern, string opName> :
  VOPCCommon <ins, "", pattern>,
  VOP <opName>,
  SIMCInstr<opName#"_e32", SISubtarget.NONE> {
  let isPseudo = 1;
}

multiclass VOPC_m <vopc op, dag outs, dag ins, string asm, list<dag> pattern,
                   string opName, bit DefExec> {
  def "" : VOPC_Pseudo <outs, ins, pattern, opName>;

  def _si : VOPC<op.SI, ins, asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.SI> {
    let Defs = !if(DefExec, [EXEC], []);
  }

  def _vi : VOPC<op.VI, ins, asm, []>,
            SIMCInstr <opName#"_e32", SISubtarget.VI> {
    let Defs = !if(DefExec, [EXEC], []);
  }
}

multiclass VOPC_Helper <vopc op, string opName,
                        dag ins32, string asm32, list<dag> pat32,
                        dag out64, dag ins64, string asm64, list<dag> pat64,
                        bit HasMods, bit DefExec> {
  defm _e32 : VOPC_m <op, (outs), ins32, opName#asm32, pat32, opName, DefExec>;

  defm _e64 : VOP3_C_m <op, out64, ins64, opName#"_e64"#asm64, pat64,
                        opName, HasMods, DefExec>;
}

multiclass VOPCInst <vopc op, string opName,
                     VOPProfile P, PatLeaf cond = COND_NULL,
                     bit DefExec = 0> : VOPC_Helper <
  op, opName,
  P.Ins32, P.Asm32, [],
  (outs SReg_64:$dst), P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set i1:$dst,
          (setcc (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                      i1:$clamp, i32:$omod)),
                 (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                 cond))],
      [(set i1:$dst, (setcc P.Src0VT:$src0, P.Src1VT:$src1, cond))]),
  P.HasModifiers, DefExec
>;

multiclass VOPCClassInst <vopc op, string opName, VOPProfile P,
                     bit DefExec = 0> : VOPC_Helper <
  op, opName,
  P.Ins32, P.Asm32, [],
  (outs SReg_64:$dst), P.Ins64, P.Asm64,
  !if(P.HasModifiers,
      [(set i1:$dst,
          (AMDGPUfp_class (P.Src0VT (VOP3Mods0Clamp0OMod P.Src0VT:$src0, i32:$src0_modifiers)), P.Src1VT:$src1))],
      [(set i1:$dst, (AMDGPUfp_class P.Src0VT:$src0, P.Src1VT:$src1))]),
  P.HasModifiers, DefExec
>;


multiclass VOPC_F32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_F32_F32_F32, cond>;

multiclass VOPC_F64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_F64_F64_F64, cond>;

multiclass VOPC_I32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_I32_I32_I32, cond>;

multiclass VOPC_I64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCInst <op, opName, VOP_I64_I64_I64, cond>;


multiclass VOPCX <vopc op, string opName, VOPProfile P,
                  PatLeaf cond = COND_NULL>
  : VOPCInst <op, opName, P, cond, 1>;

multiclass VOPCX_F32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_F32_F32_F32, cond>;

multiclass VOPCX_F64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_F64_F64_F64, cond>;

multiclass VOPCX_I32 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_I32_I32_I32, cond>;

multiclass VOPCX_I64 <vopc op, string opName, PatLeaf cond = COND_NULL> :
  VOPCX <op, opName, VOP_I64_I64_I64, cond>;

multiclass VOP3_Helper <vop3 op, string opName, dag outs, dag ins, string asm,
                        list<dag> pat, int NumSrcArgs, bit HasMods> : VOP3_m <
    op, outs, ins, opName#asm, pat, opName, NumSrcArgs, HasMods
>;

multiclass VOPC_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOP_I1_F32_I32, 0>;

multiclass VOPCX_CLASS_F32 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOP_I1_F32_I32, 1>;

multiclass VOPC_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOP_I1_F64_I32, 0>;

multiclass VOPCX_CLASS_F64 <vopc op, string opName> :
  VOPCClassInst <op, opName, VOP_I1_F64_I32, 1>;

multiclass VOP3Inst <vop3 op, string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName, P.Outs, P.Ins64, P.Asm64,
  !if(!eq(P.NumSrcArgs, 3),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1,
                                  P.Src2VT:$src2))]),
  !if(!eq(P.NumSrcArgs, 2),
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0, P.Src1VT:$src1))])
  /* P.NumSrcArgs == 1 */,
    !if(P.HasModifiers,
        [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod))))],
        [(set P.DstVT:$dst, (node P.Src0VT:$src0))]))),
  P.NumSrcArgs, P.HasModifiers
>;

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
multiclass VOP3_VCC_Inst <vop3 op, string opName,
                          VOPProfile P,
                          SDPatternOperator node = null_frag> : VOP3_Helper <
  op, opName,
  P.Outs,
  (ins InputModsNoDefault:$src0_modifiers, P.Src0RC64:$src0,
       InputModsNoDefault:$src1_modifiers, P.Src1RC64:$src1,
       InputModsNoDefault:$src2_modifiers, P.Src2RC64:$src2,
       ClampMod:$clamp,
       omod:$omod),
  " $dst, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod",
  [(set P.DstVT:$dst,
            (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers,
                                       i1:$clamp, i32:$omod)),
                  (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
                  (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers)),
                  (i1 VCC)))],
  3, 1
>;

multiclass VOP3b_Helper <vop op, RegisterClass vrc, RegisterOperand arc,
                    string opName, list<dag> pattern> :
  VOP3b_3_m <
  op, (outs vrc:$vdst, SReg_64:$sdst),
      (ins InputModsNoDefault:$src0_modifiers, arc:$src0,
           InputModsNoDefault:$src1_modifiers, arc:$src1,
           InputModsNoDefault:$src2_modifiers, arc:$src2,
           ClampMod:$clamp, omod:$omod),
  opName#" $vdst, $sdst, $src0_modifiers, $src1_modifiers, $src2_modifiers"#"$clamp"#"$omod", pattern,
  opName, opName, 1, 1
>;

multiclass VOP3b_64 <vop3 op, string opName, list<dag> pattern> :
  VOP3b_Helper <op, VReg_64, VSrc_64, opName, pattern>;

multiclass VOP3b_32 <vop3 op, string opName, list<dag> pattern> :
  VOP3b_Helper <op, VGPR_32, VSrc_32, opName, pattern>;


class Vop3ModPat<Instruction Inst, VOPProfile P, SDPatternOperator node> : Pat<
  (node (P.Src0VT (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod)),
        (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
        (P.Src2VT (VOP3Mods P.Src2VT:$src2, i32:$src2_modifiers))),
  (Inst i32:$src0_modifiers, P.Src0VT:$src0,
        i32:$src1_modifiers, P.Src1VT:$src1,
        i32:$src2_modifiers, P.Src2VT:$src2,
        i1:$clamp,
        i32:$omod)>;

//===----------------------------------------------------------------------===//
// Interpolation opcodes
//===----------------------------------------------------------------------===//

class VINTRP_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  VINTRPCommon <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class VINTRP_Real_si <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe <op>,
  SIMCInstr<opName, SISubtarget.SI>;

class VINTRP_Real_vi <bits <2> op, string opName, dag outs, dag ins,
                      string asm> :
  VINTRPCommon <outs, ins, asm, []>,
  VINTRPe_vi <op>,
  SIMCInstr<opName, SISubtarget.VI>;

multiclass VINTRP_m <bits <2> op, dag outs, dag ins, string asm,
                     string disableEncoding = "", string constraints = "",
                     list<dag> pattern = []> {
  let DisableEncoding = disableEncoding,
      Constraints = constraints in {
    def "" : VINTRP_Pseudo <NAME, outs, ins, pattern>;

    def _si : VINTRP_Real_si <op, NAME, outs, ins, asm>;

    def _vi : VINTRP_Real_vi <op, NAME, outs, ins, asm>;
  }
}

//===----------------------------------------------------------------------===//
// Vector I/O classes
//===----------------------------------------------------------------------===//

class DS_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  DS <outs, ins, "", pattern>,
  SIMCInstr <opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class DS_Real_si <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe <op>,
  SIMCInstr <opName, SISubtarget.SI>;

class DS_Real_vi <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe_vi <op>,
  SIMCInstr <opName, SISubtarget.VI>;

class DS_1A_Real_si <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe <op>,
  SIMCInstr <opName, SISubtarget.SI> {

  // Single load interpret the 2 i8imm operands as a single i16 offset.
  bits<16> offset;
  let offset0 = offset{7-0};
  let offset1 = offset{15-8};
}

class DS_1A_Real_vi <bits<8> op, string opName, dag outs, dag ins, string asm> :
  DS <outs, ins, asm, []>,
  DSe_vi <op>,
  SIMCInstr <opName, SISubtarget.VI> {

  // Single load interpret the 2 i8imm operands as a single i16 offset.
  bits<16> offset;
  let offset0 = offset{7-0};
  let offset1 = offset{15-8};
}

multiclass DS_1A_Load_m <bits<8> op, string opName, dag outs, dag ins, string asm,
                         list<dag> pat> {
  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>;

    let data0 = 0, data1 = 0 in {
      def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_Load_Helper <bits<8> op, string asm, RegisterClass regClass>
    : DS_1A_Load_m <
  op,
  asm,
  (outs regClass:$vdst),
  (ins i1imm:$gds, VGPR_32:$addr, ds_offset:$offset, M0Reg:$m0),
  asm#" $vdst, $addr"#"$offset"#" [M0]",
  []>;

multiclass DS_Load2_m <bits<8> op, string opName, dag outs, dag ins, string asm,
                       list<dag> pat> {
  let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>;

    let data0 = 0, data1 = 0 in {
      def _si : DS_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_Load2_Helper <bits<8> op, string asm, RegisterClass regClass>
    : DS_Load2_m <
  op,
  asm,
  (outs regClass:$vdst),
  (ins i1imm:$gds, VGPR_32:$addr, ds_offset0:$offset0, ds_offset1:$offset1,
        M0Reg:$m0),
  asm#" $vdst, $addr"#"$offset0"#"$offset1 [M0]",
  []>;

multiclass DS_1A_Store_m <bits<8> op, string opName, dag outs, dag ins,
                          string asm, list<dag> pat> {
  let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>;

    let data1 = 0, vdst = 0 in {
      def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_Store_Helper <bits<8> op, string asm, RegisterClass regClass>
    : DS_1A_Store_m <
  op,
  asm,
  (outs),
  (ins i1imm:$gds, VGPR_32:$addr, regClass:$data0, ds_offset:$offset, M0Reg:$m0),
  asm#" $addr, $data0"#"$offset"#" [M0]",
  []>;

multiclass DS_Store_m <bits<8> op, string opName, dag outs, dag ins,
                       string asm, list<dag> pat> {
  let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>;

    let vdst = 0 in {
      def _si : DS_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_Store2_Helper <bits<8> op, string asm, RegisterClass regClass>
    : DS_Store_m <
  op,
  asm,
  (outs),
  (ins i1imm:$gds, VGPR_32:$addr, regClass:$data0, regClass:$data1,
       ds_offset0:$offset0, ds_offset1:$offset1, M0Reg:$m0),
  asm#" $addr, $data0, $data1"#"$offset0"#"$offset1 [M0]",
  []>;

// 1 address, 1 data.
multiclass DS_1A1D_RET_m <bits<8> op, string opName, dag outs, dag ins,
                          string asm, list<dag> pat, string noRetOp> {
  let mayLoad = 1, mayStore = 1,
      hasPostISelHook = 1 // Adjusted to no return version.
      in {
    def "" : DS_Pseudo <opName, outs, ins, pat>,
             AtomicNoRet<noRetOp, 1>;

    let data1 = 0 in {
      def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_1A1D_RET <bits<8> op, string asm, RegisterClass rc,
                        string noRetOp = ""> : DS_1A1D_RET_m <
  op, asm,
  (outs rc:$vdst),
  (ins i1imm:$gds, VGPR_32:$addr, rc:$data0, ds_offset:$offset, M0Reg:$m0),
  asm#" $vdst, $addr, $data0"#"$offset"#" [M0]", [], noRetOp>;

// 1 address, 2 data.
multiclass DS_1A2D_RET_m <bits<8> op, string opName, dag outs, dag ins,
                          string asm, list<dag> pat, string noRetOp> {
  let mayLoad = 1, mayStore = 1,
      hasPostISelHook = 1 // Adjusted to no return version.
      in {
    def "" : DS_Pseudo <opName, outs, ins, pat>,
             AtomicNoRet<noRetOp, 1>;

    def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A2D_RET <bits<8> op, string asm, RegisterClass rc,
                   string noRetOp = ""> : DS_1A2D_RET_m <
  op, asm,
  (outs rc:$vdst),
  (ins i1imm:$gds, VGPR_32:$addr, rc:$data0, rc:$data1, ds_offset:$offset, M0Reg:$m0),
  asm#" $vdst, $addr, $data0, $data1"#"$offset"#" [M0]",
  [], noRetOp>;

// 1 address, 2 data.
multiclass DS_1A2D_NORET_m <bits<8> op, string opName, dag outs, dag ins,
                            string asm, list<dag> pat, string noRetOp> {
  let mayLoad = 1, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>,
             AtomicNoRet<noRetOp, 0>;

    def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
    def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass DS_1A2D_NORET <bits<8> op, string asm, RegisterClass rc,
                     string noRetOp = asm> : DS_1A2D_NORET_m <
  op, asm,
  (outs),
  (ins i1imm:$gds, VGPR_32:$addr, rc:$data0, rc:$data1, ds_offset:$offset, M0Reg:$m0),
  asm#" $addr, $data0, $data1"#"$offset"#" [M0]",
  [], noRetOp>;

// 1 address, 1 data.
multiclass DS_1A1D_NORET_m <bits<8> op, string opName, dag outs, dag ins,
                            string asm, list<dag> pat, string noRetOp> {
  let mayLoad = 1, mayStore = 1 in {
    def "" : DS_Pseudo <opName, outs, ins, pat>,
             AtomicNoRet<noRetOp, 0>;

    let data1 = 0 in {
      def _si : DS_1A_Real_si <op, opName, outs, ins, asm>;
      def _vi : DS_1A_Real_vi <op, opName, outs, ins, asm>;
    }
  }
}

multiclass DS_1A1D_NORET <bits<8> op, string asm, RegisterClass rc,
                          string noRetOp = asm> : DS_1A1D_NORET_m <
  op, asm,
  (outs),
  (ins i1imm:$gds, VGPR_32:$addr, rc:$data0, ds_offset:$offset, M0Reg:$m0),
  asm#" $addr, $data0"#"$offset"#" [M0]",
  [], noRetOp>;

//===----------------------------------------------------------------------===//
// MTBUF classes
//===----------------------------------------------------------------------===//

class MTBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MTBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;
}

class MTBUF_Real_si <bits<3> op, string opName, dag outs, dag ins,
                    string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe <op>,
  SIMCInstr<opName, SISubtarget.SI>;

class MTBUF_Real_vi <bits<4> op, string opName, dag outs, dag ins, string asm> :
  MTBUF <outs, ins, asm, []>,
  MTBUFe_vi <op>,
  SIMCInstr <opName, SISubtarget.VI>;

multiclass MTBUF_m <bits<3> op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MTBUF_Pseudo <opName, outs, ins, pattern>;

  def _si : MTBUF_Real_si <op, opName, outs, ins, asm>;

  def _vi : MTBUF_Real_vi <{0, op{2}, op{1}, op{0}}, opName, outs, ins, asm>;

}

let mayStore = 1, mayLoad = 0 in {

multiclass MTBUF_Store_Helper <bits<3> op, string opName,
                               RegisterClass regClass> : MTBUF_m <
  op, opName, (outs),
  (ins regClass:$vdata, u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc,
   i1imm:$addr64, i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr,
   SReg_128:$srsrc, i1imm:$slc, i1imm:$tfe, SCSrc_32:$soffset),
  opName#" $vdata, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayStore = 1, mayLoad = 0

let mayLoad = 1, mayStore = 0 in {

multiclass MTBUF_Load_Helper <bits<3> op, string opName,
                              RegisterClass regClass> : MTBUF_m <
  op, opName, (outs regClass:$dst),
  (ins u16imm:$offset, i1imm:$offen, i1imm:$idxen, i1imm:$glc, i1imm:$addr64,
       i8imm:$dfmt, i8imm:$nfmt, VGPR_32:$vaddr, SReg_128:$srsrc,
       i1imm:$slc, i1imm:$tfe, SCSrc_32:$soffset),
  opName#" $dst, $offset, $offen, $idxen, $glc, $addr64, $dfmt,"
        #" $nfmt, $vaddr, $srsrc, $slc, $tfe, $soffset", []
>;

} // mayLoad = 1, mayStore = 0

//===----------------------------------------------------------------------===//
// MUBUF classes
//===----------------------------------------------------------------------===//

class mubuf <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class MUBUFAddr64Table <bit is_addr64, string suffix = ""> {
  bit IsAddr64 = is_addr64;
  string OpName = NAME # suffix;
}

class MUBUF_Pseudo <string opName, dag outs, dag ins, list<dag> pattern> :
  MUBUF <outs, ins, "", pattern>,
  SIMCInstr<opName, SISubtarget.NONE> {
  let isPseudo = 1;

  // dummy fields, so that we can use let statements around multiclasses
  bits<1> offen;
  bits<1> idxen;
  bits<8> vaddr;
  bits<1> glc;
  bits<1> slc;
  bits<1> tfe;
  bits<8> soffset;
}

class MUBUF_Real_si <mubuf op, string opName, dag outs, dag ins,
                     string asm> :
  MUBUF <outs, ins, asm, []>,
  MUBUFe <op.SI>,
  SIMCInstr<opName, SISubtarget.SI> {
  let lds = 0;
}

class MUBUF_Real_vi <mubuf op, string opName, dag outs, dag ins,
                     string asm> :
  MUBUF <outs, ins, asm, []>,
  MUBUFe_vi <op.VI>,
  SIMCInstr<opName, SISubtarget.VI> {
  let lds = 0;
}

multiclass MUBUF_m <mubuf op, string opName, dag outs, dag ins, string asm,
                    list<dag> pattern> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <0>;

  let addr64 = 0 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  def _vi : MUBUF_Real_vi <op, opName, outs, ins, asm>;
}

multiclass MUBUFAddr64_m <mubuf op, string opName, dag outs,
                          dag ins, string asm, list<dag> pattern> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <1>;

  let addr64 = 1 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  // There is no VI version. If the pseudo is selected, it should be lowered
  // for VI appropriately.
}

class MUBUF_si <bits<7> op, dag outs, dag ins, string asm, list<dag> pattern> :
  MUBUF <outs, ins, asm, pattern>, MUBUFe <op> {
  let lds = 0;
}

multiclass MUBUFAtomicOffset_m <mubuf op, string opName, dag outs, dag ins,
                                string asm, list<dag> pattern, bit is_return> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <0, !if(is_return, "_RTN", "")>,
           AtomicNoRet<NAME#"_OFFSET", is_return>;

  let offen = 0, idxen = 0, tfe = 0, vaddr = 0 in {
    let addr64 = 0 in {
      def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
    }

    def _vi : MUBUF_Real_vi <op, opName, outs, ins, asm>;
  }
}

multiclass MUBUFAtomicAddr64_m <mubuf op, string opName, dag outs, dag ins,
                                string asm, list<dag> pattern, bit is_return> {

  def "" : MUBUF_Pseudo <opName, outs, ins, pattern>,
           MUBUFAddr64Table <1, !if(is_return, "_RTN", "")>,
           AtomicNoRet<NAME#"_ADDR64", is_return>;

  let offen = 0, idxen = 0, addr64 = 1, tfe = 0, soffset = 128 in {
    def _si : MUBUF_Real_si <op, opName, outs, ins, asm>;
  }

  // There is no VI version. If the pseudo is selected, it should be lowered
  // for VI appropriately.
}

multiclass MUBUF_Atomic <mubuf op, string name, RegisterClass rc,
                         ValueType vt, SDPatternOperator atomic> {

  let mayStore = 1, mayLoad = 1, hasPostISelHook = 1 in {

    // No return variants
    let glc = 0 in {

      defm _ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_addr64", (outs),
        (ins rc:$vdata, SReg_128:$srsrc, VReg_64:$vaddr,
             mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset"#"$slc", [], 0
      >;

      defm _OFFSET : MUBUFAtomicOffset_m <
        op, name#"_offset", (outs),
        (ins rc:$vdata, SReg_128:$srsrc, mbuf_offset:$offset,
             SCSrc_32:$soffset, slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#"$slc", [], 0
      >;
    } // glc = 0

    // Variant that return values
    let glc = 1, Constraints = "$vdata = $vdata_in",
        DisableEncoding = "$vdata_in"  in {

      defm _RTN_ADDR64 : MUBUFAtomicAddr64_m <
        op, name#"_rtn_addr64", (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, VReg_64:$vaddr,
             mbuf_offset:$offset, slc:$slc),
        name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset"#" glc"#"$slc",
        [(set vt:$vdata,
         (atomic (MUBUFAddr64Atomic v4i32:$srsrc, i64:$vaddr, i16:$offset,
                                    i1:$slc), vt:$vdata_in))], 1
      >;

      defm _RTN_OFFSET : MUBUFAtomicOffset_m <
        op, name#"_rtn_offset", (outs rc:$vdata),
        (ins rc:$vdata_in, SReg_128:$srsrc, mbuf_offset:$offset,
             SCSrc_32:$soffset, slc:$slc),
        name#" $vdata, $srsrc, $soffset"#"$offset"#" glc $slc",
        [(set vt:$vdata,
         (atomic (MUBUFOffsetAtomic v4i32:$srsrc, i32:$soffset, i16:$offset,
                                    i1:$slc), vt:$vdata_in))], 1
      >;

    } // glc = 1

  } // mayStore = 1, mayLoad = 1, hasPostISelHook = 1
}

multiclass MUBUF_Load_Helper <mubuf op, string name, RegisterClass regClass,
                              ValueType load_vt = i32,
                              SDPatternOperator ld = null_frag> {

  let mayLoad = 1, mayStore = 0 in {
    let offen = 0, idxen = 0, vaddr = 0 in {
      defm _OFFSET : MUBUF_m <op, name#"_offset", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc,
                           mbuf_offset:$offset, SCSrc_32:$soffset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
                           [(set load_vt:$vdata, (ld (MUBUFOffset v4i32:$srsrc,
                                                     i32:$soffset, i16:$offset,
                                                     i1:$glc, i1:$slc, i1:$tfe)))]>;
    }

    let offen = 1, idxen = 0  in {
      defm _OFFEN  : MUBUF_m <op, name#"_offen", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, VGPR_32:$vaddr,
                           SCSrc_32:$soffset, mbuf_offset:$offset, glc:$glc, slc:$slc,
                           tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 0, idxen = 1 in {
      defm _IDXEN  : MUBUF_m <op, name#"_idxen", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, VGPR_32:$vaddr,
                           mbuf_offset:$offset, SCSrc_32:$soffset, glc:$glc,
                           slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset idxen"#"$offset"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 1, idxen = 1 in {
      defm _BOTHEN : MUBUF_m <op, name#"_bothen", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, VReg_64:$vaddr,
                           SCSrc_32:$soffset, glc:$glc, slc:$slc, tfe:$tfe),
                           name#" $vdata, $vaddr, $srsrc, $soffset, idxen offen"#"$glc"#"$slc"#"$tfe", []>;
    }

    let offen = 0, idxen = 0, glc = 0, slc = 0, tfe = 0, soffset = 128 /* ZERO */ in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs regClass:$vdata),
                           (ins SReg_128:$srsrc, VReg_64:$vaddr, mbuf_offset:$offset),
                           name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset",
                           [(set load_vt:$vdata, (ld (MUBUFAddr64 v4i32:$srsrc,
                                                  i64:$vaddr, i16:$offset)))]>;
    }
  }
}

multiclass MUBUF_Store_Helper <mubuf op, string name, RegisterClass vdataClass,
                          ValueType store_vt, SDPatternOperator st> {
  let mayLoad = 0, mayStore = 1 in {
    defm : MUBUF_m <op, name, (outs),
                    (ins vdataClass:$vdata, SReg_128:$srsrc, VGPR_32:$vaddr, SCSrc_32:$soffset,
                    mbuf_offset:$offset, offen:$offen, idxen:$idxen, glc:$glc, slc:$slc,
                    tfe:$tfe),
                    name#" $vdata, $vaddr, $srsrc, $soffset"#"$offen"#"$idxen"#"$offset"#
                    "$glc"#"$slc"#"$tfe", []>;

    let offen = 0, idxen = 0, vaddr = 0 in {
      defm _OFFSET : MUBUF_m <op, name#"_offset",(outs),
                              (ins vdataClass:$vdata, SReg_128:$srsrc, mbuf_offset:$offset,
                              SCSrc_32:$soffset, glc:$glc, slc:$slc, tfe:$tfe),
                              name#" $vdata, $srsrc, $soffset"#"$offset"#"$glc"#"$slc"#"$tfe",
                              [(st store_vt:$vdata, (MUBUFOffset v4i32:$srsrc, i32:$soffset,
                                   i16:$offset, i1:$glc, i1:$slc, i1:$tfe))]>;
    } // offen = 0, idxen = 0, vaddr = 0

    let offen = 1, idxen = 0  in {
      defm _OFFEN : MUBUF_m <op, name#"_offen", (outs),
                             (ins vdataClass:$vdata, SReg_128:$srsrc, VGPR_32:$vaddr, SCSrc_32:$soffset,
                             mbuf_offset:$offset, glc:$glc, slc:$slc, tfe:$tfe),
                             name#" $vdata, $vaddr, $srsrc, $soffset offen"#"$offset"#
                             "$glc"#"$slc"#"$tfe", []>;
    } // end offen = 1, idxen = 0

    let offen = 0, idxen = 0, glc = 0, slc = 0, tfe = 0,
        soffset = 128 /* ZERO */ in {
      defm _ADDR64 : MUBUFAddr64_m <op, name#"_addr64", (outs),
                                    (ins vdataClass:$vdata, SReg_128:$srsrc, VReg_64:$vaddr, mbuf_offset:$offset),
                                    name#" $vdata, $vaddr, $srsrc, 0 addr64"#"$offset",
                                    [(st store_vt:$vdata,
                                      (MUBUFAddr64 v4i32:$srsrc, i64:$vaddr, i16:$offset))]>;
    }
  } // End mayLoad = 0, mayStore = 1
}

class FLAT_Load_Helper <bits<7> op, string asm, RegisterClass regClass> :
      FLAT <op, (outs regClass:$data),
                (ins VReg_64:$addr),
            asm#" $data, $addr, [M0, FLAT_SCRATCH]", []> {
  let glc = 0;
  let slc = 0;
  let tfe = 0;
  let mayLoad = 1;
}

class FLAT_Store_Helper <bits<7> op, string name, RegisterClass vdataClass> :
      FLAT <op, (outs), (ins vdataClass:$data, VReg_64:$addr),
          name#" $data, $addr, [M0, FLAT_SCRATCH]",
         []> {

  let mayLoad = 0;
  let mayStore = 1;

  // Encoding
  let glc = 0;
  let slc = 0;
  let tfe = 0;
}

class MIMG_Mask <string op, int channels> {
  string Op = op;
  int Channels = channels;
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass src_rc> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc",
  []> {
  let SSAMP = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_NoSampler <bits<7> op, string asm> {
  defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc, int wqm> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
  let WQM = wqm;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, int wqm> {
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32, wqm>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Sampler <bits<7> op, string asm> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, 0>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, 0>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, 0>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, 0>;
}

multiclass MIMG_Sampler_WQM <bits<7> op, string asm> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, 1>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, 1>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, 1>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, 1>;
}

class MIMG_Gather_Helper <bits<7> op, string asm,
                          RegisterClass dst_rc,
                          RegisterClass src_rc, int wqm> : MIMG <
  op,
  (outs dst_rc:$vdata),
  (ins i32imm:$dmask, i1imm:$unorm, i1imm:$glc, i1imm:$da, i1imm:$r128,
       i1imm:$tfe, i1imm:$lwe, i1imm:$slc, src_rc:$vaddr,
       SReg_256:$srsrc, SReg_128:$ssamp),
  asm#" $vdata, $dmask, $unorm, $glc, $da, $r128,"
     #" $tfe, $lwe, $slc, $vaddr, $srsrc, $ssamp",
  []> {
  let mayLoad = 1;
  let mayStore = 0;

  // DMASK was repurposed for GATHER4. 4 components are always
  // returned and DMASK works like a swizzle - it selects
  // the component to fetch. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  // Therefore, disable all code which updates DMASK by setting these two:
  let MIMG = 0;
  let hasPostISelHook = 0;
  let WQM = wqm;
}

multiclass MIMG_Gather_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, int wqm> {
  def _V1 : MIMG_Gather_Helper <op, asm, dst_rc, VGPR_32, wqm>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Gather <bits<7> op, string asm> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, 0>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, 0>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, 0>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, 0>;
}

multiclass MIMG_Gather_WQM <bits<7> op, string asm> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, 1>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, 1>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, 1>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, 1>;
}

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["4"];
  let ValueCols = [["8"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size"];
  let KeyCol = ["8"];
  let ValueCols = [["4"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

def getMaskedMIMGOp : InstrMapping {
  let FilterClass = "MIMG_Mask";
  let RowFields = ["Op"];
  let ColFields = ["Channels"];
  let KeyCol = ["4"];
  let ValueCols = [["1"], ["2"], ["3"] ];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "VOP2_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def getMCOpcodeGen : InstrMapping {
  let FilterClass = "SIMCInstr";
  let RowFields = ["PseudoInstr"];
  let ColFields = ["Subtarget"];
  let KeyCol = [!cast<string>(SISubtarget.NONE)];
  let ValueCols = [[!cast<string>(SISubtarget.SI)],[!cast<string>(SISubtarget.VI)]];
}

def getAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its version with a return value.
def getAtomicRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an atomic opcode to its returnless version.
def getAtomicNoRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

include "SIInstructions.td"
include "CIInstructions.td"
include "VIInstructions.td"
