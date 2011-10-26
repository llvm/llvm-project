//===- MipsInstrInfo.td - Target Description for Mips Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "MipsInstrFormats.td"

//===----------------------------------------------------------------------===//
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

def SDT_MipsRet          : SDTypeProfile<0, 1, [SDTCisInt<0>]>;
def SDT_MipsJmpLink      : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;
def SDT_MipsCMov         : SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>,
                                                SDTCisSameAs<1, 2>,
                                                SDTCisSameAs<3, 4>,
                                                SDTCisInt<4>]>;
def SDT_MipsCallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_MipsCallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDT_MipsMAddMSub     : SDTypeProfile<0, 4,
                                         [SDTCisVT<0, i32>, SDTCisSameAs<0, 1>,
                                          SDTCisSameAs<1, 2>,
                                          SDTCisSameAs<2, 3>]>;
def SDT_MipsDivRem       : SDTypeProfile<0, 2,
                                         [SDTCisInt<0>,
                                          SDTCisSameAs<0, 1>]>;

def SDT_MipsThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;

def SDT_MipsDynAlloc    : SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                               SDTCisVT<1, iPTR>]>;
def SDT_Sync             : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

def SDT_Ext : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                   SDTCisVT<2, i32>, SDTCisSameAs<2, 3>]>;
def SDT_Ins : SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                   SDTCisVT<2, i32>, SDTCisSameAs<2, 3>,
                                   SDTCisSameAs<0, 4>]>;

// Call
def MipsJmpLink : SDNode<"MipsISD::JmpLink",SDT_MipsJmpLink,
                         [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                          SDNPVariadic]>;

// Hi and Lo nodes are used to handle global addresses. Used on
// MipsISelLowering to lower stuff like GlobalAddress, ExternalSymbol
// static model. (nothing to do with Mips Registers Hi and Lo)
def MipsHi    : SDNode<"MipsISD::Hi", SDTIntUnaryOp>;
def MipsLo    : SDNode<"MipsISD::Lo", SDTIntUnaryOp>;
def MipsGPRel : SDNode<"MipsISD::GPRel", SDTIntUnaryOp>;

// TlsGd node is used to handle General Dynamic TLS
def MipsTlsGd : SDNode<"MipsISD::TlsGd", SDTIntUnaryOp>;

// TprelHi and TprelLo nodes are used to handle Local Exec TLS
def MipsTprelHi    : SDNode<"MipsISD::TprelHi", SDTIntUnaryOp>;
def MipsTprelLo    : SDNode<"MipsISD::TprelLo", SDTIntUnaryOp>;

// Thread pointer
def MipsThreadPointer: SDNode<"MipsISD::ThreadPointer", SDT_MipsThreadPointer>;

// Return
def MipsRet : SDNode<"MipsISD::Ret", SDT_MipsRet, [SDNPHasChain,
                     SDNPOptInGlue]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MipsCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MipsCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

// MAdd*/MSub* nodes
def MipsMAdd      : SDNode<"MipsISD::MAdd", SDT_MipsMAddMSub,
                           [SDNPOptInGlue, SDNPOutGlue]>;
def MipsMAddu     : SDNode<"MipsISD::MAddu", SDT_MipsMAddMSub,
                           [SDNPOptInGlue, SDNPOutGlue]>;
def MipsMSub      : SDNode<"MipsISD::MSub", SDT_MipsMAddMSub,
                           [SDNPOptInGlue, SDNPOutGlue]>;
def MipsMSubu     : SDNode<"MipsISD::MSubu", SDT_MipsMAddMSub,
                           [SDNPOptInGlue, SDNPOutGlue]>;

// DivRem(u) nodes
def MipsDivRem    : SDNode<"MipsISD::DivRem", SDT_MipsDivRem,
                           [SDNPOutGlue]>;
def MipsDivRemU   : SDNode<"MipsISD::DivRemU", SDT_MipsDivRem,
                           [SDNPOutGlue]>;

// Target constant nodes that are not part of any isel patterns and remain
// unchanged can cause instructions with illegal operands to be emitted.
// Wrapper node patterns give the instruction selector a chance to replace
// target constant nodes that would otherwise remain unchanged with ADDiu
// nodes. Without these wrapper node patterns, the following conditional move
// instrucion is emitted when function cmov2 in test/CodeGen/Mips/cmov.ll is
// compiled: 
//  movn  %got(d)($gp), %got(c)($gp), $4
// This instruction is illegal since movn can take only register operands.

def MipsWrapperPIC    : SDNode<"MipsISD::WrapperPIC",  SDTIntUnaryOp>;

// Pointer to dynamically allocated stack area.
def MipsDynAlloc  : SDNode<"MipsISD::DynAlloc", SDT_MipsDynAlloc,
                           [SDNPHasChain, SDNPInGlue]>;

def MipsSync : SDNode<"MipsISD::Sync", SDT_Sync, [SDNPHasChain]>;

def MipsExt :  SDNode<"MipsISD::Ext", SDT_Ext>;
def MipsIns :  SDNode<"MipsISD::Ins", SDT_Ins>;

//===----------------------------------------------------------------------===//
// Mips Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasSEInReg  : Predicate<"Subtarget.hasSEInReg()">;
def HasBitCount : Predicate<"Subtarget.hasBitCount()">;
def HasSwap     : Predicate<"Subtarget.hasSwap()">;
def HasCondMov  : Predicate<"Subtarget.hasCondMov()">;
def HasMips32    : Predicate<"Subtarget.hasMips32()">;
def HasMips32r2  : Predicate<"Subtarget.hasMips32r2()">;
def HasMips64    : Predicate<"Subtarget.hasMips64()">;
def NotMips64    : Predicate<"!Subtarget.hasMips64()">;
def HasMips64r2  : Predicate<"Subtarget.hasMips64r2()">;
def IsN64       : Predicate<"Subtarget.isABI_N64()">;
def NotN64      : Predicate<"!Subtarget.isABI_N64()">;

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Instruction operand types
def brtarget    : Operand<OtherVT>;
def calltarget  : Operand<i32>;
def simm16      : Operand<i32>;
def simm16_64   : Operand<i64>;
def shamt       : Operand<i32>;

// Unsigned Operand
def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

// Address operand
def mem : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPURegs, simm16);
  let EncoderMethod = "getMemEncoding";
}

def mem64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPU64Regs, simm16_64);
}

def mem_ea : Operand<i32> {
  let PrintMethod = "printMemOperandEA";
  let MIOperandInfo = (ops CPURegs, simm16);
  let EncoderMethod = "getMemEncoding";
}

// size operand of ext instruction
def size_ext : Operand<i32> {
  let EncoderMethod = "getSizeExtEncoding";
}

// size operand of ins instruction
def size_ins : Operand<i32> {
  let EncoderMethod = "getSizeInsEncoding";
}

// Transformation Function - get the lower 16 bits.
def LO16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() & 0xFFFF);
}]>;

// Transformation Function - get the higher 16 bits.
def HI16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() >> 16);
}]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{ return isInt<16>(N->getSExtValue()); }]>;

// Node immediate fits as 16-bit zero extended on target immediate.
// The LO16 param means that only the lower 16 bits of the node
// immediate are caught.
// e.g. addiu, sltiu
def immZExt16  : PatLeaf<(imm), [{
  if (N->getValueType(0) == MVT::i32)
    return (uint32_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
  else
    return (uint64_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
}], LO16>;

// shamt field must fit in 5 bits.
def immZExt5 : PatLeaf<(imm), [{
  return N->getZExtValue() == ((N->getZExtValue()) & 0x1f) ;
}]>;

// Mips Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr : ComplexPattern<iPTR, 2, "SelectAddr", [frameindex], []>;

//===----------------------------------------------------------------------===//
// Pattern fragment for load/store
//===----------------------------------------------------------------------===//
class UnalignedLoad<PatFrag Node> : PatFrag<(ops node:$ptr), (Node node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getMemoryVT().getSizeInBits()/8 > LD->getAlignment();
}]>;

class AlignedLoad<PatFrag Node> : PatFrag<(ops node:$ptr), (Node node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  return LD->getMemoryVT().getSizeInBits()/8 <= LD->getAlignment();
}]>;

class UnalignedStore<PatFrag Node> : PatFrag<(ops node:$val, node:$ptr),
                                             (Node node:$val, node:$ptr), [{
  StoreSDNode *SD = cast<StoreSDNode>(N);
  return SD->getMemoryVT().getSizeInBits()/8 > SD->getAlignment();
}]>;

class AlignedStore<PatFrag Node> : PatFrag<(ops node:$val, node:$ptr),
                                           (Node node:$val, node:$ptr), [{
  StoreSDNode *SD = cast<StoreSDNode>(N);
  return SD->getMemoryVT().getSizeInBits()/8 <= SD->getAlignment();
}]>;

// Load/Store PatFrags.
def sextloadi16_a   : AlignedLoad<sextloadi16>;
def zextloadi16_a   : AlignedLoad<zextloadi16>;
def extloadi16_a    : AlignedLoad<extloadi16>;
def load_a          : AlignedLoad<load>;
def sextloadi32_a   : AlignedLoad<sextloadi32>;
def zextloadi32_a   : AlignedLoad<zextloadi32>;
def extloadi32_a    : AlignedLoad<extloadi32>;
def truncstorei16_a : AlignedStore<truncstorei16>;
def store_a         : AlignedStore<store>;
def truncstorei32_a : AlignedStore<truncstorei32>;
def sextloadi16_u   : UnalignedLoad<sextloadi16>;
def zextloadi16_u   : UnalignedLoad<zextloadi16>;
def extloadi16_u    : UnalignedLoad<extloadi16>;
def load_u          : UnalignedLoad<load>;
def sextloadi32_u   : UnalignedLoad<sextloadi32>;
def zextloadi32_u   : UnalignedLoad<zextloadi32>;
def extloadi32_u    : UnalignedLoad<extloadi32>;
def truncstorei16_u : UnalignedStore<truncstorei16>;
def store_u         : UnalignedStore<store>;
def truncstorei32_u : UnalignedStore<truncstorei32>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic and logical instructions with 3 register operands.
class ArithLogicR<bits<6> op, bits<6> func, string instr_asm, SDNode OpNode,
                  InstrItinClass itin, RegisterClass RC, bit isComm = 0>:
  FR<op, func, (outs RC:$rd), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"),
     [(set RC:$rd, (OpNode RC:$rs, RC:$rt))], itin> {
  let shamt = 0;
  let isCommutable = isComm;
}

class ArithOverflowR<bits<6> op, bits<6> func, string instr_asm,
                    InstrItinClass itin, RegisterClass RC, bit isComm = 0>:
  FR<op, func, (outs RC:$rd), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"), [], itin> {
  let shamt = 0;
  let isCommutable = isComm;
}

// Arithmetic and logical instructions with 2 register operands.
class ArithLogicI<bits<6> op, string instr_asm, SDNode OpNode,
                  Operand Od, PatLeaf imm_type, RegisterClass RC> :
  FI<op, (outs RC:$rt), (ins RC:$rs, Od:$imm16),
     !strconcat(instr_asm, "\t$rt, $rs, $imm16"),
     [(set RC:$rt, (OpNode RC:$rs, imm_type:$imm16))], IIAlu>;

class ArithOverflowI<bits<6> op, string instr_asm, SDNode OpNode,
                     Operand Od, PatLeaf imm_type, RegisterClass RC> :
  FI<op, (outs RC:$rt), (ins RC:$rs, Od:$imm16),
     !strconcat(instr_asm, "\t$rt, $rs, $imm16"), [], IIAlu>;

// Arithmetic Multiply ADD/SUB
let rd = 0, shamt = 0, Defs = [HI, LO], Uses = [HI, LO] in
class MArithR<bits<6> func, string instr_asm, SDNode op, bit isComm = 0> :
  FR<0x1c, func, (outs), (ins CPURegs:$rs, CPURegs:$rt),
     !strconcat(instr_asm, "\t$rs, $rt"),
     [(op CPURegs:$rs, CPURegs:$rt, LO, HI)], IIImul> {
  let rd = 0;
  let shamt = 0;
  let isCommutable = isComm;
}

//  Logical
class LogicNOR<bits<6> op, bits<6> func, string instr_asm, RegisterClass RC>:
  FR<op, func, (outs RC:$rd), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"),
     [(set RC:$rd, (not (or RC:$rs, RC:$rt)))], IIAlu> {
  let shamt = 0;
  let isCommutable = 1;
}

// Shifts
class LogicR_shift_rotate_imm<bits<6> func, bits<5> _rs, string instr_asm,
                              SDNode OpNode>:
  FR<0x00, func, (outs CPURegs:$rd), (ins CPURegs:$rt, shamt:$shamt),
     !strconcat(instr_asm, "\t$rd, $rt, $shamt"),
     [(set CPURegs:$rd, (OpNode CPURegs:$rt, (i32 immZExt5:$shamt)))], IIAlu> {
  let rs = _rs;
}

class LogicR_shift_rotate_reg<bits<6> func, bits<5> isRotate, string instr_asm,
                              SDNode OpNode>:
  FR<0x00, func, (outs CPURegs:$rd), (ins CPURegs:$rs, CPURegs:$rt),
     !strconcat(instr_asm, "\t$rd, $rt, $rs"),
     [(set CPURegs:$rd, (OpNode CPURegs:$rt, CPURegs:$rs))], IIAlu> {
  let shamt = isRotate;
}

// Load Upper Imediate
class LoadUpper<bits<6> op, string instr_asm>:
  FI<op, (outs CPURegs:$rt), (ins uimm16:$imm16),
     !strconcat(instr_asm, "\t$rt, $imm16"), [], IIAlu> {
  let rs = 0;
}

class FMem<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
          InstrItinClass itin>: FFI<op, outs, ins, asmstr, pattern> {
  bits<21> addr;
  let Inst{25-21} = addr{20-16};
  let Inst{15-0}  = addr{15-0};
}

// Memory Load/Store
let canFoldAsLoad = 1 in
class LoadM<bits<6> op, string instr_asm, PatFrag OpNode, RegisterClass RC,
            Operand MemOpnd, bit Pseudo>:
  FMem<op, (outs RC:$rt), (ins MemOpnd:$addr),
     !strconcat(instr_asm, "\t$rt, $addr"),
     [(set RC:$rt, (OpNode addr:$addr))], IILoad> {
  let isPseudo = Pseudo;
}

class StoreM<bits<6> op, string instr_asm, PatFrag OpNode, RegisterClass RC,
             Operand MemOpnd, bit Pseudo>:
  FMem<op, (outs), (ins RC:$rt, MemOpnd:$addr),
     !strconcat(instr_asm, "\t$rt, $addr"),
     [(OpNode RC:$rt, addr:$addr)], IIStore> {
  let isPseudo = Pseudo;
}

// 32-bit load.
multiclass LoadM32<bits<6> op, string instr_asm, PatFrag OpNode,
                   bit Pseudo = 0> {
  def #NAME# : LoadM<op, instr_asm, OpNode, CPURegs, mem, Pseudo>,
               Requires<[NotN64]>;
  def _P8    : LoadM<op, instr_asm, OpNode, CPURegs, mem64, Pseudo>,
               Requires<[IsN64]>;
} 

// 64-bit load.
multiclass LoadM64<bits<6> op, string instr_asm, PatFrag OpNode,
                   bit Pseudo = 0> {
  def #NAME# : LoadM<op, instr_asm, OpNode, CPU64Regs, mem, Pseudo>,
               Requires<[NotN64]>;
  def _P8    : LoadM<op, instr_asm, OpNode, CPU64Regs, mem64, Pseudo>,
               Requires<[IsN64]>;
} 

// 32-bit store.
multiclass StoreM32<bits<6> op, string instr_asm, PatFrag OpNode,
                    bit Pseudo = 0> {
  def #NAME# : StoreM<op, instr_asm, OpNode, CPURegs, mem, Pseudo>,
               Requires<[NotN64]>;
  def _P8    : StoreM<op, instr_asm, OpNode, CPURegs, mem64, Pseudo>,
               Requires<[IsN64]>;
}

// 64-bit store.
multiclass StoreM64<bits<6> op, string instr_asm, PatFrag OpNode,
                    bit Pseudo = 0> {
  def #NAME# : StoreM<op, instr_asm, OpNode, CPU64Regs, mem, Pseudo>,
               Requires<[NotN64]>;
  def _P8    : StoreM<op, instr_asm, OpNode, CPU64Regs, mem64, Pseudo>,
               Requires<[IsN64]>;
}

// Conditional Branch
class CBranch<bits<6> op, string instr_asm, PatFrag cond_op, RegisterClass RC>:
  CBranchBase<op, (outs), (ins RC:$rs, RC:$rt, brtarget:$imm16),
              !strconcat(instr_asm, "\t$rs, $rt, $imm16"),
              [(brcond (i32 (cond_op RC:$rs, RC:$rt)), bb:$imm16)], IIBranch> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

class CBranchZero<bits<6> op, bits<5> _rt, string instr_asm, PatFrag cond_op,
                  RegisterClass RC>:
  CBranchBase<op, (outs), (ins RC:$rs, brtarget:$imm16),
              !strconcat(instr_asm, "\t$rs, $imm16"),
              [(brcond (i32 (cond_op RC:$rs, 0)), bb:$imm16)], IIBranch> {
  let rt = _rt;
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

// SetCC
class SetCC_R<bits<6> op, bits<6> func, string instr_asm, PatFrag cond_op,
              RegisterClass RC>:
  FR<op, func, (outs CPURegs:$rd), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"),
     [(set CPURegs:$rd, (cond_op RC:$rs, RC:$rt))],
     IIAlu> {
  let shamt = 0;
}

class SetCC_I<bits<6> op, string instr_asm, PatFrag cond_op, Operand Od,
              PatLeaf imm_type, RegisterClass RC>:
  FI<op, (outs CPURegs:$rt), (ins RC:$rs, Od:$imm16),
     !strconcat(instr_asm, "\t$rt, $rs, $imm16"),
     [(set CPURegs:$rt, (cond_op RC:$rs, imm_type:$imm16))],
     IIAlu>;

// Unconditional branch
let isBranch=1, isTerminator=1, isBarrier=1, hasDelaySlot = 1 in
class JumpFJ<bits<6> op, string instr_asm>:
  FJ<op, (outs), (ins brtarget:$target),
     !strconcat(instr_asm, "\t$target"), [(br bb:$target)], IIBranch>;

let isBranch=1, isTerminator=1, isBarrier=1, rd=0, hasDelaySlot = 1 in
class JumpFR<bits<6> op, bits<6> func, string instr_asm>:
  FR<op, func, (outs), (ins CPURegs:$rs),
     !strconcat(instr_asm, "\t$rs"), [(brind CPURegs:$rs)], IIBranch> {
  let rt = 0;
  let rd = 0;
  let shamt = 0;
}

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1,
  // All calls clobber the non-callee saved registers...
  Defs = [AT, V0, V1, A0, A1, A2, A3, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9,
          K0, K1, D0, D1, D2, D3, D4, D5, D6, D7, D8, D9], Uses = [GP] in {
  class JumpLink<bits<6> op, string instr_asm>:
    FJ<op, (outs), (ins calltarget:$target, variable_ops),
       !strconcat(instr_asm, "\t$target"), [(MipsJmpLink imm:$target)],
       IIBranch>;

  class JumpLinkReg<bits<6> op, bits<6> func, string instr_asm>:
    FR<op, func, (outs), (ins CPURegs:$rs, variable_ops),
       !strconcat(instr_asm, "\t$rs"), [(MipsJmpLink CPURegs:$rs)], IIBranch> {
    let rt = 0;
    let rd = 31;
    let shamt = 0;
  }

  class BranchLink<string instr_asm>:
    FI<0x1, (outs), (ins CPURegs:$rs, brtarget:$imm16, variable_ops),
       !strconcat(instr_asm, "\t$rs, $imm16"), [], IIBranch>;
}

// Mul, Div
class Mul<bits<6> func, string instr_asm, InstrItinClass itin>:
  FR<0x00, func, (outs), (ins CPURegs:$rs, CPURegs:$rt),
     !strconcat(instr_asm, "\t$rs, $rt"), [], itin> {
  let rd = 0;
  let shamt = 0;
  let isCommutable = 1;
  let Defs = [HI, LO];
}

class Div<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin>:
          FR<0x00, func, (outs), (ins CPURegs:$rs, CPURegs:$rt),
          !strconcat(instr_asm, "\t$$zero, $rs, $rt"),
          [(op CPURegs:$rs, CPURegs:$rt)], itin> {
  let rd = 0;
  let shamt = 0;
  let Defs = [HI, LO];
}

// Move from Hi/Lo
class MoveFromLOHI<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs CPURegs:$rd), (ins),
     !strconcat(instr_asm, "\t$rd"), [], IIHiLo> {
  let rs = 0;
  let rt = 0;
  let shamt = 0;
}

class MoveToLOHI<bits<6> func, string instr_asm>:
  FR<0x00, func, (outs), (ins CPURegs:$rs),
     !strconcat(instr_asm, "\t$rs"), [], IIHiLo> {
  let rt = 0;
  let rd = 0;
  let shamt = 0;
}

class EffectiveAddress<string instr_asm> :
  FMem<0x09, (outs CPURegs:$rt), (ins mem_ea:$addr),
     instr_asm, [(set CPURegs:$rt, addr:$addr)], IIAlu>;

// Count Leading Ones/Zeros in Word
class CountLeading<bits<6> func, string instr_asm, list<dag> pattern>:
  FR<0x1c, func, (outs CPURegs:$rd), (ins CPURegs:$rs),
     !strconcat(instr_asm, "\t$rd, $rs"), pattern, IIAlu>,
     Requires<[HasBitCount]> {
  let shamt = 0;
  let rt = rd;
}

// Sign Extend in Register.
class SignExtInReg<bits<5> sa, string instr_asm, ValueType vt>:
  FR<0x1f, 0x20, (outs CPURegs:$rd), (ins CPURegs:$rt),
     !strconcat(instr_asm, "\t$rd, $rt"),
     [(set CPURegs:$rd, (sext_inreg CPURegs:$rt, vt))], NoItinerary> {
  let rs = 0;
  let shamt = sa;
  let Predicates = [HasSEInReg];
}

// Byte Swap
class ByteSwap<bits<6> func, bits<5> sa, string instr_asm>:
  FR<0x1f, func, (outs CPURegs:$rd), (ins CPURegs:$rt),
     !strconcat(instr_asm, "\t$rd, $rt"),
     [(set CPURegs:$rd, (bswap CPURegs:$rt))], NoItinerary> {
  let rs = 0;
  let shamt = sa;
  let Predicates = [HasSwap];
}

// Read Hardware
class ReadHardware: FR<0x1f, 0x3b, (outs CPURegs:$rt), (ins HWRegs:$rd),
    "rdhwr\t$rt, $rd", [], IIAlu> {
  let rs = 0;
  let shamt = 0;
}

// Ext and Ins
class ExtIns<bits<6> _funct, string instr_asm, dag outs, dag ins,
             list<dag> pattern, InstrItinClass itin>:
  FR<0x1f, _funct, outs, ins, !strconcat(instr_asm, " $rt, $rs, $pos, $sz"),
     pattern, itin>, Requires<[HasMips32r2]> {
  bits<5> pos;
  bits<5> sz;
  let rd = sz;
  let shamt = pos;
}

// Atomic instructions with 2 source operands (ATOMIC_SWAP & ATOMIC_LOAD_*).
class Atomic2Ops<PatFrag Op, string Opstr> :
  MipsPseudo<(outs CPURegs:$dst), (ins CPURegs:$ptr, CPURegs:$incr),
             !strconcat("atomic_", Opstr, "\t$dst, $ptr, $incr"),
             [(set CPURegs:$dst,
              (Op CPURegs:$ptr, CPURegs:$incr))]>;

// Atomic Compare & Swap.
class AtomicCmpSwap<PatFrag Op, string Width> :
  MipsPseudo<(outs CPURegs:$dst), 
             (ins CPURegs:$ptr, CPURegs:$cmp, CPURegs:$swap),
             !strconcat("atomic_cmp_swap_", Width, 
                        "\t$dst, $ptr, $cmp, $swap"),
             [(set CPURegs:$dst,
              (Op CPURegs:$ptr, CPURegs:$cmp, CPURegs:$swap))]>;

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// As stack alignment is always done with addiu, we need a 16-bit immediate
let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN : MipsPseudo<(outs), (ins uimm16:$amt),
                                  "!ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MipsPseudo<(outs), (ins uimm16:$amt1, uimm16:$amt2),
                                  "!ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

// Some assembly macros need to avoid pseudoinstructions and assembler
// automatic reodering, we should reorder ourselves.
def MACRO     : MipsPseudo<(outs), (ins), ".set\tmacro",     []>;
def REORDER   : MipsPseudo<(outs), (ins), ".set\treorder",   []>;
def NOMACRO   : MipsPseudo<(outs), (ins), ".set\tnomacro",   []>;
def NOREORDER : MipsPseudo<(outs), (ins), ".set\tnoreorder", []>;

// These macros are inserted to prevent GAS from complaining
// when using the AT register.
def NOAT      : MipsPseudo<(outs), (ins), ".set\tnoat", []>;
def ATMACRO   : MipsPseudo<(outs), (ins), ".set\tat", []>;

// When handling PIC code the assembler needs .cpload and .cprestore
// directives. If the real instructions corresponding these directives
// are used, we have the same behavior, but get also a bunch of warnings
// from the assembler.
def CPLOAD : MipsPseudo<(outs), (ins CPURegs:$picreg), ".cpload\t$picreg", []>;
def CPRESTORE : MipsPseudo<(outs), (ins i32imm:$loc), ".cprestore\t$loc", []>;

let usesCustomInserter = 1 in {
  def ATOMIC_LOAD_ADD_I8   : Atomic2Ops<atomic_load_add_8, "load_add_8">;
  def ATOMIC_LOAD_ADD_I16  : Atomic2Ops<atomic_load_add_16, "load_add_16">;
  def ATOMIC_LOAD_ADD_I32  : Atomic2Ops<atomic_load_add_32, "load_add_32">;
  def ATOMIC_LOAD_SUB_I8   : Atomic2Ops<atomic_load_sub_8, "load_sub_8">;
  def ATOMIC_LOAD_SUB_I16  : Atomic2Ops<atomic_load_sub_16, "load_sub_16">;
  def ATOMIC_LOAD_SUB_I32  : Atomic2Ops<atomic_load_sub_32, "load_sub_32">;
  def ATOMIC_LOAD_AND_I8   : Atomic2Ops<atomic_load_and_8, "load_and_8">;
  def ATOMIC_LOAD_AND_I16  : Atomic2Ops<atomic_load_and_16, "load_and_16">;
  def ATOMIC_LOAD_AND_I32  : Atomic2Ops<atomic_load_and_32, "load_and_32">;
  def ATOMIC_LOAD_OR_I8    : Atomic2Ops<atomic_load_or_8, "load_or_8">;
  def ATOMIC_LOAD_OR_I16   : Atomic2Ops<atomic_load_or_16, "load_or_16">;
  def ATOMIC_LOAD_OR_I32   : Atomic2Ops<atomic_load_or_32, "load_or_32">;
  def ATOMIC_LOAD_XOR_I8   : Atomic2Ops<atomic_load_xor_8, "load_xor_8">;
  def ATOMIC_LOAD_XOR_I16  : Atomic2Ops<atomic_load_xor_16, "load_xor_16">;
  def ATOMIC_LOAD_XOR_I32  : Atomic2Ops<atomic_load_xor_32, "load_xor_32">;
  def ATOMIC_LOAD_NAND_I8  : Atomic2Ops<atomic_load_nand_8, "load_nand_8">;
  def ATOMIC_LOAD_NAND_I16 : Atomic2Ops<atomic_load_nand_16, "load_nand_16">;
  def ATOMIC_LOAD_NAND_I32 : Atomic2Ops<atomic_load_nand_32, "load_nand_32">;

  def ATOMIC_SWAP_I8       : Atomic2Ops<atomic_swap_8, "swap_8">;
  def ATOMIC_SWAP_I16      : Atomic2Ops<atomic_swap_16, "swap_16">;
  def ATOMIC_SWAP_I32      : Atomic2Ops<atomic_swap_32, "swap_32">;

  def ATOMIC_CMP_SWAP_I8   : AtomicCmpSwap<atomic_cmp_swap_8, "8">;
  def ATOMIC_CMP_SWAP_I16  : AtomicCmpSwap<atomic_cmp_swap_16, "16">;
  def ATOMIC_CMP_SWAP_I32  : AtomicCmpSwap<atomic_cmp_swap_32, "32">;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MipsI Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def ADDiu   : ArithLogicI<0x09, "addiu", add, simm16, immSExt16, CPURegs>;
def ADDi    : ArithOverflowI<0x08, "addi", add, simm16, immSExt16, CPURegs>;
def SLTi    : SetCC_I<0x0a, "slti", setlt, simm16, immSExt16, CPURegs>;
def SLTiu   : SetCC_I<0x0b, "sltiu", setult, simm16, immSExt16, CPURegs>;
def ANDi    : ArithLogicI<0x0c, "andi", and, uimm16, immZExt16, CPURegs>;
def ORi     : ArithLogicI<0x0d, "ori", or, uimm16, immZExt16, CPURegs>;
def XORi    : ArithLogicI<0x0e, "xori", xor, uimm16, immZExt16, CPURegs>;
def LUi     : LoadUpper<0x0f, "lui">;

/// Arithmetic Instructions (3-Operand, R-Type)
def ADDu    : ArithLogicR<0x00, 0x21, "addu", add, IIAlu, CPURegs, 1>;
def SUBu    : ArithLogicR<0x00, 0x23, "subu", sub, IIAlu, CPURegs>;
def ADD     : ArithOverflowR<0x00, 0x20, "add", IIAlu, CPURegs, 1>;
def SUB     : ArithOverflowR<0x00, 0x22, "sub", IIAlu, CPURegs>;
def SLT     : SetCC_R<0x00, 0x2a, "slt", setlt, CPURegs>;
def SLTu    : SetCC_R<0x00, 0x2b, "sltu", setult, CPURegs>;
def AND     : ArithLogicR<0x00, 0x24, "and", and, IIAlu, CPURegs, 1>;
def OR      : ArithLogicR<0x00, 0x25, "or",  or, IIAlu, CPURegs, 1>;
def XOR     : ArithLogicR<0x00, 0x26, "xor", xor, IIAlu, CPURegs, 1>;
def NOR     : LogicNOR<0x00, 0x27, "nor", CPURegs>;

/// Shift Instructions
def SLL     : LogicR_shift_rotate_imm<0x00, 0x00, "sll", shl>;
def SRL     : LogicR_shift_rotate_imm<0x02, 0x00, "srl", srl>;
def SRA     : LogicR_shift_rotate_imm<0x03, 0x00, "sra", sra>;
def SLLV    : LogicR_shift_rotate_reg<0x04, 0x00, "sllv", shl>;
def SRLV    : LogicR_shift_rotate_reg<0x06, 0x00, "srlv", srl>;
def SRAV    : LogicR_shift_rotate_reg<0x07, 0x00, "srav", sra>;

// Rotate Instructions
let Predicates = [HasMips32r2] in {
    def ROTR    : LogicR_shift_rotate_imm<0x02, 0x01, "rotr", rotr>;
    def ROTRV   : LogicR_shift_rotate_reg<0x06, 0x01, "rotrv", rotr>;
}

/// Load and Store Instructions
///  aligned
defm LB      : LoadM32<0x20, "lb",  sextloadi8>;
defm LBu     : LoadM32<0x24, "lbu", zextloadi8>;
defm LH      : LoadM32<0x21, "lh",  sextloadi16_a>;
defm LHu     : LoadM32<0x25, "lhu", zextloadi16_a>;
defm LW      : LoadM32<0x23, "lw",  load_a>;
defm SB      : StoreM32<0x28, "sb", truncstorei8>;
defm SH      : StoreM32<0x29, "sh", truncstorei16_a>;
defm SW      : StoreM32<0x2b, "sw", store_a>;

///  unaligned
defm ULH     : LoadM32<0x21, "ulh",  sextloadi16_u, 1>;
defm ULHu    : LoadM32<0x25, "ulhu", zextloadi16_u, 1>;
defm ULW     : LoadM32<0x23, "ulw",  load_u, 1>;
defm USH     : StoreM32<0x29, "ush", truncstorei16_u, 1>;
defm USW     : StoreM32<0x2b, "usw", store_u, 1>;

let hasSideEffects = 1 in
def SYNC : MipsInst<(outs), (ins i32imm:$stype), "sync $stype",
                    [(MipsSync imm:$stype)], NoItinerary, FrmOther>
{
  bits<5> stype;
  let Opcode = 0;
  let Inst{25-11} = 0;
  let Inst{10-6} = stype;
  let Inst{5-0} = 15;
}

/// Load-linked, Store-conditional
let mayLoad = 1 in
  def LL    : FMem<0x30, (outs CPURegs:$rt), (ins mem:$addr),
              "ll\t$rt, $addr", [], IILoad>;
let mayStore = 1, Constraints = "$rt = $dst" in
  def SC    : FMem<0x38, (outs CPURegs:$dst), (ins CPURegs:$rt, mem:$addr),
              "sc\t$rt, $addr", [], IIStore>;

/// Jump and Branch Instructions
def J       : JumpFJ<0x02, "j">;
let isIndirectBranch = 1 in
  def JR      : JumpFR<0x00, 0x08, "jr">;
def JAL     : JumpLink<0x03, "jal">;
def JALR    : JumpLinkReg<0x00, 0x09, "jalr">;
def BEQ     : CBranch<0x04, "beq", seteq, CPURegs>;
def BNE     : CBranch<0x05, "bne", setne, CPURegs>;
def BGEZ    : CBranchZero<0x01, 1, "bgez", setge, CPURegs>;
def BGTZ    : CBranchZero<0x07, 0, "bgtz", setgt, CPURegs>;
def BLEZ    : CBranchZero<0x06, 0, "blez", setle, CPURegs>;
def BLTZ    : CBranchZero<0x01, 0, "bltz", setlt, CPURegs>;

let rt=0x11 in
  def BGEZAL  : BranchLink<"bgezal">;
let rt=0x10 in
  def BLTZAL  : BranchLink<"bltzal">;

let isReturn=1, isTerminator=1, hasDelaySlot=1,
    isBarrier=1, hasCtrlDep=1, rd=0, rt=0, shamt=0 in
  def RET : FR <0x00, 0x08, (outs), (ins CPURegs:$target),
                "jr\t$target", [(MipsRet CPURegs:$target)], IIBranch>;

/// Multiply and Divide Instructions.
def MULT    : Mul<0x18, "mult", IIImul>;
def MULTu   : Mul<0x19, "multu", IIImul>;
def SDIV    : Div<MipsDivRem, 0x1a, "div", IIIdiv>;
def UDIV    : Div<MipsDivRemU, 0x1b, "divu", IIIdiv>;

let Defs = [HI] in
  def MTHI  : MoveToLOHI<0x11, "mthi">;
let Defs = [LO] in
  def MTLO  : MoveToLOHI<0x13, "mtlo">;

let Uses = [HI] in
  def MFHI  : MoveFromLOHI<0x10, "mfhi">;
let Uses = [LO] in
  def MFLO  : MoveFromLOHI<0x12, "mflo">;

/// Sign Ext In Register Instructions.
def SEB : SignExtInReg<0x10, "seb", i8>;
def SEH : SignExtInReg<0x18, "seh", i16>;

/// Count Leading
def CLZ : CountLeading<0x20, "clz",
                       [(set CPURegs:$rd, (ctlz CPURegs:$rs))]>;
def CLO : CountLeading<0x21, "clo",
                       [(set CPURegs:$rd, (ctlz (not CPURegs:$rs)))]>;

/// Byte Swap
def WSBW : ByteSwap<0x20, 0x2, "wsbw">;

// Conditional moves:
// These instructions are expanded in
// MipsISelLowering::EmitInstrWithCustomInserter if target does not have
// conditional move instructions.
// flag:int, data:int
class CondMovIntInt<bits<6> funct, string instr_asm> :
  FR<0, funct, (outs CPURegs:$rd),
     (ins CPURegs:$rs, CPURegs:$rt, CPURegs:$F),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"), [], NoItinerary> {
  let shamt = 0;
  let usesCustomInserter = 1;
  let Constraints = "$F = $rd";
}

def MOVZ_I : CondMovIntInt<0x0a, "movz">;
def MOVN_I : CondMovIntInt<0x0b, "movn">;

/// No operation
let addr=0 in
  def NOP   : FJ<0, (outs), (ins), "nop", [], IIAlu>;

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched. It's similar to Sparc LEA_ADDRi
def LEA_ADDiu : EffectiveAddress<"addiu\t$rt, $addr">;

// DynAlloc node points to dynamically allocated stack space.
// $sp is added to the list of implicitly used registers to prevent dead code
// elimination from removing instructions that modify $sp.
let Uses = [SP] in
def DynAlloc : EffectiveAddress<"addiu\t$rt, $addr">;

// MADD*/MSUB*
def MADD  : MArithR<0, "madd", MipsMAdd, 1>;
def MADDU : MArithR<1, "maddu", MipsMAddu, 1>;
def MSUB  : MArithR<4, "msub", MipsMSub>;
def MSUBU : MArithR<5, "msubu", MipsMSubu>;

// MUL is a assembly macro in the current used ISAs. In recent ISA's
// it is a real instruction.
def MUL   : ArithLogicR<0x1c, 0x02, "mul", mul, IIImul, CPURegs, 1>,
            Requires<[HasMips32]>;

def RDHWR : ReadHardware;

def EXT : ExtIns<0, "ext", (outs CPURegs:$rt),
                 (ins CPURegs:$rs, uimm16:$pos, size_ext:$sz),
                 [(set CPURegs:$rt,
                   (MipsExt CPURegs:$rs, immZExt5:$pos, immZExt5:$sz))],
                 NoItinerary>;

let Constraints = "$src = $rt" in
def INS : ExtIns<4, "ins", (outs CPURegs:$rt),
                 (ins CPURegs:$rs, uimm16:$pos, size_ins:$sz, CPURegs:$src),
                 [(set CPURegs:$rt,
                   (MipsIns CPURegs:$rs, immZExt5:$pos, immZExt5:$sz,
                    CPURegs:$src))],
                 NoItinerary>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : Pat<(i32 immSExt16:$in),
          (ADDiu ZERO, imm:$in)>;
def : Pat<(i32 immZExt16:$in),
          (ORi ZERO, imm:$in)>;

// Arbitrary immediates
def : Pat<(i32 imm:$imm),
          (ORi (LUi (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Carry patterns
def : Pat<(subc CPURegs:$lhs, CPURegs:$rhs),
          (SUBu CPURegs:$lhs, CPURegs:$rhs)>;
def : Pat<(addc CPURegs:$lhs, CPURegs:$rhs),
          (ADDu CPURegs:$lhs, CPURegs:$rhs)>;
def : Pat<(addc  CPURegs:$src, immSExt16:$imm),
          (ADDiu CPURegs:$src, imm:$imm)>;

// Call
def : Pat<(MipsJmpLink (i32 tglobaladdr:$dst)),
          (JAL tglobaladdr:$dst)>;
def : Pat<(MipsJmpLink (i32 texternalsym:$dst)),
          (JAL texternalsym:$dst)>;
//def : Pat<(MipsJmpLink CPURegs:$dst),
//          (JALR CPURegs:$dst)>;

// hi/lo relocs
def : Pat<(MipsHi tglobaladdr:$in), (LUi tglobaladdr:$in)>;
def : Pat<(MipsHi tblockaddress:$in), (LUi tblockaddress:$in)>;
def : Pat<(MipsLo tglobaladdr:$in), (ADDiu ZERO, tglobaladdr:$in)>;
def : Pat<(MipsLo tblockaddress:$in), (ADDiu ZERO, tblockaddress:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tglobaladdr:$lo)),
          (ADDiu CPURegs:$hi, tglobaladdr:$lo)>;
def : Pat<(add CPURegs:$hi, (MipsLo tblockaddress:$lo)),
          (ADDiu CPURegs:$hi, tblockaddress:$lo)>;

def : Pat<(MipsHi tjumptable:$in), (LUi tjumptable:$in)>;
def : Pat<(MipsLo tjumptable:$in), (ADDiu ZERO, tjumptable:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tjumptable:$lo)),
          (ADDiu CPURegs:$hi, tjumptable:$lo)>;

def : Pat<(MipsHi tconstpool:$in), (LUi tconstpool:$in)>;
def : Pat<(MipsLo tconstpool:$in), (ADDiu ZERO, tconstpool:$in)>;
def : Pat<(add CPURegs:$hi, (MipsLo tconstpool:$lo)),
          (ADDiu CPURegs:$hi, tconstpool:$lo)>;

// gp_rel relocs
def : Pat<(add CPURegs:$gp, (MipsGPRel tglobaladdr:$in)),
          (ADDiu CPURegs:$gp, tglobaladdr:$in)>;
def : Pat<(add CPURegs:$gp, (MipsGPRel tconstpool:$in)),
          (ADDiu CPURegs:$gp, tconstpool:$in)>;

// tlsgd
def : Pat<(add CPURegs:$gp, (MipsTlsGd tglobaltlsaddr:$in)),
          (ADDiu CPURegs:$gp, tglobaltlsaddr:$in)>;

// tprel hi/lo
def : Pat<(MipsTprelHi tglobaltlsaddr:$in), (LUi tglobaltlsaddr:$in)>;
def : Pat<(MipsTprelLo tglobaltlsaddr:$in), (ADDiu ZERO, tglobaltlsaddr:$in)>;
def : Pat<(add CPURegs:$hi, (MipsTprelLo tglobaltlsaddr:$lo)),
          (ADDiu CPURegs:$hi, tglobaltlsaddr:$lo)>;

// wrapper_pic
class WrapperPICPat<SDNode node>:
      Pat<(MipsWrapperPIC node:$in),
          (ADDiu GP, node:$in)>;

def : WrapperPICPat<tglobaladdr>;
def : WrapperPICPat<tconstpool>;
def : WrapperPICPat<texternalsym>;
def : WrapperPICPat<tblockaddress>;
def : WrapperPICPat<tjumptable>;

// Mips does not have "not", so we expand our way
def : Pat<(not CPURegs:$in),
          (NOR CPURegs:$in, ZERO)>;

// extended load and stores
def : Pat<(extloadi1  addr:$src), (LBu addr:$src)>;
def : Pat<(extloadi8  addr:$src), (LBu addr:$src)>;
def : Pat<(extloadi16_a addr:$src), (LHu addr:$src)>;
def : Pat<(extloadi16_u addr:$src), (ULHu addr:$src)>;

// peepholes
def : Pat<(store (i32 0), addr:$dst), (SW ZERO, addr:$dst)>;

// brcond patterns
multiclass BrcondPats<RegisterClass RC, Instruction BEQOp, Instruction BNEOp,
                      Instruction SLTOp, Instruction SLTuOp, Instruction SLTiOp,
                      Instruction SLTiuOp, Register ZEROReg> {
def : Pat<(brcond (i32 (setne RC:$lhs, 0)), bb:$dst),
          (BNEOp RC:$lhs, ZEROReg, bb:$dst)>;
def : Pat<(brcond (i32 (seteq RC:$lhs, 0)), bb:$dst),
          (BEQOp RC:$lhs, ZEROReg, bb:$dst)>;

def : Pat<(brcond (i32 (setge RC:$lhs, RC:$rhs)), bb:$dst),
          (BEQ (SLTOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (i32 (setuge RC:$lhs, RC:$rhs)), bb:$dst),
          (BEQ (SLTuOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (i32 (setge RC:$lhs, immSExt16:$rhs)), bb:$dst),
          (BEQ (SLTiOp RC:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;
def : Pat<(brcond (i32 (setuge RC:$lhs, immSExt16:$rhs)), bb:$dst),
          (BEQ (SLTiuOp RC:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;

def : Pat<(brcond (i32 (setle RC:$lhs, RC:$rhs)), bb:$dst),
          (BEQ (SLTOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
def : Pat<(brcond (i32 (setule RC:$lhs, RC:$rhs)), bb:$dst),
          (BEQ (SLTuOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;

def : Pat<(brcond RC:$cond, bb:$dst),
          (BNEOp RC:$cond, ZEROReg, bb:$dst)>;
}

defm : BrcondPats<CPURegs, BEQ, BNE, SLT, SLTu, SLTi, SLTiu, ZERO>;

// select patterns
multiclass MovzPats<RegisterClass RC, Instruction MOVZInst> {
  def : Pat<(select (i32 (setge CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLT CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setuge CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTu CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setge CPURegs:$lhs, immSExt16:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTi CPURegs:$lhs, immSExt16:$rhs), RC:$F)>;
  def : Pat<(select (i32 (setuge CPURegs:$lh, immSExt16:$rh)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTiu CPURegs:$lh, immSExt16:$rh), RC:$F)>;
  def : Pat<(select (i32 (setle CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLT CPURegs:$rhs, CPURegs:$lhs), RC:$F)>;
  def : Pat<(select (i32 (setule CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (SLTu CPURegs:$rhs, CPURegs:$lhs), RC:$F)>;
  def : Pat<(select (i32 (seteq CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVZInst RC:$T, (XOR CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select (i32 (seteq CPURegs:$lhs, 0)), RC:$T, RC:$F),
            (MOVZInst RC:$T, CPURegs:$lhs, RC:$F)>;
}

multiclass MovnPats<RegisterClass RC, Instruction MOVNInst> {
  def : Pat<(select (i32 (setne CPURegs:$lhs, CPURegs:$rhs)), RC:$T, RC:$F),
            (MOVNInst RC:$T, (XOR CPURegs:$lhs, CPURegs:$rhs), RC:$F)>;
  def : Pat<(select CPURegs:$cond, RC:$T, RC:$F),
            (MOVNInst RC:$T, CPURegs:$cond, RC:$F)>;
  def : Pat<(select (i32 (setne CPURegs:$lhs, 0)), RC:$T, RC:$F),
            (MOVNInst RC:$T, CPURegs:$lhs, RC:$F)>;
}

defm : MovzPats<CPURegs, MOVZ_I>;
defm : MovnPats<CPURegs, MOVN_I>;

// setcc patterns
multiclass SeteqPats<RegisterClass RC, Instruction SLTiuOp, Instruction XOROp,
                     Instruction SLTuOp, Register ZEROReg> {
  def : Pat<(seteq RC:$lhs, RC:$rhs),
            (SLTiuOp (XOROp RC:$lhs, RC:$rhs), 1)>;
  def : Pat<(setne RC:$lhs, RC:$rhs),
            (SLTuOp ZEROReg, (XOROp RC:$lhs, RC:$rhs))>;
}

multiclass SetlePats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : Pat<(setle RC:$lhs, RC:$rhs),
            (XORi (SLTOp RC:$rhs, RC:$lhs), 1)>;
  def : Pat<(setule RC:$lhs, RC:$rhs),
            (XORi (SLTuOp RC:$rhs, RC:$lhs), 1)>;
}

multiclass SetgtPats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : Pat<(setgt RC:$lhs, RC:$rhs),
            (SLTOp RC:$rhs, RC:$lhs)>;
  def : Pat<(setugt RC:$lhs, RC:$rhs),
            (SLTuOp RC:$rhs, RC:$lhs)>;
}

multiclass SetgePats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : Pat<(setge RC:$lhs, RC:$rhs),
            (XORi (SLTOp RC:$lhs, RC:$rhs), 1)>;
  def : Pat<(setuge RC:$lhs, RC:$rhs),
            (XORi (SLTuOp RC:$lhs, RC:$rhs), 1)>;
}

multiclass SetgeImmPats<RegisterClass RC, Instruction SLTiOp,
                        Instruction SLTiuOp> {
  def : Pat<(setge RC:$lhs, immSExt16:$rhs),
            (XORi (SLTiOp RC:$lhs, immSExt16:$rhs), 1)>;
  def : Pat<(setuge RC:$lhs, immSExt16:$rhs),
            (XORi (SLTiuOp RC:$lhs, immSExt16:$rhs), 1)>;
}

defm : SeteqPats<CPURegs, SLTiu, XOR, SLTu, ZERO>;
defm : SetlePats<CPURegs, SLT, SLTu>;
defm : SetgtPats<CPURegs, SLT, SLTu>;
defm : SetgePats<CPURegs, SLT, SLTu>;
defm : SetgeImmPats<CPURegs, SLTi, SLTiu>;

// select MipsDynAlloc
def : Pat<(MipsDynAlloc addr:$f), (DynAlloc addr:$f)>;

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//

include "MipsInstrFPU.td"
include "Mips64InstrInfo.td"

