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
// Mips profiles and nodes
//===----------------------------------------------------------------------===//

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

def SDT_MipsDynAlloc    : SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>,
                                               SDTCisSameAs<0, 1>]>;
def SDT_Sync             : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

def SDT_Ext : SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                   SDTCisVT<2, i32>, SDTCisSameAs<2, 3>]>;
def SDT_Ins : SDTypeProfile<1, 4, [SDTCisInt<0>, SDTCisSameAs<0, 1>,
                                   SDTCisVT<2, i32>, SDTCisSameAs<2, 3>,
                                   SDTCisSameAs<0, 4>]>;

def SDTMipsLoadLR  : SDTypeProfile<1, 2,
                                   [SDTCisInt<0>, SDTCisPtrTy<1>,
                                    SDTCisSameAs<0, 2>]>;

// Call
def MipsJmpLink : SDNode<"MipsISD::JmpLink",SDT_MipsJmpLink,
                         [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                          SDNPVariadic]>;

// Tail call
def MipsTailCall : SDNode<"MipsISD::TailCall", SDT_MipsJmpLink,
                          [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

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
def MipsRet : SDNode<"MipsISD::Ret", SDTNone, [SDNPHasChain, SDNPOptInGlue]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MipsCallSeqStart,
                           [SDNPHasChain, SDNPSideEffect, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MipsCallSeqEnd,
                           [SDNPHasChain, SDNPSideEffect,
                            SDNPOptInGlue, SDNPOutGlue]>;

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

def MipsWrapper    : SDNode<"MipsISD::Wrapper", SDTIntBinOp>;

// Pointer to dynamically allocated stack area.
def MipsDynAlloc  : SDNode<"MipsISD::DynAlloc", SDT_MipsDynAlloc,
                           [SDNPHasChain, SDNPInGlue]>;

def MipsSync : SDNode<"MipsISD::Sync", SDT_Sync, [SDNPHasChain,SDNPSideEffect]>;

def MipsExt :  SDNode<"MipsISD::Ext", SDT_Ext>;
def MipsIns :  SDNode<"MipsISD::Ins", SDT_Ins>;

def MipsLWL : SDNode<"MipsISD::LWL", SDTMipsLoadLR,
                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def MipsLWR : SDNode<"MipsISD::LWR", SDTMipsLoadLR,
                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def MipsSWL : SDNode<"MipsISD::SWL", SDTStore,
                     [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def MipsSWR : SDNode<"MipsISD::SWR", SDTStore,
                     [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def MipsLDL : SDNode<"MipsISD::LDL", SDTMipsLoadLR,
                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def MipsLDR : SDNode<"MipsISD::LDR", SDTMipsLoadLR,
                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def MipsSDL : SDNode<"MipsISD::SDL", SDTStore,
                     [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def MipsSDR : SDNode<"MipsISD::SDR", SDTStore,
                     [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// Mips Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//
def HasSEInReg  :     Predicate<"Subtarget.hasSEInReg()">,
                      AssemblerPredicate<"FeatureSEInReg">;
def HasBitCount :     Predicate<"Subtarget.hasBitCount()">,
                      AssemblerPredicate<"FeatureBitCount">;
def HasSwap     :     Predicate<"Subtarget.hasSwap()">,
                      AssemblerPredicate<"FeatureSwap">;
def HasCondMov  :     Predicate<"Subtarget.hasCondMov()">,
                      AssemblerPredicate<"FeatureCondMov">;
def HasMips32    :    Predicate<"Subtarget.hasMips32()">,
                      AssemblerPredicate<"FeatureMips32">;
def HasMips32r2  :    Predicate<"Subtarget.hasMips32r2()">,
                      AssemblerPredicate<"FeatureMips32r2">;
def HasMips64    :    Predicate<"Subtarget.hasMips64()">,
                      AssemblerPredicate<"FeatureMips64">;
def HasMips32r2Or64 : Predicate<"Subtarget.hasMips32r2Or64()">,
                      AssemblerPredicate<"FeatureMips32r2,FeatureMips64">;
def NotMips64    :    Predicate<"!Subtarget.hasMips64()">,
                      AssemblerPredicate<"!FeatureMips64">;
def HasMips64r2  :    Predicate<"Subtarget.hasMips64r2()">,
                      AssemblerPredicate<"FeatureMips64r2">;
def IsN64       :     Predicate<"Subtarget.isABI_N64()">,
                      AssemblerPredicate<"FeatureN64">;
def NotN64      :     Predicate<"!Subtarget.isABI_N64()">,
                      AssemblerPredicate<"!FeatureN64">;
def InMips16Mode :    Predicate<"Subtarget.inMips16Mode()">,
                      AssemblerPredicate<"FeatureMips16">;
def RelocStatic :     Predicate<"TM.getRelocationModel() == Reloc::Static">,
                      AssemblerPredicate<"FeatureMips32">;
def RelocPIC    :     Predicate<"TM.getRelocationModel() == Reloc::PIC_">,
                      AssemblerPredicate<"FeatureMips32">;
def NoNaNsFPMath :    Predicate<"TM.Options.NoNaNsFPMath">,
                      AssemblerPredicate<"FeatureMips32">;
def HasStandardEncoding : Predicate<"Subtarget.hasStandardEncoding()">,
                          AssemblerPredicate<"!FeatureMips16">;

class MipsPat<dag pattern, dag result> : Pat<pattern, result> {
  let Predicates = [HasStandardEncoding];
}

class IsBranch {
  bit isBranch = 1;
}

class IsReturn {
  bit isReturn = 1;
}

class IsCall {
  bit isCall = 1;
}

class IsTailCall {
  bit isCall = 1;
  bit isTerminator = 1;
  bit isReturn = 1;
  bit isBarrier = 1;
  bit hasExtraSrcRegAllocReq = 1;
  bit isCodeGenOnly = 1;
}

class IsAsCheapAsAMove {
  bit isAsCheapAsAMove = 1;
}

class NeverHasSideEffects {
  bit neverHasSideEffects = 1;
}

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "MipsInstrFormats.td"

//===----------------------------------------------------------------------===//
// Mips Operand, Complex Patterns and Transformations Definitions.
//===----------------------------------------------------------------------===//

// Instruction operand types
def jmptarget   : Operand<OtherVT> {
  let EncoderMethod = "getJumpTargetOpValue";
}
def brtarget    : Operand<OtherVT> {
  let EncoderMethod = "getBranchTargetOpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget";
}
def calltarget  : Operand<iPTR> {
  let EncoderMethod = "getJumpTargetOpValue";
}
def calltarget64: Operand<i64>;
def simm16      : Operand<i32> {
  let DecoderMethod= "DecodeSimm16";
}
def simm16_64   : Operand<i64>;
def shamt       : Operand<i32>;

// Unsigned Operand
def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def MipsMemAsmOperand : AsmOperandClass {
  let Name = "Mem";
  let ParserMethod = "parseMemOperand";
}

// Address operand
def mem : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPURegs, simm16);
  let EncoderMethod = "getMemEncoding";
  let ParserMatchClass = MipsMemAsmOperand;
}

def mem64 : Operand<i64> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops CPU64Regs, simm16_64);
  let EncoderMethod = "getMemEncoding";
  let ParserMatchClass = MipsMemAsmOperand;
}

def mem_ea : Operand<i32> {
  let PrintMethod = "printMemOperandEA";
  let MIOperandInfo = (ops CPURegs, simm16);
  let EncoderMethod = "getMemEncoding";
}

def mem_ea_64 : Operand<i64> {
  let PrintMethod = "printMemOperandEA";
  let MIOperandInfo = (ops CPU64Regs, simm16_64);
  let EncoderMethod = "getMemEncoding";
}

// size operand of ext instruction
def size_ext : Operand<i32> {
  let EncoderMethod = "getSizeExtEncoding";
  let DecoderMethod = "DecodeExtSize";
}

// size operand of ins instruction
def size_ins : Operand<i32> {
  let EncoderMethod = "getSizeInsEncoding";
  let DecoderMethod = "DecodeInsSize";
}

// Transformation Function - get the lower 16 bits.
def LO16 : SDNodeXForm<imm, [{
  return getImm(N, N->getZExtValue() & 0xFFFF);
}]>;

// Transformation Function - get the higher 16 bits.
def HI16 : SDNodeXForm<imm, [{
  return getImm(N, (N->getZExtValue() >> 16) & 0xFFFF);
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

// Immediate can be loaded with LUi (32-bit int with lower 16-bit cleared).
def immLow16Zero : PatLeaf<(imm), [{
  int64_t Val = N->getSExtValue();
  return isInt<32>(Val) && !(Val & 0xffff);
}]>;

// shamt field must fit in 5 bits.
def immZExt5 : ImmLeaf<i32, [{return Imm == (Imm & 0x1f);}]>;

// Mips Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr :
  ComplexPattern<iPTR, 2, "SelectAddr", [frameindex], [SDNPWantParent]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

/// Move Control Registers From/To CPU Registers
def MFC0_3OP  : MFC3OP<0x10, 0, (outs CPURegs:$rt),
                       (ins CPURegs:$rd, uimm16:$sel),"mfc0\t$rt, $rd, $sel">;
def : InstAlias<"mfc0 $rt, $rd", (MFC0_3OP CPURegs:$rt, CPURegs:$rd, 0)>;

def MTC0_3OP  : MFC3OP<0x10, 4, (outs CPURegs:$rd, uimm16:$sel),
                       (ins CPURegs:$rt),"mtc0\t$rt, $rd, $sel">;
def : InstAlias<"mtc0 $rt, $rd", (MTC0_3OP CPURegs:$rd, 0, CPURegs:$rt)>;

def MFC2_3OP  : MFC3OP<0x12, 0, (outs CPURegs:$rt),
                       (ins CPURegs:$rd, uimm16:$sel),"mfc2\t$rt, $rd, $sel">;
def : InstAlias<"mfc2 $rt, $rd", (MFC2_3OP CPURegs:$rt, CPURegs:$rd, 0)>;

def MTC2_3OP  : MFC3OP<0x12, 4, (outs CPURegs:$rd, uimm16:$sel),
                       (ins CPURegs:$rt),"mtc2\t$rt, $rd, $sel">;
def : InstAlias<"mtc2 $rt, $rd", (MTC2_3OP CPURegs:$rd, 0, CPURegs:$rt)>;

// Arithmetic and logical instructions with 3 register operands.
class ArithLogicR<bits<6> op, bits<6> func, string instr_asm, SDNode OpNode,
                  InstrItinClass itin, RegisterClass RC, bit isComm = 0>:
  FR<op, func, (outs RC:$rd), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rs, $rt"),
     [(set RC:$rd, (OpNode RC:$rs, RC:$rt))], itin> {
  let shamt = 0;
  let isCommutable = isComm;
  let isReMaterializable = 1;
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
     [(set RC:$rt, (OpNode RC:$rs, imm_type:$imm16))], IIAlu> {
  let isReMaterializable = 1;
}

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
class shift_rotate_imm<bits<6> func, bits<5> isRotate, string instr_asm,
                       SDNode OpNode, PatFrag PF, Operand ImmOpnd,
                       RegisterClass RC>:
  FR<0x00, func, (outs RC:$rd), (ins RC:$rt, ImmOpnd:$shamt),
     !strconcat(instr_asm, "\t$rd, $rt, $shamt"),
     [(set RC:$rd, (OpNode RC:$rt, PF:$shamt))], IIAlu> {
  let rs = isRotate;
}

// 32-bit shift instructions.
class shift_rotate_imm32<bits<6> func, bits<5> isRotate, string instr_asm,
                         SDNode OpNode>:
  shift_rotate_imm<func, isRotate, instr_asm, OpNode, immZExt5, shamt, CPURegs>;

class shift_rotate_reg<bits<6> func, bits<5> isRotate, string instr_asm,
                       SDNode OpNode, RegisterClass RC>:
  FR<0x00, func, (outs RC:$rd), (ins CPURegs:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rt, $rs"),
     [(set RC:$rd, (OpNode RC:$rt, CPURegs:$rs))], IIAlu> {
  let shamt = isRotate;
}

// Load Upper Imediate
class LoadUpper<bits<6> op, string instr_asm, RegisterClass RC, Operand Imm>:
  FI<op, (outs RC:$rt), (ins Imm:$imm16),
     !strconcat(instr_asm, "\t$rt, $imm16"), [], IIAlu>, IsAsCheapAsAMove {
  let rs = 0;
  let neverHasSideEffects = 1;
  let isReMaterializable = 1;
}

class FMem<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
          InstrItinClass itin>: FFI<op, outs, ins, asmstr, pattern> {
  bits<21> addr;
  let Inst{25-21} = addr{20-16};
  let Inst{15-0}  = addr{15-0};
  let DecoderMethod = "DecodeMem";
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
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : LoadM<op, instr_asm, OpNode, CPURegs, mem64, Pseudo>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 64-bit load.
multiclass LoadM64<bits<6> op, string instr_asm, PatFrag OpNode,
                   bit Pseudo = 0> {
  def #NAME# : LoadM<op, instr_asm, OpNode, CPU64Regs, mem, Pseudo>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : LoadM<op, instr_asm, OpNode, CPU64Regs, mem64, Pseudo>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 32-bit store.
multiclass StoreM32<bits<6> op, string instr_asm, PatFrag OpNode,
                    bit Pseudo = 0> {
  def #NAME# : StoreM<op, instr_asm, OpNode, CPURegs, mem, Pseudo>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : StoreM<op, instr_asm, OpNode, CPURegs, mem64, Pseudo>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 64-bit store.
multiclass StoreM64<bits<6> op, string instr_asm, PatFrag OpNode,
                    bit Pseudo = 0> {
  def #NAME# : StoreM<op, instr_asm, OpNode, CPU64Regs, mem, Pseudo>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : StoreM<op, instr_asm, OpNode, CPU64Regs, mem64, Pseudo>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// Load/Store Left/Right
let canFoldAsLoad = 1 in
class LoadLeftRight<bits<6> op, string instr_asm, SDNode OpNode,
                    RegisterClass RC, Operand MemOpnd> :
  FMem<op, (outs RC:$rt), (ins MemOpnd:$addr, RC:$src),
       !strconcat(instr_asm, "\t$rt, $addr"),
       [(set RC:$rt, (OpNode addr:$addr, RC:$src))], IILoad> {
  string Constraints = "$src = $rt";
}

class StoreLeftRight<bits<6> op, string instr_asm, SDNode OpNode,
                     RegisterClass RC, Operand MemOpnd>:
  FMem<op, (outs), (ins RC:$rt, MemOpnd:$addr),
       !strconcat(instr_asm, "\t$rt, $addr"), [(OpNode RC:$rt, addr:$addr)],
       IIStore>;

// 32-bit load left/right.
multiclass LoadLeftRightM32<bits<6> op, string instr_asm, SDNode OpNode> {
  def #NAME# : LoadLeftRight<op, instr_asm, OpNode, CPURegs, mem>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : LoadLeftRight<op, instr_asm, OpNode, CPURegs, mem64>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 64-bit load left/right.
multiclass LoadLeftRightM64<bits<6> op, string instr_asm, SDNode OpNode> {
  def #NAME# : LoadLeftRight<op, instr_asm, OpNode, CPU64Regs, mem>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : LoadLeftRight<op, instr_asm, OpNode, CPU64Regs, mem64>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 32-bit store left/right.
multiclass StoreLeftRightM32<bits<6> op, string instr_asm, SDNode OpNode> {
  def #NAME# : StoreLeftRight<op, instr_asm, OpNode, CPURegs, mem>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : StoreLeftRight<op, instr_asm, OpNode, CPURegs, mem64>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// 64-bit store left/right.
multiclass StoreLeftRightM64<bits<6> op, string instr_asm, SDNode OpNode> {
  def #NAME# : StoreLeftRight<op, instr_asm, OpNode, CPU64Regs, mem>,
               Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : StoreLeftRight<op, instr_asm, OpNode, CPU64Regs, mem64>,
               Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
    let isCodeGenOnly = 1;
  }
}

// Conditional Branch
class CBranch<bits<6> op, string instr_asm, PatFrag cond_op, RegisterClass RC>:
  BranchBase<op, (outs), (ins RC:$rs, RC:$rt, brtarget:$imm16),
             !strconcat(instr_asm, "\t$rs, $rt, $imm16"),
             [(brcond (i32 (cond_op RC:$rs, RC:$rt)), bb:$imm16)], IIBranch> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
}

class CBranchZero<bits<6> op, bits<5> _rt, string instr_asm, PatFrag cond_op,
                  RegisterClass RC>:
  BranchBase<op, (outs), (ins RC:$rs, brtarget:$imm16),
             !strconcat(instr_asm, "\t$rs, $imm16"),
             [(brcond (i32 (cond_op RC:$rs, 0)), bb:$imm16)], IIBranch> {
  let rt = _rt;
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
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

// Jump
class JumpFJ<bits<6> op, DAGOperand opnd, string instr_asm,
             SDPatternOperator operator, SDPatternOperator targetoperator>:
  FJ<op, (outs), (ins opnd:$target), !strconcat(instr_asm, "\t$target"),
     [(operator targetoperator:$target)], IIBranch> {
  let isTerminator=1;
  let isBarrier=1;
  let hasDelaySlot = 1;
  let DecoderMethod = "DecodeJumpTarget";
  let Defs = [AT];
}

// Unconditional branch
class UncondBranch<bits<6> op, string instr_asm>:
  BranchBase<op, (outs), (ins brtarget:$imm16),
             !strconcat(instr_asm, "\t$imm16"), [(br bb:$imm16)], IIBranch> {
  let rs = 0;
  let rt = 0;
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
  let hasDelaySlot = 1;
  let Predicates = [RelocPIC, HasStandardEncoding];
  let Defs = [AT];
}

// Base class for indirect branch and return instruction classes.
let isTerminator=1, isBarrier=1, hasDelaySlot = 1 in
class JumpFR<RegisterClass RC, SDPatternOperator operator = null_frag>:
  FR<0, 0x8, (outs), (ins RC:$rs), "jr\t$rs", [(operator RC:$rs)], IIBranch> {
  let rt = 0;
  let rd = 0;
  let shamt = 0;
}

// Indirect branch
class IndirectBranch<RegisterClass RC>: JumpFR<RC, brind> {
  let isBranch = 1;
  let isIndirectBranch = 1;
}

// Return instruction
class RetBase<RegisterClass RC>: JumpFR<RC> {
  let isReturn = 1;
  let isCodeGenOnly = 1;
  let hasCtrlDep = 1;
  let hasExtraSrcRegAllocReq = 1;
}

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1, Defs = [RA] in {
  class JumpLink<bits<6> op, string instr_asm>:
    FJ<op, (outs), (ins calltarget:$target),
       !strconcat(instr_asm, "\t$target"), [(MipsJmpLink imm:$target)],
       IIBranch> {
       let DecoderMethod = "DecodeJumpTarget";
       }

  class JumpLinkReg<bits<6> op, bits<6> func, string instr_asm,
                    RegisterClass RC>:
    FR<op, func, (outs), (ins RC:$rs),
       !strconcat(instr_asm, "\t$rs"), [(MipsJmpLink RC:$rs)], IIBranch> {
    let rt = 0;
    let rd = 31;
    let shamt = 0;
  }

  class BranchLink<string instr_asm, bits<5> _rt, RegisterClass RC>:
    FI<0x1, (outs), (ins RC:$rs, brtarget:$imm16),
       !strconcat(instr_asm, "\t$rs, $imm16"), [], IIBranch> {
    let rt = _rt;
  }
}

// Mul, Div
class Mult<bits<6> func, string instr_asm, InstrItinClass itin,
           RegisterClass RC, list<Register> DefRegs>:
  FR<0x00, func, (outs), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$rs, $rt"), [], itin> {
  let rd = 0;
  let shamt = 0;
  let isCommutable = 1;
  let Defs = DefRegs;
  let neverHasSideEffects = 1;
}

class Mult32<bits<6> func, string instr_asm, InstrItinClass itin>:
  Mult<func, instr_asm, itin, CPURegs, [HI, LO]>;

class Div<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin,
          RegisterClass RC, list<Register> DefRegs>:
  FR<0x00, func, (outs), (ins RC:$rs, RC:$rt),
     !strconcat(instr_asm, "\t$$zero, $rs, $rt"),
     [(op RC:$rs, RC:$rt)], itin> {
  let rd = 0;
  let shamt = 0;
  let Defs = DefRegs;
}

class Div32<SDNode op, bits<6> func, string instr_asm, InstrItinClass itin>:
  Div<op, func, instr_asm, itin, CPURegs, [HI, LO]>;

// Move from Hi/Lo
class MoveFromLOHI<bits<6> func, string instr_asm, RegisterClass RC,
                   list<Register> UseRegs>:
  FR<0x00, func, (outs RC:$rd), (ins),
     !strconcat(instr_asm, "\t$rd"), [], IIHiLo> {
  let rs = 0;
  let rt = 0;
  let shamt = 0;
  let Uses = UseRegs;
  let neverHasSideEffects = 1;
}

class MoveToLOHI<bits<6> func, string instr_asm, RegisterClass RC,
                 list<Register> DefRegs>:
  FR<0x00, func, (outs), (ins RC:$rs),
     !strconcat(instr_asm, "\t$rs"), [], IIHiLo> {
  let rt = 0;
  let rd = 0;
  let shamt = 0;
  let Defs = DefRegs;
  let neverHasSideEffects = 1;
}

class EffectiveAddress<bits<6> opc, string instr_asm, RegisterClass RC, Operand Mem> :
  FMem<opc, (outs RC:$rt), (ins Mem:$addr),
     instr_asm, [(set RC:$rt, addr:$addr)], IIAlu> {
 let isCodeGenOnly = 1;
}

// Count Leading Ones/Zeros in Word
class CountLeading0<bits<6> func, string instr_asm, RegisterClass RC>:
  FR<0x1c, func, (outs RC:$rd), (ins RC:$rs),
     !strconcat(instr_asm, "\t$rd, $rs"),
     [(set RC:$rd, (ctlz RC:$rs))], IIAlu>,
     Requires<[HasBitCount, HasStandardEncoding]> {
  let shamt = 0;
  let rt = rd;
}

class CountLeading1<bits<6> func, string instr_asm, RegisterClass RC>:
  FR<0x1c, func, (outs RC:$rd), (ins RC:$rs),
     !strconcat(instr_asm, "\t$rd, $rs"),
     [(set RC:$rd, (ctlz (not RC:$rs)))], IIAlu>,
     Requires<[HasBitCount, HasStandardEncoding]> {
  let shamt = 0;
  let rt = rd;
}

// Sign Extend in Register.
class SignExtInReg<bits<5> sa, string instr_asm, ValueType vt,
                   RegisterClass RC>:
  FR<0x1f, 0x20, (outs RC:$rd), (ins RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rt"),
     [(set RC:$rd, (sext_inreg RC:$rt, vt))], NoItinerary> {
  let rs = 0;
  let shamt = sa;
  let Predicates = [HasSEInReg, HasStandardEncoding];
}

// Subword Swap
class SubwordSwap<bits<6> func, bits<5> sa, string instr_asm, RegisterClass RC>:
  FR<0x1f, func, (outs RC:$rd), (ins RC:$rt),
     !strconcat(instr_asm, "\t$rd, $rt"), [], NoItinerary> {
  let rs = 0;
  let shamt = sa;
  let Predicates = [HasSwap, HasStandardEncoding];
  let neverHasSideEffects = 1;
}

// Read Hardware
class ReadHardware<RegisterClass CPURegClass, RegisterClass HWRegClass>
  : FR<0x1f, 0x3b, (outs CPURegClass:$rt), (ins HWRegClass:$rd),
       "rdhwr\t$rt, $rd", [], IIAlu> {
  let rs = 0;
  let shamt = 0;
}

// Ext and Ins
class ExtBase<bits<6> _funct, string instr_asm, RegisterClass RC>:
  FR<0x1f, _funct, (outs RC:$rt), (ins RC:$rs, uimm16:$pos, size_ext:$sz),
     !strconcat(instr_asm, " $rt, $rs, $pos, $sz"),
     [(set RC:$rt, (MipsExt RC:$rs, imm:$pos, imm:$sz))], NoItinerary> {
  bits<5> pos;
  bits<5> sz;
  let rd = sz;
  let shamt = pos;
  let Predicates = [HasMips32r2, HasStandardEncoding];
}

class InsBase<bits<6> _funct, string instr_asm, RegisterClass RC>:
  FR<0x1f, _funct, (outs RC:$rt),
     (ins RC:$rs, uimm16:$pos, size_ins:$sz, RC:$src),
     !strconcat(instr_asm, " $rt, $rs, $pos, $sz"),
     [(set RC:$rt, (MipsIns RC:$rs, imm:$pos, imm:$sz, RC:$src))],
     NoItinerary> {
  bits<5> pos;
  bits<5> sz;
  let rd = sz;
  let shamt = pos;
  let Predicates = [HasMips32r2, HasStandardEncoding];
  let Constraints = "$src = $rt";
}

// Atomic instructions with 2 source operands (ATOMIC_SWAP & ATOMIC_LOAD_*).
class Atomic2Ops<PatFrag Op, string Opstr, RegisterClass DRC,
                 RegisterClass PRC> :
  PseudoSE<(outs DRC:$dst), (ins PRC:$ptr, DRC:$incr),
           !strconcat("atomic_", Opstr, "\t$dst, $ptr, $incr"),
           [(set DRC:$dst, (Op PRC:$ptr, DRC:$incr))]>;

multiclass Atomic2Ops32<PatFrag Op, string Opstr> {
  def #NAME# : Atomic2Ops<Op, Opstr, CPURegs, CPURegs>,
                          Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : Atomic2Ops<Op, Opstr, CPURegs, CPU64Regs>,
                          Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}

// Atomic Compare & Swap.
class AtomicCmpSwap<PatFrag Op, string Width, RegisterClass DRC,
                    RegisterClass PRC> :
  PseudoSE<(outs DRC:$dst), (ins PRC:$ptr, DRC:$cmp, DRC:$swap),
           !strconcat("atomic_cmp_swap_", Width, "\t$dst, $ptr, $cmp, $swap"),
           [(set DRC:$dst, (Op PRC:$ptr, DRC:$cmp, DRC:$swap))]>;

multiclass AtomicCmpSwap32<PatFrag Op, string Width>  {
  def #NAME# : AtomicCmpSwap<Op, Width, CPURegs, CPURegs>,
                             Requires<[NotN64, HasStandardEncoding]>;
  def _P8    : AtomicCmpSwap<Op, Width, CPURegs, CPU64Regs>,
                             Requires<[IsN64, HasStandardEncoding]> {
    let DecoderNamespace = "Mips64";
  }
}

class LLBase<bits<6> Opc, string opstring, RegisterClass RC, Operand Mem> :
  FMem<Opc, (outs RC:$rt), (ins Mem:$addr),
       !strconcat(opstring, "\t$rt, $addr"), [], IILoad> {
  let mayLoad = 1;
}

class SCBase<bits<6> Opc, string opstring, RegisterClass RC, Operand Mem> :
  FMem<Opc, (outs RC:$dst), (ins RC:$rt, Mem:$addr),
       !strconcat(opstring, "\t$rt, $addr"), [], IIStore> {
  let mayStore = 1;
  let Constraints = "$rt = $dst";
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// Return RA.
let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1 in
def RetRA : PseudoSE<(outs), (ins), "", [(MipsRet)]>;

let Defs = [SP], Uses = [SP], hasSideEffects = 1 in {
def ADJCALLSTACKDOWN : MipsPseudo<(outs), (ins i32imm:$amt),
                                  "!ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MipsPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                  "!ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

// When handling PIC code the assembler needs .cpload and .cprestore
// directives. If the real instructions corresponding these directives
// are used, we have the same behavior, but get also a bunch of warnings
// from the assembler.
let neverHasSideEffects = 1 in
def CPRESTORE : PseudoSE<(outs), (ins i32imm:$loc, CPURegs:$gp),
                         ".cprestore\t$loc", []>;

let usesCustomInserter = 1 in {
  defm ATOMIC_LOAD_ADD_I8   : Atomic2Ops32<atomic_load_add_8, "load_add_8">;
  defm ATOMIC_LOAD_ADD_I16  : Atomic2Ops32<atomic_load_add_16, "load_add_16">;
  defm ATOMIC_LOAD_ADD_I32  : Atomic2Ops32<atomic_load_add_32, "load_add_32">;
  defm ATOMIC_LOAD_SUB_I8   : Atomic2Ops32<atomic_load_sub_8, "load_sub_8">;
  defm ATOMIC_LOAD_SUB_I16  : Atomic2Ops32<atomic_load_sub_16, "load_sub_16">;
  defm ATOMIC_LOAD_SUB_I32  : Atomic2Ops32<atomic_load_sub_32, "load_sub_32">;
  defm ATOMIC_LOAD_AND_I8   : Atomic2Ops32<atomic_load_and_8, "load_and_8">;
  defm ATOMIC_LOAD_AND_I16  : Atomic2Ops32<atomic_load_and_16, "load_and_16">;
  defm ATOMIC_LOAD_AND_I32  : Atomic2Ops32<atomic_load_and_32, "load_and_32">;
  defm ATOMIC_LOAD_OR_I8    : Atomic2Ops32<atomic_load_or_8, "load_or_8">;
  defm ATOMIC_LOAD_OR_I16   : Atomic2Ops32<atomic_load_or_16, "load_or_16">;
  defm ATOMIC_LOAD_OR_I32   : Atomic2Ops32<atomic_load_or_32, "load_or_32">;
  defm ATOMIC_LOAD_XOR_I8   : Atomic2Ops32<atomic_load_xor_8, "load_xor_8">;
  defm ATOMIC_LOAD_XOR_I16  : Atomic2Ops32<atomic_load_xor_16, "load_xor_16">;
  defm ATOMIC_LOAD_XOR_I32  : Atomic2Ops32<atomic_load_xor_32, "load_xor_32">;
  defm ATOMIC_LOAD_NAND_I8  : Atomic2Ops32<atomic_load_nand_8, "load_nand_8">;
  defm ATOMIC_LOAD_NAND_I16 : Atomic2Ops32<atomic_load_nand_16, "load_nand_16">;
  defm ATOMIC_LOAD_NAND_I32 : Atomic2Ops32<atomic_load_nand_32, "load_nand_32">;

  defm ATOMIC_SWAP_I8       : Atomic2Ops32<atomic_swap_8, "swap_8">;
  defm ATOMIC_SWAP_I16      : Atomic2Ops32<atomic_swap_16, "swap_16">;
  defm ATOMIC_SWAP_I32      : Atomic2Ops32<atomic_swap_32, "swap_32">;

  defm ATOMIC_CMP_SWAP_I8   : AtomicCmpSwap32<atomic_cmp_swap_8, "8">;
  defm ATOMIC_CMP_SWAP_I16  : AtomicCmpSwap32<atomic_cmp_swap_16, "16">;
  defm ATOMIC_CMP_SWAP_I32  : AtomicCmpSwap32<atomic_cmp_swap_32, "32">;
}

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//

class LoadImm32< string instr_asm, Operand Od, RegisterClass RC> :
  MipsAsmPseudoInst<(outs RC:$rt), (ins Od:$imm32),
                     !strconcat(instr_asm, "\t$rt, $imm32")> ;
def LoadImm32Reg : LoadImm32<"li", shamt,CPURegs>;

class LoadAddress<string instr_asm, Operand MemOpnd, RegisterClass RC> :
  MipsAsmPseudoInst<(outs RC:$rt), (ins MemOpnd:$addr),
                     !strconcat(instr_asm, "\t$rt, $addr")> ;
def LoadAddr32Reg : LoadAddress<"la", mem, CPURegs>;

class LoadAddressImm<string instr_asm, Operand Od, RegisterClass RC> :
  MipsAsmPseudoInst<(outs RC:$rt), (ins Od:$imm32),
                     !strconcat(instr_asm, "\t$rt, $imm32")> ;
def LoadAddr32Imm : LoadAddressImm<"la", shamt,CPURegs>;

//===----------------------------------------------------------------------===//
// MipsI Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def ADDiu   : ArithLogicI<0x09, "addiu", add, simm16, immSExt16, CPURegs>,
              IsAsCheapAsAMove;
def ADDi    : ArithOverflowI<0x08, "addi", add, simm16, immSExt16, CPURegs>;
def SLTi    : SetCC_I<0x0a, "slti", setlt, simm16, immSExt16, CPURegs>;
def SLTiu   : SetCC_I<0x0b, "sltiu", setult, simm16, immSExt16, CPURegs>;
def ANDi    : ArithLogicI<0x0c, "andi", and, uimm16, immZExt16, CPURegs>;
def ORi     : ArithLogicI<0x0d, "ori", or, uimm16, immZExt16, CPURegs>;
def XORi    : ArithLogicI<0x0e, "xori", xor, uimm16, immZExt16, CPURegs>;
def LUi     : LoadUpper<0x0f, "lui", CPURegs, uimm16>;

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
def SLL     : shift_rotate_imm32<0x00, 0x00, "sll", shl>;
def SRL     : shift_rotate_imm32<0x02, 0x00, "srl", srl>;
def SRA     : shift_rotate_imm32<0x03, 0x00, "sra", sra>;
def SLLV    : shift_rotate_reg<0x04, 0x00, "sllv", shl, CPURegs>;
def SRLV    : shift_rotate_reg<0x06, 0x00, "srlv", srl, CPURegs>;
def SRAV    : shift_rotate_reg<0x07, 0x00, "srav", sra, CPURegs>;

// Rotate Instructions
let Predicates = [HasMips32r2, HasStandardEncoding] in {
    def ROTR    : shift_rotate_imm32<0x02, 0x01, "rotr", rotr>;
    def ROTRV   : shift_rotate_reg<0x06, 0x01, "rotrv", rotr, CPURegs>;
}

/// Load and Store Instructions
///  aligned
defm LB      : LoadM32<0x20, "lb",  sextloadi8>;
defm LBu     : LoadM32<0x24, "lbu", zextloadi8>;
defm LH      : LoadM32<0x21, "lh",  sextloadi16>;
defm LHu     : LoadM32<0x25, "lhu", zextloadi16>;
defm LW      : LoadM32<0x23, "lw",  load>;
defm SB      : StoreM32<0x28, "sb", truncstorei8>;
defm SH      : StoreM32<0x29, "sh", truncstorei16>;
defm SW      : StoreM32<0x2b, "sw", store>;

/// load/store left/right
defm LWL : LoadLeftRightM32<0x22, "lwl", MipsLWL>;
defm LWR : LoadLeftRightM32<0x26, "lwr", MipsLWR>;
defm SWL : StoreLeftRightM32<0x2a, "swl", MipsSWL>;
defm SWR : StoreLeftRightM32<0x2e, "swr", MipsSWR>;

let hasSideEffects = 1 in
def SYNC : InstSE<(outs), (ins i32imm:$stype), "sync $stype",
                  [(MipsSync imm:$stype)], NoItinerary, FrmOther>
{
  bits<5> stype;
  let Opcode = 0;
  let Inst{25-11} = 0;
  let Inst{10-6} = stype;
  let Inst{5-0} = 15;
}

/// Load-linked, Store-conditional
def LL    : LLBase<0x30, "ll", CPURegs, mem>,
            Requires<[NotN64, HasStandardEncoding]>;
def LL_P8 : LLBase<0x30, "ll", CPURegs, mem64>,
            Requires<[IsN64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

def SC    : SCBase<0x38, "sc", CPURegs, mem>,
            Requires<[NotN64, HasStandardEncoding]>;
def SC_P8 : SCBase<0x38, "sc", CPURegs, mem64>,
            Requires<[IsN64, HasStandardEncoding]> {
  let DecoderNamespace = "Mips64";
}

/// Jump and Branch Instructions
def J       : JumpFJ<0x02, jmptarget, "j", br, bb>,
              Requires<[RelocStatic, HasStandardEncoding]>, IsBranch;
def JR      : IndirectBranch<CPURegs>;
def B       : UncondBranch<0x04, "b">;
def BEQ     : CBranch<0x04, "beq", seteq, CPURegs>;
def BNE     : CBranch<0x05, "bne", setne, CPURegs>;
def BGEZ    : CBranchZero<0x01, 1, "bgez", setge, CPURegs>;
def BGTZ    : CBranchZero<0x07, 0, "bgtz", setgt, CPURegs>;
def BLEZ    : CBranchZero<0x06, 0, "blez", setle, CPURegs>;
def BLTZ    : CBranchZero<0x01, 0, "bltz", setlt, CPURegs>;

let rt = 0, rs = 0, isBranch = 1, isTerminator = 1, isBarrier = 1,
    hasDelaySlot = 1, Defs = [RA] in
def BAL_BR: FI<0x1, (outs), (ins brtarget:$imm16), "bal\t$imm16", [], IIBranch>;

def JAL  : JumpLink<0x03, "jal">;
def JALR : JumpLinkReg<0x00, 0x09, "jalr", CPURegs>;
def BGEZAL  : BranchLink<"bgezal", 0x11, CPURegs>;
def BLTZAL  : BranchLink<"bltzal", 0x10, CPURegs>;
def TAILCALL : JumpFJ<0x02, calltarget, "j", MipsTailCall, imm>, IsTailCall;
def TAILCALL_R : JumpFR<CPURegs, MipsTailCall>, IsTailCall;

def RET : RetBase<CPURegs>;

/// Multiply and Divide Instructions.
def MULT    : Mult32<0x18, "mult", IIImul>;
def MULTu   : Mult32<0x19, "multu", IIImul>;
def SDIV    : Div32<MipsDivRem, 0x1a, "div", IIIdiv>;
def UDIV    : Div32<MipsDivRemU, 0x1b, "divu", IIIdiv>;

def MTHI : MoveToLOHI<0x11, "mthi", CPURegs, [HI]>;
def MTLO : MoveToLOHI<0x13, "mtlo", CPURegs, [LO]>;
def MFHI : MoveFromLOHI<0x10, "mfhi", CPURegs, [HI]>;
def MFLO : MoveFromLOHI<0x12, "mflo", CPURegs, [LO]>;

/// Sign Ext In Register Instructions.
def SEB : SignExtInReg<0x10, "seb", i8, CPURegs>;
def SEH : SignExtInReg<0x18, "seh", i16, CPURegs>;

/// Count Leading
def CLZ : CountLeading0<0x20, "clz", CPURegs>;
def CLO : CountLeading1<0x21, "clo", CPURegs>;

/// Word Swap Bytes Within Halfwords
def WSBH : SubwordSwap<0x20, 0x2, "wsbh", CPURegs>;

/// No operation
let addr=0 in
  def NOP   : FJ<0, (outs), (ins), "nop", [], IIAlu>;

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched. It's similar to Sparc LEA_ADDRi
def LEA_ADDiu : EffectiveAddress<0x09,"addiu\t$rt, $addr", CPURegs, mem_ea>;

// DynAlloc node points to dynamically allocated stack space.
// $sp is added to the list of implicitly used registers to prevent dead code
// elimination from removing instructions that modify $sp.
let Uses = [SP] in
def DynAlloc : EffectiveAddress<0x09,"addiu\t$rt, $addr", CPURegs, mem_ea>;

// MADD*/MSUB*
def MADD  : MArithR<0, "madd", MipsMAdd, 1>;
def MADDU : MArithR<1, "maddu", MipsMAddu, 1>;
def MSUB  : MArithR<4, "msub", MipsMSub>;
def MSUBU : MArithR<5, "msubu", MipsMSubu>;

// MUL is a assembly macro in the current used ISAs. In recent ISA's
// it is a real instruction.
def MUL   : ArithLogicR<0x1c, 0x02, "mul", mul, IIImul, CPURegs, 1>,
            Requires<[HasMips32, HasStandardEncoding]>;

def RDHWR : ReadHardware<CPURegs, HWRegs>;

def EXT : ExtBase<0, "ext", CPURegs>;
def INS : InsBase<4, "ins", CPURegs>;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : InstAlias<"move $dst,$src", (ADD CPURegs:$dst,CPURegs:$src,ZERO)>;
def : InstAlias<"bal $offset", (BGEZAL RA,brtarget:$offset)>;
def : InstAlias<"addu $rs,$rt,$imm",
                (ADDiu CPURegs:$rs,CPURegs:$rt,simm16:$imm)>;
def : InstAlias<"add $rs,$rt,$imm",
                (ADDi CPURegs:$rs,CPURegs:$rt,simm16:$imm)>;
def : InstAlias<"and $rs,$rt,$imm",
                (ANDi CPURegs:$rs,CPURegs:$rt,simm16:$imm)>;
def : InstAlias<"j $rs", (JR CPURegs:$rs)>;
def : InstAlias<"not $rt,$rs", (NOR CPURegs:$rt,CPURegs:$rs,ZERO)>;
def : InstAlias<"neg $rt,$rs", (SUB CPURegs:$rt,ZERO,CPURegs:$rs)>;
def : InstAlias<"negu $rt,$rs", (SUBu CPURegs:$rt,ZERO,CPURegs:$rs)>;
def : InstAlias<"slt $rs,$rt,$imm",
                (SLTi CPURegs:$rs,CPURegs:$rt,simm16:$imm)>;
def : InstAlias<"xor $rs,$rt,$imm",
                (XORi CPURegs:$rs,CPURegs:$rt,simm16:$imm)>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Small immediates
def : MipsPat<(i32 immSExt16:$in),
              (ADDiu ZERO, imm:$in)>;
def : MipsPat<(i32 immZExt16:$in),
              (ORi ZERO, imm:$in)>;
def : MipsPat<(i32 immLow16Zero:$in),
              (LUi (HI16 imm:$in))>;

// Arbitrary immediates
def : MipsPat<(i32 imm:$imm),
          (ORi (LUi (HI16 imm:$imm)), (LO16 imm:$imm))>;

// Carry MipsPatterns
def : MipsPat<(subc CPURegs:$lhs, CPURegs:$rhs),
              (SUBu CPURegs:$lhs, CPURegs:$rhs)>;
def : MipsPat<(addc CPURegs:$lhs, CPURegs:$rhs),
              (ADDu CPURegs:$lhs, CPURegs:$rhs)>;
def : MipsPat<(addc  CPURegs:$src, immSExt16:$imm),
              (ADDiu CPURegs:$src, imm:$imm)>;

// Call
def : MipsPat<(MipsJmpLink (i32 tglobaladdr:$dst)),
              (JAL tglobaladdr:$dst)>;
def : MipsPat<(MipsJmpLink (i32 texternalsym:$dst)),
              (JAL texternalsym:$dst)>;
//def : MipsPat<(MipsJmpLink CPURegs:$dst),
//              (JALR CPURegs:$dst)>;

// Tail call
def : MipsPat<(MipsTailCall (iPTR tglobaladdr:$dst)),
              (TAILCALL tglobaladdr:$dst)>;
def : MipsPat<(MipsTailCall (iPTR texternalsym:$dst)),
              (TAILCALL texternalsym:$dst)>;
// hi/lo relocs
def : MipsPat<(MipsHi tglobaladdr:$in), (LUi tglobaladdr:$in)>;
def : MipsPat<(MipsHi tblockaddress:$in), (LUi tblockaddress:$in)>;
def : MipsPat<(MipsHi tjumptable:$in), (LUi tjumptable:$in)>;
def : MipsPat<(MipsHi tconstpool:$in), (LUi tconstpool:$in)>;
def : MipsPat<(MipsHi tglobaltlsaddr:$in), (LUi tglobaltlsaddr:$in)>;

def : MipsPat<(MipsLo tglobaladdr:$in), (ADDiu ZERO, tglobaladdr:$in)>;
def : MipsPat<(MipsLo tblockaddress:$in), (ADDiu ZERO, tblockaddress:$in)>;
def : MipsPat<(MipsLo tjumptable:$in), (ADDiu ZERO, tjumptable:$in)>;
def : MipsPat<(MipsLo tconstpool:$in), (ADDiu ZERO, tconstpool:$in)>;
def : MipsPat<(MipsLo tglobaltlsaddr:$in), (ADDiu ZERO, tglobaltlsaddr:$in)>;

def : MipsPat<(add CPURegs:$hi, (MipsLo tglobaladdr:$lo)),
              (ADDiu CPURegs:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add CPURegs:$hi, (MipsLo tblockaddress:$lo)),
              (ADDiu CPURegs:$hi, tblockaddress:$lo)>;
def : MipsPat<(add CPURegs:$hi, (MipsLo tjumptable:$lo)),
              (ADDiu CPURegs:$hi, tjumptable:$lo)>;
def : MipsPat<(add CPURegs:$hi, (MipsLo tconstpool:$lo)),
              (ADDiu CPURegs:$hi, tconstpool:$lo)>;
def : MipsPat<(add CPURegs:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (ADDiu CPURegs:$hi, tglobaltlsaddr:$lo)>;

// gp_rel relocs
def : MipsPat<(add CPURegs:$gp, (MipsGPRel tglobaladdr:$in)),
              (ADDiu CPURegs:$gp, tglobaladdr:$in)>;
def : MipsPat<(add CPURegs:$gp, (MipsGPRel tconstpool:$in)),
              (ADDiu CPURegs:$gp, tconstpool:$in)>;

// wrapper_pic
class WrapperPat<SDNode node, Instruction ADDiuOp, RegisterClass RC>:
      MipsPat<(MipsWrapper RC:$gp, node:$in),
              (ADDiuOp RC:$gp, node:$in)>;

def : WrapperPat<tglobaladdr, ADDiu, CPURegs>;
def : WrapperPat<tconstpool, ADDiu, CPURegs>;
def : WrapperPat<texternalsym, ADDiu, CPURegs>;
def : WrapperPat<tblockaddress, ADDiu, CPURegs>;
def : WrapperPat<tjumptable, ADDiu, CPURegs>;
def : WrapperPat<tglobaltlsaddr, ADDiu, CPURegs>;

// Mips does not have "not", so we expand our way
def : MipsPat<(not CPURegs:$in),
              (NOR CPURegs:$in, ZERO)>;

// extended loads
let Predicates = [NotN64, HasStandardEncoding] in {
  def : MipsPat<(i32 (extloadi1  addr:$src)), (LBu addr:$src)>;
  def : MipsPat<(i32 (extloadi8  addr:$src)), (LBu addr:$src)>;
  def : MipsPat<(i32 (extloadi16 addr:$src)), (LHu addr:$src)>;
}
let Predicates = [IsN64, HasStandardEncoding] in {
  def : MipsPat<(i32 (extloadi1  addr:$src)), (LBu_P8 addr:$src)>;
  def : MipsPat<(i32 (extloadi8  addr:$src)), (LBu_P8 addr:$src)>;
  def : MipsPat<(i32 (extloadi16 addr:$src)), (LHu_P8 addr:$src)>;
}

// peepholes
let Predicates = [NotN64, HasStandardEncoding] in {
  def : MipsPat<(store (i32 0), addr:$dst), (SW ZERO, addr:$dst)>;
}
let Predicates = [IsN64, HasStandardEncoding] in {
  def : MipsPat<(store (i32 0), addr:$dst), (SW_P8 ZERO, addr:$dst)>;
}

// brcond patterns
multiclass BrcondPats<RegisterClass RC, Instruction BEQOp, Instruction BNEOp,
                      Instruction SLTOp, Instruction SLTuOp, Instruction SLTiOp,
                      Instruction SLTiuOp, Register ZEROReg> {
def : MipsPat<(brcond (i32 (setne RC:$lhs, 0)), bb:$dst),
              (BNEOp RC:$lhs, ZEROReg, bb:$dst)>;
def : MipsPat<(brcond (i32 (seteq RC:$lhs, 0)), bb:$dst),
              (BEQOp RC:$lhs, ZEROReg, bb:$dst)>;

def : MipsPat<(brcond (i32 (setge RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setuge RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTuOp RC:$lhs, RC:$rhs), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setge RC:$lhs, immSExt16:$rhs)), bb:$dst),
              (BEQ (SLTiOp RC:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setuge RC:$lhs, immSExt16:$rhs)), bb:$dst),
              (BEQ (SLTiuOp RC:$lhs, immSExt16:$rhs), ZERO, bb:$dst)>;

def : MipsPat<(brcond (i32 (setle RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setule RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTuOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;

def : MipsPat<(brcond RC:$cond, bb:$dst),
              (BNEOp RC:$cond, ZEROReg, bb:$dst)>;
}

defm : BrcondPats<CPURegs, BEQ, BNE, SLT, SLTu, SLTi, SLTiu, ZERO>;

// setcc patterns
multiclass SeteqPats<RegisterClass RC, Instruction SLTiuOp, Instruction XOROp,
                     Instruction SLTuOp, Register ZEROReg> {
  def : MipsPat<(seteq RC:$lhs, RC:$rhs),
                (SLTiuOp (XOROp RC:$lhs, RC:$rhs), 1)>;
  def : MipsPat<(setne RC:$lhs, RC:$rhs),
                (SLTuOp ZEROReg, (XOROp RC:$lhs, RC:$rhs))>;
}

multiclass SetlePats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : MipsPat<(setle RC:$lhs, RC:$rhs),
                (XORi (SLTOp RC:$rhs, RC:$lhs), 1)>;
  def : MipsPat<(setule RC:$lhs, RC:$rhs),
                (XORi (SLTuOp RC:$rhs, RC:$lhs), 1)>;
}

multiclass SetgtPats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : MipsPat<(setgt RC:$lhs, RC:$rhs),
                (SLTOp RC:$rhs, RC:$lhs)>;
  def : MipsPat<(setugt RC:$lhs, RC:$rhs),
                (SLTuOp RC:$rhs, RC:$lhs)>;
}

multiclass SetgePats<RegisterClass RC, Instruction SLTOp, Instruction SLTuOp> {
  def : MipsPat<(setge RC:$lhs, RC:$rhs),
                (XORi (SLTOp RC:$lhs, RC:$rhs), 1)>;
  def : MipsPat<(setuge RC:$lhs, RC:$rhs),
                (XORi (SLTuOp RC:$lhs, RC:$rhs), 1)>;
}

multiclass SetgeImmPats<RegisterClass RC, Instruction SLTiOp,
                        Instruction SLTiuOp> {
  def : MipsPat<(setge RC:$lhs, immSExt16:$rhs),
                (XORi (SLTiOp RC:$lhs, immSExt16:$rhs), 1)>;
  def : MipsPat<(setuge RC:$lhs, immSExt16:$rhs),
                (XORi (SLTiuOp RC:$lhs, immSExt16:$rhs), 1)>;
}

defm : SeteqPats<CPURegs, SLTiu, XOR, SLTu, ZERO>;
defm : SetlePats<CPURegs, SLT, SLTu>;
defm : SetgtPats<CPURegs, SLT, SLTu>;
defm : SetgePats<CPURegs, SLT, SLTu>;
defm : SetgeImmPats<CPURegs, SLTi, SLTiu>;

// select MipsDynAlloc
def : MipsPat<(MipsDynAlloc addr:$f), (DynAlloc addr:$f)>;

// bswap pattern
def : MipsPat<(bswap CPURegs:$rt), (ROTR (WSBH CPURegs:$rt), 16)>;

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//

include "MipsInstrFPU.td"
include "Mips64InstrInfo.td"
include "MipsCondMov.td"

//
// Mips16

include "Mips16InstrFormats.td"
include "Mips16InstrInfo.td"

// DSP
include "MipsDSPInstrFormats.td"
include "MipsDSPInstrInfo.td"

