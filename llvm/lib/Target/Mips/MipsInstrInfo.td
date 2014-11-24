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
def SDT_MFLOHI : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVT<1, untyped>]>;
def SDT_MTLOHI : SDTypeProfile<1, 2, [SDTCisVT<0, untyped>,
                                      SDTCisInt<1>, SDTCisSameAs<1, 2>]>;
def SDT_MipsMultDiv : SDTypeProfile<1, 2, [SDTCisVT<0, untyped>, SDTCisInt<1>,
                                    SDTCisSameAs<1, 2>]>;
def SDT_MipsMAddMSub : SDTypeProfile<1, 3,
                                     [SDTCisVT<0, untyped>, SDTCisSameAs<0, 3>,
                                      SDTCisVT<1, i32>, SDTCisSameAs<1, 2>]>;
def SDT_MipsDivRem16 : SDTypeProfile<0, 2, [SDTCisInt<0>, SDTCisSameAs<0, 1>]>;

def SDT_MipsThreadPointer : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;

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
def MipsRet : SDNode<"MipsISD::Ret", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

// These are target-independent nodes, but have target-specific formats.
def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_MipsCallSeqStart,
                           [SDNPHasChain, SDNPSideEffect, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END", SDT_MipsCallSeqEnd,
                           [SDNPHasChain, SDNPSideEffect,
                            SDNPOptInGlue, SDNPOutGlue]>;

// Nodes used to extract LO/HI registers.
def MipsMFHI : SDNode<"MipsISD::MFHI", SDT_MFLOHI>;
def MipsMFLO : SDNode<"MipsISD::MFLO", SDT_MFLOHI>;

// Node used to insert 32-bit integers to LOHI register pair.
def MipsMTLOHI : SDNode<"MipsISD::MTLOHI", SDT_MTLOHI>;

// Mult nodes.
def MipsMult  : SDNode<"MipsISD::Mult", SDT_MipsMultDiv>;
def MipsMultu : SDNode<"MipsISD::Multu", SDT_MipsMultDiv>;

// MAdd*/MSub* nodes
def MipsMAdd  : SDNode<"MipsISD::MAdd", SDT_MipsMAddMSub>;
def MipsMAddu : SDNode<"MipsISD::MAddu", SDT_MipsMAddMSub>;
def MipsMSub  : SDNode<"MipsISD::MSub", SDT_MipsMAddMSub>;
def MipsMSubu : SDNode<"MipsISD::MSubu", SDT_MipsMAddMSub>;

// DivRem(u) nodes
def MipsDivRem    : SDNode<"MipsISD::DivRem", SDT_MipsMultDiv>;
def MipsDivRemU   : SDNode<"MipsISD::DivRemU", SDT_MipsMultDiv>;
def MipsDivRem16  : SDNode<"MipsISD::DivRem16", SDT_MipsDivRem16,
                           [SDNPOutGlue]>;
def MipsDivRemU16 : SDNode<"MipsISD::DivRemU16", SDT_MipsDivRem16,
                           [SDNPOutGlue]>;

// Target constant nodes that are not part of any isel patterns and remain
// unchanged can cause instructions with illegal operands to be emitted.
// Wrapper node patterns give the instruction selector a chance to replace
// target constant nodes that would otherwise remain unchanged with ADDiu
// nodes. Without these wrapper node patterns, the following conditional move
// instruction is emitted when function cmov2 in test/CodeGen/Mips/cmov.ll is
// compiled:
//  movn  %got(d)($gp), %got(c)($gp), $4
// This instruction is illegal since movn can take only register operands.

def MipsWrapper    : SDNode<"MipsISD::Wrapper", SDTIntBinOp>;

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
def HasMips2     :    Predicate<"Subtarget->hasMips2()">,
                      AssemblerPredicate<"FeatureMips2">;
def HasMips3_32  :    Predicate<"Subtarget->hasMips3_32()">,
                      AssemblerPredicate<"FeatureMips3_32">;
def HasMips3_32r2 :   Predicate<"Subtarget->hasMips3_32r2()">,
                      AssemblerPredicate<"FeatureMips3_32r2">;
def HasMips3     :    Predicate<"Subtarget->hasMips3()">,
                      AssemblerPredicate<"FeatureMips3">;
def HasMips4_32  :    Predicate<"Subtarget->hasMips4_32()">,
                      AssemblerPredicate<"FeatureMips4_32">;
def HasMips4_32r2 :   Predicate<"Subtarget->hasMips4_32r2()">,
                      AssemblerPredicate<"FeatureMips4_32r2">;
def HasMips5_32r2 :   Predicate<"Subtarget->hasMips5_32r2()">,
                      AssemblerPredicate<"FeatureMips5_32r2">;
def HasMips32    :    Predicate<"Subtarget->hasMips32()">,
                      AssemblerPredicate<"FeatureMips32">;
def HasMips32r2  :    Predicate<"Subtarget->hasMips32r2()">,
                      AssemblerPredicate<"FeatureMips32r2">;
def HasMips32r6  :    Predicate<"Subtarget->hasMips32r6()">,
                      AssemblerPredicate<"FeatureMips32r6">;
def NotMips32r6  :    Predicate<"!Subtarget->hasMips32r6()">,
                      AssemblerPredicate<"!FeatureMips32r6">;
def IsGP64bit    :    Predicate<"Subtarget->isGP64bit()">,
                      AssemblerPredicate<"FeatureGP64Bit">;
def IsGP32bit    :    Predicate<"!Subtarget->isGP64bit()">,
                      AssemblerPredicate<"!FeatureGP64Bit">;
def HasMips64    :    Predicate<"Subtarget->hasMips64()">,
                      AssemblerPredicate<"FeatureMips64">;
def HasMips64r2  :    Predicate<"Subtarget->hasMips64r2()">,
                      AssemblerPredicate<"FeatureMips64r2">;
def HasMips64r6  :    Predicate<"Subtarget->hasMips64r6()">,
                      AssemblerPredicate<"FeatureMips64r6">;
def NotMips64r6  :    Predicate<"!Subtarget->hasMips64r6()">,
                      AssemblerPredicate<"!FeatureMips64r6">;
def IsN64       :     Predicate<"Subtarget->isABI_N64()">,
                      AssemblerPredicate<"FeatureN64">;
def InMips16Mode :    Predicate<"Subtarget->inMips16Mode()">,
                      AssemblerPredicate<"FeatureMips16">;
def HasCnMips    :    Predicate<"Subtarget->hasCnMips()">,
                      AssemblerPredicate<"FeatureCnMips">;
def RelocStatic :     Predicate<"TM.getRelocationModel() == Reloc::Static">,
                      AssemblerPredicate<"FeatureMips32">;
def RelocPIC    :     Predicate<"TM.getRelocationModel() == Reloc::PIC_">,
                      AssemblerPredicate<"FeatureMips32">;
def NoNaNsFPMath :    Predicate<"TM.Options.NoNaNsFPMath">;
def HasStdEnc :       Predicate<"Subtarget->hasStandardEncoding()">,
                      AssemblerPredicate<"!FeatureMips16">;
def NotDSP :          Predicate<"!Subtarget->hasDSP()">;
def InMicroMips    :  Predicate<"Subtarget->inMicroMipsMode()">,
                      AssemblerPredicate<"FeatureMicroMips">;
def NotInMicroMips :  Predicate<"!Subtarget->inMicroMipsMode()">,
                      AssemblerPredicate<"!FeatureMicroMips">;
def IsLE           :  Predicate<"Subtarget->isLittle()">;
def IsBE           :  Predicate<"!Subtarget->isLittle()">;
def IsNotNaCl    :    Predicate<"!Subtarget->isTargetNaCl()">;

//===----------------------------------------------------------------------===//
// Mips GPR size adjectives.
// They are mutually exclusive.
//===----------------------------------------------------------------------===//

class GPR_32 { list<Predicate> GPRPredicates = [IsGP32bit]; }
class GPR_64 { list<Predicate> GPRPredicates = [IsGP64bit]; }

//===----------------------------------------------------------------------===//
// Mips ISA/ASE membership and instruction group membership adjectives.
// They are mutually exclusive.
//===----------------------------------------------------------------------===//

// FIXME: I'd prefer to use additive predicates to build the instruction sets
//        but we are short on assembler feature bits at the moment. Using a
//        subtractive predicate will hopefully keep us under the 32 predicate
//        limit long enough to develop an alternative way to handle P1||P2
//        predicates.
class ISA_MIPS1_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [NotMips32r6, NotMips64r6];
}
class ISA_MIPS2    { list<Predicate> InsnPredicates = [HasMips2]; }
class ISA_MIPS2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips2, NotMips32r6, NotMips64r6];
}
class ISA_MIPS3    { list<Predicate> InsnPredicates = [HasMips3]; }
class ISA_MIPS3_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips3, NotMips32r6, NotMips64r6];
}
class ISA_MIPS32   { list<Predicate> InsnPredicates = [HasMips32]; }
class ISA_MIPS32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips32, NotMips32r6, NotMips64r6];
}
class ISA_MIPS32R2 { list<Predicate> InsnPredicates = [HasMips32r2]; }
class ISA_MIPS32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips32r2, NotMips32r6, NotMips64r6];
}
class ISA_MIPS64   { list<Predicate> InsnPredicates = [HasMips64]; }
class ISA_MIPS64_NOT_64R6 {
  list<Predicate> InsnPredicates = [HasMips64, NotMips64r6];
}
class ISA_MIPS64R2 { list<Predicate> InsnPredicates = [HasMips64r2]; }
class ISA_MIPS32R6 { list<Predicate> InsnPredicates = [HasMips32r6]; }
class ISA_MIPS64R6 { list<Predicate> InsnPredicates = [HasMips64r6]; }

// The portions of MIPS-III that were also added to MIPS32
class INSN_MIPS3_32 { list<Predicate> InsnPredicates = [HasMips3_32]; }

// The portions of MIPS-III that were also added to MIPS32 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS3_32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips3_32, NotMips32r6, NotMips64r6];
}

// The portions of MIPS-III that were also added to MIPS32
class INSN_MIPS3_32R2 { list<Predicate> InsnPredicates = [HasMips3_32r2]; }

// The portions of MIPS-IV that were also added to MIPS32 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS4_32_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips4_32, NotMips32r6, NotMips64r6];
}

// The portions of MIPS-IV that were also added to MIPS32r2 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS4_32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips4_32r2, NotMips32r6, NotMips64r6];
}

// The portions of MIPS-V that were also added to MIPS32r2 but were removed in
// MIPS32r6 and MIPS64r6.
class INSN_MIPS5_32R2_NOT_32R6_64R6 {
  list<Predicate> InsnPredicates = [HasMips5_32r2, NotMips32r6, NotMips64r6];
}

//===----------------------------------------------------------------------===//

class MipsPat<dag pattern, dag result> : Pat<pattern, result>, PredicateControl {
  let EncodingPredicates = [HasStdEnc];
}

class MipsInstAlias<string Asm, dag Result, bit Emit = 0b1> :
  InstAlias<Asm, Result, Emit>, PredicateControl;

class IsCommutable {
  bit isCommutable = 1;
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

def MipsJumpTargetAsmOperand : AsmOperandClass {
  let Name = "JumpTarget";
  let ParserMethod = "ParseJumpTarget";
  let PredicateMethod = "isImm";
  let RenderMethod = "addImmOperands";
}

// Instruction operand types
def jmptarget   : Operand<OtherVT> {
  let EncoderMethod = "getJumpTargetOpValue";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}
def brtarget    : Operand<OtherVT> {
  let EncoderMethod = "getBranchTargetOpValue";
  let OperandType = "OPERAND_PCREL";
  let DecoderMethod = "DecodeBranchTarget";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}
def calltarget  : Operand<iPTR> {
  let EncoderMethod = "getJumpTargetOpValue";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def simm9 : Operand<i32>;
def simm10 : Operand<i32>;
def simm11 : Operand<i32>;

def simm16      : Operand<i32> {
  let DecoderMethod= "DecodeSimm16";
}

def simm19_lsl2 : Operand<i32> {
  let EncoderMethod = "getSimm19Lsl2Encoding";
  let DecoderMethod = "DecodeSimm19Lsl2";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def simm18_lsl3 : Operand<i32> {
  let EncoderMethod = "getSimm18Lsl3Encoding";
  let DecoderMethod = "DecodeSimm18Lsl3";
  let ParserMatchClass = MipsJumpTargetAsmOperand;
}

def simm20      : Operand<i32> {
}

def uimm20      : Operand<i32> {
}

def uimm10      : Operand<i32> {
}

def simm16_64   : Operand<i64> {
  let DecoderMethod = "DecodeSimm16";
}

// Zero
def uimmz       : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

// Unsigned Operand
def uimm2 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm3 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm5       : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm6 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm16      : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def pcrel16      : Operand<i32> {
}

def MipsMemAsmOperand : AsmOperandClass {
  let Name = "Mem";
  let ParserMethod = "parseMemOperand";
}

def MipsMemSimm11AsmOperand : AsmOperandClass {
  let Name = "MemOffsetSimm11";
  let SuperClasses = [MipsMemAsmOperand];
  let RenderMethod = "addMemOperands";
  let ParserMethod = "parseMemOperand";
  let PredicateMethod = "isMemWithSimmOffset<11>";
  //let DiagnosticType = "Simm11";
}

def MipsInvertedImmoperand : AsmOperandClass {
  let Name = "InvNum";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseInvNum";
}

def InvertedImOperand : Operand<i32> {
  let ParserMatchClass = MipsInvertedImmoperand;
}

def InvertedImOperand64 : Operand<i64> {
  let ParserMatchClass = MipsInvertedImmoperand;
}

class mem_generic : Operand<iPTR> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ptr_rc, simm16);
  let EncoderMethod = "getMemEncoding";
  let ParserMatchClass = MipsMemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

// Address operand
def mem : mem_generic;

// MSA specific address operand
def mem_msa : mem_generic {
  let MIOperandInfo = (ops ptr_rc, simm10);
  let EncoderMethod = "getMSAMemEncoding";
}

def mem_simm9 : mem_generic {
  let MIOperandInfo = (ops ptr_rc, simm9);
  let EncoderMethod = "getMemEncoding";
}

def mem_simm11 : mem_generic {
  let MIOperandInfo = (ops ptr_rc, simm11);
  let EncoderMethod = "getMemEncoding";
  let ParserMatchClass = MipsMemSimm11AsmOperand;
}

def mem_ea : Operand<iPTR> {
  let PrintMethod = "printMemOperandEA";
  let MIOperandInfo = (ops ptr_rc, simm16);
  let EncoderMethod = "getMemEncoding";
  let OperandType = "OPERAND_MEMORY";
}

def PtrRC : Operand<iPTR> {
  let MIOperandInfo = (ops ptr_rc);
  let DecoderMethod = "DecodePtrRegisterClass";
  let ParserMatchClass = GPR32AsmOperand;
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

// Plus 1.
def Plus1 : SDNodeXForm<imm, [{ return getImm(N, N->getSExtValue() + 1); }]>;

// Node immediate is zero (e.g. insve.d)
def immz : PatLeaf<(imm), [{ return N->getSExtValue() == 0; }]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt8  : PatLeaf<(imm), [{ return isInt<8>(N->getSExtValue()); }]>;

// Node immediate fits as 16-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt16  : PatLeaf<(imm), [{ return isInt<16>(N->getSExtValue()); }]>;

// Node immediate fits as 15-bit sign extended on target immediate.
// e.g. addi, andi
def immSExt15  : PatLeaf<(imm), [{ return isInt<15>(N->getSExtValue()); }]>;

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

// True if (N + 1) fits in 16-bit field.
def immSExt16Plus1 : PatLeaf<(imm), [{
  return isInt<17>(N->getSExtValue()) && isInt<16>(N->getSExtValue() + 1);
}]>;

// Mips Address Mode! SDNode frameindex could possibily be a match
// since load and store instructions from stack used it.
def addr :
  ComplexPattern<iPTR, 2, "selectIntAddr", [frameindex]>;

def addrRegImm :
  ComplexPattern<iPTR, 2, "selectAddrRegImm", [frameindex]>;

def addrRegReg :
  ComplexPattern<iPTR, 2, "selectAddrRegReg", [frameindex]>;

def addrDefault :
  ComplexPattern<iPTR, 2, "selectAddrDefault", [frameindex]>;

def addrimm10 : ComplexPattern<iPTR, 2, "selectIntAddrMSA", [frameindex]>;

//===----------------------------------------------------------------------===//
// Instructions specific format
//===----------------------------------------------------------------------===//

// Arithmetic and logical instructions with 3 register operands.
class ArithLogicR<string opstr, RegisterOperand RO, bit isComm = 0,
                  InstrItinClass Itin = NoItinerary,
                  SDPatternOperator OpNode = null_frag>:
  InstSE<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (OpNode RO:$rs, RO:$rt))], Itin, FrmR, opstr> {
  let isCommutable = isComm;
  let isReMaterializable = 1;
  let TwoOperandAliasConstraint = "$rd = $rs";
}

// Arithmetic and logical instructions with 2 register operands.
class ArithLogicI<string opstr, Operand Od, RegisterOperand RO,
                  InstrItinClass Itin = NoItinerary,
                  SDPatternOperator imm_type = null_frag,
                  SDPatternOperator OpNode = null_frag> :
  InstSE<(outs RO:$rt), (ins RO:$rs, Od:$imm16),
         !strconcat(opstr, "\t$rt, $rs, $imm16"),
         [(set RO:$rt, (OpNode RO:$rs, imm_type:$imm16))],
         Itin, FrmI, opstr> {
  let isReMaterializable = 1;
  let TwoOperandAliasConstraint = "$rs = $rt";
}

// Arithmetic Multiply ADD/SUB
class MArithR<string opstr, InstrItinClass itin, bit isComm = 0> :
  InstSE<(outs), (ins GPR32Opnd:$rs, GPR32Opnd:$rt),
         !strconcat(opstr, "\t$rs, $rt"), [], itin, FrmR, opstr> {
  let Defs = [HI0, LO0];
  let Uses = [HI0, LO0];
  let isCommutable = isComm;
}

//  Logical
class LogicNOR<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set RO:$rd, (not (or RO:$rs, RO:$rt)))], II_NOR, FrmR, opstr> {
  let isCommutable = 1;
}

// Shifts
class shift_rotate_imm<string opstr, Operand ImmOpnd,
                       RegisterOperand RO, InstrItinClass itin,
                       SDPatternOperator OpNode = null_frag,
                       SDPatternOperator PF = null_frag> :
  InstSE<(outs RO:$rd), (ins RO:$rt, ImmOpnd:$shamt),
         !strconcat(opstr, "\t$rd, $rt, $shamt"),
         [(set RO:$rd, (OpNode RO:$rt, PF:$shamt))], itin, FrmR, opstr> {
  let TwoOperandAliasConstraint = "$rt = $rd";
}

class shift_rotate_reg<string opstr, RegisterOperand RO, InstrItinClass itin,
                       SDPatternOperator OpNode = null_frag>:
  InstSE<(outs RO:$rd), (ins RO:$rt, GPR32Opnd:$rs),
         !strconcat(opstr, "\t$rd, $rt, $rs"),
         [(set RO:$rd, (OpNode RO:$rt, GPR32Opnd:$rs))], itin, FrmR,
         opstr>;

// Load Upper Imediate
class LoadUpper<string opstr, RegisterOperand RO, Operand Imm>:
  InstSE<(outs RO:$rt), (ins Imm:$imm16), !strconcat(opstr, "\t$rt, $imm16"),
         [], II_LUI, FrmI, opstr>, IsAsCheapAsAMove {
  let neverHasSideEffects = 1;
  let isReMaterializable = 1;
}

// Memory Load/Store
class Load<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
           InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr> :
  InstSE<(outs RO:$rt), (ins mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode Addr:$addr))], Itin, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  let canFoldAsLoad = 1;
  let mayLoad = 1;
}

class Store<string opstr, DAGOperand RO, SDPatternOperator OpNode = null_frag,
            InstrItinClass Itin = NoItinerary, ComplexPattern Addr = addr> :
  InstSE<(outs), (ins RO:$rt, mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, Addr:$addr)], Itin, FrmI, opstr> {
  let DecoderMethod = "DecodeMem";
  let mayStore = 1;
}

// Load/Store Left/Right
let canFoldAsLoad = 1 in
class LoadLeftRight<string opstr, SDNode OpNode, RegisterOperand RO,
                    InstrItinClass Itin> :
  InstSE<(outs RO:$rt), (ins mem:$addr, RO:$src),
         !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, (OpNode addr:$addr, RO:$src))], Itin, FrmI> {
  let DecoderMethod = "DecodeMem";
  string Constraints = "$src = $rt";
}

class StoreLeftRight<string opstr, SDNode OpNode, RegisterOperand RO,
                     InstrItinClass Itin> :
  InstSE<(outs), (ins RO:$rt, mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(OpNode RO:$rt, addr:$addr)], Itin, FrmI> {
  let DecoderMethod = "DecodeMem";
}

// Conditional Branch
class CBranch<string opstr, DAGOperand opnd, PatFrag cond_op,
              RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, RO:$rt, opnd:$offset),
         !strconcat(opstr, "\t$rs, $rt, $offset"),
         [(brcond (i32 (cond_op RO:$rs, RO:$rt)), bb:$offset)], IIBranch,
         FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
}

class CBranchZero<string opstr, DAGOperand opnd, PatFrag cond_op,
                  RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, opnd:$offset),
         !strconcat(opstr, "\t$rs, $offset"),
         [(brcond (i32 (cond_op RO:$rs, 0)), bb:$offset)], IIBranch,
         FrmI, opstr> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let Defs = [AT];
}

// SetCC
class SetCC_R<string opstr, PatFrag cond_op, RegisterOperand RO> :
  InstSE<(outs GPR32Opnd:$rd), (ins RO:$rs, RO:$rt),
         !strconcat(opstr, "\t$rd, $rs, $rt"),
         [(set GPR32Opnd:$rd, (cond_op RO:$rs, RO:$rt))],
         II_SLT_SLTU, FrmR, opstr>;

class SetCC_I<string opstr, PatFrag cond_op, Operand Od, PatLeaf imm_type,
              RegisterOperand RO>:
  InstSE<(outs GPR32Opnd:$rt), (ins RO:$rs, Od:$imm16),
         !strconcat(opstr, "\t$rt, $rs, $imm16"),
         [(set GPR32Opnd:$rt, (cond_op RO:$rs, imm_type:$imm16))],
         II_SLTI_SLTIU, FrmI, opstr>;

// Jump
class JumpFJ<DAGOperand opnd, string opstr, SDPatternOperator operator,
             SDPatternOperator targetoperator, string bopstr> :
  InstSE<(outs), (ins opnd:$target), !strconcat(opstr, "\t$target"),
         [(operator targetoperator:$target)], IIBranch, FrmJ, bopstr> {
  let isTerminator=1;
  let isBarrier=1;
  let hasDelaySlot = 1;
  let DecoderMethod = "DecodeJumpTarget";
  let Defs = [AT];
}

// Unconditional branch
class UncondBranch<Instruction BEQInst> :
  PseudoSE<(outs), (ins brtarget:$offset), [(br bb:$offset)], IIBranch>,
  PseudoInstExpansion<(BEQInst ZERO, ZERO, brtarget:$offset)> {
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
  let hasDelaySlot = 1;
  let AdditionalPredicates = [RelocPIC];
  let Defs = [AT];
}

// Base class for indirect branch and return instruction classes.
let isTerminator=1, isBarrier=1, hasDelaySlot = 1 in
class JumpFR<string opstr, RegisterOperand RO,
             SDPatternOperator operator = null_frag>:
  InstSE<(outs), (ins RO:$rs), "jr\t$rs", [(operator RO:$rs)], IIBranch,
         FrmR, opstr>;

// Indirect branch
class IndirectBranch<string opstr, RegisterOperand RO> : JumpFR<opstr, RO> {
  let isBranch = 1;
  let isIndirectBranch = 1;
}

// Jump and Link (Call)
let isCall=1, hasDelaySlot=1, Defs = [RA] in {
  class JumpLink<string opstr, DAGOperand opnd> :
    InstSE<(outs), (ins opnd:$target), !strconcat(opstr, "\t$target"),
           [(MipsJmpLink imm:$target)], IIBranch, FrmJ, opstr> {
    let DecoderMethod = "DecodeJumpTarget";
  }

  class JumpLinkRegPseudo<RegisterOperand RO, Instruction JALRInst,
                          Register RetReg, RegisterOperand ResRO = RO>:
    PseudoSE<(outs), (ins RO:$rs), [(MipsJmpLink RO:$rs)], IIBranch>,
    PseudoInstExpansion<(JALRInst RetReg, ResRO:$rs)>;

  class JumpLinkReg<string opstr, RegisterOperand RO>:
    InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
           [], IIBranch, FrmR>;

  class BGEZAL_FT<string opstr, DAGOperand opnd, RegisterOperand RO> :
    InstSE<(outs), (ins RO:$rs, opnd:$offset),
           !strconcat(opstr, "\t$rs, $offset"), [], IIBranch, FrmI, opstr>;

}

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, hasDelaySlot = 1,
    hasExtraSrcRegAllocReq = 1, Defs = [AT] in {
  class TailCall<Instruction JumpInst> :
    PseudoSE<(outs), (ins calltarget:$target), [], IIBranch>,
    PseudoInstExpansion<(JumpInst jmptarget:$target)>;

  class TailCallReg<RegisterOperand RO, Instruction JRInst,
                    RegisterOperand ResRO = RO> :
    PseudoSE<(outs), (ins RO:$rs), [(MipsTailCall RO:$rs)], IIBranch>,
    PseudoInstExpansion<(JRInst ResRO:$rs)>;
}

class BAL_BR_Pseudo<Instruction RealInst> :
  PseudoSE<(outs), (ins brtarget:$offset), [], IIBranch>,
  PseudoInstExpansion<(RealInst ZERO, brtarget:$offset)> {
  let isBranch = 1;
  let isTerminator = 1;
  let isBarrier = 1;
  let hasDelaySlot = 1;
  let Defs = [RA];
}

// Syscall
class SYS_FT<string opstr> :
  InstSE<(outs), (ins uimm20:$code_),
         !strconcat(opstr, "\t$code_"), [], NoItinerary, FrmI, opstr>;
// Break
class BRK_FT<string opstr> :
  InstSE<(outs), (ins uimm10:$code_1, uimm10:$code_2),
         !strconcat(opstr, "\t$code_1, $code_2"), [], NoItinerary,
         FrmOther, opstr>;

// (D)Eret
class ER_FT<string opstr> :
  InstSE<(outs), (ins),
         opstr, [], NoItinerary, FrmOther, opstr>;

// Interrupts
class DEI_FT<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt), (ins),
         !strconcat(opstr, "\t$rt"), [], NoItinerary, FrmOther, opstr>;

// Wait
class WAIT_FT<string opstr> :
  InstSE<(outs), (ins), opstr, [], NoItinerary, FrmOther, opstr>;

// Sync
let hasSideEffects = 1 in
class SYNC_FT<string opstr> :
  InstSE<(outs), (ins i32imm:$stype), "sync $stype", [(MipsSync imm:$stype)],
         NoItinerary, FrmOther, opstr>;

let hasSideEffects = 1 in
class TEQ_FT<string opstr, RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, RO:$rt, uimm16:$code_),
         !strconcat(opstr, "\t$rs, $rt, $code_"), [], NoItinerary,
         FrmI, opstr>;

class TEQI_FT<string opstr, RegisterOperand RO> :
  InstSE<(outs), (ins RO:$rs, uimm16:$imm16),
         !strconcat(opstr, "\t$rs, $imm16"), [], NoItinerary, FrmOther, opstr>;
// Mul, Div
class Mult<string opstr, InstrItinClass itin, RegisterOperand RO,
           list<Register> DefRegs> :
  InstSE<(outs), (ins RO:$rs, RO:$rt), !strconcat(opstr, "\t$rs, $rt"), [],
         itin, FrmR, opstr> {
  let isCommutable = 1;
  let Defs = DefRegs;
  let neverHasSideEffects = 1;
}

// Pseudo multiply/divide instruction with explicit accumulator register
// operands.
class MultDivPseudo<Instruction RealInst, RegisterClass R0, RegisterOperand R1,
                    SDPatternOperator OpNode, InstrItinClass Itin,
                    bit IsComm = 1, bit HasSideEffects = 0,
                    bit UsesCustomInserter = 0> :
  PseudoSE<(outs R0:$ac), (ins R1:$rs, R1:$rt),
           [(set R0:$ac, (OpNode R1:$rs, R1:$rt))], Itin>,
  PseudoInstExpansion<(RealInst R1:$rs, R1:$rt)> {
  let isCommutable = IsComm;
  let hasSideEffects = HasSideEffects;
  let usesCustomInserter = UsesCustomInserter;
}

// Pseudo multiply add/sub instruction with explicit accumulator register
// operands.
class MAddSubPseudo<Instruction RealInst, SDPatternOperator OpNode,
                    InstrItinClass itin>
  : PseudoSE<(outs ACC64:$ac),
             (ins GPR32Opnd:$rs, GPR32Opnd:$rt, ACC64:$acin),
             [(set ACC64:$ac,
              (OpNode GPR32Opnd:$rs, GPR32Opnd:$rt, ACC64:$acin))],
             itin>,
    PseudoInstExpansion<(RealInst GPR32Opnd:$rs, GPR32Opnd:$rt)> {
  string Constraints = "$acin = $ac";
}

class Div<string opstr, InstrItinClass itin, RegisterOperand RO,
          list<Register> DefRegs> :
  InstSE<(outs), (ins RO:$rs, RO:$rt), !strconcat(opstr, "\t$$zero, $rs, $rt"),
         [], itin, FrmR, opstr> {
  let Defs = DefRegs;
}

// Move from Hi/Lo
class PseudoMFLOHI<RegisterClass DstRC, RegisterClass SrcRC, SDNode OpNode>
  : PseudoSE<(outs DstRC:$rd), (ins SrcRC:$hilo),
             [(set DstRC:$rd, (OpNode SrcRC:$hilo))], II_MFHI_MFLO>;

class MoveFromLOHI<string opstr, RegisterOperand RO, Register UseReg>:
  InstSE<(outs RO:$rd), (ins), !strconcat(opstr, "\t$rd"), [], II_MFHI_MFLO,
         FrmR, opstr> {
  let Uses = [UseReg];
  let neverHasSideEffects = 1;
}

class PseudoMTLOHI<RegisterClass DstRC, RegisterClass SrcRC>
  : PseudoSE<(outs DstRC:$lohi), (ins SrcRC:$lo, SrcRC:$hi),
             [(set DstRC:$lohi, (MipsMTLOHI SrcRC:$lo, SrcRC:$hi))],
             II_MTHI_MTLO>;

class MoveToLOHI<string opstr, RegisterOperand RO, list<Register> DefRegs>:
  InstSE<(outs), (ins RO:$rs), !strconcat(opstr, "\t$rs"), [], II_MTHI_MTLO,
  FrmR, opstr> {
  let Defs = DefRegs;
  let neverHasSideEffects = 1;
}

class EffectiveAddress<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt), (ins mem_ea:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [(set RO:$rt, addr:$addr)], NoItinerary, FrmI,
         !strconcat(opstr, "_lea")> {
  let isCodeGenOnly = 1;
  let DecoderMethod = "DecodeMem";
}

// Count Leading Ones/Zeros in Word
class CountLeading0<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
         [(set RO:$rd, (ctlz RO:$rs))], II_CLZ, FrmR, opstr>;

class CountLeading1<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rs), !strconcat(opstr, "\t$rd, $rs"),
         [(set RO:$rd, (ctlz (not RO:$rs)))], II_CLO, FrmR, opstr>;

// Sign Extend in Register.
class SignExtInReg<string opstr, ValueType vt, RegisterOperand RO,
                   InstrItinClass itin> :
  InstSE<(outs RO:$rd), (ins RO:$rt), !strconcat(opstr, "\t$rd, $rt"),
         [(set RO:$rd, (sext_inreg RO:$rt, vt))], itin, FrmR, opstr>;

// Subword Swap
class SubwordSwap<string opstr, RegisterOperand RO>:
  InstSE<(outs RO:$rd), (ins RO:$rt), !strconcat(opstr, "\t$rd, $rt"), [],
         NoItinerary, FrmR, opstr> {
  let neverHasSideEffects = 1;
}

// Read Hardware
class ReadHardware<RegisterOperand CPURegOperand, RegisterOperand RO> :
  InstSE<(outs CPURegOperand:$rt), (ins RO:$rd), "rdhwr\t$rt, $rd", [],
         II_RDHWR, FrmR>;

// Ext and Ins
class ExtBase<string opstr, RegisterOperand RO, Operand PosOpnd,
              SDPatternOperator Op = null_frag>:
  InstSE<(outs RO:$rt), (ins RO:$rs, PosOpnd:$pos, size_ext:$size),
         !strconcat(opstr, " $rt, $rs, $pos, $size"),
         [(set RO:$rt, (Op RO:$rs, imm:$pos, imm:$size))], NoItinerary,
         FrmR, opstr>, ISA_MIPS32R2;

class InsBase<string opstr, RegisterOperand RO, Operand PosOpnd,
              SDPatternOperator Op = null_frag>:
  InstSE<(outs RO:$rt), (ins RO:$rs, PosOpnd:$pos, size_ins:$size, RO:$src),
         !strconcat(opstr, " $rt, $rs, $pos, $size"),
         [(set RO:$rt, (Op RO:$rs, imm:$pos, imm:$size, RO:$src))],
         NoItinerary, FrmR, opstr>, ISA_MIPS32R2 {
  let Constraints = "$src = $rt";
}

// Atomic instructions with 2 source operands (ATOMIC_SWAP & ATOMIC_LOAD_*).
class Atomic2Ops<PatFrag Op, RegisterClass DRC> :
  PseudoSE<(outs DRC:$dst), (ins PtrRC:$ptr, DRC:$incr),
           [(set DRC:$dst, (Op iPTR:$ptr, DRC:$incr))]>;

// Atomic Compare & Swap.
class AtomicCmpSwap<PatFrag Op, RegisterClass DRC> :
  PseudoSE<(outs DRC:$dst), (ins PtrRC:$ptr, DRC:$cmp, DRC:$swap),
           [(set DRC:$dst, (Op iPTR:$ptr, DRC:$cmp, DRC:$swap))]>;

class LLBase<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt), (ins mem:$addr), !strconcat(opstr, "\t$rt, $addr"),
         [], NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMem";
  let mayLoad = 1;
}

class SCBase<string opstr, RegisterOperand RO> :
  InstSE<(outs RO:$dst), (ins RO:$rt, mem:$addr),
         !strconcat(opstr, "\t$rt, $addr"), [], NoItinerary, FrmI> {
  let DecoderMethod = "DecodeMem";
  let mayStore = 1;
  let Constraints = "$rt = $dst";
}

class MFC3OP<string asmstr, RegisterOperand RO> :
  InstSE<(outs RO:$rt, RO:$rd, uimm16:$sel), (ins),
         !strconcat(asmstr, "\t$rt, $rd, $sel"), [], NoItinerary, FrmFR>;

class TrapBase<Instruction RealInst>
  : PseudoSE<(outs), (ins), [(trap)], NoItinerary>,
    PseudoInstExpansion<(RealInst 0, 0)> {
  let isBarrier = 1;
  let isTerminator = 1;
  let isCodeGenOnly = 1;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

// Return RA.
let isReturn=1, isTerminator=1, hasDelaySlot=1, isBarrier=1, hasCtrlDep=1 in
def RetRA : PseudoSE<(outs), (ins), [(MipsRet)]>;

let Defs = [SP], Uses = [SP], hasSideEffects = 1 in {
def ADJCALLSTACKDOWN : MipsPseudo<(outs), (ins i32imm:$amt),
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP   : MipsPseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}

let usesCustomInserter = 1 in {
  def ATOMIC_LOAD_ADD_I8   : Atomic2Ops<atomic_load_add_8, GPR32>;
  def ATOMIC_LOAD_ADD_I16  : Atomic2Ops<atomic_load_add_16, GPR32>;
  def ATOMIC_LOAD_ADD_I32  : Atomic2Ops<atomic_load_add_32, GPR32>;
  def ATOMIC_LOAD_SUB_I8   : Atomic2Ops<atomic_load_sub_8, GPR32>;
  def ATOMIC_LOAD_SUB_I16  : Atomic2Ops<atomic_load_sub_16, GPR32>;
  def ATOMIC_LOAD_SUB_I32  : Atomic2Ops<atomic_load_sub_32, GPR32>;
  def ATOMIC_LOAD_AND_I8   : Atomic2Ops<atomic_load_and_8, GPR32>;
  def ATOMIC_LOAD_AND_I16  : Atomic2Ops<atomic_load_and_16, GPR32>;
  def ATOMIC_LOAD_AND_I32  : Atomic2Ops<atomic_load_and_32, GPR32>;
  def ATOMIC_LOAD_OR_I8    : Atomic2Ops<atomic_load_or_8, GPR32>;
  def ATOMIC_LOAD_OR_I16   : Atomic2Ops<atomic_load_or_16, GPR32>;
  def ATOMIC_LOAD_OR_I32   : Atomic2Ops<atomic_load_or_32, GPR32>;
  def ATOMIC_LOAD_XOR_I8   : Atomic2Ops<atomic_load_xor_8, GPR32>;
  def ATOMIC_LOAD_XOR_I16  : Atomic2Ops<atomic_load_xor_16, GPR32>;
  def ATOMIC_LOAD_XOR_I32  : Atomic2Ops<atomic_load_xor_32, GPR32>;
  def ATOMIC_LOAD_NAND_I8  : Atomic2Ops<atomic_load_nand_8, GPR32>;
  def ATOMIC_LOAD_NAND_I16 : Atomic2Ops<atomic_load_nand_16, GPR32>;
  def ATOMIC_LOAD_NAND_I32 : Atomic2Ops<atomic_load_nand_32, GPR32>;

  def ATOMIC_SWAP_I8       : Atomic2Ops<atomic_swap_8, GPR32>;
  def ATOMIC_SWAP_I16      : Atomic2Ops<atomic_swap_16, GPR32>;
  def ATOMIC_SWAP_I32      : Atomic2Ops<atomic_swap_32, GPR32>;

  def ATOMIC_CMP_SWAP_I8   : AtomicCmpSwap<atomic_cmp_swap_8, GPR32>;
  def ATOMIC_CMP_SWAP_I16  : AtomicCmpSwap<atomic_cmp_swap_16, GPR32>;
  def ATOMIC_CMP_SWAP_I32  : AtomicCmpSwap<atomic_cmp_swap_32, GPR32>;
}

/// Pseudo instructions for loading and storing accumulator registers.
let isPseudo = 1, isCodeGenOnly = 1 in {
  def LOAD_ACC64  : Load<"", ACC64>;
  def STORE_ACC64 : Store<"", ACC64>;
}

// We need these two pseudo instructions to avoid offset calculation for long
// branches.  See the comment in file MipsLongBranch.cpp for detailed
// explanation.

// Expands to: lui $dst, %hi($tgt - $baltgt)
def LONG_BRANCH_LUi : PseudoSE<(outs GPR32Opnd:$dst),
  (ins brtarget:$tgt, brtarget:$baltgt), []>;

// Expands to: addiu $dst, $src, %lo($tgt - $baltgt)
def LONG_BRANCH_ADDiu : PseudoSE<(outs GPR32Opnd:$dst),
  (ins GPR32Opnd:$src, brtarget:$tgt, brtarget:$baltgt), []>;

//===----------------------------------------------------------------------===//
// Instruction definition
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// MipsI Instructions
//===----------------------------------------------------------------------===//

/// Arithmetic Instructions (ALU Immediate)
def ADDiu : MMRel, ArithLogicI<"addiu", simm16, GPR32Opnd, II_ADDIU, immSExt16,
                               add>,
            ADDI_FM<0x9>, IsAsCheapAsAMove;
def ADDi  : MMRel, ArithLogicI<"addi", simm16, GPR32Opnd>, ADDI_FM<0x8>,
            ISA_MIPS1_NOT_32R6_64R6;
def SLTi  : MMRel, SetCC_I<"slti", setlt, simm16, immSExt16, GPR32Opnd>,
            SLTI_FM<0xa>;
def SLTiu : MMRel, SetCC_I<"sltiu", setult, simm16, immSExt16, GPR32Opnd>,
            SLTI_FM<0xb>;
def ANDi  : MMRel, ArithLogicI<"andi", uimm16, GPR32Opnd, II_ANDI, immZExt16,
                               and>,
            ADDI_FM<0xc>;
def ORi   : MMRel, ArithLogicI<"ori", uimm16, GPR32Opnd, II_ORI, immZExt16,
                               or>,
            ADDI_FM<0xd>;
def XORi  : MMRel, ArithLogicI<"xori", uimm16, GPR32Opnd, II_XORI, immZExt16,
                               xor>,
            ADDI_FM<0xe>;
def LUi   : MMRel, LoadUpper<"lui", GPR32Opnd, uimm16>, LUI_FM;

/// Arithmetic Instructions (3-Operand, R-Type)
def ADDu  : MMRel, ArithLogicR<"addu", GPR32Opnd, 1, II_ADDU, add>,
            ADD_FM<0, 0x21>;
def SUBu  : MMRel, ArithLogicR<"subu", GPR32Opnd, 0, II_SUBU, sub>,
            ADD_FM<0, 0x23>;
let Defs = [HI0, LO0] in
def MUL   : MMRel, ArithLogicR<"mul", GPR32Opnd, 1, II_MUL, mul>,
            ADD_FM<0x1c, 2>, ISA_MIPS32_NOT_32R6_64R6;
def ADD   : MMRel, ArithLogicR<"add", GPR32Opnd>, ADD_FM<0, 0x20>;
def SUB   : MMRel, ArithLogicR<"sub", GPR32Opnd>, ADD_FM<0, 0x22>;
def SLT   : MMRel, SetCC_R<"slt", setlt, GPR32Opnd>, ADD_FM<0, 0x2a>;
def SLTu  : MMRel, SetCC_R<"sltu", setult, GPR32Opnd>, ADD_FM<0, 0x2b>;
def AND   : MMRel, ArithLogicR<"and", GPR32Opnd, 1, II_AND, and>,
            ADD_FM<0, 0x24>;
def OR    : MMRel, ArithLogicR<"or", GPR32Opnd, 1, II_OR, or>,
            ADD_FM<0, 0x25>;
def XOR   : MMRel, ArithLogicR<"xor", GPR32Opnd, 1, II_XOR, xor>,
            ADD_FM<0, 0x26>;
def NOR   : MMRel, LogicNOR<"nor", GPR32Opnd>, ADD_FM<0, 0x27>;

/// Shift Instructions
def SLL  : MMRel, shift_rotate_imm<"sll", uimm5, GPR32Opnd, II_SLL, shl,
                                   immZExt5>, SRA_FM<0, 0>;
def SRL  : MMRel, shift_rotate_imm<"srl", uimm5, GPR32Opnd, II_SRL, srl,
                                   immZExt5>, SRA_FM<2, 0>;
def SRA  : MMRel, shift_rotate_imm<"sra", uimm5, GPR32Opnd, II_SRA, sra,
                                   immZExt5>, SRA_FM<3, 0>;
def SLLV : MMRel, shift_rotate_reg<"sllv", GPR32Opnd, II_SLLV, shl>,
           SRLV_FM<4, 0>;
def SRLV : MMRel, shift_rotate_reg<"srlv", GPR32Opnd, II_SRLV, srl>,
           SRLV_FM<6, 0>;
def SRAV : MMRel, shift_rotate_reg<"srav", GPR32Opnd, II_SRAV, sra>,
           SRLV_FM<7, 0>;

// Rotate Instructions
def ROTR  : MMRel, shift_rotate_imm<"rotr", uimm5, GPR32Opnd, II_ROTR, rotr,
                                    immZExt5>,
            SRA_FM<2, 1>, ISA_MIPS32R2;
def ROTRV : MMRel, shift_rotate_reg<"rotrv", GPR32Opnd, II_ROTRV, rotr>,
            SRLV_FM<6, 1>, ISA_MIPS32R2;

/// Load and Store Instructions
///  aligned
def LB  : Load<"lb", GPR32Opnd, sextloadi8, II_LB>, MMRel, LW_FM<0x20>;
def LBu : Load<"lbu", GPR32Opnd, zextloadi8, II_LBU, addrDefault>, MMRel,
          LW_FM<0x24>;
def LH  : Load<"lh", GPR32Opnd, sextloadi16, II_LH, addrDefault>, MMRel,
          LW_FM<0x21>;
def LHu : Load<"lhu", GPR32Opnd, zextloadi16, II_LHU>, MMRel, LW_FM<0x25>;
def LW  : Load<"lw", GPR32Opnd, load, II_LW, addrDefault>, MMRel,
          LW_FM<0x23>;
def SB  : Store<"sb", GPR32Opnd, truncstorei8, II_SB>, MMRel, LW_FM<0x28>;
def SH  : Store<"sh", GPR32Opnd, truncstorei16, II_SH>, MMRel, LW_FM<0x29>;
def SW  : Store<"sw", GPR32Opnd, store, II_SW>, MMRel, LW_FM<0x2b>;

/// load/store left/right
let EncodingPredicates = []<Predicate>, // FIXME: Lack of HasStdEnc is probably a bug
    AdditionalPredicates = [NotInMicroMips] in {
def LWL : LoadLeftRight<"lwl", MipsLWL, GPR32Opnd, II_LWL>, LW_FM<0x22>,
          ISA_MIPS1_NOT_32R6_64R6;
def LWR : LoadLeftRight<"lwr", MipsLWR, GPR32Opnd, II_LWR>, LW_FM<0x26>,
          ISA_MIPS1_NOT_32R6_64R6;
def SWL : StoreLeftRight<"swl", MipsSWL, GPR32Opnd, II_SWL>, LW_FM<0x2a>,
          ISA_MIPS1_NOT_32R6_64R6;
def SWR : StoreLeftRight<"swr", MipsSWR, GPR32Opnd, II_SWR>, LW_FM<0x2e>,
          ISA_MIPS1_NOT_32R6_64R6;
}

def SYNC : MMRel, SYNC_FT<"sync">, SYNC_FM, ISA_MIPS32;
def TEQ : MMRel, TEQ_FT<"teq", GPR32Opnd>, TEQ_FM<0x34>;
def TGE : MMRel, TEQ_FT<"tge", GPR32Opnd>, TEQ_FM<0x30>;
def TGEU : MMRel, TEQ_FT<"tgeu", GPR32Opnd>, TEQ_FM<0x31>;
def TLT : MMRel, TEQ_FT<"tlt", GPR32Opnd>, TEQ_FM<0x32>;
def TLTU : MMRel, TEQ_FT<"tltu", GPR32Opnd>, TEQ_FM<0x33>;
def TNE : MMRel, TEQ_FT<"tne", GPR32Opnd>, TEQ_FM<0x36>;

def TEQI : MMRel, TEQI_FT<"teqi", GPR32Opnd>, TEQI_FM<0xc>,
           ISA_MIPS2_NOT_32R6_64R6;
def TGEI : MMRel, TEQI_FT<"tgei", GPR32Opnd>, TEQI_FM<0x8>,
           ISA_MIPS2_NOT_32R6_64R6;
def TGEIU : MMRel, TEQI_FT<"tgeiu", GPR32Opnd>, TEQI_FM<0x9>,
           ISA_MIPS2_NOT_32R6_64R6;
def TLTI : MMRel, TEQI_FT<"tlti", GPR32Opnd>, TEQI_FM<0xa>,
           ISA_MIPS2_NOT_32R6_64R6;
def TTLTIU : MMRel, TEQI_FT<"tltiu", GPR32Opnd>, TEQI_FM<0xb>,
           ISA_MIPS2_NOT_32R6_64R6;
def TNEI : MMRel, TEQI_FT<"tnei", GPR32Opnd>, TEQI_FM<0xe>,
           ISA_MIPS2_NOT_32R6_64R6;

def BREAK : MMRel, BRK_FT<"break">, BRK_FM<0xd>;
def SYSCALL : MMRel, SYS_FT<"syscall">, SYS_FM<0xc>;
def TRAP : TrapBase<BREAK>;
def SDBBP : SYS_FT<"sdbbp">, SDBBP_FM, ISA_MIPS32_NOT_32R6_64R6;

def ERET : MMRel, ER_FT<"eret">, ER_FM<0x18>, INSN_MIPS3_32;
def DERET : MMRel, ER_FT<"deret">, ER_FM<0x1f>, ISA_MIPS32;

def EI : MMRel, DEI_FT<"ei", GPR32Opnd>, EI_FM<1>, ISA_MIPS32R2;
def DI : MMRel, DEI_FT<"di", GPR32Opnd>, EI_FM<0>, ISA_MIPS32R2;

let EncodingPredicates = []<Predicate>, // FIXME: Lack of HasStdEnc is probably a bug
    AdditionalPredicates = [NotInMicroMips] in {
def WAIT : WAIT_FT<"wait">, WAIT_FM;

/// Load-linked, Store-conditional
def LL : LLBase<"ll", GPR32Opnd>, LW_FM<0x30>, ISA_MIPS2_NOT_32R6_64R6;
def SC : SCBase<"sc", GPR32Opnd>, LW_FM<0x38>, ISA_MIPS2_NOT_32R6_64R6;
}

/// Jump and Branch Instructions
def J       : MMRel, JumpFJ<jmptarget, "j", br, bb, "j">, FJ<2>,
              AdditionalRequires<[RelocStatic]>, IsBranch;
def JR      : MMRel, IndirectBranch<"jr", GPR32Opnd>, MTLO_FM<8>;
def BEQ     : MMRel, CBranch<"beq", brtarget, seteq, GPR32Opnd>, BEQ_FM<4>;
def BNE     : MMRel, CBranch<"bne", brtarget, setne, GPR32Opnd>, BEQ_FM<5>;
def BGEZ    : MMRel, CBranchZero<"bgez", brtarget, setge, GPR32Opnd>,
              BGEZ_FM<1, 1>;
def BGTZ    : MMRel, CBranchZero<"bgtz", brtarget, setgt, GPR32Opnd>,
              BGEZ_FM<7, 0>;
def BLEZ    : MMRel, CBranchZero<"blez", brtarget, setle, GPR32Opnd>,
              BGEZ_FM<6, 0>;
def BLTZ    : MMRel, CBranchZero<"bltz", brtarget, setlt, GPR32Opnd>,
              BGEZ_FM<1, 0>;
def B       : UncondBranch<BEQ>;

def JAL  : MMRel, JumpLink<"jal", calltarget>, FJ<3>;
let AdditionalPredicates = [NotInMicroMips] in {
  def JALR : JumpLinkReg<"jalr", GPR32Opnd>, JALR_FM;
  def JALRPseudo : JumpLinkRegPseudo<GPR32Opnd, JALR, RA>;
}

// FIXME: JALX really requires either MIPS16 or microMIPS in addition to MIPS32.
def JALX  : JumpLink<"jalx", calltarget>, FJ<0x1D>, ISA_MIPS32_NOT_32R6_64R6;
def BGEZAL : MMRel, BGEZAL_FT<"bgezal", brtarget, GPR32Opnd>, BGEZAL_FM<0x11>,
             ISA_MIPS1_NOT_32R6_64R6;
def BLTZAL : MMRel, BGEZAL_FT<"bltzal", brtarget, GPR32Opnd>, BGEZAL_FM<0x10>,
             ISA_MIPS1_NOT_32R6_64R6;
def BAL_BR : BAL_BR_Pseudo<BGEZAL>;
def TAILCALL : TailCall<J>;
def TAILCALL_R : TailCallReg<GPR32Opnd, JR>;

// Indirect branches are matched as PseudoIndirectBranch/PseudoIndirectBranch64
// then are expanded to JR, JR64, JALR, or JALR64 depending on the ISA.
class PseudoIndirectBranchBase<RegisterOperand RO> :
    MipsPseudo<(outs), (ins RO:$rs), [(brind RO:$rs)], IIBranch> {
  let isTerminator=1;
  let isBarrier=1;
  let hasDelaySlot = 1;
  let isBranch = 1;
  let isIndirectBranch = 1;
}

def PseudoIndirectBranch : PseudoIndirectBranchBase<GPR32Opnd>;

// Return instructions are matched as a RetRA instruction, then ar expanded
// into PseudoReturn/PseudoReturn64 after register allocation. Finally,
// MipsAsmPrinter expands this into JR, JR64, JALR, or JALR64 depending on the
// ISA.
class PseudoReturnBase<RegisterOperand RO> : MipsPseudo<(outs), (ins RO:$rs),
                                                        [], IIBranch> {
  let isTerminator = 1;
  let isBarrier = 1;
  let hasDelaySlot = 1;
  let isReturn = 1;
  let isCodeGenOnly = 1;
  let hasCtrlDep = 1;
  let hasExtraSrcRegAllocReq = 1;
}

def PseudoReturn : PseudoReturnBase<GPR32Opnd>;

// Exception handling related node and instructions.
// The conversion sequence is:
// ISD::EH_RETURN -> MipsISD::EH_RETURN ->
// MIPSeh_return -> (stack change + indirect branch)
//
// MIPSeh_return takes the place of regular return instruction
// but takes two arguments (V1, V0) which are used for storing
// the offset and return address respectively.
def SDT_MipsEHRET : SDTypeProfile<0, 2, [SDTCisInt<0>, SDTCisPtrTy<1>]>;

def MIPSehret : SDNode<"MipsISD::EH_RETURN", SDT_MipsEHRET,
                      [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

let Uses = [V0, V1], isTerminator = 1, isReturn = 1, isBarrier = 1 in {
  def MIPSeh_return32 : MipsPseudo<(outs), (ins GPR32:$spoff, GPR32:$dst),
                                [(MIPSehret GPR32:$spoff, GPR32:$dst)]>;
  def MIPSeh_return64 : MipsPseudo<(outs), (ins GPR64:$spoff,
                                                GPR64:$dst),
                                [(MIPSehret GPR64:$spoff, GPR64:$dst)]>;
}

/// Multiply and Divide Instructions.
def MULT  : MMRel, Mult<"mult", II_MULT, GPR32Opnd, [HI0, LO0]>,
            MULT_FM<0, 0x18>, ISA_MIPS1_NOT_32R6_64R6;
def MULTu : MMRel, Mult<"multu", II_MULTU, GPR32Opnd, [HI0, LO0]>,
            MULT_FM<0, 0x19>, ISA_MIPS1_NOT_32R6_64R6;
def SDIV  : MMRel, Div<"div", II_DIV, GPR32Opnd, [HI0, LO0]>,
            MULT_FM<0, 0x1a>, ISA_MIPS1_NOT_32R6_64R6;
def UDIV  : MMRel, Div<"divu", II_DIVU, GPR32Opnd, [HI0, LO0]>,
            MULT_FM<0, 0x1b>, ISA_MIPS1_NOT_32R6_64R6;

def MTHI : MMRel, MoveToLOHI<"mthi", GPR32Opnd, [HI0]>, MTLO_FM<0x11>,
           ISA_MIPS1_NOT_32R6_64R6;
def MTLO : MMRel, MoveToLOHI<"mtlo", GPR32Opnd, [LO0]>, MTLO_FM<0x13>,
           ISA_MIPS1_NOT_32R6_64R6;
let EncodingPredicates = []<Predicate>, // FIXME: Lack of HasStdEnc is probably a bug
    AdditionalPredicates = [NotInMicroMips] in {
def MFHI : MMRel, MoveFromLOHI<"mfhi", GPR32Opnd, AC0>, MFLO_FM<0x10>,
           ISA_MIPS1_NOT_32R6_64R6;
def MFLO : MMRel, MoveFromLOHI<"mflo", GPR32Opnd, AC0>, MFLO_FM<0x12>,
           ISA_MIPS1_NOT_32R6_64R6;
}

/// Sign Ext In Register Instructions.
def SEB : MMRel, SignExtInReg<"seb", i8, GPR32Opnd, II_SEB>,
          SEB_FM<0x10, 0x20>, ISA_MIPS32R2;
def SEH : MMRel, SignExtInReg<"seh", i16, GPR32Opnd, II_SEH>,
          SEB_FM<0x18, 0x20>, ISA_MIPS32R2;

/// Count Leading
def CLZ : MMRel, CountLeading0<"clz", GPR32Opnd>, CLO_FM<0x20>,
          ISA_MIPS32_NOT_32R6_64R6;
def CLO : MMRel, CountLeading1<"clo", GPR32Opnd>, CLO_FM<0x21>,
          ISA_MIPS32_NOT_32R6_64R6;

/// Word Swap Bytes Within Halfwords
def WSBH : MMRel, SubwordSwap<"wsbh", GPR32Opnd>, SEB_FM<2, 0x20>, ISA_MIPS32R2;

/// No operation.
def NOP : PseudoSE<(outs), (ins), []>, PseudoInstExpansion<(SLL ZERO, ZERO, 0)>;

// FrameIndexes are legalized when they are operands from load/store
// instructions. The same not happens for stack address copies, so an
// add op with mem ComplexPattern is used and the stack address copy
// can be matched. It's similar to Sparc LEA_ADDRi
def LEA_ADDiu : MMRel, EffectiveAddress<"addiu", GPR32Opnd>, LW_FM<9>;

// MADD*/MSUB*
def MADD  : MMRel, MArithR<"madd", II_MADD, 1>, MULT_FM<0x1c, 0>,
            ISA_MIPS32_NOT_32R6_64R6;
def MADDU : MMRel, MArithR<"maddu", II_MADDU, 1>, MULT_FM<0x1c, 1>,
            ISA_MIPS32_NOT_32R6_64R6;
def MSUB  : MMRel, MArithR<"msub", II_MSUB>, MULT_FM<0x1c, 4>,
            ISA_MIPS32_NOT_32R6_64R6;
def MSUBU : MMRel, MArithR<"msubu", II_MSUBU>, MULT_FM<0x1c, 5>,
            ISA_MIPS32_NOT_32R6_64R6;

let AdditionalPredicates = [NotDSP] in {
def PseudoMULT  : MultDivPseudo<MULT, ACC64, GPR32Opnd, MipsMult, II_MULT>,
                  ISA_MIPS1_NOT_32R6_64R6;
def PseudoMULTu : MultDivPseudo<MULTu, ACC64, GPR32Opnd, MipsMultu, II_MULTU>,
                  ISA_MIPS1_NOT_32R6_64R6;
def PseudoMFHI : PseudoMFLOHI<GPR32, ACC64, MipsMFHI>, ISA_MIPS1_NOT_32R6_64R6;
def PseudoMFLO : PseudoMFLOHI<GPR32, ACC64, MipsMFLO>, ISA_MIPS1_NOT_32R6_64R6;
def PseudoMTLOHI : PseudoMTLOHI<ACC64, GPR32>, ISA_MIPS1_NOT_32R6_64R6;
def PseudoMADD  : MAddSubPseudo<MADD, MipsMAdd, II_MADD>,
                  ISA_MIPS32_NOT_32R6_64R6;
def PseudoMADDU : MAddSubPseudo<MADDU, MipsMAddu, II_MADDU>,
                  ISA_MIPS32_NOT_32R6_64R6;
def PseudoMSUB  : MAddSubPseudo<MSUB, MipsMSub, II_MSUB>,
                  ISA_MIPS32_NOT_32R6_64R6;
def PseudoMSUBU : MAddSubPseudo<MSUBU, MipsMSubu, II_MSUBU>,
                  ISA_MIPS32_NOT_32R6_64R6;
}

def PseudoSDIV : MultDivPseudo<SDIV, ACC64, GPR32Opnd, MipsDivRem, II_DIV,
                               0, 1, 1>, ISA_MIPS1_NOT_32R6_64R6;
def PseudoUDIV : MultDivPseudo<UDIV, ACC64, GPR32Opnd, MipsDivRemU, II_DIVU,
                               0, 1, 1>, ISA_MIPS1_NOT_32R6_64R6;

def RDHWR : ReadHardware<GPR32Opnd, HWRegsOpnd>, RDHWR_FM;

def EXT : MMRel, ExtBase<"ext", GPR32Opnd, uimm5, MipsExt>, EXT_FM<0>;
def INS : MMRel, InsBase<"ins", GPR32Opnd, uimm5, MipsIns>, EXT_FM<4>;

/// Move Control Registers From/To CPU Registers
def MFC0 : MFC3OP<"mfc0", GPR32Opnd>, MFC3OP_FM<0x10, 0>, ISA_MIPS32;
def MTC0 : MFC3OP<"mtc0", GPR32Opnd>, MFC3OP_FM<0x10, 4>, ISA_MIPS32;
def MFC2 : MFC3OP<"mfc2", GPR32Opnd>, MFC3OP_FM<0x12, 0>;
def MTC2 : MFC3OP<"mtc2", GPR32Opnd>, MFC3OP_FM<0x12, 4>;

class Barrier<string asmstr> : InstSE<(outs), (ins), asmstr, [], NoItinerary,
                                      FrmOther>;
def SSNOP : Barrier<"ssnop">, BARRIER_FM<1>;
def EHB : Barrier<"ehb">, BARRIER_FM<3>;
def PAUSE : Barrier<"pause">, BARRIER_FM<5>, ISA_MIPS32R2;

// JR_HB and JALR_HB are defined here using the new style naming
// scheme because some of this code is shared with Mips32r6InstrInfo.td
// and because of that it doesn't follow the naming convention of the
// rest of the file. To avoid a mixture of old vs new style, the new
// style was chosen.
class JR_HB_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rs");
  list<dag> Pattern = [];
}

class JALR_HB_DESC_BASE<string instr_asm, RegisterOperand GPROpnd> {
  dag OutOperandList = (outs GPROpnd:$rd);
  dag InOperandList = (ins GPROpnd:$rs);
  string AsmString = !strconcat(instr_asm, "\t$rd, $rs");
  list<dag> Pattern = [];
}

class JR_HB_DESC : InstSE<(outs), (ins), "", [], NoItinerary, FrmJ>,
                   JR_HB_DESC_BASE<"jr.hb", GPR32Opnd> {
  let isBranch=1;
  let isIndirectBranch=1;
  let hasDelaySlot=1;
  let isTerminator=1;
  let isBarrier=1;
}

class JALR_HB_DESC : InstSE<(outs), (ins), "", [], NoItinerary, FrmJ>,
                     JALR_HB_DESC_BASE<"jalr.hb", GPR32Opnd> {
  let isIndirectBranch=1;
  let hasDelaySlot=1;
}

class JR_HB_ENC : JR_HB_FM<8>;
class JALR_HB_ENC : JALR_HB_FM<9>;

def JR_HB : JR_HB_DESC, JR_HB_ENC, ISA_MIPS32_NOT_32R6_64R6;
def JALR_HB : JALR_HB_DESC, JALR_HB_ENC, ISA_MIPS32;

class TLB<string asmstr> : InstSE<(outs), (ins), asmstr, [], NoItinerary,
                                      FrmOther>;
def TLBP : TLB<"tlbp">, COP0_TLB_FM<0x08>;
def TLBR : TLB<"tlbr">, COP0_TLB_FM<0x01>;
def TLBWI : TLB<"tlbwi">, COP0_TLB_FM<0x02>;
def TLBWR : TLB<"tlbwr">, COP0_TLB_FM<0x06>;

class CacheOp<string instr_asm, Operand MemOpnd> :
    InstSE<(outs), (ins  MemOpnd:$addr, uimm5:$hint),
           !strconcat(instr_asm, "\t$hint, $addr"), [], NoItinerary, FrmOther> {
  let DecoderMethod = "DecodeCacheOp";
}

def CACHE : CacheOp<"cache", mem>, CACHEOP_FM<0b101111>,
            INSN_MIPS3_32_NOT_32R6_64R6;
def PREF :  CacheOp<"pref", mem>, CACHEOP_FM<0b110011>,
            INSN_MIPS3_32_NOT_32R6_64R6;

//===----------------------------------------------------------------------===//
// Instruction aliases
//===----------------------------------------------------------------------===//
def : MipsInstAlias<"move $dst, $src",
                    (ADDu GPR32Opnd:$dst, GPR32Opnd:$src,ZERO), 1>,
      GPR_32 {
  let AdditionalPredicates = [NotInMicroMips];
}
def : MipsInstAlias<"bal $offset", (BGEZAL ZERO, brtarget:$offset), 0>,
      ISA_MIPS1_NOT_32R6_64R6;
def : MipsInstAlias<"addu $rs, $rt, $imm",
                    (ADDiu GPR32Opnd:$rs, GPR32Opnd:$rt, simm16:$imm), 0>;
def : MipsInstAlias<"add $rs, $rt, $imm",
                    (ADDi GPR32Opnd:$rs, GPR32Opnd:$rt, simm16:$imm), 0>;
def : MipsInstAlias<"and $rs, $rt, $imm",
                    (ANDi GPR32Opnd:$rs, GPR32Opnd:$rt, simm16:$imm), 0>;
def : MipsInstAlias<"and $rs, $imm",
                    (ANDi GPR32Opnd:$rs, GPR32Opnd:$rs, simm16:$imm), 0>;
def : MipsInstAlias<"j $rs", (JR GPR32Opnd:$rs), 0>;
let Predicates = [NotInMicroMips] in {
def : MipsInstAlias<"jalr $rs", (JALR RA, GPR32Opnd:$rs), 0>;
}
def : MipsInstAlias<"jal $rs", (JALR RA, GPR32Opnd:$rs), 0>;
def : MipsInstAlias<"jal $rd,$rs", (JALR GPR32Opnd:$rd, GPR32Opnd:$rs), 0>;
def : MipsInstAlias<"jalr.hb $rs", (JALR_HB RA, GPR32Opnd:$rs), 1>, ISA_MIPS32;
def : MipsInstAlias<"not $rt, $rs",
                    (NOR GPR32Opnd:$rt, GPR32Opnd:$rs, ZERO), 0>;
def : MipsInstAlias<"neg $rt, $rs",
                    (SUB GPR32Opnd:$rt, ZERO, GPR32Opnd:$rs), 1>;
def : MipsInstAlias<"negu $rt",
                    (SUBu GPR32Opnd:$rt, ZERO, GPR32Opnd:$rt), 0>;
def : MipsInstAlias<"negu $rt, $rs",
                    (SUBu GPR32Opnd:$rt, ZERO, GPR32Opnd:$rs), 1>;
def : MipsInstAlias<"slt $rs, $rt, $imm",
                    (SLTi GPR32Opnd:$rs, GPR32Opnd:$rt, simm16:$imm), 0>;
def : MipsInstAlias<"sltu $rt, $rs, $imm",
                    (SLTiu GPR32Opnd:$rt, GPR32Opnd:$rs, simm16:$imm), 0>;
def : MipsInstAlias<"xor $rs, $rt, $imm",
                    (XORi GPR32Opnd:$rs, GPR32Opnd:$rt, uimm16:$imm), 0>;
def : MipsInstAlias<"or $rs, $rt, $imm",
                    (ORi GPR32Opnd:$rs, GPR32Opnd:$rt, uimm16:$imm), 0>;
def : MipsInstAlias<"or $rs, $imm",
                    (ORi GPR32Opnd:$rs, GPR32Opnd:$rs, uimm16:$imm), 0>;
def : MipsInstAlias<"nop", (SLL ZERO, ZERO, 0), 1>;
def : MipsInstAlias<"mfc0 $rt, $rd", (MFC0 GPR32Opnd:$rt, GPR32Opnd:$rd, 0), 0>;
def : MipsInstAlias<"mtc0 $rt, $rd", (MTC0 GPR32Opnd:$rt, GPR32Opnd:$rd, 0), 0>;
def : MipsInstAlias<"mfc2 $rt, $rd", (MFC2 GPR32Opnd:$rt, GPR32Opnd:$rd, 0), 0>;
def : MipsInstAlias<"mtc2 $rt, $rd", (MTC2 GPR32Opnd:$rt, GPR32Opnd:$rd, 0), 0>;
def : MipsInstAlias<"b $offset", (BEQ ZERO, ZERO, brtarget:$offset), 0>;
def : MipsInstAlias<"bnez $rs,$offset",
                    (BNE GPR32Opnd:$rs, ZERO, brtarget:$offset), 0>;
def : MipsInstAlias<"beqz $rs,$offset",
                    (BEQ GPR32Opnd:$rs, ZERO, brtarget:$offset), 0>;
def : MipsInstAlias<"syscall", (SYSCALL 0), 1>;
    
def : MipsInstAlias<"break", (BREAK 0, 0), 1>;
def : MipsInstAlias<"break $imm", (BREAK uimm10:$imm, 0), 1>;
def : MipsInstAlias<"ei", (EI ZERO), 1>;
def : MipsInstAlias<"di", (DI ZERO), 1>;

def  : MipsInstAlias<"teq $rs, $rt", (TEQ GPR32Opnd:$rs, GPR32Opnd:$rt, 0), 1>;
def  : MipsInstAlias<"tge $rs, $rt", (TGE GPR32Opnd:$rs, GPR32Opnd:$rt, 0), 1>;
def  : MipsInstAlias<"tgeu $rs, $rt", (TGEU GPR32Opnd:$rs, GPR32Opnd:$rt, 0),
                     1>;
def  : MipsInstAlias<"tlt $rs, $rt", (TLT GPR32Opnd:$rs, GPR32Opnd:$rt, 0), 1>;
def  : MipsInstAlias<"tltu $rs, $rt", (TLTU GPR32Opnd:$rs, GPR32Opnd:$rt, 0),
                     1>;
def  : MipsInstAlias<"tne $rs, $rt", (TNE GPR32Opnd:$rs, GPR32Opnd:$rt, 0), 1>;
def  : MipsInstAlias<"sll $rd, $rt, $rs",
                     (SLLV GPR32Opnd:$rd, GPR32Opnd:$rt, GPR32Opnd:$rs), 0>;
def : MipsInstAlias<"sub, $rd, $rs, $imm",
                    (ADDi GPR32Opnd:$rd, GPR32Opnd:$rs,
                          InvertedImOperand:$imm), 0>;
def : MipsInstAlias<"sub $rs, $imm",
                    (ADDi GPR32Opnd:$rs, GPR32Opnd:$rs, InvertedImOperand:$imm),
                    0>;
def : MipsInstAlias<"subu, $rd, $rs, $imm",
                    (ADDiu GPR32Opnd:$rd, GPR32Opnd:$rs,
                           InvertedImOperand:$imm), 0>;
def : MipsInstAlias<"subu $rs, $imm", (ADDiu GPR32Opnd:$rs, GPR32Opnd:$rs,
                                             InvertedImOperand:$imm), 0>;
def : MipsInstAlias<"sra $rd, $rt, $rs",
                    (SRAV GPR32Opnd:$rd, GPR32Opnd:$rt, GPR32Opnd:$rs), 0>;
def : MipsInstAlias<"srl $rd, $rt, $rs",
                    (SRLV GPR32Opnd:$rd, GPR32Opnd:$rt, GPR32Opnd:$rs), 0>;
def : MipsInstAlias<"sdbbp", (SDBBP 0)>, ISA_MIPS32_NOT_32R6_64R6;
def : MipsInstAlias<"sync",
                    (SYNC 0), 1>, ISA_MIPS2;
//===----------------------------------------------------------------------===//
// Assembler Pseudo Instructions
//===----------------------------------------------------------------------===//

class LoadImm32< string instr_asm, Operand Od, RegisterOperand RO> :
  MipsAsmPseudoInst<(outs RO:$rt), (ins Od:$imm32),
                     !strconcat(instr_asm, "\t$rt, $imm32")> ;
def LoadImm32Reg : LoadImm32<"li", uimm5, GPR32Opnd>;

class LoadAddress<string instr_asm, Operand MemOpnd, RegisterOperand RO> :
  MipsAsmPseudoInst<(outs RO:$rt), (ins MemOpnd:$addr),
                     !strconcat(instr_asm, "\t$rt, $addr")> ;
def LoadAddr32Reg : LoadAddress<"la", mem, GPR32Opnd>;

class LoadAddressImm<string instr_asm, Operand Od, RegisterOperand RO> :
  MipsAsmPseudoInst<(outs RO:$rt), (ins Od:$imm32),
                     !strconcat(instr_asm, "\t$rt, $imm32")> ;
def LoadAddr32Imm : LoadAddressImm<"la", uimm5, GPR32Opnd>;

//===----------------------------------------------------------------------===//
//  Arbitrary patterns that map to one or more instructions
//===----------------------------------------------------------------------===//

// Load/store pattern templates.
class LoadRegImmPat<Instruction LoadInst, ValueType ValTy, PatFrag Node> :
  MipsPat<(ValTy (Node addrRegImm:$a)), (LoadInst addrRegImm:$a)>;

class StoreRegImmPat<Instruction StoreInst, ValueType ValTy> :
  MipsPat<(store ValTy:$v, addrRegImm:$a), (StoreInst ValTy:$v, addrRegImm:$a)>;

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
def : MipsPat<(subc GPR32:$lhs, GPR32:$rhs),
              (SUBu GPR32:$lhs, GPR32:$rhs)>;
let AdditionalPredicates = [NotDSP] in {
  def : MipsPat<(addc GPR32:$lhs, GPR32:$rhs),
                (ADDu GPR32:$lhs, GPR32:$rhs)>;
  def : MipsPat<(addc  GPR32:$src, immSExt16:$imm),
                (ADDiu GPR32:$src, imm:$imm)>;
}

// SYNC
def : MipsPat<(MipsSync (i32 immz)),
              (SYNC 0)>, ISA_MIPS2;

// Call
def : MipsPat<(MipsJmpLink (i32 tglobaladdr:$dst)),
              (JAL tglobaladdr:$dst)>;
def : MipsPat<(MipsJmpLink (i32 texternalsym:$dst)),
              (JAL texternalsym:$dst)>;
//def : MipsPat<(MipsJmpLink GPR32:$dst),
//              (JALR GPR32:$dst)>;

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
def : MipsPat<(MipsHi texternalsym:$in), (LUi texternalsym:$in)>;

def : MipsPat<(MipsLo tglobaladdr:$in), (ADDiu ZERO, tglobaladdr:$in)>;
def : MipsPat<(MipsLo tblockaddress:$in), (ADDiu ZERO, tblockaddress:$in)>;
def : MipsPat<(MipsLo tjumptable:$in), (ADDiu ZERO, tjumptable:$in)>;
def : MipsPat<(MipsLo tconstpool:$in), (ADDiu ZERO, tconstpool:$in)>;
def : MipsPat<(MipsLo tglobaltlsaddr:$in), (ADDiu ZERO, tglobaltlsaddr:$in)>;
def : MipsPat<(MipsLo texternalsym:$in), (ADDiu ZERO, texternalsym:$in)>;

def : MipsPat<(add GPR32:$hi, (MipsLo tglobaladdr:$lo)),
              (ADDiu GPR32:$hi, tglobaladdr:$lo)>;
def : MipsPat<(add GPR32:$hi, (MipsLo tblockaddress:$lo)),
              (ADDiu GPR32:$hi, tblockaddress:$lo)>;
def : MipsPat<(add GPR32:$hi, (MipsLo tjumptable:$lo)),
              (ADDiu GPR32:$hi, tjumptable:$lo)>;
def : MipsPat<(add GPR32:$hi, (MipsLo tconstpool:$lo)),
              (ADDiu GPR32:$hi, tconstpool:$lo)>;
def : MipsPat<(add GPR32:$hi, (MipsLo tglobaltlsaddr:$lo)),
              (ADDiu GPR32:$hi, tglobaltlsaddr:$lo)>;

// gp_rel relocs
def : MipsPat<(add GPR32:$gp, (MipsGPRel tglobaladdr:$in)),
              (ADDiu GPR32:$gp, tglobaladdr:$in)>;
def : MipsPat<(add GPR32:$gp, (MipsGPRel tconstpool:$in)),
              (ADDiu GPR32:$gp, tconstpool:$in)>;

// wrapper_pic
class WrapperPat<SDNode node, Instruction ADDiuOp, RegisterClass RC>:
      MipsPat<(MipsWrapper RC:$gp, node:$in),
              (ADDiuOp RC:$gp, node:$in)>;

def : WrapperPat<tglobaladdr, ADDiu, GPR32>;
def : WrapperPat<tconstpool, ADDiu, GPR32>;
def : WrapperPat<texternalsym, ADDiu, GPR32>;
def : WrapperPat<tblockaddress, ADDiu, GPR32>;
def : WrapperPat<tjumptable, ADDiu, GPR32>;
def : WrapperPat<tglobaltlsaddr, ADDiu, GPR32>;

// Mips does not have "not", so we expand our way
def : MipsPat<(not GPR32:$in),
              (NOR GPR32Opnd:$in, ZERO)>;

// extended loads
def : MipsPat<(i32 (extloadi1  addr:$src)), (LBu addr:$src)>;
def : MipsPat<(i32 (extloadi8  addr:$src)), (LBu addr:$src)>;
def : MipsPat<(i32 (extloadi16 addr:$src)), (LHu addr:$src)>;

// peepholes
def : MipsPat<(store (i32 0), addr:$dst), (SW ZERO, addr:$dst)>;

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
def : MipsPat<(brcond (i32 (setgt RC:$lhs, immSExt16Plus1:$rhs)), bb:$dst),
              (BEQ (SLTiOp RC:$lhs, (Plus1 imm:$rhs)), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setugt RC:$lhs, immSExt16Plus1:$rhs)), bb:$dst),
              (BEQ (SLTiuOp RC:$lhs, (Plus1 imm:$rhs)), ZERO, bb:$dst)>;

def : MipsPat<(brcond (i32 (setle RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;
def : MipsPat<(brcond (i32 (setule RC:$lhs, RC:$rhs)), bb:$dst),
              (BEQ (SLTuOp RC:$rhs, RC:$lhs), ZERO, bb:$dst)>;

def : MipsPat<(brcond RC:$cond, bb:$dst),
              (BNEOp RC:$cond, ZEROReg, bb:$dst)>;
}

defm : BrcondPats<GPR32, BEQ, BNE, SLT, SLTu, SLTi, SLTiu, ZERO>;

def : MipsPat<(brcond (i32 (setlt i32:$lhs, 1)), bb:$dst),
              (BLEZ i32:$lhs, bb:$dst)>;
def : MipsPat<(brcond (i32 (setgt i32:$lhs, -1)), bb:$dst),
              (BGEZ i32:$lhs, bb:$dst)>;

// setcc patterns
multiclass SeteqPats<RegisterClass RC, Instruction SLTiuOp, Instruction XOROp,
                     Instruction SLTuOp, Register ZEROReg> {
  def : MipsPat<(seteq RC:$lhs, 0),
                (SLTiuOp RC:$lhs, 1)>;
  def : MipsPat<(setne RC:$lhs, 0),
                (SLTuOp ZEROReg, RC:$lhs)>;
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

defm : SeteqPats<GPR32, SLTiu, XOR, SLTu, ZERO>;
defm : SetlePats<GPR32, SLT, SLTu>;
defm : SetgtPats<GPR32, SLT, SLTu>;
defm : SetgePats<GPR32, SLT, SLTu>;
defm : SetgeImmPats<GPR32, SLTi, SLTiu>;

// bswap pattern
def : MipsPat<(bswap GPR32:$rt), (ROTR (WSBH GPR32:$rt), 16)>;

// Load halfword/word patterns.
let AddedComplexity = 40 in {
  def : LoadRegImmPat<LBu, i32, zextloadi8>;
  def : LoadRegImmPat<LH, i32, sextloadi16>;
  def : LoadRegImmPat<LW, i32, load>;
}

//===----------------------------------------------------------------------===//
// Floating Point Support
//===----------------------------------------------------------------------===//

include "MipsInstrFPU.td"
include "Mips64InstrInfo.td"
include "MipsCondMov.td"

include "Mips32r6InstrInfo.td"
include "Mips64r6InstrInfo.td"

//
// Mips16

include "Mips16InstrFormats.td"
include "Mips16InstrInfo.td"

// DSP
include "MipsDSPInstrFormats.td"
include "MipsDSPInstrInfo.td"

// MSA
include "MipsMSAInstrFormats.td"
include "MipsMSAInstrInfo.td"

// Micromips
include "MicroMipsInstrFormats.td"
include "MicroMipsInstrInfo.td"
include "MicroMipsInstrFPU.td"
