//===-- X86InstrInfo.td - Main X86 Instruction Definition --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 instruction set, defining the instructions, and
// properties of the instructions which are needed for code generation, machine
// code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 specific DAG Nodes.
//

def SDTIntShiftDOp: SDTypeProfile<1, 3,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                   SDTCisInt<0>, SDTCisInt<3>]>;

def SDTX86CmpTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisSameAs<1, 2>]>;

def SDTX86Cmpsd : SDTypeProfile<1, 3, [SDTCisVT<0, f64>, SDTCisSameAs<1, 2>, SDTCisVT<3, i8>]>;
def SDTX86Cmpss : SDTypeProfile<1, 3, [SDTCisVT<0, f32>, SDTCisSameAs<1, 2>, SDTCisVT<3, i8>]>;

def SDTX86Cmov    : SDTypeProfile<1, 4,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>, SDTCisVT<4, i32>]>;

// Unary and binary operator instructions that set EFLAGS as a side-effect.
def SDTUnaryArithWithFlags : SDTypeProfile<2, 1,
                                           [SDTCisInt<0>, SDTCisVT<1, i32>]>;

def SDTBinaryArithWithFlags : SDTypeProfile<2, 2,
                                            [SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<0, 3>,
                                             SDTCisInt<0>, SDTCisVT<1, i32>]>;

// SDTBinaryArithWithFlagsInOut - RES1, EFLAGS = op LHS, RHS, EFLAGS
def SDTBinaryArithWithFlagsInOut : SDTypeProfile<2, 3,
                                            [SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<0, 3>,
                                             SDTCisInt<0>,
                                             SDTCisVT<1, i32>,
                                             SDTCisVT<4, i32>]>;
// RES1, RES2, FLAGS = op LHS, RHS
def SDT2ResultBinaryArithWithFlags : SDTypeProfile<3, 2,
                                            [SDTCisSameAs<0, 1>,
                                             SDTCisSameAs<0, 2>,
                                             SDTCisSameAs<0, 3>,
                                             SDTCisInt<0>, SDTCisVT<1, i32>]>;
def SDTX86BrCond  : SDTypeProfile<0, 3,
                                  [SDTCisVT<0, OtherVT>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86SetCC   : SDTypeProfile<1, 2,
                                  [SDTCisVT<0, i8>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;
def SDTX86SetCC_C : SDTypeProfile<1, 2,
                                  [SDTCisInt<0>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86sahf : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i8>]>;

def SDTX86rdrand : SDTypeProfile<2, 0, [SDTCisInt<0>, SDTCisVT<1, i32>]>;

def SDTX86cas : SDTypeProfile<0, 3, [SDTCisPtrTy<0>, SDTCisInt<1>,
                                     SDTCisVT<2, i8>]>;
def SDTX86caspair : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDTX86atomicBinary : SDTypeProfile<2, 3, [SDTCisInt<0>, SDTCisInt<1>,
                                SDTCisPtrTy<2>, SDTCisInt<3>,SDTCisInt<4>]>;
def SDTX86Ret     : SDTypeProfile<0, -1, [SDTCisVT<0, i16>]>;

def SDT_X86CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>]>;
def SDT_X86CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>]>;

def SDT_X86Call   : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def SDT_X86VASTART_SAVE_XMM_REGS : SDTypeProfile<0, -1, [SDTCisVT<0, i8>,
                                                         SDTCisVT<1, iPTR>,
                                                         SDTCisVT<2, iPTR>]>;

def SDT_X86VAARG_64 : SDTypeProfile<1, -1, [SDTCisPtrTy<0>,
                                            SDTCisPtrTy<1>,
                                            SDTCisVT<2, i32>,
                                            SDTCisVT<3, i8>,
                                            SDTCisVT<4, i32>]>;

def SDTX86RepStr  : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;

def SDTX86Void    : SDTypeProfile<0, 0, []>;

def SDTX86Wrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def SDT_X86TLSADDR : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TLSBASEADDR : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TLSCALL : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86SEG_ALLOCA : SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;

def SDT_X86WIN_FTOL : SDTypeProfile<0, 1, [SDTCisFP<0>]>;

def SDT_X86EHRET : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TCRET : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i32>]>;

def SDT_X86MEMBARRIER : SDTypeProfile<0, 0, []>;

def X86MemBarrier : SDNode<"X86ISD::MEMBARRIER", SDT_X86MEMBARRIER,
                            [SDNPHasChain,SDNPSideEffect]>;
def X86MFence : SDNode<"X86ISD::MFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;
def X86SFence : SDNode<"X86ISD::SFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;
def X86LFence : SDNode<"X86ISD::LFENCE", SDT_X86MEMBARRIER,
                        [SDNPHasChain]>;


def X86bsf     : SDNode<"X86ISD::BSF",      SDTUnaryArithWithFlags>;
def X86bsr     : SDNode<"X86ISD::BSR",      SDTUnaryArithWithFlags>;
def X86shld    : SDNode<"X86ISD::SHLD",     SDTIntShiftDOp>;
def X86shrd    : SDNode<"X86ISD::SHRD",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest>;
def X86bt      : SDNode<"X86ISD::BT",       SDTX86CmpTest>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC>;
def X86setcc_c : SDNode<"X86ISD::SETCC_CARRY", SDTX86SetCC_C>;

def X86sahf    : SDNode<"X86ISD::SAHF",     SDTX86sahf>;

def X86rdrand  : SDNode<"X86ISD::RDRAND",   SDTX86rdrand,
                        [SDNPHasChain, SDNPSideEffect]>;

def X86cas : SDNode<"X86ISD::LCMPXCHG_DAG", SDTX86cas,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86cas8 : SDNode<"X86ISD::LCMPXCHG8_DAG", SDTX86caspair,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86cas16 : SDNode<"X86ISD::LCMPXCHG16_DAG", SDTX86caspair,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;

def X86AtomAdd64 : SDNode<"X86ISD::ATOMADD64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomSub64 : SDNode<"X86ISD::ATOMSUB64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomOr64 : SDNode<"X86ISD::ATOMOR64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomXor64 : SDNode<"X86ISD::ATOMXOR64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomAnd64 : SDNode<"X86ISD::ATOMAND64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomNand64 : SDNode<"X86ISD::ATOMNAND64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86AtomSwap64 : SDNode<"X86ISD::ATOMSWAP64_DAG", SDTX86atomicBinary,
                        [SDNPHasChain, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86retflag : SDNode<"X86ISD::RET_FLAG", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def X86vastart_save_xmm_regs :
                 SDNode<"X86ISD::VASTART_SAVE_XMM_REGS",
                        SDT_X86VASTART_SAVE_XMM_REGS,
                        [SDNPHasChain, SDNPVariadic]>;
def X86vaarg64 :
                 SDNode<"X86ISD::VAARG_64", SDT_X86VAARG_64,
                        [SDNPHasChain, SDNPMayLoad, SDNPMayStore,
                         SDNPMemOperand]>;
def X86callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_X86CallSeqStart,
                        [SDNPHasChain, SDNPOutGlue]>;
def X86callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_X86CallSeqEnd,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def X86call    : SDNode<"X86ISD::CALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                         SDNPVariadic]>;

def X86rep_stos: SDNode<"X86ISD::REP_STOS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore]>;
def X86rep_movs: SDNode<"X86ISD::REP_MOVS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad]>;

def X86rdtsc   : SDNode<"X86ISD::RDTSC_DAG", SDTX86Void,
                        [SDNPHasChain, SDNPOutGlue, SDNPSideEffect]>;

def X86Wrapper    : SDNode<"X86ISD::Wrapper",     SDTX86Wrapper>;
def X86WrapperRIP : SDNode<"X86ISD::WrapperRIP",  SDTX86Wrapper>;

def X86tlsaddr : SDNode<"X86ISD::TLSADDR", SDT_X86TLSADDR,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def X86tlsbaseaddr : SDNode<"X86ISD::TLSBASEADDR", SDT_X86TLSBASEADDR,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def X86ehret : SDNode<"X86ISD::EH_RETURN", SDT_X86EHRET,
                        [SDNPHasChain]>;

def X86eh_sjlj_setjmp  : SDNode<"X86ISD::EH_SJLJ_SETJMP",
                                SDTypeProfile<1, 1, [SDTCisInt<0>,
                                                     SDTCisPtrTy<1>]>,
                                [SDNPHasChain, SDNPSideEffect]>;
def X86eh_sjlj_longjmp : SDNode<"X86ISD::EH_SJLJ_LONGJMP",
                                SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>,
                                [SDNPHasChain, SDNPSideEffect]>;

def X86tcret : SDNode<"X86ISD::TC_RETURN", SDT_X86TCRET,
                        [SDNPHasChain,  SDNPOptInGlue, SDNPVariadic]>;

def X86add_flag  : SDNode<"X86ISD::ADD",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86sub_flag  : SDNode<"X86ISD::SUB",  SDTBinaryArithWithFlags>;
def X86smul_flag : SDNode<"X86ISD::SMUL", SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86umul_flag : SDNode<"X86ISD::UMUL", SDT2ResultBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86adc_flag  : SDNode<"X86ISD::ADC",  SDTBinaryArithWithFlagsInOut>;
def X86sbb_flag  : SDNode<"X86ISD::SBB",  SDTBinaryArithWithFlagsInOut>;

def X86inc_flag  : SDNode<"X86ISD::INC",  SDTUnaryArithWithFlags>;
def X86dec_flag  : SDNode<"X86ISD::DEC",  SDTUnaryArithWithFlags>;
def X86or_flag   : SDNode<"X86ISD::OR",   SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86xor_flag  : SDNode<"X86ISD::XOR",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86and_flag  : SDNode<"X86ISD::AND",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86andn_flag : SDNode<"X86ISD::ANDN", SDTBinaryArithWithFlags>;

def X86blsi_flag : SDNode<"X86ISD::BLSI",  SDTUnaryArithWithFlags>;
def X86blsmsk_flag : SDNode<"X86ISD::BLSMSK",  SDTUnaryArithWithFlags>;
def X86blsr_flag : SDNode<"X86ISD::BLSR",  SDTUnaryArithWithFlags>;

def X86mul_imm : SDNode<"X86ISD::MUL_IMM", SDTIntBinOp>;

def X86WinAlloca : SDNode<"X86ISD::WIN_ALLOCA", SDTX86Void,
                          [SDNPHasChain, SDNPInGlue, SDNPOutGlue]>;

def X86SegAlloca : SDNode<"X86ISD::SEG_ALLOCA", SDT_X86SEG_ALLOCA,
                          [SDNPHasChain]>;

def X86TLSCall : SDNode<"X86ISD::TLSCALL", SDT_X86TLSCALL,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def X86WinFTOL : SDNode<"X86ISD::WIN_FTOL", SDT_X86WIN_FTOL,
                        [SDNPHasChain, SDNPOutGlue]>;

//===----------------------------------------------------------------------===//
// X86 Operand Definitions.
//

// A version of ptr_rc which excludes SP, ESP, and RSP. This is used for
// the index operand of an address, to conform to x86 encoding restrictions.
def ptr_rc_nosp : PointerLikeRegClass<1>;

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
def X86MemAsmOperand : AsmOperandClass { 
 let Name = "Mem"; let PredicateMethod = "isMem"; 
}
def X86Mem8AsmOperand : AsmOperandClass { 
  let Name = "Mem8"; let PredicateMethod = "isMem8";
}
def X86Mem16AsmOperand : AsmOperandClass { 
  let Name = "Mem16"; let PredicateMethod = "isMem16";
}
def X86Mem32AsmOperand : AsmOperandClass { 
  let Name = "Mem32"; let PredicateMethod = "isMem32";
}
def X86Mem64AsmOperand : AsmOperandClass { 
  let Name = "Mem64"; let PredicateMethod = "isMem64";
}
def X86Mem80AsmOperand : AsmOperandClass { 
  let Name = "Mem80"; let PredicateMethod = "isMem80";
}
def X86Mem128AsmOperand : AsmOperandClass { 
  let Name = "Mem128"; let PredicateMethod = "isMem128";
}
def X86Mem256AsmOperand : AsmOperandClass { 
  let Name = "Mem256"; let PredicateMethod = "isMem256";
}

// Gather mem operands
def X86MemVX32Operand : AsmOperandClass {
  let Name = "MemVX32"; let PredicateMethod = "isMemVX32";
}
def X86MemVY32Operand : AsmOperandClass {
  let Name = "MemVY32"; let PredicateMethod = "isMemVY32";
}
def X86MemVX64Operand : AsmOperandClass {
  let Name = "MemVX64"; let PredicateMethod = "isMemVX64";
}
def X86MemVY64Operand : AsmOperandClass {
  let Name = "MemVY64"; let PredicateMethod = "isMemVY64";
}

def X86AbsMemAsmOperand : AsmOperandClass {
  let Name = "AbsMem";
  let SuperClasses = [X86MemAsmOperand];
}
class X86MemOperand<string printMethod> : Operand<iPTR> {
  let PrintMethod = printMethod;
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}

let OperandType = "OPERAND_MEMORY" in {
def opaque32mem : X86MemOperand<"printopaquemem">;
def opaque48mem : X86MemOperand<"printopaquemem">;
def opaque80mem : X86MemOperand<"printopaquemem">;
def opaque512mem : X86MemOperand<"printopaquemem">;

def i8mem   : X86MemOperand<"printi8mem"> { 
  let ParserMatchClass = X86Mem8AsmOperand; }
def i16mem  : X86MemOperand<"printi16mem"> { 
  let ParserMatchClass = X86Mem16AsmOperand; }
def i32mem  : X86MemOperand<"printi32mem"> { 
  let ParserMatchClass = X86Mem32AsmOperand; }
def i64mem  : X86MemOperand<"printi64mem"> { 
  let ParserMatchClass = X86Mem64AsmOperand; }
def i128mem : X86MemOperand<"printi128mem"> { 
  let ParserMatchClass = X86Mem128AsmOperand; }
def i256mem : X86MemOperand<"printi256mem"> { 
  let ParserMatchClass = X86Mem256AsmOperand; }
def f32mem  : X86MemOperand<"printf32mem"> { 
  let ParserMatchClass = X86Mem32AsmOperand; }
def f64mem  : X86MemOperand<"printf64mem"> { 
  let ParserMatchClass = X86Mem64AsmOperand; }
def f80mem  : X86MemOperand<"printf80mem"> { 
  let ParserMatchClass = X86Mem80AsmOperand; }
def f128mem : X86MemOperand<"printf128mem"> { 
  let ParserMatchClass = X86Mem128AsmOperand; }
def f256mem : X86MemOperand<"printf256mem">{ 
  let ParserMatchClass = X86Mem256AsmOperand; }

// Gather mem operands
def vx32mem : X86MemOperand<"printi32mem">{
  let MIOperandInfo = (ops ptr_rc, i8imm, VR128, i32imm, i8imm);
  let ParserMatchClass = X86MemVX32Operand; }
def vy32mem : X86MemOperand<"printi32mem">{
  let MIOperandInfo = (ops ptr_rc, i8imm, VR256, i32imm, i8imm);
  let ParserMatchClass = X86MemVY32Operand; }
def vx64mem : X86MemOperand<"printi64mem">{
  let MIOperandInfo = (ops ptr_rc, i8imm, VR128, i32imm, i8imm);
  let ParserMatchClass = X86MemVX64Operand; }
def vy64mem : X86MemOperand<"printi64mem">{
  let MIOperandInfo = (ops ptr_rc, i8imm, VR256, i32imm, i8imm);
  let ParserMatchClass = X86MemVY64Operand; }
}

// A version of i8mem for use on x86-64 that uses GR64_NOREX instead of
// plain GR64, so that it doesn't potentially require a REX prefix.
def i8mem_NOREX : Operand<i64> {
  let PrintMethod = "printi8mem";
  let MIOperandInfo = (ops GR64_NOREX, i8imm, GR64_NOREX_NOSP, i32imm, i8imm);
  let ParserMatchClass = X86Mem8AsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

// GPRs available for tailcall.
// It represents GR32_TC, GR64_TC or GR64_TCW64.
def ptr_rc_tailcall : PointerLikeRegClass<2>;

// Special i32mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i32mem_TC : Operand<i32> {
  let PrintMethod = "printi32mem";
  let MIOperandInfo = (ops ptr_rc_tailcall, i8imm, ptr_rc_tailcall,
                       i32imm, i8imm);
  let ParserMatchClass = X86Mem32AsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

// Special i64mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i64mem_TC : Operand<i64> {
  let PrintMethod = "printi64mem";
  let MIOperandInfo = (ops ptr_rc_tailcall, i8imm,
                       ptr_rc_tailcall, i32imm, i8imm);
  let ParserMatchClass = X86Mem64AsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

let OperandType = "OPERAND_PCREL",
    ParserMatchClass = X86AbsMemAsmOperand,
    PrintMethod = "printPCRelImm" in {
def i32imm_pcrel : Operand<i32>;
def i16imm_pcrel : Operand<i16>;

def offset8 : Operand<i64>;
def offset16 : Operand<i64>;
def offset32 : Operand<i64>;
def offset64 : Operand<i64>;

// Branch targets have OtherVT type and print as pc-relative values.
def brtarget : Operand<OtherVT>;
def brtarget8 : Operand<OtherVT>;

}

def SSECC : Operand<i8> {
  let PrintMethod = "printSSECC";
  let OperandType = "OPERAND_IMMEDIATE";
}

def AVXCC : Operand<i8> {
  let PrintMethod = "printAVXCC";
  let OperandType = "OPERAND_IMMEDIATE";
}

class ImmSExtAsmOperandClass : AsmOperandClass {
  let SuperClasses = [ImmAsmOperand];
  let RenderMethod = "addImmOperands";
}

class ImmZExtAsmOperandClass : AsmOperandClass {
  let SuperClasses = [ImmAsmOperand];
  let RenderMethod = "addImmOperands";
}

// Sign-extended immediate classes. We don't need to define the full lattice
// here because there is no instruction with an ambiguity between ImmSExti64i32
// and ImmSExti32i8.
//
// The strange ranges come from the fact that the assembler always works with
// 64-bit immediates, but for a 16-bit target value we want to accept both "-1"
// (which will be a -1ULL), and "0xFF" (-1 in 16-bits).

// [0, 0x7FFFFFFF]                                            |
//   [0xFFFFFFFF80000000, 0xFFFFFFFFFFFFFFFF]
def ImmSExti64i32AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti64i32";
}

// [0, 0x0000007F] | [0x000000000000FF80, 0x000000000000FFFF] |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti16i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti16i8";
  let SuperClasses = [ImmSExti64i32AsmOperand];
}

// [0, 0x0000007F] | [0x00000000FFFFFF80, 0x00000000FFFFFFFF] |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti32i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti32i8";
}

// [0, 0x000000FF]
def ImmZExtu32u8AsmOperand : ImmZExtAsmOperandClass {
  let Name = "ImmZExtu32u8";
}


// [0, 0x0000007F]                                            |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti64i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti64i8";
  let SuperClasses = [ImmSExti16i8AsmOperand, ImmSExti32i8AsmOperand,
                      ImmSExti64i32AsmOperand];
}

// A couple of more descriptive operand definitions.
// 16-bits but only 8 bits are significant.
def i16i8imm  : Operand<i16> {
  let ParserMatchClass = ImmSExti16i8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32> {
  let ParserMatchClass = ImmSExti32i8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}
// 32-bits but only 8 bits are significant, and those 8 bits are unsigned.
def u32u8imm  : Operand<i32> {
  let ParserMatchClass = ImmZExtu32u8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 64-bits but only 32 bits are significant.
def i64i32imm  : Operand<i64> {
  let ParserMatchClass = ImmSExti64i32AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 64-bits but only 32 bits are significant, and those bits are treated as being
// pc relative.
def i64i32imm_pcrel : Operand<i64> {
  let PrintMethod = "printPCRelImm";
  let ParserMatchClass = X86AbsMemAsmOperand;
  let OperandType = "OPERAND_PCREL";
}

// 64-bits but only 8 bits are significant.
def i64i8imm   : Operand<i64> {
  let ParserMatchClass = ImmSExti64i8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

def lea64_32mem : Operand<i32> {
  let PrintMethod = "printi32mem";
  let AsmOperandLowerMethod = "lower_lea64_32mem";
  let MIOperandInfo = (ops GR32, i8imm, GR32_NOSP, i32imm, i8imm);
  let ParserMatchClass = X86MemAsmOperand;
}


//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86 specific addressing mode.
def addr      : ComplexPattern<iPTR, 5, "SelectAddr", [], [SDNPWantParent]>;
def lea32addr : ComplexPattern<i32, 5, "SelectLEAAddr",
                               [add, sub, mul, X86mul_imm, shl, or, frameindex],
                               []>;
def tls32addr : ComplexPattern<i32, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def tls32baseaddr : ComplexPattern<i32, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def lea64addr : ComplexPattern<i64, 5, "SelectLEAAddr",
                        [add, sub, mul, X86mul_imm, shl, or, frameindex,
                         X86WrapperRIP], []>;

def tls64addr : ComplexPattern<i64, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def tls64baseaddr : ComplexPattern<i64, 5, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def HasCMov      : Predicate<"Subtarget->hasCMov()">;
def NoCMov       : Predicate<"!Subtarget->hasCMov()">;

def HasMMX       : Predicate<"Subtarget->hasMMX()">;
def Has3DNow     : Predicate<"Subtarget->has3DNow()">;
def Has3DNowA    : Predicate<"Subtarget->has3DNowA()">;
def HasSSE1      : Predicate<"Subtarget->hasSSE1()">;
def UseSSE1      : Predicate<"Subtarget->hasSSE1() && !Subtarget->hasAVX()">;
def HasSSE2      : Predicate<"Subtarget->hasSSE2()">;
def UseSSE2      : Predicate<"Subtarget->hasSSE2() && !Subtarget->hasAVX()">;
def HasSSE3      : Predicate<"Subtarget->hasSSE3()">;
def UseSSE3      : Predicate<"Subtarget->hasSSE3() && !Subtarget->hasAVX()">;
def HasSSSE3     : Predicate<"Subtarget->hasSSSE3()">;
def UseSSSE3     : Predicate<"Subtarget->hasSSSE3() && !Subtarget->hasAVX()">;
def HasSSE41     : Predicate<"Subtarget->hasSSE41()">;
def UseSSE41     : Predicate<"Subtarget->hasSSE41() && !Subtarget->hasAVX()">;
def HasSSE42     : Predicate<"Subtarget->hasSSE42()">;
def UseSSE42     : Predicate<"Subtarget->hasSSE42() && !Subtarget->hasAVX()">;
def HasSSE4A     : Predicate<"Subtarget->hasSSE4A()">;
def HasAVX       : Predicate<"Subtarget->hasAVX()">;
def HasAVX2      : Predicate<"Subtarget->hasAVX2()">;
def HasAVX1Only  : Predicate<"Subtarget->hasAVX() && !Subtarget->hasAVX2()">;

def HasPOPCNT    : Predicate<"Subtarget->hasPOPCNT()">;
def HasAES       : Predicate<"Subtarget->hasAES()">;
def HasPCLMUL    : Predicate<"Subtarget->hasPCLMUL()">;
def HasFMA       : Predicate<"Subtarget->hasFMA()">;
def HasFMA4      : Predicate<"Subtarget->hasFMA4()">;
def HasXOP       : Predicate<"Subtarget->hasXOP()">;
def HasMOVBE     : Predicate<"Subtarget->hasMOVBE()">;
def HasRDRAND    : Predicate<"Subtarget->hasRDRAND()">;
def HasF16C      : Predicate<"Subtarget->hasF16C()">;
def HasFSGSBase  : Predicate<"Subtarget->hasFSGSBase()">;
def HasLZCNT     : Predicate<"Subtarget->hasLZCNT()">;
def HasBMI       : Predicate<"Subtarget->hasBMI()">;
def HasBMI2      : Predicate<"Subtarget->hasBMI2()">;
def HasRTM       : Predicate<"Subtarget->hasRTM()">;
def FPStackf32   : Predicate<"!Subtarget->hasSSE1()">;
def FPStackf64   : Predicate<"!Subtarget->hasSSE2()">;
def HasCmpxchg16b: Predicate<"Subtarget->hasCmpxchg16b()">;
def In32BitMode  : Predicate<"!Subtarget->is64Bit()">,
                             AssemblerPredicate<"!Mode64Bit">;
def In64BitMode  : Predicate<"Subtarget->is64Bit()">,
                             AssemblerPredicate<"Mode64Bit">;
def IsWin64      : Predicate<"Subtarget->isTargetWin64()">;
def IsNaCl       : Predicate<"Subtarget->isTargetNaCl()">;
def NotNaCl      : Predicate<"!Subtarget->isTargetNaCl()">;
def SmallCode    : Predicate<"TM.getCodeModel() == CodeModel::Small">;
def KernelCode   : Predicate<"TM.getCodeModel() == CodeModel::Kernel">;
def FarData      : Predicate<"TM.getCodeModel() != CodeModel::Small &&"
                             "TM.getCodeModel() != CodeModel::Kernel">;
def NearData     : Predicate<"TM.getCodeModel() == CodeModel::Small ||"
                             "TM.getCodeModel() == CodeModel::Kernel">;
def IsStatic     : Predicate<"TM.getRelocationModel() == Reloc::Static">;
def IsNotPIC     : Predicate<"TM.getRelocationModel() != Reloc::PIC_">;
def OptForSize   : Predicate<"OptForSize">;
def OptForSpeed  : Predicate<"!OptForSize">;
def FastBTMem    : Predicate<"!Subtarget->isBTMemSlow()">;
def CallImmAddr  : Predicate<"Subtarget->IsLegalToCallImmediateAddr(TM)">;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

//===----------------------------------------------------------------------===//
// Pattern fragments.
//

// X86 specific condition code. These correspond to CondCode in
// X86InstrInfo.h. They must be kept in synch.
def X86_COND_A   : PatLeaf<(i8 0)>;  // alt. COND_NBE
def X86_COND_AE  : PatLeaf<(i8 1)>;  // alt. COND_NC
def X86_COND_B   : PatLeaf<(i8 2)>;  // alt. COND_C
def X86_COND_BE  : PatLeaf<(i8 3)>;  // alt. COND_NA
def X86_COND_E   : PatLeaf<(i8 4)>;  // alt. COND_Z
def X86_COND_G   : PatLeaf<(i8 5)>;  // alt. COND_NLE
def X86_COND_GE  : PatLeaf<(i8 6)>;  // alt. COND_NL
def X86_COND_L   : PatLeaf<(i8 7)>;  // alt. COND_NGE
def X86_COND_LE  : PatLeaf<(i8 8)>;  // alt. COND_NG
def X86_COND_NE  : PatLeaf<(i8 9)>;  // alt. COND_NZ
def X86_COND_NO  : PatLeaf<(i8 10)>;
def X86_COND_NP  : PatLeaf<(i8 11)>; // alt. COND_PO
def X86_COND_NS  : PatLeaf<(i8 12)>;
def X86_COND_O   : PatLeaf<(i8 13)>;
def X86_COND_P   : PatLeaf<(i8 14)>; // alt. COND_PE
def X86_COND_S   : PatLeaf<(i8 15)>;

let FastIselShouldIgnore = 1 in { // FastIsel should ignore all simm8 instrs.
  def i16immSExt8  : ImmLeaf<i16, [{ return Imm == (int8_t)Imm; }]>;
  def i32immSExt8  : ImmLeaf<i32, [{ return Imm == (int8_t)Imm; }]>;
  def i64immSExt8  : ImmLeaf<i64, [{ return Imm == (int8_t)Imm; }]>;
}

def i64immSExt32 : ImmLeaf<i64, [{ return Imm == (int32_t)Imm; }]>;


// i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
// unsigned field.
def i64immZExt32 : ImmLeaf<i64, [{ return (uint64_t)Imm == (uint32_t)Imm; }]>;

def i64immZExt32SExt8 : ImmLeaf<i64, [{
  return (uint64_t)Imm == (uint32_t)Imm && (int32_t)Imm == (int8_t)Imm;
}]>;

// Helper fragments for loads.
// It's always safe to treat a anyext i16 load as a i32 load if the i16 is
// known to be 32-bit aligned or better. Ditto for i8 to i16.
def loadi16 : PatFrag<(ops node:$ptr), (i16 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 2 && !LD->isVolatile();
  return false;
}]>;

def loadi16_anyext : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)),[{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 2 && !LD->isVolatile();
  return false;
}]>;

def loadi32 : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 4 && !LD->isVolatile();
  return false;
}]>;

def loadi8  : PatFrag<(ops node:$ptr), (i8  (load node:$ptr))>;
def loadi64 : PatFrag<(ops node:$ptr), (i64 (load node:$ptr))>;
def loadf32 : PatFrag<(ops node:$ptr), (f32 (load node:$ptr))>;
def loadf64 : PatFrag<(ops node:$ptr), (f64 (load node:$ptr))>;
def loadf80 : PatFrag<(ops node:$ptr), (f80 (load node:$ptr))>;

def sextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (sextloadi8 node:$ptr))>;
def sextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (sextloadi8 node:$ptr))>;
def sextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (sextloadi16 node:$ptr))>;
def sextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (sextloadi8 node:$ptr))>;
def sextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi8i1   : PatFrag<(ops node:$ptr), (i8  (zextloadi1 node:$ptr))>;
def zextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (zextloadi1 node:$ptr))>;
def zextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (zextloadi1 node:$ptr))>;
def zextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (zextloadi8 node:$ptr))>;
def zextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (zextloadi8 node:$ptr))>;
def zextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;
def zextloadi64i1  : PatFrag<(ops node:$ptr), (i64 (zextloadi1 node:$ptr))>;
def zextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (zextloadi8 node:$ptr))>;
def zextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

def extloadi8i1    : PatFrag<(ops node:$ptr), (i8  (extloadi1 node:$ptr))>;
def extloadi16i1   : PatFrag<(ops node:$ptr), (i16 (extloadi1 node:$ptr))>;
def extloadi32i1   : PatFrag<(ops node:$ptr), (i32 (extloadi1 node:$ptr))>;
def extloadi16i8   : PatFrag<(ops node:$ptr), (i16 (extloadi8 node:$ptr))>;
def extloadi32i8   : PatFrag<(ops node:$ptr), (i32 (extloadi8 node:$ptr))>;
def extloadi32i16  : PatFrag<(ops node:$ptr), (i32 (extloadi16 node:$ptr))>;
def extloadi64i1   : PatFrag<(ops node:$ptr), (i64 (extloadi1 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8 node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;


// An 'and' node with a single use.
def and_su : PatFrag<(ops node:$lhs, node:$rhs), (and node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;
// An 'srl' node with a single use.
def srl_su : PatFrag<(ops node:$lhs, node:$rhs), (srl node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;
// An 'trunc' node with a single use.
def trunc_su : PatFrag<(ops node:$src), (trunc node:$src), [{
  return N->hasOneUse();
}]>;

//===----------------------------------------------------------------------===//
// Instruction list.
//

// Nop
let neverHasSideEffects = 1 in {
  def NOOP : I<0x90, RawFrm, (outs), (ins), "nop", [], IIC_NOP>;
  def NOOPW : I<0x1f, MRM0m, (outs), (ins i16mem:$zero),
                "nop{w}\t$zero", [], IIC_NOP>, TB, OpSize;
  def NOOPL : I<0x1f, MRM0m, (outs), (ins i32mem:$zero),
                "nop{l}\t$zero", [], IIC_NOP>, TB;
}


// Constructing a stack frame.
def ENTER : Ii16<0xC8, RawFrmImm8, (outs), (ins i16imm:$len, i8imm:$lvl),
                 "enter\t$len, $lvl", [], IIC_ENTER>;

let Defs = [EBP, ESP], Uses = [EBP, ESP], mayLoad = 1, neverHasSideEffects=1 in
def LEAVE    : I<0xC9, RawFrm,
                 (outs), (ins), "leave", [], IIC_LEAVE>,
                 Requires<[In32BitMode]>;

let Defs = [RBP,RSP], Uses = [RBP,RSP], mayLoad = 1, neverHasSideEffects = 1 in
def LEAVE64  : I<0xC9, RawFrm,
                 (outs), (ins), "leave", [], IIC_LEAVE>,
                 Requires<[In64BitMode]>;

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions.
//

let Defs = [ESP], Uses = [ESP], neverHasSideEffects=1 in {
let mayLoad = 1 in {
def POP16r  : I<0x58, AddRegFrm, (outs GR16:$reg), (ins), "pop{w}\t$reg", [],
                IIC_POP_REG16>, OpSize;
def POP32r  : I<0x58, AddRegFrm, (outs GR32:$reg), (ins), "pop{l}\t$reg", [],
                IIC_POP_REG>;
def POP16rmr: I<0x8F, MRM0r, (outs GR16:$reg), (ins), "pop{w}\t$reg", [],
                IIC_POP_REG>, OpSize;
def POP16rmm: I<0x8F, MRM0m, (outs i16mem:$dst), (ins), "pop{w}\t$dst", [],
                IIC_POP_MEM>, OpSize;
def POP32rmr: I<0x8F, MRM0r, (outs GR32:$reg), (ins), "pop{l}\t$reg", [],
                IIC_POP_REG>;
def POP32rmm: I<0x8F, MRM0m, (outs i32mem:$dst), (ins), "pop{l}\t$dst", [],
                IIC_POP_MEM>;

def POPF16   : I<0x9D, RawFrm, (outs), (ins), "popf{w}", [], IIC_POP_F>, OpSize;
def POPF32   : I<0x9D, RawFrm, (outs), (ins), "popf{l|d}", [], IIC_POP_FD>,
               Requires<[In32BitMode]>;
}

let mayStore = 1 in {
def PUSH16r  : I<0x50, AddRegFrm, (outs), (ins GR16:$reg), "push{w}\t$reg",[],
                 IIC_PUSH_REG>, OpSize;
def PUSH32r  : I<0x50, AddRegFrm, (outs), (ins GR32:$reg), "push{l}\t$reg",[],
                 IIC_PUSH_REG>;
def PUSH16rmr: I<0xFF, MRM6r, (outs), (ins GR16:$reg), "push{w}\t$reg",[],
                 IIC_PUSH_REG>, OpSize;
def PUSH16rmm: I<0xFF, MRM6m, (outs), (ins i16mem:$src), "push{w}\t$src",[],
                 IIC_PUSH_MEM>,
  OpSize;
def PUSH32rmr: I<0xFF, MRM6r, (outs), (ins GR32:$reg), "push{l}\t$reg",[],
                 IIC_PUSH_REG>;
def PUSH32rmm: I<0xFF, MRM6m, (outs), (ins i32mem:$src), "push{l}\t$src",[],
                 IIC_PUSH_MEM>;

def PUSHi8   : Ii8<0x6a, RawFrm, (outs), (ins i32i8imm:$imm),
                      "push{l}\t$imm", [], IIC_PUSH_IMM>;
def PUSHi16  : Ii16<0x68, RawFrm, (outs), (ins i16imm:$imm),
                      "push{w}\t$imm", [], IIC_PUSH_IMM>, OpSize;
def PUSHi32  : Ii32<0x68, RawFrm, (outs), (ins i32imm:$imm),
                      "push{l}\t$imm", [], IIC_PUSH_IMM>;

def PUSHF16  : I<0x9C, RawFrm, (outs), (ins), "pushf{w}", [], IIC_PUSH_F>,
                 OpSize;
def PUSHF32  : I<0x9C, RawFrm, (outs), (ins), "pushf{l|d}", [], IIC_PUSH_F>,
               Requires<[In32BitMode]>;

}
}

let Defs = [RSP], Uses = [RSP], neverHasSideEffects=1 in {
let mayLoad = 1 in {
def POP64r   : I<0x58, AddRegFrm,
                 (outs GR64:$reg), (ins), "pop{q}\t$reg", [], IIC_POP_REG>;
def POP64rmr: I<0x8F, MRM0r, (outs GR64:$reg), (ins), "pop{q}\t$reg", [],
                IIC_POP_REG>;
def POP64rmm: I<0x8F, MRM0m, (outs i64mem:$dst), (ins), "pop{q}\t$dst", [],
                IIC_POP_MEM>;
}
let mayStore = 1 in {
def PUSH64r  : I<0x50, AddRegFrm,
                 (outs), (ins GR64:$reg), "push{q}\t$reg", [], IIC_PUSH_REG>;
def PUSH64rmr: I<0xFF, MRM6r, (outs), (ins GR64:$reg), "push{q}\t$reg", [],
                 IIC_PUSH_REG>;
def PUSH64rmm: I<0xFF, MRM6m, (outs), (ins i64mem:$src), "push{q}\t$src", [],
                 IIC_PUSH_MEM>;
}
}

let Defs = [RSP], Uses = [RSP], neverHasSideEffects = 1, mayStore = 1 in {
def PUSH64i8   : Ii8<0x6a, RawFrm, (outs), (ins i64i8imm:$imm),
                     "push{q}\t$imm", [], IIC_PUSH_IMM>;
def PUSH64i16  : Ii16<0x68, RawFrm, (outs), (ins i16imm:$imm),
                      "push{q}\t$imm", [], IIC_PUSH_IMM>;
def PUSH64i32  : Ii32<0x68, RawFrm, (outs), (ins i64i32imm:$imm),
                      "push{q}\t$imm", [], IIC_PUSH_IMM>;
}

let Defs = [RSP, EFLAGS], Uses = [RSP], mayLoad = 1, neverHasSideEffects=1 in
def POPF64   : I<0x9D, RawFrm, (outs), (ins), "popfq", [], IIC_POP_FD>,
               Requires<[In64BitMode]>;
let Defs = [RSP], Uses = [RSP, EFLAGS], mayStore = 1, neverHasSideEffects=1 in
def PUSHF64    : I<0x9C, RawFrm, (outs), (ins), "pushfq", [], IIC_PUSH_F>,
                 Requires<[In64BitMode]>;



let Defs = [EDI, ESI, EBP, EBX, EDX, ECX, EAX, ESP], Uses = [ESP],
    mayLoad=1, neverHasSideEffects=1 in {
def POPA32   : I<0x61, RawFrm, (outs), (ins), "popa{l}", [], IIC_POP_A>,
               Requires<[In32BitMode]>;
}
let Defs = [ESP], Uses = [EDI, ESI, EBP, EBX, EDX, ECX, EAX, ESP],
    mayStore=1, neverHasSideEffects=1 in {
def PUSHA32  : I<0x60, RawFrm, (outs), (ins), "pusha{l}", [], IIC_PUSH_A>,
               Requires<[In32BitMode]>;
}

let Constraints = "$src = $dst" in {    // GR32 = bswap GR32
def BSWAP32r : I<0xC8, AddRegFrm,
                 (outs GR32:$dst), (ins GR32:$src),
                 "bswap{l}\t$dst",
                 [(set GR32:$dst, (bswap GR32:$src))], IIC_BSWAP>, TB;

def BSWAP64r : RI<0xC8, AddRegFrm, (outs GR64:$dst), (ins GR64:$src),
                  "bswap{q}\t$dst",
                  [(set GR64:$dst, (bswap GR64:$src))], IIC_BSWAP>, TB;
} // Constraints = "$src = $dst"

// Bit scan instructions.
let Defs = [EFLAGS] in {
def BSF16rr  : I<0xBC, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsf GR16:$src))],
                  IIC_BSF>, TB, OpSize;
def BSF16rm  : I<0xBC, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsf (loadi16 addr:$src)))],
                  IIC_BSF>, TB, OpSize;
def BSF32rr  : I<0xBC, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsf GR32:$src))], IIC_BSF>, TB;
def BSF32rm  : I<0xBC, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsf (loadi32 addr:$src)))],
                 IIC_BSF>, TB;
def BSF64rr  : RI<0xBC, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsf GR64:$src))],
                  IIC_BSF>, TB;
def BSF64rm  : RI<0xBC, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsf (loadi64 addr:$src)))],
                  IIC_BSF>, TB;

def BSR16rr  : I<0xBD, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsr GR16:$src))], IIC_BSR>,
                 TB, OpSize;
def BSR16rm  : I<0xBD, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, EFLAGS, (X86bsr (loadi16 addr:$src)))],
                 IIC_BSR>, TB,
                 OpSize;
def BSR32rr  : I<0xBD, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsr GR32:$src))], IIC_BSR>, TB;
def BSR32rm  : I<0xBD, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, EFLAGS, (X86bsr (loadi32 addr:$src)))],
                 IIC_BSR>, TB;
def BSR64rr  : RI<0xBD, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsr GR64:$src))], IIC_BSR>, TB;
def BSR64rm  : RI<0xBD, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, EFLAGS, (X86bsr (loadi64 addr:$src)))],
                  IIC_BSR>, TB;
} // Defs = [EFLAGS]


// These uses the DF flag in the EFLAGS register to inc or dec EDI and ESI
let Defs = [EDI,ESI], Uses = [EDI,ESI,EFLAGS] in {
def MOVSB : I<0xA4, RawFrm, (outs), (ins), "movsb", [], IIC_MOVS>;
def MOVSW : I<0xA5, RawFrm, (outs), (ins), "movsw", [], IIC_MOVS>, OpSize;
def MOVSD : I<0xA5, RawFrm, (outs), (ins), "movs{l|d}", [], IIC_MOVS>;
def MOVSQ : RI<0xA5, RawFrm, (outs), (ins), "movsq", [], IIC_MOVS>;
}

// These uses the DF flag in the EFLAGS register to inc or dec EDI and ESI
let Defs = [EDI], Uses = [AL,EDI,EFLAGS] in
def STOSB : I<0xAA, RawFrm, (outs), (ins), "stosb", [], IIC_STOS>;
let Defs = [EDI], Uses = [AX,EDI,EFLAGS] in
def STOSW : I<0xAB, RawFrm, (outs), (ins), "stosw", [], IIC_STOS>, OpSize;
let Defs = [EDI], Uses = [EAX,EDI,EFLAGS] in
def STOSD : I<0xAB, RawFrm, (outs), (ins), "stos{l|d}", [], IIC_STOS>;
let Defs = [RCX,RDI], Uses = [RAX,RCX,RDI,EFLAGS] in
def STOSQ : RI<0xAB, RawFrm, (outs), (ins), "stosq", [], IIC_STOS>;

def SCAS8 : I<0xAE, RawFrm, (outs), (ins), "scasb", [], IIC_SCAS>;
def SCAS16 : I<0xAF, RawFrm, (outs), (ins), "scasw", [], IIC_SCAS>, OpSize;
def SCAS32 : I<0xAF, RawFrm, (outs), (ins), "scas{l|d}", [], IIC_SCAS>;
def SCAS64 : RI<0xAF, RawFrm, (outs), (ins), "scasq", [], IIC_SCAS>;

def CMPS8 : I<0xA6, RawFrm, (outs), (ins), "cmpsb", [], IIC_CMPS>;
def CMPS16 : I<0xA7, RawFrm, (outs), (ins), "cmpsw", [], IIC_CMPS>, OpSize;
def CMPS32 : I<0xA7, RawFrm, (outs), (ins), "cmps{l|d}", [], IIC_CMPS>;
def CMPS64 : RI<0xA7, RawFrm, (outs), (ins), "cmpsq", [], IIC_CMPS>;


//===----------------------------------------------------------------------===//
//  Move Instructions.
//

let neverHasSideEffects = 1 in {
def MOV8rr  : I<0x88, MRMDestReg, (outs GR8 :$dst), (ins GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
def MOV16rr : I<0x89, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", [], IIC_MOV>, OpSize;
def MOV32rr : I<0x89, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
def MOV64rr : RI<0x89, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
}
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV8ri  : Ii8 <0xB0, AddRegFrm, (outs GR8 :$dst), (ins i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(set GR8:$dst, imm:$src)], IIC_MOV>;
def MOV16ri : Ii16<0xB8, AddRegFrm, (outs GR16:$dst), (ins i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(set GR16:$dst, imm:$src)], IIC_MOV>, OpSize;
def MOV32ri : Ii32<0xB8, AddRegFrm, (outs GR32:$dst), (ins i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, imm:$src)], IIC_MOV>;
def MOV64ri : RIi64<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64imm:$src),
                    "movabs{q}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, imm:$src)], IIC_MOV>;
def MOV64ri32 : RIi32<0xC7, MRM0r, (outs GR64:$dst), (ins i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, i64immSExt32:$src)], IIC_MOV>;
}

def MOV8mi  : Ii8 <0xC6, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(store (i8 imm:$src), addr:$dst)], IIC_MOV_MEM>;
def MOV16mi : Ii16<0xC7, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(store (i16 imm:$src), addr:$dst)], IIC_MOV_MEM>, OpSize;
def MOV32mi : Ii32<0xC7, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(store (i32 imm:$src), addr:$dst)], IIC_MOV_MEM>;
def MOV64mi32 : RIi32<0xC7, MRM0m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(store i64immSExt32:$src, addr:$dst)], IIC_MOV_MEM>;

/// moffs8, moffs16 and moffs32 versions of moves.  The immediate is a
/// 32-bit offset from the PC.  These are only valid in x86-32 mode.
def MOV8o8a : Ii32 <0xA0, RawFrm, (outs), (ins offset8:$src),
                   "mov{b}\t{$src, %al|AL, $src}", [], IIC_MOV_MEM>,
                   Requires<[In32BitMode]>;
def MOV16o16a : Ii32 <0xA1, RawFrm, (outs), (ins offset16:$src),
                      "mov{w}\t{$src, %ax|AL, $src}", [], IIC_MOV_MEM>, OpSize,
                     Requires<[In32BitMode]>;
def MOV32o32a : Ii32 <0xA1, RawFrm, (outs), (ins offset32:$src),
                      "mov{l}\t{$src, %eax|EAX, $src}", [], IIC_MOV_MEM>,
                     Requires<[In32BitMode]>;
def MOV8ao8 : Ii32 <0xA2, RawFrm, (outs offset8:$dst), (ins),
                   "mov{b}\t{%al, $dst|$dst, AL}", [], IIC_MOV_MEM>,
                  Requires<[In32BitMode]>;
def MOV16ao16 : Ii32 <0xA3, RawFrm, (outs offset16:$dst), (ins),
                      "mov{w}\t{%ax, $dst|$dst, AL}", [], IIC_MOV_MEM>, OpSize,
                     Requires<[In32BitMode]>;
def MOV32ao32 : Ii32 <0xA3, RawFrm, (outs offset32:$dst), (ins),
                      "mov{l}\t{%eax, $dst|$dst, EAX}", [], IIC_MOV_MEM>,
                     Requires<[In32BitMode]>;

// FIXME: These definitions are utterly broken
// Just leave them commented out for now because they're useless outside
// of the large code model, and most compilers won't generate the instructions
// in question.
/*
def MOV64o8a : RIi8<0xA0, RawFrm, (outs), (ins offset8:$src),
                      "mov{q}\t{$src, %rax|RAX, $src}", []>;
def MOV64o64a : RIi32<0xA1, RawFrm, (outs), (ins offset64:$src),
                       "mov{q}\t{$src, %rax|RAX, $src}", []>;
def MOV64ao8 : RIi8<0xA2, RawFrm, (outs offset8:$dst), (ins),
                       "mov{q}\t{%rax, $dst|$dst, RAX}", []>;
def MOV64ao64 : RIi32<0xA3, RawFrm, (outs offset64:$dst), (ins),
                       "mov{q}\t{%rax, $dst|$dst, RAX}", []>;
*/


let isCodeGenOnly = 1 in {
def MOV8rr_REV : I<0x8A, MRMSrcReg, (outs GR8:$dst), (ins GR8:$src),
                   "mov{b}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
def MOV16rr_REV : I<0x8B, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                    "mov{w}\t{$src, $dst|$dst, $src}", [], IIC_MOV>, OpSize;
def MOV32rr_REV : I<0x8B, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                    "mov{l}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
def MOV64rr_REV : RI<0x8B, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                     "mov{q}\t{$src, $dst|$dst, $src}", [], IIC_MOV>;
}

let canFoldAsLoad = 1, isReMaterializable = 1 in {
def MOV8rm  : I<0x8A, MRMSrcMem, (outs GR8 :$dst), (ins i8mem :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(set GR8:$dst, (loadi8 addr:$src))], IIC_MOV_MEM>;
def MOV16rm : I<0x8B, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(set GR16:$dst, (loadi16 addr:$src))], IIC_MOV_MEM>, OpSize;
def MOV32rm : I<0x8B, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(set GR32:$dst, (loadi32 addr:$src))], IIC_MOV_MEM>;
def MOV64rm : RI<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(set GR64:$dst, (load addr:$src))], IIC_MOV_MEM>;
}

def MOV8mr  : I<0x88, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(store GR8:$src, addr:$dst)], IIC_MOV_MEM>;
def MOV16mr : I<0x89, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(store GR16:$src, addr:$dst)], IIC_MOV_MEM>, OpSize;
def MOV32mr : I<0x89, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(store GR32:$src, addr:$dst)], IIC_MOV_MEM>;
def MOV64mr : RI<0x89, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(store GR64:$src, addr:$dst)], IIC_MOV_MEM>;

// Versions of MOV8rr, MOV8mr, and MOV8rm that use i8mem_NOREX and GR8_NOREX so
// that they can be used for copying and storing h registers, which can't be
// encoded when a REX prefix is present.
let isCodeGenOnly = 1 in {
let neverHasSideEffects = 1 in
def MOV8rr_NOREX : I<0x88, MRMDestReg,
                     (outs GR8_NOREX:$dst), (ins GR8_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", [], IIC_MOV>;
let mayStore = 1 in
def MOV8mr_NOREX : I<0x88, MRMDestMem,
                     (outs), (ins i8mem_NOREX:$dst, GR8_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", [],
                     IIC_MOV_MEM>;
let mayLoad = 1, neverHasSideEffects = 1,
    canFoldAsLoad = 1, isReMaterializable = 1 in
def MOV8rm_NOREX : I<0x8A, MRMSrcMem,
                     (outs GR8_NOREX:$dst), (ins i8mem_NOREX:$src),
                     "mov{b}\t{$src, $dst|$dst, $src}  # NOREX", [],
                     IIC_MOV_MEM>;
}


// Condition code ops, incl. set if equal/not equal/...
let Defs = [EFLAGS], Uses = [AH] in
def SAHF     : I<0x9E, RawFrm, (outs),  (ins), "sahf",
                 [(set EFLAGS, (X86sahf AH))], IIC_AHF>;
let Defs = [AH], Uses = [EFLAGS], neverHasSideEffects = 1 in
def LAHF     : I<0x9F, RawFrm, (outs),  (ins), "lahf", [],
                IIC_AHF>;  // AH = flags


//===----------------------------------------------------------------------===//
// Bit tests instructions: BT, BTS, BTR, BTC.

let Defs = [EFLAGS] in {
def BT16rr : I<0xA3, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
               "bt{w}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR16:$src1, GR16:$src2))], IIC_BT_RR>,
               OpSize, TB;
def BT32rr : I<0xA3, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
               "bt{l}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR32:$src1, GR32:$src2))], IIC_BT_RR>, TB;
def BT64rr : RI<0xA3, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
               "bt{q}\t{$src2, $src1|$src1, $src2}",
               [(set EFLAGS, (X86bt GR64:$src1, GR64:$src2))], IIC_BT_RR>, TB;

// Unlike with the register+register form, the memory+register form of the
// bt instruction does not ignore the high bits of the index. From ISel's
// perspective, this is pretty bizarre. Make these instructions disassembly
// only for now.

def BT16mr : I<0xA3, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
               "bt{w}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi16 addr:$src1), GR16:$src2),
//                (implicit EFLAGS)]
               [], IIC_BT_MR
               >, OpSize, TB, Requires<[FastBTMem]>;
def BT32mr : I<0xA3, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
               "bt{l}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi32 addr:$src1), GR32:$src2),
//                (implicit EFLAGS)]
               [], IIC_BT_MR
               >, TB, Requires<[FastBTMem]>;
def BT64mr : RI<0xA3, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
               "bt{q}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi64 addr:$src1), GR64:$src2),
//                (implicit EFLAGS)]
                [], IIC_BT_MR
                >, TB;

def BT16ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR16:$src1, i16immSExt8:$src2))],
                IIC_BT_RI>, OpSize, TB;
def BT32ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR32:$src1, i32immSExt8:$src2))],
                IIC_BT_RI>, TB;
def BT64ri8 : RIi8<0xBA, MRM4r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt GR64:$src1, i64immSExt8:$src2))],
                IIC_BT_RI>, TB;

// Note that these instructions don't need FastBTMem because that
// only applies when the other operand is in a register. When it's
// an immediate, bt is still fast.
def BT16mi8 : Ii8<0xBA, MRM4m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi16 addr:$src1), i16immSExt8:$src2))
                 ], IIC_BT_MI>, OpSize, TB;
def BT32mi8 : Ii8<0xBA, MRM4m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi32 addr:$src1), i32immSExt8:$src2))
                 ], IIC_BT_MI>, TB;
def BT64mi8 : RIi8<0xBA, MRM4m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(set EFLAGS, (X86bt (loadi64 addr:$src1),
                                     i64immSExt8:$src2))], IIC_BT_MI>, TB;


def BTC16rr : I<0xBB, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "btc{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>,
                OpSize, TB;
def BTC32rr : I<0xBB, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "btc{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>, TB;
def BTC64rr : RI<0xBB, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "btc{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>, TB;
def BTC16mr : I<0xBB, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "btc{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>,
                OpSize, TB;
def BTC32mr : I<0xBB, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "btc{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTC64mr : RI<0xBB, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "btc{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTC16ri8 : Ii8<0xBA, MRM7r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "btc{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>,
                    OpSize, TB;
def BTC32ri8 : Ii8<0xBA, MRM7r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "btc{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTC64ri8 : RIi8<0xBA, MRM7r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "btc{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTC16mi8 : Ii8<0xBA, MRM7m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "btc{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>,
                    OpSize, TB;
def BTC32mi8 : Ii8<0xBA, MRM7m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "btc{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;
def BTC64mi8 : RIi8<0xBA, MRM7m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "btc{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;

def BTR16rr : I<0xB3, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "btr{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>,
                OpSize, TB;
def BTR32rr : I<0xB3, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "btr{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>, TB;
def BTR64rr : RI<0xB3, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "btr{q}\t{$src2, $src1|$src1, $src2}", []>, TB;
def BTR16mr : I<0xB3, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "btr{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>,
                OpSize, TB;
def BTR32mr : I<0xB3, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "btr{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTR64mr : RI<0xB3, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "btr{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTR16ri8 : Ii8<0xBA, MRM6r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "btr{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>,
                    OpSize, TB;
def BTR32ri8 : Ii8<0xBA, MRM6r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "btr{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTR64ri8 : RIi8<0xBA, MRM6r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "btr{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTR16mi8 : Ii8<0xBA, MRM6m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "btr{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>,
                    OpSize, TB;
def BTR32mi8 : Ii8<0xBA, MRM6m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "btr{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;
def BTR64mi8 : RIi8<0xBA, MRM6m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "btr{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;

def BTS16rr : I<0xAB, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
                "bts{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>,
                OpSize, TB;
def BTS32rr : I<0xAB, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
                "bts{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>, TB;
def BTS64rr : RI<0xAB, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "bts{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RR>, TB;
def BTS16mr : I<0xAB, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
                "bts{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>,
                OpSize, TB;
def BTS32mr : I<0xAB, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
                "bts{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTS64mr : RI<0xAB, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "bts{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MR>, TB;
def BTS16ri8 : Ii8<0xBA, MRM5r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                    "bts{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>,
                    OpSize, TB;
def BTS32ri8 : Ii8<0xBA, MRM5r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                    "bts{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTS64ri8 : RIi8<0xBA, MRM5r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "bts{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_RI>, TB;
def BTS16mi8 : Ii8<0xBA, MRM5m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                    "bts{w}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>,
                    OpSize, TB;
def BTS32mi8 : Ii8<0xBA, MRM5m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                    "bts{l}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;
def BTS64mi8 : RIi8<0xBA, MRM5m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "bts{q}\t{$src2, $src1|$src1, $src2}", [], IIC_BTX_MI>, TB;
} // Defs = [EFLAGS]


//===----------------------------------------------------------------------===//
// Atomic support
//

// Atomic swap. These are just normal xchg instructions. But since a memory
// operand is referenced, the atomicity is ensured.
multiclass ATOMIC_SWAP<bits<8> opc8, bits<8> opc, string mnemonic, string frag,
                       InstrItinClass itin> {
  let Constraints = "$val = $dst" in {
    def #NAME#8rm  : I<opc8, MRMSrcMem, (outs GR8:$dst),
                       (ins GR8:$val, i8mem:$ptr),
                       !strconcat(mnemonic, "{b}\t{$val, $ptr|$ptr, $val}"),
                       [(set
                          GR8:$dst,
                          (!cast<PatFrag>(frag # "_8") addr:$ptr, GR8:$val))],
                       itin>;
    def #NAME#16rm : I<opc, MRMSrcMem, (outs GR16:$dst),
                       (ins GR16:$val, i16mem:$ptr),
                       !strconcat(mnemonic, "{w}\t{$val, $ptr|$ptr, $val}"),
                       [(set
                          GR16:$dst,
                          (!cast<PatFrag>(frag # "_16") addr:$ptr, GR16:$val))],
                       itin>, OpSize;
    def #NAME#32rm : I<opc, MRMSrcMem, (outs GR32:$dst),
                       (ins GR32:$val, i32mem:$ptr),
                       !strconcat(mnemonic, "{l}\t{$val, $ptr|$ptr, $val}"),
                       [(set
                          GR32:$dst,
                          (!cast<PatFrag>(frag # "_32") addr:$ptr, GR32:$val))],
                       itin>;
    def #NAME#64rm : RI<opc, MRMSrcMem, (outs GR64:$dst),
                        (ins GR64:$val, i64mem:$ptr),
                        !strconcat(mnemonic, "{q}\t{$val, $ptr|$ptr, $val}"),
                        [(set
                          GR64:$dst,
                          (!cast<PatFrag>(frag # "_64") addr:$ptr, GR64:$val))],
                        itin>;
  }
}

defm XCHG    : ATOMIC_SWAP<0x86, 0x87, "xchg", "atomic_swap", IIC_XCHG_MEM>;

// Swap between registers.
let Constraints = "$val = $dst" in {
def XCHG8rr : I<0x86, MRMSrcReg, (outs GR8:$dst), (ins GR8:$val, GR8:$src),
                "xchg{b}\t{$val, $src|$src, $val}", [], IIC_XCHG_REG>;
def XCHG16rr : I<0x87, MRMSrcReg, (outs GR16:$dst), (ins GR16:$val, GR16:$src),
                 "xchg{w}\t{$val, $src|$src, $val}", [], IIC_XCHG_REG>, OpSize;
def XCHG32rr : I<0x87, MRMSrcReg, (outs GR32:$dst), (ins GR32:$val, GR32:$src),
                 "xchg{l}\t{$val, $src|$src, $val}", [], IIC_XCHG_REG>;
def XCHG64rr : RI<0x87, MRMSrcReg, (outs GR64:$dst), (ins GR64:$val,GR64:$src),
                  "xchg{q}\t{$val, $src|$src, $val}", [], IIC_XCHG_REG>;
}

// Swap between EAX and other registers.
def XCHG16ar : I<0x90, AddRegFrm, (outs), (ins GR16:$src),
                  "xchg{w}\t{$src, %ax|AX, $src}", [], IIC_XCHG_REG>, OpSize;
def XCHG32ar : I<0x90, AddRegFrm, (outs), (ins GR32:$src),
                  "xchg{l}\t{$src, %eax|EAX, $src}", [], IIC_XCHG_REG>,
                  Requires<[In32BitMode]>;
// Uses GR32_NOAX in 64-bit mode to prevent encoding using the 0x90 NOP encoding.
// xchg %eax, %eax needs to clear upper 32-bits of RAX so is not a NOP.
def XCHG32ar64 : I<0x90, AddRegFrm, (outs), (ins GR32_NOAX:$src),
                   "xchg{l}\t{$src, %eax|EAX, $src}", [], IIC_XCHG_REG>,
                   Requires<[In64BitMode]>;
def XCHG64ar : RI<0x90, AddRegFrm, (outs), (ins GR64:$src),
                  "xchg{q}\t{$src, %rax|RAX, $src}", [], IIC_XCHG_REG>;



def XADD8rr : I<0xC0, MRMDestReg, (outs GR8:$dst), (ins GR8:$src),
                "xadd{b}\t{$src, $dst|$dst, $src}", [], IIC_XADD_REG>, TB;
def XADD16rr : I<0xC1, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                 "xadd{w}\t{$src, $dst|$dst, $src}", [], IIC_XADD_REG>, TB,
                 OpSize;
def XADD32rr  : I<0xC1, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                 "xadd{l}\t{$src, $dst|$dst, $src}", [], IIC_XADD_REG>, TB;
def XADD64rr  : RI<0xC1, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
                   "xadd{q}\t{$src, $dst|$dst, $src}", [], IIC_XADD_REG>, TB;

let mayLoad = 1, mayStore = 1 in {
def XADD8rm   : I<0xC0, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                 "xadd{b}\t{$src, $dst|$dst, $src}", [], IIC_XADD_MEM>, TB;
def XADD16rm  : I<0xC1, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                 "xadd{w}\t{$src, $dst|$dst, $src}", [], IIC_XADD_MEM>, TB,
                 OpSize;
def XADD32rm  : I<0xC1, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "xadd{l}\t{$src, $dst|$dst, $src}", [], IIC_XADD_MEM>, TB;
def XADD64rm  : RI<0xC1, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                   "xadd{q}\t{$src, $dst|$dst, $src}", [], IIC_XADD_MEM>, TB;

}

def CMPXCHG8rr : I<0xB0, MRMDestReg, (outs GR8:$dst), (ins GR8:$src),
                   "cmpxchg{b}\t{$src, $dst|$dst, $src}", [],
                   IIC_CMPXCHG_REG8>, TB;
def CMPXCHG16rr : I<0xB1, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                    "cmpxchg{w}\t{$src, $dst|$dst, $src}", [],
                    IIC_CMPXCHG_REG>, TB, OpSize;
def CMPXCHG32rr  : I<0xB1, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                     "cmpxchg{l}\t{$src, $dst|$dst, $src}", [],
                     IIC_CMPXCHG_REG>, TB;
def CMPXCHG64rr  : RI<0xB1, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
                      "cmpxchg{q}\t{$src, $dst|$dst, $src}", [],
                      IIC_CMPXCHG_REG>, TB;

let mayLoad = 1, mayStore = 1 in {
def CMPXCHG8rm   : I<0xB0, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                     "cmpxchg{b}\t{$src, $dst|$dst, $src}", [],
                     IIC_CMPXCHG_MEM8>, TB;
def CMPXCHG16rm  : I<0xB1, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                     "cmpxchg{w}\t{$src, $dst|$dst, $src}", [],
                     IIC_CMPXCHG_MEM>, TB, OpSize;
def CMPXCHG32rm  : I<0xB1, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                     "cmpxchg{l}\t{$src, $dst|$dst, $src}", [],
                     IIC_CMPXCHG_MEM>, TB;
def CMPXCHG64rm  : RI<0xB1, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                      "cmpxchg{q}\t{$src, $dst|$dst, $src}", [],
                      IIC_CMPXCHG_MEM>, TB;
}

let Defs = [EAX, EDX, EFLAGS], Uses = [EAX, EBX, ECX, EDX] in
def CMPXCHG8B : I<0xC7, MRM1m, (outs), (ins i64mem:$dst),
                  "cmpxchg8b\t$dst", [], IIC_CMPXCHG_8B>, TB;

let Defs = [RAX, RDX, EFLAGS], Uses = [RAX, RBX, RCX, RDX] in
def CMPXCHG16B : RI<0xC7, MRM1m, (outs), (ins i128mem:$dst),
                    "cmpxchg16b\t$dst", [], IIC_CMPXCHG_16B>,
                    TB, Requires<[HasCmpxchg16b]>;



// Lock instruction prefix
def LOCK_PREFIX : I<0xF0, RawFrm, (outs),  (ins), "lock", []>;

// Rex64 instruction prefix
def REX64_PREFIX : I<0x48, RawFrm, (outs),  (ins), "rex64", []>;

// Data16 instruction prefix
def DATA16_PREFIX : I<0x66, RawFrm, (outs),  (ins), "data16", []>;

// Repeat string operation instruction prefixes
// These uses the DF flag in the EFLAGS register to inc or dec ECX
let Defs = [ECX], Uses = [ECX,EFLAGS] in {
// Repeat (used with INS, OUTS, MOVS, LODS and STOS)
def REP_PREFIX : I<0xF3, RawFrm, (outs),  (ins), "rep", []>;
// Repeat while not equal (used with CMPS and SCAS)
def REPNE_PREFIX : I<0xF2, RawFrm, (outs),  (ins), "repne", []>;
}


// String manipulation instructions
def LODSB : I<0xAC, RawFrm, (outs), (ins), "lodsb", [], IIC_LODS>;
def LODSW : I<0xAD, RawFrm, (outs), (ins), "lodsw", [], IIC_LODS>, OpSize;
def LODSD : I<0xAD, RawFrm, (outs), (ins), "lods{l|d}", [], IIC_LODS>;
def LODSQ : RI<0xAD, RawFrm, (outs), (ins), "lodsq", [], IIC_LODS>;

def OUTSB : I<0x6E, RawFrm, (outs), (ins), "outsb", [], IIC_OUTS>;
def OUTSW : I<0x6F, RawFrm, (outs), (ins), "outsw", [], IIC_OUTS>, OpSize;
def OUTSD : I<0x6F, RawFrm, (outs), (ins), "outs{l|d}", [], IIC_OUTS>;


// Flag instructions
def CLC : I<0xF8, RawFrm, (outs), (ins), "clc", [], IIC_CLC>;
def STC : I<0xF9, RawFrm, (outs), (ins), "stc", [], IIC_STC>;
def CLI : I<0xFA, RawFrm, (outs), (ins), "cli", [], IIC_CLI>;
def STI : I<0xFB, RawFrm, (outs), (ins), "sti", [], IIC_STI>;
def CLD : I<0xFC, RawFrm, (outs), (ins), "cld", [], IIC_CLD>;
def STD : I<0xFD, RawFrm, (outs), (ins), "std", [], IIC_STD>;
def CMC : I<0xF5, RawFrm, (outs), (ins), "cmc", [], IIC_CMC>;

def CLTS : I<0x06, RawFrm, (outs), (ins), "clts", [], IIC_CLTS>, TB;

// Table lookup instructions
def XLAT : I<0xD7, RawFrm, (outs), (ins), "xlatb", [], IIC_XLAT>;

// ASCII Adjust After Addition
// sets AL, AH and CF and AF of EFLAGS and uses AL and AF of EFLAGS
def AAA : I<0x37, RawFrm, (outs), (ins), "aaa", [], IIC_AAA>,
            Requires<[In32BitMode]>;

// ASCII Adjust AX Before Division
// sets AL, AH and EFLAGS and uses AL and AH
def AAD8i8 : Ii8<0xD5, RawFrm, (outs), (ins i8imm:$src),
                 "aad\t$src", [], IIC_AAD>, Requires<[In32BitMode]>;

// ASCII Adjust AX After Multiply
// sets AL, AH and EFLAGS and uses AL
def AAM8i8 : Ii8<0xD4, RawFrm, (outs), (ins i8imm:$src),
                 "aam\t$src", [], IIC_AAM>, Requires<[In32BitMode]>;

// ASCII Adjust AL After Subtraction - sets
// sets AL, AH and CF and AF of EFLAGS and uses AL and AF of EFLAGS
def AAS : I<0x3F, RawFrm, (outs), (ins), "aas", [], IIC_AAS>,
            Requires<[In32BitMode]>;

// Decimal Adjust AL after Addition
// sets AL, CF and AF of EFLAGS and uses AL, CF and AF of EFLAGS
def DAA : I<0x27, RawFrm, (outs), (ins), "daa", [], IIC_DAA>,
            Requires<[In32BitMode]>;

// Decimal Adjust AL after Subtraction
// sets AL, CF and AF of EFLAGS and uses AL, CF and AF of EFLAGS
def DAS : I<0x2F, RawFrm, (outs), (ins), "das", [], IIC_DAS>,
            Requires<[In32BitMode]>;

// Check Array Index Against Bounds
def BOUNDS16rm : I<0x62, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                   "bound\t{$src, $dst|$dst, $src}", [], IIC_BOUND>, OpSize,
                   Requires<[In32BitMode]>;
def BOUNDS32rm : I<0x62, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                   "bound\t{$src, $dst|$dst, $src}", [], IIC_BOUND>,
                   Requires<[In32BitMode]>;

// Adjust RPL Field of Segment Selector
def ARPL16rr : I<0x63, MRMDestReg, (outs GR16:$src), (ins GR16:$dst),
                 "arpl\t{$src, $dst|$dst, $src}", [], IIC_ARPL_REG>,
                 Requires<[In32BitMode]>;
def ARPL16mr : I<0x63, MRMSrcMem, (outs GR16:$src), (ins i16mem:$dst),
                 "arpl\t{$src, $dst|$dst, $src}", [], IIC_ARPL_MEM>,
                 Requires<[In32BitMode]>;

//===----------------------------------------------------------------------===//
// MOVBE Instructions
//
let Predicates = [HasMOVBE] in {
  def MOVBE16rm : I<0xF0, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                    "movbe{w}\t{$src, $dst|$dst, $src}",
                    [(set GR16:$dst, (bswap (loadi16 addr:$src)))], IIC_MOVBE>,
                    OpSize, T8;
  def MOVBE32rm : I<0xF0, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                    "movbe{l}\t{$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (bswap (loadi32 addr:$src)))], IIC_MOVBE>,
                    T8;
  def MOVBE64rm : RI<0xF0, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                     "movbe{q}\t{$src, $dst|$dst, $src}",
                     [(set GR64:$dst, (bswap (loadi64 addr:$src)))], IIC_MOVBE>,
                     T8;
  def MOVBE16mr : I<0xF1, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                    "movbe{w}\t{$src, $dst|$dst, $src}",
                    [(store (bswap GR16:$src), addr:$dst)], IIC_MOVBE>,
                    OpSize, T8;
  def MOVBE32mr : I<0xF1, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                    "movbe{l}\t{$src, $dst|$dst, $src}",
                    [(store (bswap GR32:$src), addr:$dst)], IIC_MOVBE>,
                    T8;
  def MOVBE64mr : RI<0xF1, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                     "movbe{q}\t{$src, $dst|$dst, $src}",
                     [(store (bswap GR64:$src), addr:$dst)], IIC_MOVBE>,
                     T8;
}

//===----------------------------------------------------------------------===//
// RDRAND Instruction
//
let Predicates = [HasRDRAND], Defs = [EFLAGS] in {
  def RDRAND16r : I<0xC7, MRM6r, (outs GR16:$dst), (ins),
                    "rdrand{w}\t$dst",
                    [(set GR16:$dst, EFLAGS, (X86rdrand))]>, OpSize, TB;
  def RDRAND32r : I<0xC7, MRM6r, (outs GR32:$dst), (ins),
                    "rdrand{l}\t$dst",
                    [(set GR32:$dst, EFLAGS, (X86rdrand))]>, TB;
  def RDRAND64r : RI<0xC7, MRM6r, (outs GR64:$dst), (ins),
                     "rdrand{q}\t$dst",
                     [(set GR64:$dst, EFLAGS, (X86rdrand))]>, TB;
}

//===----------------------------------------------------------------------===//
// LZCNT Instruction
//
let Predicates = [HasLZCNT], Defs = [EFLAGS] in {
  def LZCNT16rr : I<0xBD, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                    "lzcnt{w}\t{$src, $dst|$dst, $src}",
                    [(set GR16:$dst, (ctlz GR16:$src)), (implicit EFLAGS)]>, XS,
                    OpSize;
  def LZCNT16rm : I<0xBD, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                    "lzcnt{w}\t{$src, $dst|$dst, $src}",
                    [(set GR16:$dst, (ctlz (loadi16 addr:$src))),
                     (implicit EFLAGS)]>, XS, OpSize;

  def LZCNT32rr : I<0xBD, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                    "lzcnt{l}\t{$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (ctlz GR32:$src)), (implicit EFLAGS)]>, XS;
  def LZCNT32rm : I<0xBD, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                    "lzcnt{l}\t{$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (ctlz (loadi32 addr:$src))),
                     (implicit EFLAGS)]>, XS;

  def LZCNT64rr : RI<0xBD, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                     "lzcnt{q}\t{$src, $dst|$dst, $src}",
                     [(set GR64:$dst, (ctlz GR64:$src)), (implicit EFLAGS)]>,
                     XS;
  def LZCNT64rm : RI<0xBD, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                     "lzcnt{q}\t{$src, $dst|$dst, $src}",
                     [(set GR64:$dst, (ctlz (loadi64 addr:$src))),
                      (implicit EFLAGS)]>, XS;
}

//===----------------------------------------------------------------------===//
// BMI Instructions
//
let Predicates = [HasBMI], Defs = [EFLAGS] in {
  def TZCNT16rr : I<0xBC, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                    "tzcnt{w}\t{$src, $dst|$dst, $src}",
                    [(set GR16:$dst, (cttz GR16:$src)), (implicit EFLAGS)]>, XS,
                    OpSize;
  def TZCNT16rm : I<0xBC, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                    "tzcnt{w}\t{$src, $dst|$dst, $src}",
                    [(set GR16:$dst, (cttz (loadi16 addr:$src))),
                     (implicit EFLAGS)]>, XS, OpSize;

  def TZCNT32rr : I<0xBC, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                    "tzcnt{l}\t{$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (cttz GR32:$src)), (implicit EFLAGS)]>, XS;
  def TZCNT32rm : I<0xBC, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                    "tzcnt{l}\t{$src, $dst|$dst, $src}",
                    [(set GR32:$dst, (cttz (loadi32 addr:$src))),
                     (implicit EFLAGS)]>, XS;

  def TZCNT64rr : RI<0xBC, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                     "tzcnt{q}\t{$src, $dst|$dst, $src}",
                     [(set GR64:$dst, (cttz GR64:$src)), (implicit EFLAGS)]>,
                     XS;
  def TZCNT64rm : RI<0xBC, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                     "tzcnt{q}\t{$src, $dst|$dst, $src}",
                     [(set GR64:$dst, (cttz (loadi64 addr:$src))),
                      (implicit EFLAGS)]>, XS;
}

multiclass bmi_bls<string mnemonic, Format RegMRM, Format MemMRM,
                  RegisterClass RC, X86MemOperand x86memop, SDNode OpNode,
                  PatFrag ld_frag> {
  def rr : I<0xF3, RegMRM, (outs RC:$dst), (ins RC:$src),
             !strconcat(mnemonic, "\t{$src, $dst|$dst, $src}"),
             [(set RC:$dst, EFLAGS, (OpNode RC:$src))]>, T8, VEX_4V;
  def rm : I<0xF3, MemMRM, (outs RC:$dst), (ins x86memop:$src),
             !strconcat(mnemonic, "\t{$src, $dst|$dst, $src}"),
             [(set RC:$dst, EFLAGS, (OpNode (ld_frag addr:$src)))]>,
             T8, VEX_4V;
}

let Predicates = [HasBMI], Defs = [EFLAGS] in {
  defm BLSR32 : bmi_bls<"blsr{l}", MRM1r, MRM1m, GR32, i32mem,
                        X86blsr_flag, loadi32>;
  defm BLSR64 : bmi_bls<"blsr{q}", MRM1r, MRM1m, GR64, i64mem,
                        X86blsr_flag, loadi64>, VEX_W;
  defm BLSMSK32 : bmi_bls<"blsmsk{l}", MRM2r, MRM2m, GR32, i32mem,
                          X86blsmsk_flag, loadi32>;
  defm BLSMSK64 : bmi_bls<"blsmsk{q}", MRM2r, MRM2m, GR64, i64mem,
                          X86blsmsk_flag, loadi64>, VEX_W;
  defm BLSI32 : bmi_bls<"blsi{l}", MRM3r, MRM3m, GR32, i32mem,
                        X86blsi_flag, loadi32>;
  defm BLSI64 : bmi_bls<"blsi{q}", MRM3r, MRM3m, GR64, i64mem,
                        X86blsi_flag, loadi64>, VEX_W;
}

multiclass bmi_bextr_bzhi<bits<8> opc, string mnemonic, RegisterClass RC,
                          X86MemOperand x86memop, Intrinsic Int,
                          PatFrag ld_frag> {
  def rr : I<opc, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
             !strconcat(mnemonic, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (Int RC:$src1, RC:$src2)), (implicit EFLAGS)]>,
             T8, VEX_4VOp3;
  def rm : I<opc, MRMSrcMem, (outs RC:$dst), (ins x86memop:$src1, RC:$src2),
             !strconcat(mnemonic, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (Int (ld_frag addr:$src1), RC:$src2)),
              (implicit EFLAGS)]>, T8, VEX_4VOp3;
}

let Predicates = [HasBMI], Defs = [EFLAGS] in {
  defm BEXTR32 : bmi_bextr_bzhi<0xF7, "bextr{l}", GR32, i32mem,
                                int_x86_bmi_bextr_32, loadi32>;
  defm BEXTR64 : bmi_bextr_bzhi<0xF7, "bextr{q}", GR64, i64mem,
                                int_x86_bmi_bextr_64, loadi64>, VEX_W;
}

let Predicates = [HasBMI2], Defs = [EFLAGS] in {
  defm BZHI32 : bmi_bextr_bzhi<0xF5, "bzhi{l}", GR32, i32mem,
                               int_x86_bmi_bzhi_32, loadi32>;
  defm BZHI64 : bmi_bextr_bzhi<0xF5, "bzhi{q}", GR64, i64mem,
                               int_x86_bmi_bzhi_64, loadi64>, VEX_W;
}

multiclass bmi_pdep_pext<string mnemonic, RegisterClass RC,
                         X86MemOperand x86memop, Intrinsic Int,
                         PatFrag ld_frag> {
  def rr : I<0xF5, MRMSrcReg, (outs RC:$dst), (ins RC:$src1, RC:$src2),
             !strconcat(mnemonic, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (Int RC:$src1, RC:$src2))]>,
             VEX_4V;
  def rm : I<0xF5, MRMSrcMem, (outs RC:$dst), (ins RC:$src1, x86memop:$src2),
             !strconcat(mnemonic, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
             [(set RC:$dst, (Int RC:$src1, (ld_frag addr:$src2)))]>, VEX_4V;
}

let Predicates = [HasBMI2] in {
  defm PDEP32 : bmi_pdep_pext<"pdep{l}", GR32, i32mem,
                               int_x86_bmi_pdep_32, loadi32>, T8XD;
  defm PDEP64 : bmi_pdep_pext<"pdep{q}", GR64, i64mem,
                               int_x86_bmi_pdep_64, loadi64>, T8XD, VEX_W;
  defm PEXT32 : bmi_pdep_pext<"pext{l}", GR32, i32mem,
                               int_x86_bmi_pext_32, loadi32>, T8XS;
  defm PEXT64 : bmi_pdep_pext<"pext{q}", GR64, i64mem,
                               int_x86_bmi_pext_64, loadi64>, T8XS, VEX_W;
}

//===----------------------------------------------------------------------===//
// Subsystems.
//===----------------------------------------------------------------------===//

include "X86InstrArithmetic.td"
include "X86InstrCMovSetCC.td"
include "X86InstrExtension.td"
include "X86InstrControl.td"
include "X86InstrShiftRotate.td"

// X87 Floating Point Stack.
include "X86InstrFPStack.td"

// SIMD support (SSE, MMX and AVX)
include "X86InstrFragmentsSIMD.td"

// FMA - Fused Multiply-Add support (requires FMA)
include "X86InstrFMA.td"

// XOP
include "X86InstrXOP.td"

// SSE, MMX and 3DNow! vector support.
include "X86InstrSSE.td"
include "X86InstrMMX.td"
include "X86Instr3DNow.td"

include "X86InstrVMX.td"
include "X86InstrSVM.td"

include "X86InstrTSX.td"

// System instructions.
include "X86InstrSystem.td"

// Compiler Pseudo Instructions and Pat Patterns
include "X86InstrCompiler.td"

//===----------------------------------------------------------------------===//
// Assembler Mnemonic Aliases
//===----------------------------------------------------------------------===//

def : MnemonicAlias<"call", "calll">, Requires<[In32BitMode]>;
def : MnemonicAlias<"call", "callq">, Requires<[In64BitMode]>;

def : MnemonicAlias<"cbw",  "cbtw">;
def : MnemonicAlias<"cwde", "cwtl">;
def : MnemonicAlias<"cwd",  "cwtd">;
def : MnemonicAlias<"cdq", "cltd">;
def : MnemonicAlias<"cdqe", "cltq">;
def : MnemonicAlias<"cqo", "cqto">;

// lret maps to lretl, it is not ambiguous with lretq.
def : MnemonicAlias<"lret", "lretl">;

def : MnemonicAlias<"leavel", "leave">, Requires<[In32BitMode]>;
def : MnemonicAlias<"leaveq", "leave">, Requires<[In64BitMode]>;

def : MnemonicAlias<"loopz", "loope">;
def : MnemonicAlias<"loopnz", "loopne">;

def : MnemonicAlias<"pop", "popl">, Requires<[In32BitMode]>;
def : MnemonicAlias<"pop", "popq">, Requires<[In64BitMode]>;
def : MnemonicAlias<"popf", "popfl">, Requires<[In32BitMode]>;
def : MnemonicAlias<"popf", "popfq">, Requires<[In64BitMode]>;
def : MnemonicAlias<"popfd",  "popfl">;

// FIXME: This is wrong for "push reg".  "push %bx" should turn into pushw in
// all modes.  However: "push (addr)" and "push $42" should default to
// pushl/pushq depending on the current mode.  Similar for "pop %bx"
def : MnemonicAlias<"push", "pushl">, Requires<[In32BitMode]>;
def : MnemonicAlias<"push", "pushq">, Requires<[In64BitMode]>;
def : MnemonicAlias<"pushf", "pushfl">, Requires<[In32BitMode]>;
def : MnemonicAlias<"pushf", "pushfq">, Requires<[In64BitMode]>;
def : MnemonicAlias<"pushfd", "pushfl">;

def : MnemonicAlias<"repe", "rep">;
def : MnemonicAlias<"repz", "rep">;
def : MnemonicAlias<"repnz", "repne">;

def : MnemonicAlias<"retl", "ret">, Requires<[In32BitMode]>;
def : MnemonicAlias<"retq", "ret">, Requires<[In64BitMode]>;

def : MnemonicAlias<"salb", "shlb">;
def : MnemonicAlias<"salw", "shlw">;
def : MnemonicAlias<"sall", "shll">;
def : MnemonicAlias<"salq", "shlq">;

def : MnemonicAlias<"smovb", "movsb">;
def : MnemonicAlias<"smovw", "movsw">;
def : MnemonicAlias<"smovl", "movsl">;
def : MnemonicAlias<"smovq", "movsq">;

def : MnemonicAlias<"ud2a", "ud2">;
def : MnemonicAlias<"verrw", "verr">;

// System instruction aliases.
def : MnemonicAlias<"iret", "iretl">;
def : MnemonicAlias<"sysret", "sysretl">;
def : MnemonicAlias<"sysexit", "sysexitl">;

def : MnemonicAlias<"lgdtl", "lgdt">, Requires<[In32BitMode]>;
def : MnemonicAlias<"lgdtq", "lgdt">, Requires<[In64BitMode]>;
def : MnemonicAlias<"lidtl", "lidt">, Requires<[In32BitMode]>;
def : MnemonicAlias<"lidtq", "lidt">, Requires<[In64BitMode]>;
def : MnemonicAlias<"sgdtl", "sgdt">, Requires<[In32BitMode]>;
def : MnemonicAlias<"sgdtq", "sgdt">, Requires<[In64BitMode]>;
def : MnemonicAlias<"sidtl", "sidt">, Requires<[In32BitMode]>;
def : MnemonicAlias<"sidtq", "sidt">, Requires<[In64BitMode]>;


// Floating point stack aliases.
def : MnemonicAlias<"fcmovz",   "fcmove">;
def : MnemonicAlias<"fcmova",   "fcmovnbe">;
def : MnemonicAlias<"fcmovnae", "fcmovb">;
def : MnemonicAlias<"fcmovna",  "fcmovbe">;
def : MnemonicAlias<"fcmovae",  "fcmovnb">;
def : MnemonicAlias<"fcomip",   "fcompi">;
def : MnemonicAlias<"fildq",    "fildll">;
def : MnemonicAlias<"fistpq",   "fistpll">;
def : MnemonicAlias<"fisttpq",  "fisttpll">;
def : MnemonicAlias<"fldcww",   "fldcw">;
def : MnemonicAlias<"fnstcww", "fnstcw">;
def : MnemonicAlias<"fnstsww", "fnstsw">;
def : MnemonicAlias<"fucomip",  "fucompi">;
def : MnemonicAlias<"fwait",    "wait">;


class CondCodeAlias<string Prefix,string Suffix, string OldCond, string NewCond>
  : MnemonicAlias<!strconcat(Prefix, OldCond, Suffix),
                  !strconcat(Prefix, NewCond, Suffix)>;

/// IntegerCondCodeMnemonicAlias - This multiclass defines a bunch of
/// MnemonicAlias's that canonicalize the condition code in a mnemonic, for
/// example "setz" -> "sete".
multiclass IntegerCondCodeMnemonicAlias<string Prefix, string Suffix> {
  def C   : CondCodeAlias<Prefix, Suffix, "c",   "b">;   // setc   -> setb
  def Z   : CondCodeAlias<Prefix, Suffix, "z" ,  "e">;   // setz   -> sete
  def NA  : CondCodeAlias<Prefix, Suffix, "na",  "be">;  // setna  -> setbe
  def NB  : CondCodeAlias<Prefix, Suffix, "nb",  "ae">;  // setnb  -> setae
  def NC  : CondCodeAlias<Prefix, Suffix, "nc",  "ae">;  // setnc  -> setae
  def NG  : CondCodeAlias<Prefix, Suffix, "ng",  "le">;  // setng  -> setle
  def NL  : CondCodeAlias<Prefix, Suffix, "nl",  "ge">;  // setnl  -> setge
  def NZ  : CondCodeAlias<Prefix, Suffix, "nz",  "ne">;  // setnz  -> setne
  def PE  : CondCodeAlias<Prefix, Suffix, "pe",  "p">;   // setpe  -> setp
  def PO  : CondCodeAlias<Prefix, Suffix, "po",  "np">;  // setpo  -> setnp

  def NAE : CondCodeAlias<Prefix, Suffix, "nae", "b">;   // setnae -> setb
  def NBE : CondCodeAlias<Prefix, Suffix, "nbe", "a">;   // setnbe -> seta
  def NGE : CondCodeAlias<Prefix, Suffix, "nge", "l">;   // setnge -> setl
  def NLE : CondCodeAlias<Prefix, Suffix, "nle", "g">;   // setnle -> setg
}

// Aliases for set<CC>
defm : IntegerCondCodeMnemonicAlias<"set", "">;
// Aliases for j<CC>
defm : IntegerCondCodeMnemonicAlias<"j", "">;
// Aliases for cmov<CC>{w,l,q}
defm : IntegerCondCodeMnemonicAlias<"cmov", "w">;
defm : IntegerCondCodeMnemonicAlias<"cmov", "l">;
defm : IntegerCondCodeMnemonicAlias<"cmov", "q">;


//===----------------------------------------------------------------------===//
// Assembler Instruction Aliases
//===----------------------------------------------------------------------===//

// aad/aam default to base 10 if no operand is specified.
def : InstAlias<"aad", (AAD8i8 10)>;
def : InstAlias<"aam", (AAM8i8 10)>;

// Disambiguate the mem/imm form of bt-without-a-suffix as btl.
def : InstAlias<"bt $imm, $mem", (BT32mi8 i32mem:$mem, i32i8imm:$imm)>;

// clr aliases.
def : InstAlias<"clrb $reg", (XOR8rr  GR8 :$reg, GR8 :$reg)>;
def : InstAlias<"clrw $reg", (XOR16rr GR16:$reg, GR16:$reg)>;
def : InstAlias<"clrl $reg", (XOR32rr GR32:$reg, GR32:$reg)>;
def : InstAlias<"clrq $reg", (XOR64rr GR64:$reg, GR64:$reg)>;

// div and idiv aliases for explicit A register.
def : InstAlias<"divb $src, %al",  (DIV8r  GR8 :$src)>;
def : InstAlias<"divw $src, %ax",  (DIV16r GR16:$src)>;
def : InstAlias<"divl $src, %eax", (DIV32r GR32:$src)>;
def : InstAlias<"divq $src, %rax", (DIV64r GR64:$src)>;
def : InstAlias<"divb $src, %al",  (DIV8m  i8mem :$src)>;
def : InstAlias<"divw $src, %ax",  (DIV16m i16mem:$src)>;
def : InstAlias<"divl $src, %eax", (DIV32m i32mem:$src)>;
def : InstAlias<"divq $src, %rax", (DIV64m i64mem:$src)>;
def : InstAlias<"idivb $src, %al",  (IDIV8r  GR8 :$src)>;
def : InstAlias<"idivw $src, %ax",  (IDIV16r GR16:$src)>;
def : InstAlias<"idivl $src, %eax", (IDIV32r GR32:$src)>;
def : InstAlias<"idivq $src, %rax", (IDIV64r GR64:$src)>;
def : InstAlias<"idivb $src, %al",  (IDIV8m  i8mem :$src)>;
def : InstAlias<"idivw $src, %ax",  (IDIV16m i16mem:$src)>;
def : InstAlias<"idivl $src, %eax", (IDIV32m i32mem:$src)>;
def : InstAlias<"idivq $src, %rax", (IDIV64m i64mem:$src)>;



// Various unary fpstack operations default to operating on on ST1.
// For example, "fxch" -> "fxch %st(1)"
def : InstAlias<"faddp",        (ADD_FPrST0  ST1), 0>;
def : InstAlias<"fsubp",        (SUBR_FPrST0 ST1)>;
def : InstAlias<"fsubrp",       (SUB_FPrST0  ST1)>;
def : InstAlias<"fmulp",        (MUL_FPrST0  ST1)>;
def : InstAlias<"fdivp",        (DIVR_FPrST0 ST1)>;
def : InstAlias<"fdivrp",       (DIV_FPrST0  ST1)>;
def : InstAlias<"fxch",         (XCH_F       ST1)>;
def : InstAlias<"fcomi",        (COM_FIr     ST1)>;
def : InstAlias<"fcompi",       (COM_FIPr    ST1)>;
def : InstAlias<"fucom",        (UCOM_Fr     ST1)>;
def : InstAlias<"fucomp",       (UCOM_FPr    ST1)>;
def : InstAlias<"fucomi",       (UCOM_FIr    ST1)>;
def : InstAlias<"fucompi",      (UCOM_FIPr   ST1)>;

// Handle fmul/fadd/fsub/fdiv instructions with explicitly written st(0) op.
// For example, "fadd %st(4), %st(0)" -> "fadd %st(4)".  We also disambiguate
// instructions like "fadd %st(0), %st(0)" as "fadd %st(0)" for consistency with
// gas.
multiclass FpUnaryAlias<string Mnemonic, Instruction Inst, bit EmitAlias = 1> {
 def : InstAlias<!strconcat(Mnemonic, " $op, %st(0)"),
                 (Inst RST:$op), EmitAlias>;
 def : InstAlias<!strconcat(Mnemonic, " %st(0), %st(0)"),
                 (Inst ST0), EmitAlias>;
}

defm : FpUnaryAlias<"fadd",   ADD_FST0r>;
defm : FpUnaryAlias<"faddp",  ADD_FPrST0, 0>;
defm : FpUnaryAlias<"fsub",   SUB_FST0r>;
defm : FpUnaryAlias<"fsubp",  SUBR_FPrST0>;
defm : FpUnaryAlias<"fsubr",  SUBR_FST0r>;
defm : FpUnaryAlias<"fsubrp", SUB_FPrST0>;
defm : FpUnaryAlias<"fmul",   MUL_FST0r>;
defm : FpUnaryAlias<"fmulp",  MUL_FPrST0>;
defm : FpUnaryAlias<"fdiv",   DIV_FST0r>;
defm : FpUnaryAlias<"fdivp",  DIVR_FPrST0>;
defm : FpUnaryAlias<"fdivr",  DIVR_FST0r>;
defm : FpUnaryAlias<"fdivrp", DIV_FPrST0>;
defm : FpUnaryAlias<"fcomi",   COM_FIr, 0>;
defm : FpUnaryAlias<"fucomi",  UCOM_FIr, 0>;
defm : FpUnaryAlias<"fcompi",   COM_FIPr>;
defm : FpUnaryAlias<"fucompi",  UCOM_FIPr>;


// Handle "f{mulp,addp} st(0), $op" the same as "f{mulp,addp} $op", since they
// commute.  We also allow fdiv[r]p/fsubrp even though they don't commute,
// solely because gas supports it.
def : InstAlias<"faddp %st(0), $op", (ADD_FPrST0 RST:$op), 0>;
def : InstAlias<"fmulp %st(0), $op", (MUL_FPrST0 RST:$op)>;
def : InstAlias<"fsubp %st(0), $op", (SUBR_FPrST0 RST:$op)>;
def : InstAlias<"fsubrp %st(0), $op", (SUB_FPrST0 RST:$op)>;
def : InstAlias<"fdivp %st(0), $op", (DIVR_FPrST0 RST:$op)>;
def : InstAlias<"fdivrp %st(0), $op", (DIV_FPrST0 RST:$op)>;

// We accept "fnstsw %eax" even though it only writes %ax.
def : InstAlias<"fnstsw %eax", (FNSTSW16r)>;
def : InstAlias<"fnstsw %al" , (FNSTSW16r)>;
def : InstAlias<"fnstsw"     , (FNSTSW16r)>;

// lcall and ljmp aliases.  This seems to be an odd mapping in 64-bit mode, but
// this is compatible with what GAS does.
def : InstAlias<"lcall $seg, $off", (FARCALL32i i32imm:$off, i16imm:$seg)>;
def : InstAlias<"ljmp $seg, $off",  (FARJMP32i  i32imm:$off, i16imm:$seg)>;
def : InstAlias<"lcall *$dst",      (FARCALL32m opaque48mem:$dst)>;
def : InstAlias<"ljmp *$dst",       (FARJMP32m  opaque48mem:$dst)>;

// "imul <imm>, B" is an alias for "imul <imm>, B, B".
def : InstAlias<"imulw $imm, $r", (IMUL16rri  GR16:$r, GR16:$r, i16imm:$imm)>;
def : InstAlias<"imulw $imm, $r", (IMUL16rri8 GR16:$r, GR16:$r, i16i8imm:$imm)>;
def : InstAlias<"imull $imm, $r", (IMUL32rri  GR32:$r, GR32:$r, i32imm:$imm)>;
def : InstAlias<"imull $imm, $r", (IMUL32rri8 GR32:$r, GR32:$r, i32i8imm:$imm)>;
def : InstAlias<"imulq $imm, $r",(IMUL64rri32 GR64:$r, GR64:$r,i64i32imm:$imm)>;
def : InstAlias<"imulq $imm, $r", (IMUL64rri8 GR64:$r, GR64:$r, i64i8imm:$imm)>;

// inb %dx -> inb %al, %dx
def : InstAlias<"inb %dx", (IN8rr)>;
def : InstAlias<"inw %dx", (IN16rr)>;
def : InstAlias<"inl %dx", (IN32rr)>;
def : InstAlias<"inb $port", (IN8ri i8imm:$port)>;
def : InstAlias<"inw $port", (IN16ri i8imm:$port)>;
def : InstAlias<"inl $port", (IN32ri i8imm:$port)>;


// jmp and call aliases for lcall and ljmp.  jmp $42,$5 -> ljmp
def : InstAlias<"call $seg, $off",  (FARCALL32i i32imm:$off, i16imm:$seg)>;
def : InstAlias<"jmp $seg, $off",   (FARJMP32i  i32imm:$off, i16imm:$seg)>;
def : InstAlias<"callw $seg, $off", (FARCALL16i i16imm:$off, i16imm:$seg)>;
def : InstAlias<"jmpw $seg, $off",  (FARJMP16i  i16imm:$off, i16imm:$seg)>;
def : InstAlias<"calll $seg, $off", (FARCALL32i i32imm:$off, i16imm:$seg)>;
def : InstAlias<"jmpl $seg, $off",  (FARJMP32i  i32imm:$off, i16imm:$seg)>;

// Force mov without a suffix with a segment and mem to prefer the 'l' form of
// the move.  All segment/mem forms are equivalent, this has the shortest
// encoding.
def : InstAlias<"mov $mem, $seg", (MOV32sm SEGMENT_REG:$seg, i32mem:$mem)>;
def : InstAlias<"mov $seg, $mem", (MOV32ms i32mem:$mem, SEGMENT_REG:$seg)>;

// Match 'movq <largeimm>, <reg>' as an alias for movabsq.
def : InstAlias<"movq $imm, $reg", (MOV64ri GR64:$reg, i64imm:$imm)>;

// Match 'movq GR64, MMX' as an alias for movd.
def : InstAlias<"movq $src, $dst",
                (MMX_MOVD64to64rr VR64:$dst, GR64:$src), 0>;
def : InstAlias<"movq $src, $dst",
                (MMX_MOVD64from64rr GR64:$dst, VR64:$src), 0>;

// movsd with no operands (as opposed to the SSE scalar move of a double) is an
// alias for movsl. (as in rep; movsd)
def : InstAlias<"movsd", (MOVSD)>;

// movsx aliases
def : InstAlias<"movsx $src, $dst", (MOVSX16rr8 GR16:$dst, GR8:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX16rm8 GR16:$dst, i8mem:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX32rr8 GR32:$dst, GR8:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX32rr16 GR32:$dst, GR16:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX64rr8 GR64:$dst, GR8:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX64rr16 GR64:$dst, GR16:$src), 0>;
def : InstAlias<"movsx $src, $dst", (MOVSX64rr32 GR64:$dst, GR32:$src), 0>;

// movzx aliases
def : InstAlias<"movzx $src, $dst", (MOVZX16rr8 GR16:$dst, GR8:$src), 0>;
def : InstAlias<"movzx $src, $dst", (MOVZX16rm8 GR16:$dst, i8mem:$src), 0>;
def : InstAlias<"movzx $src, $dst", (MOVZX32rr8 GR32:$dst, GR8:$src), 0>;
def : InstAlias<"movzx $src, $dst", (MOVZX32rr16 GR32:$dst, GR16:$src), 0>;
def : InstAlias<"movzx $src, $dst", (MOVZX64rr8_Q GR64:$dst, GR8:$src), 0>;
def : InstAlias<"movzx $src, $dst", (MOVZX64rr16_Q GR64:$dst, GR16:$src), 0>;
// Note: No GR32->GR64 movzx form.

// outb %dx -> outb %al, %dx
def : InstAlias<"outb %dx", (OUT8rr)>;
def : InstAlias<"outw %dx", (OUT16rr)>;
def : InstAlias<"outl %dx", (OUT32rr)>;
def : InstAlias<"outb $port", (OUT8ir i8imm:$port)>;
def : InstAlias<"outw $port", (OUT16ir i8imm:$port)>;
def : InstAlias<"outl $port", (OUT32ir i8imm:$port)>;

// 'sldt <mem>' can be encoded with either sldtw or sldtq with the same
// effect (both store to a 16-bit mem).  Force to sldtw to avoid ambiguity
// errors, since its encoding is the most compact.
def : InstAlias<"sldt $mem", (SLDT16m i16mem:$mem)>;

// shld/shrd op,op -> shld op, op, CL
def : InstAlias<"shldw $r2, $r1", (SHLD16rrCL GR16:$r1, GR16:$r2)>;
def : InstAlias<"shldl $r2, $r1", (SHLD32rrCL GR32:$r1, GR32:$r2)>;
def : InstAlias<"shldq $r2, $r1", (SHLD64rrCL GR64:$r1, GR64:$r2)>;
def : InstAlias<"shrdw $r2, $r1", (SHRD16rrCL GR16:$r1, GR16:$r2)>;
def : InstAlias<"shrdl $r2, $r1", (SHRD32rrCL GR32:$r1, GR32:$r2)>;
def : InstAlias<"shrdq $r2, $r1", (SHRD64rrCL GR64:$r1, GR64:$r2)>;

def : InstAlias<"shldw $reg, $mem", (SHLD16mrCL i16mem:$mem, GR16:$reg)>;
def : InstAlias<"shldl $reg, $mem", (SHLD32mrCL i32mem:$mem, GR32:$reg)>;
def : InstAlias<"shldq $reg, $mem", (SHLD64mrCL i64mem:$mem, GR64:$reg)>;
def : InstAlias<"shrdw $reg, $mem", (SHRD16mrCL i16mem:$mem, GR16:$reg)>;
def : InstAlias<"shrdl $reg, $mem", (SHRD32mrCL i32mem:$mem, GR32:$reg)>;
def : InstAlias<"shrdq $reg, $mem", (SHRD64mrCL i64mem:$mem, GR64:$reg)>;

/*  FIXME: This is disabled because the asm matcher is currently incapable of
 *  matching a fixed immediate like $1.
// "shl X, $1" is an alias for "shl X".
multiclass ShiftRotateByOneAlias<string Mnemonic, string Opc> {
 def : InstAlias<!strconcat(Mnemonic, "b $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "8r1")) GR8:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "w $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "16r1")) GR16:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "l $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "32r1")) GR32:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "q $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "64r1")) GR64:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "b $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "8m1")) i8mem:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "w $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "16m1")) i16mem:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "l $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "32m1")) i32mem:$op)>;
 def : InstAlias<!strconcat(Mnemonic, "q $op, $$1"),
                 (!cast<Instruction>(!strconcat(Opc, "64m1")) i64mem:$op)>;
}

defm : ShiftRotateByOneAlias<"rcl", "RCL">;
defm : ShiftRotateByOneAlias<"rcr", "RCR">;
defm : ShiftRotateByOneAlias<"rol", "ROL">;
defm : ShiftRotateByOneAlias<"ror", "ROR">;
FIXME */

// test: We accept "testX <reg>, <mem>" and "testX <mem>, <reg>" as synonyms.
def : InstAlias<"testb $val, $mem", (TEST8rm  GR8 :$val, i8mem :$mem)>;
def : InstAlias<"testw $val, $mem", (TEST16rm GR16:$val, i16mem:$mem)>;
def : InstAlias<"testl $val, $mem", (TEST32rm GR32:$val, i32mem:$mem)>;
def : InstAlias<"testq $val, $mem", (TEST64rm GR64:$val, i64mem:$mem)>;

// xchg: We accept "xchgX <reg>, <mem>" and "xchgX <mem>, <reg>" as synonyms.
def : InstAlias<"xchgb $mem, $val", (XCHG8rm  GR8 :$val, i8mem :$mem)>;
def : InstAlias<"xchgw $mem, $val", (XCHG16rm GR16:$val, i16mem:$mem)>;
def : InstAlias<"xchgl $mem, $val", (XCHG32rm GR32:$val, i32mem:$mem)>;
def : InstAlias<"xchgq $mem, $val", (XCHG64rm GR64:$val, i64mem:$mem)>;

// xchg: We accept "xchgX <reg>, %eax" and "xchgX %eax, <reg>" as synonyms.
def : InstAlias<"xchgw %ax, $src", (XCHG16ar GR16:$src)>;
def : InstAlias<"xchgl %eax, $src", (XCHG32ar GR32:$src)>, Requires<[In32BitMode]>;
def : InstAlias<"xchgl %eax, $src", (XCHG32ar64 GR32_NOAX:$src)>, Requires<[In64BitMode]>;
def : InstAlias<"xchgq %rax, $src", (XCHG64ar GR64:$src)>;
