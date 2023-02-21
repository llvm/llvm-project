//===-- X86InstrInfo.td - Main X86 Instruction Properties --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 properties of the instructions which are needed 
// for code generation, machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// X86 specific DAG Nodes.
//

def SDTX86CmpTest : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisInt<1>,
                                         SDTCisSameAs<1, 2>]>;
def SDTX86FCmp    : SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisFP<1>,
                                         SDTCisSameAs<1, 2>]>;

def SDTX86Cmov    : SDTypeProfile<1, 4,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>, SDTCisVT<4, i32>]>;

// Unary and binary operator instructions that set EFLAGS as a side-effect.
def SDTUnaryArithWithFlags : SDTypeProfile<2, 1,
                                           [SDTCisSameAs<0, 2>,
                                            SDTCisInt<0>, SDTCisVT<1, i32>]>;

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

def SDTX86rdpkru : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def SDTX86wrpkru : SDTypeProfile<0, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>,
                                        SDTCisVT<2, i32>]>;

def SDTX86cas : SDTypeProfile<0, 3, [SDTCisPtrTy<0>, SDTCisInt<1>,
                                     SDTCisVT<2, i8>]>;
def SDTX86cas8pair : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;
def SDTX86cas16pair : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i64>]>;

def SDTLockBinaryArithWithFlags : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                                       SDTCisPtrTy<1>,
                                                       SDTCisInt<2>]>;

def SDTLockUnaryArithWithFlags : SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                                      SDTCisPtrTy<1>]>;

def SDTX86Ret     : SDTypeProfile<0, -1, [SDTCisVT<0, i32>]>;

def SDT_X86CallSeqStart : SDCallSeqStart<[SDTCisVT<0, i32>,
                                          SDTCisVT<1, i32>]>;
def SDT_X86CallSeqEnd   : SDCallSeqEnd<[SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32>]>;

def SDT_X86Call   : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def SDT_X86NtBrind : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def SDT_X86VASTART_SAVE_XMM_REGS : SDTypeProfile<0, -1, [SDTCisVT<0, i8>,
                                                         SDTCisPtrTy<1>]>;

def SDT_X86VAARG : SDTypeProfile<1, -1, [SDTCisPtrTy<0>,
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

def SDT_X86DYN_ALLOCA : SDTypeProfile<0, 1, [SDTCisVT<0, iPTR>]>;

def SDT_X86SEG_ALLOCA : SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;

def SDT_X86PROBED_ALLOCA : SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;

def SDT_X86EHRET : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TCRET : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i32>]>;

def SDT_X86ENQCMD : SDTypeProfile<1, 2, [SDTCisVT<0, i32>,
                                         SDTCisPtrTy<1>, SDTCisSameAs<1, 2>]>;

def SDT_X86AESENCDECKL : SDTypeProfile<2, 2, [SDTCisVT<0, v2i64>,
                                              SDTCisVT<1, i32>,
                                              SDTCisVT<2, v2i64>,
                                              SDTCisPtrTy<3>]>;

def SDTX86Cmpccxadd : SDTypeProfile<1, 4, [SDTCisSameAs<0, 2>,
                                           SDTCisPtrTy<1>, SDTCisSameAs<2, 3>,
                                           SDTCisVT<4, i8>]>;

def X86MFence : SDNode<"X86ISD::MFENCE", SDTNone, [SDNPHasChain]>;


def X86bsf     : SDNode<"X86ISD::BSF",      SDTUnaryArithWithFlags>;
def X86bsr     : SDNode<"X86ISD::BSR",      SDTUnaryArithWithFlags>;
def X86fshl    : SDNode<"X86ISD::FSHL",     SDTIntShiftDOp>;
def X86fshr    : SDNode<"X86ISD::FSHR",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest>;
def X86fcmp    : SDNode<"X86ISD::FCMP",     SDTX86FCmp>;
def X86strict_fcmp : SDNode<"X86ISD::STRICT_FCMP", SDTX86FCmp, [SDNPHasChain]>;
def X86strict_fcmps : SDNode<"X86ISD::STRICT_FCMPS", SDTX86FCmp, [SDNPHasChain]>;
def X86bt      : SDNode<"X86ISD::BT",       SDTX86CmpTest>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC>;
def X86setcc_c : SDNode<"X86ISD::SETCC_CARRY", SDTX86SetCC_C>;

def X86rdrand  : SDNode<"X86ISD::RDRAND",   SDTX86rdrand,
                        [SDNPHasChain, SDNPSideEffect]>;

def X86rdseed  : SDNode<"X86ISD::RDSEED",   SDTX86rdrand,
                        [SDNPHasChain, SDNPSideEffect]>;

def X86rdpkru : SDNode<"X86ISD::RDPKRU",    SDTX86rdpkru,
                       [SDNPHasChain, SDNPSideEffect]>;
def X86wrpkru : SDNode<"X86ISD::WRPKRU",    SDTX86wrpkru,
                       [SDNPHasChain, SDNPSideEffect]>;

def X86cas : SDNode<"X86ISD::LCMPXCHG_DAG", SDTX86cas,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86cas8 : SDNode<"X86ISD::LCMPXCHG8_DAG", SDTX86cas8pair,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;
def X86cas16 : SDNode<"X86ISD::LCMPXCHG16_DAG", SDTX86cas16pair,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad, SDNPMemOperand]>;

def X86retflag : SDNode<"X86ISD::RET_FLAG", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def X86iret : SDNode<"X86ISD::IRET", SDTX86Ret,
                        [SDNPHasChain, SDNPOptInGlue]>;

def X86vastart_save_xmm_regs :
                 SDNode<"X86ISD::VASTART_SAVE_XMM_REGS",
                        SDT_X86VASTART_SAVE_XMM_REGS,
                        [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPVariadic]>;
def X86vaarg64 :
                 SDNode<"X86ISD::VAARG_64", SDT_X86VAARG,
                        [SDNPHasChain, SDNPMayLoad, SDNPMayStore,
                         SDNPMemOperand]>;
def X86vaargx32 :
                 SDNode<"X86ISD::VAARG_X32", SDT_X86VAARG,
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

def X86call_rvmarker  : SDNode<"X86ISD::CALL_RVMARKER",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                         SDNPVariadic]>;


def X86NoTrackCall : SDNode<"X86ISD::NT_CALL", SDT_X86Call,
                            [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue,
                             SDNPVariadic]>;
def X86NoTrackBrind : SDNode<"X86ISD::NT_BRIND", SDT_X86NtBrind,
                             [SDNPHasChain]>;

def X86rep_stos: SDNode<"X86ISD::REP_STOS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore]>;
def X86rep_movs: SDNode<"X86ISD::REP_MOVS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInGlue, SDNPOutGlue, SDNPMayStore,
                         SDNPMayLoad]>;

def X86Wrapper    : SDNode<"X86ISD::Wrapper",     SDTX86Wrapper>;
def X86WrapperRIP : SDNode<"X86ISD::WrapperRIP",  SDTX86Wrapper>;

def X86RecoverFrameAlloc : SDNode<"ISD::LOCAL_RECOVER",
                                  SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                                       SDTCisInt<1>]>>;

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
def X86eh_sjlj_setup_dispatch : SDNode<"X86ISD::EH_SJLJ_SETUP_DISPATCH",
                                       SDTypeProfile<0, 0, []>,
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

def X86or_flag   : SDNode<"X86ISD::OR",   SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86xor_flag  : SDNode<"X86ISD::XOR",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;
def X86and_flag  : SDNode<"X86ISD::AND",  SDTBinaryArithWithFlags,
                          [SDNPCommutative]>;

def X86lock_add  : SDNode<"X86ISD::LADD",  SDTLockBinaryArithWithFlags,
                          [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                           SDNPMemOperand]>;
def X86lock_sub  : SDNode<"X86ISD::LSUB",  SDTLockBinaryArithWithFlags,
                          [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                           SDNPMemOperand]>;
def X86lock_or  : SDNode<"X86ISD::LOR",  SDTLockBinaryArithWithFlags,
                         [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                          SDNPMemOperand]>;
def X86lock_xor  : SDNode<"X86ISD::LXOR",  SDTLockBinaryArithWithFlags,
                          [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                           SDNPMemOperand]>;
def X86lock_and  : SDNode<"X86ISD::LAND",  SDTLockBinaryArithWithFlags,
                          [SDNPHasChain, SDNPMayStore, SDNPMayLoad,
                           SDNPMemOperand]>;

def X86bextr  : SDNode<"X86ISD::BEXTR",  SDTIntBinOp>;
def X86bextri : SDNode<"X86ISD::BEXTRI", SDTIntBinOp>;

def X86bzhi   : SDNode<"X86ISD::BZHI",   SDTIntBinOp>;

def X86pdep   : SDNode<"X86ISD::PDEP",   SDTIntBinOp>;
def X86pext   : SDNode<"X86ISD::PEXT",   SDTIntBinOp>;

def X86mul_imm : SDNode<"X86ISD::MUL_IMM", SDTIntBinOp>;

def X86DynAlloca : SDNode<"X86ISD::DYN_ALLOCA", SDT_X86DYN_ALLOCA,
                          [SDNPHasChain, SDNPOutGlue]>;

def X86SegAlloca : SDNode<"X86ISD::SEG_ALLOCA", SDT_X86SEG_ALLOCA,
                          [SDNPHasChain]>;

def X86ProbedAlloca : SDNode<"X86ISD::PROBED_ALLOCA", SDT_X86PROBED_ALLOCA,
                          [SDNPHasChain]>;

def X86TLSCall : SDNode<"X86ISD::TLSCALL", SDT_X86TLSCALL,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def X86lwpins : SDNode<"X86ISD::LWPINS",
                       SDTypeProfile<1, 3, [SDTCisVT<0, i32>, SDTCisInt<1>,
                                            SDTCisVT<2, i32>, SDTCisVT<3, i32>]>,
                       [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPSideEffect]>;

def X86umwait : SDNode<"X86ISD::UMWAIT",
                       SDTypeProfile<1, 3, [SDTCisVT<0, i32>, SDTCisInt<1>,
                                            SDTCisVT<2, i32>, SDTCisVT<3, i32>]>,
                       [SDNPHasChain, SDNPSideEffect]>;

def X86tpause : SDNode<"X86ISD::TPAUSE",
                       SDTypeProfile<1, 3, [SDTCisVT<0, i32>, SDTCisInt<1>,
                                            SDTCisVT<2, i32>, SDTCisVT<3, i32>]>,
                       [SDNPHasChain, SDNPSideEffect]>;

def X86enqcmd : SDNode<"X86ISD::ENQCMD", SDT_X86ENQCMD,
                       [SDNPHasChain, SDNPSideEffect]>;
def X86enqcmds : SDNode<"X86ISD::ENQCMDS", SDT_X86ENQCMD,
                       [SDNPHasChain, SDNPSideEffect]>;
def X86testui : SDNode<"X86ISD::TESTUI",
                       SDTypeProfile<1, 0, [SDTCisVT<0, i32>]>,
                       [SDNPHasChain, SDNPSideEffect]>;

def X86aesenc128kl : SDNode<"X86ISD::AESENC128KL", SDT_X86AESENCDECKL,
                            [SDNPHasChain, SDNPMayLoad, SDNPSideEffect,
                             SDNPMemOperand]>;
def X86aesdec128kl : SDNode<"X86ISD::AESDEC128KL", SDT_X86AESENCDECKL,
                            [SDNPHasChain, SDNPMayLoad, SDNPSideEffect,
                             SDNPMemOperand]>;
def X86aesenc256kl : SDNode<"X86ISD::AESENC256KL", SDT_X86AESENCDECKL,
                            [SDNPHasChain, SDNPMayLoad, SDNPSideEffect,
                             SDNPMemOperand]>;
def X86aesdec256kl : SDNode<"X86ISD::AESDEC256KL", SDT_X86AESENCDECKL,
                            [SDNPHasChain, SDNPMayLoad, SDNPSideEffect,
                             SDNPMemOperand]>;

def X86cmpccxadd : SDNode<"X86ISD::CMPCCXADD", SDTX86Cmpccxadd,
                          [SDNPHasChain, SDNPMayLoad, SDNPMayStore,
                           SDNPMemOperand]>;

//===----------------------------------------------------------------------===//
// X86 Operand Definitions.
//

// A version of ptr_rc which excludes SP, ESP, and RSP. This is used for
// the index operand of an address, to conform to x86 encoding restrictions.
def ptr_rc_nosp : PointerLikeRegClass<1>;

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
def X86MemAsmOperand : AsmOperandClass {
 let Name = "Mem";
}
let RenderMethod = "addMemOperands", SuperClasses = [X86MemAsmOperand] in {
  def X86Mem8AsmOperand   : AsmOperandClass { let Name = "Mem8"; }
  def X86Mem16AsmOperand  : AsmOperandClass { let Name = "Mem16"; }
  def X86Mem32AsmOperand  : AsmOperandClass { let Name = "Mem32"; }
  def X86Mem64AsmOperand  : AsmOperandClass { let Name = "Mem64"; }
  def X86Mem80AsmOperand  : AsmOperandClass { let Name = "Mem80"; }
  def X86Mem128AsmOperand : AsmOperandClass { let Name = "Mem128"; }
  def X86Mem256AsmOperand : AsmOperandClass { let Name = "Mem256"; }
  def X86Mem512AsmOperand : AsmOperandClass { let Name = "Mem512"; }
  // Gather mem operands
  def X86Mem64_RC128Operand  : AsmOperandClass { let Name = "Mem64_RC128"; }
  def X86Mem128_RC128Operand : AsmOperandClass { let Name = "Mem128_RC128"; }
  def X86Mem256_RC128Operand : AsmOperandClass { let Name = "Mem256_RC128"; }
  def X86Mem128_RC256Operand : AsmOperandClass { let Name = "Mem128_RC256"; }
  def X86Mem256_RC256Operand : AsmOperandClass { let Name = "Mem256_RC256"; }

  def X86Mem64_RC128XOperand  : AsmOperandClass { let Name = "Mem64_RC128X"; }
  def X86Mem128_RC128XOperand : AsmOperandClass { let Name = "Mem128_RC128X"; }
  def X86Mem256_RC128XOperand : AsmOperandClass { let Name = "Mem256_RC128X"; }
  def X86Mem128_RC256XOperand : AsmOperandClass { let Name = "Mem128_RC256X"; }
  def X86Mem256_RC256XOperand : AsmOperandClass { let Name = "Mem256_RC256X"; }
  def X86Mem512_RC256XOperand : AsmOperandClass { let Name = "Mem512_RC256X"; }
  def X86Mem256_RC512Operand  : AsmOperandClass { let Name = "Mem256_RC512"; }
  def X86Mem512_RC512Operand  : AsmOperandClass { let Name = "Mem512_RC512"; }

  def X86SibMemOperand : AsmOperandClass { let Name = "SibMem"; }
}

def X86AbsMemAsmOperand : AsmOperandClass {
  let Name = "AbsMem";
  let SuperClasses = [X86MemAsmOperand];
}

class X86MemOperand<string printMethod,
                    AsmOperandClass parserMatchClass = X86MemAsmOperand,
                    int size = 0> : Operand<iPTR> {
  let PrintMethod = printMethod;
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc_nosp, i32imm, SEGMENT_REG);
  let ParserMatchClass = parserMatchClass;
  let OperandType = "OPERAND_MEMORY";
  int Size = size;
}

// Gather mem operands
class X86VMemOperand<RegisterClass RC, string printMethod,
                     AsmOperandClass parserMatchClass, int size = 0>
    : X86MemOperand<printMethod, parserMatchClass, size> {
  let MIOperandInfo = (ops ptr_rc, i8imm, RC, i32imm, SEGMENT_REG);
}

def anymem : X86MemOperand<"printMemReference">;
def X86any_fcmp : PatFrags<(ops node:$lhs, node:$rhs),
                          [(X86strict_fcmp node:$lhs, node:$rhs),
                           (X86fcmp node:$lhs, node:$rhs)]>;

// FIXME: Right now we allow any size during parsing, but we might want to
// restrict to only unsized memory.
def opaquemem : X86MemOperand<"printMemReference">;

def sibmem: X86MemOperand<"printMemReference", X86SibMemOperand>;

def i8mem   : X86MemOperand<"printbytemem",   X86Mem8AsmOperand, 8>;
def i16mem  : X86MemOperand<"printwordmem",  X86Mem16AsmOperand, 16>;
def i32mem  : X86MemOperand<"printdwordmem",  X86Mem32AsmOperand, 32>;
def i64mem  : X86MemOperand<"printqwordmem",  X86Mem64AsmOperand, 64>;
def i128mem : X86MemOperand<"printxmmwordmem", X86Mem128AsmOperand, 128>;
def i256mem : X86MemOperand<"printymmwordmem", X86Mem256AsmOperand, 256>;
def i512mem : X86MemOperand<"printzmmwordmem", X86Mem512AsmOperand, 512>;
def f16mem  : X86MemOperand<"printwordmem",   X86Mem16AsmOperand, 16>;
def f32mem  : X86MemOperand<"printdwordmem",  X86Mem32AsmOperand, 32>;
def f64mem  : X86MemOperand<"printqwordmem",  X86Mem64AsmOperand, 64>;
def f80mem  : X86MemOperand<"printtbytemem",  X86Mem80AsmOperand, 80>;
def f128mem : X86MemOperand<"printxmmwordmem", X86Mem128AsmOperand, 128>;
def f256mem : X86MemOperand<"printymmwordmem", X86Mem256AsmOperand, 256>;
def f512mem : X86MemOperand<"printzmmwordmem", X86Mem512AsmOperand, 512>;

// Gather mem operands
def vx64mem  : X86VMemOperand<VR128,  "printqwordmem",  X86Mem64_RC128Operand, 64>;
def vx128mem : X86VMemOperand<VR128,  "printxmmwordmem", X86Mem128_RC128Operand, 128>;
def vx256mem : X86VMemOperand<VR128,  "printymmwordmem", X86Mem256_RC128Operand, 256>;
def vy128mem : X86VMemOperand<VR256,  "printxmmwordmem", X86Mem128_RC256Operand, 128>;
def vy256mem : X86VMemOperand<VR256,  "printymmwordmem", X86Mem256_RC256Operand, 256>;

def vx64xmem  : X86VMemOperand<VR128X, "printqwordmem",  X86Mem64_RC128XOperand, 64>;
def vx128xmem : X86VMemOperand<VR128X, "printxmmwordmem", X86Mem128_RC128XOperand, 128>;
def vx256xmem : X86VMemOperand<VR128X, "printymmwordmem", X86Mem256_RC128XOperand, 256>;
def vy128xmem : X86VMemOperand<VR256X, "printxmmwordmem", X86Mem128_RC256XOperand, 128>;
def vy256xmem : X86VMemOperand<VR256X, "printymmwordmem", X86Mem256_RC256XOperand, 256>;
def vy512xmem : X86VMemOperand<VR256X, "printzmmwordmem", X86Mem512_RC256XOperand, 512>;
def vz256mem  : X86VMemOperand<VR512,  "printymmwordmem", X86Mem256_RC512Operand, 256>;
def vz512mem  : X86VMemOperand<VR512,  "printzmmwordmem", X86Mem512_RC512Operand, 512>;

// A version of i8mem for use on x86-64 and x32 that uses a NOREX GPR instead
// of a plain GPR, so that it doesn't potentially require a REX prefix.
def ptr_rc_norex : PointerLikeRegClass<2>;
def ptr_rc_norex_nosp : PointerLikeRegClass<3>;

def i8mem_NOREX : X86MemOperand<"printbytemem", X86Mem8AsmOperand, 8> {
  let MIOperandInfo = (ops ptr_rc_norex, i8imm, ptr_rc_norex_nosp, i32imm,
                       SEGMENT_REG);
}

// GPRs available for tailcall.
// It represents GR32_TC, GR64_TC or GR64_TCW64.
def ptr_rc_tailcall : PointerLikeRegClass<4>;

// Special i32mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i32mem_TC : X86MemOperand<"printdwordmem", X86Mem32AsmOperand, 32> {
  let MIOperandInfo = (ops ptr_rc_tailcall, i8imm, ptr_rc_tailcall,
                       i32imm, SEGMENT_REG);
}

// Special i64mem for addresses of load folding tail calls. These are not
// allowed to use callee-saved registers since they must be scheduled
// after callee-saved register are popped.
def i64mem_TC : X86MemOperand<"printqwordmem", X86Mem64AsmOperand, 64> {
  let MIOperandInfo = (ops ptr_rc_tailcall, i8imm,
                       ptr_rc_tailcall, i32imm, SEGMENT_REG);
}

// Special parser to detect 16-bit mode to select 16-bit displacement.
def X86AbsMem16AsmOperand : AsmOperandClass {
  let Name = "AbsMem16";
  let RenderMethod = "addAbsMemOperands";
  let SuperClasses = [X86AbsMemAsmOperand];
}

// Branch targets print as pc-relative values.
class BranchTargetOperand<ValueType ty> : Operand<ty> {
  let OperandType = "OPERAND_PCREL";
  let PrintMethod = "printPCRelImm";
  let ParserMatchClass = X86AbsMemAsmOperand;
}

def i32imm_brtarget : BranchTargetOperand<i32>;
def i16imm_brtarget : BranchTargetOperand<i16>;

// 64-bits but only 32 bits are significant, and those bits are treated as being
// pc relative.
def i64i32imm_brtarget : BranchTargetOperand<i64>;

def brtarget : BranchTargetOperand<OtherVT>;
def brtarget8 : BranchTargetOperand<OtherVT>;
def brtarget16 : BranchTargetOperand<OtherVT> {
  let ParserMatchClass = X86AbsMem16AsmOperand;
}
def brtarget32 : BranchTargetOperand<OtherVT>;

let RenderMethod = "addSrcIdxOperands" in {
  def X86SrcIdx8Operand : AsmOperandClass {
    let Name = "SrcIdx8";
    let SuperClasses = [X86Mem8AsmOperand];
  }
  def X86SrcIdx16Operand : AsmOperandClass {
    let Name = "SrcIdx16";
    let SuperClasses = [X86Mem16AsmOperand];
  }
  def X86SrcIdx32Operand : AsmOperandClass {
    let Name = "SrcIdx32";
    let SuperClasses = [X86Mem32AsmOperand];
  }
  def X86SrcIdx64Operand : AsmOperandClass {
    let Name = "SrcIdx64";
    let SuperClasses = [X86Mem64AsmOperand];
  }
} // RenderMethod = "addSrcIdxOperands"

let RenderMethod = "addDstIdxOperands" in {
 def X86DstIdx8Operand : AsmOperandClass {
   let Name = "DstIdx8";
   let SuperClasses = [X86Mem8AsmOperand];
 }
 def X86DstIdx16Operand : AsmOperandClass {
   let Name = "DstIdx16";
   let SuperClasses = [X86Mem16AsmOperand];
 }
 def X86DstIdx32Operand : AsmOperandClass {
   let Name = "DstIdx32";
   let SuperClasses = [X86Mem32AsmOperand];
 }
 def X86DstIdx64Operand : AsmOperandClass {
   let Name = "DstIdx64";
   let SuperClasses = [X86Mem64AsmOperand];
 }
} // RenderMethod = "addDstIdxOperands"

let RenderMethod = "addMemOffsOperands" in {
  def X86MemOffs16_8AsmOperand : AsmOperandClass {
    let Name = "MemOffs16_8";
    let SuperClasses = [X86Mem8AsmOperand];
  }
  def X86MemOffs16_16AsmOperand : AsmOperandClass {
    let Name = "MemOffs16_16";
    let SuperClasses = [X86Mem16AsmOperand];
  }
  def X86MemOffs16_32AsmOperand : AsmOperandClass {
    let Name = "MemOffs16_32";
    let SuperClasses = [X86Mem32AsmOperand];
  }
  def X86MemOffs32_8AsmOperand : AsmOperandClass {
    let Name = "MemOffs32_8";
    let SuperClasses = [X86Mem8AsmOperand];
  }
  def X86MemOffs32_16AsmOperand : AsmOperandClass {
    let Name = "MemOffs32_16";
    let SuperClasses = [X86Mem16AsmOperand];
  }
  def X86MemOffs32_32AsmOperand : AsmOperandClass {
    let Name = "MemOffs32_32";
    let SuperClasses = [X86Mem32AsmOperand];
  }
  def X86MemOffs32_64AsmOperand : AsmOperandClass {
    let Name = "MemOffs32_64";
    let SuperClasses = [X86Mem64AsmOperand];
  }
  def X86MemOffs64_8AsmOperand : AsmOperandClass {
    let Name = "MemOffs64_8";
    let SuperClasses = [X86Mem8AsmOperand];
  }
  def X86MemOffs64_16AsmOperand : AsmOperandClass {
    let Name = "MemOffs64_16";
    let SuperClasses = [X86Mem16AsmOperand];
  }
  def X86MemOffs64_32AsmOperand : AsmOperandClass {
    let Name = "MemOffs64_32";
    let SuperClasses = [X86Mem32AsmOperand];
  }
  def X86MemOffs64_64AsmOperand : AsmOperandClass {
    let Name = "MemOffs64_64";
    let SuperClasses = [X86Mem64AsmOperand];
  }
} // RenderMethod = "addMemOffsOperands"

class X86SrcIdxOperand<string printMethod, AsmOperandClass parserMatchClass>
    : X86MemOperand<printMethod, parserMatchClass> {
  let MIOperandInfo = (ops ptr_rc, SEGMENT_REG);
}

class X86DstIdxOperand<string printMethod, AsmOperandClass parserMatchClass>
    : X86MemOperand<printMethod, parserMatchClass> {
  let MIOperandInfo = (ops ptr_rc);
}

def srcidx8  : X86SrcIdxOperand<"printSrcIdx8",  X86SrcIdx8Operand>;
def srcidx16 : X86SrcIdxOperand<"printSrcIdx16", X86SrcIdx16Operand>;
def srcidx32 : X86SrcIdxOperand<"printSrcIdx32", X86SrcIdx32Operand>;
def srcidx64 : X86SrcIdxOperand<"printSrcIdx64", X86SrcIdx64Operand>;
def dstidx8  : X86DstIdxOperand<"printDstIdx8",  X86DstIdx8Operand>;
def dstidx16 : X86DstIdxOperand<"printDstIdx16", X86DstIdx16Operand>;
def dstidx32 : X86DstIdxOperand<"printDstIdx32", X86DstIdx32Operand>;
def dstidx64 : X86DstIdxOperand<"printDstIdx64", X86DstIdx64Operand>;

class X86MemOffsOperand<Operand immOperand, string printMethod,
                        AsmOperandClass parserMatchClass>
    : X86MemOperand<printMethod, parserMatchClass> {
  let MIOperandInfo = (ops immOperand, SEGMENT_REG);
}

def offset16_8  : X86MemOffsOperand<i16imm, "printMemOffs8",
                                    X86MemOffs16_8AsmOperand>;
def offset16_16 : X86MemOffsOperand<i16imm, "printMemOffs16",
                                    X86MemOffs16_16AsmOperand>;
def offset16_32 : X86MemOffsOperand<i16imm, "printMemOffs32",
                                    X86MemOffs16_32AsmOperand>;
def offset32_8  : X86MemOffsOperand<i32imm, "printMemOffs8",
                                    X86MemOffs32_8AsmOperand>;
def offset32_16 : X86MemOffsOperand<i32imm, "printMemOffs16",
                                    X86MemOffs32_16AsmOperand>;
def offset32_32 : X86MemOffsOperand<i32imm, "printMemOffs32",
                                    X86MemOffs32_32AsmOperand>;
def offset32_64 : X86MemOffsOperand<i32imm, "printMemOffs64",
                                    X86MemOffs32_64AsmOperand>;
def offset64_8  : X86MemOffsOperand<i64imm, "printMemOffs8",
                                    X86MemOffs64_8AsmOperand>;
def offset64_16 : X86MemOffsOperand<i64imm, "printMemOffs16",
                                    X86MemOffs64_16AsmOperand>;
def offset64_32 : X86MemOffsOperand<i64imm, "printMemOffs32",
                                    X86MemOffs64_32AsmOperand>;
def offset64_64 : X86MemOffsOperand<i64imm, "printMemOffs64",
                                    X86MemOffs64_64AsmOperand>;

def ccode : Operand<i8> {
  let PrintMethod = "printCondCode";
  let OperandNamespace = "X86";
  let OperandType = "OPERAND_COND_CODE";
}

class ImmSExtAsmOperandClass : AsmOperandClass {
  let SuperClasses = [ImmAsmOperand];
  let RenderMethod = "addImmOperands";
}

def X86GR32orGR64AsmOperand : AsmOperandClass {
  let Name = "GR32orGR64";
}
def GR32orGR64 : RegisterOperand<GR32> {
  let ParserMatchClass = X86GR32orGR64AsmOperand;
}

def X86GR16orGR32orGR64AsmOperand : AsmOperandClass {
  let Name = "GR16orGR32orGR64";
}
def GR16orGR32orGR64 : RegisterOperand<GR16> {
  let ParserMatchClass = X86GR16orGR32orGR64AsmOperand;
}

def AVX512RCOperand : AsmOperandClass {
  let Name = "AVX512RC";
}
def AVX512RC : Operand<i32> {
  let PrintMethod = "printRoundingControl";
  let OperandNamespace = "X86";
  let OperandType = "OPERAND_ROUNDING_CONTROL";
  let ParserMatchClass = AVX512RCOperand;
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

// [0, 0x0000007F]                                            |
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmSExti64i8AsmOperand : ImmSExtAsmOperandClass {
  let Name = "ImmSExti64i8";
  let SuperClasses = [ImmSExti16i8AsmOperand, ImmSExti32i8AsmOperand,
                      ImmSExti64i32AsmOperand];
}

// 4-bit immediate used by some XOP instructions
// [0, 0xF]
def ImmUnsignedi4AsmOperand : AsmOperandClass {
  let Name = "ImmUnsignedi4";
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "InvalidImmUnsignedi4";
}

// Unsigned immediate used by SSE/AVX instructions
// [0, 0xFF]
//   [0xFFFFFFFFFFFFFF80, 0xFFFFFFFFFFFFFFFF]
def ImmUnsignedi8AsmOperand : AsmOperandClass {
  let Name = "ImmUnsignedi8";
  let RenderMethod = "addImmOperands";
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

// 64-bits but only 32 bits are significant.
def i64i32imm  : Operand<i64> {
  let ParserMatchClass = ImmSExti64i32AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 64-bits but only 8 bits are significant.
def i64i8imm   : Operand<i64> {
  let ParserMatchClass = ImmSExti64i8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Unsigned 4-bit immediate used by some XOP instructions.
def u4imm : Operand<i8> {
  let PrintMethod = "printU8Imm";
  let ParserMatchClass = ImmUnsignedi4AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// Unsigned 8-bit immediate used by SSE/AVX instructions.
def u8imm : Operand<i8> {
  let PrintMethod = "printU8Imm";
  let ParserMatchClass = ImmUnsignedi8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 16-bit immediate but only 8-bits are significant and they are unsigned.
// Used by BT instructions.
def i16u8imm : Operand<i16> {
  let PrintMethod = "printU8Imm";
  let ParserMatchClass = ImmUnsignedi8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 32-bit immediate but only 8-bits are significant and they are unsigned.
// Used by some SSE/AVX instructions that use intrinsics.
def i32u8imm : Operand<i32> {
  let PrintMethod = "printU8Imm";
  let ParserMatchClass = ImmUnsignedi8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

// 64-bit immediate but only 8-bits are significant and they are unsigned.
// Used by BT instructions.
def i64u8imm : Operand<i64> {
  let PrintMethod = "printU8Imm";
  let ParserMatchClass = ImmUnsignedi8AsmOperand;
  let OperandType = "OPERAND_IMMEDIATE";
}

def lea64_32mem : Operand<i32> {
  let PrintMethod = "printMemReference";
  let MIOperandInfo = (ops GR64, i8imm, GR64_NOSP, i32imm, SEGMENT_REG);
  let ParserMatchClass = X86MemAsmOperand;
}

// Memory operands that use 64-bit pointers in both ILP32 and LP64.
def lea64mem : Operand<i64> {
  let PrintMethod = "printMemReference";
  let MIOperandInfo = (ops GR64, i8imm, GR64_NOSP, i32imm, SEGMENT_REG);
  let ParserMatchClass = X86MemAsmOperand;
}

let RenderMethod = "addMaskPairOperands" in {
  def VK1PairAsmOperand : AsmOperandClass { let Name = "VK1Pair"; }
  def VK2PairAsmOperand : AsmOperandClass { let Name = "VK2Pair"; }
  def VK4PairAsmOperand : AsmOperandClass { let Name = "VK4Pair"; }
  def VK8PairAsmOperand : AsmOperandClass { let Name = "VK8Pair"; }
  def VK16PairAsmOperand : AsmOperandClass { let Name = "VK16Pair"; }
}

def VK1Pair : RegisterOperand<VK1PAIR, "printVKPair"> {
  let ParserMatchClass = VK1PairAsmOperand;
}

def VK2Pair : RegisterOperand<VK2PAIR, "printVKPair"> {
  let ParserMatchClass = VK2PairAsmOperand;
}

def VK4Pair : RegisterOperand<VK4PAIR, "printVKPair"> {
  let ParserMatchClass = VK4PairAsmOperand;
}

def VK8Pair : RegisterOperand<VK8PAIR, "printVKPair"> {
  let ParserMatchClass = VK8PairAsmOperand;
}

def VK16Pair : RegisterOperand<VK16PAIR, "printVKPair"> {
  let ParserMatchClass = VK16PairAsmOperand;
}

//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86-specific addressing mode.
def addr      : ComplexPattern<iPTR, 5, "selectAddr", [], [SDNPWantParent]>;
def lea32addr : ComplexPattern<i32, 5, "selectLEAAddr",
                               [add, sub, mul, X86mul_imm, shl, or, xor, frameindex],
                               []>;
// In 64-bit mode 32-bit LEAs can use RIP-relative addressing.
def lea64_32addr : ComplexPattern<i32, 5, "selectLEA64_32Addr",
                                  [add, sub, mul, X86mul_imm, shl, or, xor,
                                   frameindex, X86WrapperRIP],
                                  []>;

def tls32addr : ComplexPattern<i32, 5, "selectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def tls32baseaddr : ComplexPattern<i32, 5, "selectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def lea64addr : ComplexPattern<i64, 5, "selectLEAAddr",
                        [add, sub, mul, X86mul_imm, shl, or, xor, frameindex,
                         X86WrapperRIP], []>;

def tls64addr : ComplexPattern<i64, 5, "selectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def tls64baseaddr : ComplexPattern<i64, 5, "selectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

def vectoraddr : ComplexPattern<iPTR, 5, "selectVectorAddr", [],[SDNPWantParent]>;

// A relocatable immediate is an operand that can be relocated by the linker to
// an immediate, such as a regular symbol in non-PIC code.
def relocImm : ComplexPattern<iAny, 1, "selectRelocImm",
                              [X86Wrapper], [], 0>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def TruePredicate : Predicate<"true">;

def HasCMOV      : Predicate<"Subtarget->canUseCMOV()">;
def NoCMOV       : Predicate<"!Subtarget->canUseCMOV()">;

def HasNOPL      : Predicate<"Subtarget->hasNOPL()">;
def HasMMX       : Predicate<"Subtarget->hasMMX()">;
def Has3DNow     : Predicate<"Subtarget->hasThreeDNow()">;
def Has3DNowA    : Predicate<"Subtarget->hasThreeDNowA()">;
def HasSSE1      : Predicate<"Subtarget->hasSSE1()">;
def UseSSE1      : Predicate<"Subtarget->hasSSE1() && !Subtarget->hasAVX()">;
def HasSSE2      : Predicate<"Subtarget->hasSSE2()">;
def UseSSE2      : Predicate<"Subtarget->hasSSE2() && !Subtarget->hasAVX()">;
def HasSSE3      : Predicate<"Subtarget->hasSSE3()">;
def UseSSE3      : Predicate<"Subtarget->hasSSE3() && !Subtarget->hasAVX()">;
def HasSSSE3     : Predicate<"Subtarget->hasSSSE3()">;
def UseSSSE3     : Predicate<"Subtarget->hasSSSE3() && !Subtarget->hasAVX()">;
def HasSSE41     : Predicate<"Subtarget->hasSSE41()">;
def NoSSE41      : Predicate<"!Subtarget->hasSSE41()">;
def UseSSE41     : Predicate<"Subtarget->hasSSE41() && !Subtarget->hasAVX()">;
def HasSSE42     : Predicate<"Subtarget->hasSSE42()">;
def UseSSE42     : Predicate<"Subtarget->hasSSE42() && !Subtarget->hasAVX()">;
def HasSSE4A     : Predicate<"Subtarget->hasSSE4A()">;
def NoAVX        : Predicate<"!Subtarget->hasAVX()">;
def HasAVX       : Predicate<"Subtarget->hasAVX()">;
def HasAVX2      : Predicate<"Subtarget->hasAVX2()">;
def HasAVX1Only  : Predicate<"Subtarget->hasAVX() && !Subtarget->hasAVX2()">;
def HasAVX512    : Predicate<"Subtarget->hasAVX512()">;
def UseAVX       : Predicate<"Subtarget->hasAVX() && !Subtarget->hasAVX512()">;
def UseAVX2      : Predicate<"Subtarget->hasAVX2() && !Subtarget->hasAVX512()">;
def NoAVX512     : Predicate<"!Subtarget->hasAVX512()">;
def HasCDI       : Predicate<"Subtarget->hasCDI()">;
def HasVPOPCNTDQ : Predicate<"Subtarget->hasVPOPCNTDQ()">;
def HasPFI       : Predicate<"Subtarget->hasPFI()">;
def HasERI       : Predicate<"Subtarget->hasERI()">;
def HasDQI       : Predicate<"Subtarget->hasDQI()">;
def NoDQI        : Predicate<"!Subtarget->hasDQI()">;
def HasBWI       : Predicate<"Subtarget->hasBWI()">;
def NoBWI        : Predicate<"!Subtarget->hasBWI()">;
def HasVLX       : Predicate<"Subtarget->hasVLX()">;
def NoVLX        : Predicate<"!Subtarget->hasVLX()">;
def NoVLX_Or_NoBWI : Predicate<"!Subtarget->hasVLX() || !Subtarget->hasBWI()">;
def NoVLX_Or_NoDQI : Predicate<"!Subtarget->hasVLX() || !Subtarget->hasDQI()">;
def HasPKU       : Predicate<"Subtarget->hasPKU()">;
def HasVNNI      : Predicate<"Subtarget->hasVNNI()">;
def HasVP2INTERSECT : Predicate<"Subtarget->hasVP2INTERSECT()">;
def HasBF16      : Predicate<"Subtarget->hasBF16()">;
def HasFP16      : Predicate<"Subtarget->hasFP16()">;
def HasAVXVNNIINT8 : Predicate<"Subtarget->hasAVXVNNIINT8()">;
def HasAVXVNNI : Predicate <"Subtarget->hasAVXVNNI()">;
def NoVLX_Or_NoVNNI : Predicate<"!Subtarget->hasVLX() || !Subtarget->hasVNNI()">;

def HasBITALG    : Predicate<"Subtarget->hasBITALG()">;
def HasPOPCNT    : Predicate<"Subtarget->hasPOPCNT()">;
def HasAES       : Predicate<"Subtarget->hasAES()">;
def HasVAES      : Predicate<"Subtarget->hasVAES()">;
def NoVLX_Or_NoVAES : Predicate<"!Subtarget->hasVLX() || !Subtarget->hasVAES()">;
def HasFXSR      : Predicate<"Subtarget->hasFXSR()">;
def HasX87       : Predicate<"Subtarget->hasX87()">;
def HasXSAVE     : Predicate<"Subtarget->hasXSAVE()">;
def HasXSAVEOPT  : Predicate<"Subtarget->hasXSAVEOPT()">;
def HasXSAVEC    : Predicate<"Subtarget->hasXSAVEC()">;
def HasXSAVES    : Predicate<"Subtarget->hasXSAVES()">;
def HasPCLMUL    : Predicate<"Subtarget->hasPCLMUL()">;
def NoVLX_Or_NoVPCLMULQDQ :
                    Predicate<"!Subtarget->hasVLX() || !Subtarget->hasVPCLMULQDQ()">;
def HasVPCLMULQDQ : Predicate<"Subtarget->hasVPCLMULQDQ()">;
def HasGFNI      : Predicate<"Subtarget->hasGFNI()">;
def HasFMA       : Predicate<"Subtarget->hasFMA()">;
def HasFMA4      : Predicate<"Subtarget->hasFMA4()">;
def NoFMA4       : Predicate<"!Subtarget->hasFMA4()">;
def HasXOP       : Predicate<"Subtarget->hasXOP()">;
def HasTBM       : Predicate<"Subtarget->hasTBM()">;
def NoTBM        : Predicate<"!Subtarget->hasTBM()">;
def HasLWP       : Predicate<"Subtarget->hasLWP()">;
def HasMOVBE     : Predicate<"Subtarget->hasMOVBE()">;
def HasRDRAND    : Predicate<"Subtarget->hasRDRAND()">;
def HasF16C      : Predicate<"Subtarget->hasF16C()">;
def HasFSGSBase  : Predicate<"Subtarget->hasFSGSBase()">;
def HasLZCNT     : Predicate<"Subtarget->hasLZCNT()">;
def HasBMI       : Predicate<"Subtarget->hasBMI()">;
def HasBMI2      : Predicate<"Subtarget->hasBMI2()">;
def NoBMI2       : Predicate<"!Subtarget->hasBMI2()">;
def HasVBMI      : Predicate<"Subtarget->hasVBMI()">;
def HasVBMI2     : Predicate<"Subtarget->hasVBMI2()">;
def HasIFMA      : Predicate<"Subtarget->hasIFMA()">;
def HasAVXIFMA   : Predicate<"Subtarget->hasAVXIFMA()">;
def NoVLX_Or_NoIFMA : Predicate<"!Subtarget->hasVLX() || !Subtarget->hasIFMA()">;
def HasRTM       : Predicate<"Subtarget->hasRTM()">;
def HasADX       : Predicate<"Subtarget->hasADX()">;
def HasSHA       : Predicate<"Subtarget->hasSHA()">;
def HasSGX       : Predicate<"Subtarget->hasSGX()">;
def HasRDSEED    : Predicate<"Subtarget->hasRDSEED()">;
def HasSSEPrefetch : Predicate<"Subtarget->hasSSEPrefetch()">;
def NoSSEPrefetch : Predicate<"!Subtarget->hasSSEPrefetch()">;
def HasPRFCHW    : Predicate<"Subtarget->hasPRFCHW()">;
def HasPREFETCHI : Predicate<"Subtarget->hasPREFETCHI()">;
def HasPrefetchW : Predicate<"Subtarget->hasPrefetchW()">;
def HasPREFETCHWT1 : Predicate<"Subtarget->hasPREFETCHWT1()">;
def HasLAHFSAHF  : Predicate<"Subtarget->hasLAHFSAHF()">;
def HasLAHFSAHF64 : Predicate<"Subtarget->hasLAHFSAHF64()">;
def HasMWAITX    : Predicate<"Subtarget->hasMWAITX()">;
def HasCLZERO    : Predicate<"Subtarget->hasCLZERO()">;
def HasCLDEMOTE  : Predicate<"Subtarget->hasCLDEMOTE()">;
def HasMOVDIRI   : Predicate<"Subtarget->hasMOVDIRI()">;
def HasMOVDIR64B : Predicate<"Subtarget->hasMOVDIR64B()">;
def HasPTWRITE   : Predicate<"Subtarget->hasPTWRITE()">;
def FPStackf32   : Predicate<"!Subtarget->hasSSE1()">;
def FPStackf64   : Predicate<"!Subtarget->hasSSE2()">;
def HasSHSTK     : Predicate<"Subtarget->hasSHSTK()">;
def HasCLFLUSH   : Predicate<"Subtarget->hasCLFLUSH()">;
def HasCLFLUSHOPT : Predicate<"Subtarget->hasCLFLUSHOPT()">;
def HasCLWB      : Predicate<"Subtarget->hasCLWB()">;
def HasWBNOINVD  : Predicate<"Subtarget->hasWBNOINVD()">;
def HasRDPID     : Predicate<"Subtarget->hasRDPID()">;
def HasRDPRU     : Predicate<"Subtarget->hasRDPRU()">;
def HasWAITPKG   : Predicate<"Subtarget->hasWAITPKG()">;
def HasINVPCID   : Predicate<"Subtarget->hasINVPCID()">;
def HasCX8       : Predicate<"Subtarget->hasCX8()">;
def HasCX16      : Predicate<"Subtarget->hasCX16()">;
def HasPCONFIG   : Predicate<"Subtarget->hasPCONFIG()">;
def HasENQCMD    : Predicate<"Subtarget->hasENQCMD()">;
def HasAMXFP16   : Predicate<"Subtarget->hasAMXFP16()">;
def HasCMPCCXADD : Predicate<"Subtarget->hasCMPCCXADD()">;
def HasAVXNECONVERT : Predicate<"Subtarget->hasAVXNECONVERT()">;
def HasKL        : Predicate<"Subtarget->hasKL()">;
def HasRAOINT    : Predicate<"Subtarget->hasRAOINT()">;
def HasWIDEKL    : Predicate<"Subtarget->hasWIDEKL()">;
def HasHRESET    : Predicate<"Subtarget->hasHRESET()">;
def HasSERIALIZE : Predicate<"Subtarget->hasSERIALIZE()">;
def HasTSXLDTRK  : Predicate<"Subtarget->hasTSXLDTRK()">;
def HasAMXTILE   : Predicate<"Subtarget->hasAMXTILE()">;
def HasAMXBF16   : Predicate<"Subtarget->hasAMXBF16()">;
def HasAMXINT8   : Predicate<"Subtarget->hasAMXINT8()">;
def HasUINTR     : Predicate<"Subtarget->hasUINTR()">;
def HasCRC32     : Predicate<"Subtarget->hasCRC32()">;

def HasX86_64    : Predicate<"Subtarget->hasX86_64()">;
def Not64BitMode : Predicate<"!Subtarget->is64Bit()">,
                             AssemblerPredicate<(all_of (not Is64Bit)), "Not 64-bit mode">;
def In64BitMode  : Predicate<"Subtarget->is64Bit()">,
                             AssemblerPredicate<(all_of Is64Bit), "64-bit mode">;
def IsLP64  : Predicate<"Subtarget->isTarget64BitLP64()">;
def NotLP64 : Predicate<"!Subtarget->isTarget64BitLP64()">;
def In16BitMode  : Predicate<"Subtarget->is16Bit()">,
                             AssemblerPredicate<(all_of Is16Bit), "16-bit mode">;
def Not16BitMode : Predicate<"!Subtarget->is16Bit()">,
                             AssemblerPredicate<(all_of (not Is16Bit)), "Not 16-bit mode">;
def In32BitMode  : Predicate<"Subtarget->is32Bit()">,
                             AssemblerPredicate<(all_of Is32Bit), "32-bit mode">;
def IsWin64      : Predicate<"Subtarget->isTargetWin64()">;
def NotWin64     : Predicate<"!Subtarget->isTargetWin64()">;
def NotWin64WithoutFP : Predicate<"!Subtarget->isTargetWin64() ||"
                                  "Subtarget->getFrameLowering()->hasFP(*MF)"> {
  let RecomputePerFunction = 1;
}
def IsPS         : Predicate<"Subtarget->isTargetPS()">;
def NotPS        : Predicate<"!Subtarget->isTargetPS()">;
def IsNaCl       : Predicate<"Subtarget->isTargetNaCl()">;
def NotNaCl      : Predicate<"!Subtarget->isTargetNaCl()">;
def SmallCode    : Predicate<"TM.getCodeModel() == CodeModel::Small">;
def KernelCode   : Predicate<"TM.getCodeModel() == CodeModel::Kernel">;
def NearData     : Predicate<"TM.getCodeModel() == CodeModel::Small ||"
                             "TM.getCodeModel() == CodeModel::Kernel">;
def IsNotPIC     : Predicate<"!TM.isPositionIndependent()">;

// We could compute these on a per-module basis but doing so requires accessing
// the Function object through the <Target>Subtarget and objections were raised
// to that (see post-commit review comments for r301750).
let RecomputePerFunction = 1 in {
  def OptForSize   : Predicate<"shouldOptForSize(MF)">;
  def OptForMinSize : Predicate<"MF->getFunction().hasMinSize()">;
  def OptForSpeed  : Predicate<"!shouldOptForSize(MF)">;
  def UseIncDec : Predicate<"!Subtarget->slowIncDec() || "
                            "shouldOptForSize(MF)">;
  def NoSSE41_Or_OptForSize : Predicate<"shouldOptForSize(MF) || "
                                        "!Subtarget->hasSSE41()">;
}

def CallImmAddr  : Predicate<"Subtarget->isLegalToCallImmediateAddr()">;
def FavorMemIndirectCall  : Predicate<"!Subtarget->slowTwoMemOps()">;
def HasFastMem32 : Predicate<"!Subtarget->isUnalignedMem32Slow()">;
def HasFastLZCNT : Predicate<"Subtarget->hasFastLZCNT()">;
def HasFastSHLDRotate : Predicate<"Subtarget->hasFastSHLDRotate()">;
def HasERMSB : Predicate<"Subtarget->hasERMSB()">;
def HasFSRM : Predicate<"Subtarget->hasFSRM()">;
def HasMFence    : Predicate<"Subtarget->hasMFence()">;
def UseIndirectThunkCalls : Predicate<"Subtarget->useIndirectThunkCalls()">;
def NotUseIndirectThunkCalls : Predicate<"!Subtarget->useIndirectThunkCalls()">;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

//===----------------------------------------------------------------------===//
// Pattern fragments.
//

// X86 specific condition code. These correspond to CondCode in
// X86InstrInfo.h. They must be kept in synch.
def X86_COND_O   : PatLeaf<(i8 0)>;
def X86_COND_NO  : PatLeaf<(i8 1)>;
def X86_COND_B   : PatLeaf<(i8 2)>;  // alt. COND_C
def X86_COND_AE  : PatLeaf<(i8 3)>;  // alt. COND_NC
def X86_COND_E   : PatLeaf<(i8 4)>;  // alt. COND_Z
def X86_COND_NE  : PatLeaf<(i8 5)>;  // alt. COND_NZ
def X86_COND_BE  : PatLeaf<(i8 6)>;  // alt. COND_NA
def X86_COND_A   : PatLeaf<(i8 7)>;  // alt. COND_NBE
def X86_COND_S   : PatLeaf<(i8 8)>;
def X86_COND_NS  : PatLeaf<(i8 9)>;
def X86_COND_P   : PatLeaf<(i8 10)>; // alt. COND_PE
def X86_COND_NP  : PatLeaf<(i8 11)>; // alt. COND_PO
def X86_COND_L   : PatLeaf<(i8 12)>; // alt. COND_NGE
def X86_COND_GE  : PatLeaf<(i8 13)>; // alt. COND_NL
def X86_COND_LE  : PatLeaf<(i8 14)>; // alt. COND_NG
def X86_COND_G   : PatLeaf<(i8 15)>; // alt. COND_NLE

def i16immSExt8  : ImmLeaf<i16, [{ return isInt<8>(Imm); }]>;
def i32immSExt8  : ImmLeaf<i32, [{ return isInt<8>(Imm); }]>;
def i64immSExt8  : ImmLeaf<i64, [{ return isInt<8>(Imm); }]>;
def i64immSExt32 : ImmLeaf<i64, [{ return isInt<32>(Imm); }]>;
def i64timmSExt32 : TImmLeaf<i64, [{ return isInt<32>(Imm); }]>;

def i16relocImmSExt8 : PatLeaf<(i16 relocImm), [{
  return isSExtAbsoluteSymbolRef(8, N);
}]>;
def i32relocImmSExt8 : PatLeaf<(i32 relocImm), [{
  return isSExtAbsoluteSymbolRef(8, N);
}]>;
def i64relocImmSExt8 : PatLeaf<(i64 relocImm), [{
  return isSExtAbsoluteSymbolRef(8, N);
}]>;
def i64relocImmSExt32 : PatLeaf<(i64 relocImm), [{
  return isSExtAbsoluteSymbolRef(32, N);
}]>;

// If we have multiple users of an immediate, it's much smaller to reuse
// the register, rather than encode the immediate in every instruction.
// This has the risk of increasing register pressure from stretched live
// ranges, however, the immediates should be trivial to rematerialize by
// the RA in the event of high register pressure.
// TODO : This is currently enabled for stores and binary ops. There are more
// cases for which this can be enabled, though this catches the bulk of the
// issues.
// TODO2 : This should really also be enabled under O2, but there's currently
// an issue with RA where we don't pull the constants into their users
// when we rematerialize them. I'll follow-up on enabling O2 after we fix that
// issue.
// TODO3 : This is currently limited to single basic blocks (DAG creation
// pulls block immediates to the top and merges them if necessary).
// Eventually, it would be nice to allow ConstantHoisting to merge constants
// globally for potentially added savings.
//
def imm_su : PatLeaf<(imm), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i64immSExt32_su : PatLeaf<(i64immSExt32), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;

def relocImm8_su : PatLeaf<(i8 relocImm), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def relocImm16_su : PatLeaf<(i16 relocImm), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def relocImm32_su : PatLeaf<(i32 relocImm), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;

def i16relocImmSExt8_su : PatLeaf<(i16relocImmSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i32relocImmSExt8_su : PatLeaf<(i32relocImmSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i64relocImmSExt8_su : PatLeaf<(i64relocImmSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i64relocImmSExt32_su : PatLeaf<(i64relocImmSExt32), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;

def i16immSExt8_su : PatLeaf<(i16immSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i32immSExt8_su : PatLeaf<(i32immSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;
def i64immSExt8_su : PatLeaf<(i64immSExt8), [{
    return !shouldAvoidImmediateInstFormsForSize(N);
}]>;

// i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
// unsigned field.
def i64immZExt32 : ImmLeaf<i64, [{ return isUInt<32>(Imm); }]>;

def i64immZExt32SExt8 : ImmLeaf<i64, [{
  return isUInt<32>(Imm) && isInt<8>(static_cast<int32_t>(Imm));
}]>;

// Helper fragments for loads.

// It's safe to fold a zextload/extload from i1 as a regular i8 load. The
// upper bits are guaranteed to be zero and we were going to emit a MOV8rm
// which might get folded during peephole anyway.
def loadi8 : PatFrag<(ops node:$ptr), (i8 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  return ExtType == ISD::NON_EXTLOAD || ExtType == ISD::EXTLOAD ||
         ExtType == ISD::ZEXTLOAD;
}]>;

// It's always safe to treat a anyext i16 load as a i32 load if the i16 is
// known to be 32-bit aligned or better. Ditto for i8 to i16.
def loadi16 : PatFrag<(ops node:$ptr), (i16 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD && EnablePromoteAnyextLoad)
    return LD->getAlign() >= 2 && LD->isSimple();
  return false;
}]>;

def loadi32 : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD && EnablePromoteAnyextLoad)
    return LD->getAlign() >= 4 && LD->isSimple();
  return false;
}]>;

def loadi64  : PatFrag<(ops node:$ptr), (i64 (load node:$ptr))>;
def loadf16  : PatFrag<(ops node:$ptr), (f16 (load node:$ptr))>;
def loadf32  : PatFrag<(ops node:$ptr), (f32 (load node:$ptr))>;
def loadf64  : PatFrag<(ops node:$ptr), (f64 (load node:$ptr))>;
def loadf80  : PatFrag<(ops node:$ptr), (f80 (load node:$ptr))>;
def loadf128 : PatFrag<(ops node:$ptr), (f128 (load node:$ptr))>;
def alignedloadf128 : PatFrag<(ops node:$ptr), (f128 (load node:$ptr)), [{
  LoadSDNode *Ld = cast<LoadSDNode>(N);
  return Ld->getAlign() >= Ld->getMemoryVT().getStoreSize();
}]>;
def memopf128 : PatFrag<(ops node:$ptr), (f128 (load node:$ptr)), [{
  LoadSDNode *Ld = cast<LoadSDNode>(N);
  return Subtarget->hasSSEUnalignedMem() ||
         Ld->getAlign() >= Ld->getMemoryVT().getStoreSize();
}]>;

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

// We can treat an i8/i16 extending load to i64 as a 32 bit load if its known
// to be 4 byte aligned or better.
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType != ISD::EXTLOAD)
    return false;
  if (LD->getMemoryVT() == MVT::i32)
    return true;

  return LD->getAlign() >= 4 && LD->isSimple();
}]>;

// binary op with only one user
class binop_oneuse<SDPatternOperator operator>
    : PatFrag<(ops node:$A, node:$B),
              (operator node:$A, node:$B), [{
  return N->hasOneUse();
}]>;

def add_su : binop_oneuse<add>;
def and_su : binop_oneuse<and>;
def srl_su : binop_oneuse<srl>;

// unary op with only one user
class unop_oneuse<SDPatternOperator operator>
    : PatFrag<(ops node:$A),
              (operator node:$A), [{
  return N->hasOneUse();
}]>;


def ineg_su : unop_oneuse<ineg>;
def trunc_su : unop_oneuse<trunc>;

//===----------------------------------------------------------------------===//
// X86 Type infomation definitions
//===----------------------------------------------------------------------===//

/// X86TypeInfo - This is a bunch of information that describes relevant X86
/// information about value types.  For example, it can tell you what the
/// register class and preferred load to use.
class X86TypeInfo<ValueType vt, string instrsuffix, RegisterClass regclass,
                  PatFrag loadnode, X86MemOperand memoperand, ImmType immkind,
                  Operand immoperand, SDPatternOperator immoperator,
                  Operand imm8operand, SDPatternOperator imm8operator,
                  bit hasOddOpcode, OperandSize opSize,
                  bit hasREX_W> {
  /// VT - This is the value type itself.
  ValueType VT = vt;

  /// InstrSuffix - This is the suffix used on instructions with this type.  For
  /// example, i8 -> "b", i16 -> "w", i32 -> "l", i64 -> "q".
  string InstrSuffix = instrsuffix;

  /// RegClass - This is the register class associated with this type.  For
  /// example, i8 -> GR8, i16 -> GR16, i32 -> GR32, i64 -> GR64.
  RegisterClass RegClass = regclass;

  /// LoadNode - This is the load node associated with this type.  For
  /// example, i8 -> loadi8, i16 -> loadi16, i32 -> loadi32, i64 -> loadi64.
  PatFrag LoadNode = loadnode;

  /// MemOperand - This is the memory operand associated with this type.  For
  /// example, i8 -> i8mem, i16 -> i16mem, i32 -> i32mem, i64 -> i64mem.
  X86MemOperand MemOperand = memoperand;

  /// ImmEncoding - This is the encoding of an immediate of this type.  For
  /// example, i8 -> Imm8, i16 -> Imm16, i32 -> Imm32.  Note that i64 -> Imm32
  /// since the immediate fields of i64 instructions is a 32-bit sign extended
  /// value.
  ImmType ImmEncoding = immkind;

  /// ImmOperand - This is the operand kind of an immediate of this type.  For
  /// example, i8 -> i8imm, i16 -> i16imm, i32 -> i32imm.  Note that i64 ->
  /// i64i32imm since the immediate fields of i64 instructions is a 32-bit sign
  /// extended value.
  Operand ImmOperand = immoperand;

  /// ImmOperator - This is the operator that should be used to match an
  /// immediate of this kind in a pattern (e.g. imm, or i64immSExt32).
  SDPatternOperator ImmOperator = immoperator;

  /// Imm8Operand - This is the operand kind to use for an imm8 of this type.
  /// For example, i8 -> <invalid>, i16 -> i16i8imm, i32 -> i32i8imm.  This is
  /// only used for instructions that have a sign-extended imm8 field form.
  Operand Imm8Operand = imm8operand;

  /// Imm8Operator - This is the operator that should be used to match an 8-bit
  /// sign extended immediate of this kind in a pattern (e.g. imm16immSExt8).
  SDPatternOperator Imm8Operator = imm8operator;

  /// HasOddOpcode - This bit is true if the instruction should have an odd (as
  /// opposed to even) opcode.  Operations on i8 are usually even, operations on
  /// other datatypes are odd.
  bit HasOddOpcode = hasOddOpcode;

  /// OpSize - Selects whether the instruction needs a 0x66 prefix based on
  /// 16-bit vs 32-bit mode. i8/i64 set this to OpSizeFixed. i16 sets this
  /// to Opsize16. i32 sets this to OpSize32.
  OperandSize OpSize = opSize;

  /// HasREX_W - This bit is set to true if the instruction should have
  /// the 0x40 REX prefix.  This is set for i64 types.
  bit HasREX_W = hasREX_W;
}

def invalid_node : SDNode<"<<invalid_node>>", SDTIntLeaf,[],"<<invalid_node>>">;


def Xi8  : X86TypeInfo<i8, "b", GR8, loadi8, i8mem,
                       Imm8, i8imm, imm_su, i8imm, invalid_node,
                       0, OpSizeFixed, 0>;
def Xi16 : X86TypeInfo<i16, "w", GR16, loadi16, i16mem,
                       Imm16, i16imm, imm_su, i16i8imm, i16immSExt8_su,
                       1, OpSize16, 0>;
def Xi32 : X86TypeInfo<i32, "l", GR32, loadi32, i32mem,
                       Imm32, i32imm, imm_su, i32i8imm, i32immSExt8_su,
                       1, OpSize32, 0>;
def Xi64 : X86TypeInfo<i64, "q", GR64, loadi64, i64mem,
                       Imm32S, i64i32imm, i64immSExt32_su, i64i8imm, i64immSExt8_su,
                       1, OpSizeFixed, 1>;

//===----------------------------------------------------------------------===//
// Subsystems.
//===----------------------------------------------------------------------===//

include "X86InstrMisc.td"
include "X86InstrTBM.td"
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
include "X86InstrAVX512.td"
include "X86InstrMMX.td"
include "X86Instr3DNow.td"

include "X86InstrVMX.td"
include "X86InstrSVM.td"
include "X86InstrSNP.td"

include "X86InstrTSX.td"
include "X86InstrSGX.td"

include "X86InstrTDX.td"

// Key Locker instructions
include "X86InstrKL.td"

// AMX instructions
include "X86InstrAMX.td"

// RAO-INT instructions
include "X86InstrRAOINT.td"

// System instructions.
include "X86InstrSystem.td"

// Compiler Pseudo Instructions and Pat Patterns
include "X86InstrCompiler.td"
include "X86InstrVecCompiler.td"

// Assembler mnemonic/instruction aliases
include "X86InstrAsmAlias.td"

