//===- X86InstrInfo.td - Describe the X86 Instruction Set --*- tablegen -*-===//
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

def SDTX86CmpTest : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;

def SDTX86Cmov    : SDTypeProfile<1, 4,
                                  [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>,
                                   SDTCisVT<3, i8>, SDTCisVT<4, i32>]>;

def SDTUnaryArithOvf  : SDTypeProfile<1, 1,
                                      [SDTCisInt<0>]>;
def SDTBinaryArithOvf : SDTypeProfile<1, 2,
                                      [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                                       SDTCisInt<0>]>;

def SDTX86BrCond  : SDTypeProfile<0, 3,
                                  [SDTCisVT<0, OtherVT>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86SetCC   : SDTypeProfile<1, 2,
                                  [SDTCisVT<0, i8>,
                                   SDTCisVT<1, i8>, SDTCisVT<2, i32>]>;

def SDTX86cas : SDTypeProfile<0, 3, [SDTCisPtrTy<0>, SDTCisInt<1>, 
                                     SDTCisVT<2, i8>]>;
def SDTX86cas8 : SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>;

def SDTX86atomicBinary : SDTypeProfile<2, 3, [SDTCisInt<0>, SDTCisInt<1>,
                                SDTCisPtrTy<2>, SDTCisInt<3>,SDTCisInt<4>]>;
def SDTX86Ret     : SDTypeProfile<0, -1, [SDTCisVT<0, i16>]>;

def SDT_X86CallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_X86CallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                         SDTCisVT<1, i32> ]>;

def SDT_X86Call   : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;

def SDTX86RepStr  : SDTypeProfile<0, 1, [SDTCisVT<0, OtherVT>]>;

def SDTX86RdTsc   : SDTypeProfile<0, 0, []>;

def SDTX86Wrapper : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>, SDTCisPtrTy<0>]>;

def SDT_X86TLSADDR : SDTypeProfile<1, 1, [SDTCisPtrTy<0>, SDTCisInt<1>]>;

def SDT_X86TLSTP : SDTypeProfile<1, 0, [SDTCisPtrTy<0>]>;

def SDT_X86EHRET : SDTypeProfile<0, 1, [SDTCisInt<0>]>;

def SDT_X86TCRET : SDTypeProfile<0, 2, [SDTCisPtrTy<0>, SDTCisVT<1, i32>]>;

def X86bsf     : SDNode<"X86ISD::BSF",      SDTIntUnaryOp>;
def X86bsr     : SDNode<"X86ISD::BSR",      SDTIntUnaryOp>;
def X86shld    : SDNode<"X86ISD::SHLD",     SDTIntShiftDOp>;
def X86shrd    : SDNode<"X86ISD::SHRD",     SDTIntShiftDOp>;

def X86cmp     : SDNode<"X86ISD::CMP" ,     SDTX86CmpTest>;

def X86bt      : SDNode<"X86ISD::BT",       SDTX86CmpTest>;

def X86cmov    : SDNode<"X86ISD::CMOV",     SDTX86Cmov>;
def X86brcond  : SDNode<"X86ISD::BRCOND",   SDTX86BrCond,
                        [SDNPHasChain]>;
def X86setcc   : SDNode<"X86ISD::SETCC",    SDTX86SetCC>;

def X86cas : SDNode<"X86ISD::LCMPXCHG_DAG", SDTX86cas,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad]>;
def X86cas8 : SDNode<"X86ISD::LCMPXCHG8_DAG", SDTX86cas8,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad]>;
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
                        [SDNPHasChain, SDNPOptInFlag]>;

def X86callseq_start :
                 SDNode<"ISD::CALLSEQ_START", SDT_X86CallSeqStart,
                        [SDNPHasChain, SDNPOutFlag]>;
def X86callseq_end :
                 SDNode<"ISD::CALLSEQ_END",   SDT_X86CallSeqEnd,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;       

def X86call    : SDNode<"X86ISD::CALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;

def X86tailcall: SDNode<"X86ISD::TAILCALL",     SDT_X86Call,
                        [SDNPHasChain, SDNPOutFlag, SDNPOptInFlag]>;

def X86rep_stos: SDNode<"X86ISD::REP_STOS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore]>;
def X86rep_movs: SDNode<"X86ISD::REP_MOVS", SDTX86RepStr,
                        [SDNPHasChain, SDNPInFlag, SDNPOutFlag, SDNPMayStore,
                         SDNPMayLoad]>;

def X86rdtsc   : SDNode<"X86ISD::RDTSC_DAG",SDTX86RdTsc,
                        [SDNPHasChain, SDNPOutFlag, SDNPSideEffect]>;

def X86Wrapper    : SDNode<"X86ISD::Wrapper",     SDTX86Wrapper>;
def X86WrapperRIP : SDNode<"X86ISD::WrapperRIP",  SDTX86Wrapper>;

def X86tlsaddr : SDNode<"X86ISD::TLSADDR", SDT_X86TLSADDR,
                        [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def X86TLStp : SDNode<"X86ISD::THREAD_POINTER", SDT_X86TLSTP, []>;

def X86ehret : SDNode<"X86ISD::EH_RETURN", SDT_X86EHRET,
                        [SDNPHasChain]>;

def X86tcret : SDNode<"X86ISD::TC_RETURN", SDT_X86TCRET, 
                        [SDNPHasChain,  SDNPOptInFlag]>;

def X86add_ovf  : SDNode<"X86ISD::ADD",  SDTBinaryArithOvf>;
def X86sub_ovf  : SDNode<"X86ISD::SUB",  SDTBinaryArithOvf>;
def X86smul_ovf : SDNode<"X86ISD::SMUL", SDTBinaryArithOvf>;
def X86umul_ovf : SDNode<"X86ISD::UMUL", SDTUnaryArithOvf>;

//===----------------------------------------------------------------------===//
// X86 Operand Definitions.
//

// *mem - Operand definitions for the funky X86 addressing mode operands.
//
class X86MemOperand<string printMethod> : Operand<iPTR> {
  let PrintMethod = printMethod;
  let MIOperandInfo = (ops ptr_rc, i8imm, ptr_rc, i32imm);
}

def i8mem   : X86MemOperand<"printi8mem">;
def i16mem  : X86MemOperand<"printi16mem">;
def i32mem  : X86MemOperand<"printi32mem">;
def i64mem  : X86MemOperand<"printi64mem">;
def i128mem : X86MemOperand<"printi128mem">;
def f32mem  : X86MemOperand<"printf32mem">;
def f64mem  : X86MemOperand<"printf64mem">;
def f80mem  : X86MemOperand<"printf80mem">;
def f128mem : X86MemOperand<"printf128mem">;

def lea32mem : Operand<i32> {
  let PrintMethod = "printi32mem";
  let MIOperandInfo = (ops GR32, i8imm, GR32, i32imm);
}

def SSECC : Operand<i8> {
  let PrintMethod = "printSSECC";
}

def piclabel: Operand<i32> {
  let PrintMethod = "printPICLabel";
}

// A couple of more descriptive operand definitions.
// 16-bits but only 8 bits are significant.
def i16i8imm  : Operand<i16>;
// 32-bits but only 8 bits are significant.
def i32i8imm  : Operand<i32>;

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;

//===----------------------------------------------------------------------===//
// X86 Complex Pattern Definitions.
//

// Define X86 specific addressing mode.
def addr      : ComplexPattern<iPTR, 4, "SelectAddr", [], []>;
def lea32addr : ComplexPattern<i32, 4, "SelectLEAAddr",
                               [add, mul, shl, or, frameindex], []>;

//===----------------------------------------------------------------------===//
// X86 Instruction Predicate Definitions.
def HasMMX       : Predicate<"Subtarget->hasMMX()">;
def HasSSE1      : Predicate<"Subtarget->hasSSE1()">;
def HasSSE2      : Predicate<"Subtarget->hasSSE2()">;
def HasSSE3      : Predicate<"Subtarget->hasSSE3()">;
def HasSSSE3     : Predicate<"Subtarget->hasSSSE3()">;
def HasSSE41     : Predicate<"Subtarget->hasSSE41()">;
def HasSSE42     : Predicate<"Subtarget->hasSSE42()">;
def FPStackf32   : Predicate<"!Subtarget->hasSSE1()">;
def FPStackf64   : Predicate<"!Subtarget->hasSSE2()">;
def In32BitMode  : Predicate<"!Subtarget->is64Bit()">;
def In64BitMode  : Predicate<"Subtarget->is64Bit()">;
def SmallCode    : Predicate<"TM.getCodeModel() == CodeModel::Small">;
def NotSmallCode : Predicate<"TM.getCodeModel() != CodeModel::Small">;
def IsStatic     : Predicate<"TM.getRelocationModel() == Reloc::Static">;
def OptForSpeed  : Predicate<"!OptForSize">;
def FastBTMem    : Predicate<"!Subtarget->isBTMemSlow()">;

//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

//===----------------------------------------------------------------------===//
// Pattern fragments...
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

def i16immSExt8  : PatLeaf<(i16 imm), [{
  // i16immSExt8 predicate - True if the 16-bit immediate fits in a 8-bit
  // sign extended field.
  return (int16_t)N->getZExtValue() == (int8_t)N->getZExtValue();
}]>;

def i32immSExt8  : PatLeaf<(i32 imm), [{
  // i32immSExt8 predicate - True if the 32-bit immediate fits in a 8-bit
  // sign extended field.
  return (int32_t)N->getZExtValue() == (int8_t)N->getZExtValue();
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

def loadi16_anyext : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)), [{
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

def nvloadi32 : PatFrag<(ops node:$ptr), (i32 (unindexedload node:$ptr)), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  if (LD->isVolatile())
    return false;
  ISD::LoadExtType ExtType = LD->getExtensionType();
  if (ExtType == ISD::NON_EXTLOAD)
    return true;
  if (ExtType == ISD::EXTLOAD)
    return LD->getAlignment() >= 4;
  return false;
}]>;

def gsload : PatFrag<(ops node:$ptr), (load node:$ptr), [{
  LoadSDNode *LD = cast<LoadSDNode>(N);
  const Value *Src = LD->getSrcValue();
  if (!Src)
    return false;
  if (const PointerType *PT = dyn_cast<PointerType>(Src->getType()))
    return PT->getAddressSpace() == 256;
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

def zextloadi8i1   : PatFrag<(ops node:$ptr), (i8  (zextloadi1 node:$ptr))>;
def zextloadi16i1  : PatFrag<(ops node:$ptr), (i16 (zextloadi1 node:$ptr))>;
def zextloadi32i1  : PatFrag<(ops node:$ptr), (i32 (zextloadi1 node:$ptr))>;
def zextloadi16i8  : PatFrag<(ops node:$ptr), (i16 (zextloadi8 node:$ptr))>;
def zextloadi32i8  : PatFrag<(ops node:$ptr), (i32 (zextloadi8 node:$ptr))>;
def zextloadi32i16 : PatFrag<(ops node:$ptr), (i32 (zextloadi16 node:$ptr))>;

def extloadi8i1    : PatFrag<(ops node:$ptr), (i8  (extloadi1 node:$ptr))>;
def extloadi16i1   : PatFrag<(ops node:$ptr), (i16 (extloadi1 node:$ptr))>;
def extloadi32i1   : PatFrag<(ops node:$ptr), (i32 (extloadi1 node:$ptr))>;
def extloadi16i8   : PatFrag<(ops node:$ptr), (i16 (extloadi8 node:$ptr))>;
def extloadi32i8   : PatFrag<(ops node:$ptr), (i32 (extloadi8 node:$ptr))>;
def extloadi32i16  : PatFrag<(ops node:$ptr), (i32 (extloadi16 node:$ptr))>;


// An 'and' node with a single use.
def and_su : PatFrag<(ops node:$lhs, node:$rhs), (and node:$lhs, node:$rhs), [{
  return N->hasOneUse();
}]>;

// 'shld' and 'shrd' instruction patterns. Note that even though these have
// the srl and shl in their patterns, the C++ code must still check for them,
// because predicates are tested before children nodes are explored.

def shrd : PatFrag<(ops node:$src1, node:$amt1, node:$src2, node:$amt2),
                   (or (srl node:$src1, node:$amt1),
                       (shl node:$src2, node:$amt2)), [{
  assert(N->getOpcode() == ISD::OR);
  return N->getOperand(0).getOpcode() == ISD::SRL &&
         N->getOperand(1).getOpcode() == ISD::SHL &&
         isa<ConstantSDNode>(N->getOperand(0).getOperand(1)) &&
         isa<ConstantSDNode>(N->getOperand(1).getOperand(1)) &&
         N->getOperand(0).getConstantOperandVal(1) ==
         N->getValueSizeInBits(0) - N->getOperand(1).getConstantOperandVal(1);
}]>;

def shld : PatFrag<(ops node:$src1, node:$amt1, node:$src2, node:$amt2),
                   (or (shl node:$src1, node:$amt1),
                       (srl node:$src2, node:$amt2)), [{
  assert(N->getOpcode() == ISD::OR);
  return N->getOperand(0).getOpcode() == ISD::SHL &&
         N->getOperand(1).getOpcode() == ISD::SRL &&
         isa<ConstantSDNode>(N->getOperand(0).getOperand(1)) &&
         isa<ConstantSDNode>(N->getOperand(1).getOperand(1)) &&
         N->getOperand(0).getConstantOperandVal(1) ==
         N->getValueSizeInBits(0) - N->getOperand(1).getConstantOperandVal(1);
}]>;

//===----------------------------------------------------------------------===//
// Instruction list...
//

// ADJCALLSTACKDOWN/UP implicitly use/def ESP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber EFLAGS.
let Defs = [ESP, EFLAGS], Uses = [ESP] in {
def ADJCALLSTACKDOWN32 : I<0, Pseudo, (outs), (ins i32imm:$amt),
                           "#ADJCALLSTACKDOWN",
                           [(X86callseq_start timm:$amt)]>,
                          Requires<[In32BitMode]>;
def ADJCALLSTACKUP32   : I<0, Pseudo, (outs), (ins i32imm:$amt1, i32imm:$amt2),
                           "#ADJCALLSTACKUP",
                           [(X86callseq_end timm:$amt1, timm:$amt2)]>,
                          Requires<[In32BitMode]>;
}

// Nop
let neverHasSideEffects = 1 in
  def NOOP : I<0x90, RawFrm, (outs), (ins), "nop", []>;

// PIC base
let neverHasSideEffects = 1, isNotDuplicable = 1, Uses = [ESP] in
  def MOVPC32r : Ii32<0xE8, Pseudo, (outs GR32:$reg), (ins piclabel:$label),
                      "call\t$label\n\tpop{l}\t$reg", []>;

//===----------------------------------------------------------------------===//
//  Control Flow Instructions...
//

// Return instructions.
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, FPForm = SpecialFP, FPFormBits = SpecialFP.Value in {
  def RET    : I   <0xC3, RawFrm, (outs), (ins variable_ops),
                    "ret",
                    [(X86retflag 0)]>;
  def RETI   : Ii16<0xC2, RawFrm, (outs), (ins i16imm:$amt, variable_ops),
                    "ret\t$amt",
                    [(X86retflag imm:$amt)]>;
}

// All branches are RawFrm, Void, Branch, and Terminators
let isBranch = 1, isTerminator = 1 in
  class IBr<bits<8> opcode, dag ins, string asm, list<dag> pattern> :
        I<opcode, RawFrm, (outs), ins, asm, pattern>;

let isBranch = 1, isBarrier = 1 in
  def JMP : IBr<0xE9, (ins brtarget:$dst), "jmp\t$dst", [(br bb:$dst)]>;

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def JMP32r     : I<0xFF, MRM4r, (outs), (ins GR32:$dst), "jmp{l}\t{*}$dst",
                     [(brind GR32:$dst)]>;
  def JMP32m     : I<0xFF, MRM4m, (outs), (ins i32mem:$dst), "jmp{l}\t{*}$dst",
                     [(brind (loadi32 addr:$dst))]>;
}

// Conditional branches
let Uses = [EFLAGS] in {
def JE  : IBr<0x84, (ins brtarget:$dst), "je\t$dst",
              [(X86brcond bb:$dst, X86_COND_E, EFLAGS)]>, TB;
def JNE : IBr<0x85, (ins brtarget:$dst), "jne\t$dst",
              [(X86brcond bb:$dst, X86_COND_NE, EFLAGS)]>, TB;
def JL  : IBr<0x8C, (ins brtarget:$dst), "jl\t$dst",
              [(X86brcond bb:$dst, X86_COND_L, EFLAGS)]>, TB;
def JLE : IBr<0x8E, (ins brtarget:$dst), "jle\t$dst",
              [(X86brcond bb:$dst, X86_COND_LE, EFLAGS)]>, TB;
def JG  : IBr<0x8F, (ins brtarget:$dst), "jg\t$dst",
              [(X86brcond bb:$dst, X86_COND_G, EFLAGS)]>, TB;
def JGE : IBr<0x8D, (ins brtarget:$dst), "jge\t$dst",
              [(X86brcond bb:$dst, X86_COND_GE, EFLAGS)]>, TB;

def JB  : IBr<0x82, (ins brtarget:$dst), "jb\t$dst",
              [(X86brcond bb:$dst, X86_COND_B, EFLAGS)]>, TB;
def JBE : IBr<0x86, (ins brtarget:$dst), "jbe\t$dst",
              [(X86brcond bb:$dst, X86_COND_BE, EFLAGS)]>, TB;
def JA  : IBr<0x87, (ins brtarget:$dst), "ja\t$dst",
              [(X86brcond bb:$dst, X86_COND_A, EFLAGS)]>, TB;
def JAE : IBr<0x83, (ins brtarget:$dst), "jae\t$dst",
              [(X86brcond bb:$dst, X86_COND_AE, EFLAGS)]>, TB;

def JS  : IBr<0x88, (ins brtarget:$dst), "js\t$dst",
              [(X86brcond bb:$dst, X86_COND_S, EFLAGS)]>, TB;
def JNS : IBr<0x89, (ins brtarget:$dst), "jns\t$dst",
              [(X86brcond bb:$dst, X86_COND_NS, EFLAGS)]>, TB;
def JP  : IBr<0x8A, (ins brtarget:$dst), "jp\t$dst",
              [(X86brcond bb:$dst, X86_COND_P, EFLAGS)]>, TB;
def JNP : IBr<0x8B, (ins brtarget:$dst), "jnp\t$dst",
              [(X86brcond bb:$dst, X86_COND_NP, EFLAGS)]>, TB;
def JO  : IBr<0x80, (ins brtarget:$dst), "jo\t$dst",
              [(X86brcond bb:$dst, X86_COND_O, EFLAGS)]>, TB;
def JNO : IBr<0x81, (ins brtarget:$dst), "jno\t$dst",
              [(X86brcond bb:$dst, X86_COND_NO, EFLAGS)]>, TB;
} // Uses = [EFLAGS]

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. ESP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [EAX, ECX, EDX, FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [ESP] in {
    def CALLpcrel32 : Ii32<0xE8, RawFrm, (outs), (ins i32imm:$dst,variable_ops),
                           "call\t${dst:call}", []>;
    def CALL32r     : I<0xFF, MRM2r, (outs), (ins GR32:$dst, variable_ops),
                        "call\t{*}$dst", [(X86call GR32:$dst)]>;
    def CALL32m     : I<0xFF, MRM2m, (outs), (ins i32mem:$dst, variable_ops),
                        "call\t{*}$dst", [(X86call (loadi32 addr:$dst))]>;
  }

// Tail call stuff.

def TAILCALL : I<0, Pseudo, (outs), (ins),
                         "#TAILCALL",
                         []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
def TCRETURNdi : I<0, Pseudo, (outs), (ins i32imm:$dst, i32imm:$offset, variable_ops),
                 "#TC_RETURN $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
def TCRETURNri : I<0, Pseudo, (outs), (ins GR32:$dst, i32imm:$offset, variable_ops),
                 "#TC_RETURN $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in

  def TAILJMPd : IBr<0xE9, (ins i32imm:$dst), "jmp\t${dst:call}  # TAILCALL",
                 []>;
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
  def TAILJMPr : I<0xFF, MRM4r, (outs), (ins GR32:$dst), "jmp{l}\t{*}$dst  # TAILCALL",
                 []>;     
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
  def TAILJMPm : I<0xFF, MRM4m, (outs), (ins i32mem:$dst),
                   "jmp\t{*}$dst  # TAILCALL", []>;

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
let Defs = [EBP, ESP], Uses = [EBP, ESP], mayLoad = 1, neverHasSideEffects=1 in
def LEAVE    : I<0xC9, RawFrm,
                 (outs), (ins), "leave", []>;

let Defs = [ESP], Uses = [ESP], neverHasSideEffects=1 in {
let mayLoad = 1 in
def POP32r   : I<0x58, AddRegFrm, (outs GR32:$reg), (ins), "pop{l}\t$reg", []>;

let mayStore = 1 in
def PUSH32r  : I<0x50, AddRegFrm, (outs), (ins GR32:$reg), "push{l}\t$reg",[]>;
}

let Defs = [ESP, EFLAGS], Uses = [ESP], mayLoad = 1, neverHasSideEffects=1 in
def POPFD    : I<0x9D, RawFrm, (outs), (ins), "popf", []>;
let Defs = [ESP], Uses = [ESP, EFLAGS], mayStore = 1, neverHasSideEffects=1 in
def PUSHFD   : I<0x9C, RawFrm, (outs), (ins), "pushf", []>;

let isTwoAddress = 1 in                               // GR32 = bswap GR32
  def BSWAP32r : I<0xC8, AddRegFrm,
                   (outs GR32:$dst), (ins GR32:$src),
                   "bswap{l}\t$dst", 
                   [(set GR32:$dst, (bswap GR32:$src))]>, TB;


// Bit scan instructions.
let Defs = [EFLAGS] in {
def BSF16rr  : I<0xBC, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, (X86bsf GR16:$src)), (implicit EFLAGS)]>, TB;
def BSF16rm  : I<0xBC, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsf{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, (X86bsf (loadi16 addr:$src))),
                  (implicit EFLAGS)]>, TB;
def BSF32rr  : I<0xBC, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, (X86bsf GR32:$src)), (implicit EFLAGS)]>, TB;
def BSF32rm  : I<0xBC, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsf{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, (X86bsf (loadi32 addr:$src))),
                  (implicit EFLAGS)]>, TB;

def BSR16rr  : I<0xBD, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, (X86bsr GR16:$src)), (implicit EFLAGS)]>, TB;
def BSR16rm  : I<0xBD, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                 "bsr{w}\t{$src, $dst|$dst, $src}",
                 [(set GR16:$dst, (X86bsr (loadi16 addr:$src))),
                  (implicit EFLAGS)]>, TB;
def BSR32rr  : I<0xBD, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, (X86bsr GR32:$src)), (implicit EFLAGS)]>, TB;
def BSR32rm  : I<0xBD, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                 "bsr{l}\t{$src, $dst|$dst, $src}",
                 [(set GR32:$dst, (X86bsr (loadi32 addr:$src))),
                  (implicit EFLAGS)]>, TB;
} // Defs = [EFLAGS]

let neverHasSideEffects = 1 in
def LEA16r   : I<0x8D, MRMSrcMem,
                 (outs GR16:$dst), (ins i32mem:$src),
                 "lea{w}\t{$src|$dst}, {$dst|$src}", []>, OpSize;
let isReMaterializable = 1 in
def LEA32r   : I<0x8D, MRMSrcMem,
                 (outs GR32:$dst), (ins lea32mem:$src),
                 "lea{l}\t{$src|$dst}, {$dst|$src}",
                 [(set GR32:$dst, lea32addr:$src)]>, Requires<[In32BitMode]>;

let Defs = [ECX,EDI,ESI], Uses = [ECX,EDI,ESI] in {
def REP_MOVSB : I<0xA4, RawFrm, (outs), (ins), "{rep;movsb|rep movsb}",
                  [(X86rep_movs i8)]>, REP;
def REP_MOVSW : I<0xA5, RawFrm, (outs), (ins), "{rep;movsw|rep movsw}",
                  [(X86rep_movs i16)]>, REP, OpSize;
def REP_MOVSD : I<0xA5, RawFrm, (outs), (ins), "{rep;movsl|rep movsd}",
                  [(X86rep_movs i32)]>, REP;
}

let Defs = [ECX,EDI], Uses = [AL,ECX,EDI] in
def REP_STOSB : I<0xAA, RawFrm, (outs), (ins), "{rep;stosb|rep stosb}",
                  [(X86rep_stos i8)]>, REP;
let Defs = [ECX,EDI], Uses = [AX,ECX,EDI] in
def REP_STOSW : I<0xAB, RawFrm, (outs), (ins), "{rep;stosw|rep stosw}",
                  [(X86rep_stos i16)]>, REP, OpSize;
let Defs = [ECX,EDI], Uses = [EAX,ECX,EDI] in
def REP_STOSD : I<0xAB, RawFrm, (outs), (ins), "{rep;stosl|rep stosd}",
                  [(X86rep_stos i32)]>, REP;

let Defs = [RAX, RDX] in
def RDTSC : I<0x31, RawFrm, (outs), (ins), "rdtsc", [(X86rdtsc)]>,
            TB;

let isBarrier = 1, hasCtrlDep = 1 in {
def TRAP    : I<0x0B, RawFrm, (outs), (ins), "ud2", [(trap)]>, TB;
}

//===----------------------------------------------------------------------===//
//  Input/Output Instructions...
//
let Defs = [AL], Uses = [DX] in
def IN8rr  : I<0xEC, RawFrm, (outs), (ins),
               "in{b}\t{%dx, %al|%AL, %DX}", []>;
let Defs = [AX], Uses = [DX] in
def IN16rr : I<0xED, RawFrm, (outs), (ins),
               "in{w}\t{%dx, %ax|%AX, %DX}", []>,  OpSize;
let Defs = [EAX], Uses = [DX] in
def IN32rr : I<0xED, RawFrm, (outs), (ins),
               "in{l}\t{%dx, %eax|%EAX, %DX}", []>;

let Defs = [AL] in
def IN8ri  : Ii8<0xE4, RawFrm, (outs), (ins i16i8imm:$port),
                  "in{b}\t{$port, %al|%AL, $port}", []>;
let Defs = [AX] in
def IN16ri : Ii8<0xE5, RawFrm, (outs), (ins i16i8imm:$port),
                  "in{w}\t{$port, %ax|%AX, $port}", []>, OpSize;
let Defs = [EAX] in
def IN32ri : Ii8<0xE5, RawFrm, (outs), (ins i16i8imm:$port),
                  "in{l}\t{$port, %eax|%EAX, $port}", []>;

let Uses = [DX, AL] in
def OUT8rr  : I<0xEE, RawFrm, (outs), (ins),
                "out{b}\t{%al, %dx|%DX, %AL}", []>;
let Uses = [DX, AX] in
def OUT16rr : I<0xEF, RawFrm, (outs), (ins),
                "out{w}\t{%ax, %dx|%DX, %AX}", []>, OpSize;
let Uses = [DX, EAX] in
def OUT32rr : I<0xEF, RawFrm, (outs), (ins),
                "out{l}\t{%eax, %dx|%DX, %EAX}", []>;

let Uses = [AL] in
def OUT8ir  : Ii8<0xE6, RawFrm, (outs), (ins i16i8imm:$port),
                   "out{b}\t{%al, $port|$port, %AL}", []>;
let Uses = [AX] in
def OUT16ir : Ii8<0xE7, RawFrm, (outs), (ins i16i8imm:$port),
                   "out{w}\t{%ax, $port|$port, %AX}", []>, OpSize;
let Uses = [EAX] in
def OUT32ir : Ii8<0xE7, RawFrm, (outs), (ins i16i8imm:$port),
                   "out{l}\t{%eax, $port|$port, %EAX}", []>;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//
let neverHasSideEffects = 1 in {
def MOV8rr  : I<0x88, MRMDestReg, (outs GR8 :$dst), (ins GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}", []>;
def MOV16rr : I<0x89, MRMDestReg, (outs GR16:$dst), (ins GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32rr : I<0x89, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
}
let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV8ri  : Ii8 <0xB0, AddRegFrm, (outs GR8 :$dst), (ins i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(set GR8:$dst, imm:$src)]>;
def MOV16ri : Ii16<0xB8, AddRegFrm, (outs GR16:$dst), (ins i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(set GR16:$dst, imm:$src)]>, OpSize;
def MOV32ri : Ii32<0xB8, AddRegFrm, (outs GR32:$dst), (ins i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, imm:$src)]>;
}
def MOV8mi  : Ii8 <0xC6, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src),
                   "mov{b}\t{$src, $dst|$dst, $src}",
                   [(store (i8 imm:$src), addr:$dst)]>;
def MOV16mi : Ii16<0xC7, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src),
                   "mov{w}\t{$src, $dst|$dst, $src}",
                   [(store (i16 imm:$src), addr:$dst)]>, OpSize;
def MOV32mi : Ii32<0xC7, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src),
                   "mov{l}\t{$src, $dst|$dst, $src}",
                   [(store (i32 imm:$src), addr:$dst)]>;

let canFoldAsLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV8rm  : I<0x8A, MRMSrcMem, (outs GR8 :$dst), (ins i8mem :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(set GR8:$dst, (load addr:$src))]>;
def MOV16rm : I<0x8B, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(set GR16:$dst, (load addr:$src))]>, OpSize;
def MOV32rm : I<0x8B, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(set GR32:$dst, (load addr:$src))]>;
}

def MOV8mr  : I<0x88, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src),
                "mov{b}\t{$src, $dst|$dst, $src}",
                [(store GR8:$src, addr:$dst)]>;
def MOV16mr : I<0x89, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}",
                [(store GR16:$src, addr:$dst)]>, OpSize;
def MOV32mr : I<0x89, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}",
                [(store GR32:$src, addr:$dst)]>;
                
//===----------------------------------------------------------------------===//
//  Fixed-Register Multiplication and Division Instructions...
//

// Extra precision multiplication
let Defs = [AL,AH,EFLAGS], Uses = [AL] in
def MUL8r  : I<0xF6, MRM4r, (outs),  (ins GR8:$src), "mul{b}\t$src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, GR8:$src)),
                (implicit EFLAGS)]>;     // AL,AH = AL*GR8

let Defs = [AX,DX,EFLAGS], Uses = [AX], neverHasSideEffects = 1 in
def MUL16r : I<0xF7, MRM4r, (outs),  (ins GR16:$src),
               "mul{w}\t$src", 
               []>, OpSize;    // AX,DX = AX*GR16

let Defs = [EAX,EDX,EFLAGS], Uses = [EAX], neverHasSideEffects = 1 in
def MUL32r : I<0xF7, MRM4r, (outs),  (ins GR32:$src),
               "mul{l}\t$src",
               []>; // EAX,EDX = EAX*GR32

let Defs = [AL,AH,EFLAGS], Uses = [AL] in
def MUL8m  : I<0xF6, MRM4m, (outs), (ins i8mem :$src),
               "mul{b}\t$src",
               // FIXME: Used for 8-bit mul, ignore result upper 8 bits.
               // This probably ought to be moved to a def : Pat<> if the
               // syntax can be accepted.
               [(set AL, (mul AL, (loadi8 addr:$src))),
                (implicit EFLAGS)]>;   // AL,AH = AL*[mem8]

let mayLoad = 1, neverHasSideEffects = 1 in {
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def MUL16m : I<0xF7, MRM4m, (outs), (ins i16mem:$src),
               "mul{w}\t$src",
               []>, OpSize; // AX,DX = AX*[mem16]

let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
def MUL32m : I<0xF7, MRM4m, (outs), (ins i32mem:$src),
              "mul{l}\t$src",
              []>;          // EAX,EDX = EAX*[mem32]
}

let neverHasSideEffects = 1 in {
let Defs = [AL,AH,EFLAGS], Uses = [AL] in
def IMUL8r  : I<0xF6, MRM5r, (outs),  (ins GR8:$src), "imul{b}\t$src", []>;
              // AL,AH = AL*GR8
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def IMUL16r : I<0xF7, MRM5r, (outs),  (ins GR16:$src), "imul{w}\t$src", []>,
              OpSize;    // AX,DX = AX*GR16
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX] in
def IMUL32r : I<0xF7, MRM5r, (outs),  (ins GR32:$src), "imul{l}\t$src", []>;
              // EAX,EDX = EAX*GR32
let mayLoad = 1 in {
let Defs = [AL,AH,EFLAGS], Uses = [AL] in
def IMUL8m  : I<0xF6, MRM5m, (outs), (ins i8mem :$src),
                "imul{b}\t$src", []>;    // AL,AH = AL*[mem8]
let Defs = [AX,DX,EFLAGS], Uses = [AX] in
def IMUL16m : I<0xF7, MRM5m, (outs), (ins i16mem:$src),
                "imul{w}\t$src", []>, OpSize; // AX,DX = AX*[mem16]
let Defs = [EAX,EDX], Uses = [EAX] in
def IMUL32m : I<0xF7, MRM5m, (outs), (ins i32mem:$src),
                "imul{l}\t$src", []>;  // EAX,EDX = EAX*[mem32]
}
} // neverHasSideEffects

// unsigned division/remainder
let Defs = [AL,AH,EFLAGS], Uses = [AX] in
def DIV8r  : I<0xF6, MRM6r, (outs),  (ins GR8:$src),          // AX/r8 = AL,AH
               "div{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def DIV16r : I<0xF7, MRM6r, (outs),  (ins GR16:$src),         // DX:AX/r16 = AX,DX
               "div{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def DIV32r : I<0xF7, MRM6r, (outs),  (ins GR32:$src),         // EDX:EAX/r32 = EAX,EDX
               "div{l}\t$src", []>;
let mayLoad = 1 in {
let Defs = [AL,AH,EFLAGS], Uses = [AX] in
def DIV8m  : I<0xF6, MRM6m, (outs), (ins i8mem:$src),       // AX/[mem8] = AL,AH
               "div{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def DIV16m : I<0xF7, MRM6m, (outs), (ins i16mem:$src),      // DX:AX/[mem16] = AX,DX
               "div{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def DIV32m : I<0xF7, MRM6m, (outs), (ins i32mem:$src),      // EDX:EAX/[mem32] = EAX,EDX
               "div{l}\t$src", []>;
}

// Signed division/remainder.
let Defs = [AL,AH,EFLAGS], Uses = [AX] in
def IDIV8r : I<0xF6, MRM7r, (outs),  (ins GR8:$src),          // AX/r8 = AL,AH
               "idiv{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def IDIV16r: I<0xF7, MRM7r, (outs),  (ins GR16:$src),         // DX:AX/r16 = AX,DX
               "idiv{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def IDIV32r: I<0xF7, MRM7r, (outs),  (ins GR32:$src),         // EDX:EAX/r32 = EAX,EDX
               "idiv{l}\t$src", []>;
let mayLoad = 1, mayLoad = 1 in {
let Defs = [AL,AH,EFLAGS], Uses = [AX] in
def IDIV8m : I<0xF6, MRM7m, (outs), (ins i8mem:$src),      // AX/[mem8] = AL,AH
               "idiv{b}\t$src", []>;
let Defs = [AX,DX,EFLAGS], Uses = [AX,DX] in
def IDIV16m: I<0xF7, MRM7m, (outs), (ins i16mem:$src),     // DX:AX/[mem16] = AX,DX
               "idiv{w}\t$src", []>, OpSize;
let Defs = [EAX,EDX,EFLAGS], Uses = [EAX,EDX] in
def IDIV32m: I<0xF7, MRM7m, (outs), (ins i32mem:$src),     // EDX:EAX/[mem32] = EAX,EDX
               "idiv{l}\t$src", []>;
}

//===----------------------------------------------------------------------===//
//  Two address Instructions.
//
let isTwoAddress = 1 in {

// Conditional moves
let Uses = [EFLAGS] in {
let isCommutable = 1 in {
def CMOVB16rr : I<0x42, MRMSrcReg,       // if <u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovb\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rr : I<0x42, MRMSrcReg,       // if <u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovb\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rr: I<0x43, MRMSrcReg,       // if >=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovae\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rr: I<0x43, MRMSrcReg,       // if >=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovae\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rr : I<0x44, MRMSrcReg,       // if ==, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmove\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rr : I<0x44, MRMSrcReg,       // if ==, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmove\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rr: I<0x45, MRMSrcReg,       // if !=, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovne\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rr: I<0x45, MRMSrcReg,       // if !=, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovne\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rr: I<0x46, MRMSrcReg,       // if <=u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovbe\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rr: I<0x46, MRMSrcReg,       // if <=u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovbe\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rr : I<0x47, MRMSrcReg,       // if >u, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmova\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rr : I<0x47, MRMSrcReg,       // if >u, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmova\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rr : I<0x4C, MRMSrcReg,       // if <s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovl\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rr : I<0x4C, MRMSrcReg,       // if <s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovl\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rr: I<0x4D, MRMSrcReg,       // if >=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovge\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rr: I<0x4D, MRMSrcReg,       // if >=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovge\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rr: I<0x4E, MRMSrcReg,       // if <=s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovle\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rr: I<0x4E, MRMSrcReg,       // if <=s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovle\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rr : I<0x4F, MRMSrcReg,       // if >s, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovg\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rr : I<0x4F, MRMSrcReg,       // if >s, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovg\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rr : I<0x48, MRMSrcReg,       // if signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovs\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rr : I<0x48, MRMSrcReg,       // if signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovs\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rr: I<0x49, MRMSrcReg,       // if !signed, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovns\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rr: I<0x49, MRMSrcReg,       // if !signed, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovns\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rr : I<0x4A, MRMSrcReg,       // if parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovp\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rr : I<0x4A, MRMSrcReg,       // if parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovp\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rr : I<0x4B, MRMSrcReg,       // if !parity, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovnp\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rr : I<0x4B, MRMSrcReg,       // if !parity, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovnp\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rr : I<0x40, MRMSrcReg,       // if overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovo\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rr : I<0x40, MRMSrcReg,       // if overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovo\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rr : I<0x41, MRMSrcReg,       // if !overflow, GR16 = GR16
                  (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                  "cmovno\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, GR16:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rr : I<0x41, MRMSrcReg,       // if !overflow, GR32 = GR32
                  (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "cmovno\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, GR32:$src2,
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // isCommutable = 1

def CMOVB16rm : I<0x42, MRMSrcMem,       // if <u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovb\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                  TB, OpSize;
def CMOVB32rm : I<0x42, MRMSrcMem,       // if <u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovb\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_B, EFLAGS))]>,
                   TB;
def CMOVAE16rm: I<0x43, MRMSrcMem,       // if >=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovae\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB, OpSize;
def CMOVAE32rm: I<0x43, MRMSrcMem,       // if >=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovae\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_AE, EFLAGS))]>,
                   TB;
def CMOVE16rm : I<0x44, MRMSrcMem,       // if ==, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmove\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB, OpSize;
def CMOVE32rm : I<0x44, MRMSrcMem,       // if ==, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmove\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_E, EFLAGS))]>,
                   TB;
def CMOVNE16rm: I<0x45, MRMSrcMem,       // if !=, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovne\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB, OpSize;
def CMOVNE32rm: I<0x45, MRMSrcMem,       // if !=, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovne\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NE, EFLAGS))]>,
                   TB;
def CMOVBE16rm: I<0x46, MRMSrcMem,       // if <=u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovbe\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB, OpSize;
def CMOVBE32rm: I<0x46, MRMSrcMem,       // if <=u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovbe\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_BE, EFLAGS))]>,
                   TB;
def CMOVA16rm : I<0x47, MRMSrcMem,       // if >u, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmova\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB, OpSize;
def CMOVA32rm : I<0x47, MRMSrcMem,       // if >u, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmova\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_A, EFLAGS))]>,
                   TB;
def CMOVL16rm : I<0x4C, MRMSrcMem,       // if <s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovl\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB, OpSize;
def CMOVL32rm : I<0x4C, MRMSrcMem,       // if <s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovl\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_L, EFLAGS))]>,
                   TB;
def CMOVGE16rm: I<0x4D, MRMSrcMem,       // if >=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovge\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB, OpSize;
def CMOVGE32rm: I<0x4D, MRMSrcMem,       // if >=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovge\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_GE, EFLAGS))]>,
                   TB;
def CMOVLE16rm: I<0x4E, MRMSrcMem,       // if <=s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovle\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB, OpSize;
def CMOVLE32rm: I<0x4E, MRMSrcMem,       // if <=s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovle\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_LE, EFLAGS))]>,
                   TB;
def CMOVG16rm : I<0x4F, MRMSrcMem,       // if >s, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovg\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB, OpSize;
def CMOVG32rm : I<0x4F, MRMSrcMem,       // if >s, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovg\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_G, EFLAGS))]>,
                   TB;
def CMOVS16rm : I<0x48, MRMSrcMem,       // if signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovs\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB, OpSize;
def CMOVS32rm : I<0x48, MRMSrcMem,       // if signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovs\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_S, EFLAGS))]>,
                  TB;
def CMOVNS16rm: I<0x49, MRMSrcMem,       // if !signed, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovns\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB, OpSize;
def CMOVNS32rm: I<0x49, MRMSrcMem,       // if !signed, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovns\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_NS, EFLAGS))]>,
                  TB;
def CMOVP16rm : I<0x4A, MRMSrcMem,       // if parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovp\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB, OpSize;
def CMOVP32rm : I<0x4A, MRMSrcMem,       // if parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovp\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_P, EFLAGS))]>,
                  TB;
def CMOVNP16rm : I<0x4B, MRMSrcMem,       // if !parity, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovnp\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB, OpSize;
def CMOVNP32rm : I<0x4B, MRMSrcMem,       // if !parity, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovnp\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NP, EFLAGS))]>,
                  TB;
def CMOVO16rm : I<0x40, MRMSrcMem,       // if overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovo\t{$src2, $dst|$dst, $src2}",
                  [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB, OpSize;
def CMOVO32rm : I<0x40, MRMSrcMem,       // if overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovo\t{$src2, $dst|$dst, $src2}",
                  [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                   X86_COND_O, EFLAGS))]>,
                  TB;
def CMOVNO16rm : I<0x41, MRMSrcMem,       // if !overflow, GR16 = [mem16]
                  (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                  "cmovno\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB, OpSize;
def CMOVNO32rm : I<0x41, MRMSrcMem,       // if !overflow, GR32 = [mem32]
                  (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                  "cmovno\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                    X86_COND_NO, EFLAGS))]>,
                  TB;
} // Uses = [EFLAGS]


// unary instructions
let CodeSize = 2 in {
let Defs = [EFLAGS] in {
def NEG8r  : I<0xF6, MRM3r, (outs GR8 :$dst), (ins GR8 :$src), "neg{b}\t$dst",
               [(set GR8:$dst, (ineg GR8:$src))]>;
def NEG16r : I<0xF7, MRM3r, (outs GR16:$dst), (ins GR16:$src), "neg{w}\t$dst",
               [(set GR16:$dst, (ineg GR16:$src))]>, OpSize;
def NEG32r : I<0xF7, MRM3r, (outs GR32:$dst), (ins GR32:$src), "neg{l}\t$dst",
               [(set GR32:$dst, (ineg GR32:$src))]>;
let isTwoAddress = 0 in {
  def NEG8m  : I<0xF6, MRM3m, (outs), (ins i8mem :$dst), "neg{b}\t$dst",
                 [(store (ineg (loadi8 addr:$dst)), addr:$dst)]>;
  def NEG16m : I<0xF7, MRM3m, (outs), (ins i16mem:$dst), "neg{w}\t$dst",
                 [(store (ineg (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
  def NEG32m : I<0xF7, MRM3m, (outs), (ins i32mem:$dst), "neg{l}\t$dst",
                 [(store (ineg (loadi32 addr:$dst)), addr:$dst)]>;

}
} // Defs = [EFLAGS]

// Match xor -1 to not. Favors these over a move imm + xor to save code size.
let AddedComplexity = 15 in {
def NOT8r  : I<0xF6, MRM2r, (outs GR8 :$dst), (ins GR8 :$src), "not{b}\t$dst",
               [(set GR8:$dst, (not GR8:$src))]>;
def NOT16r : I<0xF7, MRM2r, (outs GR16:$dst), (ins GR16:$src), "not{w}\t$dst",
               [(set GR16:$dst, (not GR16:$src))]>, OpSize;
def NOT32r : I<0xF7, MRM2r, (outs GR32:$dst), (ins GR32:$src), "not{l}\t$dst",
               [(set GR32:$dst, (not GR32:$src))]>;
}
let isTwoAddress = 0 in {
  def NOT8m  : I<0xF6, MRM2m, (outs), (ins i8mem :$dst), "not{b}\t$dst",
                 [(store (not (loadi8 addr:$dst)), addr:$dst)]>;
  def NOT16m : I<0xF7, MRM2m, (outs), (ins i16mem:$dst), "not{w}\t$dst",
                 [(store (not (loadi16 addr:$dst)), addr:$dst)]>, OpSize;
  def NOT32m : I<0xF7, MRM2m, (outs), (ins i32mem:$dst), "not{l}\t$dst",
                 [(store (not (loadi32 addr:$dst)), addr:$dst)]>;
}
} // CodeSize

// TODO: inc/dec is slow for P4, but fast for Pentium-M.
let Defs = [EFLAGS] in {
let CodeSize = 2 in
def INC8r  : I<0xFE, MRM0r, (outs GR8 :$dst), (ins GR8 :$src), "inc{b}\t$dst",
               [(set GR8:$dst, (add GR8:$src, 1))]>;
let isConvertibleToThreeAddress = 1, CodeSize = 1 in {  // Can xform into LEA.
def INC16r : I<0x40, AddRegFrm, (outs GR16:$dst), (ins GR16:$src), "inc{w}\t$dst",
               [(set GR16:$dst, (add GR16:$src, 1))]>,
             OpSize, Requires<[In32BitMode]>;
def INC32r : I<0x40, AddRegFrm, (outs GR32:$dst), (ins GR32:$src), "inc{l}\t$dst",
               [(set GR32:$dst, (add GR32:$src, 1))]>, Requires<[In32BitMode]>;
}
let isTwoAddress = 0, CodeSize = 2 in {
  def INC8m  : I<0xFE, MRM0m, (outs), (ins i8mem :$dst), "inc{b}\t$dst",
               [(store (add (loadi8 addr:$dst), 1), addr:$dst)]>;
  def INC16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst), "inc{w}\t$dst",
               [(store (add (loadi16 addr:$dst), 1), addr:$dst)]>,
               OpSize, Requires<[In32BitMode]>;
  def INC32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst), "inc{l}\t$dst",
               [(store (add (loadi32 addr:$dst), 1), addr:$dst)]>,
               Requires<[In32BitMode]>;
}

let CodeSize = 2 in
def DEC8r  : I<0xFE, MRM1r, (outs GR8 :$dst), (ins GR8 :$src), "dec{b}\t$dst",
               [(set GR8:$dst, (add GR8:$src, -1))]>;
let isConvertibleToThreeAddress = 1, CodeSize = 1 in {   // Can xform into LEA.
def DEC16r : I<0x48, AddRegFrm, (outs GR16:$dst), (ins GR16:$src), "dec{w}\t$dst",
               [(set GR16:$dst, (add GR16:$src, -1))]>,
             OpSize, Requires<[In32BitMode]>;
def DEC32r : I<0x48, AddRegFrm, (outs GR32:$dst), (ins GR32:$src), "dec{l}\t$dst",
               [(set GR32:$dst, (add GR32:$src, -1))]>, Requires<[In32BitMode]>;
}

let isTwoAddress = 0, CodeSize = 2 in {
  def DEC8m  : I<0xFE, MRM1m, (outs), (ins i8mem :$dst), "dec{b}\t$dst",
               [(store (add (loadi8 addr:$dst), -1), addr:$dst)]>;
  def DEC16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst), "dec{w}\t$dst",
               [(store (add (loadi16 addr:$dst), -1), addr:$dst)]>,
               OpSize, Requires<[In32BitMode]>;
  def DEC32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst), "dec{l}\t$dst",
               [(store (add (loadi32 addr:$dst), -1), addr:$dst)]>,
               Requires<[In32BitMode]>;
}
} // Defs = [EFLAGS]

// Logical operators...
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // X = AND Y, Z   --> X = AND Z, Y
def AND8rr   : I<0x20, MRMDestReg,
                (outs GR8 :$dst), (ins GR8 :$src1, GR8 :$src2),
                "and{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (and GR8:$src1, GR8:$src2))]>;
def AND16rr  : I<0x21, MRMDestReg,
                 (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                 "and{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (and GR16:$src1, GR16:$src2))]>, OpSize;
def AND32rr  : I<0x21, MRMDestReg, 
                 (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                 "and{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (and GR32:$src1, GR32:$src2))]>;
}

def AND8rm   : I<0x22, MRMSrcMem, 
                 (outs GR8 :$dst), (ins GR8 :$src1, i8mem :$src2),
                 "and{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (and GR8:$src1, (load addr:$src2)))]>;
def AND16rm  : I<0x23, MRMSrcMem, 
                 (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                 "and{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, (and GR16:$src1, (load addr:$src2)))]>, OpSize;
def AND32rm  : I<0x23, MRMSrcMem,
                 (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "and{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, (and GR32:$src1, (load addr:$src2)))]>;

def AND8ri   : Ii8<0x80, MRM4r, 
                   (outs GR8 :$dst), (ins GR8 :$src1, i8imm :$src2),
                   "and{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (and GR8:$src1, imm:$src2))]>;
def AND16ri  : Ii16<0x81, MRM4r, 
                    (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                    "and{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (and GR16:$src1, imm:$src2))]>, OpSize;
def AND32ri  : Ii32<0x81, MRM4r, 
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "and{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (and GR32:$src1, imm:$src2))]>;
def AND16ri8 : Ii8<0x83, MRM4r, 
                   (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                   "and{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (and GR16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def AND32ri8 : Ii8<0x83, MRM4r, 
                   (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "and{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (and GR32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def AND8mr   : I<0x20, MRMDestMem,
                   (outs), (ins i8mem :$dst, GR8 :$src),
                   "and{b}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR8:$src), addr:$dst)]>;
  def AND16mr  : I<0x21, MRMDestMem,
                   (outs), (ins i16mem:$dst, GR16:$src),
                   "and{w}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR16:$src), addr:$dst)]>,
                   OpSize;
  def AND32mr  : I<0x21, MRMDestMem,
                   (outs), (ins i32mem:$dst, GR32:$src),
                   "and{l}\t{$src, $dst|$dst, $src}",
                   [(store (and (load addr:$dst), GR32:$src), addr:$dst)]>;
  def AND8mi   : Ii8<0x80, MRM4m,
                     (outs), (ins i8mem :$dst, i8imm :$src),
                     "and{b}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def AND16mi  : Ii16<0x81, MRM4m,
                      (outs), (ins i16mem:$dst, i16imm:$src),
                      "and{w}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                      OpSize;
  def AND32mi  : Ii32<0x81, MRM4m,
                      (outs), (ins i32mem:$dst, i32imm:$src),
                      "and{l}\t{$src, $dst|$dst, $src}",
                      [(store (and (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def AND16mi8 : Ii8<0x83, MRM4m,
                     (outs), (ins i16mem:$dst, i16i8imm :$src),
                     "and{w}\t{$src, $dst|$dst, $src}",
                [(store (and (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def AND32mi8 : Ii8<0x83, MRM4m,
                     (outs), (ins i32mem:$dst, i32i8imm :$src),
                     "and{l}\t{$src, $dst|$dst, $src}",
                [(store (and (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
}


let isCommutable = 1 in {   // X = OR Y, Z   --> X = OR Z, Y
def OR8rr    : I<0x08, MRMDestReg, (outs GR8 :$dst), (ins GR8 :$src1, GR8 :$src2),
                 "or{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (or GR8:$src1, GR8:$src2))]>;
def OR16rr   : I<0x09, MRMDestReg, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                 "or{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (or GR16:$src1, GR16:$src2))]>, OpSize;
def OR32rr   : I<0x09, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                 "or{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (or GR32:$src1, GR32:$src2))]>;
}
def OR8rm    : I<0x0A, MRMSrcMem , (outs GR8 :$dst), (ins GR8 :$src1, i8mem :$src2),
                 "or{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (or GR8:$src1, (load addr:$src2)))]>;
def OR16rm   : I<0x0B, MRMSrcMem , (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2),
                 "or{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, (or GR16:$src1, (load addr:$src2)))]>, OpSize;
def OR32rm   : I<0x0B, MRMSrcMem , (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "or{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, (or GR32:$src1, (load addr:$src2)))]>;

def OR8ri    : Ii8 <0x80, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                    "or{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, (or GR8:$src1, imm:$src2))]>;
def OR16ri   : Ii16<0x81, MRM1r, (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                    "or{w}\t{$src2, $dst|$dst, $src2}", 
                    [(set GR16:$dst, (or GR16:$src1, imm:$src2))]>, OpSize;
def OR32ri   : Ii32<0x81, MRM1r, (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "or{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (or GR32:$src1, imm:$src2))]>;

def OR16ri8  : Ii8<0x83, MRM1r, (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                   "or{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (or GR16:$src1, i16immSExt8:$src2))]>, OpSize;
def OR32ri8  : Ii8<0x83, MRM1r, (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "or{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (or GR32:$src1, i32immSExt8:$src2))]>;
let isTwoAddress = 0 in {
  def OR8mr  : I<0x08, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src),
                 "or{b}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR8:$src), addr:$dst)]>;
  def OR16mr : I<0x09, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR16:$src), addr:$dst)]>, OpSize;
  def OR32mr : I<0x09, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), GR32:$src), addr:$dst)]>;
  def OR8mi    : Ii8<0x80, MRM1m, (outs), (ins i8mem :$dst, i8imm:$src),
                 "or{b}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def OR16mi   : Ii16<0x81, MRM1m, (outs), (ins i16mem:$dst, i16imm:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                 OpSize;
  def OR32mi   : Ii32<0x81, MRM1m, (outs), (ins i32mem:$dst, i32imm:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def OR16mi8  : Ii8<0x83, MRM1m, (outs), (ins i16mem:$dst, i16i8imm:$src),
                 "or{w}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def OR32mi8  : Ii8<0x83, MRM1m, (outs), (ins i32mem:$dst, i32i8imm:$src),
                 "or{l}\t{$src, $dst|$dst, $src}",
                 [(store (or (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
} // isTwoAddress = 0


let isCommutable = 1 in { // X = XOR Y, Z --> X = XOR Z, Y
  def XOR8rr   : I<0x30, MRMDestReg,
                   (outs GR8 :$dst), (ins GR8 :$src1, GR8 :$src2),
                   "xor{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (xor GR8:$src1, GR8:$src2))]>;
  def XOR16rr  : I<0x31, MRMDestReg, 
                   (outs GR16:$dst), (ins GR16:$src1, GR16:$src2), 
                   "xor{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (xor GR16:$src1, GR16:$src2))]>, OpSize;
  def XOR32rr  : I<0x31, MRMDestReg, 
                   (outs GR32:$dst), (ins GR32:$src1, GR32:$src2), 
                   "xor{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (xor GR32:$src1, GR32:$src2))]>;
} // isCommutable = 1

def XOR8rm   : I<0x32, MRMSrcMem , 
                 (outs GR8 :$dst), (ins GR8:$src1, i8mem :$src2), 
                 "xor{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (xor GR8:$src1, (load addr:$src2)))]>;
def XOR16rm  : I<0x33, MRMSrcMem , 
                 (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2), 
                 "xor{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (xor GR16:$src1, (load addr:$src2)))]>,
                 OpSize;
def XOR32rm  : I<0x33, MRMSrcMem , 
                 (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2), 
                 "xor{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (xor GR32:$src1, (load addr:$src2)))]>;

def XOR8ri   : Ii8<0x80, MRM6r, 
                   (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2), 
                   "xor{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (xor GR8:$src1, imm:$src2))]>;
def XOR16ri  : Ii16<0x81, MRM6r, 
                    (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2), 
                    "xor{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (xor GR16:$src1, imm:$src2))]>, OpSize;
def XOR32ri  : Ii32<0x81, MRM6r, 
                    (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2), 
                    "xor{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (xor GR32:$src1, imm:$src2))]>;
def XOR16ri8 : Ii8<0x83, MRM6r, 
                   (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                   "xor{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (xor GR16:$src1, i16immSExt8:$src2))]>,
                   OpSize;
def XOR32ri8 : Ii8<0x83, MRM6r, 
                   (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "xor{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (xor GR32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def XOR8mr   : I<0x30, MRMDestMem,
                   (outs), (ins i8mem :$dst, GR8 :$src),
                   "xor{b}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR8:$src), addr:$dst)]>;
  def XOR16mr  : I<0x31, MRMDestMem,
                   (outs), (ins i16mem:$dst, GR16:$src),
                   "xor{w}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR16:$src), addr:$dst)]>,
                   OpSize;
  def XOR32mr  : I<0x31, MRMDestMem,
                   (outs), (ins i32mem:$dst, GR32:$src),
                   "xor{l}\t{$src, $dst|$dst, $src}",
                   [(store (xor (load addr:$dst), GR32:$src), addr:$dst)]>;
  def XOR8mi   : Ii8<0x80, MRM6m,
                     (outs), (ins i8mem :$dst, i8imm :$src),
                     "xor{b}\t{$src, $dst|$dst, $src}",
                    [(store (xor (loadi8 addr:$dst), imm:$src), addr:$dst)]>;
  def XOR16mi  : Ii16<0x81, MRM6m,
                      (outs), (ins i16mem:$dst, i16imm:$src),
                      "xor{w}\t{$src, $dst|$dst, $src}",
                   [(store (xor (loadi16 addr:$dst), imm:$src), addr:$dst)]>,
                      OpSize;
  def XOR32mi  : Ii32<0x81, MRM6m,
                      (outs), (ins i32mem:$dst, i32imm:$src),
                      "xor{l}\t{$src, $dst|$dst, $src}",
                   [(store (xor (loadi32 addr:$dst), imm:$src), addr:$dst)]>;
  def XOR16mi8 : Ii8<0x83, MRM6m,
                     (outs), (ins i16mem:$dst, i16i8imm :$src),
                     "xor{w}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i16immSExt8:$src), addr:$dst)]>,
                     OpSize;
  def XOR32mi8 : Ii8<0x83, MRM6m,
                     (outs), (ins i32mem:$dst, i32i8imm :$src),
                     "xor{l}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i32immSExt8:$src), addr:$dst)]>;
} // isTwoAddress = 0
} // Defs = [EFLAGS]

// Shift instructions
let Defs = [EFLAGS] in {
let Uses = [CL] in {
def SHL8rCL  : I<0xD2, MRM4r, (outs GR8 :$dst), (ins GR8 :$src),
                 "shl{b}\t{%cl, $dst|$dst, %CL}",
                 [(set GR8:$dst, (shl GR8:$src, CL))]>;
def SHL16rCL : I<0xD3, MRM4r, (outs GR16:$dst), (ins GR16:$src),
                 "shl{w}\t{%cl, $dst|$dst, %CL}",
                 [(set GR16:$dst, (shl GR16:$src, CL))]>, OpSize;
def SHL32rCL : I<0xD3, MRM4r, (outs GR32:$dst), (ins GR32:$src),
                 "shl{l}\t{%cl, $dst|$dst, %CL}",
                 [(set GR32:$dst, (shl GR32:$src, CL))]>;
} // Uses = [CL]

def SHL8ri   : Ii8<0xC0, MRM4r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "shl{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (shl GR8:$src1, (i8 imm:$src2)))]>;
let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
def SHL16ri  : Ii8<0xC1, MRM4r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "shl{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (shl GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHL32ri  : Ii8<0xC1, MRM4r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "shl{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (shl GR32:$src1, (i8 imm:$src2)))]>;
// NOTE: We don't use shifts of a register by one, because 'add reg,reg' is
// cheaper.
} // isConvertibleToThreeAddress = 1

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def SHL8mCL  : I<0xD2, MRM4m, (outs), (ins i8mem :$dst),
                   "shl{b}\t{%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SHL16mCL : I<0xD3, MRM4m, (outs), (ins i16mem:$dst),
                   "shl{w}\t{%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def SHL32mCL : I<0xD3, MRM4m, (outs), (ins i32mem:$dst),
                   "shl{l}\t{%cl, $dst|$dst, %CL}",
                   [(store (shl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SHL8mi   : Ii8<0xC0, MRM4m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "shl{b}\t{$src, $dst|$dst, $src}",
                  [(store (shl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHL16mi  : Ii8<0xC1, MRM4m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "shl{w}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHL32mi  : Ii8<0xC1, MRM4m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "shl{l}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SHL8m1   : I<0xD0, MRM4m, (outs), (ins i8mem :$dst),
                   "shl{b}\t$dst",
                  [(store (shl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SHL16m1  : I<0xD1, MRM4m, (outs), (ins i16mem:$dst),
                   "shl{w}\t$dst",
                 [(store (shl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def SHL32m1  : I<0xD1, MRM4m, (outs), (ins i32mem:$dst),
                   "shl{l}\t$dst",
                 [(store (shl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
}

let Uses = [CL] in {
def SHR8rCL  : I<0xD2, MRM5r, (outs GR8 :$dst), (ins GR8 :$src),
                 "shr{b}\t{%cl, $dst|$dst, %CL}",
                 [(set GR8:$dst, (srl GR8:$src, CL))]>;
def SHR16rCL : I<0xD3, MRM5r, (outs GR16:$dst), (ins GR16:$src),
                 "shr{w}\t{%cl, $dst|$dst, %CL}",
                 [(set GR16:$dst, (srl GR16:$src, CL))]>, OpSize;
def SHR32rCL : I<0xD3, MRM5r, (outs GR32:$dst), (ins GR32:$src),
                 "shr{l}\t{%cl, $dst|$dst, %CL}",
                 [(set GR32:$dst, (srl GR32:$src, CL))]>;
}

def SHR8ri   : Ii8<0xC0, MRM5r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                   "shr{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (srl GR8:$src1, (i8 imm:$src2)))]>;
def SHR16ri  : Ii8<0xC1, MRM5r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "shr{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (srl GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def SHR32ri  : Ii8<0xC1, MRM5r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "shr{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (srl GR32:$src1, (i8 imm:$src2)))]>;

// Shift by 1
def SHR8r1   : I<0xD0, MRM5r, (outs GR8:$dst), (ins GR8:$src1),
                 "shr{b}\t$dst",
                 [(set GR8:$dst, (srl GR8:$src1, (i8 1)))]>;
def SHR16r1  : I<0xD1, MRM5r, (outs GR16:$dst), (ins GR16:$src1),
                 "shr{w}\t$dst",
                 [(set GR16:$dst, (srl GR16:$src1, (i8 1)))]>, OpSize;
def SHR32r1  : I<0xD1, MRM5r, (outs GR32:$dst), (ins GR32:$src1),
                 "shr{l}\t$dst",
                 [(set GR32:$dst, (srl GR32:$src1, (i8 1)))]>;

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def SHR8mCL  : I<0xD2, MRM5m, (outs), (ins i8mem :$dst),
                   "shr{b}\t{%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SHR16mCL : I<0xD3, MRM5m, (outs), (ins i16mem:$dst),
                   "shr{w}\t{%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi16 addr:$dst), CL), addr:$dst)]>,
                   OpSize;
  def SHR32mCL : I<0xD3, MRM5m, (outs), (ins i32mem:$dst),
                   "shr{l}\t{%cl, $dst|$dst, %CL}",
                   [(store (srl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SHR8mi   : Ii8<0xC0, MRM5m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "shr{b}\t{$src, $dst|$dst, $src}",
                  [(store (srl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SHR16mi  : Ii8<0xC1, MRM5m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "shr{w}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SHR32mi  : Ii8<0xC1, MRM5m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "shr{l}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SHR8m1   : I<0xD0, MRM5m, (outs), (ins i8mem :$dst),
                   "shr{b}\t$dst",
                  [(store (srl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SHR16m1  : I<0xD1, MRM5m, (outs), (ins i16mem:$dst),
                   "shr{w}\t$dst",
                 [(store (srl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,OpSize;
  def SHR32m1  : I<0xD1, MRM5m, (outs), (ins i32mem:$dst),
                   "shr{l}\t$dst",
                 [(store (srl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
}

let Uses = [CL] in {
def SAR8rCL  : I<0xD2, MRM7r, (outs GR8 :$dst), (ins GR8 :$src),
                 "sar{b}\t{%cl, $dst|$dst, %CL}",
                 [(set GR8:$dst, (sra GR8:$src, CL))]>;
def SAR16rCL : I<0xD3, MRM7r, (outs GR16:$dst), (ins GR16:$src),
                 "sar{w}\t{%cl, $dst|$dst, %CL}",
                 [(set GR16:$dst, (sra GR16:$src, CL))]>, OpSize;
def SAR32rCL : I<0xD3, MRM7r, (outs GR32:$dst), (ins GR32:$src),
                 "sar{l}\t{%cl, $dst|$dst, %CL}",
                 [(set GR32:$dst, (sra GR32:$src, CL))]>;
}

def SAR8ri   : Ii8<0xC0, MRM7r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "sar{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (sra GR8:$src1, (i8 imm:$src2)))]>;
def SAR16ri  : Ii8<0xC1, MRM7r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "sar{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (sra GR16:$src1, (i8 imm:$src2)))]>,
                   OpSize;
def SAR32ri  : Ii8<0xC1, MRM7r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "sar{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (sra GR32:$src1, (i8 imm:$src2)))]>;

// Shift by 1
def SAR8r1   : I<0xD0, MRM7r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "sar{b}\t$dst",
                 [(set GR8:$dst, (sra GR8:$src1, (i8 1)))]>;
def SAR16r1  : I<0xD1, MRM7r, (outs GR16:$dst), (ins GR16:$src1),
                 "sar{w}\t$dst",
                 [(set GR16:$dst, (sra GR16:$src1, (i8 1)))]>, OpSize;
def SAR32r1  : I<0xD1, MRM7r, (outs GR32:$dst), (ins GR32:$src1),
                 "sar{l}\t$dst",
                 [(set GR32:$dst, (sra GR32:$src1, (i8 1)))]>;

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def SAR8mCL  : I<0xD2, MRM7m, (outs), (ins i8mem :$dst),
                   "sar{b}\t{%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi8 addr:$dst), CL), addr:$dst)]>;
  def SAR16mCL : I<0xD3, MRM7m, (outs), (ins i16mem:$dst),
                   "sar{w}\t{%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def SAR32mCL : I<0xD3, MRM7m, (outs), (ins i32mem:$dst), 
                   "sar{l}\t{%cl, $dst|$dst, %CL}",
                   [(store (sra (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def SAR8mi   : Ii8<0xC0, MRM7m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "sar{b}\t{$src, $dst|$dst, $src}",
                  [(store (sra (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def SAR16mi  : Ii8<0xC1, MRM7m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "sar{w}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def SAR32mi  : Ii8<0xC1, MRM7m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "sar{l}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Shift by 1
  def SAR8m1   : I<0xD0, MRM7m, (outs), (ins i8mem :$dst),
                   "sar{b}\t$dst",
                  [(store (sra (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def SAR16m1  : I<0xD1, MRM7m, (outs), (ins i16mem:$dst),
                   "sar{w}\t$dst",
                 [(store (sra (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def SAR32m1  : I<0xD1, MRM7m, (outs), (ins i32mem:$dst),
                   "sar{l}\t$dst",
                 [(store (sra (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
}

// Rotate instructions
// FIXME: provide shorter instructions when imm8 == 1
let Uses = [CL] in {
def ROL8rCL  : I<0xD2, MRM0r, (outs GR8 :$dst), (ins GR8 :$src),
                 "rol{b}\t{%cl, $dst|$dst, %CL}",
                 [(set GR8:$dst, (rotl GR8:$src, CL))]>;
def ROL16rCL : I<0xD3, MRM0r, (outs GR16:$dst), (ins GR16:$src),
                 "rol{w}\t{%cl, $dst|$dst, %CL}",
                 [(set GR16:$dst, (rotl GR16:$src, CL))]>, OpSize;
def ROL32rCL : I<0xD3, MRM0r, (outs GR32:$dst), (ins GR32:$src),
                 "rol{l}\t{%cl, $dst|$dst, %CL}",
                 [(set GR32:$dst, (rotl GR32:$src, CL))]>;
}

def ROL8ri   : Ii8<0xC0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "rol{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotl GR8:$src1, (i8 imm:$src2)))]>;
def ROL16ri  : Ii8<0xC1, MRM0r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "rol{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotl GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def ROL32ri  : Ii8<0xC1, MRM0r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "rol{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotl GR32:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROL8r1   : I<0xD0, MRM0r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "rol{b}\t$dst",
                 [(set GR8:$dst, (rotl GR8:$src1, (i8 1)))]>;
def ROL16r1  : I<0xD1, MRM0r, (outs GR16:$dst), (ins GR16:$src1),
                 "rol{w}\t$dst",
                 [(set GR16:$dst, (rotl GR16:$src1, (i8 1)))]>, OpSize;
def ROL32r1  : I<0xD1, MRM0r, (outs GR32:$dst), (ins GR32:$src1),
                 "rol{l}\t$dst",
                 [(set GR32:$dst, (rotl GR32:$src1, (i8 1)))]>;

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def ROL8mCL  : I<0xD2, MRM0m, (outs), (ins i8mem :$dst),
                   "rol{b}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi8 addr:$dst), CL), addr:$dst)]>;
  def ROL16mCL : I<0xD3, MRM0m, (outs), (ins i16mem:$dst),
                   "rol{w}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def ROL32mCL : I<0xD3, MRM0m, (outs), (ins i32mem:$dst),
                   "rol{l}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotl (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def ROL8mi   : Ii8<0xC0, MRM0m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "rol{b}\t{$src, $dst|$dst, $src}",
                 [(store (rotl (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROL16mi  : Ii8<0xC1, MRM0m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "rol{w}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROL32mi  : Ii8<0xC1, MRM0m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "rol{l}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Rotate by 1
  def ROL8m1   : I<0xD0, MRM0m, (outs), (ins i8mem :$dst),
                   "rol{b}\t$dst",
                 [(store (rotl (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def ROL16m1  : I<0xD1, MRM0m, (outs), (ins i16mem:$dst),
                   "rol{w}\t$dst",
                [(store (rotl (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def ROL32m1  : I<0xD1, MRM0m, (outs), (ins i32mem:$dst),
                   "rol{l}\t$dst",
                [(store (rotl (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
}

let Uses = [CL] in {
def ROR8rCL  : I<0xD2, MRM1r, (outs GR8 :$dst), (ins GR8 :$src),
                 "ror{b}\t{%cl, $dst|$dst, %CL}",
                 [(set GR8:$dst, (rotr GR8:$src, CL))]>;
def ROR16rCL : I<0xD3, MRM1r, (outs GR16:$dst), (ins GR16:$src),
                 "ror{w}\t{%cl, $dst|$dst, %CL}",
                 [(set GR16:$dst, (rotr GR16:$src, CL))]>, OpSize;
def ROR32rCL : I<0xD3, MRM1r, (outs GR32:$dst), (ins GR32:$src),
                 "ror{l}\t{%cl, $dst|$dst, %CL}",
                 [(set GR32:$dst, (rotr GR32:$src, CL))]>;
}

def ROR8ri   : Ii8<0xC0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1, i8imm:$src2),
                   "ror{b}\t{$src2, $dst|$dst, $src2}",
                   [(set GR8:$dst, (rotr GR8:$src1, (i8 imm:$src2)))]>;
def ROR16ri  : Ii8<0xC1, MRM1r, (outs GR16:$dst), (ins GR16:$src1, i8imm:$src2),
                   "ror{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (rotr GR16:$src1, (i8 imm:$src2)))]>, OpSize;
def ROR32ri  : Ii8<0xC1, MRM1r, (outs GR32:$dst), (ins GR32:$src1, i8imm:$src2),
                   "ror{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (rotr GR32:$src1, (i8 imm:$src2)))]>;

// Rotate by 1
def ROR8r1   : I<0xD0, MRM1r, (outs GR8 :$dst), (ins GR8 :$src1),
                 "ror{b}\t$dst",
                 [(set GR8:$dst, (rotr GR8:$src1, (i8 1)))]>;
def ROR16r1  : I<0xD1, MRM1r, (outs GR16:$dst), (ins GR16:$src1),
                 "ror{w}\t$dst",
                 [(set GR16:$dst, (rotr GR16:$src1, (i8 1)))]>, OpSize;
def ROR32r1  : I<0xD1, MRM1r, (outs GR32:$dst), (ins GR32:$src1),
                 "ror{l}\t$dst",
                 [(set GR32:$dst, (rotr GR32:$src1, (i8 1)))]>;

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def ROR8mCL  : I<0xD2, MRM1m, (outs), (ins i8mem :$dst),
                   "ror{b}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi8 addr:$dst), CL), addr:$dst)]>;
  def ROR16mCL : I<0xD3, MRM1m, (outs), (ins i16mem:$dst),
                   "ror{w}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi16 addr:$dst), CL), addr:$dst)]>, OpSize;
  def ROR32mCL : I<0xD3, MRM1m, (outs), (ins i32mem:$dst), 
                   "ror{l}\t{%cl, $dst|$dst, %CL}",
                   [(store (rotr (loadi32 addr:$dst), CL), addr:$dst)]>;
  }
  def ROR8mi   : Ii8<0xC0, MRM1m, (outs), (ins i8mem :$dst, i8imm:$src),
                     "ror{b}\t{$src, $dst|$dst, $src}",
                 [(store (rotr (loadi8 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
  def ROR16mi  : Ii8<0xC1, MRM1m, (outs), (ins i16mem:$dst, i8imm:$src),
                     "ror{w}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi16 addr:$dst), (i8 imm:$src)), addr:$dst)]>,
                     OpSize;
  def ROR32mi  : Ii8<0xC1, MRM1m, (outs), (ins i32mem:$dst, i8imm:$src),
                     "ror{l}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi32 addr:$dst), (i8 imm:$src)), addr:$dst)]>;

  // Rotate by 1
  def ROR8m1   : I<0xD0, MRM1m, (outs), (ins i8mem :$dst),
                   "ror{b}\t$dst",
                 [(store (rotr (loadi8 addr:$dst), (i8 1)), addr:$dst)]>;
  def ROR16m1  : I<0xD1, MRM1m, (outs), (ins i16mem:$dst),
                   "ror{w}\t$dst",
                [(store (rotr (loadi16 addr:$dst), (i8 1)), addr:$dst)]>,
                     OpSize;
  def ROR32m1  : I<0xD1, MRM1m, (outs), (ins i32mem:$dst),
                   "ror{l}\t$dst",
                [(store (rotr (loadi32 addr:$dst), (i8 1)), addr:$dst)]>;
}



// Double shift instructions (generalizations of rotate)
let Uses = [CL] in {
def SHLD32rrCL : I<0xA5, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                   "shld{l}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set GR32:$dst, (X86shld GR32:$src1, GR32:$src2, CL))]>, TB;
def SHRD32rrCL : I<0xAD, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                   "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set GR32:$dst, (X86shrd GR32:$src1, GR32:$src2, CL))]>, TB;
def SHLD16rrCL : I<0xA5, MRMDestReg, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                   "shld{w}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set GR16:$dst, (X86shld GR16:$src1, GR16:$src2, CL))]>,
                   TB, OpSize;
def SHRD16rrCL : I<0xAD, MRMDestReg, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2),
                   "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                   [(set GR16:$dst, (X86shrd GR16:$src1, GR16:$src2, CL))]>,
                   TB, OpSize;
}

let isCommutable = 1 in {  // These instructions commute to each other.
def SHLD32rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$src3),
                     "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (X86shld GR32:$src1, GR32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHRD32rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, i8imm:$src3),
                     "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR32:$dst, (X86shrd GR32:$src1, GR32:$src2,
                                      (i8 imm:$src3)))]>,
                 TB;
def SHLD16rri8 : Ii8<0xA4, MRMDestReg,
                     (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, i8imm:$src3),
                     "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86shld GR16:$src1, GR16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
def SHRD16rri8 : Ii8<0xAC, MRMDestReg,
                     (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, i8imm:$src3),
                     "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                     [(set GR16:$dst, (X86shrd GR16:$src1, GR16:$src2,
                                      (i8 imm:$src3)))]>,
                     TB, OpSize;
}

let isTwoAddress = 0 in {
  let Uses = [CL] in {
  def SHLD32mrCL : I<0xA5, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                     "shld{l}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                     [(store (X86shld (loadi32 addr:$dst), GR32:$src2, CL),
                       addr:$dst)]>, TB;
  def SHRD32mrCL : I<0xAD, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                    "shrd{l}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi32 addr:$dst), GR32:$src2, CL),
                      addr:$dst)]>, TB;
  }
  def SHLD32mri8 : Ii8<0xA4, MRMDestMem,
                      (outs), (ins i32mem:$dst, GR32:$src2, i8imm:$src3),
                      "shld{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi32 addr:$dst), GR32:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB;
  def SHRD32mri8 : Ii8<0xAC, MRMDestMem, 
                       (outs), (ins i32mem:$dst, GR32:$src2, i8imm:$src3),
                       "shrd{l}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                       [(store (X86shrd (loadi32 addr:$dst), GR32:$src2,
                                         (i8 imm:$src3)), addr:$dst)]>,
                       TB;

  let Uses = [CL] in {
  def SHLD16mrCL : I<0xA5, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                     "shld{w}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                     [(store (X86shld (loadi16 addr:$dst), GR16:$src2, CL),
                       addr:$dst)]>, TB, OpSize;
  def SHRD16mrCL : I<0xAD, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                    "shrd{w}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi16 addr:$dst), GR16:$src2, CL),
                      addr:$dst)]>, TB, OpSize;
  }
  def SHLD16mri8 : Ii8<0xA4, MRMDestMem,
                      (outs), (ins i16mem:$dst, GR16:$src2, i8imm:$src3),
                      "shld{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi16 addr:$dst), GR16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                      TB, OpSize;
  def SHRD16mri8 : Ii8<0xAC, MRMDestMem, 
                       (outs), (ins i16mem:$dst, GR16:$src2, i8imm:$src3),
                       "shrd{w}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shrd (loadi16 addr:$dst), GR16:$src2,
                                        (i8 imm:$src3)), addr:$dst)]>,
                       TB, OpSize;
}
} // Defs = [EFLAGS]


// Arithmetic.
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // X = ADD Y, Z   --> X = ADD Z, Y
// Register-Register Addition
def ADD8rr    : I<0x00, MRMDestReg, (outs GR8 :$dst),
                                    (ins GR8 :$src1, GR8 :$src2),
                  "add{b}\t{$src2, $dst|$dst, $src2}",
                  [(set GR8:$dst, (add GR8:$src1, GR8:$src2)),
                   (implicit EFLAGS)]>;

let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
// Register-Register Addition
def ADD16rr  : I<0x01, MRMDestReg, (outs GR16:$dst),
                                   (ins GR16:$src1, GR16:$src2),
                 "add{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (add GR16:$src1, GR16:$src2)),
                  (implicit EFLAGS)]>, OpSize;
def ADD32rr  : I<0x01, MRMDestReg, (outs GR32:$dst),
                                   (ins GR32:$src1, GR32:$src2),
                 "add{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (add GR32:$src1, GR32:$src2)),
                  (implicit EFLAGS)]>;
} // end isConvertibleToThreeAddress
} // end isCommutable

// Register-Memory Addition
def ADD8rm   : I<0x02, MRMSrcMem, (outs GR8 :$dst),
                                  (ins GR8 :$src1, i8mem :$src2),
                 "add{b}\t{$src2, $dst|$dst, $src2}",
                 [(set GR8:$dst, (add GR8:$src1, (load addr:$src2))),
                  (implicit EFLAGS)]>;
def ADD16rm  : I<0x03, MRMSrcMem, (outs GR16:$dst),
                                  (ins GR16:$src1, i16mem:$src2),
                 "add{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (add GR16:$src1, (load addr:$src2))),
                  (implicit EFLAGS)]>, OpSize;
def ADD32rm  : I<0x03, MRMSrcMem, (outs GR32:$dst),
                                  (ins GR32:$src1, i32mem:$src2),
                 "add{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (add GR32:$src1, (load addr:$src2))),
                  (implicit EFLAGS)]>;

// Register-Integer Addition
def ADD8ri    : Ii8<0x80, MRM0r, (outs GR8:$dst), (ins GR8:$src1, i8imm:$src2),
                    "add{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, (add GR8:$src1, imm:$src2)),
                     (implicit EFLAGS)]>;

let isConvertibleToThreeAddress = 1 in {   // Can transform into LEA.
// Register-Integer Addition
def ADD16ri  : Ii16<0x81, MRM0r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "add{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (add GR16:$src1, imm:$src2)),
                     (implicit EFLAGS)]>, OpSize;
def ADD32ri  : Ii32<0x81, MRM0r, (outs GR32:$dst),
                                 (ins GR32:$src1, i32imm:$src2),
                    "add{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (add GR32:$src1, imm:$src2)),
                     (implicit EFLAGS)]>;
def ADD16ri8 : Ii8<0x83, MRM0r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "add{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (add GR16:$src1, i16immSExt8:$src2)),
                    (implicit EFLAGS)]>, OpSize;
def ADD32ri8 : Ii8<0x83, MRM0r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "add{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (add GR32:$src1, i32immSExt8:$src2)),
                    (implicit EFLAGS)]>;
}

let isTwoAddress = 0 in {
  // Memory-Register Addition
  def ADD8mr   : I<0x00, MRMDestMem, (outs), (ins i8mem:$dst, GR8:$src2),
                   "add{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR8:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD16mr  : I<0x01, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "add{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR16:$src2), addr:$dst),
                    (implicit EFLAGS)]>, OpSize;
  def ADD32mr  : I<0x01, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                   "add{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (load addr:$dst), GR32:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD8mi   : Ii8<0x80, MRM0m, (outs), (ins i8mem :$dst, i8imm :$src2),
                     "add{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (add (loadi8 addr:$dst), imm:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def ADD16mi  : Ii16<0x81, MRM0m, (outs), (ins i16mem:$dst, i16imm:$src2),
                      "add{w}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (loadi16 addr:$dst), imm:$src2), addr:$dst),
                   (implicit EFLAGS)]>, OpSize;
  def ADD32mi  : Ii32<0x81, MRM0m, (outs), (ins i32mem:$dst, i32imm:$src2),
                      "add{l}\t{$src2, $dst|$dst, $src2}",
                      [(store (add (loadi32 addr:$dst), imm:$src2), addr:$dst),
                       (implicit EFLAGS)]>;
  def ADD16mi8 : Ii8<0x83, MRM0m, (outs), (ins i16mem:$dst, i16i8imm :$src2),
                     "add{w}\t{$src2, $dst|$dst, $src2}",
                     [(store (add (load addr:$dst), i16immSExt8:$src2),
                                  addr:$dst),
                      (implicit EFLAGS)]>, OpSize;
  def ADD32mi8 : Ii8<0x83, MRM0m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "add{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), i32immSExt8:$src2),
                               addr:$dst),
                   (implicit EFLAGS)]>;
}

let Uses = [EFLAGS] in {
let isCommutable = 1 in {  // X = ADC Y, Z --> X = ADC Z, Y
def ADC32rr  : I<0x11, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                 "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, GR32:$src2))]>;
}
def ADC32rm  : I<0x13, MRMSrcMem , (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, (load addr:$src2)))]>;
def ADC32ri  : Ii32<0x81, MRM2r, (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, imm:$src2))]>;
def ADC32ri8 : Ii8<0x83, MRM2r, (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "adc{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (adde GR32:$src1, i32immSExt8:$src2))]>;

let isTwoAddress = 0 in {
  def ADC32mr  : I<0x11, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2),
                   "adc{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (adde (load addr:$dst), GR32:$src2), addr:$dst)]>;
  def ADC32mi  : Ii32<0x81, MRM2m, (outs), (ins i32mem:$dst, i32imm:$src2),
                      "adc{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def ADC32mi8 : Ii8<0x83, MRM2m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "adc{l}\t{$src2, $dst|$dst, $src2}",
               [(store (adde (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}
} // Uses = [EFLAGS]

// Register-Register Subtraction
def SUB8rr  : I<0x28, MRMDestReg, (outs GR8:$dst), (ins GR8:$src1, GR8:$src2),
                "sub{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (sub GR8:$src1, GR8:$src2)),
                 (implicit EFLAGS)]>;
def SUB16rr : I<0x29, MRMDestReg, (outs GR16:$dst), (ins GR16:$src1,GR16:$src2),
                "sub{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, (sub GR16:$src1, GR16:$src2)),
                 (implicit EFLAGS)]>, OpSize;
def SUB32rr : I<0x29, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1,GR32:$src2),
                "sub{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, (sub GR32:$src1, GR32:$src2)),
                 (implicit EFLAGS)]>;

// Register-Memory Subtraction
def SUB8rm  : I<0x2A, MRMSrcMem, (outs GR8 :$dst),
                                 (ins GR8 :$src1, i8mem :$src2),
                "sub{b}\t{$src2, $dst|$dst, $src2}",
                [(set GR8:$dst, (sub GR8:$src1, (load addr:$src2))),
                 (implicit EFLAGS)]>;
def SUB16rm : I<0x2B, MRMSrcMem, (outs GR16:$dst),
                                 (ins GR16:$src1, i16mem:$src2),
                "sub{w}\t{$src2, $dst|$dst, $src2}",
                [(set GR16:$dst, (sub GR16:$src1, (load addr:$src2))),
                 (implicit EFLAGS)]>, OpSize;
def SUB32rm : I<0x2B, MRMSrcMem, (outs GR32:$dst),
                                 (ins GR32:$src1, i32mem:$src2),
                "sub{l}\t{$src2, $dst|$dst, $src2}",
                [(set GR32:$dst, (sub GR32:$src1, (load addr:$src2))),
                 (implicit EFLAGS)]>;

// Register-Integer Subtraction
def SUB8ri   : Ii8 <0x80, MRM5r, (outs GR8:$dst),
                                 (ins GR8:$src1, i8imm:$src2),
                    "sub{b}\t{$src2, $dst|$dst, $src2}",
                    [(set GR8:$dst, (sub GR8:$src1, imm:$src2)),
                     (implicit EFLAGS)]>;
def SUB16ri  : Ii16<0x81, MRM5r, (outs GR16:$dst),
                                 (ins GR16:$src1, i16imm:$src2),
                    "sub{w}\t{$src2, $dst|$dst, $src2}",
                    [(set GR16:$dst, (sub GR16:$src1, imm:$src2)),
                     (implicit EFLAGS)]>, OpSize;
def SUB32ri  : Ii32<0x81, MRM5r, (outs GR32:$dst),
                                 (ins GR32:$src1, i32imm:$src2),
                    "sub{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (sub GR32:$src1, imm:$src2)),
                     (implicit EFLAGS)]>;
def SUB16ri8 : Ii8<0x83, MRM5r, (outs GR16:$dst),
                                (ins GR16:$src1, i16i8imm:$src2),
                   "sub{w}\t{$src2, $dst|$dst, $src2}",
                   [(set GR16:$dst, (sub GR16:$src1, i16immSExt8:$src2)),
                    (implicit EFLAGS)]>, OpSize;
def SUB32ri8 : Ii8<0x83, MRM5r, (outs GR32:$dst),
                                (ins GR32:$src1, i32i8imm:$src2),
                   "sub{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (sub GR32:$src1, i32immSExt8:$src2)),
                    (implicit EFLAGS)]>;

let isTwoAddress = 0 in {
  // Memory-Register Subtraction
  def SUB8mr   : I<0x28, MRMDestMem, (outs), (ins i8mem :$dst, GR8 :$src2),
                   "sub{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR8:$src2), addr:$dst),
                    (implicit EFLAGS)]>;
  def SUB16mr  : I<0x29, MRMDestMem, (outs), (ins i16mem:$dst, GR16:$src2),
                   "sub{w}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR16:$src2), addr:$dst),
                    (implicit EFLAGS)]>, OpSize;
  def SUB32mr  : I<0x29, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2), 
                   "sub{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (sub (load addr:$dst), GR32:$src2), addr:$dst),
                    (implicit EFLAGS)]>;

  // Memory-Integer Subtraction
  def SUB8mi   : Ii8<0x80, MRM5m, (outs), (ins i8mem :$dst, i8imm:$src2), 
                     "sub{b}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (loadi8 addr:$dst), imm:$src2), addr:$dst),
                      (implicit EFLAGS)]>;
  def SUB16mi  : Ii16<0x81, MRM5m, (outs), (ins i16mem:$dst, i16imm:$src2), 
                      "sub{w}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (loadi16 addr:$dst), imm:$src2),addr:$dst),
                       (implicit EFLAGS)]>, OpSize;
  def SUB32mi  : Ii32<0x81, MRM5m, (outs), (ins i32mem:$dst, i32imm:$src2), 
                      "sub{l}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (loadi32 addr:$dst), imm:$src2),addr:$dst),
                       (implicit EFLAGS)]>;
  def SUB16mi8 : Ii8<0x83, MRM5m, (outs), (ins i16mem:$dst, i16i8imm :$src2), 
                     "sub{w}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (load addr:$dst), i16immSExt8:$src2),
                             addr:$dst),
                      (implicit EFLAGS)]>, OpSize;
  def SUB32mi8 : Ii8<0x83, MRM5m, (outs), (ins i32mem:$dst, i32i8imm :$src2),
                     "sub{l}\t{$src2, $dst|$dst, $src2}",
                     [(store (sub (load addr:$dst), i32immSExt8:$src2),
                             addr:$dst),
                      (implicit EFLAGS)]>;
}

let Uses = [EFLAGS] in {
def SBB32rr    : I<0x19, MRMDestReg, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
                  "sbb{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (sube GR32:$src1, GR32:$src2))]>;

let isTwoAddress = 0 in {
  def SBB32mr  : I<0x19, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src2), 
                   "sbb{l}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (load addr:$dst), GR32:$src2), addr:$dst)]>;
  def SBB8mi  : Ii32<0x80, MRM3m, (outs), (ins i8mem:$dst, i8imm:$src2), 
                      "sbb{b}\t{$src2, $dst|$dst, $src2}",
                   [(store (sube (loadi8 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB32mi  : Ii32<0x81, MRM3m, (outs), (ins i32mem:$dst, i32imm:$src2), 
                      "sbb{l}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (loadi32 addr:$dst), imm:$src2), addr:$dst)]>;
  def SBB32mi8 : Ii8<0x83, MRM3m, (outs), (ins i32mem:$dst, i32i8imm :$src2), 
                     "sbb{l}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i32immSExt8:$src2), addr:$dst)]>;
}
def SBB32rm  : I<0x1B, MRMSrcMem, (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (sube GR32:$src1, (load addr:$src2)))]>;
def SBB32ri  : Ii32<0x81, MRM3r, (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                    "sbb{l}\t{$src2, $dst|$dst, $src2}",
                    [(set GR32:$dst, (sube GR32:$src1, imm:$src2))]>;
def SBB32ri8 : Ii8<0x83, MRM3r, (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                   "sbb{l}\t{$src2, $dst|$dst, $src2}",
                   [(set GR32:$dst, (sube GR32:$src1, i32immSExt8:$src2))]>;
} // Uses = [EFLAGS]
} // Defs = [EFLAGS]

let Defs = [EFLAGS] in {
let isCommutable = 1 in {  // X = IMUL Y, Z --> X = IMUL Z, Y
// Register-Register Signed Integer Multiply
def IMUL16rr : I<0xAF, MRMSrcReg, (outs GR16:$dst), (ins GR16:$src1,GR16:$src2),
                 "imul{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (mul GR16:$src1, GR16:$src2)),
                  (implicit EFLAGS)]>, TB, OpSize;
def IMUL32rr : I<0xAF, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src1,GR32:$src2),
                 "imul{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (mul GR32:$src1, GR32:$src2)),
                  (implicit EFLAGS)]>, TB;
}

// Register-Memory Signed Integer Multiply
def IMUL16rm : I<0xAF, MRMSrcMem, (outs GR16:$dst),
                                  (ins GR16:$src1, i16mem:$src2),
                 "imul{w}\t{$src2, $dst|$dst, $src2}",
                 [(set GR16:$dst, (mul GR16:$src1, (load addr:$src2))),
                  (implicit EFLAGS)]>, TB, OpSize;
def IMUL32rm : I<0xAF, MRMSrcMem, (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2),
                 "imul{l}\t{$src2, $dst|$dst, $src2}",
                 [(set GR32:$dst, (mul GR32:$src1, (load addr:$src2))),
                  (implicit EFLAGS)]>, TB;
} // Defs = [EFLAGS]
} // end Two Address instructions

// Suprisingly enough, these are not two address instructions!
let Defs = [EFLAGS] in {
// Register-Integer Signed Integer Multiply
def IMUL16rri  : Ii16<0x69, MRMSrcReg,                      // GR16 = GR16*I16
                      (outs GR16:$dst), (ins GR16:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, (mul GR16:$src1, imm:$src2)),
                       (implicit EFLAGS)]>, OpSize;
def IMUL32rri  : Ii32<0x69, MRMSrcReg,                      // GR32 = GR32*I32
                      (outs GR32:$dst), (ins GR32:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, (mul GR32:$src1, imm:$src2)),
                       (implicit EFLAGS)]>;
def IMUL16rri8 : Ii8<0x6B, MRMSrcReg,                       // GR16 = GR16*I8
                     (outs GR16:$dst), (ins GR16:$src1, i16i8imm:$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, (mul GR16:$src1, i16immSExt8:$src2)),
                      (implicit EFLAGS)]>, OpSize;
def IMUL32rri8 : Ii8<0x6B, MRMSrcReg,                       // GR32 = GR32*I8
                     (outs GR32:$dst), (ins GR32:$src1, i32i8imm:$src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, (mul GR32:$src1, i32immSExt8:$src2)),
                      (implicit EFLAGS)]>;

// Memory-Integer Signed Integer Multiply
def IMUL16rmi  : Ii16<0x69, MRMSrcMem,                      // GR16 = [mem16]*I16
                      (outs GR16:$dst), (ins i16mem:$src1, i16imm:$src2),
                      "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR16:$dst, (mul (load addr:$src1), imm:$src2)),
                       (implicit EFLAGS)]>, OpSize;
def IMUL32rmi  : Ii32<0x69, MRMSrcMem,                      // GR32 = [mem32]*I32
                      (outs GR32:$dst), (ins i32mem:$src1, i32imm:$src2),
                      "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR32:$dst, (mul (load addr:$src1), imm:$src2)),
                       (implicit EFLAGS)]>;
def IMUL16rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR16 = [mem16]*I8
                     (outs GR16:$dst), (ins i16mem:$src1, i16i8imm :$src2),
                     "imul{w}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR16:$dst, (mul (load addr:$src1),
                                       i16immSExt8:$src2)),
                      (implicit EFLAGS)]>, OpSize;
def IMUL32rmi8 : Ii8<0x6B, MRMSrcMem,                       // GR32 = [mem32]*I8
                     (outs GR32:$dst), (ins i32mem:$src1, i32i8imm: $src2),
                     "imul{l}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                     [(set GR32:$dst, (mul (load addr:$src1),
                                           i32immSExt8:$src2)),
                      (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
// Test instructions are just like AND, except they don't generate a result.
//
let Defs = [EFLAGS] in {
let isCommutable = 1 in {   // TEST X, Y   --> TEST Y, X
def TEST8rr  : I<0x84, MRMDestReg, (outs),  (ins GR8:$src1, GR8:$src2),
                     "test{b}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and_su GR8:$src1, GR8:$src2), 0),
                      (implicit EFLAGS)]>;
def TEST16rr : I<0x85, MRMDestReg, (outs),  (ins GR16:$src1, GR16:$src2),
                     "test{w}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and_su GR16:$src1, GR16:$src2), 0),
                      (implicit EFLAGS)]>,
                 OpSize;
def TEST32rr : I<0x85, MRMDestReg, (outs),  (ins GR32:$src1, GR32:$src2),
                     "test{l}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and_su GR32:$src1, GR32:$src2), 0),
                      (implicit EFLAGS)]>;
}

def TEST8rm  : I<0x84, MRMSrcMem, (outs),  (ins GR8 :$src1, i8mem :$src2),
                     "test{b}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and GR8:$src1, (loadi8 addr:$src2)), 0),
                      (implicit EFLAGS)]>;
def TEST16rm : I<0x85, MRMSrcMem, (outs),  (ins GR16:$src1, i16mem:$src2),
                     "test{w}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and GR16:$src1, (loadi16 addr:$src2)), 0),
                      (implicit EFLAGS)]>, OpSize;
def TEST32rm : I<0x85, MRMSrcMem, (outs),  (ins GR32:$src1, i32mem:$src2),
                     "test{l}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and GR32:$src1, (loadi32 addr:$src2)), 0),
                      (implicit EFLAGS)]>;

def TEST8ri  : Ii8 <0xF6, MRM0r,                     // flags = GR8  & imm8
                    (outs),  (ins GR8:$src1, i8imm:$src2),
                    "test{b}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and_su GR8:$src1, imm:$src2), 0),
                     (implicit EFLAGS)]>;
def TEST16ri : Ii16<0xF7, MRM0r,                     // flags = GR16 & imm16
                    (outs),  (ins GR16:$src1, i16imm:$src2),
                    "test{w}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and_su GR16:$src1, imm:$src2), 0),
                     (implicit EFLAGS)]>, OpSize;
def TEST32ri : Ii32<0xF7, MRM0r,                     // flags = GR32 & imm32
                    (outs),  (ins GR32:$src1, i32imm:$src2),
                    "test{l}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and_su GR32:$src1, imm:$src2), 0),
                     (implicit EFLAGS)]>;

def TEST8mi  : Ii8 <0xF6, MRM0m,                   // flags = [mem8]  & imm8
                    (outs), (ins i8mem:$src1, i8imm:$src2),
                    "test{b}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and (loadi8 addr:$src1), imm:$src2), 0),
                     (implicit EFLAGS)]>;
def TEST16mi : Ii16<0xF7, MRM0m,                   // flags = [mem16] & imm16
                    (outs), (ins i16mem:$src1, i16imm:$src2),
                    "test{w}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and (loadi16 addr:$src1), imm:$src2), 0),
                     (implicit EFLAGS)]>, OpSize;
def TEST32mi : Ii32<0xF7, MRM0m,                   // flags = [mem32] & imm32
                    (outs), (ins i32mem:$src1, i32imm:$src2),
                    "test{l}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (and (loadi32 addr:$src1), imm:$src2), 0),
                     (implicit EFLAGS)]>;
} // Defs = [EFLAGS]


// Condition code ops, incl. set if equal/not equal/...
let Defs = [EFLAGS], Uses = [AH], neverHasSideEffects = 1 in
def SAHF     : I<0x9E, RawFrm, (outs),  (ins), "sahf", []>;  // flags = AH
let Defs = [AH], Uses = [EFLAGS], neverHasSideEffects = 1 in
def LAHF     : I<0x9F, RawFrm, (outs),  (ins), "lahf", []>;  // AH = flags

let Uses = [EFLAGS] in {
def SETEr    : I<0x94, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sete\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_E, EFLAGS))]>,
               TB;                        // GR8 = ==
def SETEm    : I<0x94, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sete\t$dst",
                 [(store (X86setcc X86_COND_E, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = ==

def SETNEr   : I<0x95, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setne\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NE, EFLAGS))]>,
               TB;                        // GR8 = !=
def SETNEm   : I<0x95, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setne\t$dst",
                 [(store (X86setcc X86_COND_NE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !=

def SETLr    : I<0x9C, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setl\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_L, EFLAGS))]>,
               TB;                        // GR8 = <  signed
def SETLm    : I<0x9C, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setl\t$dst",
                 [(store (X86setcc X86_COND_L, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  signed

def SETGEr   : I<0x9D, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setge\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_GE, EFLAGS))]>,
               TB;                        // GR8 = >= signed
def SETGEm   : I<0x9D, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setge\t$dst",
                 [(store (X86setcc X86_COND_GE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= signed

def SETLEr   : I<0x9E, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setle\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_LE, EFLAGS))]>,
               TB;                        // GR8 = <= signed
def SETLEm   : I<0x9E, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setle\t$dst",
                 [(store (X86setcc X86_COND_LE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= signed

def SETGr    : I<0x9F, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setg\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_G, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETGm    : I<0x9F, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setg\t$dst",
                 [(store (X86setcc X86_COND_G, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETBr    : I<0x92, MRM0r,
                 (outs GR8   :$dst), (ins),
                 "setb\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_B, EFLAGS))]>,
               TB;                        // GR8 = <  unsign
def SETBm    : I<0x92, MRM0m,
                 (outs), (ins i8mem:$dst),
                 "setb\t$dst",
                 [(store (X86setcc X86_COND_B, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <  unsign

def SETAEr   : I<0x93, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setae\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_AE, EFLAGS))]>,
               TB;                        // GR8 = >= unsign
def SETAEm   : I<0x93, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setae\t$dst",
                 [(store (X86setcc X86_COND_AE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >= unsign

def SETBEr   : I<0x96, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setbe\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_BE, EFLAGS))]>,
               TB;                        // GR8 = <= unsign
def SETBEm   : I<0x96, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setbe\t$dst",
                 [(store (X86setcc X86_COND_BE, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <= unsign

def SETAr    : I<0x97, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seta\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_A, EFLAGS))]>,
               TB;                        // GR8 = >  signed
def SETAm    : I<0x97, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seta\t$dst",
                 [(store (X86setcc X86_COND_A, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = >  signed

def SETSr    : I<0x98, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "sets\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_S, EFLAGS))]>,
               TB;                        // GR8 = <sign bit>
def SETSm    : I<0x98, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "sets\t$dst",
                 [(store (X86setcc X86_COND_S, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = <sign bit>
def SETNSr   : I<0x99, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setns\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NS, EFLAGS))]>,
               TB;                        // GR8 = !<sign bit>
def SETNSm   : I<0x99, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setns\t$dst",
                 [(store (X86setcc X86_COND_NS, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = !<sign bit>

def SETPr    : I<0x9A, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_P, EFLAGS))]>,
               TB;                        // GR8 = parity
def SETPm    : I<0x9A, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setp\t$dst",
                 [(store (X86setcc X86_COND_P, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = parity
def SETNPr   : I<0x9B, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setnp\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NP, EFLAGS))]>,
               TB;                        // GR8 = not parity
def SETNPm   : I<0x9B, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setnp\t$dst",
                 [(store (X86setcc X86_COND_NP, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not parity

def SETOr    : I<0x90, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "seto\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_O, EFLAGS))]>,
               TB;                        // GR8 = overflow
def SETOm    : I<0x90, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "seto\t$dst",
                 [(store (X86setcc X86_COND_O, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = overflow
def SETNOr   : I<0x91, MRM0r, 
                 (outs GR8   :$dst), (ins),
                 "setno\t$dst",
                 [(set GR8:$dst, (X86setcc X86_COND_NO, EFLAGS))]>,
               TB;                        // GR8 = not overflow
def SETNOm   : I<0x91, MRM0m, 
                 (outs), (ins i8mem:$dst),
                 "setno\t$dst",
                 [(store (X86setcc X86_COND_NO, EFLAGS), addr:$dst)]>,
               TB;                        // [mem8] = not overflow
} // Uses = [EFLAGS]


// Integer comparisons
let Defs = [EFLAGS] in {
def CMP8rr  : I<0x38, MRMDestReg,
                (outs), (ins GR8 :$src1, GR8 :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR8:$src1, GR8:$src2), (implicit EFLAGS)]>;
def CMP16rr : I<0x39, MRMDestReg,
                (outs), (ins GR16:$src1, GR16:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR16:$src1, GR16:$src2), (implicit EFLAGS)]>, OpSize;
def CMP32rr : I<0x39, MRMDestReg,
                (outs), (ins GR32:$src1, GR32:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR32:$src1, GR32:$src2), (implicit EFLAGS)]>;
def CMP8mr  : I<0x38, MRMDestMem,
                (outs), (ins i8mem :$src1, GR8 :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi8 addr:$src1), GR8:$src2),
                 (implicit EFLAGS)]>;
def CMP16mr : I<0x39, MRMDestMem,
                (outs), (ins i16mem:$src1, GR16:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi16 addr:$src1), GR16:$src2),
                 (implicit EFLAGS)]>, OpSize;
def CMP32mr : I<0x39, MRMDestMem,
                (outs), (ins i32mem:$src1, GR32:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp (loadi32 addr:$src1), GR32:$src2),
                 (implicit EFLAGS)]>;
def CMP8rm  : I<0x3A, MRMSrcMem,
                (outs), (ins GR8 :$src1, i8mem :$src2),
                "cmp{b}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR8:$src1, (loadi8 addr:$src2)),
                 (implicit EFLAGS)]>;
def CMP16rm : I<0x3B, MRMSrcMem,
                (outs), (ins GR16:$src1, i16mem:$src2),
                "cmp{w}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR16:$src1, (loadi16 addr:$src2)),
                 (implicit EFLAGS)]>, OpSize;
def CMP32rm : I<0x3B, MRMSrcMem,
                (outs), (ins GR32:$src1, i32mem:$src2),
                "cmp{l}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp GR32:$src1, (loadi32 addr:$src2)),
                 (implicit EFLAGS)]>;
def CMP8ri  : Ii8<0x80, MRM7r,
                  (outs), (ins GR8:$src1, i8imm:$src2),
                  "cmp{b}\t{$src2, $src1|$src1, $src2}",
                  [(X86cmp GR8:$src1, imm:$src2), (implicit EFLAGS)]>;
def CMP16ri : Ii16<0x81, MRM7r,
                   (outs), (ins GR16:$src1, i16imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp GR16:$src1, imm:$src2),
                    (implicit EFLAGS)]>, OpSize;
def CMP32ri : Ii32<0x81, MRM7r,
                   (outs), (ins GR32:$src1, i32imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp GR32:$src1, imm:$src2), (implicit EFLAGS)]>;
def CMP8mi  : Ii8 <0x80, MRM7m,
                   (outs), (ins i8mem :$src1, i8imm :$src2),
                   "cmp{b}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi8 addr:$src1), imm:$src2),
                    (implicit EFLAGS)]>;
def CMP16mi : Ii16<0x81, MRM7m,
                   (outs), (ins i16mem:$src1, i16imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi16 addr:$src1), imm:$src2),
                    (implicit EFLAGS)]>, OpSize;
def CMP32mi : Ii32<0x81, MRM7m,
                   (outs), (ins i32mem:$src1, i32imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi32 addr:$src1), imm:$src2),
                    (implicit EFLAGS)]>;
def CMP16ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR16:$src1, i16i8imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp GR16:$src1, i16immSExt8:$src2),
                    (implicit EFLAGS)]>, OpSize;
def CMP16mi8 : Ii8<0x83, MRM7m,
                   (outs), (ins i16mem:$src1, i16i8imm:$src2),
                   "cmp{w}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi16 addr:$src1), i16immSExt8:$src2),
                    (implicit EFLAGS)]>, OpSize;
def CMP32mi8 : Ii8<0x83, MRM7m,
                   (outs), (ins i32mem:$src1, i32i8imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp (loadi32 addr:$src1), i32immSExt8:$src2),
                    (implicit EFLAGS)]>;
def CMP32ri8 : Ii8<0x83, MRM7r,
                   (outs), (ins GR32:$src1, i32i8imm:$src2),
                   "cmp{l}\t{$src2, $src1|$src1, $src2}",
                   [(X86cmp GR32:$src1, i32immSExt8:$src2),
                    (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Bit tests.
// TODO: BTC, BTR, and BTS
let Defs = [EFLAGS] in {
def BT16rr : I<0xA3, MRMDestReg, (outs), (ins GR16:$src1, GR16:$src2),
               "bt{w}\t{$src2, $src1|$src1, $src2}",
               [(X86bt GR16:$src1, GR16:$src2),
                (implicit EFLAGS)]>, OpSize, TB;
def BT32rr : I<0xA3, MRMDestReg, (outs), (ins GR32:$src1, GR32:$src2),
               "bt{l}\t{$src2, $src1|$src1, $src2}",
               [(X86bt GR32:$src1, GR32:$src2),
                (implicit EFLAGS)]>, TB;

// Unlike with the register+register form, the memory+register form of the
// bt instruction does not ignore the high bits of the index. From ISel's
// perspective, this is pretty bizarre. Disable these instructions for now.
//def BT16mr : I<0xA3, MRMDestMem, (outs), (ins i16mem:$src1, GR16:$src2),
//               "bt{w}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi16 addr:$src1), GR16:$src2),
//                (implicit EFLAGS)]>, OpSize, TB, Requires<[FastBTMem]>;
//def BT32mr : I<0xA3, MRMDestMem, (outs), (ins i32mem:$src1, GR32:$src2),
//               "bt{l}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi32 addr:$src1), GR32:$src2),
//                (implicit EFLAGS)]>, TB, Requires<[FastBTMem]>;

def BT16ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR16:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(X86bt GR16:$src1, i16immSExt8:$src2),
                 (implicit EFLAGS)]>, OpSize, TB;
def BT32ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR32:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(X86bt GR32:$src1, i32immSExt8:$src2),
                 (implicit EFLAGS)]>, TB;
// Note that these instructions don't need FastBTMem because that
// only applies when the other operand is in a register. When it's
// an immediate, bt is still fast.
def BT16mi8 : Ii8<0xBA, MRM4m, (outs), (ins i16mem:$src1, i16i8imm:$src2),
                "bt{w}\t{$src2, $src1|$src1, $src2}",
                [(X86bt (loadi16 addr:$src1), i16immSExt8:$src2),
                 (implicit EFLAGS)]>, OpSize, TB;
def BT32mi8 : Ii8<0xBA, MRM4m, (outs), (ins i32mem:$src1, i32i8imm:$src2),
                "bt{l}\t{$src2, $src1|$src1, $src2}",
                [(X86bt (loadi32 addr:$src1), i32immSExt8:$src2),
                 (implicit EFLAGS)]>, TB;
} // Defs = [EFLAGS]

// Sign/Zero extenders
// Use movsbl intead of movsbw; we don't care about the high 16 bits
// of the register here. This has a smaller encoding and avoids a
// partial-register update.
def MOVSX16rr8 : I<0xBE, MRMSrcReg, (outs GR16:$dst), (ins GR8 :$src),
                   "movs{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR16:$dst, (sext GR8:$src))]>, TB;
def MOVSX16rm8 : I<0xBE, MRMSrcMem, (outs GR16:$dst), (ins i8mem :$src),
                   "movs{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR16:$dst, (sextloadi16i8 addr:$src))]>, TB;
def MOVSX32rr8 : I<0xBE, MRMSrcReg, (outs GR32:$dst), (ins GR8 :$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR8:$src))]>, TB;
def MOVSX32rm8 : I<0xBE, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movs{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i8 addr:$src))]>, TB;
def MOVSX32rr16: I<0xBF, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sext GR16:$src))]>, TB;
def MOVSX32rm16: I<0xBF, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movs{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (sextloadi32i16 addr:$src))]>, TB;

// Use movzbl intead of movzbw; we don't care about the high 16 bits
// of the register here. This has a smaller encoding and avoids a
// partial-register update.
def MOVZX16rr8 : I<0xB6, MRMSrcReg, (outs GR16:$dst), (ins GR8 :$src),
                   "movz{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR16:$dst, (zext GR8:$src))]>, TB;
def MOVZX16rm8 : I<0xB6, MRMSrcMem, (outs GR16:$dst), (ins i8mem :$src),
                   "movz{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR16:$dst, (zextloadi16i8 addr:$src))]>, TB;
def MOVZX32rr8 : I<0xB6, MRMSrcReg, (outs GR32:$dst), (ins GR8 :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR8:$src))]>, TB;
def MOVZX32rm8 : I<0xB6, MRMSrcMem, (outs GR32:$dst), (ins i8mem :$src),
                   "movz{bl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i8 addr:$src))]>, TB;
def MOVZX32rr16: I<0xB7, MRMSrcReg, (outs GR32:$dst), (ins GR16:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zext GR16:$src))]>, TB;
def MOVZX32rm16: I<0xB7, MRMSrcMem, (outs GR32:$dst), (ins i16mem:$src),
                   "movz{wl|x}\t{$src, $dst|$dst, $src}",
                   [(set GR32:$dst, (zextloadi32i16 addr:$src))]>, TB;

let neverHasSideEffects = 1 in {
  let Defs = [AX], Uses = [AL] in
  def CBW : I<0x98, RawFrm, (outs), (ins),
              "{cbtw|cbw}", []>, OpSize;   // AX = signext(AL)
  let Defs = [EAX], Uses = [AX] in
  def CWDE : I<0x98, RawFrm, (outs), (ins),
              "{cwtl|cwde}", []>;   // EAX = signext(AX)

  let Defs = [AX,DX], Uses = [AX] in
  def CWD : I<0x99, RawFrm, (outs), (ins),
              "{cwtd|cwd}", []>, OpSize; // DX:AX = signext(AX)
  let Defs = [EAX,EDX], Uses = [EAX] in
  def CDQ : I<0x99, RawFrm, (outs), (ins),
              "{cltd|cdq}", []>; // EDX:EAX = signext(EAX)
}

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map movr0 to xor.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
let Defs = [EFLAGS], isReMaterializable = 1, isAsCheapAsAMove = 1 in {
def MOV8r0   : I<0x30, MRMInitReg, (outs GR8 :$dst), (ins),
                 "xor{b}\t$dst, $dst",
                 [(set GR8:$dst, 0)]>;
// Use xorl instead of xorw since we don't care about the high 16 bits,
// it's smaller, and it avoids a partial-register update.
def MOV16r0  : I<0x31, MRMInitReg,  (outs GR16:$dst), (ins),
                 "xor{l}\t${dst:subreg32}, ${dst:subreg32}",
                 [(set GR16:$dst, 0)]>;
def MOV32r0  : I<0x31, MRMInitReg,  (outs GR32:$dst), (ins),
                 "xor{l}\t$dst, $dst",
                 [(set GR32:$dst, 0)]>;
}

// Basic operations on GR16 / GR32 subclasses GR16_ and GR32_ which contains only
// those registers that have GR8 sub-registers (i.e. AX - DX, EAX - EDX).
let neverHasSideEffects = 1 in {
def MOV16to16_ : I<0x89, MRMDestReg, (outs GR16_:$dst), (ins GR16:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32to32_ : I<0x89, MRMDestReg, (outs GR32_:$dst), (ins GR32:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
                
def MOV16_rr : I<0x89, MRMDestReg, (outs GR16_:$dst), (ins GR16_:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32_rr : I<0x89, MRMDestReg, (outs GR32_:$dst), (ins GR32_:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
} // neverHasSideEffects

let canFoldAsLoad = 1, mayLoad = 1, isReMaterializable = 1, mayHaveSideEffects = 1 in {
def MOV16_rm : I<0x8B, MRMSrcMem, (outs GR16_:$dst), (ins i16mem:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32_rm : I<0x8B, MRMSrcMem, (outs GR32_:$dst), (ins i32mem:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
}
let mayStore = 1, neverHasSideEffects = 1 in {
def MOV16_mr : I<0x89, MRMDestMem, (outs), (ins i16mem:$dst, GR16_:$src),
                "mov{w}\t{$src, $dst|$dst, $src}", []>, OpSize;
def MOV32_mr : I<0x89, MRMDestMem, (outs), (ins i32mem:$dst, GR32_:$src),
                "mov{l}\t{$src, $dst|$dst, $src}", []>;
}

//===----------------------------------------------------------------------===//
// Thread Local Storage Instructions
//

let Uses = [EBX] in
def TLS_addr32 : I<0, Pseudo, (outs GR32:$dst), (ins i32imm:$sym),
                  "leal\t${sym:mem}(,%ebx,1), $dst",
                  [(set GR32:$dst, (X86tlsaddr tglobaltlsaddr:$sym))]>;

let AddedComplexity = 10 in
def TLS_gs_rr  : I<0, Pseudo, (outs GR32:$dst), (ins GR32:$src),
                  "movl\t%gs:($src), $dst",
                  [(set GR32:$dst, (load (add X86TLStp, GR32:$src)))]>;

let AddedComplexity = 15 in
def TLS_gs_ri : I<0x8B, Pseudo, (outs GR32:$dst), (ins i32imm:$src),
                  "movl\t%gs:${src:mem}, $dst",
                  [(set GR32:$dst,
                    (load (add X86TLStp, (X86Wrapper tglobaltlsaddr:$src))))]>,
                  SegGS;

def TLS_tp : I<0x8B, Pseudo, (outs GR32:$dst), (ins),
               "movl\t%gs:0, $dst",
               [(set GR32:$dst, X86TLStp)]>, SegGS;

let AddedComplexity = 5 in
def GS_MOV32rm : I<0x8B, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
                   "movl\t%gs:$src, $dst",
                   [(set GR32:$dst, (gsload addr:$src))]>, SegGS;

//===----------------------------------------------------------------------===//
// DWARF Pseudo Instructions
//

def DWARF_LOC   : I<0, Pseudo, (outs),
                    (ins i32imm:$line, i32imm:$col, i32imm:$file),
                    ".loc\t${file:debug} ${line:debug} ${col:debug}",
                    [(dwarf_loc (i32 imm:$line), (i32 imm:$col),
                      (i32 imm:$file))]>;

//===----------------------------------------------------------------------===//
// EH Pseudo Instructions
//
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1 in {
def EH_RETURN   : I<0xC3, RawFrm, (outs), (ins GR32:$addr),
                    "ret\t#eh_return, addr: $addr",
                    [(X86ehret GR32:$addr)]>;

}

//===----------------------------------------------------------------------===//
// Atomic support
//

// Atomic swap. These are just normal xchg instructions. But since a memory
// operand is referenced, the atomicity is ensured.
let Constraints = "$val = $dst" in {
def XCHG32rm : I<0x87, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$ptr, GR32:$val),
               "xchg{l}\t{$val, $ptr|$ptr, $val}", 
               [(set GR32:$dst, (atomic_swap_32 addr:$ptr, GR32:$val))]>;
def XCHG16rm : I<0x87, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$ptr, GR16:$val),
               "xchg{w}\t{$val, $ptr|$ptr, $val}", 
               [(set GR16:$dst, (atomic_swap_16 addr:$ptr, GR16:$val))]>, 
                OpSize;
def XCHG8rm  : I<0x86, MRMSrcMem, (outs GR8:$dst), (ins i8mem:$ptr, GR8:$val),
               "xchg{b}\t{$val, $ptr|$ptr, $val}", 
               [(set GR8:$dst, (atomic_swap_8 addr:$ptr, GR8:$val))]>;
}

// Atomic compare and swap.
let Defs = [EAX, EFLAGS], Uses = [EAX] in {
def LCMPXCHG32 : I<0xB1, MRMDestMem, (outs), (ins i32mem:$ptr, GR32:$swap),
               "lock\n\tcmpxchg{l}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR32:$swap, 4)]>, TB, LOCK;
}
let Defs = [EAX, EDX, EFLAGS], Uses = [EAX, EBX, ECX, EDX] in {
def LCMPXCHG8B : I<0xC7, MRM1m, (outs), (ins i32mem:$ptr),
               "lock\n\tcmpxchg8b\t$ptr",
               [(X86cas8 addr:$ptr)]>, TB, LOCK;
}

let Defs = [AX, EFLAGS], Uses = [AX] in {
def LCMPXCHG16 : I<0xB1, MRMDestMem, (outs), (ins i16mem:$ptr, GR16:$swap),
               "lock\n\tcmpxchg{w}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR16:$swap, 2)]>, TB, OpSize, LOCK;
}
let Defs = [AL, EFLAGS], Uses = [AL] in {
def LCMPXCHG8 : I<0xB0, MRMDestMem, (outs), (ins i8mem:$ptr, GR8:$swap),
               "lock\n\tcmpxchg{b}\t{$swap, $ptr|$ptr, $swap}",
               [(X86cas addr:$ptr, GR8:$swap, 1)]>, TB, LOCK;
}

// Atomic exchange and add
let Constraints = "$val = $dst", Defs = [EFLAGS] in {
def LXADD32 : I<0xC1, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$ptr, GR32:$val),
               "lock\n\txadd{l}\t{$val, $ptr|$ptr, $val}", 
               [(set GR32:$dst, (atomic_load_add_32 addr:$ptr, GR32:$val))]>,
                TB, LOCK;
def LXADD16 : I<0xC1, MRMSrcMem, (outs GR16:$dst), (ins i16mem:$ptr, GR16:$val),
               "lock\n\txadd{w}\t{$val, $ptr|$ptr, $val}", 
               [(set GR16:$dst, (atomic_load_add_16 addr:$ptr, GR16:$val))]>,
                TB, OpSize, LOCK;
def LXADD8  : I<0xC0, MRMSrcMem, (outs GR8:$dst), (ins i8mem:$ptr, GR8:$val),
               "lock\n\txadd{b}\t{$val, $ptr|$ptr, $val}", 
               [(set GR8:$dst, (atomic_load_add_8 addr:$ptr, GR8:$val))]>,
                TB, LOCK;
}

// Atomic exchange, and, or, xor
let Constraints = "$val = $dst", Defs = [EFLAGS],
                  usesCustomDAGSchedInserter = 1 in {
def ATOMAND32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMAND32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_and_32 addr:$ptr, GR32:$val))]>;
def ATOMOR32 : I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMOR32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_or_32 addr:$ptr, GR32:$val))]>;
def ATOMXOR32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMXOR32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_xor_32 addr:$ptr, GR32:$val))]>;
def ATOMNAND32 : I<0, Pseudo,(outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMNAND32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_nand_32 addr:$ptr, GR32:$val))]>;
def ATOMMIN32: I<0, Pseudo, (outs GR32:$dst), (ins i32mem:$ptr, GR32:$val),
               "#ATOMMIN32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_min_32 addr:$ptr, GR32:$val))]>;
def ATOMMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMMAX32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_max_32 addr:$ptr, GR32:$val))]>;
def ATOMUMIN32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMIN32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_umin_32 addr:$ptr, GR32:$val))]>;
def ATOMUMAX32: I<0, Pseudo, (outs GR32:$dst),(ins i32mem:$ptr, GR32:$val),
               "#ATOMUMAX32 PSEUDO!", 
               [(set GR32:$dst, (atomic_load_umax_32 addr:$ptr, GR32:$val))]>;

def ATOMAND16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMAND16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_and_16 addr:$ptr, GR16:$val))]>;
def ATOMOR16 : I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMOR16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_or_16 addr:$ptr, GR16:$val))]>;
def ATOMXOR16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMXOR16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_xor_16 addr:$ptr, GR16:$val))]>;
def ATOMNAND16 : I<0, Pseudo,(outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMNAND16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_nand_16 addr:$ptr, GR16:$val))]>;
def ATOMMIN16: I<0, Pseudo, (outs GR16:$dst), (ins i16mem:$ptr, GR16:$val),
               "#ATOMMIN16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_min_16 addr:$ptr, GR16:$val))]>;
def ATOMMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMMAX16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_max_16 addr:$ptr, GR16:$val))]>;
def ATOMUMIN16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMIN16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_umin_16 addr:$ptr, GR16:$val))]>;
def ATOMUMAX16: I<0, Pseudo, (outs GR16:$dst),(ins i16mem:$ptr, GR16:$val),
               "#ATOMUMAX16 PSEUDO!", 
               [(set GR16:$dst, (atomic_load_umax_16 addr:$ptr, GR16:$val))]>;

def ATOMAND8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMAND8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_and_8 addr:$ptr, GR8:$val))]>;
def ATOMOR8 : I<0, Pseudo, (outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMOR8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_or_8 addr:$ptr, GR8:$val))]>;
def ATOMXOR8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMXOR8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_xor_8 addr:$ptr, GR8:$val))]>;
def ATOMNAND8 : I<0, Pseudo,(outs GR8:$dst),(ins i8mem:$ptr, GR8:$val),
               "#ATOMNAND8 PSEUDO!", 
               [(set GR8:$dst, (atomic_load_nand_8 addr:$ptr, GR8:$val))]>;
}

let Constraints = "$val1 = $dst1, $val2 = $dst2", 
                  Defs = [EFLAGS, EAX, EBX, ECX, EDX],
                  Uses = [EAX, EBX, ECX, EDX],
                  mayLoad = 1, mayStore = 1,
                  usesCustomDAGSchedInserter = 1 in {
def ATOMAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMAND6432 PSEUDO!", []>;
def ATOMOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMOR6432 PSEUDO!", []>;
def ATOMXOR6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMXOR6432 PSEUDO!", []>;
def ATOMNAND6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMNAND6432 PSEUDO!", []>;
def ATOMADD6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMADD6432 PSEUDO!", []>;
def ATOMSUB6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSUB6432 PSEUDO!", []>;
def ATOMSWAP6432 : I<0, Pseudo, (outs GR32:$dst1, GR32:$dst2),
                               (ins i64mem:$ptr, GR32:$val1, GR32:$val2),
               "#ATOMSWAP6432 PSEUDO!", []>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// ConstantPool GlobalAddress, ExternalSymbol, and JumpTable
def : Pat<(i32 (X86Wrapper tconstpool  :$dst)), (MOV32ri tconstpool  :$dst)>;
def : Pat<(i32 (X86Wrapper tjumptable  :$dst)), (MOV32ri tjumptable  :$dst)>;
def : Pat<(i32 (X86Wrapper tglobaltlsaddr:$dst)),(MOV32ri tglobaltlsaddr:$dst)>;
def : Pat<(i32 (X86Wrapper tglobaladdr :$dst)), (MOV32ri tglobaladdr :$dst)>;
def : Pat<(i32 (X86Wrapper texternalsym:$dst)), (MOV32ri texternalsym:$dst)>;

def : Pat<(add GR32:$src1, (X86Wrapper tconstpool:$src2)),
          (ADD32ri GR32:$src1, tconstpool:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper tjumptable:$src2)),
          (ADD32ri GR32:$src1, tjumptable:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper tglobaladdr :$src2)),
          (ADD32ri GR32:$src1, tglobaladdr:$src2)>;
def : Pat<(add GR32:$src1, (X86Wrapper texternalsym:$src2)),
          (ADD32ri GR32:$src1, texternalsym:$src2)>;

def : Pat<(store (i32 (X86Wrapper tglobaladdr:$src)), addr:$dst),
          (MOV32mi addr:$dst, tglobaladdr:$src)>;
def : Pat<(store (i32 (X86Wrapper texternalsym:$src)), addr:$dst),
          (MOV32mi addr:$dst, texternalsym:$src)>;

// Calls
// tailcall stuff
def : Pat<(X86tailcall GR32:$dst),
          (TAILCALL)>;

def : Pat<(X86tailcall (i32 tglobaladdr:$dst)),
          (TAILCALL)>;
def : Pat<(X86tailcall (i32 texternalsym:$dst)),
          (TAILCALL)>;

def : Pat<(X86tcret GR32:$dst, imm:$off),
          (TCRETURNri GR32:$dst, imm:$off)>;

def : Pat<(X86tcret (i32 tglobaladdr:$dst), imm:$off),
          (TCRETURNdi texternalsym:$dst, imm:$off)>;

def : Pat<(X86tcret (i32 texternalsym:$dst), imm:$off),
          (TCRETURNdi texternalsym:$dst, imm:$off)>;

def : Pat<(X86call (i32 tglobaladdr:$dst)),
          (CALLpcrel32 tglobaladdr:$dst)>;
def : Pat<(X86call (i32 texternalsym:$dst)),
          (CALLpcrel32 texternalsym:$dst)>;

// X86 specific add which produces a flag.
def : Pat<(addc GR32:$src1, GR32:$src2),
          (ADD32rr GR32:$src1, GR32:$src2)>;
def : Pat<(addc GR32:$src1, (load addr:$src2)),
          (ADD32rm GR32:$src1, addr:$src2)>;
def : Pat<(addc GR32:$src1, imm:$src2),
          (ADD32ri GR32:$src1, imm:$src2)>;
def : Pat<(addc GR32:$src1, i32immSExt8:$src2),
          (ADD32ri8 GR32:$src1, i32immSExt8:$src2)>;

def : Pat<(subc GR32:$src1, GR32:$src2),
          (SUB32rr GR32:$src1, GR32:$src2)>;
def : Pat<(subc GR32:$src1, (load addr:$src2)),
          (SUB32rm GR32:$src1, addr:$src2)>;
def : Pat<(subc GR32:$src1, imm:$src2),
          (SUB32ri GR32:$src1, imm:$src2)>;
def : Pat<(subc GR32:$src1, i32immSExt8:$src2),
          (SUB32ri8 GR32:$src1, i32immSExt8:$src2)>;

// Comparisons.

// TEST R,R is smaller than CMP R,0
def : Pat<(parallel (X86cmp GR8:$src1, 0), (implicit EFLAGS)),
          (TEST8rr GR8:$src1, GR8:$src1)>;
def : Pat<(parallel (X86cmp GR16:$src1, 0), (implicit EFLAGS)),
          (TEST16rr GR16:$src1, GR16:$src1)>;
def : Pat<(parallel (X86cmp GR32:$src1, 0), (implicit EFLAGS)),
          (TEST32rr GR32:$src1, GR32:$src1)>;

// Conditional moves with folded loads with operands swapped and conditions
// inverted.
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_B, EFLAGS),
          (CMOVAE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_B, EFLAGS),
          (CMOVAE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_AE, EFLAGS),
          (CMOVB16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_AE, EFLAGS),
          (CMOVB32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_E, EFLAGS),
          (CMOVNE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_E, EFLAGS),
          (CMOVNE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_NE, EFLAGS),
          (CMOVE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_NE, EFLAGS),
          (CMOVE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_BE, EFLAGS),
          (CMOVA16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_BE, EFLAGS),
          (CMOVA32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_A, EFLAGS),
          (CMOVBE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_A, EFLAGS),
          (CMOVBE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_L, EFLAGS),
          (CMOVGE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_L, EFLAGS),
          (CMOVGE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_GE, EFLAGS),
          (CMOVL16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_GE, EFLAGS),
          (CMOVL32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_LE, EFLAGS),
          (CMOVG16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_LE, EFLAGS),
          (CMOVG32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_G, EFLAGS),
          (CMOVLE16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_G, EFLAGS),
          (CMOVLE32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_P, EFLAGS),
          (CMOVNP16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_P, EFLAGS),
          (CMOVNP32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_NP, EFLAGS),
          (CMOVP16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_NP, EFLAGS),
          (CMOVP32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_S, EFLAGS),
          (CMOVNS16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_S, EFLAGS),
          (CMOVNS32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_NS, EFLAGS),
          (CMOVS16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_NS, EFLAGS),
          (CMOVS32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_O, EFLAGS),
          (CMOVNO16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_O, EFLAGS),
          (CMOVNO32rm GR32:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, X86_COND_NO, EFLAGS),
          (CMOVO16rm GR16:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, X86_COND_NO, EFLAGS),
          (CMOVO32rm GR32:$src2, addr:$src1)>;

// zextload bool -> zextload byte
def : Pat<(zextloadi8i1  addr:$src), (MOV8rm     addr:$src)>;
def : Pat<(zextloadi16i1 addr:$src), (MOVZX16rm8 addr:$src)>;
def : Pat<(zextloadi32i1 addr:$src), (MOVZX32rm8 addr:$src)>;

// extload bool -> extload byte
def : Pat<(extloadi8i1 addr:$src),   (MOV8rm      addr:$src)>;
def : Pat<(extloadi16i1 addr:$src),  (MOVZX16rm8  addr:$src)>,
         Requires<[In32BitMode]>;
def : Pat<(extloadi32i1 addr:$src),  (MOVZX32rm8  addr:$src)>;
def : Pat<(extloadi16i8 addr:$src),  (MOVZX16rm8  addr:$src)>,
         Requires<[In32BitMode]>;
def : Pat<(extloadi32i8 addr:$src),  (MOVZX32rm8  addr:$src)>;
def : Pat<(extloadi32i16 addr:$src), (MOVZX32rm16 addr:$src)>;

// anyext
def : Pat<(i16 (anyext GR8 :$src)), (MOVZX16rr8  GR8 :$src)>,
         Requires<[In32BitMode]>;
def : Pat<(i32 (anyext GR8 :$src)), (MOVZX32rr8  GR8 :$src)>,
         Requires<[In32BitMode]>;
def : Pat<(i32 (anyext GR16:$src)),
          (INSERT_SUBREG (i32 (IMPLICIT_DEF)), GR16:$src, x86_subreg_16bit)>;

// (and (i32 load), 255) -> (zextload i8)
def : Pat<(i32 (and (nvloadi32 addr:$src), (i32 255))),
          (MOVZX32rm8 addr:$src)>;
def : Pat<(i32 (and (nvloadi32 addr:$src), (i32 65535))),
          (MOVZX32rm16 addr:$src)>;

//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// Odd encoding trick: -128 fits into an 8-bit immediate field while
// +128 doesn't, so in this special case use a sub instead of an add.
def : Pat<(add GR16:$src1, 128),
          (SUB16ri8 GR16:$src1, -128)>;
def : Pat<(store (add (loadi16 addr:$dst), 128), addr:$dst),
          (SUB16mi8 addr:$dst, -128)>;
def : Pat<(add GR32:$src1, 128),
          (SUB32ri8 GR32:$src1, -128)>;
def : Pat<(store (add (loadi32 addr:$dst), 128), addr:$dst),
          (SUB32mi8 addr:$dst, -128)>;

// r & (2^16-1) ==> movz
def : Pat<(and GR32:$src1, 0xffff),
          (MOVZX32rr16 (i16 (EXTRACT_SUBREG GR32:$src1, x86_subreg_16bit)))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR32:$src1, 0xff),
          (MOVZX32rr8 (i8 (EXTRACT_SUBREG (MOV32to32_ GR32:$src1),
                                          x86_subreg_8bit)))>,
      Requires<[In32BitMode]>;
// r & (2^8-1) ==> movz
def : Pat<(and GR16:$src1, 0xff),
          (MOVZX16rr8 (i8 (EXTRACT_SUBREG (MOV16to16_ GR16:$src1),
                                          x86_subreg_8bit)))>,
      Requires<[In32BitMode]>;

// sext_inreg patterns
def : Pat<(sext_inreg GR32:$src, i16),
          (MOVSX32rr16 (i16 (EXTRACT_SUBREG GR32:$src, x86_subreg_16bit)))>;
def : Pat<(sext_inreg GR32:$src, i8),
          (MOVSX32rr8 (i8 (EXTRACT_SUBREG (MOV32to32_ GR32:$src),
                                          x86_subreg_8bit)))>,
      Requires<[In32BitMode]>;
def : Pat<(sext_inreg GR16:$src, i8),
          (MOVSX16rr8 (i8 (EXTRACT_SUBREG (MOV16to16_ GR16:$src),
                                          x86_subreg_8bit)))>,
      Requires<[In32BitMode]>;

// trunc patterns
def : Pat<(i16 (trunc GR32:$src)),
          (i16 (EXTRACT_SUBREG GR32:$src, x86_subreg_16bit))>;
def : Pat<(i8 (trunc GR32:$src)),
          (i8 (EXTRACT_SUBREG (MOV32to32_ GR32:$src), x86_subreg_8bit))>,
      Requires<[In32BitMode]>;
def : Pat<(i8 (trunc GR16:$src)),
          (i8 (EXTRACT_SUBREG (MOV16to16_ GR16:$src), x86_subreg_8bit))>,
      Requires<[In32BitMode]>;

// (shl x, 1) ==> (add x, x)
def : Pat<(shl GR8 :$src1, (i8 1)), (ADD8rr  GR8 :$src1, GR8 :$src1)>;
def : Pat<(shl GR16:$src1, (i8 1)), (ADD16rr GR16:$src1, GR16:$src1)>;
def : Pat<(shl GR32:$src1, (i8 1)), (ADD32rr GR32:$src1, GR32:$src1)>;

// (shl x (and y, 31)) ==> (shl x, y)
def : Pat<(shl GR8:$src1, (and CL:$amt, 31)),
          (SHL8rCL GR8:$src1)>;
def : Pat<(shl GR16:$src1, (and CL:$amt, 31)),
          (SHL16rCL GR16:$src1)>;
def : Pat<(shl GR32:$src1, (and CL:$amt, 31)),
          (SHL32rCL GR32:$src1)>;
def : Pat<(store (shl (loadi8 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHL8mCL addr:$dst)>;
def : Pat<(store (shl (loadi16 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHL16mCL addr:$dst)>;
def : Pat<(store (shl (loadi32 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHL32mCL addr:$dst)>;

def : Pat<(srl GR8:$src1, (and CL:$amt, 31)),
          (SHR8rCL GR8:$src1)>;
def : Pat<(srl GR16:$src1, (and CL:$amt, 31)),
          (SHR16rCL GR16:$src1)>;
def : Pat<(srl GR32:$src1, (and CL:$amt, 31)),
          (SHR32rCL GR32:$src1)>;
def : Pat<(store (srl (loadi8 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHR8mCL addr:$dst)>;
def : Pat<(store (srl (loadi16 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHR16mCL addr:$dst)>;
def : Pat<(store (srl (loadi32 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SHR32mCL addr:$dst)>;

def : Pat<(sra GR8:$src1, (and CL:$amt, 31)),
          (SAR8rCL GR8:$src1)>;
def : Pat<(sra GR16:$src1, (and CL:$amt, 31)),
          (SAR16rCL GR16:$src1)>;
def : Pat<(sra GR32:$src1, (and CL:$amt, 31)),
          (SAR32rCL GR32:$src1)>;
def : Pat<(store (sra (loadi8 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SAR8mCL addr:$dst)>;
def : Pat<(store (sra (loadi16 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SAR16mCL addr:$dst)>;
def : Pat<(store (sra (loadi32 addr:$dst), (and CL:$amt, 31)), addr:$dst),
          (SAR32mCL addr:$dst)>;

// (or (x >> c) | (y << (32 - c))) ==> (shrd32 x, y, c)
def : Pat<(or (srl GR32:$src1, CL:$amt),
              (shl GR32:$src2, (sub 32, CL:$amt))),
          (SHRD32rrCL GR32:$src1, GR32:$src2)>;

def : Pat<(store (or (srl (loadi32 addr:$dst), CL:$amt),
                     (shl GR32:$src2, (sub 32, CL:$amt))), addr:$dst),
          (SHRD32mrCL addr:$dst, GR32:$src2)>;

def : Pat<(or (srl GR32:$src1, (i8 (trunc ECX:$amt))),
              (shl GR32:$src2, (i8 (trunc (sub 32, ECX:$amt))))),
          (SHRD32rrCL GR32:$src1, GR32:$src2)>;

def : Pat<(store (or (srl (loadi32 addr:$dst), (i8 (trunc ECX:$amt))),
                     (shl GR32:$src2, (i8 (trunc (sub 32, ECX:$amt))))),
                 addr:$dst),
          (SHRD32mrCL addr:$dst, GR32:$src2)>;

def : Pat<(shrd GR32:$src1, (i8 imm:$amt1), GR32:$src2, (i8 imm:$amt2)),
          (SHRD32rri8 GR32:$src1, GR32:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shrd (loadi32 addr:$dst), (i8 imm:$amt1),
                       GR32:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHRD32mri8 addr:$dst, GR32:$src2, (i8 imm:$amt1))>;

// (or (x << c) | (y >> (32 - c))) ==> (shld32 x, y, c)
def : Pat<(or (shl GR32:$src1, CL:$amt),
              (srl GR32:$src2, (sub 32, CL:$amt))),
          (SHLD32rrCL GR32:$src1, GR32:$src2)>;

def : Pat<(store (or (shl (loadi32 addr:$dst), CL:$amt),
                     (srl GR32:$src2, (sub 32, CL:$amt))), addr:$dst),
          (SHLD32mrCL addr:$dst, GR32:$src2)>;

def : Pat<(or (shl GR32:$src1, (i8 (trunc ECX:$amt))),
              (srl GR32:$src2, (i8 (trunc (sub 32, ECX:$amt))))),
          (SHLD32rrCL GR32:$src1, GR32:$src2)>;

def : Pat<(store (or (shl (loadi32 addr:$dst), (i8 (trunc ECX:$amt))),
                     (srl GR32:$src2, (i8 (trunc (sub 32, ECX:$amt))))),
                 addr:$dst),
          (SHLD32mrCL addr:$dst, GR32:$src2)>;

def : Pat<(shld GR32:$src1, (i8 imm:$amt1), GR32:$src2, (i8 imm:$amt2)),
          (SHLD32rri8 GR32:$src1, GR32:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shld (loadi32 addr:$dst), (i8 imm:$amt1),
                       GR32:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHLD32mri8 addr:$dst, GR32:$src2, (i8 imm:$amt1))>;

// (or (x >> c) | (y << (16 - c))) ==> (shrd16 x, y, c)
def : Pat<(or (srl GR16:$src1, CL:$amt),
              (shl GR16:$src2, (sub 16, CL:$amt))),
          (SHRD16rrCL GR16:$src1, GR16:$src2)>;

def : Pat<(store (or (srl (loadi16 addr:$dst), CL:$amt),
                     (shl GR16:$src2, (sub 16, CL:$amt))), addr:$dst),
          (SHRD16mrCL addr:$dst, GR16:$src2)>;

def : Pat<(or (srl GR16:$src1, (i8 (trunc CX:$amt))),
              (shl GR16:$src2, (i8 (trunc (sub 16, CX:$amt))))),
          (SHRD16rrCL GR16:$src1, GR16:$src2)>;

def : Pat<(store (or (srl (loadi16 addr:$dst), (i8 (trunc CX:$amt))),
                     (shl GR16:$src2, (i8 (trunc (sub 16, CX:$amt))))),
                 addr:$dst),
          (SHRD16mrCL addr:$dst, GR16:$src2)>;

def : Pat<(shrd GR16:$src1, (i8 imm:$amt1), GR16:$src2, (i8 imm:$amt2)),
          (SHRD16rri8 GR16:$src1, GR16:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shrd (loadi16 addr:$dst), (i8 imm:$amt1),
                       GR16:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHRD16mri8 addr:$dst, GR16:$src2, (i8 imm:$amt1))>;

// (or (x << c) | (y >> (16 - c))) ==> (shld16 x, y, c)
def : Pat<(or (shl GR16:$src1, CL:$amt),
              (srl GR16:$src2, (sub 16, CL:$amt))),
          (SHLD16rrCL GR16:$src1, GR16:$src2)>;

def : Pat<(store (or (shl (loadi16 addr:$dst), CL:$amt),
                     (srl GR16:$src2, (sub 16, CL:$amt))), addr:$dst),
          (SHLD16mrCL addr:$dst, GR16:$src2)>;

def : Pat<(or (shl GR16:$src1, (i8 (trunc CX:$amt))),
              (srl GR16:$src2, (i8 (trunc (sub 16, CX:$amt))))),
          (SHLD16rrCL GR16:$src1, GR16:$src2)>;

def : Pat<(store (or (shl (loadi16 addr:$dst), (i8 (trunc CX:$amt))),
                     (srl GR16:$src2, (i8 (trunc (sub 16, CX:$amt))))),
                 addr:$dst),
          (SHLD16mrCL addr:$dst, GR16:$src2)>;

def : Pat<(shld GR16:$src1, (i8 imm:$amt1), GR16:$src2, (i8 imm:$amt2)),
          (SHLD16rri8 GR16:$src1, GR16:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shld (loadi16 addr:$dst), (i8 imm:$amt1),
                       GR16:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHLD16mri8 addr:$dst, GR16:$src2, (i8 imm:$amt1))>;

//===----------------------------------------------------------------------===//
// Overflow Patterns
//===----------------------------------------------------------------------===//

// Register-Register Addition with Overflow
def : Pat<(parallel (X86add_ovf GR8:$src1, GR8:$src2),
                    (implicit EFLAGS)),
          (ADD8rr GR8:$src1, GR8:$src2)>;

// Register-Register Addition with Overflow
def : Pat<(parallel (X86add_ovf GR16:$src1, GR16:$src2),
                    (implicit EFLAGS)),
          (ADD16rr GR16:$src1, GR16:$src2)>;
def : Pat<(parallel (X86add_ovf GR32:$src1, GR32:$src2),
                    (implicit EFLAGS)),
          (ADD32rr GR32:$src1, GR32:$src2)>;

// Register-Memory Addition with Overflow
def : Pat<(parallel (X86add_ovf GR8:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (ADD8rm GR8:$src1, addr:$src2)>;
def : Pat<(parallel (X86add_ovf GR16:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (ADD16rm GR16:$src1, addr:$src2)>;
def : Pat<(parallel (X86add_ovf GR32:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (ADD32rm GR32:$src1, addr:$src2)>;

// Register-Integer Addition with Overflow
def : Pat<(parallel (X86add_ovf GR8:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (ADD8ri GR8:$src1, imm:$src2)>;

// Register-Integer Addition with Overflow
def : Pat<(parallel (X86add_ovf GR16:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (ADD16ri GR16:$src1, imm:$src2)>;
def : Pat<(parallel (X86add_ovf GR32:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (ADD32ri GR32:$src1, imm:$src2)>;
def : Pat<(parallel (X86add_ovf GR16:$src1, i16immSExt8:$src2),
                    (implicit EFLAGS)),
          (ADD16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(parallel (X86add_ovf GR32:$src1, i32immSExt8:$src2),
                    (implicit EFLAGS)),
          (ADD32ri8 GR32:$src1, i32immSExt8:$src2)>;

// Memory-Register Addition with Overflow
def : Pat<(parallel (store (X86add_ovf (load addr:$dst), GR8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD8mr addr:$dst, GR8:$src2)>;
def : Pat<(parallel (store (X86add_ovf (load addr:$dst), GR16:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD16mr addr:$dst, GR16:$src2)>;
def : Pat<(parallel (store (X86add_ovf (load addr:$dst), GR32:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD32mr addr:$dst, GR32:$src2)>;
def : Pat<(parallel (store (X86add_ovf (loadi8 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD8mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86add_ovf (loadi16 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD16mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86add_ovf (loadi32 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD32mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86add_ovf (load addr:$dst), i16immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD16mi8 addr:$dst, i16immSExt8:$src2)>;
def : Pat<(parallel (store (X86add_ovf (load addr:$dst), i32immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD32mi8 addr:$dst, i32immSExt8:$src2)>;

// Register-Register Subtraction with Overflow
def : Pat<(parallel (X86sub_ovf GR8:$src1, GR8:$src2),
                    (implicit EFLAGS)),
          (SUB8rr GR8:$src1, GR8:$src2)>;
def : Pat<(parallel (X86sub_ovf GR16:$src1, GR16:$src2),
                    (implicit EFLAGS)),
          (SUB16rr GR16:$src1, GR16:$src2)>;
def : Pat<(parallel (X86sub_ovf GR32:$src1, GR32:$src2),
                    (implicit EFLAGS)),
          (SUB32rr GR32:$src1, GR32:$src2)>;

// Register-Memory Subtraction with Overflow
def : Pat<(parallel (X86sub_ovf GR8:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (SUB8rm GR8:$src1, addr:$src2)>;
def : Pat<(parallel (X86sub_ovf GR16:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (SUB16rm GR16:$src1, addr:$src2)>;
def : Pat<(parallel (X86sub_ovf GR32:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (SUB32rm GR32:$src1, addr:$src2)>;

// Register-Integer Subtraction with Overflow
def : Pat<(parallel (X86sub_ovf GR8:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (SUB8ri GR8:$src1, imm:$src2)>;
def : Pat<(parallel (X86sub_ovf GR16:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (SUB16ri GR16:$src1, imm:$src2)>;
def : Pat<(parallel (X86sub_ovf GR32:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (SUB32ri GR32:$src1, imm:$src2)>;
def : Pat<(parallel (X86sub_ovf GR16:$src1, i16immSExt8:$src2),
                    (implicit EFLAGS)),
          (SUB16ri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(parallel (X86sub_ovf GR32:$src1, i32immSExt8:$src2),
                    (implicit EFLAGS)),
          (SUB32ri8 GR32:$src1, i32immSExt8:$src2)>;

// Memory-Register Subtraction with Overflow
def : Pat<(parallel (store (X86sub_ovf (load addr:$dst), GR8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB8mr addr:$dst, GR8:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (load addr:$dst), GR16:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB16mr addr:$dst, GR16:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (load addr:$dst), GR32:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB32mr addr:$dst, GR32:$src2)>;

// Memory-Integer Subtraction with Overflow
def : Pat<(parallel (store (X86sub_ovf (loadi8 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB8mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (loadi16 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB16mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (loadi32 addr:$dst), imm:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB32mi addr:$dst, imm:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (load addr:$dst), i16immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB16mi8 addr:$dst, i16immSExt8:$src2)>;
def : Pat<(parallel (store (X86sub_ovf (load addr:$dst), i32immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB32mi8 addr:$dst, i32immSExt8:$src2)>;


// Register-Register Signed Integer Multiply with Overflow
def : Pat<(parallel (X86smul_ovf GR16:$src1, GR16:$src2),
                    (implicit EFLAGS)),
          (IMUL16rr GR16:$src1, GR16:$src2)>;
def : Pat<(parallel (X86smul_ovf GR32:$src1, GR32:$src2),
                    (implicit EFLAGS)),
          (IMUL32rr GR32:$src1, GR32:$src2)>;

// Register-Memory Signed Integer Multiply with Overflow
def : Pat<(parallel (X86smul_ovf GR16:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (IMUL16rm GR16:$src1, addr:$src2)>;
def : Pat<(parallel (X86smul_ovf GR32:$src1, (load addr:$src2)),
                    (implicit EFLAGS)),
          (IMUL32rm GR32:$src1, addr:$src2)>;

// Register-Integer Signed Integer Multiply with Overflow
def : Pat<(parallel (X86smul_ovf GR16:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (IMUL16rri GR16:$src1, imm:$src2)>;
def : Pat<(parallel (X86smul_ovf GR32:$src1, imm:$src2),
                    (implicit EFLAGS)),
          (IMUL32rri GR32:$src1, imm:$src2)>;
def : Pat<(parallel (X86smul_ovf GR16:$src1, i16immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL16rri8 GR16:$src1, i16immSExt8:$src2)>;
def : Pat<(parallel (X86smul_ovf GR32:$src1, i32immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL32rri8 GR32:$src1, i32immSExt8:$src2)>;

// Memory-Integer Signed Integer Multiply with Overflow
def : Pat<(parallel (X86smul_ovf (load addr:$src1), imm:$src2),
                    (implicit EFLAGS)),
          (IMUL16rmi addr:$src1, imm:$src2)>;
def : Pat<(parallel (X86smul_ovf (load addr:$src1), imm:$src2),
                    (implicit EFLAGS)),
          (IMUL32rmi addr:$src1, imm:$src2)>;
def : Pat<(parallel (X86smul_ovf (load addr:$src1), i16immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL16rmi8 addr:$src1, i16immSExt8:$src2)>;
def : Pat<(parallel (X86smul_ovf (load addr:$src1), i32immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL32rmi8 addr:$src1, i32immSExt8:$src2)>;

// Optimize multiple with overflow by 2.
let AddedComplexity = 2 in {
def : Pat<(parallel (X86smul_ovf GR16:$src1, 2),
                    (implicit EFLAGS)),
          (ADD16rr GR16:$src1, GR16:$src1)>;

def : Pat<(parallel (X86smul_ovf GR32:$src1, 2),
                    (implicit EFLAGS)),
          (ADD32rr GR32:$src1, GR32:$src1)>;
}

//===----------------------------------------------------------------------===//
// Floating Point Stack Support
//===----------------------------------------------------------------------===//

include "X86InstrFPStack.td"

//===----------------------------------------------------------------------===//
// X86-64 Support
//===----------------------------------------------------------------------===//

include "X86Instr64bit.td"

//===----------------------------------------------------------------------===//
// XMM Floating point support (requires SSE / SSE2)
//===----------------------------------------------------------------------===//

include "X86InstrSSE.td"

//===----------------------------------------------------------------------===//
// MMX and XMM Packed Integer support (requires MMX, SSE, and SSE2)
//===----------------------------------------------------------------------===//

include "X86InstrMMX.td"
