//====- X86Instr64bit.td - Describe X86-64 Instructions ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86-64 instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand Definitions.
//

// 64-bits but only 32 bits are significant.
def i64i32imm  : Operand<i64>;

// 64-bits but only 32 bits are significant, and those bits are treated as being
// pc relative.
def i64i32imm_pcrel : Operand<i64> {
  let PrintMethod = "print_pcrel_imm";
}


// 64-bits but only 8 bits are significant.
def i64i8imm   : Operand<i64> {
  let ParserMatchClass = ImmSExt8AsmOperand;
}

def lea64mem : Operand<i64> {
  let PrintMethod = "printlea64mem";
  let MIOperandInfo = (ops GR64, i8imm, GR64_NOSP, i32imm);
  let ParserMatchClass = X86MemAsmOperand;
}

def lea64_32mem : Operand<i32> {
  let PrintMethod = "printlea64_32mem";
  let AsmOperandLowerMethod = "lower_lea64_32mem";
  let MIOperandInfo = (ops GR32, i8imm, GR32_NOSP, i32imm);
  let ParserMatchClass = X86MemAsmOperand;
}

//===----------------------------------------------------------------------===//
// Complex Pattern Definitions.
//
def lea64addr : ComplexPattern<i64, 4, "SelectLEAAddr",
                        [add, sub, mul, X86mul_imm, shl, or, frameindex,
                         X86WrapperRIP], []>;

def tls64addr : ComplexPattern<i64, 4, "SelectTLSADDRAddr",
                               [tglobaltlsaddr], []>;

//===----------------------------------------------------------------------===//
// Pattern fragments.
//

def i64immSExt8  : PatLeaf<(i64 imm), [{
  // i64immSExt8 predicate - True if the 64-bit immediate fits in a 8-bit
  // sign extended field.
  return (int64_t)N->getZExtValue() == (int8_t)N->getZExtValue();
}]>;

def i64immSExt32  : PatLeaf<(i64 imm), [{
  // i64immSExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // sign extended field.
  return (int64_t)N->getZExtValue() == (int32_t)N->getZExtValue();
}]>;

def i64immZExt32  : PatLeaf<(i64 imm), [{
  // i64immZExt32 predicate - True if the 64-bit immediate fits in a 32-bit
  // unsignedsign extended field.
  return (uint64_t)N->getZExtValue() == (uint32_t)N->getZExtValue();
}]>;

def sextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (sextloadi8 node:$ptr))>;
def sextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (sextloadi16 node:$ptr))>;
def sextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (sextloadi32 node:$ptr))>;

def zextloadi64i1  : PatFrag<(ops node:$ptr), (i64 (zextloadi1 node:$ptr))>;
def zextloadi64i8  : PatFrag<(ops node:$ptr), (i64 (zextloadi8 node:$ptr))>;
def zextloadi64i16 : PatFrag<(ops node:$ptr), (i64 (zextloadi16 node:$ptr))>;
def zextloadi64i32 : PatFrag<(ops node:$ptr), (i64 (zextloadi32 node:$ptr))>;

def extloadi64i1   : PatFrag<(ops node:$ptr), (i64 (extloadi1 node:$ptr))>;
def extloadi64i8   : PatFrag<(ops node:$ptr), (i64 (extloadi8 node:$ptr))>;
def extloadi64i16  : PatFrag<(ops node:$ptr), (i64 (extloadi16 node:$ptr))>;
def extloadi64i32  : PatFrag<(ops node:$ptr), (i64 (extloadi32 node:$ptr))>;

//===----------------------------------------------------------------------===//
// Instruction list...
//

// ADJCALLSTACKDOWN/UP implicitly use/def RSP because they may be expanded into
// a stack adjustment and the codegen must know that they may modify the stack
// pointer before prolog-epilog rewriting occurs.
// Pessimistically assume ADJCALLSTACKDOWN / ADJCALLSTACKUP will become
// sub / add which can clobber EFLAGS.
let Defs = [RSP, EFLAGS], Uses = [RSP] in {
def ADJCALLSTACKDOWN64 : I<0, Pseudo, (outs), (ins i32imm:$amt),
                           "#ADJCALLSTACKDOWN",
                           [(X86callseq_start timm:$amt)]>,
                          Requires<[In64BitMode]>;
def ADJCALLSTACKUP64   : I<0, Pseudo, (outs), (ins i32imm:$amt1, i32imm:$amt2),
                           "#ADJCALLSTACKUP",
                           [(X86callseq_end timm:$amt1, timm:$amt2)]>,
                          Requires<[In64BitMode]>;
}

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. RSP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
              XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
      Uses = [RSP] in {
      
    // NOTE: this pattern doesn't match "X86call imm", because we do not know
    // that the offset between an arbitrary immediate and the call will fit in
    // the 32-bit pcrel field that we have.
    def CALL64pcrel32 : Ii32<0xE8, RawFrm,
                          (outs), (ins i64i32imm_pcrel:$dst, variable_ops),
                          "call\t$dst", []>,
                        Requires<[In64BitMode, NotWin64]>;
    def CALL64r       : I<0xFF, MRM2r, (outs), (ins GR64:$dst, variable_ops),
                          "call\t{*}$dst", [(X86call GR64:$dst)]>,
                        Requires<[NotWin64]>;
    def CALL64m       : I<0xFF, MRM2m, (outs), (ins i64mem:$dst, variable_ops),
                          "call\t{*}$dst", [(X86call (loadi64 addr:$dst))]>,
                        Requires<[NotWin64]>;
  }

  // FIXME: We need to teach codegen about single list of call-clobbered registers.
let isCall = 1 in
  // All calls clobber the non-callee saved registers. RSP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Defs = [RAX, RCX, RDX, R8, R9, R10, R11,
              FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
              MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
              XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, EFLAGS],
      Uses = [RSP] in {
    def WINCALL64pcrel32 : I<0xE8, RawFrm,
                             (outs), (ins i64i32imm_pcrel:$dst, variable_ops),
                             "call\t$dst", []>,
                           Requires<[IsWin64]>;
    def WINCALL64r       : I<0xFF, MRM2r, (outs), (ins GR64:$dst, variable_ops),
                             "call\t{*}$dst",
                             [(X86call GR64:$dst)]>, Requires<[IsWin64]>;
    def WINCALL64m       : I<0xFF, MRM2m, (outs), (ins i64mem:$dst, variable_ops),
                             "call\t{*}$dst",
                             [(X86call (loadi64 addr:$dst))]>, Requires<[IsWin64]>;
  }


let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
def TCRETURNdi64 : I<0, Pseudo, (outs), (ins i64imm:$dst, i32imm:$offset,
                                         variable_ops),
                 "#TC_RETURN $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
def TCRETURNri64 : I<0, Pseudo, (outs), (ins GR64:$dst, i32imm:$offset,
                                         variable_ops),
                 "#TC_RETURN $dst $offset",
                 []>;


let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1 in
  def TAILJMPr64 : I<0xFF, MRM4r, (outs), (ins GR64:$dst),
                   "jmp{q}\t{*}$dst  # TAILCALL",
                   []>;     

// Branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def JMP64r     : I<0xFF, MRM4r, (outs), (ins GR64:$dst), "jmp{q}\t{*}$dst",
                     [(brind GR64:$dst)]>;
  def JMP64m     : I<0xFF, MRM4m, (outs), (ins i64mem:$dst), "jmp{q}\t{*}$dst",
                     [(brind (loadi64 addr:$dst))]>;
}

//===----------------------------------------------------------------------===//
// EH Pseudo Instructions
//
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1 in {
def EH_RETURN64   : I<0xC3, RawFrm, (outs), (ins GR64:$addr),
                     "ret\t#eh_return, addr: $addr",
                     [(X86ehret GR64:$addr)]>;

}

//===----------------------------------------------------------------------===//
//  Miscellaneous Instructions...
//
let Defs = [RBP,RSP], Uses = [RBP,RSP], mayLoad = 1, neverHasSideEffects = 1 in
def LEAVE64  : I<0xC9, RawFrm,
                 (outs), (ins), "leave", []>;
let Defs = [RSP], Uses = [RSP], neverHasSideEffects=1 in {
let mayLoad = 1 in
def POP64r   : I<0x58, AddRegFrm,
                 (outs GR64:$reg), (ins), "pop{q}\t$reg", []>;
let mayStore = 1 in
def PUSH64r  : I<0x50, AddRegFrm,
                 (outs), (ins GR64:$reg), "push{q}\t$reg", []>;
}

let Defs = [RSP], Uses = [RSP], neverHasSideEffects = 1, mayStore = 1 in {
def PUSH64i8   : Ii8<0x6a, RawFrm, (outs), (ins i8imm:$imm), 
                     "push{q}\t$imm", []>;
def PUSH64i16  : Ii16<0x68, RawFrm, (outs), (ins i16imm:$imm), 
                      "push{q}\t$imm", []>;
def PUSH64i32  : Ii32<0x68, RawFrm, (outs), (ins i32imm:$imm), 
                      "push{q}\t$imm", []>;
}

let Defs = [RSP, EFLAGS], Uses = [RSP], mayLoad = 1 in
def POPFQ    : I<0x9D, RawFrm, (outs), (ins), "popf", []>, REX_W;
let Defs = [RSP], Uses = [RSP, EFLAGS], mayStore = 1 in
def PUSHFQ   : I<0x9C, RawFrm, (outs), (ins), "pushf", []>;

def LEA64_32r : I<0x8D, MRMSrcMem,
                  (outs GR32:$dst), (ins lea64_32mem:$src),
                  "lea{l}\t{$src|$dst}, {$dst|$src}",
                  [(set GR32:$dst, lea32addr:$src)]>, Requires<[In64BitMode]>;

let isReMaterializable = 1 in
def LEA64r   : RI<0x8D, MRMSrcMem, (outs GR64:$dst), (ins lea64mem:$src),
                  "lea{q}\t{$src|$dst}, {$dst|$src}",
                  [(set GR64:$dst, lea64addr:$src)]>;

let isTwoAddress = 1 in
def BSWAP64r : RI<0xC8, AddRegFrm, (outs GR64:$dst), (ins GR64:$src),
                  "bswap{q}\t$dst", 
                  [(set GR64:$dst, (bswap GR64:$src))]>, TB;

// Bit scan instructions.
let Defs = [EFLAGS] in {
def BSF64rr  : RI<0xBC, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, (X86bsf GR64:$src)), (implicit EFLAGS)]>, TB;
def BSF64rm  : RI<0xBC, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsf{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, (X86bsf (loadi64 addr:$src))),
                   (implicit EFLAGS)]>, TB;

def BSR64rr  : RI<0xBD, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, (X86bsr GR64:$src)), (implicit EFLAGS)]>, TB;
def BSR64rm  : RI<0xBD, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                  "bsr{q}\t{$src, $dst|$dst, $src}",
                  [(set GR64:$dst, (X86bsr (loadi64 addr:$src))),
                   (implicit EFLAGS)]>, TB;
} // Defs = [EFLAGS]

// Repeat string ops
let Defs = [RCX,RDI,RSI], Uses = [RCX,RDI,RSI] in
def REP_MOVSQ : RI<0xA5, RawFrm, (outs), (ins), "{rep;movsq|rep movsq}",
                   [(X86rep_movs i64)]>, REP;
let Defs = [RCX,RDI], Uses = [RAX,RCX,RDI] in
def REP_STOSQ : RI<0xAB, RawFrm, (outs), (ins), "{rep;stosq|rep stosq}",
                   [(X86rep_stos i64)]>, REP;

// Fast system-call instructions
def SYSEXIT64 : RI<0x35, RawFrm,
                   (outs), (ins), "sysexit", []>, TB;

//===----------------------------------------------------------------------===//
//  Move Instructions...
//

let neverHasSideEffects = 1 in
def MOV64rr : RI<0x89, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}", []>;

let isReMaterializable = 1, isAsCheapAsAMove = 1  in {
def MOV64ri : RIi64<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64imm:$src),
                    "movabs{q}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, imm:$src)]>;
def MOV64ri32 : RIi32<0xC7, MRM0r, (outs GR64:$dst), (ins i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(set GR64:$dst, i64immSExt32:$src)]>;
}

let canFoldAsLoad = 1 in
def MOV64rm : RI<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(set GR64:$dst, (load addr:$src))]>;

def MOV64mr : RI<0x89, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                 "mov{q}\t{$src, $dst|$dst, $src}",
                 [(store GR64:$src, addr:$dst)]>;
def MOV64mi32 : RIi32<0xC7, MRM0m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "mov{q}\t{$src, $dst|$dst, $src}",
                      [(store i64immSExt32:$src, addr:$dst)]>;

// Sign/Zero extenders

// MOVSX64rr8 always has a REX prefix and it has an 8-bit register
// operand, which makes it a rare instruction with an 8-bit register
// operand that can never access an h register. If support for h registers
// were generalized, this would require a special register class.
def MOVSX64rr8 : RI<0xBE, MRMSrcReg, (outs GR64:$dst), (ins GR8 :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR8:$src))]>, TB;
def MOVSX64rm8 : RI<0xBE, MRMSrcMem, (outs GR64:$dst), (ins i8mem :$src),
                    "movs{bq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i8 addr:$src))]>, TB;
def MOVSX64rr16: RI<0xBF, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR16:$src))]>, TB;
def MOVSX64rm16: RI<0xBF, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                    "movs{wq|x}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i16 addr:$src))]>, TB;
def MOVSX64rr32: RI<0x63, MRMSrcReg, (outs GR64:$dst), (ins GR32:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sext GR32:$src))]>;
def MOVSX64rm32: RI<0x63, MRMSrcMem, (outs GR64:$dst), (ins i32mem:$src),
                    "movs{lq|xd}\t{$src, $dst|$dst, $src}",
                    [(set GR64:$dst, (sextloadi64i32 addr:$src))]>;

// Use movzbl instead of movzbq when the destination is a register; it's
// equivalent due to implicit zero-extending, and it has a smaller encoding.
def MOVZX64rr8 : I<0xB6, MRMSrcReg, (outs GR64:$dst), (ins GR8 :$src),
                   "movz{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR64:$dst, (zext GR8:$src))]>, TB;
def MOVZX64rm8 : I<0xB6, MRMSrcMem, (outs GR64:$dst), (ins i8mem :$src),
                   "movz{bl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR64:$dst, (zextloadi64i8 addr:$src))]>, TB;
// Use movzwl instead of movzwq when the destination is a register; it's
// equivalent due to implicit zero-extending, and it has a smaller encoding.
def MOVZX64rr16: I<0xB7, MRMSrcReg, (outs GR64:$dst), (ins GR16:$src),
                   "movz{wl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR64:$dst, (zext GR16:$src))]>, TB;
def MOVZX64rm16: I<0xB7, MRMSrcMem, (outs GR64:$dst), (ins i16mem:$src),
                   "movz{wl|x}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                   [(set GR64:$dst, (zextloadi64i16 addr:$src))]>, TB;

// There's no movzlq instruction, but movl can be used for this purpose, using
// implicit zero-extension. The preferred way to do 32-bit-to-64-bit zero
// extension on x86-64 is to use a SUBREG_TO_REG to utilize implicit
// zero-extension, however this isn't possible when the 32-bit value is
// defined by a truncate or is copied from something where the high bits aren't
// necessarily all zero. In such cases, we fall back to these explicit zext
// instructions.
def MOVZX64rr32 : I<0x89, MRMDestReg, (outs GR64:$dst), (ins GR32:$src),
                    "mov{l}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                    [(set GR64:$dst, (zext GR32:$src))]>;
def MOVZX64rm32 : I<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i32mem:$src),
                    "mov{l}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                    [(set GR64:$dst, (zextloadi64i32 addr:$src))]>;

// Any instruction that defines a 32-bit result leaves the high half of the
// register. Truncate can be lowered to EXTRACT_SUBREG. CopyFromReg may
// be copying from a truncate. And x86's cmov doesn't do anything if the
// condition is false. But any other 32-bit operation will zero-extend
// up to 64 bits.
def def32 : PatLeaf<(i32 GR32:$src), [{
  return N->getOpcode() != ISD::TRUNCATE &&
         N->getOpcode() != TargetInstrInfo::EXTRACT_SUBREG &&
         N->getOpcode() != ISD::CopyFromReg &&
         N->getOpcode() != X86ISD::CMOV;
}]>;

// In the case of a 32-bit def that is known to implicitly zero-extend,
// we can use a SUBREG_TO_REG.
def : Pat<(i64 (zext def32:$src)),
          (SUBREG_TO_REG (i64 0), GR32:$src, x86_subreg_32bit)>;

let neverHasSideEffects = 1 in {
  let Defs = [RAX], Uses = [EAX] in
  def CDQE : RI<0x98, RawFrm, (outs), (ins),
               "{cltq|cdqe}", []>;     // RAX = signext(EAX)

  let Defs = [RAX,RDX], Uses = [RAX] in
  def CQO  : RI<0x99, RawFrm, (outs), (ins),
                "{cqto|cqo}", []>; // RDX:RAX = signext(RAX)
}

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions...
//

let Defs = [EFLAGS] in {
let isTwoAddress = 1 in {
let isConvertibleToThreeAddress = 1 in {
let isCommutable = 1 in
// Register-Register Addition
def ADD64rr    : RI<0x01, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (add GR64:$src1, GR64:$src2)),
                     (implicit EFLAGS)]>;

// Register-Integer Addition
def ADD64ri8  : RIi8<0x83, MRM0r, (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, i64immSExt8:$src2)),
                      (implicit EFLAGS)]>;
def ADD64ri32 : RIi32<0x81, MRM0r, (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (add GR64:$src1, i64immSExt32:$src2)),
                       (implicit EFLAGS)]>;
} // isConvertibleToThreeAddress

// Register-Memory Addition
def ADD64rm     : RI<0x03, MRMSrcMem, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                     "add{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, (add GR64:$src1, (load addr:$src2))),
                      (implicit EFLAGS)]>;
} // isTwoAddress

// Memory-Register Addition
def ADD64mr  : RI<0x01, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "add{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (add (load addr:$dst), GR64:$src2), addr:$dst),
                   (implicit EFLAGS)]>;
def ADD64mi8 : RIi8<0x83, MRM0m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "add{q}\t{$src2, $dst|$dst, $src2}",
                [(store (add (load addr:$dst), i64immSExt8:$src2), addr:$dst),
                 (implicit EFLAGS)]>;
def ADD64mi32 : RIi32<0x81, MRM0m, (outs), (ins i64mem:$dst, i64i32imm :$src2),
                      "add{q}\t{$src2, $dst|$dst, $src2}",
               [(store (add (load addr:$dst), i64immSExt32:$src2), addr:$dst),
                (implicit EFLAGS)]>;

let Uses = [EFLAGS] in {
let isTwoAddress = 1 in {
let isCommutable = 1 in
def ADC64rr  : RI<0x11, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, GR64:$src2))]>;

def ADC64rm  : RI<0x13, MRMSrcMem , (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (adde GR64:$src1, (load addr:$src2)))]>;

def ADC64ri8 : RIi8<0x83, MRM2r, (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (adde GR64:$src1, i64immSExt8:$src2))]>;
def ADC64ri32 : RIi32<0x81, MRM2r, (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (adde GR64:$src1, i64immSExt32:$src2))]>;
} // isTwoAddress

def ADC64mr  : RI<0x11, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                  "adc{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (adde (load addr:$dst), GR64:$src2), addr:$dst)]>;
def ADC64mi8 : RIi8<0x83, MRM2m, (outs), (ins i64mem:$dst, i64i8imm :$src2),
                    "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;
def ADC64mi32 : RIi32<0x81, MRM2m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "adc{q}\t{$src2, $dst|$dst, $src2}",
                 [(store (adde (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;
} // Uses = [EFLAGS]

let isTwoAddress = 1 in {
// Register-Register Subtraction
def SUB64rr  : RI<0x29, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sub GR64:$src1, GR64:$src2)),
                   (implicit EFLAGS)]>;

// Register-Memory Subtraction
def SUB64rm  : RI<0x2B, MRMSrcMem, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sub GR64:$src1, (load addr:$src2))),
                   (implicit EFLAGS)]>;

// Register-Integer Subtraction
def SUB64ri8 : RIi8<0x83, MRM5r, (outs GR64:$dst),
                                 (ins GR64:$src1, i64i8imm:$src2),
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sub GR64:$src1, i64immSExt8:$src2)),
                     (implicit EFLAGS)]>;
def SUB64ri32 : RIi32<0x81, MRM5r, (outs GR64:$dst),
                                   (ins GR64:$src1, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sub GR64:$src1, i64immSExt32:$src2)),
                       (implicit EFLAGS)]>;
} // isTwoAddress

// Memory-Register Subtraction
def SUB64mr  : RI<0x29, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sub{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sub (load addr:$dst), GR64:$src2), addr:$dst),
                   (implicit EFLAGS)]>;

// Memory-Integer Subtraction
def SUB64mi8 : RIi8<0x83, MRM5m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sub{q}\t{$src2, $dst|$dst, $src2}",
                    [(store (sub (load addr:$dst), i64immSExt8:$src2),
                            addr:$dst),
                     (implicit EFLAGS)]>;
def SUB64mi32 : RIi32<0x81, MRM5m, (outs), (ins i64mem:$dst, i64i32imm:$src2),
                      "sub{q}\t{$src2, $dst|$dst, $src2}",
                      [(store (sub (load addr:$dst), i64immSExt32:$src2),
                              addr:$dst),
                       (implicit EFLAGS)]>;

let Uses = [EFLAGS] in {
let isTwoAddress = 1 in {
def SBB64rr    : RI<0x19, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, GR64:$src2))]>;

def SBB64rm  : RI<0x1B, MRMSrcMem, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (sube GR64:$src1, (load addr:$src2)))]>;

def SBB64ri8 : RIi8<0x83, MRM3r, (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (sube GR64:$src1, i64immSExt8:$src2))]>;
def SBB64ri32 : RIi32<0x81, MRM3r, (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (sube GR64:$src1, i64immSExt32:$src2))]>;
} // isTwoAddress

def SBB64mr  : RI<0x19, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                  "sbb{q}\t{$src2, $dst|$dst, $src2}",
                  [(store (sube (load addr:$dst), GR64:$src2), addr:$dst)]>;
def SBB64mi8 : RIi8<0x83, MRM3m, (outs), (ins i64mem:$dst, i64i8imm :$src2), 
                    "sbb{q}\t{$src2, $dst|$dst, $src2}",
               [(store (sube (load addr:$dst), i64immSExt8:$src2), addr:$dst)]>;
def SBB64mi32 : RIi32<0x81, MRM3m, (outs), (ins i64mem:$dst, i64i32imm:$src2), 
                      "sbb{q}\t{$src2, $dst|$dst, $src2}",
              [(store (sube (load addr:$dst), i64immSExt32:$src2), addr:$dst)]>;
} // Uses = [EFLAGS]
} // Defs = [EFLAGS]

// Unsigned multiplication
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX], neverHasSideEffects = 1 in {
def MUL64r : RI<0xF7, MRM4r, (outs), (ins GR64:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*GR64
let mayLoad = 1 in
def MUL64m : RI<0xF7, MRM4m, (outs), (ins i64mem:$src),
                "mul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]

// Signed multiplication
def IMUL64r : RI<0xF7, MRM5r, (outs), (ins GR64:$src),
                 "imul{q}\t$src", []>;         // RAX,RDX = RAX*GR64
let mayLoad = 1 in
def IMUL64m : RI<0xF7, MRM5m, (outs), (ins i64mem:$src),
                 "imul{q}\t$src", []>;         // RAX,RDX = RAX*[mem64]
}

let Defs = [EFLAGS] in {
let isTwoAddress = 1 in {
let isCommutable = 1 in
// Register-Register Signed Integer Multiplication
def IMUL64rr : RI<0xAF, MRMSrcReg, (outs GR64:$dst),
                                   (ins GR64:$src1, GR64:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (mul GR64:$src1, GR64:$src2)),
                   (implicit EFLAGS)]>, TB;

// Register-Memory Signed Integer Multiplication
def IMUL64rm : RI<0xAF, MRMSrcMem, (outs GR64:$dst),
                                   (ins GR64:$src1, i64mem:$src2),
                  "imul{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (mul GR64:$src1, (load addr:$src2))),
                   (implicit EFLAGS)]>, TB;
} // isTwoAddress

// Suprisingly enough, these are not two address instructions!

// Register-Integer Signed Integer Multiplication
def IMUL64rri8 : RIi8<0x6B, MRMSrcReg,                      // GR64 = GR64*I8
                      (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, (mul GR64:$src1, i64immSExt8:$src2)),
                       (implicit EFLAGS)]>;
def IMUL64rri32 : RIi32<0x69, MRMSrcReg,                    // GR64 = GR64*I32
                        (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                       [(set GR64:$dst, (mul GR64:$src1, i64immSExt32:$src2)),
                        (implicit EFLAGS)]>;

// Memory-Integer Signed Integer Multiplication
def IMUL64rmi8 : RIi8<0x6B, MRMSrcMem,                      // GR64 = [mem64]*I8
                      (outs GR64:$dst), (ins i64mem:$src1, i64i8imm: $src2),
                      "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                      [(set GR64:$dst, (mul (load addr:$src1),
                                            i64immSExt8:$src2)),
                       (implicit EFLAGS)]>;
def IMUL64rmi32 : RIi32<0x69, MRMSrcMem,                   // GR64 = [mem64]*I32
                        (outs GR64:$dst), (ins i64mem:$src1, i64i32imm:$src2),
                        "imul{q}\t{$src2, $src1, $dst|$dst, $src1, $src2}",
                        [(set GR64:$dst, (mul (load addr:$src1),
                                              i64immSExt32:$src2)),
                         (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Unsigned division / remainder
let Defs = [RAX,RDX,EFLAGS], Uses = [RAX,RDX] in {
def DIV64r : RI<0xF7, MRM6r, (outs), (ins GR64:$src),        // RDX:RAX/r64 = RAX,RDX
                "div{q}\t$src", []>;
// Signed division / remainder
def IDIV64r: RI<0xF7, MRM7r, (outs), (ins GR64:$src),        // RDX:RAX/r64 = RAX,RDX
                "idiv{q}\t$src", []>;
let mayLoad = 1 in {
def DIV64m : RI<0xF7, MRM6m, (outs), (ins i64mem:$src),      // RDX:RAX/[mem64] = RAX,RDX
                "div{q}\t$src", []>;
def IDIV64m: RI<0xF7, MRM7m, (outs), (ins i64mem:$src),      // RDX:RAX/[mem64] = RAX,RDX
                "idiv{q}\t$src", []>;
}
}

// Unary instructions
let Defs = [EFLAGS], CodeSize = 2 in {
let isTwoAddress = 1 in
def NEG64r : RI<0xF7, MRM3r, (outs GR64:$dst), (ins GR64:$src), "neg{q}\t$dst",
                [(set GR64:$dst, (ineg GR64:$src)),
                 (implicit EFLAGS)]>;
def NEG64m : RI<0xF7, MRM3m, (outs), (ins i64mem:$dst), "neg{q}\t$dst",
                [(store (ineg (loadi64 addr:$dst)), addr:$dst),
                 (implicit EFLAGS)]>;

let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in
def INC64r : RI<0xFF, MRM0r, (outs GR64:$dst), (ins GR64:$src), "inc{q}\t$dst",
                [(set GR64:$dst, (add GR64:$src, 1)),
                 (implicit EFLAGS)]>;
def INC64m : RI<0xFF, MRM0m, (outs), (ins i64mem:$dst), "inc{q}\t$dst",
                [(store (add (loadi64 addr:$dst), 1), addr:$dst),
                 (implicit EFLAGS)]>;

let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in
def DEC64r : RI<0xFF, MRM1r, (outs GR64:$dst), (ins GR64:$src), "dec{q}\t$dst",
                [(set GR64:$dst, (add GR64:$src, -1)),
                 (implicit EFLAGS)]>;
def DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst), "dec{q}\t$dst",
                [(store (add (loadi64 addr:$dst), -1), addr:$dst),
                 (implicit EFLAGS)]>;

// In 64-bit mode, single byte INC and DEC cannot be encoded.
let isTwoAddress = 1, isConvertibleToThreeAddress = 1 in {
// Can transform into LEA.
def INC64_16r : I<0xFF, MRM0r, (outs GR16:$dst), (ins GR16:$src), "inc{w}\t$dst",
                  [(set GR16:$dst, (add GR16:$src, 1)),
                   (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def INC64_32r : I<0xFF, MRM0r, (outs GR32:$dst), (ins GR32:$src), "inc{l}\t$dst",
                  [(set GR32:$dst, (add GR32:$src, 1)),
                   (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
def DEC64_16r : I<0xFF, MRM1r, (outs GR16:$dst), (ins GR16:$src), "dec{w}\t$dst",
                  [(set GR16:$dst, (add GR16:$src, -1)),
                   (implicit EFLAGS)]>,
                OpSize, Requires<[In64BitMode]>;
def DEC64_32r : I<0xFF, MRM1r, (outs GR32:$dst), (ins GR32:$src), "dec{l}\t$dst",
                  [(set GR32:$dst, (add GR32:$src, -1)),
                   (implicit EFLAGS)]>,
                Requires<[In64BitMode]>;
} // isConvertibleToThreeAddress

// These are duplicates of their 32-bit counterparts. Only needed so X86 knows
// how to unfold them.
let isTwoAddress = 0, CodeSize = 2 in {
  def INC64_16m : I<0xFF, MRM0m, (outs), (ins i16mem:$dst), "inc{w}\t$dst",
                    [(store (add (loadi16 addr:$dst), 1), addr:$dst),
                     (implicit EFLAGS)]>,
                  OpSize, Requires<[In64BitMode]>;
  def INC64_32m : I<0xFF, MRM0m, (outs), (ins i32mem:$dst), "inc{l}\t$dst",
                    [(store (add (loadi32 addr:$dst), 1), addr:$dst),
                     (implicit EFLAGS)]>,
                  Requires<[In64BitMode]>;
  def DEC64_16m : I<0xFF, MRM1m, (outs), (ins i16mem:$dst), "dec{w}\t$dst",
                    [(store (add (loadi16 addr:$dst), -1), addr:$dst),
                     (implicit EFLAGS)]>,
                  OpSize, Requires<[In64BitMode]>;
  def DEC64_32m : I<0xFF, MRM1m, (outs), (ins i32mem:$dst), "dec{l}\t$dst",
                    [(store (add (loadi32 addr:$dst), -1), addr:$dst),
                     (implicit EFLAGS)]>,
                  Requires<[In64BitMode]>;
}
} // Defs = [EFLAGS], CodeSize


let Defs = [EFLAGS] in {
// Shift instructions
let isTwoAddress = 1 in {
let Uses = [CL] in
def SHL64rCL : RI<0xD3, MRM4r, (outs GR64:$dst), (ins GR64:$src),
                  "shl{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (shl GR64:$src, CL))]>;
let isConvertibleToThreeAddress = 1 in   // Can transform into LEA.
def SHL64ri  : RIi8<0xC1, MRM4r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                    "shl{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (shl GR64:$src1, (i8 imm:$src2)))]>;
// NOTE: We don't use shifts of a register by one, because 'add reg,reg' is
// cheaper.
} // isTwoAddress

let Uses = [CL] in
def SHL64mCL : RI<0xD3, MRM4m, (outs), (ins i64mem:$dst),
                  "shl{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (shl (loadi64 addr:$dst), CL), addr:$dst)]>;
def SHL64mi : RIi8<0xC1, MRM4m, (outs), (ins i64mem:$dst, i8imm:$src),
                  "shl{q}\t{$src, $dst|$dst, $src}",
                 [(store (shl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHL64m1 : RI<0xD1, MRM4m, (outs), (ins i64mem:$dst),
                  "shl{q}\t$dst",
                 [(store (shl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
let Uses = [CL] in
def SHR64rCL : RI<0xD3, MRM5r, (outs GR64:$dst), (ins GR64:$src),
                  "shr{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (srl GR64:$src, CL))]>;
def SHR64ri : RIi8<0xC1, MRM5r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                  "shr{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (srl GR64:$src1, (i8 imm:$src2)))]>;
def SHR64r1  : RI<0xD1, MRM5r, (outs GR64:$dst), (ins GR64:$src1),
                 "shr{q}\t$dst",
                 [(set GR64:$dst, (srl GR64:$src1, (i8 1)))]>;
} // isTwoAddress

let Uses = [CL] in
def SHR64mCL : RI<0xD3, MRM5m, (outs), (ins i64mem:$dst),
                  "shr{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (srl (loadi64 addr:$dst), CL), addr:$dst)]>;
def SHR64mi : RIi8<0xC1, MRM5m, (outs), (ins i64mem:$dst, i8imm:$src),
                  "shr{q}\t{$src, $dst|$dst, $src}",
                 [(store (srl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SHR64m1 : RI<0xD1, MRM5m, (outs), (ins i64mem:$dst),
                  "shr{q}\t$dst",
                 [(store (srl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
let Uses = [CL] in
def SAR64rCL : RI<0xD3, MRM7r, (outs GR64:$dst), (ins GR64:$src),
                 "sar{q}\t{%cl, $dst|$dst, %CL}",
                 [(set GR64:$dst, (sra GR64:$src, CL))]>;
def SAR64ri  : RIi8<0xC1, MRM7r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                   "sar{q}\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (sra GR64:$src1, (i8 imm:$src2)))]>;
def SAR64r1  : RI<0xD1, MRM7r, (outs GR64:$dst), (ins GR64:$src1),
                 "sar{q}\t$dst",
                 [(set GR64:$dst, (sra GR64:$src1, (i8 1)))]>;
} // isTwoAddress

let Uses = [CL] in
def SAR64mCL : RI<0xD3, MRM7m, (outs), (ins i64mem:$dst), 
                 "sar{q}\t{%cl, $dst|$dst, %CL}",
                 [(store (sra (loadi64 addr:$dst), CL), addr:$dst)]>;
def SAR64mi  : RIi8<0xC1, MRM7m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "sar{q}\t{$src, $dst|$dst, $src}",
                 [(store (sra (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def SAR64m1 : RI<0xD1, MRM7m, (outs), (ins i64mem:$dst),
                  "sar{q}\t$dst",
                 [(store (sra (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Rotate instructions
let isTwoAddress = 1 in {
let Uses = [CL] in
def ROL64rCL : RI<0xD3, MRM0r, (outs GR64:$dst), (ins GR64:$src),
                  "rol{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotl GR64:$src, CL))]>;
def ROL64ri  : RIi8<0xC1, MRM0r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                    "rol{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotl GR64:$src1, (i8 imm:$src2)))]>;
def ROL64r1  : RI<0xD1, MRM0r, (outs GR64:$dst), (ins GR64:$src1),
                  "rol{q}\t$dst",
                  [(set GR64:$dst, (rotl GR64:$src1, (i8 1)))]>;
} // isTwoAddress

let Uses = [CL] in
def ROL64mCL :  I<0xD3, MRM0m, (outs), (ins i64mem:$dst),
                  "rol{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (rotl (loadi64 addr:$dst), CL), addr:$dst)]>;
def ROL64mi  : RIi8<0xC1, MRM0m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "rol{q}\t{$src, $dst|$dst, $src}",
                [(store (rotl (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROL64m1  : RI<0xD1, MRM0m, (outs), (ins i64mem:$dst),
                 "rol{q}\t$dst",
               [(store (rotl (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

let isTwoAddress = 1 in {
let Uses = [CL] in
def ROR64rCL : RI<0xD3, MRM1r, (outs GR64:$dst), (ins GR64:$src),
                  "ror{q}\t{%cl, $dst|$dst, %CL}",
                  [(set GR64:$dst, (rotr GR64:$src, CL))]>;
def ROR64ri  : RIi8<0xC1, MRM1r, (outs GR64:$dst), (ins GR64:$src1, i8imm:$src2),
                    "ror{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (rotr GR64:$src1, (i8 imm:$src2)))]>;
def ROR64r1  : RI<0xD1, MRM1r, (outs GR64:$dst), (ins GR64:$src1),
                  "ror{q}\t$dst",
                  [(set GR64:$dst, (rotr GR64:$src1, (i8 1)))]>;
} // isTwoAddress

let Uses = [CL] in
def ROR64mCL : RI<0xD3, MRM1m, (outs), (ins i64mem:$dst), 
                  "ror{q}\t{%cl, $dst|$dst, %CL}",
                  [(store (rotr (loadi64 addr:$dst), CL), addr:$dst)]>;
def ROR64mi  : RIi8<0xC1, MRM1m, (outs), (ins i64mem:$dst, i8imm:$src),
                    "ror{q}\t{$src, $dst|$dst, $src}",
                [(store (rotr (loadi64 addr:$dst), (i8 imm:$src)), addr:$dst)]>;
def ROR64m1  : RI<0xD1, MRM1m, (outs), (ins i64mem:$dst),
                 "ror{q}\t$dst",
               [(store (rotr (loadi64 addr:$dst), (i8 1)), addr:$dst)]>;

// Double shift instructions (generalizations of rotate)
let isTwoAddress = 1 in {
let Uses = [CL] in {
def SHLD64rrCL : RI<0xA5, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(set GR64:$dst, (X86shld GR64:$src1, GR64:$src2, CL))]>, TB;
def SHRD64rrCL : RI<0xAD, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(set GR64:$dst, (X86shrd GR64:$src1, GR64:$src2, CL))]>, TB;
}

let isCommutable = 1 in {  // FIXME: Update X86InstrInfo::commuteInstruction
def SHLD64rri8 : RIi8<0xA4, MRMDestReg,
                      (outs GR64:$dst), (ins GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (X86shld GR64:$src1, GR64:$src2,
                                       (i8 imm:$src3)))]>,
                 TB;
def SHRD64rri8 : RIi8<0xAC, MRMDestReg,
                      (outs GR64:$dst), (ins GR64:$src1, GR64:$src2, i8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(set GR64:$dst, (X86shrd GR64:$src1, GR64:$src2,
                                       (i8 imm:$src3)))]>,
                 TB;
} // isCommutable
} // isTwoAddress

let Uses = [CL] in {
def SHLD64mrCL : RI<0xA5, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shld{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shld (loadi64 addr:$dst), GR64:$src2, CL),
                      addr:$dst)]>, TB;
def SHRD64mrCL : RI<0xAD, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                    "shrd{q}\t{%cl, $src2, $dst|$dst, $src2, %CL}",
                    [(store (X86shrd (loadi64 addr:$dst), GR64:$src2, CL),
                      addr:$dst)]>, TB;
}
def SHLD64mri8 : RIi8<0xA4, MRMDestMem,
                      (outs), (ins i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shld{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shld (loadi64 addr:$dst), GR64:$src2,
                                       (i8 imm:$src3)), addr:$dst)]>,
                 TB;
def SHRD64mri8 : RIi8<0xAC, MRMDestMem, 
                      (outs), (ins i64mem:$dst, GR64:$src2, i8imm:$src3),
                      "shrd{q}\t{$src3, $src2, $dst|$dst, $src2, $src3}",
                      [(store (X86shrd (loadi64 addr:$dst), GR64:$src2,
                                       (i8 imm:$src3)), addr:$dst)]>,
                 TB;
} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
//  Logical Instructions...
//

let isTwoAddress = 1 , AddedComplexity = 15 in
def NOT64r : RI<0xF7, MRM2r, (outs GR64:$dst), (ins GR64:$src), "not{q}\t$dst",
                [(set GR64:$dst, (not GR64:$src))]>;
def NOT64m : RI<0xF7, MRM2m, (outs), (ins i64mem:$dst), "not{q}\t$dst",
                [(store (not (loadi64 addr:$dst)), addr:$dst)]>;

let Defs = [EFLAGS] in {
let isTwoAddress = 1 in {
let isCommutable = 1 in
def AND64rr  : RI<0x21, MRMDestReg, 
                  (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "and{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (and GR64:$src1, GR64:$src2)),
                   (implicit EFLAGS)]>;
def AND64rm  : RI<0x23, MRMSrcMem,
                  (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "and{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (and GR64:$src1, (load addr:$src2))),
                   (implicit EFLAGS)]>;
def AND64ri8 : RIi8<0x83, MRM4r, 
                    (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "and{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (and GR64:$src1, i64immSExt8:$src2)),
                     (implicit EFLAGS)]>;
def AND64ri32  : RIi32<0x81, MRM4r, 
                       (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                       "and{q}\t{$src2, $dst|$dst, $src2}",
                       [(set GR64:$dst, (and GR64:$src1, i64immSExt32:$src2)),
                        (implicit EFLAGS)]>;
} // isTwoAddress

def AND64mr  : RI<0x21, MRMDestMem,
                  (outs), (ins i64mem:$dst, GR64:$src),
                  "and{q}\t{$src, $dst|$dst, $src}",
                  [(store (and (load addr:$dst), GR64:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def AND64mi8 : RIi8<0x83, MRM4m,
                    (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "and{q}\t{$src, $dst|$dst, $src}",
                 [(store (and (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
def AND64mi32  : RIi32<0x81, MRM4m,
                       (outs), (ins i64mem:$dst, i64i32imm:$src),
                       "and{q}\t{$src, $dst|$dst, $src}",
             [(store (and (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
              (implicit EFLAGS)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def OR64rr   : RI<0x09, MRMDestReg, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (or GR64:$src1, GR64:$src2)),
                   (implicit EFLAGS)]>;
def OR64rm   : RI<0x0B, MRMSrcMem , (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                  "or{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (or GR64:$src1, (load addr:$src2))),
                   (implicit EFLAGS)]>;
def OR64ri8  : RIi8<0x83, MRM1r, (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "or{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (or GR64:$src1, i64immSExt8:$src2)),
                     (implicit EFLAGS)]>;
def OR64ri32 : RIi32<0x81, MRM1r, (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2),
                     "or{q}\t{$src2, $dst|$dst, $src2}",
                     [(set GR64:$dst, (or GR64:$src1, i64immSExt32:$src2)),
                      (implicit EFLAGS)]>;
} // isTwoAddress

def OR64mr : RI<0x09, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                "or{q}\t{$src, $dst|$dst, $src}",
                [(store (or (load addr:$dst), GR64:$src), addr:$dst),
                 (implicit EFLAGS)]>;
def OR64mi8  : RIi8<0x83, MRM1m, (outs), (ins i64mem:$dst, i64i8imm:$src),
                    "or{q}\t{$src, $dst|$dst, $src}",
                  [(store (or (load addr:$dst), i64immSExt8:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def OR64mi32 : RIi32<0x81, MRM1m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                     "or{q}\t{$src, $dst|$dst, $src}",
              [(store (or (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
               (implicit EFLAGS)]>;

let isTwoAddress = 1 in {
let isCommutable = 1 in
def XOR64rr  : RI<0x31, MRMDestReg,  (outs GR64:$dst), (ins GR64:$src1, GR64:$src2), 
                  "xor{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (xor GR64:$src1, GR64:$src2)),
                   (implicit EFLAGS)]>;
def XOR64rm  : RI<0x33, MRMSrcMem, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2), 
                  "xor{q}\t{$src2, $dst|$dst, $src2}",
                  [(set GR64:$dst, (xor GR64:$src1, (load addr:$src2))),
                   (implicit EFLAGS)]>;
def XOR64ri8 : RIi8<0x83, MRM6r,  (outs GR64:$dst), (ins GR64:$src1, i64i8imm:$src2),
                    "xor{q}\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (xor GR64:$src1, i64immSExt8:$src2)),
                     (implicit EFLAGS)]>;
def XOR64ri32 : RIi32<0x81, MRM6r, 
                      (outs GR64:$dst), (ins GR64:$src1, i64i32imm:$src2), 
                      "xor{q}\t{$src2, $dst|$dst, $src2}",
                      [(set GR64:$dst, (xor GR64:$src1, i64immSExt32:$src2)),
                       (implicit EFLAGS)]>;
} // isTwoAddress

def XOR64mr  : RI<0x31, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                  "xor{q}\t{$src, $dst|$dst, $src}",
                  [(store (xor (load addr:$dst), GR64:$src), addr:$dst),
                   (implicit EFLAGS)]>;
def XOR64mi8 : RIi8<0x83, MRM6m, (outs), (ins i64mem:$dst, i64i8imm :$src),
                    "xor{q}\t{$src, $dst|$dst, $src}",
                 [(store (xor (load addr:$dst), i64immSExt8:$src), addr:$dst),
                  (implicit EFLAGS)]>;
def XOR64mi32 : RIi32<0x81, MRM6m, (outs), (ins i64mem:$dst, i64i32imm:$src),
                      "xor{q}\t{$src, $dst|$dst, $src}",
             [(store (xor (loadi64 addr:$dst), i64immSExt32:$src), addr:$dst),
              (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

//===----------------------------------------------------------------------===//
//  Comparison Instructions...
//

// Integer comparison
let Defs = [EFLAGS] in {
let isCommutable = 1 in
def TEST64rr : RI<0x85, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(X86cmp (and GR64:$src1, GR64:$src2), 0),
                   (implicit EFLAGS)]>;
def TEST64rm : RI<0x85, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                  "test{q}\t{$src2, $src1|$src1, $src2}",
                  [(X86cmp (and GR64:$src1, (loadi64 addr:$src2)), 0),
                   (implicit EFLAGS)]>;
def TEST64ri32 : RIi32<0xF7, MRM0r, (outs),
                                        (ins GR64:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                     [(X86cmp (and GR64:$src1, i64immSExt32:$src2), 0),
                      (implicit EFLAGS)]>;
def TEST64mi32 : RIi32<0xF7, MRM0m, (outs),
                                        (ins i64mem:$src1, i64i32imm:$src2),
                       "test{q}\t{$src2, $src1|$src1, $src2}",
                [(X86cmp (and (loadi64 addr:$src1), i64immSExt32:$src2), 0),
                 (implicit EFLAGS)]>;

def CMP64rr : RI<0x39, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(X86cmp GR64:$src1, GR64:$src2),
                  (implicit EFLAGS)]>;
def CMP64mr : RI<0x39, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(X86cmp (loadi64 addr:$src1), GR64:$src2),
                   (implicit EFLAGS)]>;
def CMP64rm : RI<0x3B, MRMSrcMem, (outs), (ins GR64:$src1, i64mem:$src2),
                 "cmp{q}\t{$src2, $src1|$src1, $src2}",
                 [(X86cmp GR64:$src1, (loadi64 addr:$src2)),
                  (implicit EFLAGS)]>;
def CMP64ri8 : RIi8<0x83, MRM7r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp GR64:$src1, i64immSExt8:$src2),
                     (implicit EFLAGS)]>;
def CMP64ri32 : RIi32<0x81, MRM7r, (outs), (ins GR64:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(X86cmp GR64:$src1, i64immSExt32:$src2),
                       (implicit EFLAGS)]>;
def CMP64mi8 : RIi8<0x83, MRM7m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                    "cmp{q}\t{$src2, $src1|$src1, $src2}",
                    [(X86cmp (loadi64 addr:$src1), i64immSExt8:$src2),
                     (implicit EFLAGS)]>;
def CMP64mi32 : RIi32<0x81, MRM7m, (outs),
                                       (ins i64mem:$src1, i64i32imm:$src2),
                      "cmp{q}\t{$src2, $src1|$src1, $src2}",
                      [(X86cmp (loadi64 addr:$src1), i64immSExt32:$src2),
                       (implicit EFLAGS)]>;
} // Defs = [EFLAGS]

// Bit tests.
// TODO: BTC, BTR, and BTS
let Defs = [EFLAGS] in {
def BT64rr : RI<0xA3, MRMDestReg, (outs), (ins GR64:$src1, GR64:$src2),
               "bt{q}\t{$src2, $src1|$src1, $src2}",
               [(X86bt GR64:$src1, GR64:$src2),
                (implicit EFLAGS)]>, TB;

// Unlike with the register+register form, the memory+register form of the
// bt instruction does not ignore the high bits of the index. From ISel's
// perspective, this is pretty bizarre. Disable these instructions for now.
//def BT64mr : RI<0xA3, MRMDestMem, (outs), (ins i64mem:$src1, GR64:$src2),
//               "bt{q}\t{$src2, $src1|$src1, $src2}",
//               [(X86bt (loadi64 addr:$src1), GR64:$src2),
//                (implicit EFLAGS)]>, TB;

def BT64ri8 : Ii8<0xBA, MRM4r, (outs), (ins GR64:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(X86bt GR64:$src1, i64immSExt8:$src2),
                 (implicit EFLAGS)]>, TB;
// Note that these instructions don't need FastBTMem because that
// only applies when the other operand is in a register. When it's
// an immediate, bt is still fast.
def BT64mi8 : Ii8<0xBA, MRM4m, (outs), (ins i64mem:$src1, i64i8imm:$src2),
                "bt{q}\t{$src2, $src1|$src1, $src2}",
                [(X86bt (loadi64 addr:$src1), i64immSExt8:$src2),
                 (implicit EFLAGS)]>, TB;
} // Defs = [EFLAGS]

// Conditional moves
let Uses = [EFLAGS], isTwoAddress = 1 in {
let isCommutable = 1 in {
def CMOVB64rr : RI<0x42, MRMSrcReg,       // if <u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovb\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_B, EFLAGS))]>, TB;
def CMOVAE64rr: RI<0x43, MRMSrcReg,       // if >=u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovae\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_AE, EFLAGS))]>, TB;
def CMOVE64rr : RI<0x44, MRMSrcReg,       // if ==, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmove\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_E, EFLAGS))]>, TB;
def CMOVNE64rr: RI<0x45, MRMSrcReg,       // if !=, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovne\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NE, EFLAGS))]>, TB;
def CMOVBE64rr: RI<0x46, MRMSrcReg,       // if <=u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovbe\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_BE, EFLAGS))]>, TB;
def CMOVA64rr : RI<0x47, MRMSrcReg,       // if >u, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmova\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_A, EFLAGS))]>, TB;
def CMOVL64rr : RI<0x4C, MRMSrcReg,       // if <s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovl\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_L, EFLAGS))]>, TB;
def CMOVGE64rr: RI<0x4D, MRMSrcReg,       // if >=s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovge\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_GE, EFLAGS))]>, TB;
def CMOVLE64rr: RI<0x4E, MRMSrcReg,       // if <=s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovle\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_LE, EFLAGS))]>, TB;
def CMOVG64rr : RI<0x4F, MRMSrcReg,       // if >s, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovg\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_G, EFLAGS))]>, TB;
def CMOVS64rr : RI<0x48, MRMSrcReg,       // if signed, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovs\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_S, EFLAGS))]>, TB;
def CMOVNS64rr: RI<0x49, MRMSrcReg,       // if !signed, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovns\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_NS, EFLAGS))]>, TB;
def CMOVP64rr : RI<0x4A, MRMSrcReg,       // if parity, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovp\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_P, EFLAGS))]>, TB;
def CMOVNP64rr : RI<0x4B, MRMSrcReg,       // if !parity, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovnp\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_NP, EFLAGS))]>, TB;
def CMOVO64rr : RI<0x40, MRMSrcReg,       // if overflow, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovo\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                    X86_COND_O, EFLAGS))]>, TB;
def CMOVNO64rr : RI<0x41, MRMSrcReg,       // if !overflow, GR64 = GR64
                   (outs GR64:$dst), (ins GR64:$src1, GR64:$src2),
                   "cmovno\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, GR64:$src2,
                                     X86_COND_NO, EFLAGS))]>, TB;
} // isCommutable = 1

def CMOVB64rm : RI<0x42, MRMSrcMem,       // if <u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovb\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_B, EFLAGS))]>, TB;
def CMOVAE64rm: RI<0x43, MRMSrcMem,       // if >=u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovae\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_AE, EFLAGS))]>, TB;
def CMOVE64rm : RI<0x44, MRMSrcMem,       // if ==, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmove\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_E, EFLAGS))]>, TB;
def CMOVNE64rm: RI<0x45, MRMSrcMem,       // if !=, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovne\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NE, EFLAGS))]>, TB;
def CMOVBE64rm: RI<0x46, MRMSrcMem,       // if <=u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovbe\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_BE, EFLAGS))]>, TB;
def CMOVA64rm : RI<0x47, MRMSrcMem,       // if >u, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmova\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_A, EFLAGS))]>, TB;
def CMOVL64rm : RI<0x4C, MRMSrcMem,       // if <s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovl\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_L, EFLAGS))]>, TB;
def CMOVGE64rm: RI<0x4D, MRMSrcMem,       // if >=s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovge\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_GE, EFLAGS))]>, TB;
def CMOVLE64rm: RI<0x4E, MRMSrcMem,       // if <=s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovle\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_LE, EFLAGS))]>, TB;
def CMOVG64rm : RI<0x4F, MRMSrcMem,       // if >s, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovg\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_G, EFLAGS))]>, TB;
def CMOVS64rm : RI<0x48, MRMSrcMem,       // if signed, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovs\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_S, EFLAGS))]>, TB;
def CMOVNS64rm: RI<0x49, MRMSrcMem,       // if !signed, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovns\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_NS, EFLAGS))]>, TB;
def CMOVP64rm : RI<0x4A, MRMSrcMem,       // if parity, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovp\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_P, EFLAGS))]>, TB;
def CMOVNP64rm : RI<0x4B, MRMSrcMem,       // if !parity, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovnp\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_NP, EFLAGS))]>, TB;
def CMOVO64rm : RI<0x40, MRMSrcMem,       // if overflow, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovo\t{$src2, $dst|$dst, $src2}",
                   [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                    X86_COND_O, EFLAGS))]>, TB;
def CMOVNO64rm : RI<0x41, MRMSrcMem,       // if !overflow, GR64 = [mem64]
                   (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2),
                   "cmovno\t{$src2, $dst|$dst, $src2}",
                    [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                     X86_COND_NO, EFLAGS))]>, TB;
} // isTwoAddress

//===----------------------------------------------------------------------===//
//  Conversion Instructions...
//

// f64 -> signed i64
def Int_CVTSD2SI64rr: RSDI<0x2D, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
                           "cvtsd2si{q}\t{$src, $dst|$dst, $src}",
                           [(set GR64:$dst,
                             (int_x86_sse2_cvtsd2si64 VR128:$src))]>;
def Int_CVTSD2SI64rm: RSDI<0x2D, MRMSrcMem, (outs GR64:$dst), (ins f128mem:$src),
                           "cvtsd2si{q}\t{$src, $dst|$dst, $src}",
                           [(set GR64:$dst, (int_x86_sse2_cvtsd2si64
                                             (load addr:$src)))]>;
def CVTTSD2SI64rr: RSDI<0x2C, MRMSrcReg, (outs GR64:$dst), (ins FR64:$src),
                        "cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint FR64:$src))]>;
def CVTTSD2SI64rm: RSDI<0x2C, MRMSrcMem, (outs GR64:$dst), (ins f64mem:$src),
                        "cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint (loadf64 addr:$src)))]>;
def Int_CVTTSD2SI64rr: RSDI<0x2C, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
                            "cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                            [(set GR64:$dst,
                              (int_x86_sse2_cvttsd2si64 VR128:$src))]>;
def Int_CVTTSD2SI64rm: RSDI<0x2C, MRMSrcMem, (outs GR64:$dst), (ins f128mem:$src),
                            "cvttsd2si{q}\t{$src, $dst|$dst, $src}",
                            [(set GR64:$dst,
                              (int_x86_sse2_cvttsd2si64
                               (load addr:$src)))]>;

// Signed i64 -> f64
def CVTSI2SD64rr: RSDI<0x2A, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "cvtsi2sd{q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (sint_to_fp GR64:$src))]>;
def CVTSI2SD64rm: RSDI<0x2A, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                       "cvtsi2sd{q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (sint_to_fp (loadi64 addr:$src)))]>;

let isTwoAddress = 1 in {
def Int_CVTSI2SD64rr: RSDI<0x2A, MRMSrcReg,
                           (outs VR128:$dst), (ins VR128:$src1, GR64:$src2),
                           "cvtsi2sd{q}\t{$src2, $dst|$dst, $src2}",
                           [(set VR128:$dst,
                             (int_x86_sse2_cvtsi642sd VR128:$src1,
                              GR64:$src2))]>;
def Int_CVTSI2SD64rm: RSDI<0x2A, MRMSrcMem,
                           (outs VR128:$dst), (ins VR128:$src1, i64mem:$src2),
                           "cvtsi2sd{q}\t{$src2, $dst|$dst, $src2}",
                           [(set VR128:$dst,
                             (int_x86_sse2_cvtsi642sd VR128:$src1,
                              (loadi64 addr:$src2)))]>;
} // isTwoAddress

// Signed i64 -> f32
def CVTSI2SS64rr: RSSI<0x2A, MRMSrcReg, (outs FR32:$dst), (ins GR64:$src),
                       "cvtsi2ss{q}\t{$src, $dst|$dst, $src}",
                       [(set FR32:$dst, (sint_to_fp GR64:$src))]>;
def CVTSI2SS64rm: RSSI<0x2A, MRMSrcMem, (outs FR32:$dst), (ins i64mem:$src),
                       "cvtsi2ss{q}\t{$src, $dst|$dst, $src}",
                       [(set FR32:$dst, (sint_to_fp (loadi64 addr:$src)))]>;

let isTwoAddress = 1 in {
  def Int_CVTSI2SS64rr : RSSI<0x2A, MRMSrcReg,
                              (outs VR128:$dst), (ins VR128:$src1, GR64:$src2),
                              "cvtsi2ss{q}\t{$src2, $dst|$dst, $src2}",
                              [(set VR128:$dst,
                                (int_x86_sse_cvtsi642ss VR128:$src1,
                                 GR64:$src2))]>;
  def Int_CVTSI2SS64rm : RSSI<0x2A, MRMSrcMem,
                              (outs VR128:$dst), (ins VR128:$src1, i64mem:$src2),
                              "cvtsi2ss{q}\t{$src2, $dst|$dst, $src2}",
                              [(set VR128:$dst,
                                (int_x86_sse_cvtsi642ss VR128:$src1,
                                 (loadi64 addr:$src2)))]>;
}

// f32 -> signed i64
def Int_CVTSS2SI64rr: RSSI<0x2D, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
                           "cvtss2si{q}\t{$src, $dst|$dst, $src}",
                           [(set GR64:$dst,
                             (int_x86_sse_cvtss2si64 VR128:$src))]>;
def Int_CVTSS2SI64rm: RSSI<0x2D, MRMSrcMem, (outs GR64:$dst), (ins f32mem:$src),
                           "cvtss2si{q}\t{$src, $dst|$dst, $src}",
                           [(set GR64:$dst, (int_x86_sse_cvtss2si64
                                             (load addr:$src)))]>;
def CVTTSS2SI64rr: RSSI<0x2C, MRMSrcReg, (outs GR64:$dst), (ins FR32:$src),
                        "cvttss2si{q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint FR32:$src))]>;
def CVTTSS2SI64rm: RSSI<0x2C, MRMSrcMem, (outs GR64:$dst), (ins f32mem:$src),
                        "cvttss2si{q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (fp_to_sint (loadf32 addr:$src)))]>;
def Int_CVTTSS2SI64rr: RSSI<0x2C, MRMSrcReg, (outs GR64:$dst), (ins VR128:$src),
                            "cvttss2si{q}\t{$src, $dst|$dst, $src}",
                            [(set GR64:$dst,
                              (int_x86_sse_cvttss2si64 VR128:$src))]>;
def Int_CVTTSS2SI64rm: RSSI<0x2C, MRMSrcMem, (outs GR64:$dst), (ins f32mem:$src),
                            "cvttss2si{q}\t{$src, $dst|$dst, $src}",
                            [(set GR64:$dst,
                              (int_x86_sse_cvttss2si64 (load addr:$src)))]>;

//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map movr0 to xor. Use xorl instead of xorq; it's
// equivalent due to implicit zero-extending, and it sometimes has a smaller
// encoding.
// FIXME: AddedComplexity gives this a higher priority than MOV64ri32. Remove
// when we have a better way to specify isel priority.
let AddedComplexity = 1 in
def : Pat<(i64 0),
          (SUBREG_TO_REG (i64 0), (MOV32r0), x86_subreg_32bit)>;


// Materialize i64 constant where top 32-bits are zero.
let AddedComplexity = 1, isReMaterializable = 1, isAsCheapAsAMove = 1 in
def MOV64ri64i32 : Ii32<0xB8, AddRegFrm, (outs GR64:$dst), (ins i64i32imm:$src),
                        "mov{l}\t{$src, ${dst:subreg32}|${dst:subreg32}, $src}",
                        [(set GR64:$dst, i64immZExt32:$src)]>;

//===----------------------------------------------------------------------===//
// Thread Local Storage Instructions
//===----------------------------------------------------------------------===//

// All calls clobber the non-callee saved registers. RSP is marked as
// a use to prevent stack-pointer assignments that appear immediately
// before calls from potentially appearing dead.
let Defs = [RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11,
            FP0, FP1, FP2, FP3, FP4, FP5, FP6, ST0, ST1,
            MM0, MM1, MM2, MM3, MM4, MM5, MM6, MM7,
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15, EFLAGS],
    Uses = [RSP] in
def TLS_addr64 : I<0, Pseudo, (outs), (ins lea64mem:$sym),
                   ".byte\t0x66; "
                   "leaq\t$sym(%rip), %rdi; "
                   ".word\t0x6666; "
                   "rex64; "
                   "call\t__tls_get_addr@PLT",
                  [(X86tlsaddr tls64addr:$sym)]>,
                  Requires<[In64BitMode]>;

let AddedComplexity = 5, isCodeGenOnly = 1 in
def MOV64GSrm : RI<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "movq\t%gs:$src, $dst",
                 [(set GR64:$dst, (gsload addr:$src))]>, SegGS;

let AddedComplexity = 5, isCodeGenOnly = 1 in
def MOV64FSrm : RI<0x8B, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
                 "movq\t%fs:$src, $dst",
                 [(set GR64:$dst, (fsload addr:$src))]>, SegFS;

//===----------------------------------------------------------------------===//
// Atomic Instructions
//===----------------------------------------------------------------------===//

let Defs = [RAX, EFLAGS], Uses = [RAX] in {
def LCMPXCHG64 : RI<0xB1, MRMDestMem, (outs), (ins i64mem:$ptr, GR64:$swap),
               "lock\n\t"
               "cmpxchgq\t$swap,$ptr",
               [(X86cas addr:$ptr, GR64:$swap, 8)]>, TB, LOCK;
}

let Constraints = "$val = $dst" in {
let Defs = [EFLAGS] in
def LXADD64 : RI<0xC1, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$ptr,GR64:$val),
               "lock\n\t"
               "xadd\t$val, $ptr",
               [(set GR64:$dst, (atomic_load_add_64 addr:$ptr, GR64:$val))]>,
                TB, LOCK;

def XCHG64rm : RI<0x87, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$ptr,GR64:$val),
                  "xchg\t$val, $ptr", 
                  [(set GR64:$dst, (atomic_swap_64 addr:$ptr, GR64:$val))]>;
}

// Optimized codegen when the non-memory output is not used.
// FIXME: Use normal add / sub instructions and add lock prefix dynamically.
def LOCK_ADD64mr : RI<0x03, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2),
                      "lock\n\t"
                      "add{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD64mi8 : RIi8<0x83, MRM0m, (outs),
                                      (ins i64mem:$dst, i64i8imm :$src2),
                    "lock\n\t"
                    "add{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_ADD64mi32 : RIi32<0x81, MRM0m, (outs),
                                        (ins i64mem:$dst, i64i32imm :$src2),
                      "lock\n\t"
                      "add{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB64mr : RI<0x29, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src2), 
                      "lock\n\t"
                      "sub{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB64mi8 : RIi8<0x83, MRM5m, (outs),
                                      (ins i64mem:$dst, i64i8imm :$src2), 
                      "lock\n\t"
                      "sub{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_SUB64mi32 : RIi32<0x81, MRM5m, (outs),
                                        (ins i64mem:$dst, i64i32imm:$src2),
                      "lock\n\t"
                      "sub{q}\t{$src2, $dst|$dst, $src2}", []>, LOCK;
def LOCK_INC64m : RI<0xFF, MRM0m, (outs), (ins i64mem:$dst),
                     "lock\n\t"
                     "inc{q}\t$dst", []>, LOCK;
def LOCK_DEC64m : RI<0xFF, MRM1m, (outs), (ins i64mem:$dst),
                      "lock\n\t"
                      "dec{q}\t$dst", []>, LOCK;

// Atomic exchange, and, or, xor
let Constraints = "$val = $dst", Defs = [EFLAGS],
                  usesCustomDAGSchedInserter = 1 in {
def ATOMAND64 : I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMAND64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_and_64 addr:$ptr, GR64:$val))]>;
def ATOMOR64 : I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMOR64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_or_64 addr:$ptr, GR64:$val))]>;
def ATOMXOR64 : I<0, Pseudo,(outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMXOR64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_xor_64 addr:$ptr, GR64:$val))]>;
def ATOMNAND64 : I<0, Pseudo,(outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMNAND64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_nand_64 addr:$ptr, GR64:$val))]>;
def ATOMMIN64: I<0, Pseudo, (outs GR64:$dst), (ins i64mem:$ptr, GR64:$val),
               "#ATOMMIN64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_min_64 addr:$ptr, GR64:$val))]>;
def ATOMMAX64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMMAX64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_max_64 addr:$ptr, GR64:$val))]>;
def ATOMUMIN64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMUMIN64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_umin_64 addr:$ptr, GR64:$val))]>;
def ATOMUMAX64: I<0, Pseudo, (outs GR64:$dst),(ins i64mem:$ptr, GR64:$val),
               "#ATOMUMAX64 PSEUDO!", 
               [(set GR64:$dst, (atomic_load_umax_64 addr:$ptr, GR64:$val))]>;
}

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// ConstantPool GlobalAddress, ExternalSymbol, and JumpTable when not in small
// code model mode, should use 'movabs'.  FIXME: This is really a hack, the
//  'movabs' predicate should handle this sort of thing.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri tconstpool  :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri tjumptable  :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri tglobaladdr :$dst)>, Requires<[FarData]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri texternalsym:$dst)>, Requires<[FarData]>;

// In static codegen with small code model, we can get the address of a label
// into a register with 'movl'.  FIXME: This is a hack, the 'imm' predicate of
// the MOV64ri64i32 should accept these.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri64i32 tconstpool  :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri64i32 tjumptable  :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri64i32 tglobaladdr :$dst)>, Requires<[SmallCode]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri64i32 texternalsym:$dst)>, Requires<[SmallCode]>;

// In kernel code model, we can get the address of a label
// into a register with 'movq'.  FIXME: This is a hack, the 'imm' predicate of
// the MOV64ri32 should accept these.
def : Pat<(i64 (X86Wrapper tconstpool  :$dst)),
          (MOV64ri32 tconstpool  :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper tjumptable  :$dst)),
          (MOV64ri32 tjumptable  :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper tglobaladdr :$dst)),
          (MOV64ri32 tglobaladdr :$dst)>, Requires<[KernelCode]>;
def : Pat<(i64 (X86Wrapper texternalsym:$dst)),
          (MOV64ri32 texternalsym:$dst)>, Requires<[KernelCode]>;

// If we have small model and -static mode, it is safe to store global addresses
// directly as immediates.  FIXME: This is really a hack, the 'imm' predicate
// for MOV64mi32 should handle this sort of thing.
def : Pat<(store (i64 (X86Wrapper tconstpool:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tconstpool:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tjumptable:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tjumptable:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper tglobaladdr:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, tglobaladdr:$src)>,
          Requires<[NearData, IsStatic]>;
def : Pat<(store (i64 (X86Wrapper texternalsym:$src)), addr:$dst),
          (MOV64mi32 addr:$dst, texternalsym:$src)>,
          Requires<[NearData, IsStatic]>;

// Calls
// Direct PC relative function call for small code model. 32-bit displacement
// sign extended to 64-bit.
def : Pat<(X86call (i64 tglobaladdr:$dst)),
          (CALL64pcrel32 tglobaladdr:$dst)>, Requires<[NotWin64]>;
def : Pat<(X86call (i64 texternalsym:$dst)),
          (CALL64pcrel32 texternalsym:$dst)>, Requires<[NotWin64]>;

def : Pat<(X86call (i64 tglobaladdr:$dst)),
          (WINCALL64pcrel32 tglobaladdr:$dst)>, Requires<[IsWin64]>;
def : Pat<(X86call (i64 texternalsym:$dst)),
          (WINCALL64pcrel32 texternalsym:$dst)>, Requires<[IsWin64]>;

// tailcall stuff
def : Pat<(X86tcret GR64:$dst, imm:$off),
          (TCRETURNri64 GR64:$dst, imm:$off)>;

def : Pat<(X86tcret (i64 tglobaladdr:$dst), imm:$off),
          (TCRETURNdi64 texternalsym:$dst, imm:$off)>;

def : Pat<(X86tcret (i64 texternalsym:$dst), imm:$off),
          (TCRETURNdi64 texternalsym:$dst, imm:$off)>;

// Comparisons.

// TEST R,R is smaller than CMP R,0
def : Pat<(parallel (X86cmp GR64:$src1, 0), (implicit EFLAGS)),
          (TEST64rr GR64:$src1, GR64:$src1)>;

// Conditional moves with folded loads with operands swapped and conditions
// inverted.
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_B, EFLAGS),
          (CMOVAE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_AE, EFLAGS),
          (CMOVB64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_E, EFLAGS),
          (CMOVNE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_NE, EFLAGS),
          (CMOVE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_BE, EFLAGS),
          (CMOVA64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_A, EFLAGS),
          (CMOVBE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_L, EFLAGS),
          (CMOVGE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_GE, EFLAGS),
          (CMOVL64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_LE, EFLAGS),
          (CMOVG64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_G, EFLAGS),
          (CMOVLE64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_P, EFLAGS),
          (CMOVNP64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_NP, EFLAGS),
          (CMOVP64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_S, EFLAGS),
          (CMOVNS64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_NS, EFLAGS),
          (CMOVS64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_O, EFLAGS),
          (CMOVNO64rm GR64:$src2, addr:$src1)>;
def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, X86_COND_NO, EFLAGS),
          (CMOVO64rm GR64:$src2, addr:$src1)>;

// zextload bool -> zextload byte
def : Pat<(zextloadi64i1 addr:$src), (MOVZX64rm8 addr:$src)>;

// extload
// When extloading from 16-bit and smaller memory locations into 64-bit registers,
// use zero-extending loads so that the entire 64-bit register is defined, avoiding
// partial-register updates.
def : Pat<(extloadi64i1 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i8 addr:$src),  (MOVZX64rm8  addr:$src)>;
def : Pat<(extloadi64i16 addr:$src), (MOVZX64rm16 addr:$src)>;
// For other extloads, use subregs, since the high contents of the register are
// defined after an extload.
def : Pat<(extloadi64i32 addr:$src),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (MOV32rm addr:$src),
                         x86_subreg_32bit)>;
def : Pat<(extloadi16i1 addr:$src), 
          (INSERT_SUBREG (i16 (IMPLICIT_DEF)), (MOV8rm addr:$src), 
                         x86_subreg_8bit)>,
         Requires<[In64BitMode]>;
def : Pat<(extloadi16i8 addr:$src), 
          (INSERT_SUBREG (i16 (IMPLICIT_DEF)), (MOV8rm addr:$src), 
                         x86_subreg_8bit)>,
         Requires<[In64BitMode]>;

// anyext
def : Pat<(i64 (anyext GR8:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR8:$src, x86_subreg_8bit)>;
def : Pat<(i64 (anyext GR16:$src)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR16:$src, x86_subreg_16bit)>;
def : Pat<(i64 (anyext GR32:$src)), 
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), GR32:$src, x86_subreg_32bit)>;
def : Pat<(i16 (anyext GR8:$src)),
          (INSERT_SUBREG (i16 (IMPLICIT_DEF)), GR8:$src, x86_subreg_8bit)>,
         Requires<[In64BitMode]>;
def : Pat<(i32 (anyext GR8:$src)),
          (INSERT_SUBREG (i32 (IMPLICIT_DEF)), GR8:$src, x86_subreg_8bit)>,
         Requires<[In64BitMode]>;

//===----------------------------------------------------------------------===//
// Some peepholes
//===----------------------------------------------------------------------===//

// Odd encoding trick: -128 fits into an 8-bit immediate field while
// +128 doesn't, so in this special case use a sub instead of an add.
def : Pat<(add GR64:$src1, 128),
          (SUB64ri8 GR64:$src1, -128)>;
def : Pat<(store (add (loadi64 addr:$dst), 128), addr:$dst),
          (SUB64mi8 addr:$dst, -128)>;

// The same trick applies for 32-bit immediate fields in 64-bit
// instructions.
def : Pat<(add GR64:$src1, 0x0000000080000000),
          (SUB64ri32 GR64:$src1, 0xffffffff80000000)>;
def : Pat<(store (add (loadi64 addr:$dst), 0x00000000800000000), addr:$dst),
          (SUB64mi32 addr:$dst, 0xffffffff80000000)>;

// r & (2^32-1) ==> movz
def : Pat<(and GR64:$src, 0x00000000FFFFFFFF),
          (MOVZX64rr32 (EXTRACT_SUBREG GR64:$src, x86_subreg_32bit))>;
// r & (2^16-1) ==> movz
def : Pat<(and GR64:$src, 0xffff),
          (MOVZX64rr16 (i16 (EXTRACT_SUBREG GR64:$src, x86_subreg_16bit)))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR64:$src, 0xff),
          (MOVZX64rr8 (i8 (EXTRACT_SUBREG GR64:$src, x86_subreg_8bit)))>;
// r & (2^8-1) ==> movz
def : Pat<(and GR32:$src1, 0xff),
           (MOVZX32rr8 (EXTRACT_SUBREG GR32:$src1, x86_subreg_8bit))>,
      Requires<[In64BitMode]>;
// r & (2^8-1) ==> movz
def : Pat<(and GR16:$src1, 0xff),
           (MOVZX16rr8 (i8 (EXTRACT_SUBREG GR16:$src1, x86_subreg_8bit)))>,
      Requires<[In64BitMode]>;

// sext_inreg patterns
def : Pat<(sext_inreg GR64:$src, i32),
          (MOVSX64rr32 (EXTRACT_SUBREG GR64:$src, x86_subreg_32bit))>;
def : Pat<(sext_inreg GR64:$src, i16),
          (MOVSX64rr16 (EXTRACT_SUBREG GR64:$src, x86_subreg_16bit))>;
def : Pat<(sext_inreg GR64:$src, i8),
          (MOVSX64rr8 (EXTRACT_SUBREG GR64:$src, x86_subreg_8bit))>;
def : Pat<(sext_inreg GR32:$src, i8),
          (MOVSX32rr8 (EXTRACT_SUBREG GR32:$src, x86_subreg_8bit))>,
      Requires<[In64BitMode]>;
def : Pat<(sext_inreg GR16:$src, i8),
          (MOVSX16rr8 (i8 (EXTRACT_SUBREG GR16:$src, x86_subreg_8bit)))>,
      Requires<[In64BitMode]>;

// trunc patterns
def : Pat<(i32 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, x86_subreg_32bit)>;
def : Pat<(i16 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, x86_subreg_16bit)>;
def : Pat<(i8 (trunc GR64:$src)),
          (EXTRACT_SUBREG GR64:$src, x86_subreg_8bit)>;
def : Pat<(i8 (trunc GR32:$src)),
          (EXTRACT_SUBREG GR32:$src, x86_subreg_8bit)>,
      Requires<[In64BitMode]>;
def : Pat<(i8 (trunc GR16:$src)),
          (EXTRACT_SUBREG GR16:$src, x86_subreg_8bit)>,
      Requires<[In64BitMode]>;

// h-register tricks.
// For now, be conservative on x86-64 and use an h-register extract only if the
// value is immediately zero-extended or stored, which are somewhat common
// cases. This uses a bunch of code to prevent a register requiring a REX prefix
// from being allocated in the same instruction as the h register, as there's
// currently no way to describe this requirement to the register allocator.

// h-register extract and zero-extend.
def : Pat<(and (srl_su GR64:$src, (i8 8)), (i64 255)),
          (SUBREG_TO_REG
            (i64 0),
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (COPY_TO_REGCLASS GR64:$src, GR64_ABCD),
                              x86_subreg_8bit_hi)),
            x86_subreg_32bit)>;
def : Pat<(and (srl_su GR32:$src, (i8 8)), (i32 255)),
          (MOVZX32_NOREXrr8
            (EXTRACT_SUBREG (COPY_TO_REGCLASS GR32:$src, GR32_ABCD),
                            x86_subreg_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(srl_su GR16:$src, (i8 8)),
          (EXTRACT_SUBREG
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (COPY_TO_REGCLASS GR16:$src, GR16_ABCD),
                              x86_subreg_8bit_hi)),
            x86_subreg_16bit)>,
      Requires<[In64BitMode]>;
def : Pat<(i32 (zext (srl_su GR16:$src, (i8 8)))),
          (MOVZX32_NOREXrr8
            (EXTRACT_SUBREG (COPY_TO_REGCLASS GR16:$src, GR16_ABCD),
                            x86_subreg_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(i64 (zext (srl_su GR16:$src, (i8 8)))),
          (SUBREG_TO_REG
            (i64 0),
            (MOVZX32_NOREXrr8
              (EXTRACT_SUBREG (COPY_TO_REGCLASS GR16:$src, GR16_ABCD),
                              x86_subreg_8bit_hi)),
            x86_subreg_32bit)>;

// h-register extract and store.
def : Pat<(store (i8 (trunc_su (srl_su GR64:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (COPY_TO_REGCLASS GR64:$src, GR64_ABCD),
                            x86_subreg_8bit_hi))>;
def : Pat<(store (i8 (trunc_su (srl_su GR32:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (COPY_TO_REGCLASS GR32:$src, GR32_ABCD),
                            x86_subreg_8bit_hi))>,
      Requires<[In64BitMode]>;
def : Pat<(store (i8 (trunc_su (srl_su GR16:$src, (i8 8)))), addr:$dst),
          (MOV8mr_NOREX
            addr:$dst,
            (EXTRACT_SUBREG (COPY_TO_REGCLASS GR16:$src, GR16_ABCD),
                            x86_subreg_8bit_hi))>,
      Requires<[In64BitMode]>;

// (shl x, 1) ==> (add x, x)
def : Pat<(shl GR64:$src1, (i8 1)), (ADD64rr GR64:$src1, GR64:$src1)>;

// (shl x (and y, 63)) ==> (shl x, y)
def : Pat<(shl GR64:$src1, (and CL:$amt, 63)),
          (SHL64rCL GR64:$src1)>;
def : Pat<(store (shl (loadi64 addr:$dst), (and CL:$amt, 63)), addr:$dst),
          (SHL64mCL addr:$dst)>;

def : Pat<(srl GR64:$src1, (and CL:$amt, 63)),
          (SHR64rCL GR64:$src1)>;
def : Pat<(store (srl (loadi64 addr:$dst), (and CL:$amt, 63)), addr:$dst),
          (SHR64mCL addr:$dst)>;

def : Pat<(sra GR64:$src1, (and CL:$amt, 63)),
          (SAR64rCL GR64:$src1)>;
def : Pat<(store (sra (loadi64 addr:$dst), (and CL:$amt, 63)), addr:$dst),
          (SAR64mCL addr:$dst)>;

// (or (x >> c) | (y << (64 - c))) ==> (shrd64 x, y, c)
def : Pat<(or (srl GR64:$src1, CL:$amt),
              (shl GR64:$src2, (sub 64, CL:$amt))),
          (SHRD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (srl (loadi64 addr:$dst), CL:$amt),
                     (shl GR64:$src2, (sub 64, CL:$amt))), addr:$dst),
          (SHRD64mrCL addr:$dst, GR64:$src2)>;

def : Pat<(or (srl GR64:$src1, (i8 (trunc RCX:$amt))),
              (shl GR64:$src2, (i8 (trunc (sub 64, RCX:$amt))))),
          (SHRD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (srl (loadi64 addr:$dst), (i8 (trunc RCX:$amt))),
                     (shl GR64:$src2, (i8 (trunc (sub 64, RCX:$amt))))),
                 addr:$dst),
          (SHRD64mrCL addr:$dst, GR64:$src2)>;

def : Pat<(shrd GR64:$src1, (i8 imm:$amt1), GR64:$src2, (i8 imm:$amt2)),
          (SHRD64rri8 GR64:$src1, GR64:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shrd (loadi64 addr:$dst), (i8 imm:$amt1),
                       GR64:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHRD64mri8 addr:$dst, GR64:$src2, (i8 imm:$amt1))>;

// (or (x << c) | (y >> (64 - c))) ==> (shld64 x, y, c)
def : Pat<(or (shl GR64:$src1, CL:$amt),
              (srl GR64:$src2, (sub 64, CL:$amt))),
          (SHLD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (shl (loadi64 addr:$dst), CL:$amt),
                     (srl GR64:$src2, (sub 64, CL:$amt))), addr:$dst),
          (SHLD64mrCL addr:$dst, GR64:$src2)>;

def : Pat<(or (shl GR64:$src1, (i8 (trunc RCX:$amt))),
              (srl GR64:$src2, (i8 (trunc (sub 64, RCX:$amt))))),
          (SHLD64rrCL GR64:$src1, GR64:$src2)>;

def : Pat<(store (or (shl (loadi64 addr:$dst), (i8 (trunc RCX:$amt))),
                     (srl GR64:$src2, (i8 (trunc (sub 64, RCX:$amt))))),
                 addr:$dst),
          (SHLD64mrCL addr:$dst, GR64:$src2)>;

def : Pat<(shld GR64:$src1, (i8 imm:$amt1), GR64:$src2, (i8 imm:$amt2)),
          (SHLD64rri8 GR64:$src1, GR64:$src2, (i8 imm:$amt1))>;

def : Pat<(store (shld (loadi64 addr:$dst), (i8 imm:$amt1),
                       GR64:$src2, (i8 imm:$amt2)), addr:$dst),
          (SHLD64mri8 addr:$dst, GR64:$src2, (i8 imm:$amt1))>;

// X86 specific add which produces a flag.
def : Pat<(addc GR64:$src1, GR64:$src2),
          (ADD64rr GR64:$src1, GR64:$src2)>;
def : Pat<(addc GR64:$src1, (load addr:$src2)),
          (ADD64rm GR64:$src1, addr:$src2)>;
def : Pat<(addc GR64:$src1, i64immSExt8:$src2),
          (ADD64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(addc GR64:$src1, i64immSExt32:$src2),
          (ADD64ri32 GR64:$src1, imm:$src2)>;

def : Pat<(subc GR64:$src1, GR64:$src2),
          (SUB64rr GR64:$src1, GR64:$src2)>;
def : Pat<(subc GR64:$src1, (load addr:$src2)),
          (SUB64rm GR64:$src1, addr:$src2)>;
def : Pat<(subc GR64:$src1, i64immSExt8:$src2),
          (SUB64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(subc GR64:$src1, imm:$src2),
          (SUB64ri32 GR64:$src1, i64immSExt32:$src2)>;

//===----------------------------------------------------------------------===//
// EFLAGS-defining Patterns
//===----------------------------------------------------------------------===//

// Register-Register Addition with EFLAGS result
def : Pat<(parallel (X86add_flag GR64:$src1, GR64:$src2),
                    (implicit EFLAGS)),
          (ADD64rr GR64:$src1, GR64:$src2)>;

// Register-Integer Addition with EFLAGS result
def : Pat<(parallel (X86add_flag GR64:$src1, i64immSExt8:$src2),
                    (implicit EFLAGS)),
          (ADD64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(parallel (X86add_flag GR64:$src1, i64immSExt32:$src2),
                    (implicit EFLAGS)),
          (ADD64ri32 GR64:$src1, i64immSExt32:$src2)>;

// Register-Memory Addition with EFLAGS result
def : Pat<(parallel (X86add_flag GR64:$src1, (loadi64 addr:$src2)),
                    (implicit EFLAGS)),
          (ADD64rm GR64:$src1, addr:$src2)>;

// Memory-Register Addition with EFLAGS result
def : Pat<(parallel (store (X86add_flag (loadi64 addr:$dst), GR64:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD64mr addr:$dst, GR64:$src2)>;
def : Pat<(parallel (store (X86add_flag (loadi64 addr:$dst), i64immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD64mi8 addr:$dst, i64immSExt8:$src2)>;
def : Pat<(parallel (store (X86add_flag (loadi64 addr:$dst), i64immSExt32:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (ADD64mi32 addr:$dst, i64immSExt32:$src2)>;

// Register-Register Subtraction with EFLAGS result
def : Pat<(parallel (X86sub_flag GR64:$src1, GR64:$src2),
                    (implicit EFLAGS)),
          (SUB64rr GR64:$src1, GR64:$src2)>;

// Register-Memory Subtraction with EFLAGS result
def : Pat<(parallel (X86sub_flag GR64:$src1, (loadi64 addr:$src2)),
                    (implicit EFLAGS)),
          (SUB64rm GR64:$src1, addr:$src2)>;

// Register-Integer Subtraction with EFLAGS result
def : Pat<(parallel (X86sub_flag GR64:$src1, i64immSExt8:$src2),
                    (implicit EFLAGS)),
          (SUB64ri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(parallel (X86sub_flag GR64:$src1, i64immSExt32:$src2),
                    (implicit EFLAGS)),
          (SUB64ri32 GR64:$src1, i64immSExt32:$src2)>;

// Memory-Register Subtraction with EFLAGS result
def : Pat<(parallel (store (X86sub_flag (loadi64 addr:$dst), GR64:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB64mr addr:$dst, GR64:$src2)>;

// Memory-Integer Subtraction with EFLAGS result
def : Pat<(parallel (store (X86sub_flag (loadi64 addr:$dst), i64immSExt8:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB64mi8 addr:$dst, i64immSExt8:$src2)>;
def : Pat<(parallel (store (X86sub_flag (loadi64 addr:$dst), i64immSExt32:$src2),
                           addr:$dst),
                    (implicit EFLAGS)),
          (SUB64mi32 addr:$dst, i64immSExt32:$src2)>;

// Register-Register Signed Integer Multiplication with EFLAGS result
def : Pat<(parallel (X86smul_flag GR64:$src1, GR64:$src2),
                    (implicit EFLAGS)),
          (IMUL64rr GR64:$src1, GR64:$src2)>;

// Register-Memory Signed Integer Multiplication with EFLAGS result
def : Pat<(parallel (X86smul_flag GR64:$src1, (loadi64 addr:$src2)),
                    (implicit EFLAGS)),
          (IMUL64rm GR64:$src1, addr:$src2)>;

// Register-Integer Signed Integer Multiplication with EFLAGS result
def : Pat<(parallel (X86smul_flag GR64:$src1, i64immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL64rri8 GR64:$src1, i64immSExt8:$src2)>;
def : Pat<(parallel (X86smul_flag GR64:$src1, i64immSExt32:$src2),
                    (implicit EFLAGS)),
          (IMUL64rri32 GR64:$src1, i64immSExt32:$src2)>;

// Memory-Integer Signed Integer Multiplication with EFLAGS result
def : Pat<(parallel (X86smul_flag (loadi64 addr:$src1), i64immSExt8:$src2),
                    (implicit EFLAGS)),
          (IMUL64rmi8 addr:$src1, i64immSExt8:$src2)>;
def : Pat<(parallel (X86smul_flag (loadi64 addr:$src1), i64immSExt32:$src2),
                    (implicit EFLAGS)),
          (IMUL64rmi32 addr:$src1, i64immSExt32:$src2)>;

// INC and DEC with EFLAGS result. Note that these do not set CF.
def : Pat<(parallel (X86inc_flag GR16:$src), (implicit EFLAGS)),
          (INC64_16r GR16:$src)>, Requires<[In64BitMode]>;
def : Pat<(parallel (store (i16 (X86inc_flag (loadi16 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (INC64_16m addr:$dst)>, Requires<[In64BitMode]>;
def : Pat<(parallel (X86dec_flag GR16:$src), (implicit EFLAGS)),
          (DEC64_16r GR16:$src)>, Requires<[In64BitMode]>;
def : Pat<(parallel (store (i16 (X86dec_flag (loadi16 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (DEC64_16m addr:$dst)>, Requires<[In64BitMode]>;

def : Pat<(parallel (X86inc_flag GR32:$src), (implicit EFLAGS)),
          (INC64_32r GR32:$src)>, Requires<[In64BitMode]>;
def : Pat<(parallel (store (i32 (X86inc_flag (loadi32 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (INC64_32m addr:$dst)>, Requires<[In64BitMode]>;
def : Pat<(parallel (X86dec_flag GR32:$src), (implicit EFLAGS)),
          (DEC64_32r GR32:$src)>, Requires<[In64BitMode]>;
def : Pat<(parallel (store (i32 (X86dec_flag (loadi32 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (DEC64_32m addr:$dst)>, Requires<[In64BitMode]>;

def : Pat<(parallel (X86inc_flag GR64:$src), (implicit EFLAGS)),
          (INC64r GR64:$src)>;
def : Pat<(parallel (store (i64 (X86inc_flag (loadi64 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (INC64m addr:$dst)>;
def : Pat<(parallel (X86dec_flag GR64:$src), (implicit EFLAGS)),
          (DEC64r GR64:$src)>;
def : Pat<(parallel (store (i64 (X86dec_flag (loadi64 addr:$dst))), addr:$dst),
                    (implicit EFLAGS)),
          (DEC64m addr:$dst)>;

//===----------------------------------------------------------------------===//
// X86-64 SSE Instructions
//===----------------------------------------------------------------------===//

// Move instructions...

def MOV64toPQIrr : RPDI<0x6E, MRMSrcReg, (outs VR128:$dst), (ins GR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set VR128:$dst,
                          (v2i64 (scalar_to_vector GR64:$src)))]>;
def MOVPQIto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins VR128:$src),
                         "mov{d|q}\t{$src, $dst|$dst, $src}",
                         [(set GR64:$dst, (vector_extract (v2i64 VR128:$src),
                                           (iPTR 0)))]>;

def MOV64toSDrr : RPDI<0x6E, MRMSrcReg, (outs FR64:$dst), (ins GR64:$src),
                       "mov{d|q}\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert GR64:$src))]>;
def MOV64toSDrm : RPDI<0x6E, MRMSrcMem, (outs FR64:$dst), (ins i64mem:$src),
                       "movq\t{$src, $dst|$dst, $src}",
                       [(set FR64:$dst, (bitconvert (loadi64 addr:$src)))]>;

def MOVSDto64rr  : RPDI<0x7E, MRMDestReg, (outs GR64:$dst), (ins FR64:$src),
                        "mov{d|q}\t{$src, $dst|$dst, $src}",
                        [(set GR64:$dst, (bitconvert FR64:$src))]>;
def MOVSDto64mr  : RPDI<0x7E, MRMDestMem, (outs), (ins i64mem:$dst, FR64:$src),
                        "movq\t{$src, $dst|$dst, $src}",
                        [(store (i64 (bitconvert FR64:$src)), addr:$dst)]>;

//===----------------------------------------------------------------------===//
// X86-64 SSE4.1 Instructions
//===----------------------------------------------------------------------===//

/// SS41I_extract32 - SSE 4.1 extract 32 bits to int reg or memory destination
multiclass SS41I_extract64<bits<8> opc, string OpcodeStr> {
  def rr : SS4AIi8<opc, MRMDestReg, (outs GR64:$dst),
                 (ins VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(set GR64:$dst,
                  (extractelt (v2i64 VR128:$src1), imm:$src2))]>, OpSize, REX_W;
  def mr : SS4AIi8<opc, MRMDestMem, (outs),
                 (ins i64mem:$dst, VR128:$src1, i32i8imm:$src2),
                 !strconcat(OpcodeStr, 
                  "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
                 [(store (extractelt (v2i64 VR128:$src1), imm:$src2),
                          addr:$dst)]>, OpSize, REX_W;
}

defm PEXTRQ      : SS41I_extract64<0x16, "pextrq">;

let isTwoAddress = 1 in {
  multiclass SS41I_insert64<bits<8> opc, string OpcodeStr> {
    def rr : SS4AIi8<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, GR64:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr, 
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (v2i64 (insertelt VR128:$src1, GR64:$src2, imm:$src3)))]>,
                   OpSize, REX_W;
    def rm : SS4AIi8<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, i64mem:$src2, i32i8imm:$src3),
                   !strconcat(OpcodeStr,
                    "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, 
                     (v2i64 (insertelt VR128:$src1, (loadi64 addr:$src2),
                                       imm:$src3)))]>, OpSize, REX_W;
  }
}

defm PINSRQ      : SS41I_insert64<0x22, "pinsrq">;
