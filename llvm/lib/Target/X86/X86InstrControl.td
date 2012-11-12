//===-- X86InstrControl.td - Control Flow Instructions -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 jump, return, call, and related instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Control Flow Instructions.
//

// Return instructions.
//
// The X86retflag return instructions are variadic because we may add ST0 and
// ST1 arguments when returning values on the x87 stack.
let isTerminator = 1, isReturn = 1, isBarrier = 1,
    hasCtrlDep = 1, FPForm = SpecialFP in {
  def RET    : I   <0xC3, RawFrm, (outs), (ins variable_ops),
                    "ret",
                    [(X86retflag 0)], IIC_RET>;
  def RETW   : I   <0xC3, RawFrm, (outs), (ins),
                    "ret{w}",
                    [], IIC_RET>, OpSize;
  def RETI   : Ii16<0xC2, RawFrm, (outs), (ins i16imm:$amt, variable_ops),
                    "ret\t$amt",
                    [(X86retflag timm:$amt)], IIC_RET_IMM>;
  def RETIW  : Ii16<0xC2, RawFrm, (outs), (ins i16imm:$amt),
                    "ret{w}\t$amt",
                    [], IIC_RET_IMM>, OpSize;
  def LRETL  : I   <0xCB, RawFrm, (outs), (ins),
                    "{l}ret{l|f}", [], IIC_RET>;
  def LRETW  : I   <0xCB, RawFrm, (outs), (ins),
                    "{l}ret{w|f}", [], IIC_RET>, OpSize;
  def LRETQ  : RI  <0xCB, RawFrm, (outs), (ins),
                    "{l}ret{q|f}", [], IIC_RET>;
  def LRETI  : Ii16<0xCA, RawFrm, (outs), (ins i16imm:$amt),
                    "{l}ret{l|f}\t$amt", [], IIC_RET>;
  def LRETIW : Ii16<0xCA, RawFrm, (outs), (ins i16imm:$amt),
                    "{l}ret{w|f}\t$amt", [], IIC_RET>, OpSize;
}

// Unconditional branches.
let isBarrier = 1, isBranch = 1, isTerminator = 1 in {
  def JMP_4 : Ii32PCRel<0xE9, RawFrm, (outs), (ins brtarget:$dst),
                        "jmp\t$dst", [(br bb:$dst)], IIC_JMP_REL>;
  def JMP_1 : Ii8PCRel<0xEB, RawFrm, (outs), (ins brtarget8:$dst),
                       "jmp\t$dst", [], IIC_JMP_REL>;
  // FIXME : Intel syntax for JMP64pcrel32 such that it is not ambiguious
  // with JMP_1.
  def JMP64pcrel32 : I<0xE9, RawFrm, (outs), (ins brtarget:$dst),
                       "jmpq\t$dst", [], IIC_JMP_REL>;
}

// Conditional Branches.
let isBranch = 1, isTerminator = 1, Uses = [EFLAGS] in {
  multiclass ICBr<bits<8> opc1, bits<8> opc4, string asm, PatFrag Cond> {
    def _1 : Ii8PCRel <opc1, RawFrm, (outs), (ins brtarget8:$dst), asm, [],
                       IIC_Jcc>;
    def _4 : Ii32PCRel<opc4, RawFrm, (outs), (ins brtarget:$dst), asm,
                       [(X86brcond bb:$dst, Cond, EFLAGS)], IIC_Jcc>, TB;
  }
}

defm JO  : ICBr<0x70, 0x80, "jo\t$dst" , X86_COND_O>;
defm JNO : ICBr<0x71, 0x81, "jno\t$dst" , X86_COND_NO>;
defm JB  : ICBr<0x72, 0x82, "jb\t$dst" , X86_COND_B>;
defm JAE : ICBr<0x73, 0x83, "jae\t$dst", X86_COND_AE>;
defm JE  : ICBr<0x74, 0x84, "je\t$dst" , X86_COND_E>;
defm JNE : ICBr<0x75, 0x85, "jne\t$dst", X86_COND_NE>;
defm JBE : ICBr<0x76, 0x86, "jbe\t$dst", X86_COND_BE>;
defm JA  : ICBr<0x77, 0x87, "ja\t$dst" , X86_COND_A>;
defm JS  : ICBr<0x78, 0x88, "js\t$dst" , X86_COND_S>;
defm JNS : ICBr<0x79, 0x89, "jns\t$dst", X86_COND_NS>;
defm JP  : ICBr<0x7A, 0x8A, "jp\t$dst" , X86_COND_P>;
defm JNP : ICBr<0x7B, 0x8B, "jnp\t$dst", X86_COND_NP>;
defm JL  : ICBr<0x7C, 0x8C, "jl\t$dst" , X86_COND_L>;
defm JGE : ICBr<0x7D, 0x8D, "jge\t$dst", X86_COND_GE>;
defm JLE : ICBr<0x7E, 0x8E, "jle\t$dst", X86_COND_LE>;
defm JG  : ICBr<0x7F, 0x8F, "jg\t$dst" , X86_COND_G>;

// jcx/jecx/jrcx instructions.
let isBranch = 1, isTerminator = 1 in {
  // These are the 32-bit versions of this instruction for the asmparser.  In
  // 32-bit mode, the address size prefix is jcxz and the unprefixed version is
  // jecxz.
  let Uses = [CX] in
    def JCXZ : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                        "jcxz\t$dst", [], IIC_JCXZ>, AdSize, Requires<[In32BitMode]>;
  let Uses = [ECX] in
    def JECXZ_32 : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                           "jecxz\t$dst", [], IIC_JCXZ>, Requires<[In32BitMode]>;

  // J*CXZ instruction: 64-bit versions of this instruction for the asmparser.
  // In 64-bit mode, the address size prefix is jecxz and the unprefixed version
  // is jrcxz.
  let Uses = [ECX] in
    def JECXZ_64 : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                            "jecxz\t$dst", [], IIC_JCXZ>, AdSize, Requires<[In64BitMode]>;
  let Uses = [RCX] in
    def JRCXZ : Ii8PCRel<0xE3, RawFrm, (outs), (ins brtarget8:$dst),
                           "jrcxz\t$dst", [], IIC_JCXZ>, Requires<[In64BitMode]>;
}

// Indirect branches
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in {
  def JMP32r     : I<0xFF, MRM4r, (outs), (ins GR32:$dst), "jmp{l}\t{*}$dst",
                     [(brind GR32:$dst)], IIC_JMP_REG>, Requires<[In32BitMode]>;
  def JMP32m     : I<0xFF, MRM4m, (outs), (ins i32mem:$dst), "jmp{l}\t{*}$dst",
                     [(brind (loadi32 addr:$dst))], IIC_JMP_MEM>, Requires<[In32BitMode]>;

  def JMP64r     : I<0xFF, MRM4r, (outs), (ins GR64:$dst), "jmp{q}\t{*}$dst",
                     [(brind GR64:$dst)], IIC_JMP_REG>, Requires<[In64BitMode]>;
  def JMP64m     : I<0xFF, MRM4m, (outs), (ins i64mem:$dst), "jmp{q}\t{*}$dst",
                     [(brind (loadi64 addr:$dst))], IIC_JMP_MEM>, Requires<[In64BitMode]>;

  def FARJMP16i  : Iseg16<0xEA, RawFrmImm16, (outs),
                          (ins i16imm:$off, i16imm:$seg),
                          "ljmp{w}\t{$seg, $off|$off, $seg}", [], IIC_JMP_FAR_PTR>, OpSize;
  def FARJMP32i  : Iseg32<0xEA, RawFrmImm16, (outs),
                          (ins i32imm:$off, i16imm:$seg),
                          "ljmp{l}\t{$seg, $off|$off, $seg}", [], IIC_JMP_FAR_PTR>;
  def FARJMP64   : RI<0xFF, MRM5m, (outs), (ins opaque80mem:$dst),
                      "ljmp{q}\t{*}$dst", [], IIC_JMP_FAR_MEM>;

  def FARJMP16m  : I<0xFF, MRM5m, (outs), (ins opaque32mem:$dst),
                     "ljmp{w}\t{*}$dst", [], IIC_JMP_FAR_MEM>, OpSize;
  def FARJMP32m  : I<0xFF, MRM5m, (outs), (ins opaque48mem:$dst),
                     "ljmp{l}\t{*}$dst", [], IIC_JMP_FAR_MEM>;
}


// Loop instructions

def LOOP   : Ii8PCRel<0xE2, RawFrm, (outs), (ins brtarget8:$dst), "loop\t$dst", [], IIC_LOOP>;
def LOOPE  : Ii8PCRel<0xE1, RawFrm, (outs), (ins brtarget8:$dst), "loope\t$dst", [], IIC_LOOPE>;
def LOOPNE : Ii8PCRel<0xE0, RawFrm, (outs), (ins brtarget8:$dst), "loopne\t$dst", [], IIC_LOOPNE>;

//===----------------------------------------------------------------------===//
//  Call Instructions...
//
let isCall = 1 in
  // All calls clobber the non-callee saved registers. ESP is marked as
  // a use to prevent stack-pointer assignments that appear immediately
  // before calls from potentially appearing dead. Uses for argument
  // registers are added manually.
  let Uses = [ESP] in {
    def CALLpcrel32 : Ii32PCRel<0xE8, RawFrm,
                           (outs), (ins i32imm_pcrel:$dst),
                           "call{l}\t$dst", [], IIC_CALL_RI>, Requires<[In32BitMode]>;
    def CALL32r     : I<0xFF, MRM2r, (outs), (ins GR32:$dst),
                        "call{l}\t{*}$dst", [(X86call GR32:$dst)], IIC_CALL_RI>,
                         Requires<[In32BitMode]>;
    def CALL32m     : I<0xFF, MRM2m, (outs), (ins i32mem:$dst),
                        "call{l}\t{*}$dst", [(X86call (loadi32 addr:$dst))], IIC_CALL_MEM>,
                        Requires<[In32BitMode]>;

    def FARCALL16i  : Iseg16<0x9A, RawFrmImm16, (outs),
                             (ins i16imm:$off, i16imm:$seg),
                             "lcall{w}\t{$seg, $off|$off, $seg}", [],
                             IIC_CALL_FAR_PTR>, OpSize;
    def FARCALL32i  : Iseg32<0x9A, RawFrmImm16, (outs),
                             (ins i32imm:$off, i16imm:$seg),
                             "lcall{l}\t{$seg, $off|$off, $seg}", [],
                             IIC_CALL_FAR_PTR>;

    def FARCALL16m  : I<0xFF, MRM3m, (outs), (ins opaque32mem:$dst),
                        "lcall{w}\t{*}$dst", [], IIC_CALL_FAR_MEM>, OpSize;
    def FARCALL32m  : I<0xFF, MRM3m, (outs), (ins opaque48mem:$dst),
                        "lcall{l}\t{*}$dst", [], IIC_CALL_FAR_MEM>;

    // callw for 16 bit code for the assembler.
    let isAsmParserOnly = 1 in
      def CALLpcrel16 : Ii16PCRel<0xE8, RawFrm,
                       (outs), (ins i16imm_pcrel:$dst),
                       "callw\t$dst", []>, OpSize;
  }


// Tail call stuff.

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1,
    isCodeGenOnly = 1 in
  let Uses = [ESP] in {
  def TCRETURNdi : PseudoI<(outs),
                     (ins i32imm_pcrel:$dst, i32imm:$offset), []>;
  def TCRETURNri : PseudoI<(outs),
                     (ins ptr_rc_tailcall:$dst, i32imm:$offset), []>;
  let mayLoad = 1 in
  def TCRETURNmi : PseudoI<(outs),
                     (ins i32mem_TC:$dst, i32imm:$offset), []>;

  // FIXME: The should be pseudo instructions that are lowered when going to
  // mcinst.
  def TAILJMPd : Ii32PCRel<0xE9, RawFrm, (outs),
                           (ins i32imm_pcrel:$dst),
                           "jmp\t$dst  # TAILCALL",
                           [], IIC_JMP_REL>;
  def TAILJMPr : I<0xFF, MRM4r, (outs), (ins ptr_rc_tailcall:$dst),
                   "", [], IIC_JMP_REG>;  // FIXME: Remove encoding when JIT is dead.
  let mayLoad = 1 in
  def TAILJMPm : I<0xFF, MRM4m, (outs), (ins i32mem_TC:$dst),
                   "jmp{l}\t{*}$dst  # TAILCALL", [], IIC_JMP_MEM>;
}


//===----------------------------------------------------------------------===//
//  Call Instructions...
//

// RSP is marked as a use to prevent stack-pointer assignments that appear
// immediately before calls from potentially appearing dead. Uses for argument
// registers are added manually.
let isCall = 1, Uses = [RSP] in {
  // NOTE: this pattern doesn't match "X86call imm", because we do not know
  // that the offset between an arbitrary immediate and the call will fit in
  // the 32-bit pcrel field that we have.
  def CALL64pcrel32 : Ii32PCRel<0xE8, RawFrm,
                        (outs), (ins i64i32imm_pcrel:$dst),
                        "call{q}\t$dst", [], IIC_CALL_RI>,
                      Requires<[In64BitMode]>;
  def CALL64r       : I<0xFF, MRM2r, (outs), (ins GR64:$dst),
                        "call{q}\t{*}$dst", [(X86call GR64:$dst)],
                        IIC_CALL_RI>,
                      Requires<[In64BitMode]>;
  def CALL64m       : I<0xFF, MRM2m, (outs), (ins i64mem:$dst),
                        "call{q}\t{*}$dst", [(X86call (loadi64 addr:$dst))],
                        IIC_CALL_MEM>,
                      Requires<[In64BitMode]>;

  def FARCALL64   : RI<0xFF, MRM3m, (outs), (ins opaque80mem:$dst),
                       "lcall{q}\t{*}$dst", [], IIC_CALL_FAR_MEM>;
}

let isCall = 1, isCodeGenOnly = 1 in
  // __chkstk(MSVC):     clobber R10, R11 and EFLAGS.
  // ___chkstk(Mingw64): clobber R10, R11, RAX and EFLAGS, and update RSP.
  let Defs = [RAX, R10, R11, RSP, EFLAGS],
      Uses = [RSP] in {
    def W64ALLOCA : Ii32PCRel<0xE8, RawFrm,
                      (outs), (ins i64i32imm_pcrel:$dst),
                      "call{q}\t$dst", [], IIC_CALL_RI>,
                    Requires<[IsWin64]>;
  }

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1,
    isCodeGenOnly = 1 in
  let Uses = [RSP],
      usesCustomInserter = 1 in {
  def TCRETURNdi64 : PseudoI<(outs),
                      (ins i64i32imm_pcrel:$dst, i32imm:$offset),
                      []>;
  def TCRETURNri64 : PseudoI<(outs),
                      (ins ptr_rc_tailcall:$dst, i32imm:$offset), []>;
  let mayLoad = 1 in
  def TCRETURNmi64 : PseudoI<(outs),
                       (ins i64mem_TC:$dst, i32imm:$offset), []>;

  def TAILJMPd64 : Ii32PCRel<0xE9, RawFrm, (outs),
                                      (ins i64i32imm_pcrel:$dst),
                   "jmp\t$dst  # TAILCALL", [], IIC_JMP_REL>;
  def TAILJMPr64 : I<0xFF, MRM4r, (outs), (ins ptr_rc_tailcall:$dst),
                     "jmp{q}\t{*}$dst  # TAILCALL", [], IIC_JMP_MEM>;

  let mayLoad = 1 in
  def TAILJMPm64 : I<0xFF, MRM4m, (outs), (ins i64mem_TC:$dst),
                     "jmp{q}\t{*}$dst  # TAILCALL", [], IIC_JMP_MEM>;
}
