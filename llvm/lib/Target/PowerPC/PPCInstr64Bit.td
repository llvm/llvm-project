//===-- PPCInstr64Bit.td - The PowerPC 64-bit Support ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the PowerPC 64-bit instructions.  These patterns are used
// both when in ppc64 mode and when in "use 64-bit extensions in 32-bit" mode.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// 64-bit operands.
//
def s16imm64 : Operand<i64> {
  let PrintMethod = "printS16ImmOperand";
  let EncoderMethod = "getImm16Encoding";
  let ParserMatchClass = PPCS16ImmAsmOperand;
}
def u16imm64 : Operand<i64> {
  let PrintMethod = "printU16ImmOperand";
  let EncoderMethod = "getImm16Encoding";
  let ParserMatchClass = PPCU16ImmAsmOperand;
}
def s17imm64 : Operand<i64> {
  // This operand type is used for addis/lis to allow the assembler parser
  // to accept immediates in the range -65536..65535 for compatibility with
  // the GNU assembler.  The operand is treated as 16-bit otherwise.
  let PrintMethod = "printS16ImmOperand";
  let EncoderMethod = "getImm16Encoding";
  let ParserMatchClass = PPCS17ImmAsmOperand;
}
def tocentry : Operand<iPTR> {
  let MIOperandInfo = (ops i64imm:$imm);
}
def PPCTLSRegOperand : AsmOperandClass {
  let Name = "TLSReg"; let PredicateMethod = "isTLSReg";
  let RenderMethod = "addTLSRegOperands";
}
def tlsreg : Operand<i64> {
  let EncoderMethod = "getTLSRegEncoding";
  let ParserMatchClass = PPCTLSRegOperand;
}
def tlsgd : Operand<i64> {}
def tlscall : Operand<i64> {
  let PrintMethod = "printTLSCall";
  let MIOperandInfo = (ops calltarget:$func, tlsgd:$sym);
  let EncoderMethod = "getTLSCallEncoding";
}

//===----------------------------------------------------------------------===//
// 64-bit transformation functions.
//

def SHL64 : SDNodeXForm<imm, [{
  // Transformation function: 63 - imm
  return getI32Imm(63 - N->getZExtValue());
}]>;

def SRL64 : SDNodeXForm<imm, [{
  // Transformation function: 64 - imm
  return N->getZExtValue() ? getI32Imm(64 - N->getZExtValue()) : getI32Imm(0);
}]>;

def HI32_48 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getZExtValue() >> 32));
}]>;

def HI48_64 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return getI32Imm((unsigned short)(N->getZExtValue() >> 48));
}]>;


//===----------------------------------------------------------------------===//
// Calls.
//

let Interpretation64Bit = 1 in {
let isTerminator = 1, isBarrier = 1, PPC970_Unit = 7 in {
  let isBranch = 1, isIndirectBranch = 1, Uses = [CTR8] in {
    def BCTR8 : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", BrB, []>,
        Requires<[In64BitMode]>;

    let isCodeGenOnly = 1 in
    def BCCTR8 : XLForm_2_br<19, 528, 0, (outs), (ins pred:$cond),
                             "b${cond:cc}ctr${cond:pm} ${cond:reg}", BrB, []>,
        Requires<[In64BitMode]>;
  }
}

let Defs = [LR8] in
  def MovePCtoLR8 : Pseudo<(outs), (ins), "#MovePCtoLR8", []>,
                    PPC970_Unit_BRU;

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7 in {
  let Defs = [CTR8], Uses = [CTR8] in {
    def BDZ8  : BForm_1<16, 18, 0, 0, (outs), (ins condbrtarget:$dst),
                        "bdz $dst">;
    def BDNZ8 : BForm_1<16, 16, 0, 0, (outs), (ins condbrtarget:$dst),
                        "bdnz $dst">;
  }

  let isReturn = 1, Defs = [CTR8], Uses = [CTR8, LR8, RM] in {
    def BDZLR8  : XLForm_2_ext<19, 16, 18, 0, 0, (outs), (ins),
                              "bdzlr", BrB, []>;
    def BDNZLR8 : XLForm_2_ext<19, 16, 16, 0, 0, (outs), (ins),
                              "bdnzlr", BrB, []>;
  }
}



let isCall = 1, PPC970_Unit = 7, Defs = [LR8] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL8  : IForm<18, 0, 1, (outs), (ins calltarget:$func),
                     "bl $func", BrB, []>;  // See Pat patterns below.

    def BL8_TLS  : IForm<18, 0, 1, (outs), (ins tlscall:$func),
                         "bl $func", BrB, []>;

    def BLA8 : IForm<18, 1, 1, (outs), (ins abscalltarget:$func),
                     "bla $func", BrB, [(PPCcall (i64 imm:$func))]>;
  }
  let Uses = [RM], isCodeGenOnly = 1 in {
    def BL8_NOP  : IForm_and_DForm_4_zero<18, 0, 1, 24,
                             (outs), (ins calltarget:$func),
                             "bl $func\n\tnop", BrB, []>;

    def BL8_NOP_TLS : IForm_and_DForm_4_zero<18, 0, 1, 24,
                                  (outs), (ins tlscall:$func),
                                  "bl $func\n\tnop", BrB, []>;

    def BLA8_NOP : IForm_and_DForm_4_zero<18, 1, 1, 24,
                             (outs), (ins abscalltarget:$func),
                             "bla $func\n\tnop", BrB,
                             [(PPCcall_nop (i64 imm:$func))]>;
  }
  let Uses = [CTR8, RM] in {
    def BCTRL8 : XLForm_2_ext<19, 528, 20, 0, 1, (outs), (ins),
                              "bctrl", BrB, [(PPCbctrl)]>,
                 Requires<[In64BitMode]>;

    let isCodeGenOnly = 1 in
    def BCCTRL8 : XLForm_2_br<19, 528, 1, (outs), (ins pred:$cond),
                              "b${cond:cc}ctrl${cond:pm} ${cond:reg}", BrB, []>,
        Requires<[In64BitMode]>;
  }
}
} // Interpretation64Bit

// Calls
def : Pat<(PPCcall (i64 tglobaladdr:$dst)),
          (BL8 tglobaladdr:$dst)>;
def : Pat<(PPCcall_nop (i64 tglobaladdr:$dst)),
          (BL8_NOP tglobaladdr:$dst)>;

def : Pat<(PPCcall (i64 texternalsym:$dst)),
          (BL8 texternalsym:$dst)>;
def : Pat<(PPCcall_nop (i64 texternalsym:$dst)),
          (BL8_NOP texternalsym:$dst)>;

// Atomic operations
let usesCustomInserter = 1 in {
  let Defs = [CR0] in {
    def ATOMIC_LOAD_ADD_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_ADD_I64",
      [(set i64:$dst, (atomic_load_add_64 xoaddr:$ptr, i64:$incr))]>;
    def ATOMIC_LOAD_SUB_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_SUB_I64",
      [(set i64:$dst, (atomic_load_sub_64 xoaddr:$ptr, i64:$incr))]>;
    def ATOMIC_LOAD_OR_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_OR_I64",
      [(set i64:$dst, (atomic_load_or_64 xoaddr:$ptr, i64:$incr))]>;
    def ATOMIC_LOAD_XOR_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_XOR_I64",
      [(set i64:$dst, (atomic_load_xor_64 xoaddr:$ptr, i64:$incr))]>;
    def ATOMIC_LOAD_AND_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_AND_i64",
      [(set i64:$dst, (atomic_load_and_64 xoaddr:$ptr, i64:$incr))]>;
    def ATOMIC_LOAD_NAND_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$incr), "#ATOMIC_LOAD_NAND_I64",
      [(set i64:$dst, (atomic_load_nand_64 xoaddr:$ptr, i64:$incr))]>;

    def ATOMIC_CMP_SWAP_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$old, g8rc:$new), "#ATOMIC_CMP_SWAP_I64",
      [(set i64:$dst, (atomic_cmp_swap_64 xoaddr:$ptr, i64:$old, i64:$new))]>;

    def ATOMIC_SWAP_I64 : Pseudo<
      (outs g8rc:$dst), (ins memrr:$ptr, g8rc:$new), "#ATOMIC_SWAP_I64",
      [(set i64:$dst, (atomic_swap_64 xoaddr:$ptr, i64:$new))]>;
  }
}

// Instructions to support atomic operations
def LDARX : XForm_1<31,  84, (outs g8rc:$rD), (ins memrr:$ptr),
                   "ldarx $rD, $ptr", LdStLDARX,
                   [(set i64:$rD, (PPClarx xoaddr:$ptr))]>;

let Defs = [CR0] in
def STDCX : XForm_1<31, 214, (outs), (ins g8rc:$rS, memrr:$dst),
                   "stdcx. $rS, $dst", LdStSTDCX,
                   [(PPCstcx i64:$rS, xoaddr:$dst)]>,
                   isDOT;

let Interpretation64Bit = 1 in {
let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNdi8 :Pseudo< (outs),
                        (ins calltarget:$dst, i32imm:$offset),
                 "#TC_RETURNd8 $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNai8 :Pseudo<(outs), (ins abscalltarget:$func, i32imm:$offset),
                 "#TC_RETURNa8 $func $offset",
                 [(PPCtc_return (i64 imm:$func), imm:$offset)]>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNri8 : Pseudo<(outs), (ins CTRRC8:$dst, i32imm:$offset),
                 "#TC_RETURNr8 $dst $offset",
                 []>;

let isCodeGenOnly = 1 in {

let isTerminator = 1, isBarrier = 1, PPC970_Unit = 7, isBranch = 1,
    isIndirectBranch = 1, isCall = 1, isReturn = 1, Uses = [CTR8, RM] in
def TAILBCTR8 : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", BrB, []>,
    Requires<[In64BitMode]>;


let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILB8   : IForm<18, 0, 0, (outs), (ins calltarget:$dst),
                  "b $dst", BrB,
                  []>;


let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILBA8   : IForm<18, 0, 0, (outs), (ins abscalltarget:$dst),
                  "ba $dst", BrB,
                  []>;

}
} // Interpretation64Bit

def : Pat<(PPCtc_return (i64 tglobaladdr:$dst),  imm:$imm),
          (TCRETURNdi8 tglobaladdr:$dst, imm:$imm)>;

def : Pat<(PPCtc_return (i64 texternalsym:$dst), imm:$imm),
          (TCRETURNdi8 texternalsym:$dst, imm:$imm)>;

def : Pat<(PPCtc_return CTRRC8:$dst, imm:$imm),
          (TCRETURNri8 CTRRC8:$dst, imm:$imm)>;


// 64-bit CR instructions
let Interpretation64Bit = 1 in {
let neverHasSideEffects = 1 in {
def MTOCRF8: XFXForm_5a<31, 144, (outs crbitm:$FXM), (ins g8rc:$ST),
                        "mtocrf $FXM, $ST", BrMCRX>,
            PPC970_DGroup_First, PPC970_Unit_CRU;

def MTCRF8 : XFXForm_5<31, 144, (outs), (ins i32imm:$FXM, g8rc:$rS),
                      "mtcrf $FXM, $rS", BrMCRX>,
            PPC970_MicroCode, PPC970_Unit_CRU;

let hasExtraSrcRegAllocReq = 1 in // to enable post-ra anti-dep breaking.
def MFOCRF8: XFXForm_5a<31, 19, (outs g8rc:$rT), (ins crbitm:$FXM),
                        "mfocrf $rT, $FXM", SprMFCR>,
             PPC970_DGroup_First, PPC970_Unit_CRU;

def MFCR8 : XFXForm_3<31, 19, (outs g8rc:$rT), (ins),
                     "mfcr $rT", SprMFCR>,
                     PPC970_MicroCode, PPC970_Unit_CRU;
} // neverHasSideEffects = 1

let hasSideEffects = 1, isBarrier = 1, usesCustomInserter = 1 in {
  let Defs = [CTR8] in
  def EH_SjLj_SetJmp64  : Pseudo<(outs gprc:$dst), (ins memr:$buf),
                            "#EH_SJLJ_SETJMP64",
                            [(set i32:$dst, (PPCeh_sjlj_setjmp addr:$buf))]>,
                          Requires<[In64BitMode]>;
  let isTerminator = 1 in
  def EH_SjLj_LongJmp64 : Pseudo<(outs), (ins memr:$buf),
                            "#EH_SJLJ_LONGJMP64",
                            [(PPCeh_sjlj_longjmp addr:$buf)]>,
                          Requires<[In64BitMode]>;
}

//===----------------------------------------------------------------------===//
// 64-bit SPR manipulation instrs.

let Uses = [CTR8] in {
def MFCTR8 : XFXForm_1_ext<31, 339, 9, (outs g8rc:$rT), (ins),
                           "mfctr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Pattern = [(PPCmtctr i64:$rS)], Defs = [CTR8] in {
def MTCTR8 : XFXForm_7_ext<31, 467, 9, (outs), (ins g8rc:$rS),
                           "mtctr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let hasSideEffects = 1, isCodeGenOnly = 1, Defs = [CTR8] in {
let Pattern = [(int_ppc_mtctr i64:$rS)] in
def MTCTR8loop : XFXForm_7_ext<31, 467, 9, (outs), (ins g8rc:$rS),
                               "mtctr $rS", SprMTSPR>,
                 PPC970_DGroup_First, PPC970_Unit_FXU;
}

let isCodeGenOnly = 1, Pattern = [(set i64:$rT, readcyclecounter)] in
def MFTB8 : XFXForm_1_ext<31, 339, 268, (outs g8rc:$rT), (ins),
                          "mfspr $rT, 268", SprMFTB>,
            PPC970_DGroup_First, PPC970_Unit_FXU;
// Note that encoding mftb using mfspr is now the preferred form,
// and has been since at least ISA v2.03. The mftb instruction has
// now been phased out. Using mfspr, however, is known not to work on
// the POWER3.

let Defs = [X1], Uses = [X1] in
def DYNALLOC8 : Pseudo<(outs g8rc:$result), (ins g8rc:$negsize, memri:$fpsi),"#DYNALLOC8",
                       [(set i64:$result,
                             (PPCdynalloc i64:$negsize, iaddr:$fpsi))]>;

let Defs = [LR8] in {
def MTLR8  : XFXForm_7_ext<31, 467, 8, (outs), (ins g8rc:$rS),
                           "mtlr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Uses = [LR8] in {
def MFLR8  : XFXForm_1_ext<31, 339, 8, (outs g8rc:$rT), (ins),
                           "mflr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
} // Interpretation64Bit

//===----------------------------------------------------------------------===//
// Fixed point instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.
let Interpretation64Bit = 1 in {
let neverHasSideEffects = 1 in {

let isReMaterializable = 1, isAsCheapAsAMove = 1, isMoveImm = 1 in {
def LI8  : DForm_2_r0<14, (outs g8rc:$rD), (ins s16imm64:$imm),
                      "li $rD, $imm", IntSimple,
                      [(set i64:$rD, imm64SExt16:$imm)]>;
def LIS8 : DForm_2_r0<15, (outs g8rc:$rD), (ins s17imm64:$imm),
                      "lis $rD, $imm", IntSimple,
                      [(set i64:$rD, imm16ShiftedSExt:$imm)]>;
}

// Logical ops.
defm NAND8: XForm_6r<31, 476, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "nand", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (not (and i64:$rS, i64:$rB)))]>;
defm AND8 : XForm_6r<31,  28, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "and", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (and i64:$rS, i64:$rB))]>;
defm ANDC8: XForm_6r<31,  60, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "andc", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (and i64:$rS, (not i64:$rB)))]>;
defm OR8  : XForm_6r<31, 444, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "or", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (or i64:$rS, i64:$rB))]>;
defm NOR8 : XForm_6r<31, 124, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "nor", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (not (or i64:$rS, i64:$rB)))]>;
defm ORC8 : XForm_6r<31, 412, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "orc", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (or i64:$rS, (not i64:$rB)))]>;
defm EQV8 : XForm_6r<31, 284, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "eqv", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (not (xor i64:$rS, i64:$rB)))]>;
defm XOR8 : XForm_6r<31, 316, (outs g8rc:$rA), (ins g8rc:$rS, g8rc:$rB),
                     "xor", "$rA, $rS, $rB", IntSimple,
                     [(set i64:$rA, (xor i64:$rS, i64:$rB))]>;

// Logical ops with immediate.
let Defs = [CR0] in {
def ANDIo8  : DForm_4<28, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                      "andi. $dst, $src1, $src2", IntGeneral,
                      [(set i64:$dst, (and i64:$src1, immZExt16:$src2))]>,
                      isDOT;
def ANDISo8 : DForm_4<29, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                     "andis. $dst, $src1, $src2", IntGeneral,
                    [(set i64:$dst, (and i64:$src1, imm16ShiftedZExt:$src2))]>,
                     isDOT;
}
def ORI8    : DForm_4<24, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                      "ori $dst, $src1, $src2", IntSimple,
                      [(set i64:$dst, (or i64:$src1, immZExt16:$src2))]>;
def ORIS8   : DForm_4<25, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                      "oris $dst, $src1, $src2", IntSimple,
                    [(set i64:$dst, (or i64:$src1, imm16ShiftedZExt:$src2))]>;
def XORI8   : DForm_4<26, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                      "xori $dst, $src1, $src2", IntSimple,
                      [(set i64:$dst, (xor i64:$src1, immZExt16:$src2))]>;
def XORIS8  : DForm_4<27, (outs g8rc:$dst), (ins g8rc:$src1, u16imm:$src2),
                      "xoris $dst, $src1, $src2", IntSimple,
                   [(set i64:$dst, (xor i64:$src1, imm16ShiftedZExt:$src2))]>;

defm ADD8  : XOForm_1r<31, 266, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "add", "$rT, $rA, $rB", IntSimple,
                       [(set i64:$rT, (add i64:$rA, i64:$rB))]>;
// ADD8 has a special form: reg = ADD8(reg, sym@tls) for use by the
// initial-exec thread-local storage model.
def ADD8TLS  : XOForm_1<31, 266, 0, (outs g8rc:$rT), (ins g8rc:$rA, tlsreg:$rB),
                        "add $rT, $rA, $rB", IntSimple,
                        [(set i64:$rT, (add i64:$rA, tglobaltlsaddr:$rB))]>;
                     
defm ADDC8 : XOForm_1rc<31, 10, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                        "addc", "$rT, $rA, $rB", IntGeneral,
                        [(set i64:$rT, (addc i64:$rA, i64:$rB))]>,
                        PPC970_DGroup_Cracked;
let Defs = [CARRY] in
def ADDIC8 : DForm_2<12, (outs g8rc:$rD), (ins g8rc:$rA, s16imm64:$imm),
                     "addic $rD, $rA, $imm", IntGeneral,
                     [(set i64:$rD, (addc i64:$rA, imm64SExt16:$imm))]>;
def ADDI8  : DForm_2<14, (outs g8rc:$rD), (ins g8rc_nox0:$rA, s16imm64:$imm),
                     "addi $rD, $rA, $imm", IntSimple,
                     [(set i64:$rD, (add i64:$rA, imm64SExt16:$imm))]>;
def ADDIS8 : DForm_2<15, (outs g8rc:$rD), (ins g8rc_nox0:$rA, s17imm64:$imm),
                     "addis $rD, $rA, $imm", IntSimple,
                     [(set i64:$rD, (add i64:$rA, imm16ShiftedSExt:$imm))]>;

let Defs = [CARRY] in {
def SUBFIC8: DForm_2< 8, (outs g8rc:$rD), (ins g8rc:$rA, s16imm64:$imm),
                     "subfic $rD, $rA, $imm", IntGeneral,
                     [(set i64:$rD, (subc imm64SExt16:$imm, i64:$rA))]>;
defm SUBFC8 : XOForm_1r<31, 8, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                        "subfc", "$rT, $rA, $rB", IntGeneral,
                        [(set i64:$rT, (subc i64:$rB, i64:$rA))]>,
                        PPC970_DGroup_Cracked;
}
defm SUBF8 : XOForm_1r<31, 40, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "subf", "$rT, $rA, $rB", IntGeneral,
                       [(set i64:$rT, (sub i64:$rB, i64:$rA))]>;
defm NEG8    : XOForm_3r<31, 104, 0, (outs g8rc:$rT), (ins g8rc:$rA),
                        "neg", "$rT, $rA", IntSimple,
                        [(set i64:$rT, (ineg i64:$rA))]>;
let Uses = [CARRY] in {
defm ADDE8   : XOForm_1rc<31, 138, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                          "adde", "$rT, $rA, $rB", IntGeneral,
                          [(set i64:$rT, (adde i64:$rA, i64:$rB))]>;
defm ADDME8  : XOForm_3rc<31, 234, 0, (outs g8rc:$rT), (ins g8rc:$rA),
                          "addme", "$rT, $rA", IntGeneral,
                          [(set i64:$rT, (adde i64:$rA, -1))]>;
defm ADDZE8  : XOForm_3rc<31, 202, 0, (outs g8rc:$rT), (ins g8rc:$rA),
                          "addze", "$rT, $rA", IntGeneral,
                          [(set i64:$rT, (adde i64:$rA, 0))]>;
defm SUBFE8  : XOForm_1rc<31, 136, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                          "subfe", "$rT, $rA, $rB", IntGeneral,
                          [(set i64:$rT, (sube i64:$rB, i64:$rA))]>;
defm SUBFME8 : XOForm_3rc<31, 232, 0, (outs g8rc:$rT), (ins g8rc:$rA),
                          "subfme", "$rT, $rA", IntGeneral,
                          [(set i64:$rT, (sube -1, i64:$rA))]>;
defm SUBFZE8 : XOForm_3rc<31, 200, 0, (outs g8rc:$rT), (ins g8rc:$rA),
                          "subfze", "$rT, $rA", IntGeneral,
                          [(set i64:$rT, (sube 0, i64:$rA))]>;
}


defm MULHD : XOForm_1r<31, 73, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "mulhd", "$rT, $rA, $rB", IntMulHW,
                       [(set i64:$rT, (mulhs i64:$rA, i64:$rB))]>;
defm MULHDU : XOForm_1r<31, 9, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "mulhdu", "$rT, $rA, $rB", IntMulHWU,
                       [(set i64:$rT, (mulhu i64:$rA, i64:$rB))]>;
}
} // Interpretation64Bit

let isCompare = 1, neverHasSideEffects = 1 in {
  def CMPD   : XForm_16_ext<31, 0, (outs crrc:$crD), (ins g8rc:$rA, g8rc:$rB),
                            "cmpd $crD, $rA, $rB", IntCompare>, isPPC64;
  def CMPLD  : XForm_16_ext<31, 32, (outs crrc:$crD), (ins g8rc:$rA, g8rc:$rB),
                            "cmpld $crD, $rA, $rB", IntCompare>, isPPC64;
  def CMPDI  : DForm_5_ext<11, (outs crrc:$crD), (ins g8rc:$rA, s16imm:$imm),
                           "cmpdi $crD, $rA, $imm", IntCompare>, isPPC64;
  def CMPLDI : DForm_6_ext<10, (outs crrc:$dst), (ins g8rc:$src1, u16imm:$src2),
                           "cmpldi $dst, $src1, $src2", IntCompare>, isPPC64;
}

let neverHasSideEffects = 1 in {
defm SLD  : XForm_6r<31,  27, (outs g8rc:$rA), (ins g8rc:$rS, gprc:$rB),
                     "sld", "$rA, $rS, $rB", IntRotateD,
                     [(set i64:$rA, (PPCshl i64:$rS, i32:$rB))]>, isPPC64;
defm SRD  : XForm_6r<31, 539, (outs g8rc:$rA), (ins g8rc:$rS, gprc:$rB),
                     "srd", "$rA, $rS, $rB", IntRotateD,
                     [(set i64:$rA, (PPCsrl i64:$rS, i32:$rB))]>, isPPC64;
defm SRAD : XForm_6rc<31, 794, (outs g8rc:$rA), (ins g8rc:$rS, gprc:$rB),
                      "srad", "$rA, $rS, $rB", IntRotateD,
                      [(set i64:$rA, (PPCsra i64:$rS, i32:$rB))]>, isPPC64;

let Interpretation64Bit = 1 in { 
defm EXTSB8 : XForm_11r<31, 954, (outs g8rc:$rA), (ins g8rc:$rS),
                        "extsb", "$rA, $rS", IntSimple,
                        [(set i64:$rA, (sext_inreg i64:$rS, i8))]>;
defm EXTSH8 : XForm_11r<31, 922, (outs g8rc:$rA), (ins g8rc:$rS),
                        "extsh", "$rA, $rS", IntSimple,
                        [(set i64:$rA, (sext_inreg i64:$rS, i16))]>;
} // Interpretation64Bit

// For fast-isel:
let isCodeGenOnly = 1 in {
def EXTSB8_32_64 : XForm_11<31, 954, (outs g8rc:$rA), (ins gprc:$rS),
                           "extsb $rA, $rS", IntSimple, []>, isPPC64;
def EXTSH8_32_64 : XForm_11<31, 922, (outs g8rc:$rA), (ins gprc:$rS),
                           "extsh $rA, $rS", IntSimple, []>, isPPC64;
} // isCodeGenOnly for fast-isel

defm EXTSW  : XForm_11r<31, 986, (outs g8rc:$rA), (ins g8rc:$rS),
                        "extsw", "$rA, $rS", IntSimple,
                        [(set i64:$rA, (sext_inreg i64:$rS, i32))]>, isPPC64;
let Interpretation64Bit = 1 in
defm EXTSW_32_64 : XForm_11r<31, 986, (outs g8rc:$rA), (ins gprc:$rS),
                             "extsw", "$rA, $rS", IntSimple,
                             [(set i64:$rA, (sext i32:$rS))]>, isPPC64;

defm SRADI  : XSForm_1rc<31, 413, (outs g8rc:$rA), (ins g8rc:$rS, u6imm:$SH),
                         "sradi", "$rA, $rS, $SH", IntRotateDI,
                         [(set i64:$rA, (sra i64:$rS, (i32 imm:$SH)))]>, isPPC64;
defm CNTLZD : XForm_11r<31, 58, (outs g8rc:$rA), (ins g8rc:$rS),
                        "cntlzd", "$rA, $rS", IntGeneral,
                        [(set i64:$rA, (ctlz i64:$rS))]>;
def POPCNTD : XForm_11<31, 506, (outs g8rc:$rA), (ins g8rc:$rS),
                       "popcntd $rA, $rS", IntGeneral,
                       [(set i64:$rA, (ctpop i64:$rS))]>;

// popcntw also does a population count on the high 32 bits (storing the
// results in the high 32-bits of the output). We'll ignore that here (which is
// safe because we never separately use the high part of the 64-bit registers).
def POPCNTW : XForm_11<31, 378, (outs gprc:$rA), (ins gprc:$rS),
                       "popcntw $rA, $rS", IntGeneral,
                       [(set i32:$rA, (ctpop i32:$rS))]>;

defm DIVD  : XOForm_1r<31, 489, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "divd", "$rT, $rA, $rB", IntDivD,
                       [(set i64:$rT, (sdiv i64:$rA, i64:$rB))]>, isPPC64,
                       PPC970_DGroup_First, PPC970_DGroup_Cracked;
defm DIVDU : XOForm_1r<31, 457, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "divdu", "$rT, $rA, $rB", IntDivD,
                       [(set i64:$rT, (udiv i64:$rA, i64:$rB))]>, isPPC64,
                       PPC970_DGroup_First, PPC970_DGroup_Cracked;
defm MULLD : XOForm_1r<31, 233, 0, (outs g8rc:$rT), (ins g8rc:$rA, g8rc:$rB),
                       "mulld", "$rT, $rA, $rB", IntMulHD,
                       [(set i64:$rT, (mul i64:$rA, i64:$rB))]>, isPPC64;
def MULLI8 : DForm_2<7, (outs g8rc:$rD), (ins g8rc:$rA, s16imm64:$imm),
                       "mulli $rD, $rA, $imm", IntMulLI,
                       [(set i64:$rD, (mul i64:$rA, imm64SExt16:$imm))]>;
}

let neverHasSideEffects = 1 in {
let isCommutable = 1 in {
defm RLDIMI : MDForm_1r<30, 3, (outs g8rc:$rA),
                        (ins g8rc:$rSi, g8rc:$rS, u6imm:$SH, u6imm:$MBE),
                        "rldimi", "$rA, $rS, $SH, $MBE", IntRotateDI,
                        []>, isPPC64, RegConstraint<"$rSi = $rA">,
                        NoEncode<"$rSi">;
}

// Rotate instructions.
defm RLDCL  : MDSForm_1r<30, 8,
                        (outs g8rc:$rA), (ins g8rc:$rS, gprc:$rB, u6imm:$MBE),
                        "rldcl", "$rA, $rS, $rB, $MBE", IntRotateD,
                        []>, isPPC64;
defm RLDCR  : MDSForm_1r<30, 9,
                        (outs g8rc:$rA), (ins g8rc:$rS, gprc:$rB, u6imm:$MBE),
                        "rldcr", "$rA, $rS, $rB, $MBE", IntRotateD,
                        []>, isPPC64;
defm RLDICL : MDForm_1r<30, 0,
                        (outs g8rc:$rA), (ins g8rc:$rS, u6imm:$SH, u6imm:$MBE),
                        "rldicl", "$rA, $rS, $SH, $MBE", IntRotateDI,
                        []>, isPPC64;
// For fast-isel:
let isCodeGenOnly = 1 in
def RLDICL_32_64 : MDForm_1<30, 0,
                           (outs g8rc:$rA),
                           (ins gprc:$rS, u6imm:$SH, u6imm:$MBE),
                           "rldicl $rA, $rS, $SH, $MBE", IntRotateDI,
                           []>, isPPC64;
// End fast-isel.
defm RLDICR : MDForm_1r<30, 1,
                        (outs g8rc:$rA), (ins g8rc:$rS, u6imm:$SH, u6imm:$MBE),
                        "rldicr", "$rA, $rS, $SH, $MBE", IntRotateDI,
                        []>, isPPC64;
defm RLDIC  : MDForm_1r<30, 2,
                        (outs g8rc:$rA), (ins g8rc:$rS, u6imm:$SH, u6imm:$MBE),
                        "rldic", "$rA, $rS, $SH, $MBE", IntRotateDI,
                        []>, isPPC64;

let Interpretation64Bit = 1 in {
defm RLWINM8 : MForm_2r<21, (outs g8rc:$rA),
                        (ins g8rc:$rS, u5imm:$SH, u5imm:$MB, u5imm:$ME),
                        "rlwinm", "$rA, $rS, $SH, $MB, $ME", IntGeneral,
                        []>;

let isSelect = 1 in
def ISEL8   : AForm_4<31, 15,
                     (outs g8rc:$rT), (ins g8rc_nox0:$rA, g8rc:$rB, crbitrc:$cond),
                     "isel $rT, $rA, $rB, $cond", IntGeneral,
                     []>;
}  // Interpretation64Bit
}  // neverHasSideEffects = 1
}  // End FXU Operations.


//===----------------------------------------------------------------------===//
// Load/Store instructions.
//


// Sign extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
let Interpretation64Bit = 1 in
def LHA8: DForm_1<42, (outs g8rc:$rD), (ins memri:$src),
                  "lha $rD, $src", LdStLHA,
                  [(set i64:$rD, (sextloadi16 iaddr:$src))]>,
                  PPC970_DGroup_Cracked;
def LWA  : DSForm_1<58, 2, (outs g8rc:$rD), (ins memrix:$src),
                    "lwa $rD, $src", LdStLWA,
                    [(set i64:$rD,
                          (aligned4sextloadi32 ixaddr:$src))]>, isPPC64,
                    PPC970_DGroup_Cracked;
let Interpretation64Bit = 1 in
def LHAX8: XForm_1<31, 343, (outs g8rc:$rD), (ins memrr:$src),
                   "lhax $rD, $src", LdStLHA,
                   [(set i64:$rD, (sextloadi16 xaddr:$src))]>,
                   PPC970_DGroup_Cracked;
def LWAX : XForm_1<31, 341, (outs g8rc:$rD), (ins memrr:$src),
                   "lwax $rD, $src", LdStLHA,
                   [(set i64:$rD, (sextloadi32 xaddr:$src))]>, isPPC64,
                   PPC970_DGroup_Cracked;
// For fast-isel:
let isCodeGenOnly = 1, mayLoad = 1 in {
def LWA_32  : DSForm_1<58, 2, (outs gprc:$rD), (ins memrix:$src),
                      "lwa $rD, $src", LdStLWA, []>, isPPC64,
                      PPC970_DGroup_Cracked;
def LWAX_32 : XForm_1<31, 341, (outs gprc:$rD), (ins memrr:$src),
                     "lwax $rD, $src", LdStLHA, []>, isPPC64,
                     PPC970_DGroup_Cracked;
} // end fast-isel isCodeGenOnly

// Update forms.
let mayLoad = 1, neverHasSideEffects = 1 in {
let Interpretation64Bit = 1 in
def LHAU8 : DForm_1<43, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                    (ins memri:$addr),
                    "lhau $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
// NO LWAU!

let Interpretation64Bit = 1 in
def LHAUX8 : XForm_1<31, 375, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                    (ins memrr:$addr),
                    "lhaux $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                    NoEncode<"$ea_result">;
def LWAUX : XForm_1<31, 373, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                    (ins memrr:$addr),
                    "lwaux $rD, $addr", LdStLHAU,
                    []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                    NoEncode<"$ea_result">, isPPC64;
}
}

let Interpretation64Bit = 1 in {
// Zero extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LBZ8 : DForm_1<34, (outs g8rc:$rD), (ins memri:$src),
                  "lbz $rD, $src", LdStLoad,
                  [(set i64:$rD, (zextloadi8 iaddr:$src))]>;
def LHZ8 : DForm_1<40, (outs g8rc:$rD), (ins memri:$src),
                  "lhz $rD, $src", LdStLoad,
                  [(set i64:$rD, (zextloadi16 iaddr:$src))]>;
def LWZ8 : DForm_1<32, (outs g8rc:$rD), (ins memri:$src),
                  "lwz $rD, $src", LdStLoad,
                  [(set i64:$rD, (zextloadi32 iaddr:$src))]>, isPPC64;

def LBZX8 : XForm_1<31,  87, (outs g8rc:$rD), (ins memrr:$src),
                   "lbzx $rD, $src", LdStLoad,
                   [(set i64:$rD, (zextloadi8 xaddr:$src))]>;
def LHZX8 : XForm_1<31, 279, (outs g8rc:$rD), (ins memrr:$src),
                   "lhzx $rD, $src", LdStLoad,
                   [(set i64:$rD, (zextloadi16 xaddr:$src))]>;
def LWZX8 : XForm_1<31,  23, (outs g8rc:$rD), (ins memrr:$src),
                   "lwzx $rD, $src", LdStLoad,
                   [(set i64:$rD, (zextloadi32 xaddr:$src))]>;
                   
                   
// Update forms.
let mayLoad = 1, neverHasSideEffects = 1 in {
def LBZU8 : DForm_1<35, (outs g8rc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lbzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LHZU8 : DForm_1<41, (outs g8rc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lhzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LWZU8 : DForm_1<33, (outs g8rc:$rD, ptr_rc_nor0:$ea_result), (ins memri:$addr),
                    "lwzu $rD, $addr", LdStLoadUpd,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;

def LBZUX8 : XForm_1<31, 119, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lbzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;
def LHZUX8 : XForm_1<31, 311, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lhzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;
def LWZUX8 : XForm_1<31, 55, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "lwzux $rD, $addr", LdStLoadUpd,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">;
}
}
} // Interpretation64Bit


// Full 8-byte loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LD   : DSForm_1<58, 0, (outs g8rc:$rD), (ins memrix:$src),
                    "ld $rD, $src", LdStLD,
                    [(set i64:$rD, (aligned4load ixaddr:$src))]>, isPPC64;
// The following three definitions are selected for small code model only.
// Otherwise, we need to create two instructions to form a 32-bit offset,
// so we have a custom matcher for TOC_ENTRY in PPCDAGToDAGIsel::Select().
def LDtoc: Pseudo<(outs g8rc:$rD), (ins tocentry:$disp, g8rc:$reg),
                  "#LDtoc",
                  [(set i64:$rD,
                     (PPCtoc_entry tglobaladdr:$disp, i64:$reg))]>, isPPC64;
def LDtocJTI: Pseudo<(outs g8rc:$rD), (ins tocentry:$disp, g8rc:$reg),
                  "#LDtocJTI",
                  [(set i64:$rD,
                     (PPCtoc_entry tjumptable:$disp, i64:$reg))]>, isPPC64;
def LDtocCPT: Pseudo<(outs g8rc:$rD), (ins tocentry:$disp, g8rc:$reg),
                  "#LDtocCPT",
                  [(set i64:$rD,
                     (PPCtoc_entry tconstpool:$disp, i64:$reg))]>, isPPC64;

let hasSideEffects = 1, isCodeGenOnly = 1 in {
let RST = 2, DS = 2 in
def LDinto_toc: DSForm_1a<58, 0, (outs), (ins g8rc:$reg),
                    "ld 2, 8($reg)", LdStLD,
                    [(PPCload_toc i64:$reg)]>, isPPC64;
                    
let RST = 2, DS = 10, RA = 1 in
def LDtoc_restore : DSForm_1a<58, 0, (outs), (ins),
                    "ld 2, 40(1)", LdStLD,
                    [(PPCtoc_restore)]>, isPPC64;
}
def LDX  : XForm_1<31,  21, (outs g8rc:$rD), (ins memrr:$src),
                   "ldx $rD, $src", LdStLD,
                   [(set i64:$rD, (load xaddr:$src))]>, isPPC64;
def LDBRX : XForm_1<31,  532, (outs g8rc:$rD), (ins memrr:$src),
                   "ldbrx $rD, $src", LdStLoad,
                   [(set i64:$rD, (PPClbrx xoaddr:$src, i64))]>, isPPC64;

let mayLoad = 1, neverHasSideEffects = 1 in {
def LDU  : DSForm_1<58, 1, (outs g8rc:$rD, ptr_rc_nor0:$ea_result), (ins memrix:$addr),
                    "ldu $rD, $addr", LdStLDU,
                    []>, RegConstraint<"$addr.reg = $ea_result">, isPPC64,
                    NoEncode<"$ea_result">;

def LDUX : XForm_1<31, 53, (outs g8rc:$rD, ptr_rc_nor0:$ea_result),
                   (ins memrr:$addr),
                   "ldux $rD, $addr", LdStLDU,
                   []>, RegConstraint<"$addr.ptrreg = $ea_result">,
                   NoEncode<"$ea_result">, isPPC64;
}
}

def : Pat<(PPCload ixaddr:$src),
          (LD ixaddr:$src)>;
def : Pat<(PPCload xaddr:$src),
          (LDX xaddr:$src)>;

// Support for medium and large code model.
def ADDIStocHA: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, tocentry:$disp),
                       "#ADDIStocHA",
                       [(set i64:$rD,
                         (PPCaddisTocHA i64:$reg, tglobaladdr:$disp))]>,
                       isPPC64;
def LDtocL: Pseudo<(outs g8rc:$rD), (ins tocentry:$disp, g8rc_nox0:$reg),
                   "#LDtocL",
                   [(set i64:$rD,
                     (PPCldTocL tglobaladdr:$disp, i64:$reg))]>, isPPC64;
def ADDItocL: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, tocentry:$disp),
                     "#ADDItocL",
                     [(set i64:$rD,
                       (PPCaddiTocL i64:$reg, tglobaladdr:$disp))]>, isPPC64;

// Support for thread-local storage.
def ADDISgotTprelHA: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                         "#ADDISgotTprelHA",
                         [(set i64:$rD,
                           (PPCaddisGotTprelHA i64:$reg,
                                               tglobaltlsaddr:$disp))]>,
                  isPPC64;
def LDgotTprelL: Pseudo<(outs g8rc:$rD), (ins s16imm64:$disp, g8rc_nox0:$reg),
                        "#LDgotTprelL",
                        [(set i64:$rD,
                          (PPCldGotTprelL tglobaltlsaddr:$disp, i64:$reg))]>,
                 isPPC64;
def : Pat<(PPCaddTls i64:$in, tglobaltlsaddr:$g),
          (ADD8TLS $in, tglobaltlsaddr:$g)>;
def ADDIStlsgdHA: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                         "#ADDIStlsgdHA",
                         [(set i64:$rD,
                           (PPCaddisTlsgdHA i64:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;
def ADDItlsgdL : Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                       "#ADDItlsgdL",
                       [(set i64:$rD,
                         (PPCaddiTlsgdL i64:$reg, tglobaltlsaddr:$disp))]>,
                 isPPC64;
def GETtlsADDR : Pseudo<(outs g8rc:$rD), (ins g8rc:$reg, tlsgd:$sym),
                        "#GETtlsADDR",
                        [(set i64:$rD,
                          (PPCgetTlsAddr i64:$reg, tglobaltlsaddr:$sym))]>,
                 isPPC64;
def ADDIStlsldHA: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                         "#ADDIStlsldHA",
                         [(set i64:$rD,
                           (PPCaddisTlsldHA i64:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;
def ADDItlsldL : Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                       "#ADDItlsldL",
                       [(set i64:$rD,
                         (PPCaddiTlsldL i64:$reg, tglobaltlsaddr:$disp))]>,
                 isPPC64;
def GETtlsldADDR : Pseudo<(outs g8rc:$rD), (ins g8rc:$reg, tlsgd:$sym),
                          "#GETtlsldADDR",
                          [(set i64:$rD,
                            (PPCgetTlsldAddr i64:$reg, tglobaltlsaddr:$sym))]>,
                   isPPC64;
def ADDISdtprelHA: Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                          "#ADDISdtprelHA",
                          [(set i64:$rD,
                            (PPCaddisDtprelHA i64:$reg,
                                              tglobaltlsaddr:$disp))]>,
                   isPPC64;
def ADDIdtprelL : Pseudo<(outs g8rc:$rD), (ins g8rc_nox0:$reg, s16imm64:$disp),
                         "#ADDIdtprelL",
                         [(set i64:$rD,
                           (PPCaddiDtprelL i64:$reg, tglobaltlsaddr:$disp))]>,
                  isPPC64;

let PPC970_Unit = 2 in {
let Interpretation64Bit = 1 in {
// Truncating stores.                       
def STB8 : DForm_1<38, (outs), (ins g8rc:$rS, memri:$src),
                   "stb $rS, $src", LdStStore,
                   [(truncstorei8 i64:$rS, iaddr:$src)]>;
def STH8 : DForm_1<44, (outs), (ins g8rc:$rS, memri:$src),
                   "sth $rS, $src", LdStStore,
                   [(truncstorei16 i64:$rS, iaddr:$src)]>;
def STW8 : DForm_1<36, (outs), (ins g8rc:$rS, memri:$src),
                   "stw $rS, $src", LdStStore,
                   [(truncstorei32 i64:$rS, iaddr:$src)]>;
def STBX8 : XForm_8<31, 215, (outs), (ins g8rc:$rS, memrr:$dst),
                   "stbx $rS, $dst", LdStStore,
                   [(truncstorei8 i64:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
def STHX8 : XForm_8<31, 407, (outs), (ins g8rc:$rS, memrr:$dst),
                   "sthx $rS, $dst", LdStStore,
                   [(truncstorei16 i64:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
def STWX8 : XForm_8<31, 151, (outs), (ins g8rc:$rS, memrr:$dst),
                   "stwx $rS, $dst", LdStStore,
                   [(truncstorei32 i64:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
} // Interpretation64Bit

// Normal 8-byte stores.
def STD  : DSForm_1<62, 0, (outs), (ins g8rc:$rS, memrix:$dst),
                    "std $rS, $dst", LdStSTD,
                    [(aligned4store i64:$rS, ixaddr:$dst)]>, isPPC64;
def STDX  : XForm_8<31, 149, (outs), (ins g8rc:$rS, memrr:$dst),
                   "stdx $rS, $dst", LdStSTD,
                   [(store i64:$rS, xaddr:$dst)]>, isPPC64,
                   PPC970_DGroup_Cracked;
def STDBRX: XForm_8<31, 660, (outs), (ins g8rc:$rS, memrr:$dst),
                   "stdbrx $rS, $dst", LdStStore,
                   [(PPCstbrx i64:$rS, xoaddr:$dst, i64)]>, isPPC64,
                   PPC970_DGroup_Cracked;
}

// Stores with Update (pre-inc).
let PPC970_Unit = 2, mayStore = 1 in {
let Interpretation64Bit = 1 in {
def STBU8 : DForm_1<39, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memri:$dst),
                   "stbu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STHU8 : DForm_1<45, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memri:$dst),
                   "sthu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STWU8 : DForm_1<37, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memri:$dst),
                   "stwu $rS, $dst", LdStStoreUpd, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">;
def STDU : DSForm_1<62, 1, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memrix:$dst),
                   "stdu $rS, $dst", LdStSTDU, []>,
                   RegConstraint<"$dst.reg = $ea_res">, NoEncode<"$ea_res">,
                   isPPC64;

def STBUX8: XForm_8<31, 247, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memrr:$dst),
                    "stbux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.ptrreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
def STHUX8: XForm_8<31, 439, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memrr:$dst),
                    "sthux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.ptrreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
def STWUX8: XForm_8<31, 183, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memrr:$dst),
                    "stwux $rS, $dst", LdStStoreUpd, []>,
                    RegConstraint<"$dst.ptrreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked;
} // Interpretation64Bit

def STDUX : XForm_8<31, 181, (outs ptr_rc_nor0:$ea_res), (ins g8rc:$rS, memrr:$dst),
                    "stdux $rS, $dst", LdStSTDU, []>,
                    RegConstraint<"$dst.ptrreg = $ea_res">, NoEncode<"$ea_res">,
                    PPC970_DGroup_Cracked, isPPC64;
}

// Patterns to match the pre-inc stores.  We can't put the patterns on
// the instruction definitions directly as ISel wants the address base
// and offset to be separate operands, not a single complex operand.
def : Pat<(pre_truncsti8 i64:$rS, iPTR:$ptrreg, iaddroff:$ptroff),
          (STBU8 $rS, iaddroff:$ptroff, $ptrreg)>;
def : Pat<(pre_truncsti16 i64:$rS, iPTR:$ptrreg, iaddroff:$ptroff),
          (STHU8 $rS, iaddroff:$ptroff, $ptrreg)>;
def : Pat<(pre_truncsti32 i64:$rS, iPTR:$ptrreg, iaddroff:$ptroff),
          (STWU8 $rS, iaddroff:$ptroff, $ptrreg)>;
def : Pat<(aligned4pre_store i64:$rS, iPTR:$ptrreg, iaddroff:$ptroff),
          (STDU $rS, iaddroff:$ptroff, $ptrreg)>;

def : Pat<(pre_truncsti8 i64:$rS, iPTR:$ptrreg, iPTR:$ptroff),
          (STBUX8 $rS, $ptrreg, $ptroff)>;
def : Pat<(pre_truncsti16 i64:$rS, iPTR:$ptrreg, iPTR:$ptroff),
          (STHUX8 $rS, $ptrreg, $ptroff)>;
def : Pat<(pre_truncsti32 i64:$rS, iPTR:$ptrreg, iPTR:$ptroff),
          (STWUX8 $rS, $ptrreg, $ptroff)>;
def : Pat<(pre_store i64:$rS, iPTR:$ptrreg, iPTR:$ptroff),
          (STDUX $rS, $ptrreg, $ptroff)>;


//===----------------------------------------------------------------------===//
// Floating point instructions.
//


let PPC970_Unit = 3, neverHasSideEffects = 1,
    Uses = [RM] in {  // FPU Operations.
defm FCFID  : XForm_26r<63, 846, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fcfid", "$frD, $frB", FPGeneral,
                        [(set f64:$frD, (PPCfcfid f64:$frB))]>, isPPC64;
defm FCTID  : XForm_26r<63, 814, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fctid", "$frD, $frB", FPGeneral,
                        []>, isPPC64;
defm FCTIDZ : XForm_26r<63, 815, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fctidz", "$frD, $frB", FPGeneral,
                        [(set f64:$frD, (PPCfctidz f64:$frB))]>, isPPC64;

defm FCFIDU  : XForm_26r<63, 974, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fcfidu", "$frD, $frB", FPGeneral,
                        [(set f64:$frD, (PPCfcfidu f64:$frB))]>, isPPC64;
defm FCFIDS  : XForm_26r<59, 846, (outs f4rc:$frD), (ins f8rc:$frB),
                        "fcfids", "$frD, $frB", FPGeneral,
                        [(set f32:$frD, (PPCfcfids f64:$frB))]>, isPPC64;
defm FCFIDUS : XForm_26r<59, 974, (outs f4rc:$frD), (ins f8rc:$frB),
                        "fcfidus", "$frD, $frB", FPGeneral,
                        [(set f32:$frD, (PPCfcfidus f64:$frB))]>, isPPC64;
defm FCTIDUZ : XForm_26r<63, 943, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fctiduz", "$frD, $frB", FPGeneral,
                        [(set f64:$frD, (PPCfctiduz f64:$frB))]>, isPPC64;
defm FCTIWUZ : XForm_26r<63, 143, (outs f8rc:$frD), (ins f8rc:$frB),
                        "fctiwuz", "$frD, $frB", FPGeneral,
                        [(set f64:$frD, (PPCfctiwuz f64:$frB))]>, isPPC64;
}


//===----------------------------------------------------------------------===//
// Instruction Patterns
//

// Extensions and truncates to/from 32-bit regs.
def : Pat<(i64 (zext i32:$in)),
          (RLDICL (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $in, sub_32),
                  0, 32)>;
def : Pat<(i64 (anyext i32:$in)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $in, sub_32)>;
def : Pat<(i32 (trunc i64:$in)),
          (EXTRACT_SUBREG $in, sub_32)>;

// Extending loads with i64 targets.
def : Pat<(zextloadi1 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(zextloadi1 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi1 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(extloadi1 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi8 iaddr:$src),
          (LBZ8 iaddr:$src)>;
def : Pat<(extloadi8 xaddr:$src),
          (LBZX8 xaddr:$src)>;
def : Pat<(extloadi16 iaddr:$src),
          (LHZ8 iaddr:$src)>;
def : Pat<(extloadi16 xaddr:$src),
          (LHZX8 xaddr:$src)>;
def : Pat<(extloadi32 iaddr:$src),
          (LWZ8 iaddr:$src)>;
def : Pat<(extloadi32 xaddr:$src),
          (LWZX8 xaddr:$src)>;

// Standard shifts.  These are represented separately from the real shifts above
// so that we can distinguish between shifts that allow 6-bit and 7-bit shift
// amounts.
def : Pat<(sra i64:$rS, i32:$rB),
          (SRAD $rS, $rB)>;
def : Pat<(srl i64:$rS, i32:$rB),
          (SRD $rS, $rB)>;
def : Pat<(shl i64:$rS, i32:$rB),
          (SLD $rS, $rB)>;

// SHL/SRL
def : Pat<(shl i64:$in, (i32 imm:$imm)),
          (RLDICR $in, imm:$imm, (SHL64 imm:$imm))>;
def : Pat<(srl i64:$in, (i32 imm:$imm)),
          (RLDICL $in, (SRL64 imm:$imm), imm:$imm)>;

// ROTL
def : Pat<(rotl i64:$in, i32:$sh),
          (RLDCL $in, $sh, 0)>;
def : Pat<(rotl i64:$in, (i32 imm:$imm)),
          (RLDICL $in, imm:$imm, 0)>;

// Hi and Lo for Darwin Global Addresses.
def : Pat<(PPChi tglobaladdr:$in, 0), (LIS8 tglobaladdr:$in)>;
def : Pat<(PPClo tglobaladdr:$in, 0), (LI8  tglobaladdr:$in)>;
def : Pat<(PPChi tconstpool:$in , 0), (LIS8 tconstpool:$in)>;
def : Pat<(PPClo tconstpool:$in , 0), (LI8  tconstpool:$in)>;
def : Pat<(PPChi tjumptable:$in , 0), (LIS8 tjumptable:$in)>;
def : Pat<(PPClo tjumptable:$in , 0), (LI8  tjumptable:$in)>;
def : Pat<(PPChi tblockaddress:$in, 0), (LIS8 tblockaddress:$in)>;
def : Pat<(PPClo tblockaddress:$in, 0), (LI8  tblockaddress:$in)>;
def : Pat<(PPChi tglobaltlsaddr:$g, i64:$in),
          (ADDIS8 $in, tglobaltlsaddr:$g)>;
def : Pat<(PPClo tglobaltlsaddr:$g, i64:$in),
          (ADDI8 $in, tglobaltlsaddr:$g)>;
def : Pat<(add i64:$in, (PPChi tglobaladdr:$g, 0)),
          (ADDIS8 $in, tglobaladdr:$g)>;
def : Pat<(add i64:$in, (PPChi tconstpool:$g, 0)),
          (ADDIS8 $in, tconstpool:$g)>;
def : Pat<(add i64:$in, (PPChi tjumptable:$g, 0)),
          (ADDIS8 $in, tjumptable:$g)>;
def : Pat<(add i64:$in, (PPChi tblockaddress:$g, 0)),
          (ADDIS8 $in, tblockaddress:$g)>;

// Patterns to match r+r indexed loads and stores for
// addresses without at least 4-byte alignment.
def : Pat<(i64 (unaligned4sextloadi32 xoaddr:$src)),
          (LWAX xoaddr:$src)>;
def : Pat<(i64 (unaligned4load xoaddr:$src)),
          (LDX xoaddr:$src)>;
def : Pat<(unaligned4store i64:$rS, xoaddr:$dst),
          (STDX $rS, xoaddr:$dst)>;

