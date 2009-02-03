//===- PPCInstr64Bit.td - The PowerPC 64-bit Support -------*- tablegen -*-===//
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
}
def u16imm64 : Operand<i64> {
  let PrintMethod = "printU16ImmOperand";
}
def symbolHi64 : Operand<i64> {
  let PrintMethod = "printSymbolHi";
}
def symbolLo64 : Operand<i64> {
  let PrintMethod = "printSymbolLo";
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

let Defs = [LR8] in
  def MovePCtoLR8 : Pseudo<(outs), (ins piclabel:$label), "bl $label", []>,
                    PPC970_Unit_BRU;

// Macho ABI Calls.
let isCall = 1, PPC970_Unit = 7, 
  // All calls clobber the PPC64 non-callee saved registers.
  Defs = [X0,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,
          LR8,CTR8,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL8_Macho  : IForm<18, 0, 1,
                           (outs), (ins calltarget:$func, variable_ops), 
                           "bl $func", BrB, []>;  // See Pat patterns below.
    def BLA8_Macho : IForm<18, 1, 1,
                           (outs), (ins aaddr:$func, variable_ops),
                           "bla $func", BrB, [(PPCcall_Macho (i64 imm:$func))]>;
  }
  let Uses = [CTR8, RM] in {
    def BCTRL8_Macho : XLForm_2_ext<19, 528, 20, 0, 1, 
                                 (outs), (ins variable_ops),
                                 "bctrl", BrB,
                                 [(PPCbctrl_Macho)]>, Requires<[In64BitMode]>;
  }
}

// ELF 64 ABI Calls = Macho ABI Calls
// Used to define BL8_ELF and BLA8_ELF
let isCall = 1, PPC970_Unit = 7, 
  // All calls clobber the PPC64 non-callee saved registers.
  Defs = [X0,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,
          F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,
          V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,
          LR8,CTR8,
          CR0,CR1,CR5,CR6,CR7] in {
  // Convenient aliases for call instructions
  let Uses = [RM] in {
    def BL8_ELF  : IForm<18, 0, 1,
                         (outs), (ins calltarget:$func, variable_ops), 
                         "bl $func", BrB, []>;  // See Pat patterns below.                            
    def BLA8_ELF : IForm<18, 1, 1,
                         (outs), (ins aaddr:$func, variable_ops),
                         "bla $func", BrB, [(PPCcall_ELF (i64 imm:$func))]>;
  }
  let Uses = [CTR8, RM] in {
    def BCTRL8_ELF : XLForm_2_ext<19, 528, 20, 0, 1,
                               (outs), (ins variable_ops),
                               "bctrl", BrB,
                               [(PPCbctrl_ELF)]>, Requires<[In64BitMode]>;
  }
}


// Calls
def : Pat<(PPCcall_Macho (i64 tglobaladdr:$dst)),
          (BL8_Macho tglobaladdr:$dst)>;
def : Pat<(PPCcall_Macho (i64 texternalsym:$dst)),
          (BL8_Macho texternalsym:$dst)>;

def : Pat<(PPCcall_ELF (i64 tglobaladdr:$dst)),
          (BL8_ELF tglobaladdr:$dst)>;
def : Pat<(PPCcall_ELF (i64 texternalsym:$dst)),
          (BL8_ELF texternalsym:$dst)>;

// Atomic operations
let usesCustomDAGSchedInserter = 1 in {
  let Uses = [CR0] in {
    def ATOMIC_LOAD_ADD_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_ADD_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_add_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_SUB_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_SUB_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_sub_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_OR_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_OR_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_or_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_XOR_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_XOR_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_xor_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_AND_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_AND_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_and_64 xoaddr:$ptr, G8RC:$incr))]>;
    def ATOMIC_LOAD_NAND_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$incr),
      "${:comment} ATOMIC_LOAD_NAND_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_load_nand_64 xoaddr:$ptr, G8RC:$incr))]>;

    def ATOMIC_CMP_SWAP_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$old, G8RC:$new),
      "${:comment} ATOMIC_CMP_SWAP_I64 PSEUDO!",
      [(set G8RC:$dst, 
                    (atomic_cmp_swap_64 xoaddr:$ptr, G8RC:$old, G8RC:$new))]>;

    def ATOMIC_SWAP_I64 : Pseudo<
      (outs G8RC:$dst), (ins memrr:$ptr, G8RC:$new),
      "${:comment} ATOMIC_SWAP_I64 PSEUDO!",
      [(set G8RC:$dst, (atomic_swap_64 xoaddr:$ptr, G8RC:$new))]>;
  }
}

// Instructions to support atomic operations
def LDARX : XForm_1<31,  84, (outs G8RC:$rD), (ins memrr:$ptr),
                   "ldarx $rD, $ptr", LdStLDARX,
                   [(set G8RC:$rD, (PPClarx xoaddr:$ptr))]>;

let Defs = [CR0] in
def STDCX : XForm_1<31, 214, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stdcx. $rS, $dst", LdStSTDCX,
                   [(PPCstcx G8RC:$rS, xoaddr:$dst)]>,
                   isDOT;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNdi8 :Pseudo< (outs),
                        (ins calltarget:$dst, i32imm:$offset, variable_ops),
                 "#TC_RETURNd8 $dst $offset",
                 []>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNai8 :Pseudo<(outs), (ins aaddr:$func, i32imm:$offset, variable_ops),
                 "#TC_RETURNa8 $func $offset",
                 [(PPCtc_return (i64 imm:$func), imm:$offset)]>;

let isCall = 1, isTerminator = 1, isReturn = 1, isBarrier = 1, Uses = [RM] in
def TCRETURNri8 : Pseudo<(outs), (ins CTRRC8:$dst, i32imm:$offset, variable_ops),
                 "#TC_RETURNr8 $dst $offset",
                 []>;


let isTerminator = 1, isBarrier = 1, PPC970_Unit = 7, isBranch = 1,
    isIndirectBranch = 1, isCall = 1, isReturn = 1, Uses = [CTR, RM] in
def TAILBCTR8 : XLForm_2_ext<19, 528, 20, 0, 0, (outs), (ins), "bctr", BrB, []>,
    Requires<[In64BitMode]>;



let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILB8   : IForm<18, 0, 0, (outs), (ins calltarget:$dst),
                  "b $dst", BrB,
                  []>;


let isBranch = 1, isTerminator = 1, hasCtrlDep = 1, PPC970_Unit = 7,
    isBarrier = 1, isCall = 1, isReturn = 1, Uses = [RM] in
def TAILBA8   : IForm<18, 0, 0, (outs), (ins aaddr:$dst),
                  "ba $dst", BrB,
                  []>;

def : Pat<(PPCtc_return (i64 tglobaladdr:$dst),  imm:$imm),
          (TCRETURNdi8 tglobaladdr:$dst, imm:$imm)>;

def : Pat<(PPCtc_return (i64 texternalsym:$dst), imm:$imm),
          (TCRETURNdi8 texternalsym:$dst, imm:$imm)>;

def : Pat<(PPCtc_return CTRRC8:$dst, imm:$imm),
          (TCRETURNri8 CTRRC8:$dst, imm:$imm)>;


//===----------------------------------------------------------------------===//
// 64-bit SPR manipulation instrs.

let Uses = [CTR8] in {
def MFCTR8 : XFXForm_1_ext<31, 339, 9, (outs G8RC:$rT), (ins),
                           "mfctr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Pattern = [(PPCmtctr G8RC:$rS)], Defs = [CTR8] in {
def MTCTR8 : XFXForm_7_ext<31, 467, 9, (outs), (ins G8RC:$rS),
                           "mtctr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}

let Defs = [X1], Uses = [X1] in
def DYNALLOC8 : Pseudo<(outs G8RC:$result), (ins G8RC:$negsize, memri:$fpsi),
                       "${:comment} DYNALLOC8 $result, $negsize, $fpsi",
                       [(set G8RC:$result,
                             (PPCdynalloc G8RC:$negsize, iaddr:$fpsi))]>;

let Defs = [LR8] in {
def MTLR8  : XFXForm_7_ext<31, 467, 8, (outs), (ins G8RC:$rS),
                           "mtlr $rS", SprMTSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}
let Uses = [LR8] in {
def MFLR8  : XFXForm_1_ext<31, 339, 8, (outs G8RC:$rT), (ins),
                           "mflr $rT", SprMFSPR>,
             PPC970_DGroup_First, PPC970_Unit_FXU;
}

//===----------------------------------------------------------------------===//
// Fixed point instructions.
//

let PPC970_Unit = 1 in {  // FXU Operations.

// Copies, extends, truncates.
def OR4To8  : XForm_6<31, 444, (outs G8RC:$rA), (ins GPRC:$rS, GPRC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;
def OR8To4  : XForm_6<31, 444, (outs GPRC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   []>;

def LI8  : DForm_2_r0<14, (outs G8RC:$rD), (ins symbolLo64:$imm),
                      "li $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, immSExt16:$imm)]>;
def LIS8 : DForm_2_r0<15, (outs G8RC:$rD), (ins symbolHi64:$imm),
                      "lis $rD, $imm", IntGeneral,
                      [(set G8RC:$rD, imm16ShiftedSExt:$imm)]>;

// Logical ops.
def NAND8: XForm_6<31, 476, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "nand $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (and G8RC:$rS, G8RC:$rB)))]>;
def AND8 : XForm_6<31,  28, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "and $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (and G8RC:$rS, G8RC:$rB))]>;
def ANDC8: XForm_6<31,  60, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "andc $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (and G8RC:$rS, (not G8RC:$rB)))]>;
def OR8  : XForm_6<31, 444, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "or $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (or G8RC:$rS, G8RC:$rB))]>;
def NOR8 : XForm_6<31, 124, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "nor $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (or G8RC:$rS, G8RC:$rB)))]>;
def ORC8 : XForm_6<31, 412, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "orc $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (or G8RC:$rS, (not G8RC:$rB)))]>;
def EQV8 : XForm_6<31, 284, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "eqv $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (not (xor G8RC:$rS, G8RC:$rB)))]>;
def XOR8 : XForm_6<31, 316, (outs G8RC:$rA), (ins G8RC:$rS, G8RC:$rB),
                   "xor $rA, $rS, $rB", IntGeneral,
                   [(set G8RC:$rA, (xor G8RC:$rS, G8RC:$rB))]>;

// Logical ops with immediate.
def ANDIo8  : DForm_4<28, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "andi. $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (and G8RC:$src1, immZExt16:$src2))]>,
                      isDOT;
def ANDISo8 : DForm_4<29, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                     "andis. $dst, $src1, $src2", IntGeneral,
                    [(set G8RC:$dst, (and G8RC:$src1,imm16ShiftedZExt:$src2))]>,
                     isDOT;
def ORI8    : DForm_4<24, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "ori $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (or G8RC:$src1, immZExt16:$src2))]>;
def ORIS8   : DForm_4<25, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "oris $dst, $src1, $src2", IntGeneral,
                    [(set G8RC:$dst, (or G8RC:$src1, imm16ShiftedZExt:$src2))]>;
def XORI8   : DForm_4<26, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "xori $dst, $src1, $src2", IntGeneral,
                      [(set G8RC:$dst, (xor G8RC:$src1, immZExt16:$src2))]>;
def XORIS8  : DForm_4<27, (outs G8RC:$dst), (ins G8RC:$src1, u16imm:$src2),
                      "xoris $dst, $src1, $src2", IntGeneral,
                   [(set G8RC:$dst, (xor G8RC:$src1, imm16ShiftedZExt:$src2))]>;

def ADD8  : XOForm_1<31, 266, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "add $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (add G8RC:$rA, G8RC:$rB))]>;
                     
def ADDC8 : XOForm_1<31, 10, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "addc $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (addc G8RC:$rA, G8RC:$rB))]>,
                     PPC970_DGroup_Cracked;
def ADDE8 : XOForm_1<31, 138, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "adde $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (adde G8RC:$rA, G8RC:$rB))]>;
                     
def ADDI8  : DForm_2<14, (outs G8RC:$rD), (ins G8RC:$rA, s16imm64:$imm),
                     "addi $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (add G8RC:$rA, immSExt16:$imm))]>;
def ADDIS8 : DForm_2<15, (outs G8RC:$rD), (ins G8RC:$rA, symbolHi64:$imm),
                     "addis $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (add G8RC:$rA, imm16ShiftedSExt:$imm))]>;

def SUBFIC8: DForm_2< 8, (outs G8RC:$rD), (ins G8RC:$rA, s16imm64:$imm),
                     "subfic $rD, $rA, $imm", IntGeneral,
                     [(set G8RC:$rD, (subc immSExt16:$imm, G8RC:$rA))]>;
def SUBF8 : XOForm_1<31, 40, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "subf $rT, $rA, $rB", IntGeneral,
                     [(set G8RC:$rT, (sub G8RC:$rB, G8RC:$rA))]>;
def SUBFC8 : XOForm_1<31, 8, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                      "subfc $rT, $rA, $rB", IntGeneral,
                      [(set G8RC:$rT, (subc G8RC:$rB, G8RC:$rA))]>,
                      PPC970_DGroup_Cracked;

def SUBFE8 : XOForm_1<31, 136, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                      "subfe $rT, $rA, $rB", IntGeneral,
                      [(set G8RC:$rT, (sube G8RC:$rB, G8RC:$rA))]>;
def ADDME8  : XOForm_3<31, 234, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "addme $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (adde G8RC:$rA, immAllOnes))]>;
def ADDZE8  : XOForm_3<31, 202, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "addze $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (adde G8RC:$rA, 0))]>;
def NEG8    : XOForm_3<31, 104, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "neg $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (ineg G8RC:$rA))]>;
def SUBFME8 : XOForm_3<31, 232, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "subfme $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (sube immAllOnes, G8RC:$rA))]>;
def SUBFZE8 : XOForm_3<31, 200, 0, (outs G8RC:$rT), (ins G8RC:$rA),
                       "subfze $rT, $rA", IntGeneral,
                       [(set G8RC:$rT, (sube 0, G8RC:$rA))]>;



def MULHD : XOForm_1<31, 73, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulhd $rT, $rA, $rB", IntMulHW,
                     [(set G8RC:$rT, (mulhs G8RC:$rA, G8RC:$rB))]>;
def MULHDU : XOForm_1<31, 9, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulhdu $rT, $rA, $rB", IntMulHWU,
                     [(set G8RC:$rT, (mulhu G8RC:$rA, G8RC:$rB))]>;

def CMPD   : XForm_16_ext<31, 0, (outs CRRC:$crD), (ins G8RC:$rA, G8RC:$rB),
                          "cmpd $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPLD  : XForm_16_ext<31, 32, (outs CRRC:$crD), (ins G8RC:$rA, G8RC:$rB),
                          "cmpld $crD, $rA, $rB", IntCompare>, isPPC64;
def CMPDI  : DForm_5_ext<11, (outs CRRC:$crD), (ins G8RC:$rA, s16imm:$imm),
                         "cmpdi $crD, $rA, $imm", IntCompare>, isPPC64;
def CMPLDI : DForm_6_ext<10, (outs CRRC:$dst), (ins G8RC:$src1, u16imm:$src2),
                         "cmpldi $dst, $src1, $src2", IntCompare>, isPPC64;

def SLD  : XForm_6<31,  27, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "sld $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCshl G8RC:$rS, GPRC:$rB))]>, isPPC64;
def SRD  : XForm_6<31, 539, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "srd $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCsrl G8RC:$rS, GPRC:$rB))]>, isPPC64;
def SRAD : XForm_6<31, 794, (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB),
                   "srad $rA, $rS, $rB", IntRotateD,
                   [(set G8RC:$rA, (PPCsra G8RC:$rS, GPRC:$rB))]>, isPPC64;
                   
def EXTSB8 : XForm_11<31, 954, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsb $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i8))]>;
def EXTSH8 : XForm_11<31, 922, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsh $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i16))]>;

def EXTSW  : XForm_11<31, 986, (outs G8RC:$rA), (ins G8RC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext_inreg G8RC:$rS, i32))]>, isPPC64;
/// EXTSW_32 - Just like EXTSW, but works on '32-bit' registers.
def EXTSW_32 : XForm_11<31, 986, (outs GPRC:$rA), (ins GPRC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set GPRC:$rA, (PPCextsw_32 GPRC:$rS))]>, isPPC64;
def EXTSW_32_64 : XForm_11<31, 986, (outs G8RC:$rA), (ins GPRC:$rS),
                      "extsw $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (sext GPRC:$rS))]>, isPPC64;

def SRADI  : XSForm_1<31, 413, (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH),
                      "sradi $rA, $rS, $SH", IntRotateD,
                      [(set G8RC:$rA, (sra G8RC:$rS, (i32 imm:$SH)))]>, isPPC64;
def CNTLZD : XForm_11<31, 58, (outs G8RC:$rA), (ins G8RC:$rS),
                      "cntlzd $rA, $rS", IntGeneral,
                      [(set G8RC:$rA, (ctlz G8RC:$rS))]>;

def DIVD  : XOForm_1<31, 489, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "divd $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (sdiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def DIVDU : XOForm_1<31, 457, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "divdu $rT, $rA, $rB", IntDivD,
                     [(set G8RC:$rT, (udiv G8RC:$rA, G8RC:$rB))]>, isPPC64,
                     PPC970_DGroup_First, PPC970_DGroup_Cracked;
def MULLD : XOForm_1<31, 233, 0, (outs G8RC:$rT), (ins G8RC:$rA, G8RC:$rB),
                     "mulld $rT, $rA, $rB", IntMulHD,
                     [(set G8RC:$rT, (mul G8RC:$rA, G8RC:$rB))]>, isPPC64;


let isCommutable = 1 in {
def RLDIMI : MDForm_1<30, 3,
                      (outs G8RC:$rA), (ins G8RC:$rSi, G8RC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldimi $rA, $rS, $SH, $MB", IntRotateD,
                      []>, isPPC64, RegConstraint<"$rSi = $rA">,
                      NoEncode<"$rSi">;
}

// Rotate instructions.
def RLDCL  : MDForm_1<30, 0,
                      (outs G8RC:$rA), (ins G8RC:$rS, GPRC:$rB, u6imm:$MB),
                      "rldcl $rA, $rS, $rB, $MB", IntRotateD,
                      []>, isPPC64;
def RLDICL : MDForm_1<30, 0,
                      (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH, u6imm:$MB),
                      "rldicl $rA, $rS, $SH, $MB", IntRotateD,
                      []>, isPPC64;
def RLDICR : MDForm_1<30, 1,
                      (outs G8RC:$rA), (ins G8RC:$rS, u6imm:$SH, u6imm:$ME),
                      "rldicr $rA, $rS, $SH, $ME", IntRotateD,
                      []>, isPPC64;
}  // End FXU Operations.


//===----------------------------------------------------------------------===//
// Load/Store instructions.
//


// Sign extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LHA8: DForm_1<42, (outs G8RC:$rD), (ins memri:$src),
                  "lha $rD, $src", LdStLHA,
                  [(set G8RC:$rD, (sextloadi16 iaddr:$src))]>,
                  PPC970_DGroup_Cracked;
def LWA  : DSForm_1<58, 2, (outs G8RC:$rD), (ins memrix:$src),
                    "lwa $rD, $src", LdStLWA,
                    [(set G8RC:$rD, (sextloadi32 ixaddr:$src))]>, isPPC64,
                    PPC970_DGroup_Cracked;
def LHAX8: XForm_1<31, 343, (outs G8RC:$rD), (ins memrr:$src),
                   "lhax $rD, $src", LdStLHA,
                   [(set G8RC:$rD, (sextloadi16 xaddr:$src))]>,
                   PPC970_DGroup_Cracked;
def LWAX : XForm_1<31, 341, (outs G8RC:$rD), (ins memrr:$src),
                   "lwax $rD, $src", LdStLHA,
                   [(set G8RC:$rD, (sextloadi32 xaddr:$src))]>, isPPC64,
                   PPC970_DGroup_Cracked;

// Update forms.
let mayLoad = 1 in
def LHAU8 : DForm_1<43, (outs G8RC:$rD, ptr_rc:$ea_result), (ins symbolLo:$disp,
                            ptr_rc:$rA),
                    "lhau $rD, $disp($rA)", LdStGeneral,
                    []>, RegConstraint<"$rA = $ea_result">,
                    NoEncode<"$ea_result">;
// NO LWAU!

}

// Zero extending loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LBZ8 : DForm_1<34, (outs G8RC:$rD), (ins memri:$src),
                  "lbz $rD, $src", LdStGeneral,
                  [(set G8RC:$rD, (zextloadi8 iaddr:$src))]>;
def LHZ8 : DForm_1<40, (outs G8RC:$rD), (ins memri:$src),
                  "lhz $rD, $src", LdStGeneral,
                  [(set G8RC:$rD, (zextloadi16 iaddr:$src))]>;
def LWZ8 : DForm_1<32, (outs G8RC:$rD), (ins memri:$src),
                  "lwz $rD, $src", LdStGeneral,
                  [(set G8RC:$rD, (zextloadi32 iaddr:$src))]>, isPPC64;

def LBZX8 : XForm_1<31,  87, (outs G8RC:$rD), (ins memrr:$src),
                   "lbzx $rD, $src", LdStGeneral,
                   [(set G8RC:$rD, (zextloadi8 xaddr:$src))]>;
def LHZX8 : XForm_1<31, 279, (outs G8RC:$rD), (ins memrr:$src),
                   "lhzx $rD, $src", LdStGeneral,
                   [(set G8RC:$rD, (zextloadi16 xaddr:$src))]>;
def LWZX8 : XForm_1<31,  23, (outs G8RC:$rD), (ins memrr:$src),
                   "lwzx $rD, $src", LdStGeneral,
                   [(set G8RC:$rD, (zextloadi32 xaddr:$src))]>;
                   
                   
// Update forms.
let mayLoad = 1 in {
def LBZU8 : DForm_1<35, (outs G8RC:$rD, ptr_rc:$ea_result), (ins memri:$addr),
                    "lbzu $rD, $addr", LdStGeneral,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LHZU8 : DForm_1<41, (outs G8RC:$rD, ptr_rc:$ea_result), (ins memri:$addr),
                    "lhzu $rD, $addr", LdStGeneral,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
def LWZU8 : DForm_1<33, (outs G8RC:$rD, ptr_rc:$ea_result), (ins memri:$addr),
                    "lwzu $rD, $addr", LdStGeneral,
                    []>, RegConstraint<"$addr.reg = $ea_result">,
                    NoEncode<"$ea_result">;
}
}


// Full 8-byte loads.
let canFoldAsLoad = 1, PPC970_Unit = 2 in {
def LD   : DSForm_1<58, 0, (outs G8RC:$rD), (ins memrix:$src),
                    "ld $rD, $src", LdStLD,
                    [(set G8RC:$rD, (load ixaddr:$src))]>, isPPC64;
def LDX  : XForm_1<31,  21, (outs G8RC:$rD), (ins memrr:$src),
                   "ldx $rD, $src", LdStLD,
                   [(set G8RC:$rD, (load xaddr:$src))]>, isPPC64;
                   
let mayLoad = 1 in
def LDU  : DSForm_1<58, 1, (outs G8RC:$rD, ptr_rc:$ea_result), (ins memrix:$addr),
                    "ldu $rD, $addr", LdStLD,
                    []>, RegConstraint<"$addr.reg = $ea_result">, isPPC64,
                    NoEncode<"$ea_result">;

}

let PPC970_Unit = 2 in {
// Truncating stores.                       
def STB8 : DForm_1<38, (outs), (ins G8RC:$rS, memri:$src),
                   "stb $rS, $src", LdStGeneral,
                   [(truncstorei8 G8RC:$rS, iaddr:$src)]>;
def STH8 : DForm_1<44, (outs), (ins G8RC:$rS, memri:$src),
                   "sth $rS, $src", LdStGeneral,
                   [(truncstorei16 G8RC:$rS, iaddr:$src)]>;
def STW8 : DForm_1<36, (outs), (ins G8RC:$rS, memri:$src),
                   "stw $rS, $src", LdStGeneral,
                   [(truncstorei32 G8RC:$rS, iaddr:$src)]>;
def STBX8 : XForm_8<31, 215, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stbx $rS, $dst", LdStGeneral,
                   [(truncstorei8 G8RC:$rS, xaddr:$dst)]>, 
                   PPC970_DGroup_Cracked;
def STHX8 : XForm_8<31, 407, (outs), (ins G8RC:$rS, memrr:$dst),
                   "sthx $rS, $dst", LdStGeneral,
                   [(truncstorei16 G8RC:$rS, xaddr:$dst)]>, 
                   PPC970_DGroup_Cracked;
def STWX8 : XForm_8<31, 151, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stwx $rS, $dst", LdStGeneral,
                   [(truncstorei32 G8RC:$rS, xaddr:$dst)]>,
                   PPC970_DGroup_Cracked;
// Normal 8-byte stores.
def STD  : DSForm_1<62, 0, (outs), (ins G8RC:$rS, memrix:$dst),
                    "std $rS, $dst", LdStSTD,
                    [(store G8RC:$rS, ixaddr:$dst)]>, isPPC64;
def STDX  : XForm_8<31, 149, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stdx $rS, $dst", LdStSTD,
                   [(store G8RC:$rS, xaddr:$dst)]>, isPPC64,
                   PPC970_DGroup_Cracked;
}

let PPC970_Unit = 2 in {

def STBU8 : DForm_1<38, (outs ptr_rc:$ea_res), (ins G8RC:$rS,
                             symbolLo:$ptroff, ptr_rc:$ptrreg),
                    "stbu $rS, $ptroff($ptrreg)", LdStGeneral,
                    [(set ptr_rc:$ea_res,
                          (pre_truncsti8 G8RC:$rS, ptr_rc:$ptrreg, 
                                         iaddroff:$ptroff))]>,
                    RegConstraint<"$ptrreg = $ea_res">, NoEncode<"$ea_res">;
def STHU8 : DForm_1<45, (outs ptr_rc:$ea_res), (ins G8RC:$rS,
                             symbolLo:$ptroff, ptr_rc:$ptrreg),
                    "sthu $rS, $ptroff($ptrreg)", LdStGeneral,
                    [(set ptr_rc:$ea_res,
                        (pre_truncsti16 G8RC:$rS, ptr_rc:$ptrreg, 
                                        iaddroff:$ptroff))]>,
                    RegConstraint<"$ptrreg = $ea_res">, NoEncode<"$ea_res">;
def STWU8 : DForm_1<37, (outs ptr_rc:$ea_res), (ins G8RC:$rS,
                             symbolLo:$ptroff, ptr_rc:$ptrreg),
                    "stwu $rS, $ptroff($ptrreg)", LdStGeneral,
                    [(set ptr_rc:$ea_res, (pre_store G8RC:$rS, ptr_rc:$ptrreg, 
                                                     iaddroff:$ptroff))]>,
                    RegConstraint<"$ptrreg = $ea_res">, NoEncode<"$ea_res">;


def STDU : DSForm_1<62, 1, (outs ptr_rc:$ea_res), (ins G8RC:$rS,
                                s16immX4:$ptroff, ptr_rc:$ptrreg),
                    "stdu $rS, $ptroff($ptrreg)", LdStSTD,
                    [(set ptr_rc:$ea_res, (pre_store G8RC:$rS, ptr_rc:$ptrreg, 
                                                     iaddroff:$ptroff))]>,
                    RegConstraint<"$ptrreg = $ea_res">, NoEncode<"$ea_res">,
                    isPPC64;

let mayStore = 1 in
def STDUX : XForm_8<31, 181, (outs), (ins G8RC:$rS, memrr:$dst),
                   "stdux $rS, $dst", LdStSTD,
                   []>, isPPC64;

// STD_32/STDX_32 - Just like STD/STDX, but uses a '32-bit' input register.
def STD_32  : DSForm_1<62, 0, (outs), (ins GPRC:$rT, memrix:$dst),
                       "std $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, ixaddr:$dst)]>, isPPC64;
def STDX_32  : XForm_8<31, 149, (outs), (ins GPRC:$rT, memrr:$dst),
                       "stdx $rT, $dst", LdStSTD,
                       [(PPCstd_32  GPRC:$rT, xaddr:$dst)]>, isPPC64,
                       PPC970_DGroup_Cracked;
}



//===----------------------------------------------------------------------===//
// Floating point instructions.
//


let PPC970_Unit = 3, Uses = [RM] in {  // FPU Operations.
def FCFID  : XForm_26<63, 846, (outs F8RC:$frD), (ins F8RC:$frB),
                      "fcfid $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfcfid F8RC:$frB))]>, isPPC64;
def FCTIDZ : XForm_26<63, 815, (outs F8RC:$frD), (ins F8RC:$frB),
                      "fctidz $frD, $frB", FPGeneral,
                      [(set F8RC:$frD, (PPCfctidz F8RC:$frB))]>, isPPC64;
}


//===----------------------------------------------------------------------===//
// Instruction Patterns
//

// Extensions and truncates to/from 32-bit regs.
def : Pat<(i64 (zext GPRC:$in)),
          (RLDICL (OR4To8 GPRC:$in, GPRC:$in), 0, 32)>;
def : Pat<(i64 (anyext GPRC:$in)),
          (OR4To8 GPRC:$in, GPRC:$in)>;
def : Pat<(i32 (trunc G8RC:$in)),
          (OR8To4 G8RC:$in, G8RC:$in)>;

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
def : Pat<(sra G8RC:$rS, GPRC:$rB),
          (SRAD G8RC:$rS, GPRC:$rB)>;
def : Pat<(srl G8RC:$rS, GPRC:$rB),
          (SRD G8RC:$rS, GPRC:$rB)>;
def : Pat<(shl G8RC:$rS, GPRC:$rB),
          (SLD G8RC:$rS, GPRC:$rB)>;

// SHL/SRL
def : Pat<(shl G8RC:$in, (i32 imm:$imm)),
          (RLDICR G8RC:$in, imm:$imm, (SHL64 imm:$imm))>;
def : Pat<(srl G8RC:$in, (i32 imm:$imm)),
          (RLDICL G8RC:$in, (SRL64 imm:$imm), imm:$imm)>;

// ROTL
def : Pat<(rotl G8RC:$in, GPRC:$sh),
          (RLDCL G8RC:$in, GPRC:$sh, 0)>;
def : Pat<(rotl G8RC:$in, (i32 imm:$imm)),
          (RLDICL G8RC:$in, imm:$imm, 0)>;

// Hi and Lo for Darwin Global Addresses.
def : Pat<(PPChi tglobaladdr:$in, 0), (LIS8 tglobaladdr:$in)>;
def : Pat<(PPClo tglobaladdr:$in, 0), (LI8  tglobaladdr:$in)>;
def : Pat<(PPChi tconstpool:$in , 0), (LIS8 tconstpool:$in)>;
def : Pat<(PPClo tconstpool:$in , 0), (LI8  tconstpool:$in)>;
def : Pat<(PPChi tjumptable:$in , 0), (LIS8 tjumptable:$in)>;
def : Pat<(PPClo tjumptable:$in , 0), (LI8  tjumptable:$in)>;
def : Pat<(add G8RC:$in, (PPChi tglobaladdr:$g, 0)),
          (ADDIS8 G8RC:$in, tglobaladdr:$g)>;
def : Pat<(add G8RC:$in, (PPChi tconstpool:$g, 0)),
          (ADDIS8 G8RC:$in, tconstpool:$g)>;
def : Pat<(add G8RC:$in, (PPChi tjumptable:$g, 0)),
          (ADDIS8 G8RC:$in, tjumptable:$g)>;
