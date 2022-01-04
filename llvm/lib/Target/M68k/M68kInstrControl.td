//===-- M68kInstrControl.td - Control Flow Instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the M68k jump, return, call, and related instructions.
/// Here is the current status of the file:
///
///  Machine:
///
///       BRA   [x]     BSR  [ ]     Bcc [ ]     DBcc [ ]     FBcc [ ]
///       FDBcc [ ]     FNOP [ ]     FPn [ ]     FScc [ ]     FTST [ ]
///       JMP   [~]     JSR  [x]     NOP [x]     RTD  [!]     RTR  [ ]
///       RTS   [x]     Scc  [x]     TST [ ]
///
///  Pseudo:
///
///          RET [x]
///    TCRETURNj [x]   TCRETURNq [x]
///     TAILJMPj [x]    TAILJMPq [x]
///
///  Map:
///
///   [ ] - was not touched at all
///   [!] - requires extarnal stuff implemented
///   [~] - in progress but usable
///   [x] - done
///
///
///                                   NOTE
///      Though branch and jump instructions are using memory operands they
///      DO NOT read the jump address from memory, they just calculate EA
///      and jump there.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NOP
//===----------------------------------------------------------------------===//

let hasSideEffects = 0 in {
  def NOP : MxInst<(outs), (ins), "nop", [], MxEncFixed<0x4E71>>;
}


//===----------------------------------------------------------------------===//
// Conditions
//===----------------------------------------------------------------------===//

/// CC—Carry clear      GE—Greater than or equal
/// LS—Lower or same    PL—Plus
/// CS—Carry set        GT—Greater than
/// LT—Less than        T—Always true*
/// EQ—Equal            HI—Higher
/// MI—Minus            VC—Overflow clear
/// F—Never true*       LE—Less than or equal
/// NE—Not equal        VS—Overflow set
///
/// *Not applicable to the Bcc instructions.
def MxCCt  : MxBead4Bits<0b0000>;
def MxCCf  : MxBead4Bits<0b0001>;
def MxCChi : MxBead4Bits<0b0010>;
def MxCCls : MxBead4Bits<0b0011>;
def MxCCcc : MxBead4Bits<0b0100>;
def MxCCcs : MxBead4Bits<0b0101>;
def MxCCne : MxBead4Bits<0b0110>;
def MxCCeq : MxBead4Bits<0b0111>;
def MxCCvc : MxBead4Bits<0b1000>;
def MxCCvs : MxBead4Bits<0b1001>;
def MxCCpl : MxBead4Bits<0b1010>;
def MxCCmi : MxBead4Bits<0b1011>;
def MxCCge : MxBead4Bits<0b1100>;
def MxCClt : MxBead4Bits<0b1101>;
def MxCCgt : MxBead4Bits<0b1110>;
def MxCCle : MxBead4Bits<0b1111>;

/// --------------------------------+---------+---------
///  F  E  D  C | B  A  9  8 | 7  6 | 5  4  3 | 2  1  0
/// --------------------------------+---------+---------
///  0  1  0  1 | CONDITION  | 1  1 |   MODE  |   REG
/// ----------------------------------------------------
class MxSccEncoding<MxEncEA EA, MxEncExt EXT, MxBead4Bits CC>
    : MxEncoding<EA.Reg, EA.DA, EA.Mode, MxBead2Bits<0b11>, CC, MxBead4Bits<0b0101>,
                 EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>;

let Uses = [CCR] in {
class MxSccR<string CC>
    : MxInst<(outs MxDRD8:$dst), (ins), "s"#CC#"\t$dst",
             [(set i8:$dst, (MxSetCC !cast<PatLeaf>("MxCOND"#CC), CCR))],
             MxSccEncoding<MxEncEAd_0, MxExtEmpty,
                           !cast<MxBead4Bits>("MxCC"#CC)>>;

class MxSccM<string CC, MxOperand MEMOpd, ComplexPattern MEMPat,
             MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins MEMOpd:$dst), "s"#CC#"\t$dst",
             [(store (MxSetCC !cast<PatLeaf>("MxCOND"#CC), CCR), MEMPat:$dst)],
             MxSccEncoding<EA, EXT, !cast<MxBead4Bits>("MxCC"#CC)>>;
}

foreach cc = [ "cc", "ls", "lt", "eq", "mi", "f", "ne", "ge",
               "cs", "pl", "gt", "t", "hi", "vc", "le", "vs"] in {
def SET#"d8"#cc : MxSccR<cc>;
def SET#"j8"#cc : MxSccM<cc, MxType8.JOp, MxType8.JPat, MxEncEAj_0, MxExtEmpty>;
def SET#"p8"#cc : MxSccM<cc, MxType8.POp, MxType8.PPat, MxEncEAp_0, MxExtI16_0>;
}

//===----------------------------------------------------------------------===//
// Jumps
//===----------------------------------------------------------------------===//

///------------------------------+---------+---------
/// F  E  D  C  B  A  9  8  7  6 | 5  4  3 | 2  1  0
///------------------------------+---------+---------
/// 0  1  0  0  1  1  1  0  1  1 |  MODE   |   REG
///------------------------------+---------+---------
let isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1 in
class MxJMP<MxOperand LOCOp, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins LOCOp:$dst), "jmp\t$dst", [(brind iPTR:$dst)],
             MxEncoding<EA.Reg, EA.DA, EA.Mode, MxBead2Bits<0b11>,
                        MxBead4Bits<0b1110>, MxBead4Bits<0b0100>,
                        EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>>;

def JMP32j : MxJMP<MxARI32, MxEncEAj_0, MxExtEmpty>;


// FIXME Support 16 bit indirect jump.
// Currently M68k does not allow 16 bit indirect jumps use sext operands
// def JMP16r     : MxInst<(outs), (ins M68k_ARI16:$dst),
//                             "jmp\t$dst",
//                             [(brind AR16:$dst)]>;

//===----------------------------------------------------------------------===//
// Branches
//===----------------------------------------------------------------------===//

/// --------------------------------------------------
///  F  E  D  C | B  A  9  8 | 7  6  5  4  3  2  1  0
/// --------------------------------------------------
///  0  1  1  0 | CONDITION |   8-BIT DISPLACEMENT
/// --------------------------------------------------
///  16-BIT DISPLACEMENT IF 8-BIT DISPLACEMENT = $00
/// --------------------------------------------------
///  32-BIT DISPLACEMENT IF 8-BIT DISPLACEMENT = $FF
/// --------------------------------------------------
let isBranch = 1, isTerminator = 1, Uses = [CCR] in
class MxBcc<string cc, Operand TARGET, MxEncoding ENC = MxEncEmpty>
    : MxInst<(outs), (ins TARGET:$dst), "b"#cc#"\t$dst", [], ENC>;

foreach cc = [ "cc", "ls", "lt", "eq", "mi", "ne", "ge",
               "cs", "pl", "gt", "hi", "vc", "le", "vs"] in {
  def B#cc#"8"
    : MxBcc<cc, MxBrTarget8,
            MxEncoding<MxBead8Disp<0>,
                       !cast<MxBead4Bits>("MxCC"#cc), MxBead4Bits<0x6>>>;
  def B#cc#"16"
    : MxBcc<cc, MxBrTarget16,
            MxEncoding<MxBead4Bits<0x0>,
                       MxBead4Bits<0x0>, !cast<MxBead4Bits>("MxCC"#cc),
                       MxBead4Bits<0x6>, MxBead16Imm<0>>>;
}

foreach cc = [ "cc", "ls", "lt", "eq", "mi", "ne", "ge",
               "cs", "pl", "gt", "hi", "vc", "le", "vs"] in {
def : Pat<(MxBrCond bb:$target, !cast<PatLeaf>("MxCOND"#cc), CCR),
          (!cast<Instruction>("B"#cc#"8") MxBrTarget8:$target)>;
}

/// -------------------------------------------------
///  F  E  D  C  B  A  9  8 | 7  6  5  4  3  2  1  0
/// -------------------------------------------------
///  0  1  1  0  0  0  0  0 |   8-BIT DISPLACEMENT
/// -------------------------------------------------
///  16-BIT DISPLACEMENT IF 8-BIT DISPLACEMENT = $00
/// -------------------------------------------------
///  32-BIT DISPLACEMENT IF 8-BIT DISPLACEMENT = $FF
/// -------------------------------------------------
let isBranch = 1, isTerminator = 1, isBarrier=1 in
class MxBra<Operand TARGET, MxEncoding ENC = MxEncEmpty>
    : MxInst<(outs), (ins TARGET:$dst), "bra\t$dst", [], ENC>;

def BRA8  : MxBra<MxBrTarget8,
                  MxEncoding<MxBead8Disp<0>, MxBead4Bits<0x0>,
                             MxBead4Bits<0x6>>>;
def BRA16 : MxBra<MxBrTarget16,
                  MxEncoding<MxBead4Bits<0x0>, MxBead4Bits<0x0>,
                             MxBead4Bits<0x0>, MxBead4Bits<0x6>,
                             MxBead16Imm<0>>>;

def : Pat<(br bb:$target), (BRA8 MxBrTarget8:$target)>;


//===----------------------------------------------------------------------===//
// Call
//===----------------------------------------------------------------------===//

// All calls clobber the non-callee saved registers. %SP is marked as
// a use to prevent stack-pointer assignments that appear immediately
// before calls from potentially appearing dead. Uses for argument
// registers are added manually.
let Uses = [SP] in
let isCall = 1 in
///------------------------------+---------+---------
/// F  E  D  C  B  A  9  8  7  6 | 5  4  3 | 2  1  0
///------------------------------+---------+---------
/// 0  1  0  0  1  1  1  0  1  0 |  MODE   |   REG
///------------------------------+---------+---------
class MxCall<MxOperand LOCOp, MxEncEA EA, MxEncExt EXT>
    : MxInst<(outs), (ins LOCOp:$dst), "jsr\t$dst", [],
             MxEncoding<EA.Reg, EA.DA, EA.Mode, MxBead2Bits<0b10>,
                        MxBead4Bits<0b1110>, MxBead4Bits<0b0100>,
                        EXT.Imm, EXT.B8, EXT.Scale, EXT.WL, EXT.DAReg>>;

def CALLk : MxCall<MxPCI32, MxEncEAk,   MxExtBrief_0>;
def CALLq : MxCall<MxPCD32, MxEncEAq,   MxExtI16_0>;
def CALLb : MxCall<MxAL32,  MxEncEAb,   MxExtI32_0>;
def CALLj : MxCall<MxARI32, MxEncEAj_0, MxExtEmpty>;

multiclass CallPat<MxCall callOp, Predicate pred> {
  let Predicates = [pred] in {
    def : Pat<(MxCall (i32 tglobaladdr:$dst)),  (callOp tglobaladdr:$dst)>;
    def : Pat<(MxCall (i32 texternalsym:$dst)), (callOp texternalsym:$dst)>;
    def : Pat<(MxCall (i32 imm:$dst)),          (callOp imm:$dst)>;
  }
}

defm : CallPat<CALLq, IsPIC>;
defm : CallPat<CALLb, IsNotPIC>;

def : Pat<(MxCall iPTR:$dst), (CALLj MxARI32:$dst)>;

//===----------------------------------------------------------------------===//
// Tail Call
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {
let Uses = [SP] in {
let isCall = 1, isTerminator = 1, isBarrier = 1 in {

let isReturn = 1 in
def TCRETURNq : MxPseudo<(outs), (ins MxPCD32:$dst,    i32imm:$adj)>;
def TAILJMPq  : MxPseudo<(outs), (ins MxPCD32:$dst)>;

// NOTE j does not mean load and jump M68k jmp just calculates EA and jumps
// and it is using Mem form like (An) thus j letter.
let isReturn = 1 in
def TCRETURNj : MxPseudo<(outs), (ins MxARI32_TC:$dst, i32imm:$adj)>;
def TAILJMPj  : MxPseudo<(outs), (ins MxARI32_TC:$dst)>;
} // isCall = 1, isTerminator = 1, isBarrier = 1
} // Uses = [SP]
} // isCodeGenOnly = 1

//===----------------------------------------------------------------------===//
// Return
//===----------------------------------------------------------------------===//

// TODO Implement LINK/UNLK

let isTerminator = 1, isReturn = 1, isBarrier = 1, hasCtrlDep = 1 in {

def RTS : MxInst<(outs), (ins), "rts", [], MxEncFixed<0x4E75>>;

let isCodeGenOnly = 1 in
def RET : MxPseudo<(outs), (ins i32imm:$adj, variable_ops),
                   [(MxRet timm:$adj)]>;
} // isTerminator = 1, isReturn = 1, isBarrier = 1, hasCtrlDep = 1

//===----------------------------------------------------------------------===//
// SETCC_C Patterns
//===----------------------------------------------------------------------===//

// Use subx to materialize carry bit.
let Uses = [CCR], Defs = [CCR], isPseudo = 1 in {
// FIXME These are pseudo ops that should be replaced with Pat<> patterns.
// However, Pat<> can't replicate the destination reg into the inputs of the
// result.
def SETCS_C8d : MxPseudo<(outs MxDRD8:$dst), (ins),
                         [(set MxDRD8:$dst, (MxSetCC_C MxCONDcs, CCR))]>;
def SETCS_C16d : MxPseudo<(outs MxDRD16:$dst), (ins),
                          [(set MxDRD16:$dst, (MxSetCC_C MxCONDcs, CCR))]>;
def SETCS_C32d : MxPseudo<(outs MxXRD32:$dst), (ins),
                          [(set MxXRD32:$dst, (MxSetCC_C MxCONDcs, CCR))]>;
} // Uses = [CCR], Defs = [CCR], isPseudo = 1


def : Pat<(i16 (anyext (i8 (MxSetCC_C MxCONDcs, CCR)))), (SETCS_C16d)>;
def : Pat<(i32 (anyext (i8 (MxSetCC_C MxCONDcs, CCR)))), (SETCS_C32d)>;

def : Pat<(i16 (sext (i8 (MxSetCC_C MxCONDcs, CCR)))), (SETCS_C16d)>;
def : Pat<(i32 (sext (i8 (MxSetCC_C MxCONDcs, CCR)))), (SETCS_C32d)>;

// We canonicalize 'scs' to "(and (subx reg,reg), 1)" on the hope that the and
// will be eliminated and that the subx can be extended up to a wider type.  When
// this happens, it is great.  However, if we are left with an 8-bit subx and an
// and, we might as well just match it as a setb.
def : Pat<(and (i8 (MxSetCC_C MxCONDcs, CCR)), 1), (SETd8cs)>;

// (add OP, SETB) -> (addx OP, (move 0))
def : Pat<(add (and (i8 (MxSetCC_C MxCONDcs, CCR)), 1), MxDRD8:$op),
          (ADDX8dd MxDRD8:$op, (MOV8di 0))>;
def : Pat<(add (and (i32 (MxSetCC_C MxCONDcs, CCR)), 1), MxXRD32:$op),
          (ADDX32dd MxDRD32:$op, (MOV32ri 0))>;

// (sub OP, SETB) -> (subx OP, (move 0))
def : Pat<(sub MxDRD8:$op, (and (i8 (MxSetCC_C MxCONDcs, CCR)), 1)),
          (SUBX8dd MxDRD8:$op, (MOV8di 0))>;
def : Pat<(sub MxXRD32:$op, (and (i32 (MxSetCC_C MxCONDcs, CCR)), 1)),
          (SUBX32dd MxDRD32:$op, (MOV32ri 0))>;

// (sub OP, SETCC_CARRY) -> (addx OP, (move 0))
def : Pat<(sub MxDRD8:$op, (i8 (MxSetCC_C MxCONDcs, CCR))),
          (ADDX8dd MxDRD8:$op, (MOV8di 0))>;
def : Pat<(sub MxXRD32:$op, (i32 (MxSetCC_C MxCONDcs, CCR))),
          (ADDX32dd MxDRD32:$op, (MOV32ri 0))>;
