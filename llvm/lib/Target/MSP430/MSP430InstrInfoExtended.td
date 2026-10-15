//===-- MSP430InstrInfoExtended.td - MSP430X Instruction defs -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the MSP430X extended instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

// TODO: Most extended instructions have a .A version that enables 20-bit
// operands. This requires support for the large memory model.
//
// Only the extended instructions that are useful in the small memory model
// (16-bit pointers) are implemented so far. Most MOVX forms are not needed
// there and are larger and slower than MOV. MOVX.A @Rs, x(Rd) is the
// exception, because 20-bit registers like the DMA address registers can only
// be written with a single 20-bit store.

//===----------------------------------------------------------------------===//
//  Operands
//
def Rpt2ImmAsmOperand : AsmOperandClass {
  let Name = "Rpt2Imm";
  let RenderMethod = "addImmOperands";
}

def rpt2imm : Operand<i8>,
              ImmLeaf<i8, [{return Imm >= 1 && Imm <= 4;}]> {
  let ParserMatchClass = Rpt2ImmAsmOperand;
  let EncoderMethod = "getRpt2ImmOpValue";
  let DecoderMethod = "decodeRptImm";
}

def Rpt4ImmAsmOperand : AsmOperandClass {
  let Name = "Rpt4Imm";
  let RenderMethod = "addImmOperands";
}

def rpt4imm : Operand<i8>,
              ImmLeaf<i8, [{return Imm >= 1 && Imm <= 16;}]> {
  let ParserMatchClass = Rpt4ImmAsmOperand;
  let EncoderMethod = "getRpt4ImmOpValue";
  let DecoderMethod = "decodeRptImm";
}

// POPM encodes Rdst - n + 1 for Rdst, while PUSHM encodes Rdst.
def popmreg : RegisterOperand<GR16> {
  let EncoderMethod = "getPOPMRegOpValue";
  let DecoderMethod = "decodePOPMRegOperand";
}

// Memory operand with a 20-bit offset. Bits 19:16 go into the extension word.
// Only constant offsets are accepted, because there are no 20-bit fixups yet.
def Mem20AsmOperand : AsmOperandClass {
  let Name = "Mem20";
  let RenderMethod = "addMemOperands";
}

def memdst20 : Operand<i32> {
  let PrintMethod = "printSrcMemOperand";
  let MIOperandInfo = (ops GR16, i32imm);
  let ParserMatchClass = Mem20AsmOperand;
  let EncoderMethod = "getMemOpValue";
  let DecoderMethod = "decodeMem20Operand";
}

//===----------------------------------------------------------------------===//
//  Shift Instructions
//
let Predicates = [HasMSP430X] in {
let Constraints = "$rs = $rd", Defs = [SR] in {
  multiclass IIExtExcRot<string Mnemonic, bits<2> OpCode,
                         list<Register> uses> {
    let Uses = uses in
    def _16 : IIExtExcRotForm16<OpCode,
                                (outs GR16:$rd), (ins GR16:$rs, rpt2imm:$cnt),
                                !strconcat(Mnemonic, "\t$cnt, $rd"), []>;
  }

  defm RRCM : IIExtExcRot<"rrcm", 0b00, [SR]>;
  defm RRAM : IIExtExcRot<"rram", 0b01, []>;
  defm RRUM : IIExtExcRot<"rrum", 0b11, []>;
  defm RLAM : IIExtExcRot<"rlam", 0b10, []>;

  multiclass IIExtRot<string Mnemonic, bits<3> OpCode, bit ZC,
                      list<Register> uses> {
    let Uses = uses in {
    def _8  :  II8rExtended<OpCode, ZC, 0,
                            (outs GR8:$rd), (ins GR8:$rs),
                            !strconcat(Mnemonic, ".b\t$rd"), []>;
    def _16 : II16rExtended<OpCode, ZC, 0,
                            (outs GR16:$rd), (ins GR16:$rs),
                            !strconcat(Mnemonic, "\t$rd"), []>;
    }
  }
  defm RRCX : IIExtRot<"rrcx", 0b000, 0, [SR]>;
  defm RRAX : IIExtRot<"rrax", 0b010, 0, []>;
  // RRUX is encoded as RRCX with the ZC flag set.
  defm RRUX : IIExtRot<"rrux", 0b000, 1, []>;

} // Defs = [SR], Constraints = "$rs = $rd"

//===----------------------------------------------------------------------===//
//  Arithmetic Instructions
//
let Defs = [SR], Constraints = "$src2 = $rd" in {
  def ADDX8rr  : I8rExtended<0b0101, 0, 0,
                             (outs GR8:$rd), (ins GR8:$src2, GR8:$rs),
                             "addx.b\t$rs, $rd", []>;
  def ADDX16rr : I16rExtended<0b0101, 0, 0,
                              (outs GR16:$rd), (ins GR16:$src2, GR16:$rs),
                              "addx\t$rs, $rd", []>;
}

// Alias printing is disabled for the same reason as for RLA.
def RLAX8r  : InstAlias<"rlax.b\t$dst", (ADDX8rr  GR8:$dst,  GR8:$dst),  0>;
def RLAX16r : InstAlias<"rlax\t$dst",   (ADDX16rr GR16:$dst, GR16:$dst), 0>;

//===----------------------------------------------------------------------===//
//  Move Instructions
//
let mayLoad = 1, mayStore = 1, hasSideEffects = 0 in {
def MOVXAmn  : IAmnExtended<0b0100, (outs), (ins memdst20:$dst, indreg:$rs),
                            "movx.a\t$rs, $dst", []>;
def MOVX16mn : I16mnExtended<0b0100, (outs), (ins memdst20:$dst, indreg:$rs),
                             "movx\t$rs, $dst", []>;
}

//===----------------------------------------------------------------------===//
//  Push/Pop Multiple Registers
//
// $rs is the highest register of the range. The other registers depend on the
// count, so codegen must add them as implicit operands when it builds the
// instruction.
let Defs = [SP], Uses = [SP], hasSideEffects = 0 in {
let mayStore = 1 in
def PUSHM16r : IIExtExcPushPopForm16<0, (outs), (ins rpt4imm:$cnt, GR16:$rs),
                                     "pushm\t$cnt, $rs", []>;
let mayLoad = 1 in
def POPM16r  : IIExtExcPushPopForm16<1, (outs), (ins rpt4imm:$cnt, popmreg:$rs),
                                     "popm\t$cnt, $rs", []>;
}
} // Predicates = [HasMSP430X]
