//===-- Nios2InstrFormats.td - Nios2 Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe NIOS2 instructions format
//
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<3> val> {
  bits<3> Value = val;
}

def Pseudo : Format<0>;
def FrmI : Format<1>;
def FrmR : Format<2>;
def FrmJ : Format<3>;
def FrmOther : Format<4>; // Instruction w/ a custom format

// Generic Nios2 Format
class Nios2Inst<dag outs, dag ins, string asmstr, list<dag> pattern, Format f>
    : Instruction {
  field bits<32> Inst;
  Format Form = f;

  let Namespace = "Nios2";

  let Size = 4;

  bits<6> Opcode = 0;

  // Bottom 6 bits are the 'opcode' field
  let Inst{5 - 0} = Opcode;

  let OutOperandList = outs;
  let InOperandList = ins;

  let AsmString = asmstr;
  let Pattern = pattern;

  //
  // Attributes specific to Nios2 instructions:
  //
  bits<3> FormBits = Form.Value;

  // TSFlags layout should be kept in sync with Nios2InstrInfo.h.
  let TSFlags{2 - 0} = FormBits;

  let DecoderNamespace = "Nios2";
}

// Nios2 Instruction Format
class InstSE<dag outs, dag ins, string asmstr, list<dag> pattern, Format f>
    : Nios2Inst<outs, ins, asmstr, pattern, f> {
}

//===----------------------------------------------------------------------===//
// Format I instruction class in Nios2 : <|A|B|immediate|opcode|>
//===----------------------------------------------------------------------===//

class FI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstSE<outs, ins, asmstr, pattern, FrmI> {
  bits<5> rA;
  bits<5> rB;
  bits<16> imm;

  let Opcode = op;

  let Inst{31 - 27} = rA;
  let Inst{26 - 22} = rB;
  let Inst{21 - 6} = imm;
}

//===----------------------------------------------------------------------===//
// Format R instruction : <|A|B|C|opx|imm|opcode|>
//===----------------------------------------------------------------------===//

class FR<bits<6> opx, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstSE<outs, ins, asmstr, pattern, FrmR> {
  bits<5> rA;
  bits<5> rB;
  bits<5> rC;
  bits<5> imm = 0;

  // opcode is always 0x3a for R instr.
  let Opcode = 0x3a;

  let Inst{31 - 27} = rA;
  let Inst{26 - 22} = rB;
  let Inst{21 - 17} = rC;
  // opx stands for opcode extension
  let Inst{16 - 11} = opx;
  // optional 5-bit immediate value
  let Inst{10 - 6}  = imm;
}

//===----------------------------------------------------------------------===//
// Format J instruction class in Nios2 : <|address|opcode|>
//===----------------------------------------------------------------------===//

class FJ<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstSE<outs, ins, asmstr, pattern, FrmJ> {
  bits<26> addr;

  let Opcode = op;

  let Inst{31 - 6} = addr;
}
