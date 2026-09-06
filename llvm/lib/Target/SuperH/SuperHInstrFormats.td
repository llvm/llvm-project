//===-- SuperHInstrFormats.td - SuperH Instruction Formats ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
//  The instruction formats in this file are defined based on the instruction
//  notation defined in the SuperH 4a specification, table 3.3.
//  
//  The SuperH architecture uses a "compressed" instruction set where all
//  instructions (except DSP instructions) are 16 bits wide, as such opcodes
//  often embed the registers, displacement, etc. within the instruction opcode
//  itself.
//
//  SuperH Instructions are generally split up in the following categories:
//   * 0000nnnniiiiiiii (OP, Rn, imm)         OP_N4_I8
//   * 0000nnnndddddddd (OP, Rn, disp)        OP_N4_D8
//   * 0000nnnnmmmm0000 (OP, Rn, Rm, OP)      OP_N4_M4
//   * 00000000nnnndddd (OP, Rn, disp)        OP_N4_D4
//   * 00000000mmmmdddd (OP, Rm, disp)        OP_M4_D4
//   * 0000nnnnmmmmdddd (OP, Rn, Rm, disp)    OP_N4_M4_D4
//   * 00000000dddddddd (OP, disp)            OP_D8
//   * 0000dddddddddddd (OP, disp)            OP_D12
//   * 00000000iiiiiiii (OP, imm)             OP_I8
//   * 0000nnnn00000000 (OP, Rn, OP)          OP_N4
//   * 0000mmmm00000000 (OP, Rm, OP)          OP_M4
//   * 0000000000000000 (OP)                  OP
//
//  These classes are simplified by giving them a naming scheme based on their
//  encoding, eg. 0000nnnniiiiiiii is SHInstOP_N4_I8
//
//  TODO: Implement the DSP extensions into this scheme.
//
//===----------------------------------------------------------------------===//

class SHInst<dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : Instruction {
  let Namespace = "SH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  // Whether the instruction is incompatible with delay slots.
  bit isDelayIllegal = 0;
  let TSFlags{0} = isDelayIllegal;
}


// Base of all 16-bit SuperH instructions
class SHInst16<dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst<outs, ins, asmstr, pattern> {
  field bits<16> SoftFail = 0;
  field bits<16> Inst;
  bits<16> Opcode = 0;
  let Size = 2;
}

class SHInst32<dag outs, dag ins, string asmstr, list<dag> pattern = []>
    : SHInst<outs, ins, asmstr, pattern> {
  field bits<16> SoftFail = 0;
  field bits<16> Inst;
  bits<32> Opcode = 0;
  let Size = 4;
}

// SuperH Pseudo Instruction
// These are not real instructions but are expanded to real instructions
// in later passes implemented in C++.
class SHPseudo<dag outs, dag ins, string asmstr="; error: this should not be emitted", list<dag> pattern = []>
    : SHInst<outs, ins, asmstr, pattern> {
  let Pattern = pattern;

  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

class SHSelectPseudo<dag outs, dag ins, string asmstr, list<dag> pattern> 
      : SHPseudo<outs, ins, asmstr, pattern> {
  let usesCustomInserter = 1;
  let Uses = [SR];
}

class SHLoadPseudo<dag outs, dag ins, string asmstr, list<dag> pattern> 
      : SHPseudo<outs, ins, asmstr, pattern> {
  let mayLoad = 1;
}

class SHStorePseudo<dag outs, dag ins, string asmstr, list<dag> pattern> 
      : SHPseudo<outs, ins, asmstr, pattern> {
  let mayStore = 1;
}




//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//  Instruction Formats
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// No-operand Instruction
// <opopopopopopopop>
//===----------------------------------------------------------------------===//
class Inst<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst = Opcode;
}

//===----------------------------------------------------------------------===//
// Register -> Register Instruction
// <opop|rnrn|rmrm|opop>
//===----------------------------------------------------------------------===//
class InstRmRn<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-12} = Opcode{15-12};
  let Inst{3-0} = Opcode{3-0};

  // Operands
  bits<4> Rn;
  bits<4> Rm;
  let Inst{11-8} = Rn;
  let Inst{7-4} = Rm;
}

//===----------------------------------------------------------------------===//
// (Any) -> Register Instruction
// <opop|rnrn|opopopop>
//===----------------------------------------------------------------------===//
class InstRn<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<4> Rn;
  let Inst{11-8} = Rn;
}

//===----------------------------------------------------------------------===//
// Register -> (Any) Instruction
// <opop|rmrm|opopopop>
//===----------------------------------------------------------------------===//
class InstRm<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<4> Rm;
  let Inst{11-8} = Rm;
}

//===----------------------------------------------------------------------===//
// Immediate -> Register Instruction
// <opop|rnrn|iiiiiiii>
//===----------------------------------------------------------------------===//
class InstRnI8<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {

  // Opcode
  let Opcode = opcode;
  let Inst{15-12} = Opcode{15-12};

  // Operands
  bits<4> Rn;
  bits<8> imm;
  let Inst{11-8} = Rn;
  let Inst{7-0} = imm;
}

//===----------------------------------------------------------------------===//
// (disp:8) -> Register Instruction
// <opop|rnrn|dddddddd>
//===----------------------------------------------------------------------===//
class InstRnD8<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-12} = Opcode{15-12};

  // Operands
  bits<4> Rn;
  bits<8> disp;
  let Inst{11-8} = Rn;
  let Inst{7-0} = disp;
}

//===----------------------------------------------------------------------===//
// (Register + disp:4) -> Register Instruction
// <opop|rnrn|rmrm|dddd>
//===----------------------------------------------------------------------===//
class InstRmRnD4<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-12} = Opcode{15-12};

  // Operands
  bits<4> Rn;
  bits<4> Rm;
  bits<4> disp;
  let Inst{11-8} = Rn;
  let Inst{7-4} = Rm;
  let Inst{3-0} = disp;
}

//===----------------------------------------------------------------------===//
// (disp:4) -> Register Instruction
// <opopopop|rnrn|dddd>
//===----------------------------------------------------------------------===//
class InstRnD4<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<4> Rn;
  bits<4> disp;
  let Inst{7-4} = Rn;
  let Inst{3-0} = disp;
}

//===----------------------------------------------------------------------===//
// (disp:4 + Register) -> Reserved Register Instruction
// <opopopop|rmrm|dddd>
//===----------------------------------------------------------------------===//
class InstRmD4<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<4> Rm;
  bits<4> disp;
  let Inst{7-4} = Rm;
  let Inst{3-0} = disp;
}

//===----------------------------------------------------------------------===//
// (disp:8) Instruction
// <opopopop|dddddddd>
//===----------------------------------------------------------------------===//
class InstD8<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<8> disp;
  let Inst{7-0} = disp;
}

//===----------------------------------------------------------------------===//
// (imm:8) Instruction
// <opopopop|iiiiiiii>
//===----------------------------------------------------------------------===//
class InstI8<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<8> imm;
  let Inst{7-0} = imm;
}

//===----------------------------------------------------------------------===//
// (disp:12) Instruction
// <opop|dddddddddddd>
//===----------------------------------------------------------------------===//
class InstD12<bits<16> opcode, dag outs, dag ins, string asmstr, list<dag> pattern = []> 
    : SHInst16<outs, ins, asmstr, pattern> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{15-12} = Opcode{15-12};

  // Operands
  bits<12> disp;
  let Inst{11-0} = disp;
}