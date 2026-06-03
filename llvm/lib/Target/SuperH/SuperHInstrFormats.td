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

// Base of all 16-bit SuperH instructions
class SHInst <dag outs, dag ins, string asmstr, list<dag> pattern> : Instruction {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;
  
  bits<16> Opcode = 0;

  let Namespace = "SH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  let Size = 2;
}

// SuperH Psuedo Instruction
class SHPseudo<dag outs, dag ins, string asmstr="; error: this should not be emitted", list<dag> pattern = []>
    : SHInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

//===----------------------------------------------------------------------===//
//  Instruction Formats
//===----------------------------------------------------------------------===//

class SHInstOP_N4_I8 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {

  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = Opcode{3-0};

  // Operands
  bits<4> n;
  bits<8> imm;
  let Inst{7-4} = n;
  let Inst{15-8} = imm;
}

class SHInstOP_N4_D8 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  
  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = Opcode{3-0};

  // Operands
  bits<4> n;
  bits<8> disp;
  let Inst{7-4} = n;
  let Inst{15-8} = disp;
}

class SHInstOP_N4_M4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = Opcode{3-0};
  let Inst{15-11} = Opcode{15-11};

  // Operands
  bits<4> n;
  bits<4> m;
  let Inst{7-4} = n;
  let Inst{11-8} = m;
}

class SHInstOP_N4_D4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<4> n;
  bits<4> disp;
  let Inst{11-8} = n;
  let Inst{15-12} = disp;
}

class SHInstOP_M4_D4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<4> m;
  bits<4> disp;
  let Inst{11-8} = m;
  let Inst{15-12} = disp;
}

class SHInstOP_N4_M4_D4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = opcode{3-0};

  // Operands
  bits<4> n;
  bits<4> m;
  bits<4> disp;
  let Inst{7-4} = n;
  let Inst{11-8} = m;
  let Inst{15-12} = disp;
}

class SHInstOP_D8 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<8> disp;
  let Inst{15-8} = disp;
}

class SHInstOP_I8 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{7-0} = Opcode{7-0};

  // Operands
  bits<8> imm;
  let Inst{15-8} = imm;
}

class SHInstOP_N4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = Opcode{3-0};
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<4> n;
  let Inst{7-4} = n;
}

class SHInstOP_M4 <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst{3-0} = Opcode{3-0};
  let Inst{15-8} = Opcode{15-8};

  // Operands
  bits<4> m;
  let Inst{7-4} = m;
}

class SHInstOP <bits<16> opcode, dag outs, dag ins, string asmstr> 
  : SHInst<outs, ins, asmstr, []> {
  
  // Opcode
  let Opcode = opcode;
  let Inst = Opcode;
}

