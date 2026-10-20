//===-- Toy64InstrFormats.td - Toy64 Instruction Formats -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the binary instruction formats used by Toy64.
//
// Toy64 currently uses fixed-width 32-bit instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base Toy64 instruction
//===----------------------------------------------------------------------===//

class Toy64Inst<dag outs, dag ins, string asmstr, list<dag> pattern = []> : 
Instruction {
  // Actual machine encoding.
  field bits<32> Inst;

  // Used later by the disassembler.
  field bits<32> SoftFail = 0;

  let Namespace = "Toy64";
  let Size = 4;

  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;

  // Arithmetic instructions do not access memory and have no hidden
  // side effects.
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

//===----------------------------------------------------------------------===//
// R-type instruction
//
//  31       25 24    20 19    15 14  12 11     7 6       0
// +-----------+--------+--------+------+---------+---------+
// | funct7    | rs2    | rs1    |funct3| rd      | opcode  |
// +-----------+--------+--------+------+---------+---------+
//
//===----------------------------------------------------------------------===//

class Toy64RType<bits<7> funct7, bits<3> funct3, bits<7> opcode, dag outs,
    dag ins, string asmstr, list<dag> pattern = [] > : 
    Toy64Inst<outs, ins, asmstr, pattern> {

  bits<5> rd;
  bits<5> rs1;
  bits<5> rs2;

  let Inst{31-25} = funct7;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode;
}

//===----------------------------------------------------------------------===//
// I-type instruction
//
//  31                20 19    15 14  12 11     7 6       0
// +--------------------+--------+------+---------+---------+
// | immediate[11:0]    | rs1    |funct3| rd      | opcode  |
// +--------------------+--------+------+---------+---------+
//
//===----------------------------------------------------------------------===//

class Toy64IType<bits<3> funct3, bits<7> opcode, dag outs, dag ins,
    string asmstr, list<dag> pattern = [] > : 
    Toy64Inst<outs, ins, asmstr, pattern> {

  bits<5> rd;
  bits<5> rs1;
  bits<12> imm12;

  let Inst{31-20} = imm12;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = opcode;
}


//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class Toy64Pseudo<
    dag outs,
    dag ins,
    string asmstr = "",
    list<dag> pattern = []
> : Toy64Inst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  // Pseudos do not have a real machine encoding.
  let Inst{31-0} = 0;
}