//===- IA64InstrFormats.td - IA64 Instruction Formats --*- tablegen -*-=//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  - Warning: the stuff in here isn't really being used, so is mostly
//             junk. It'll get fixed as the JIT gets built.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

class InstIA64<bits<4> op, dag OOL, dag IOL, string asmstr> : Instruction { 
  // IA64 instruction baseline
  field bits<41> Inst;
  let Namespace = "IA64";
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let AsmString = asmstr;

  let Inst{40-37} = op;
}

//"Each Itanium instruction is categorized into one of six types."
//We should have:
// A, I, M, F, B, L+X

class AForm<bits<4> opcode, bits<6> qpReg, dag OOL, dag IOL, string asmstr> : 
  InstIA64<opcode, OOL, IOL, asmstr> {

  let Inst{5-0} = qpReg;
}

class AForm_DAG<bits<4> opcode, bits<6> qpReg, dag OOL, dag IOL, string asmstr,
      list<dag> pattern> : 
  InstIA64<opcode, OOL, IOL, asmstr> {

  let Pattern = pattern;
  let Inst{5-0} = qpReg;
}

let isBranch = 1, isTerminator = 1 in
class BForm<bits<4> opcode, bits<6> x6, bits<3> btype, dag OOL, dag IOL, string asmstr> :
  InstIA64<opcode, OOL, IOL, asmstr> {

  let Inst{32-27} = x6;
  let Inst{8-6} = btype;
}

class MForm<bits<4> opcode, bits<6> x6, dag OOL, dag IOL, string asmstr> :
  InstIA64<opcode, OOL, IOL, asmstr> {
    bits<7> Ra;
    bits<7> Rb;
    bits<16> disp;

    let Inst{35-30} = x6;
//  let Inst{20-16} = Rb;
    let Inst{15-0} = disp;
}

class RawForm<bits<4> opcode, bits<26> rest, dag OOL, dag IOL, string asmstr> :
  InstIA64<opcode, OOL, IOL, asmstr> {
    let Inst{25-0} = rest;
}

// Pseudo instructions.
class PseudoInstIA64<dag OOL, dag IOL, string nm> : InstIA64<0, OOL, IOL, nm>  {
}

class PseudoInstIA64_DAG<dag OOL, dag IOL, string nm, list<dag> pattern>
  : InstIA64<0, OOL, IOL, nm> {
  let Pattern = pattern;
}

