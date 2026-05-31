//===-- SuperHInstrFormats.td - SuperH Instruction Formats ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class SHInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {

  let Namespace = "SH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  
  field bits<16> Inst;
  let Size = 1;
}

class SHDSPInst<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {

  let Namespace = "SH";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
  
  field bits<32> Inst;
  let Size = 2;
}

// SH PSEUDO INSTRUCTION
class SHPseudo<dag outs, dag ins, list<dag> pattern = []>
    : SHInst<outs, ins, "; error: this should not be emitted", pattern> {
  let isPseudo = 1;
}
