//===-- XCoreInstrFormats.td - XCore Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
class InstXCore<dag outs, dag ins, string asmstr, list<dag> pattern>
    : Instruction {
  field bits<32> Inst;

  let Namespace = "XCore";
  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

// XCore pseudo instructions format
class PseudoInstXCore<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstXCore<outs, ins, asmstr, pattern>;

//===----------------------------------------------------------------------===//
// Instruction formats
//===----------------------------------------------------------------------===//

class _F3R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FL3R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _F2RUS<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FL2RUS<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FRU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FLRU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FLU6<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FU10<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FLU10<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _F2R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FRUS<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _FL2R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _F1R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _F0R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _L4R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _L5R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}

class _L6R<dag outs, dag ins, string asmstr, list<dag> pattern>
    : InstXCore<outs, ins, asmstr, pattern> {
  let Inst{31-0} = 0;
}
