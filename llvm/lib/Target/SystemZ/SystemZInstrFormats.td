//===- SystemZInstrFormats.td - SystemZ Instruction Formats ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<5> val> {
  bits<5> Value = val;
}

def Pseudo   : Format<0>;
def EForm    : Format<1>;
def IForm    : Format<2>;
def RIForm   : Format<3>;
def RIEForm  : Format<4>;
def RILForm  : Format<5>;
def RISForm  : Format<6>;
def RRForm   : Format<7>;
def RREForm  : Format<8>;
def RRFForm  : Format<9>;
def RRRForm  : Format<10>;
def RRSForm  : Format<11>;
def RSForm   : Format<12>;
def RSIForm  : Format<13>;
def RSILForm : Format<14>;
def RSYForm  : Format<15>;
def RXForm   : Format<16>;
def RXEForm  : Format<17>;
def RXFForm  : Format<18>;
def RXYForm  : Format<19>;
def SForm    : Format<20>;
def SIForm   : Format<21>;
def SILForm  : Format<22>;
def SIYForm  : Format<23>;
def SSForm   : Format<24>;
def SSEForm  : Format<25>;
def SSFForm  : Format<26>;

class InstSystemZ<bits<16> op, Format f, dag outs, dag ins> : Instruction {
  let Namespace = "SystemZ";

  bits<16> Opcode = op;

  Format Form = f;
  bits<5> FormBits = Form.Value;

  dag OutOperandList = outs;
  dag InOperandList = ins;
}

class I8<bits<8> op, Format f, dag outs, dag ins, string asmstr, 
         list<dag> pattern> 
  : InstSystemZ<0, f, outs, ins> {
  let Opcode{0-7} = op;
  let Opcode{8-15} = 0;

  let Pattern = pattern;
  let AsmString = asmstr;
}

class I12<bits<12> op, Format f, dag outs, dag ins, string asmstr, 
         list<dag> pattern> 
  : InstSystemZ<0, f, outs, ins> {
  let Opcode{0-11} = op;
  let Opcode{12-15} = 0;

  let Pattern = pattern;
  let AsmString = asmstr;
}

class I16<bits<16> op, Format f, dag outs, dag ins, string asmstr,
         list<dag> pattern>
  : InstSystemZ<op, f, outs, ins> {
  let Pattern = pattern;
  let AsmString = asmstr;
}

class RRI<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I8<op, RRForm, outs, ins, asmstr, pattern>;

class RII<bits<12> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I12<op, RIForm, outs, ins, asmstr, pattern>;

class RILI<bits<12> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I12<op, RILForm, outs, ins, asmstr, pattern>;

class RREI<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I16<op, RREForm, outs, ins, asmstr, pattern>;

class RXI<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I8<op, RXForm, outs, ins, asmstr, pattern> {
  let AddedComplexity = 1;
}

class RXYI<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I16<op, RXYForm, outs, ins, asmstr, pattern>;

class RSI<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I8<op, RSForm, outs, ins, asmstr, pattern> {
  let AddedComplexity = 1;
}

class RSYI<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I16<op, RSYForm, outs, ins, asmstr, pattern>;

class SII<bits<8> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I8<op, SIForm, outs, ins, asmstr, pattern> {
  let AddedComplexity = 1;
}

class SIYI<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I16<op, SIYForm, outs, ins, asmstr, pattern>;

class SILI<bits<16> op, dag outs, dag ins, string asmstr, list<dag> pattern>
  : I16<op, SILForm, outs, ins, asmstr, pattern>;


//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<0, Pseudo, outs, ins> {

  let Pattern = pattern;
  let AsmString = asmstr;
}
