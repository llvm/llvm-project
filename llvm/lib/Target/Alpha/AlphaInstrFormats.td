//===- AlphaInstrFormats.td - Alpha Instruction Formats ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

//3.3:
//Memory
//Branch
//Operate
//Floating-point
//PALcode

def u8imm   : Operand<i64>;
def s14imm  : Operand<i64>;
def s16imm  : Operand<i64>;
def s21imm  : Operand<i64>;
def s64imm  : Operand<i64>;
def u64imm  : Operand<i64>;

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//
// Alpha instruction baseline
class InstAlpha<bits<6> op, string asmstr, InstrItinClass itin> : Instruction {
  field bits<32> Inst;
  let Namespace = "Alpha";
  let AsmString = asmstr;
  let Inst{31-26} = op;
  let Itinerary = itin;
}


//3.3.1
class MForm<bits<6> opcode, bit load, string asmstr, list<dag> pattern, InstrItinClass itin> 
        : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let canFoldAsLoad = load;
  let Defs = [R28]; //We may use this for frame index calculations, so reserve it here

  bits<5> Ra;
  bits<16> disp;
  bits<5> Rb;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-0} = disp;
}
class MfcForm<bits<6> opcode, bits<16> fc, string asmstr, InstrItinClass itin> 
        : InstAlpha<opcode, asmstr, itin> {    
  bits<5> Ra;

  let OutOperandList = (ops GPRC:$RA);
  let InOperandList = (ops);
  let Inst{25-21} = Ra;
  let Inst{20-16} = 0;
  let Inst{15-0} = fc;
}
class MfcPForm<bits<6> opcode, bits<16> fc, string asmstr, InstrItinClass itin> 
        : InstAlpha<opcode, asmstr, itin> {    
  let OutOperandList = (ops);
  let InOperandList = (ops);
  let Inst{25-21} = 0;
  let Inst{20-16} = 0;
  let Inst{15-0} = fc;
}

class MbrForm<bits<6> opcode, bits<2> TB, dag OL, string asmstr, InstrItinClass itin>
    : InstAlpha<opcode, asmstr, itin> {
  bits<5> Ra;
  bits<5> Rb;
  bits<14> disp;

  let OutOperandList = (ops);
  let InOperandList = OL;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-14} = TB;
  let Inst{13-0} = disp;
}
class MbrpForm<bits<6> opcode, bits<2> TB, dag OL, string asmstr, list<dag> pattern, InstrItinClass itin>
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern=pattern;
  bits<5> Ra;
  bits<5> Rb;
  bits<14> disp;

  let OutOperandList = (ops);
  let InOperandList = OL;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-14} = TB;
  let Inst{13-0} = disp;
}

//3.3.2
def target : Operand<OtherVT> {}

let isBranch = 1, isTerminator = 1, hasCtrlDep = 1 in {
class BFormN<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
   : InstAlpha<opcode, asmstr, itin> {
  let OutOperandList = (ops);
  let InOperandList = OL;
  bits<64> Opc; //dummy
  bits<5> Ra;
  bits<21> disp;

  let Inst{25-21} = Ra;
  let Inst{20-0} = disp;
}
}

let isBranch = 1, isTerminator = 1 in
class BFormD<bits<6> opcode, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (ops);
  let InOperandList = (ops target:$DISP);
  bits<5> Ra;
  bits<21> disp;

  let Inst{25-21} = Ra;
  let Inst{20-0} = disp;
}

//3.3.3
class OForm<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (outs GPRC:$RC);
  let InOperandList = (ins GPRC:$RA, GPRC:$RB);

  bits<5> Rc;
  bits<5> Ra;
  bits<5> Rb;
  bits<7> Function = fun;

  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm2<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (outs GPRC:$RC);
  let InOperandList = (ins GPRC:$RB);

  bits<5> Rc;
  bits<5> Rb;
  bits<7> Function = fun;

  let Inst{25-21} = 31;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm4<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (outs GPRC:$RDEST);
  let InOperandList = (ins GPRC:$RCOND, GPRC:$RTRUE, GPRC:$RFALSE);
  let Constraints = "$RFALSE = $RDEST";
  let DisableEncoding = "$RFALSE";

  bits<5> Rc;
  bits<5> Ra;
  bits<5> Rb;
  bits<7> Function = fun;

//  let isTwoAddress = 1;
  let Inst{25-21} = Ra;
  let Inst{20-16} = Rb;
  let Inst{15-13} = 0;
  let Inst{12} = 0;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}


class OFormL<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (outs GPRC:$RC);
  let InOperandList = (ins GPRC:$RA, u8imm:$L);

  bits<5> Rc;
  bits<5> Ra;
  bits<8> LIT;
  bits<7> Function = fun;

  let Inst{25-21} = Ra;
  let Inst{20-13} = LIT;
  let Inst{12} = 1;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

class OForm4L<bits<6> opcode, bits<7> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;
  let OutOperandList = (outs GPRC:$RDEST);
  let InOperandList = (ins GPRC:$RCOND, s64imm:$RTRUE, GPRC:$RFALSE);
  let Constraints = "$RFALSE = $RDEST";
  let DisableEncoding = "$RFALSE";

  bits<5> Rc;
  bits<5> Ra;
  bits<8> LIT;
  bits<7> Function = fun;

//  let isTwoAddress = 1;
  let Inst{25-21} = Ra;
  let Inst{20-13} = LIT;
  let Inst{12} = 1;
  let Inst{11-5} = Function;
  let Inst{4-0} = Rc;
}

//3.3.4
class FPForm<bits<6> opcode, bits<11> fun, string asmstr, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<opcode, asmstr, itin> {
  let Pattern = pattern;

  bits<5> Fc;
  bits<5> Fa;
  bits<5> Fb;
  bits<11> Function = fun;

  let Inst{25-21} = Fa;
  let Inst{20-16} = Fb;
  let Inst{15-5} = Function;
  let Inst{4-0} = Fc;
}

//3.3.5
class PALForm<bits<6> opcode, dag OL, string asmstr, InstrItinClass itin>
    : InstAlpha<opcode, asmstr, itin> {
  let OutOperandList = (ops);
  let InOperandList = OL;
  bits<26> Function;

  let Inst{25-0} = Function;
}


// Pseudo instructions.
class PseudoInstAlpha<dag OOL, dag IOL, string nm, list<dag> pattern, InstrItinClass itin> 
    : InstAlpha<0, nm, itin>  {
  let OutOperandList = OOL;
  let InOperandList = IOL;
  let Pattern = pattern;

}
