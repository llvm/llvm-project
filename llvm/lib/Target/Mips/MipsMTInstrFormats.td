//===-- MipsMTInstrFormats.td - Mips Instruction Formats ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe the MIPS MT instructions format
//
//  opcode - operation code.
//  rt     - destination register
//
//===----------------------------------------------------------------------===//

class MipsMTInst : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther>,
                   PredicateControl {
  let DecoderNamespace = "Mips";
  let EncodingPredicates = [HasStdEnc];
}

class OPCODE1<bits<1> Val> {
  bits<1> Value = Val;
}

def OPCODE_SC_DMT : OPCODE1<0b0>;
def OPCODE_SC_EMT : OPCODE1<0b1>;

class FIELD5<bits<5> Val> {
  bits<5> Value = Val;
}

def FIELD5_1_DMT_EMT  : FIELD5<0b00001>;
def FIELD5_2_DMT_EMT  : FIELD5<0b01111>;

class COP0_MFMC0_MT<FIELD5 Op1, FIELD5 Op2, OPCODE1 sc> : MipsMTInst {
  bits<32> Inst;

  bits<5> rt;
  let Inst{31-26} = 0b010000; // COP0
  let Inst{25-21} = 0b01011;  // MFMC0
  let Inst{20-16} = rt;
  let Inst{15-11} = Op1.Value;
  let Inst{10-6}  = Op2.Value;
  let Inst{5}     = sc.Value;
  let Inst{4-3}   = 0b00;
  let Inst{2-0}   = 0b001;
}
