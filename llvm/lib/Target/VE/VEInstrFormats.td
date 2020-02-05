//===-- VEInstrFormats.td - VE Instruction Formats ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class InstVE<dag outs, dag ins, string asmstr, list<dag> pattern>
   : Instruction {
  field bits<64> Inst;

  let Namespace = "VE";
  let Size = 8;

  bits<8> op;
  let Inst{0-7} = op;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;

  let DecoderNamespace = "VE";
  field bits<64> SoftFail = 0;
}

class RM<bits<8>opVal, dag outs, dag ins, string asmstr, list<dag> pattern=[]>
   : InstVE<outs, ins, asmstr, pattern> {
  bits<1>  cx = 0;
  bits<7>  sx;
  bits<1>  cy = 0;
  bits<7>  sy;
  bits<1>  cz = 0;
  bits<7>  sz;
  bits<32> imm32 = 0;
  let op = opVal;
  let Inst{15} = cx;
  let Inst{14-8} = sx;
  let Inst{23} = cy;
  let Inst{22-16} = sy;
  let Inst{31} = cz;
  let Inst{30-24} = sz;
  let Inst{63-32}  = imm32;
}

class RR<bits<8>opVal, dag outs, dag ins, string asmstr, list<dag> pattern=[]>
   : RM<opVal, outs, ins, asmstr, pattern> {
  bits<1> cw = 0;
  bits<1> cw2 = 0;
  bits<4> cfw = 0;
  let imm32{0-23} = 0;
  let imm32{24} = cw;
  let imm32{25} = cw2;
  let imm32{26-27} = 0;
  let imm32{28-31} = cfw;
}

class CF<bits<8>opVal, dag outs, dag ins, string asmstr, list<dag> pattern=[]>
   : RM<opVal, outs, ins, asmstr, pattern> {
  bits<1>  cx2;
  bits<2>  bpf;
  bits<4>  cf;
  let cx = 0;
  let sx{6} = cx2;
  let sx{5-4} = bpf;
  let sx{3-0} = cf;
}

// Pseudo instructions.
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern=[]>
   : InstVE<outs, ins, asmstr, pattern> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}
