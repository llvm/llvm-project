//===-- MipsInstrFormats.td - Mips Instruction Formats -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe MIPS instructions format
//
//  CPU INSTRUCTION FORMATS
//
//  opcode  - operation code.
//  rs      - src reg.
//  rt      - dst reg (on a 2 regs instr) or src reg (on a 3 reg instr).
//  rd      - dst reg, only used on 3 regs instr.
//  shamt   - only used on shift instructions, contains the shift amount.
//  funct   - combined with opcode field give us an operation code.
//
//===----------------------------------------------------------------------===//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<4> val> {
  bits<4> Value = val;
}

def Pseudo    : Format<0>;
def FrmR      : Format<1>;
def FrmI      : Format<2>;
def FrmJ      : Format<3>;
def FrmFR     : Format<4>;
def FrmFI     : Format<5>;
def FrmOther  : Format<6>; // Instruction w/ a custom format

// Generic Mips Format
class MipsInst<dag outs, dag ins, string asmstr, list<dag> pattern,
               InstrItinClass itin, Format f>: Instruction
{
  field bits<32> Inst;
  Format Form = f;

  let Namespace = "Mips";

  let Size = 4;

  bits<6> Opcode = 0;

  // Top 6 bits are the 'opcode' field
  let Inst{31-26} = Opcode;

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;
  let Itinerary   = itin;

  //
  // Attributes specific to Mips instructions...
  //
  bits<4> FormBits = Form.Value;

  // TSFlags layout should be kept in sync with MipsInstrInfo.h.
  let TSFlags{3-0}   = FormBits;

  let DecoderNamespace = "Mips";

  field bits<32> SoftFail = 0;
}

// Mips32/64 Instruction Format
class InstSE<dag outs, dag ins, string asmstr, list<dag> pattern,
             InstrItinClass itin, Format f>:
  MipsInst<outs, ins, asmstr, pattern, itin, f> {
  let Predicates = [HasStandardEncoding];
}

// Mips Pseudo Instructions Format
class MipsPseudo<dag outs, dag ins, string asmstr, list<dag> pattern>:
  MipsInst<outs, ins, asmstr, pattern, IIPseudo, Pseudo> {
  let isCodeGenOnly = 1;
  let isPseudo = 1;
}

// Mips32/64 Pseudo Instruction Format
class PseudoSE<dag outs, dag ins, string asmstr, list<dag> pattern>:
  MipsPseudo<outs, ins, asmstr, pattern> {
  let Predicates = [HasStandardEncoding];
}

// Pseudo-instructions for alternate assembly syntax (never used by codegen).
// These are aliases that require C++ handling to convert to the target
// instruction, while InstAliases can be handled directly by tblgen.
class MipsAsmPseudoInst<dag outs, dag ins, string asmstr>:
  MipsInst<outs, ins, asmstr, [], IIPseudo, Pseudo> {
  let isPseudo = 1;
  let Pattern = [];
}
//===----------------------------------------------------------------------===//
// Format R instruction class in Mips : <|opcode|rs|rt|rd|shamt|funct|>
//===----------------------------------------------------------------------===//

class FR<bits<6> op, bits<6> _funct, dag outs, dag ins, string asmstr,
         list<dag> pattern, InstrItinClass itin>:
  InstSE<outs, ins, asmstr, pattern, itin, FrmR>
{
  bits<5>  rd;
  bits<5>  rs;
  bits<5>  rt;
  bits<5>  shamt;
  bits<6>  funct;

  let Opcode = op;
  let funct  = _funct;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-6}  = shamt;
  let Inst{5-0}   = funct;
}

//===----------------------------------------------------------------------===//
// Format I instruction class in Mips : <|opcode|rs|rt|immediate|>
//===----------------------------------------------------------------------===//

class FI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: InstSE<outs, ins, asmstr, pattern, itin, FrmI>
{
  bits<5>  rt;
  bits<5>  rs;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

class BranchBase<bits<6> op, dag outs, dag ins, string asmstr,
                  list<dag> pattern, InstrItinClass itin>:
  InstSE<outs, ins, asmstr, pattern, itin, FrmI>
{
  bits<5>  rs;
  bits<5>  rt;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = rs;
  let Inst{20-16} = rt;
  let Inst{15-0}  = imm16;
}

//===----------------------------------------------------------------------===//
// Format J instruction class in Mips : <|opcode|address|>
//===----------------------------------------------------------------------===//

class FJ<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern,
         InstrItinClass itin>: InstSE<outs, ins, asmstr, pattern, itin, FrmJ>
{
  bits<26> addr;

  let Opcode = op;

  let Inst{25-0} = addr;
}

 //===----------------------------------------------------------------------===//
// MFC instruction class in Mips : <|op|mf|rt|rd|0000000|sel|>
//===----------------------------------------------------------------------===//
class MFC3OP<bits<6> op, bits<5> _mfmt, dag outs, dag ins, string asmstr>:
  InstSE<outs, ins, asmstr, [], NoItinerary, FrmFR>
{
  bits<5> mfmt;
  bits<5> rt;
  bits<5> rd;
  bits<3> sel;

  let Opcode = op;
  let mfmt = _mfmt;

  let Inst{25-21} = mfmt;
  let Inst{20-16} = rt;
  let Inst{15-11} = rd;
  let Inst{10-3}  = 0;
  let Inst{2-0}   = sel;
}

//===----------------------------------------------------------------------===//
//
//  FLOATING POINT INSTRUCTION FORMATS
//
//  opcode  - operation code.
//  fs      - src reg.
//  ft      - dst reg (on a 2 regs instr) or src reg (on a 3 reg instr).
//  fd      - dst reg, only used on 3 regs instr.
//  fmt     - double or single precision.
//  funct   - combined with opcode field give us an operation code.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Format FR instruction class in Mips : <|opcode|fmt|ft|fs|fd|funct|>
//===----------------------------------------------------------------------===//

class FFR<bits<6> op, bits<6> _funct, bits<5> _fmt, dag outs, dag ins,
          string asmstr, list<dag> pattern> :
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmFR>
{
  bits<5>  fd;
  bits<5>  fs;
  bits<5>  ft;
  bits<5>  fmt;
  bits<6>  funct;

  let Opcode = op;
  let funct  = _funct;
  let fmt    = _fmt;

  let Inst{25-21} = fmt;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = funct;
}

//===----------------------------------------------------------------------===//
// Format FI instruction class in Mips : <|opcode|base|ft|immediate|>
//===----------------------------------------------------------------------===//

class FFI<bits<6> op, dag outs, dag ins, string asmstr, list<dag> pattern>:
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmFI>
{
  bits<5>  ft;
  bits<5>  base;
  bits<16> imm16;

  let Opcode = op;

  let Inst{25-21} = base;
  let Inst{20-16} = ft;
  let Inst{15-0}  = imm16;
}

//===----------------------------------------------------------------------===//
// Compare instruction class in Mips : <|010001|fmt|ft|fs|0000011|condcode|>
//===----------------------------------------------------------------------===//

class FCC<bits<5> _fmt, dag outs, dag ins, string asmstr, list<dag> pattern> :
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmOther>
{
  bits<5>  fs;
  bits<5>  ft;
  bits<4>  cc;
  bits<5>  fmt;

  let Opcode = 0x11;
  let fmt    = _fmt;

  let Inst{25-21} = fmt;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6}  = 0;
  let Inst{5-4}   = 0b11;
  let Inst{3-0}   = cc;
}


class FCMOV<bits<1> _tf, dag outs, dag ins, string asmstr,
            list<dag> pattern> :
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmOther>
{
  bits<5>  rd;
  bits<5>  rs;
  bits<3>  cc;
  bits<1>  tf;

  let Opcode = 0;
  let tf = _tf;

  let Inst{25-21} = rs;
  let Inst{20-18} = cc;
  let Inst{17} = 0;
  let Inst{16} = tf;
  let Inst{15-11} = rd;
  let Inst{10-6}  = 0;
  let Inst{5-0}   = 1;
}

class FFCMOV<bits<5> _fmt, bits<1> _tf, dag outs, dag ins, string asmstr,
             list<dag> pattern> :
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmOther>
{
  bits<5>  fd;
  bits<5>  fs;
  bits<3>  cc;
  bits<5>  fmt;
  bits<1>  tf;

  let Opcode = 17;
  let fmt = _fmt;
  let tf = _tf;

  let Inst{25-21} = fmt;
  let Inst{20-18} = cc;
  let Inst{17} = 0;
  let Inst{16} = tf;
  let Inst{15-11} = fs;
  let Inst{10-6}  = fd;
  let Inst{5-0}   = 17;
}

// FP unary instructions without patterns.
class FFR1<bits<6> funct, bits<5> fmt, string opstr, string fmtstr,
           RegisterClass DstRC, RegisterClass SrcRC> :
  FFR<0x11, funct, fmt, (outs DstRC:$fd), (ins SrcRC:$fs),
      !strconcat(opstr, ".", fmtstr, "\t$fd, $fs"), []> {
  let ft = 0;
}

// FP unary instructions with patterns.
class FFR1P<bits<6> funct, bits<5> fmt, string opstr, string fmtstr,
            RegisterClass DstRC, RegisterClass SrcRC, SDNode OpNode> :
  FFR<0x11, funct, fmt, (outs DstRC:$fd), (ins SrcRC:$fs),
      !strconcat(opstr, ".", fmtstr, "\t$fd, $fs"),
      [(set DstRC:$fd, (OpNode SrcRC:$fs))]> {
  let ft = 0;
}

class FFR2P<bits<6> funct, bits<5> fmt, string opstr,
            string fmtstr, RegisterClass RC, SDNode OpNode> :
  FFR<0x11, funct, fmt, (outs RC:$fd), (ins RC:$fs, RC:$ft),
      !strconcat(opstr, ".", fmtstr, "\t$fd, $fs, $ft"),
      [(set RC:$fd, (OpNode RC:$fs, RC:$ft))]>;

// Floating point madd/msub/nmadd/nmsub.
class FFMADDSUB<bits<3> funct, bits<3> fmt, dag outs, dag ins, string asmstr,
                list<dag> pattern>
  : InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmOther> {
  bits<5> fd;
  bits<5> fr;
  bits<5> fs;
  bits<5> ft;

  let Opcode = 0x13;
  let Inst{25-21} = fr;
  let Inst{20-16} = ft;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-3} = funct;
  let Inst{2-0} = fmt;
}

// FP indexed load/store instructions.
class FFMemIdx<bits<6> funct, dag outs, dag ins, string asmstr,
               list<dag> pattern> :
  InstSE<outs, ins, asmstr, pattern, NoItinerary, FrmOther>
{
  bits<5>  base;
  bits<5>  index;
  bits<5>  fs;
  bits<5>  fd;

  let Opcode = 0x13;

  let Inst{25-21} = base;
  let Inst{20-16} = index;
  let Inst{15-11} = fs;
  let Inst{10-6} = fd;
  let Inst{5-0} = funct;
}
