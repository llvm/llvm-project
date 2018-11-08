//===-- MSP430InstrFormats.td - MSP430 Instruction Formats -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Describe MSP430 instructions format here
//

// Format specifies the encoding used by the instruction.  This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<2> val> {
  bits<2> Value = val;
}

def PseudoFrm   : Format<0>;
def SingleOpFrm : Format<1>;
def DoubleOpFrm : Format<2>;
def CondJumpFrm : Format<3>;

class SourceMode<bits<2> val> {
  bits<2> Value = val;
}

def SrcReg      : SourceMode<0>;
def SrcMem      : SourceMode<1>;
def SrcIndReg   : SourceMode<2>;
def SrcPostInc  : SourceMode<3>;
def SrcImm      : SourceMode<3>;

class DestMode<bit val> {
  bit Value = val;
}

def DstReg      : DestMode<0>;
def DstMem      : DestMode<1>;

class SizeVal<bits<3> val> {
  bits<3> Value = val;
}

def SizeUnknown : SizeVal<0>; // Unknown / unset size
def SizeSpecial : SizeVal<1>; // Special instruction, e.g. pseudo
def Size2Bytes  : SizeVal<2>;
def Size4Bytes  : SizeVal<3>;
def Size6Bytes  : SizeVal<4>;

// Generic MSP430 Format
class MSP430Inst<dag outs, dag ins, SizeVal sz, Format f,
                 string asmstr> : Instruction {
  field bits<16> Inst;

  let Namespace = "MSP430";

  dag OutOperandList = outs;
  dag InOperandList  = ins;

  Format Form = f;
  SizeVal Sz = sz;

  // Define how we want to layout our TargetSpecific information field... This
  // should be kept up-to-date with the fields in the MSP430InstrInfo.h file.
  let TSFlags{1-0} = Form.Value;
  let TSFlags{4-2} = Sz.Value;

  let AsmString   = asmstr;
}

// FIXME: Create different classes for different addressing modes.

// MSP430 Double Operand (Format I) Instructions
class IForm<bits<4> opcode, DestMode dest, bit bw, SourceMode src, SizeVal sz,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, sz, DoubleOpFrm, asmstr> {
  let Pattern = pattern;

  DestMode ad = dest;
  SourceMode as = src;
  
  let Inst{12-15} = opcode;
  let Inst{7}     = ad.Value;
  let Inst{6}     = bw;
  let Inst{4-5}   = as.Value;
}

// 8 bit IForm instructions
class IForm8<bits<4> opcode, DestMode dest, SourceMode src, SizeVal sz,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm<opcode, dest, 1, src, sz, outs, ins, asmstr, pattern>;

class I8rr<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcReg, Size2Bytes, outs, ins, asmstr, pattern>;

class I8ri<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcImm, Size4Bytes, outs, ins, asmstr, pattern>;

class I8rm<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstReg, SrcMem, Size4Bytes, outs, ins, asmstr, pattern>;

class I8mr<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcReg, Size4Bytes, outs, ins, asmstr, pattern>;

class I8mi<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcImm, Size6Bytes, outs, ins, asmstr, pattern>;

class I8mm<bits<4> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm8<opcode, DstMem, SrcMem, Size6Bytes, outs, ins, asmstr, pattern>;

// 16 bit IForm instructions
class IForm16<bits<4> opcode, DestMode dest, SourceMode src, SizeVal sz,
              dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm<opcode, dest, 0, src, sz, outs, ins, asmstr, pattern>;

class I16rr<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcReg, Size2Bytes, outs, ins, asmstr, pattern>;

class I16ri<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcImm, Size4Bytes, outs, ins, asmstr, pattern>;

class I16rm<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstReg, SrcMem, Size4Bytes, outs, ins, asmstr, pattern>;

class I16mr<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcReg, Size4Bytes, outs, ins, asmstr, pattern>;

class I16mi<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcImm, Size6Bytes, outs, ins, asmstr, pattern>;

class I16mm<bits<4> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IForm16<opcode, DstMem, SrcMem, Size6Bytes, outs, ins, asmstr, pattern>;

// MSP430 Single Operand (Format II) Instructions
class IIForm<bits<9> opcode, bit bw, SourceMode src, SizeVal sz,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, sz, SingleOpFrm, asmstr> {
  let Pattern = pattern;
  
  SourceMode as = src;

  let Inst{7-15} = opcode;
  let Inst{6}    = bw;
  let Inst{4-5}  = as.Value;
}

// 8 bit IIForm instructions
class IIForm8<bits<9> opcode, SourceMode src, SizeVal sz,
              dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm<opcode, 1, src, sz, outs, ins, asmstr, pattern>;

class II8r<bits<9> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm8<opcode, SrcReg, Size2Bytes, outs, ins, asmstr, pattern>;

class II8m<bits<9> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm8<opcode, SrcMem, Size4Bytes, outs, ins, asmstr, pattern>;

class II8i<bits<9> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm8<opcode, SrcImm, Size4Bytes, outs, ins, asmstr, pattern>;

// 16 bit IIForm instructions
class IIForm16<bits<9> opcode, SourceMode src, SizeVal sz,
               dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm<opcode, 0, src, sz, outs, ins, asmstr, pattern>;

class II16r<bits<9> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm16<opcode, SrcReg, Size2Bytes, outs, ins, asmstr, pattern>;

class II16m<bits<9> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm16<opcode, SrcMem, Size4Bytes, outs, ins, asmstr, pattern>;

class II16i<bits<9> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
  : IIForm16<opcode, SrcImm, Size4Bytes, outs, ins, asmstr, pattern>;

// MSP430 Conditional Jumps Instructions
class CJForm<bits<3> opcode, bits<3> cond,
             dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, Size2Bytes, CondJumpFrm, asmstr> {
  let Pattern = pattern;
  
  let Inst{13-15} = opcode;
  let Inst{10-12} = cond;
}

// Pseudo instructions
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
  : MSP430Inst<outs, ins, SizeSpecial, PseudoFrm, asmstr> {
  let Pattern = pattern;
  let Inst{15-0} = 0;
}
