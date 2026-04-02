//===-- ConnexInstrFormats.td - Connex Instruction Formats -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Inspired from lib/Target/BPF/BPFInstrFormats.td (from Oct 2025)
class ConnexOpClass<bits<3> val> {
  bits<3> Value = val;
}

def Connex_LD    : ConnexOpClass<0x0>;
def Connex_LDX   : ConnexOpClass<0x1>;
def Connex_ST    : ConnexOpClass<0x2>;
def Connex_STX   : ConnexOpClass<0x3>;
def Connex_ALU   : ConnexOpClass<0x4>;
def Connex_JMP   : ConnexOpClass<0x5>;
def Connex_JMP32 : ConnexOpClass<0x6>;
def Connex_ALU64 : ConnexOpClass<0x7>;



// Inspired from lib/Target/BPF/BPFInstrFormats.td (from Oct 2025)
class ConnexArithOp<bits<4> val> {
  bits<4> Value = val;
}

def Connex_ADD  : ConnexArithOp<0x0>;
def Connex_SUB  : ConnexArithOp<0x1>;
def Connex_MUL  : ConnexArithOp<0x2>;
def Connex_DIV  : ConnexArithOp<0x3>;
def Connex_OR   : ConnexArithOp<0x4>;
def Connex_AND  : ConnexArithOp<0x5>;
def Connex_LSH  : ConnexArithOp<0x6>;
def Connex_RSH  : ConnexArithOp<0x7>;
def Connex_NEG  : ConnexArithOp<0x8>;
def Connex_MOD  : ConnexArithOp<0x9>;
def Connex_XOR  : ConnexArithOp<0xa>;
def Connex_MOV  : ConnexArithOp<0xb>;
def Connex_ARSH : ConnexArithOp<0xc>;
def Connex_END  : ConnexArithOp<0xd>;


// Inspired from lib/Target/BPF/BPFInstrFormats.td (from Oct 2025)
class ConnexWidthModifer<bits<2> val> {
  bits<2> Value = val;
}

def Connex_W  : ConnexWidthModifer<0x0>;
def Connex_H  : ConnexWidthModifer<0x1>;
def Connex_B  : ConnexWidthModifer<0x2>;
def Connex_DW : ConnexWidthModifer<0x3>;



// Inspired from lib/Target/BPF/BPFInstrFormats.td (from Oct 2025)
class ConnexModeModifer<bits<3> val> {
  bits<3> Value = val;
}

def Connex_IMM  : ConnexModeModifer<0x0>;
def Connex_ABS  : ConnexModeModifer<0x1>;
def Connex_IND  : ConnexModeModifer<0x2>;
def Connex_MEM  : ConnexModeModifer<0x3>;
def Connex_MEMSX  : ConnexModeModifer<0x4>;
def Connex_ATOMIC : ConnexModeModifer<0x6>;





class InstConnex<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
  let Size = 8;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  // bits<3> ConnexClass;
  // let Inst{58-56} = ConnexClass;
  // Inspired from lib/Target/BPF/BPFInstrFormats.td (from Oct 2025)
  ConnexOpClass ConnexClass;
  let Inst{58-56} = ConnexClass.Value;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}




class InstConnex2<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Instruction {
  field bits<64> Inst;
  field bits<64> SoftFail = 0;
  let Size = 8;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  // bits<3> ConnexClass;
  // let Inst{58-56} = ConnexClass;

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}


// Pseudo instructions
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstConnex<outs, ins, asmstr, pattern> {
  let Inst{63-0} = 0;
  let isPseudo = 1;
}



// Inspired from BPFInstrInfo.td (from Oct 2025)
//For load and store instructions the 8-bit 'code' field is divided as:
//
//  +--------+--------+-------------------+
//  | 3 bits | 2 bits |   3 bits          |
//  |  mode  |  size  | instruction class |
//  +--------+--------+-------------------+
//  (MSB)                             (LSB)
class TYPE_LD_ST<bits<3> mode, bits<2> size,
                 dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstConnex<outs, ins, asmstr, pattern> {

  let Inst{63-61} = mode;
  let Inst{60-59} = size;
}




// Inspired from book "Getting started with LLVM Core Libraries", 2014, page 141

// Inspired from SparcInstrFormats.td:

// Instruction with 16 bits immediate operand
class Connex_IMM16_FMT<bits<6> opcode>: Instruction {
  field bits<32> Inst;

  bits<16> imm;
  bits<5> wl; // Connex-S left
  bits<5> wd; // Connex-S dest

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-26} = opcode;
  let Inst{25-10} = imm;
  let Inst{9-5} = wl;
  let Inst{4-0} = wd;
}

class Connex_IMM16_FMT2<bits<6> opcode>: Instruction {
  field bits<32> Inst;

  bits<16> imm;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-26} = opcode;
  let Inst{25-10} = imm;
}

class Connex_IMM16_FMT3<bits<6> opcode>: Instruction {
  field bits<32> Inst;

  bits<16> imm;
  bits<5> wd; // Connex-S dest

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-26} = opcode;
  let Inst{25-10} = imm;
  let Inst{4-0} = wd;
}

class Connex_IMM16_SYM_FMT<bits<6> opcode>: Instruction {
  field bits<32> Inst;

  // We comment it - otherwise we get error in TableGen
  //   <<CodeGenInstruction.h:196: Assertion `i < OperandList.size() &&
  //   "Invalid flat operand #"' failed.>>: bits<16> imm;
  bits<5> wd; // dest

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-26} = opcode;
  let Inst{4-0} = wd;
}

// Non-immediate instruction
class Connex_NI_FMT<bits<9> opcode>: Instruction {
  field bits<32> Inst;

  bits<8> reserved;
  bits<5> wd;
  bits<5> wl;
  bits<5> wr;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-23} = opcode;
  let Inst{22-15} = reserved;
  let Inst{14-10} = wl;
  let Inst{9-5} = wr;
  let Inst{4-0} = wd;
}

// Non-immediate instruction for ISHLV(_SPECIAL), ISHRV(_SPECIAL)
class Connex_NI_FMT_ISHV<bits<9> opcode>: Instruction {
  field bits<32> Inst;

  // wr is Right, wl is Left, wd is Dest - see ConnexISA.pdf
  // bits<8> reserved;
  bits<5> wr;
  bits<5> wl;
  bits<5> wd;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{31-23} = opcode;
  // let Inst{22-15} = reserved;
  let Inst{14-10} = wr;
  let Inst{9-5} = wl;
  let Inst{4-0} = wd;
}

class NonImmediateInstruction<bits<9> opcode, dag outs, dag ins,
                              string asmstr, list<dag> pattern>
       : Instruction {
  field bits<32> Inst;
  let Inst{31-23} = opcode;

  // We require to put the Namespace field, otherwise we receive:
  //   "error:No instructions defined!"
  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

class ImmediateInstruction<bits<6> opcode, dag outs, dag ins, string asmstr,
                           list<dag> pattern>
      : Instruction {
  field bits<32> Inst;
  let Inst{31-26} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = asmstr;
  let Pattern = pattern;
}


// Inspired from lib/Target/Mips/Mips.td
// The overall idea of the PredicateControl class is to chop the Predicates list
// into subsets that are usually overridden independently. This allows
// subclasses to partially override the predicates of their superclasses without
// having to re-add all the existing predicates.
class PredicateControl {
  // Predicates for the encoding scheme in use such as HasStdEnc
  list<Predicate> EncodingPredicates = [];
  // Predicates for the GPR size such as IsGP64bit
  list<Predicate> GPRPredicates = [];
  // Predicates for the FGR size and layout such as IsFP64bit
  list<Predicate> FGRPredicates = [];
  // Predicates for the instruction group membership such as ISA's and ASE's
  list<Predicate> InsnPredicates = [];
  // Predicate for marking the instruction as usable in hard-float mode only.
  list<Predicate> HardFloatPredicate = [];
  // Predicates for anything else
  list<Predicate> AdditionalPredicates = [];
  list<Predicate> Predicates = !listconcat(EncodingPredicates,
                                           GPRPredicates,
                                           FGRPredicates,
                                           InsnPredicates,
                                           HardFloatPredicate,
                                           AdditionalPredicates);
}

// Inspired from lib/Target/Mips/MipsInstrFormats.td
// Format specifies the encoding used by the instruction. This is part of the
// ad-hoc solution used to emit machine instruction encodings by our machine
// code emitter.
class Format<bits<4> val> {
  bits<4> Value = val;
}

def FrmOther  : Format<6>; // Instruction w/ a custom format

class ConnexMipsInst<dag outs, dag ins, string asmstr, list<dag> pattern,
                     InstrItinClass itin, Format f>: Instruction
{
  field bits<32> Inst;
  Format Form = f;

  let Namespace = "Connex";

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

  let DecoderNamespace = "Connex";

  field bits<32> SoftFail = 0;
}

// Inspired from lib/Target/Mips/MipsMSAInstrFormats.td:
class MSAInst : ConnexMipsInst<(outs), (ins), "", [], NoItinerary, FrmOther>,
                PredicateControl {
}

class MSA_1R_FMT<bits<9> opcode>: MSAInst {
  bits<5> wl;

  let Inst{31-23} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{9-5} = wl;
}

class MSA_1R_FMT_dest<bits<9> opcode>: MSAInst {
  bits<5> wd;

  let Inst{31-23} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{9-5} = wd;
}

class MSA_1R_FMT_dest_imm<bits<6> opcode>: MSAInst {
  bits<5> wd; // Connex-S dest
  bits<16> imm;

  let Inst{31-26} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{25-10} = imm;
  let Inst{4-0} = wd;
}

class MSA_1R_FMT_left_imm<bits<6> opcode>: MSAInst {
  bits<5> wl;   // Connex-S left
  bits<16> imm; // Connex-S immediate

  let Inst{31-26} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{25-10} = imm;
  let Inst{9-5} = wl;
}

class MSA_2R_FMT<bits<9> opcode>: MSAInst {
  bits<5> wl; // Connex-S left
  bits<5> wd; // Connex-S dest

  let Inst{31-23} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{9-5} = wl;
  let Inst{4-0} = wd;
}

class MSA_2R_FMT2<bits<9> opcode>: MSAInst {
  bits<5> wr; // Connex-S right
  bits<5> wl; // Connex-S left

  let Inst{31-23} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{14-10} = wr;
  let Inst{9-5} = wl;
}

class MSA_RR_FMT<bits<9> opcode>: MSAInst {
  bits<5> wr; // Connex-S right
  bits<5> wl; // Connex-S left

  let Inst{31-23} = opcode;

  let Namespace = "Connex";
  let DecoderNamespace = "Connex";

  let Inst{14-10} = wr;
  let Inst{9-5} = wl;
}

class MSA_3R_FMT<bits<9> opcode>: MSAInst {
  bits<5> wr; // Connex-S right
  bits<5> wl; // Connex-S left
  bits<5> wd; // Connex-S dest

  let Inst{31-23} = opcode;

  let Namespace = "Connex";

  let DecoderNamespace = "Connex";

  let Inst{14-10} = wr;
  let Inst{9-5} = wl;
  let Inst{4-0} = wd;
}

class MSA_3R_FMT2<bits<9> opcode>: MSAInst {
  bits<5> wr;    // Connex-S right
  // bits<5> ws; // Connex-S left
  bits<5> wd;    // Connex-S dest

  let Inst{31-23} = opcode;

  let Namespace = "Connex";

  let DecoderNamespace = "Connex";

  let Inst{14-10} = wr;
  // let Inst{9-5} = ws;
  let Inst{4-0} = wd;
}

class MSA_LDIX_LDSH_MULT_H_DESC_BASE<string instr_asm,
                                     RegisterOperand ROWD,
                                     InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins);
  string AsmString = !strconcat(!strconcat("$wd = ", instr_asm),
                                " ; // MSA_LDIX_LDSH_MULT");
  // Note: LDI is matched using custom matching code in MipsSEISelDAGToDAG.cpp
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
    // We need to put this since we don't specify a DAG pattern in Pattern
  InstrItinClass Itinerary = itin;
}

class MSA_RR_PREFIX_DESC_BASE<string instr_asm,
                              RegisterOperand ROWS,
                              InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins ROWS:$wr, ROWS:$wl);
  string AsmString = !strconcat(instr_asm,
                              " ( $wr, $wl ); // MSA_RR generic instruction");
  list<dag> Pattern = [];

  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}

class MSA_RR_INFIX_DESC_BASE<string instr_asm,
                             RegisterOperand ROWS,
                             InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins ROWS:$wr, ROWS:$wl);
  string AsmString = !strconcat(
                       !strconcat("$wr ", instr_asm),
                       " $wl ; // MSA_RR_INFIX instruction");
  list<dag> Pattern = [];

  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
}

///////////////////////////////////////////////////////////////////////////////
////////////////////BEGIN (i)read/(i)write SPECS///////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// SDTMaskedGather, SDTMaskedScatter, masked_gather, masked_scatter
//   are found now in include/llvm/Target/TargetSelectionDAG.td.
//   But SDTMaskedGather, SDTMaskedScatter taken now 4 parameters.
def SDTMaskedGather2: SDTypeProfile<2, 3, [       // masked gather
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<1, 3>,
  SDTCisPtrTy<4>, SDTCVecEltisVT<1, i1>, SDTCisSameNumEltsAs<0, 1>
]>;

/*
// So: 3 input operands, 1 result.
//   Params are: mask, value, index; results are: ptr
//   Params are 0, 1, 2 and result is 3.
//   Operands 0 and 1 have vector type; also with same number of elements.
//   Operands 0 and 2 have identical types.
//       --> Opnd 2 is vector of i16 elements
//   Operand 3 (result 0) has pointer type.
//   Operand 0 is vector type with element type of i1.
def SDTMaskedScatter: SDTypeProfile<1, 3, [       // masked scatter
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<0, i1>, SDTCisPtrTy<3>
]>;

def masked_scatter : SDNode<"ISD::MSCATTER",  SDTMaskedScatter,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
*/

def SDTMaskedScatter2: SDTypeProfile<1, 3, [       // masked scatter
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<0, i1>, SDTCisPtrTy<3>
]>;

def masked_gather2  : SDNode<"ISD::MGATHER",  SDTMaskedGather2,
                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def masked_scatter2 : SDNode<"ISD::MSCATTER",  SDTMaskedScatter2,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// We inspire from MSP430:
def addr : ComplexPattern<iPTR, 2, "SelectAddr", [], []>;

def simm10 : Operand<i32>;
def simm16      : Operand<i32> {
  let DecoderMethod= "DecodeSimm16";
}

def MipsMemAsmOperand : AsmOperandClass {
  let Name = "Mem";
  let ParserMethod = "parseMemOperand";
}

class mem_generic : Operand<iPTR> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ptr_rc, simm16);
  let EncoderMethod = "getMemEncoding";
  let ParserMatchClass = MipsMemAsmOperand;
  let OperandType = "OPERAND_MEMORY";
}

// MSA specific address operand
def mem_msa : mem_generic {
  let MIOperandInfo = (ops ptr_rc, simm10);
  let EncoderMethod = "getMSAMemEncoding";
}

def mem_msa2 : mem_generic {
  let MIOperandInfo = (ops VectorHOpnd);
  let EncoderMethod = "getMSAMemEncoding";
}

// Address operands
def MEMtest : Operand<i16> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops i16imm);
}

// From [LLVM]/llvm/lib/Target/Mips/MipsMSAInstrInfo.td
def uimm4_ptr : Operand<iPTR> {
  let PrintMethod = "printUnsignedImm";
}

def immAlex : ComplexPattern<iPTR, 1, "SelectAddr", []>;
def immLeafAlex : ImmLeaf<i64, [{return 1;}]>; // TODO: make sure we can retrn 1
def uimm8 : Operand<i64> {
  let PrintMethod = "printUnsignedImm8";
}

// This is inspired from Mips MSA LD_DESC_BASE and got changed to have
//    immediate address operand.
//   Note that MSA_I10_LDI_DESC_BASE loads in a vector register an
//    immediate vector value.
class LD_DESC_BASE<
                   SDPatternOperator OpNode,
                   ValueType TyNode, RegisterOperand ROWD,
                   Operand MemOpnd = uimm4_ptr,
                   ImmLeaf Addr = immLeafAlex,
                   InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins MemOpnd:$imm);
  string AsmString = "$wd = LS[$imm]; // IREAD (or Mips MSA's LD)";

  list<dag> Pattern = [(set ROWD:$wd, (TyNode (OpNode Addr:$imm)))];
  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}

class ST_DESC_BASE<
                   SDPatternOperator OpNode,
                   ValueType TyNode, RegisterOperand ROWS,
                   Operand MemOpnd = uimm4_ptr, ImmLeaf Addr = immLeafAlex,
                   InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins ROWS:$wl, MemOpnd:$imm);
  string AsmString = "LS[$imm] = $wl ; // IWRITE (or Mips MSA's ST)";
  list<dag> Pattern = [(OpNode (TyNode ROWS:$wl), Addr:$imm)];
  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}

/*
In a good sense LD_INDIRECT_DESC_BASE is similar to
    the gather of X86 AVX - remember I can implement gather with Connex's
    Read, by loading (the rather small array in each column of LS
    with LS[i] = Rsrc = array[i]).

LD_INDIRECT_DESC_BASE is similar to:
    - VLOAD
The Mips equivalent is LD_DESC_BASE which uses stack -
    def addrimm10 : ComplexPattern<iPTR, 2, "selectIntAddrMSA", [frameindex]>;
  The instructions selected look like:
    ld.w    $w0, 32($fp)

I need to instr-select:
    Rdst = LS[Rsrc]

  list<dag> Pattern = [(set ROWD:$wd, (TyNode (OpNode Addr:$addrsrc)))];

// From http://llvm.org/docs/doxygen/html/SelectionDAGNodes_8h_source.html:
  // In the both nodes address is Op1, mask is Op2:
  // MaskedGatherSDNode  (Chain, src0, mask, base, index),
                              // src0 is a passthru value
  // MaskedScatterSDNode (Chain, value, mask, base, index)
  // Mask is a vector of i1 elements
  const SDValue &getBasePtr() const { return getOperand(3); }
  const SDValue &getIndex()   const { return getOperand(4); }
  const SDValue &getMask()    const { return getOperand(2); }
  const SDValue &getValue()   const { return getOperand(1); }
                                             // This is pass-thru
*/

/*
// From include/llvm/Target/TargetSelectionDAG.td

// SDTypeProfile - This profile describes the type requirements of a Selection
// DAG node.
class SDTypeProfile<int numresults, int numoperands,
                    list<SDTypeConstraint> constraints> {
  int NumResults = numresults;
  int NumOperands = numoperands;
  list<SDTypeConstraint> Constraints = constraints;
}

// So: 3 input operands, 2 results.
//   Params are: passthru, mask, index; results are: vector of i1, the
//                vector with the values loaded by the gather instruction (ptr)
//   Params are 0, 1, 2 and results are 3, 4.
//   Operands 0 and 1 have vector type; also with same number of elements.
//   Operands 0 and 2 have identical types.
//   Operands 1 and 3 have identical types.
//       --> Opnd 3 (result 0?) is i1 vector
//   Operand 4 (result 1?) has pointer type.
//   Operand 1 is vector type with element type of i1.
//  Note that ConnexTargetLowering::LowerMGATHER() treats a ~different
//     masked_gather, which is machine-independent, NOT like this one, with
//     different parameters.
def SDTMaskedGather: SDTypeProfile<2, 3, [       // masked gather
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<1, 3>,
  SDTCisPtrTy<4>, SDTCVecEltisVT<1, i1>, SDTCisSameNumEltsAs<0, 1>
]>;

def masked_gather  : SDNode<"ISD::MGATHER",  SDTMaskedGather,
                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
*/

// Note (defined in Target.td): def ptr_rc : PointerLikeRegClass<0>;
//
/* IMPORTANT: because of the SDNPMemOperand attribute of masked_gather it seems
      we need to make the index operator a memory operand.
   It also seems we need to make it a scalar operand by using iPTR and use a C++
       method that returns a vector type.
   If we try to use VectorHOpnd instead of vectoraddr in the Pattern, we get
       errors.
*/
// Gather mem operands
def ScatterGatherMemOperand : Operand<iPTR> {
  let PrintMethod = "printScatterGatherMemOperand";
  let MIOperandInfo = (ops VectorH);
}

/* 1 means selectVectorAddr takes 1 extra argument, in this case reference
  int Index which we set with N->getIndex(). Otherwise, the 3rd(? maybe 2nd now)
  parameter of masked_gather would receive the base pointer IIRC. */
let WantsParent = true in // Inspired from X86/X86InstrFragments.td
def vectoraddr : ComplexPattern<iPTR, 1, "selectVectorAddr">;

/*
Note: From
         llvm.org/docs/LangRef.html#masked-vector-gather-and-scatter-intrinsics
  ‘llvm.masked.gather.*‘ Intrinsics
     <<Overview:
       Reads scalar values from arbitrary memory locations and gathers them into
                                                                    one vector.
       The memory locations are provided in the vector of pointers ‘ptrs‘.
       The memory is accessed according to the provided mask.
       The mask holds a bit for each vector lane, and is used to prevent memory
                                               accesses to the masked-off lanes.
       The masked-off lanes in the result vector are taken from the
                                corresponding lanes of the ‘passthru‘ operand.>>
*/

// Inspired from [REPO]/llvm/lib/Target/X86/X86InstrAVX512.td
class LD_INDIRECT_MASKED_DESC_BASE<RegisterOperand ROWD,
                                   // AVOIDING_USE_OF_PASSTHRU_REGISTER:
                                     RegisterOperand ROWSP = ROWD,
                                   InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd, BoolMaskOpnd:$dstmask);
  dag InOperandList = (ins
                         // AVOIDING_USE_OF_PASSTHRU_REGISTER:
                           ROWSP:$wsp, // passthru register
                         BoolMaskOpnd:$srcmask, // mask register
                         ScatterGatherMemOperand:$wr // index register
                      );
  string AsmString = "$wd = LS[$wr]; // READ (gather)";
  list<dag> Pattern = [(set ROWD:$wd, BoolMaskOpnd:$dstmask,
                          (masked_gather2
                                    // AVOIDING_USE_OF_PASSTHRU_REGISTER:
                                    ROWSP:$wsp,
                                    BoolMaskOpnd:$srcmask,
                                    vectoraddr:$wr
                          )
                          )];
  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}

// NEW32-TODO
class LD_INDIRECT_DESC_BASE<RegisterOperand ROWD,
                            RegisterOperand ROWSI = ROWD,
                            InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs ROWD:$wd);
  dag InOperandList = (ins ROWSI:$wr); // index register
  string AsmString = "$wd = LS[$wr]; // READ 32bits index (gather)";
  list<dag> Pattern = [];

  bit hasSideEffects = 1;
          // We need to put this since we don't specify a DAG pattern in Pattern

  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}

// NEW32-TODO
class ST_INDIRECT_DESC_BASE<
                RegisterOperand ROWSV,
                RegisterOperand ROWSI = ROWSV,
                InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins
                         ROWSI:$wr, // index register
                         ROWSV:$wl
                      );
  string AsmString = "LS[$wr] = $wl ; // WRITE 32bits index (scatter)";
  list<dag> Pattern = [];

  // We need to put this since we don't specify a DAG pattern in Pattern
  bit hasSideEffects = 1;

  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}

/*
// REPEATED: So: 3 input operands, 1 result.
//   Params are: mask, value, index; results are: ptr
//   Params are 0, 1, 2 and result is 3.
//   Operands 0 and 1 have vector type; also with same number of elements.
//   Operands 0 and 2 have identical types.
//       --> Opnd 2 is vector of i16 elements
//   Operand 3 (result 0) has pointer type.
//   Operand 0 is vector type with element type of i1.
def SDTMaskedScatter: SDTypeProfile<1, 3, [       // masked scatter
  SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameNumEltsAs<0, 1>,
  SDTCVecEltisVT<0, i1>, SDTCisPtrTy<3>
]>;

def masked_scatter : SDNode<"ISD::MSCATTER",  SDTMaskedScatter,
                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
*/

class ST_INDIRECT_MASKED_DESC_BASE<
                RegisterOperand ROWV,
                InstrItinClass itin = NoItinerary> {
  dag OutOperandList = (outs BoolMaskOpnd:$dstmask);
  dag InOperandList = (ins ROWV:$wl, // value register
                           BoolMaskOpnd:$wd, // mask register
                           ScatterGatherMemOperand:$wr // index register
                      );
  string AsmString = "LS[$wr] = $wl ; // WRITE (scatter)";
  list<dag> Pattern = [(set BoolMaskOpnd:$dstmask,
                         (masked_scatter2
                          // See for pattern, multiclass avx512_scatter,
                          //      def mr for pattern:
                          ROWV:$wl, BoolMaskOpnd:$wd, vectoraddr:$wr)
                      )];
  InstrItinClass Itinerary = itin;
  string DecoderMethod = "DecodeMSA128Mem";
}
///////////////////////////////////////////////////////////////////////////////
/////////////////////END (i)read/(i)write SPECS////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

