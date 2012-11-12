//===-- SparcInstrInfo.td - Target Description for Sparc Target -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Sparc instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "SparcInstrFormats.td"

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

// HasV9 - This predicate is true when the target processor supports V9
// instructions.  Note that the machine may be running in 32-bit mode.
def HasV9   : Predicate<"Subtarget.isV9()">;

// HasNoV9 - This predicate is true when the target doesn't have V9
// instructions.  Use of this is just a hack for the isel not having proper
// costs for V8 instructions that are more expensive than their V9 ones.
def HasNoV9 : Predicate<"!Subtarget.isV9()">;

// HasVIS - This is true when the target processor has VIS extensions.
def HasVIS : Predicate<"Subtarget.isVIS()">;

// UseDeprecatedInsts - This predicate is true when the target processor is a
// V8, or when it is V9 but the V8 deprecated instructions are efficient enough
// to use when appropriate.  In either of these cases, the instruction selector
// will pick deprecated instructions.
def UseDeprecatedInsts : Predicate<"Subtarget.useDeprecatedV8Instructions()">;

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def simm11  : PatLeaf<(imm), [{ return isInt<11>(N->getSExtValue()); }]>;

def simm13  : PatLeaf<(imm), [{ return isInt<13>(N->getSExtValue()); }]>;

def LO10 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant((unsigned)N->getZExtValue() & 1023,
                                   MVT::i32);
}]>;

def HI22 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return CurDAG->getTargetConstant((unsigned)N->getZExtValue() >> 10, MVT::i32);
}]>;

def SETHIimm : PatLeaf<(imm), [{
  return (((unsigned)N->getZExtValue() >> 10) << 10) ==
         (unsigned)N->getZExtValue();
}], HI22>;

// Addressing modes.
def ADDRrr : ComplexPattern<i32, 2, "SelectADDRrr", [], []>;
def ADDRri : ComplexPattern<i32, 2, "SelectADDRri", [frameindex], []>;

// Address operands
def MEMrr : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops IntRegs, IntRegs);
}
def MEMri : Operand<i32> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops IntRegs, i32imm);
}

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;
def calltarget : Operand<i32>;

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i32>;

def SDTSPcmpfcc : 
SDTypeProfile<0, 2, [SDTCisFP<0>, SDTCisSameAs<0, 1>]>;
def SDTSPbrcc : 
SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;
def SDTSPselectcc :
SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<1, 2>, SDTCisVT<3, i32>]>;
def SDTSPFTOI :
SDTypeProfile<1, 1, [SDTCisVT<0, f32>, SDTCisFP<1>]>;
def SDTSPITOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f32>]>;

def SPcmpicc : SDNode<"SPISD::CMPICC", SDTIntBinOp, [SDNPOutGlue]>;
def SPcmpfcc : SDNode<"SPISD::CMPFCC", SDTSPcmpfcc, [SDNPOutGlue]>;
def SPbricc : SDNode<"SPISD::BRICC", SDTSPbrcc, [SDNPHasChain, SDNPInGlue]>;
def SPbrfcc : SDNode<"SPISD::BRFCC", SDTSPbrcc, [SDNPHasChain, SDNPInGlue]>;

def SPhi    : SDNode<"SPISD::Hi", SDTIntUnaryOp>;
def SPlo    : SDNode<"SPISD::Lo", SDTIntUnaryOp>;

def SPftoi  : SDNode<"SPISD::FTOI", SDTSPFTOI>;
def SPitof  : SDNode<"SPISD::ITOF", SDTSPITOF>;

def SPselecticc : SDNode<"SPISD::SELECT_ICC", SDTSPselectcc, [SDNPInGlue]>;
def SPselectfcc : SDNode<"SPISD::SELECT_FCC", SDTSPselectcc, [SDNPInGlue]>;

//  These are target-independent nodes, but have target-specific formats.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32> ]>;
def SDT_SPCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SDT_SPCall    : SDTypeProfile<0, -1, [SDTCisVT<0, i32>]>;
def call          : SDNode<"SPISD::CALL", SDT_SPCall,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                            SDNPVariadic]>;

def SDT_SPRet     : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;
def retflag       : SDNode<"SPISD::RET_FLAG", SDT_SPRet,
                           [SDNPHasChain, SDNPOptInGlue]>;

def flushw        : SDNode<"SPISD::FLUSHW", SDTNone,
                           [SDNPHasChain, SDNPSideEffect, SDNPMayStore]>;

def getPCX        : Operand<i32> {
  let PrintMethod = "printGetPCX";
}

//===----------------------------------------------------------------------===//
// SPARC Flag Conditions
//===----------------------------------------------------------------------===//

// Note that these values must be kept in sync with the CCOp::CondCode enum
// values.
class ICC_VAL<int N> : PatLeaf<(i32 N)>;
def ICC_NE  : ICC_VAL< 9>;  // Not Equal
def ICC_E   : ICC_VAL< 1>;  // Equal
def ICC_G   : ICC_VAL<10>;  // Greater
def ICC_LE  : ICC_VAL< 2>;  // Less or Equal
def ICC_GE  : ICC_VAL<11>;  // Greater or Equal
def ICC_L   : ICC_VAL< 3>;  // Less
def ICC_GU  : ICC_VAL<12>;  // Greater Unsigned
def ICC_LEU : ICC_VAL< 4>;  // Less or Equal Unsigned
def ICC_CC  : ICC_VAL<13>;  // Carry Clear/Great or Equal Unsigned
def ICC_CS  : ICC_VAL< 5>;  // Carry Set/Less Unsigned
def ICC_POS : ICC_VAL<14>;  // Positive
def ICC_NEG : ICC_VAL< 6>;  // Negative
def ICC_VC  : ICC_VAL<15>;  // Overflow Clear
def ICC_VS  : ICC_VAL< 7>;  // Overflow Set

class FCC_VAL<int N> : PatLeaf<(i32 N)>;
def FCC_U   : FCC_VAL<23>;  // Unordered
def FCC_G   : FCC_VAL<22>;  // Greater
def FCC_UG  : FCC_VAL<21>;  // Unordered or Greater
def FCC_L   : FCC_VAL<20>;  // Less
def FCC_UL  : FCC_VAL<19>;  // Unordered or Less
def FCC_LG  : FCC_VAL<18>;  // Less or Greater
def FCC_NE  : FCC_VAL<17>;  // Not Equal
def FCC_E   : FCC_VAL<25>;  // Equal
def FCC_UE  : FCC_VAL<24>;  // Unordered or Equal
def FCC_GE  : FCC_VAL<25>;  // Greater or Equal
def FCC_UGE : FCC_VAL<26>;  // Unordered or Greater or Equal
def FCC_LE  : FCC_VAL<27>;  // Less or Equal
def FCC_ULE : FCC_VAL<28>;  // Unordered or Less or Equal
def FCC_O   : FCC_VAL<29>;  // Ordered

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

/// F3_12 multiclass - Define a normal F3_1/F3_2 pattern in one shot.
multiclass F3_12<string OpcStr, bits<6> Op3Val, SDNode OpNode> {
  def rr  : F3_1<2, Op3Val, 
                 (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat(OpcStr, " $b, $c, $dst"),
                 [(set IntRegs:$dst, (OpNode IntRegs:$b, IntRegs:$c))]>;
  def ri  : F3_2<2, Op3Val,
                 (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $b, $c, $dst"),
                 [(set IntRegs:$dst, (OpNode IntRegs:$b, simm13:$c))]>;
}

/// F3_12np multiclass - Define a normal F3_1/F3_2 pattern in one shot, with no
/// pattern.
multiclass F3_12np<string OpcStr, bits<6> Op3Val> {
  def rr  : F3_1<2, Op3Val, 
                 (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                 !strconcat(OpcStr, " $b, $c, $dst"), []>;
  def ri  : F3_2<2, Op3Val,
                 (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $b, $c, $dst"), []>;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// Pseudo instructions.
class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSP<outs, ins, asmstr, pattern>;

// GETPCX for PIC
let Defs = [O7] in {
  def GETPCX : Pseudo<(outs getPCX:$getpcseq), (ins), "$getpcseq", [] >;
}

let Defs = [O6], Uses = [O6] in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i32imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP : Pseudo<(outs), (ins i32imm:$amt1, i32imm:$amt2),
                            "!ADJCALLSTACKUP $amt1",
                            [(callseq_end timm:$amt1, timm:$amt2)]>;
}

let hasSideEffects = 1, mayStore = 1 in {
  let rd = 0, rs1 = 0, rs2 = 0 in
    def FLUSHW : F3_1<0b10, 0b101011, (outs), (ins),
                      "flushw",
                      [(flushw)]>, Requires<[HasV9]>;
  let rd = 0, rs1 = 1, simm13 = 3 in
    def TA3 : F3_2<0b10, 0b111010, (outs), (ins),
                   "ta 3",
                   [(flushw)]>;
}

def UNIMP : F2_1<0b000, (outs), (ins i32imm:$val),
                "unimp $val", []>;

// FpMOVD/FpNEGD/FpABSD - These are lowered to single-precision ops by the 
// fpmover pass.
let Predicates = [HasNoV9] in {  // Only emit these in V8 mode.
  def FpMOVD : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$src),
                      "!FpMOVD $src, $dst", []>;
  def FpNEGD : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$src),
                      "!FpNEGD $src, $dst",
                      [(set DFPRegs:$dst, (fneg DFPRegs:$src))]>;
  def FpABSD : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$src),
                      "!FpABSD $src, $dst",
                      [(set DFPRegs:$dst, (fabs DFPRegs:$src))]>;
}

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded after
// instruction selection into a branch sequence.  This has to handle all
// permutations of selection between i32/f32/f64 on ICC and FCC.
  // Expanded after instruction selection.
let Uses = [ICC], usesCustomInserter = 1 in { 
  def SELECT_CC_Int_ICC
   : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_ICC PSEUDO!",
            [(set IntRegs:$dst, (SPselecticc IntRegs:$T, IntRegs:$F,
                                             imm:$Cond))]>;
  def SELECT_CC_FP_ICC
   : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_ICC PSEUDO!",
            [(set FPRegs:$dst, (SPselecticc FPRegs:$T, FPRegs:$F,
                                            imm:$Cond))]>;

  def SELECT_CC_DFP_ICC
   : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_ICC PSEUDO!",
            [(set DFPRegs:$dst, (SPselecticc DFPRegs:$T, DFPRegs:$F,
                                             imm:$Cond))]>;
}

let usesCustomInserter = 1, Uses = [FCC] in {

  def SELECT_CC_Int_FCC
   : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_FCC PSEUDO!",
            [(set IntRegs:$dst, (SPselectfcc IntRegs:$T, IntRegs:$F,
                                             imm:$Cond))]>;

  def SELECT_CC_FP_FCC
   : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_FCC PSEUDO!",
            [(set FPRegs:$dst, (SPselectfcc FPRegs:$T, FPRegs:$F,
                                            imm:$Cond))]>;
  def SELECT_CC_DFP_FCC
   : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_FCC PSEUDO!",
            [(set DFPRegs:$dst, (SPselectfcc DFPRegs:$T, DFPRegs:$F,
                                             imm:$Cond))]>;
}


// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, hasDelaySlot = 1, isBarrier = 1 in {
  let rd = O7.Num, rs1 = G0.Num in
    def RETL: F3_2<2, 0b111000, (outs), (ins i32imm:$val),
                   "jmp %o7+$val", [(retflag simm13:$val)]>;

  let rd = I7.Num, rs1 = G0.Num in
    def RET: F3_2<2, 0b111000, (outs), (ins i32imm:$val),
                  "jmp %i7+$val", []>;
}

// Section B.1 - Load Integer Instructions, p. 90
def LDSBrr : F3_1<3, 0b001001,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextloadi8 ADDRrr:$addr))]>;
def LDSBri : F3_2<3, 0b001001,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldsb [$addr], $dst",
                  [(set IntRegs:$dst, (sextloadi8 ADDRri:$addr))]>;
def LDSHrr : F3_1<3, 0b001010,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextloadi16 ADDRrr:$addr))]>;
def LDSHri : F3_2<3, 0b001010,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldsh [$addr], $dst",
                  [(set IntRegs:$dst, (sextloadi16 ADDRri:$addr))]>;
def LDUBrr : F3_1<3, 0b000001,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextloadi8 ADDRrr:$addr))]>;
def LDUBri : F3_2<3, 0b000001,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldub [$addr], $dst",
                  [(set IntRegs:$dst, (zextloadi8 ADDRri:$addr))]>;
def LDUHrr : F3_1<3, 0b000010,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextloadi16 ADDRrr:$addr))]>;
def LDUHri : F3_2<3, 0b000010,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "lduh [$addr], $dst",
                  [(set IntRegs:$dst, (zextloadi16 ADDRri:$addr))]>;
def LDrr   : F3_1<3, 0b000000,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRrr:$addr))]>;
def LDri   : F3_2<3, 0b000000,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set IntRegs:$dst, (load ADDRri:$addr))]>;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFrr  : F3_1<3, 0b100000,
                  (outs FPRegs:$dst), (ins MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRrr:$addr))]>;
def LDFri  : F3_2<3, 0b100000,
                  (outs FPRegs:$dst), (ins MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set FPRegs:$dst, (load ADDRri:$addr))]>;
def LDDFrr : F3_1<3, 0b100011,
                  (outs DFPRegs:$dst), (ins MEMrr:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRrr:$addr))]>;
def LDDFri : F3_2<3, 0b100011,
                  (outs DFPRegs:$dst), (ins MEMri:$addr),
                  "ldd [$addr], $dst",
                  [(set DFPRegs:$dst, (load ADDRri:$addr))]>;

// Section B.4 - Store Integer Instructions, p. 95
def STBrr : F3_1<3, 0b000101,
                 (outs), (ins MEMrr:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstorei8 IntRegs:$src, ADDRrr:$addr)]>;
def STBri : F3_2<3, 0b000101,
                 (outs), (ins MEMri:$addr, IntRegs:$src),
                 "stb $src, [$addr]",
                 [(truncstorei8 IntRegs:$src, ADDRri:$addr)]>;
def STHrr : F3_1<3, 0b000110,
                 (outs), (ins MEMrr:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstorei16 IntRegs:$src, ADDRrr:$addr)]>;
def STHri : F3_2<3, 0b000110,
                 (outs), (ins MEMri:$addr, IntRegs:$src),
                 "sth $src, [$addr]",
                 [(truncstorei16 IntRegs:$src, ADDRri:$addr)]>;
def STrr  : F3_1<3, 0b000100,
                 (outs), (ins MEMrr:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRrr:$addr)]>;
def STri  : F3_2<3, 0b000100,
                 (outs), (ins MEMri:$addr, IntRegs:$src),
                 "st $src, [$addr]",
                 [(store IntRegs:$src, ADDRri:$addr)]>;

// Section B.5 - Store Floating-point Instructions, p. 97
def STFrr   : F3_1<3, 0b100100,
                   (outs), (ins MEMrr:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRrr:$addr)]>;
def STFri   : F3_2<3, 0b100100,
                   (outs), (ins MEMri:$addr, FPRegs:$src),
                   "st $src, [$addr]",
                   [(store FPRegs:$src, ADDRri:$addr)]>;
def STDFrr  : F3_1<3, 0b100111,
                   (outs), (ins MEMrr:$addr, DFPRegs:$src),
                   "std  $src, [$addr]",
                   [(store DFPRegs:$src, ADDRrr:$addr)]>;
def STDFri  : F3_2<3, 0b100111,
                   (outs), (ins MEMri:$addr, DFPRegs:$src),
                   "std $src, [$addr]",
                   [(store DFPRegs:$src, ADDRri:$addr)]>;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100,
                 (outs IntRegs:$dst), (ins i32imm:$src),
                 "sethi $src, $dst",
                 [(set IntRegs:$dst, SETHIimm:$src)]>;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm22 = 0 in
  def NOP : F2_1<0b100, (outs), (ins), "nop", []>;

// Section B.11 - Logical Instructions, p. 106
defm AND    : F3_12<"and", 0b000001, and>;

def ANDNrr  : F3_1<2, 0b000101,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "andn $b, $c, $dst",
                   [(set IntRegs:$dst, (and IntRegs:$b, (not IntRegs:$c)))]>;
def ANDNri  : F3_2<2, 0b000101,
                   (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                   "andn $b, $c, $dst", []>;

defm OR     : F3_12<"or", 0b000010, or>;

def ORNrr   : F3_1<2, 0b000110,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "orn $b, $c, $dst",
                   [(set IntRegs:$dst, (or IntRegs:$b, (not IntRegs:$c)))]>;
def ORNri   : F3_2<2, 0b000110,
                   (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                   "orn $b, $c, $dst", []>;
defm XOR    : F3_12<"xor", 0b000011, xor>;

def XNORrr  : F3_1<2, 0b000111,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "xnor $b, $c, $dst",
                   [(set IntRegs:$dst, (not (xor IntRegs:$b, IntRegs:$c)))]>;
def XNORri  : F3_2<2, 0b000111,
                   (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                   "xnor $b, $c, $dst", []>;

// Section B.12 - Shift Instructions, p. 107
defm SLL : F3_12<"sll", 0b100101, shl>;
defm SRL : F3_12<"srl", 0b100110, srl>;
defm SRA : F3_12<"sra", 0b100111, sra>;

// Section B.13 - Add Instructions, p. 108
defm ADD   : F3_12<"add", 0b000000, add>;

// "LEA" forms of add (patterns to make tblgen happy)
def LEA_ADDri   : F3_2<2, 0b000000,
                   (outs IntRegs:$dst), (ins MEMri:$addr),
                   "add ${addr:arith}, $dst",
                   [(set IntRegs:$dst, ADDRri:$addr)]>;

let Defs = [ICC] in                   
  defm ADDCC  : F3_12<"addcc", 0b010000, addc>;

let Uses = [ICC] in
  defm ADDX  : F3_12<"addx", 0b001000, adde>;

// Section B.15 - Subtract Instructions, p. 110
defm SUB    : F3_12  <"sub"  , 0b000100, sub>;
let Uses = [ICC] in 
  defm SUBX   : F3_12  <"subx" , 0b001100, sube>;

let Defs = [ICC] in 
  defm SUBCC  : F3_12  <"subcc", 0b010100, SPcmpicc>;

let Uses = [ICC], Defs = [ICC] in
  def SUBXCCrr: F3_1<2, 0b011100, 
                (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                "subxcc $b, $c, $dst", []>;


// Section B.18 - Multiply Instructions, p. 113
let Defs = [Y] in {
  defm UMUL : F3_12np<"umul", 0b001010>;
  defm SMUL : F3_12  <"smul", 0b001011, mul>;
}

// Section B.19 - Divide Instructions, p. 115
let Defs = [Y] in {
  defm UDIV : F3_12np<"udiv", 0b001110>;
  defm SDIV : F3_12np<"sdiv", 0b001111>;
}

// Section B.20 - SAVE and RESTORE, p. 117
defm SAVE    : F3_12np<"save"   , 0b111100>;
defm RESTORE : F3_12np<"restore", 0b111101>;

// Section B.21 - Branch on Integer Condition Codes Instructions, p. 119

// conditional branch class:
class BranchSP<bits<4> cc, dag ins, string asmstr, list<dag> pattern>
 : F2_2<cc, 0b010, (outs), ins, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

let isBarrier = 1 in
  def BA   : BranchSP<0b1000, (ins brtarget:$dst),
                      "ba $dst",
                      [(br bb:$dst)]>;

// FIXME: the encoding for the JIT should look at the condition field.
let Uses = [ICC] in
  def BCOND : BranchSP<0, (ins brtarget:$dst, CCOp:$cc),
                         "b$cc $dst",
                        [(SPbricc bb:$dst, imm:$cc)]>;


// Section B.22 - Branch on Floating-point Condition Codes Instructions, p. 121

// floating-point conditional branch class:
class FPBranchSP<bits<4> cc, dag ins, string asmstr, list<dag> pattern>
 : F2_2<cc, 0b110, (outs), ins, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

// FIXME: the encoding for the JIT should look at the condition field.
let Uses = [FCC] in
  def FBCOND  : FPBranchSP<0, (ins brtarget:$dst, CCOp:$cc),
                              "fb$cc $dst",
                              [(SPbrfcc bb:$dst, imm:$cc)]>;


// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
let Uses = [O6],
    hasDelaySlot = 1, isCall = 1,
    Defs = [O0, O1, O2, O3, O4, O5, O7, G1, G2, G3, G4, G5, G6, G7,
    D0, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
        ICC, FCC, Y] in {
  def CALL : InstSP<(outs), (ins calltarget:$dst, variable_ops),
                    "call $dst", []> {
    bits<30> disp;
    let op = 1;
    let Inst{29-0} = disp;
  }
  
  // indirect calls
  def JMPLrr : F3_1<2, 0b111000,
                    (outs), (ins MEMrr:$ptr, variable_ops),
                    "call $ptr",
                    [(call ADDRrr:$ptr)]>;
  def JMPLri : F3_2<2, 0b111000,
                    (outs), (ins MEMri:$ptr, variable_ops),
                    "call $ptr",
                    [(call ADDRri:$ptr)]>;
}

// Section B.28 - Read State Register Instructions
let Uses = [Y] in 
  def RDY : F3_1<2, 0b101000,
                 (outs IntRegs:$dst), (ins),
                 "rd %y, $dst", []>;

// Section B.29 - Write State Register Instructions
let Defs = [Y] in {
  def WRYrr : F3_1<2, 0b110000,
                   (outs), (ins IntRegs:$b, IntRegs:$c),
                   "wr $b, $c, %y", []>;
  def WRYri : F3_2<2, 0b110000,
                   (outs), (ins IntRegs:$b, i32imm:$c),
                   "wr $b, $c, %y", []>;
}
// Convert Integer to Floating-point Instructions, p. 141
def FITOS : F3_3<2, 0b110100, 0b011000100,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fitos $src, $dst",
                 [(set FPRegs:$dst, (SPitof FPRegs:$src))]>;
def FITOD : F3_3<2, 0b110100, 0b011001000, 
                 (outs DFPRegs:$dst), (ins FPRegs:$src),
                 "fitod $src, $dst",
                 [(set DFPRegs:$dst, (SPitof FPRegs:$src))]>;

// Convert Floating-point to Integer Instructions, p. 142
def FSTOI : F3_3<2, 0b110100, 0b011010001,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fstoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi FPRegs:$src))]>;
def FDTOI : F3_3<2, 0b110100, 0b011010010,
                 (outs FPRegs:$dst), (ins DFPRegs:$src),
                 "fdtoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi DFPRegs:$src))]>;

// Convert between Floating-point Formats Instructions, p. 143
def FSTOD : F3_3<2, 0b110100, 0b011001001, 
                 (outs DFPRegs:$dst), (ins FPRegs:$src),
                 "fstod $src, $dst",
                 [(set DFPRegs:$dst, (fextend FPRegs:$src))]>;
def FDTOS : F3_3<2, 0b110100, 0b011000110,
                 (outs FPRegs:$dst), (ins DFPRegs:$src),
                 "fdtos $src, $dst",
                 [(set FPRegs:$dst, (fround DFPRegs:$src))]>;

// Floating-point Move Instructions, p. 144
def FMOVS : F3_3<2, 0b110100, 0b000000001,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fmovs $src, $dst", []>;
def FNEGS : F3_3<2, 0b110100, 0b000000101, 
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fnegs $src, $dst",
                 [(set FPRegs:$dst, (fneg FPRegs:$src))]>;
def FABSS : F3_3<2, 0b110100, 0b000001001, 
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fabss $src, $dst",
                 [(set FPRegs:$dst, (fabs FPRegs:$src))]>;


// Floating-point Square Root Instructions, p.145
def FSQRTS : F3_3<2, 0b110100, 0b000101001, 
                  (outs FPRegs:$dst), (ins FPRegs:$src),
                  "fsqrts $src, $dst",
                  [(set FPRegs:$dst, (fsqrt FPRegs:$src))]>;
def FSQRTD : F3_3<2, 0b110100, 0b000101010, 
                  (outs DFPRegs:$dst), (ins DFPRegs:$src),
                  "fsqrtd $src, $dst",
                  [(set DFPRegs:$dst, (fsqrt DFPRegs:$src))]>;



// Floating-point Add and Subtract Instructions, p. 146
def FADDS  : F3_3<2, 0b110100, 0b001000001,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fadds $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fadd FPRegs:$src1, FPRegs:$src2))]>;
def FADDD  : F3_3<2, 0b110100, 0b001000010,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "faddd $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fadd DFPRegs:$src1, DFPRegs:$src2))]>;
def FSUBS  : F3_3<2, 0b110100, 0b001000101,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fsubs $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fsub FPRegs:$src1, FPRegs:$src2))]>;
def FSUBD  : F3_3<2, 0b110100, 0b001000110,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "fsubd $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fsub DFPRegs:$src1, DFPRegs:$src2))]>;

// Floating-point Multiply and Divide Instructions, p. 147
def FMULS  : F3_3<2, 0b110100, 0b001001001,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fmuls $src1, $src2, $dst",
                  [(set FPRegs:$dst, (fmul FPRegs:$src1, FPRegs:$src2))]>;
def FMULD  : F3_3<2, 0b110100, 0b001001010,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "fmuld $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fmul DFPRegs:$src1, DFPRegs:$src2))]>;
def FSMULD : F3_3<2, 0b110100, 0b001101001,
                  (outs DFPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fsmuld $src1, $src2, $dst",
                  [(set DFPRegs:$dst, (fmul (fextend FPRegs:$src1),
                                            (fextend FPRegs:$src2)))]>;
def FDIVS  : F3_3<2, 0b110100, 0b001001101,
                 (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                 "fdivs $src1, $src2, $dst",
                 [(set FPRegs:$dst, (fdiv FPRegs:$src1, FPRegs:$src2))]>;
def FDIVD  : F3_3<2, 0b110100, 0b001001110,
                 (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                 "fdivd $src1, $src2, $dst",
                 [(set DFPRegs:$dst, (fdiv DFPRegs:$src1, DFPRegs:$src2))]>;

// Floating-point Compare Instructions, p. 148
// Note: the 2nd template arg is different for these guys.
// Note 2: the result of a FCMP is not available until the 2nd cycle
// after the instr is retired, but there is no interlock. This behavior
// is modelled with a forced noop after the instruction.
let Defs = [FCC] in {
  def FCMPS  : F3_3<2, 0b110101, 0b001010001,
                   (outs), (ins FPRegs:$src1, FPRegs:$src2),
                   "fcmps $src1, $src2\n\tnop",
                   [(SPcmpfcc FPRegs:$src1, FPRegs:$src2)]>;
  def FCMPD  : F3_3<2, 0b110101, 0b001010010,
                   (outs), (ins DFPRegs:$src1, DFPRegs:$src2),
                   "fcmpd $src1, $src2\n\tnop",
                   [(SPcmpfcc DFPRegs:$src1, DFPRegs:$src2)]>;
}

//===----------------------------------------------------------------------===//
// V9 Instructions
//===----------------------------------------------------------------------===//

// V9 Conditional Moves.
let Predicates = [HasV9], Constraints = "$T = $dst" in {
  // Move Integer Register on Condition (MOVcc) p. 194 of the V9 manual.
  // FIXME: Add instruction encodings for the JIT some day.
  let Uses = [ICC] in {
    def MOVICCrr
      : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, CCOp:$cc),
               "mov$cc %icc, $F, $dst",
               [(set IntRegs:$dst,
                           (SPselecticc IntRegs:$F, IntRegs:$T, imm:$cc))]>;
    def MOVICCri
      : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, i32imm:$F, CCOp:$cc),
               "mov$cc %icc, $F, $dst",
               [(set IntRegs:$dst,
                            (SPselecticc simm11:$F, IntRegs:$T, imm:$cc))]>;
  }

  let Uses = [FCC] in {
    def MOVFCCrr
      : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, CCOp:$cc),
               "mov$cc %fcc0, $F, $dst",
               [(set IntRegs:$dst,
                           (SPselectfcc IntRegs:$F, IntRegs:$T, imm:$cc))]>;
    def MOVFCCri
      : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, i32imm:$F, CCOp:$cc),
               "mov$cc %fcc0, $F, $dst",
               [(set IntRegs:$dst,
                            (SPselectfcc simm11:$F, IntRegs:$T, imm:$cc))]>;
  }

  let Uses = [ICC] in {
    def FMOVS_ICC
      : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, CCOp:$cc),
               "fmovs$cc %icc, $F, $dst",
               [(set FPRegs:$dst,
                           (SPselecticc FPRegs:$F, FPRegs:$T, imm:$cc))]>;
    def FMOVD_ICC
      : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, CCOp:$cc),
               "fmovd$cc %icc, $F, $dst",
               [(set DFPRegs:$dst,
                           (SPselecticc DFPRegs:$F, DFPRegs:$T, imm:$cc))]>;
  }

  let Uses = [FCC] in {
    def FMOVS_FCC
      : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, CCOp:$cc),
               "fmovs$cc %fcc0, $F, $dst",
               [(set FPRegs:$dst,
                           (SPselectfcc FPRegs:$F, FPRegs:$T, imm:$cc))]>;
    def FMOVD_FCC
      : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, CCOp:$cc),
               "fmovd$cc %fcc0, $F, $dst",
               [(set DFPRegs:$dst,
                           (SPselectfcc DFPRegs:$F, DFPRegs:$T, imm:$cc))]>;
  }

}

// Floating-Point Move Instructions, p. 164 of the V9 manual.
let Predicates = [HasV9] in {
  def FMOVD : F3_3<2, 0b110100, 0b000000010,
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fmovd $src, $dst", []>;
  def FNEGD : F3_3<2, 0b110100, 0b000000110, 
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fnegd $src, $dst",
                   [(set DFPRegs:$dst, (fneg DFPRegs:$src))]>;
  def FABSD : F3_3<2, 0b110100, 0b000001010, 
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fabsd $src, $dst",
                   [(set DFPRegs:$dst, (fabs DFPRegs:$src))]>;
}

// POPCrr - This does a ctpop of a 64-bit register.  As such, we have to clear
// the top 32-bits before using it.  To do this clearing, we use a SLLri X,0.
def POPCrr : F3_1<2, 0b101110, 
                  (outs IntRegs:$dst), (ins IntRegs:$src),
                  "popc $src, $dst", []>, Requires<[HasV9]>;
def : Pat<(ctpop IntRegs:$src),
          (POPCrr (SLLri IntRegs:$src, 0))>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Small immediates.
def : Pat<(i32 simm13:$val),
          (ORri G0, imm:$val)>;
// Arbitrary immediates.
def : Pat<(i32 imm:$val),
          (ORri (SETHIi (HI22 imm:$val)), (LO10 imm:$val))>;

// subc
def : Pat<(subc IntRegs:$b, IntRegs:$c),
          (SUBCCrr IntRegs:$b, IntRegs:$c)>;
def : Pat<(subc IntRegs:$b, simm13:$val),
          (SUBCCri IntRegs:$b, imm:$val)>;

// Global addresses, constant pool entries
def : Pat<(SPhi tglobaladdr:$in), (SETHIi tglobaladdr:$in)>;
def : Pat<(SPlo tglobaladdr:$in), (ORri G0, tglobaladdr:$in)>;
def : Pat<(SPhi tconstpool:$in), (SETHIi tconstpool:$in)>;
def : Pat<(SPlo tconstpool:$in), (ORri G0, tconstpool:$in)>;

// Add reg, lo.  This is used when taking the addr of a global/constpool entry.
def : Pat<(add IntRegs:$r, (SPlo tglobaladdr:$in)),
          (ADDri IntRegs:$r, tglobaladdr:$in)>;
def : Pat<(add IntRegs:$r, (SPlo tconstpool:$in)),
          (ADDri IntRegs:$r, tconstpool:$in)>;

// Calls: 
def : Pat<(call tglobaladdr:$dst),
          (CALL tglobaladdr:$dst)>;
def : Pat<(call texternalsym:$dst),
          (CALL texternalsym:$dst)>;

// Map integer extload's to zextloads.
def : Pat<(i32 (extloadi1 ADDRrr:$src)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (extloadi1 ADDRri:$src)), (LDUBri ADDRri:$src)>;
def : Pat<(i32 (extloadi8 ADDRrr:$src)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (extloadi8 ADDRri:$src)), (LDUBri ADDRri:$src)>;
def : Pat<(i32 (extloadi16 ADDRrr:$src)), (LDUHrr ADDRrr:$src)>;
def : Pat<(i32 (extloadi16 ADDRri:$src)), (LDUHri ADDRri:$src)>;

// zextload bool -> zextload byte
def : Pat<(i32 (zextloadi1 ADDRrr:$src)), (LDUBrr ADDRrr:$src)>;
def : Pat<(i32 (zextloadi1 ADDRri:$src)), (LDUBri ADDRri:$src)>;
