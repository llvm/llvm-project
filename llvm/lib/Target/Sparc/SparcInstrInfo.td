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

// True when generating 32-bit code.
def Is32Bit : Predicate<"!Subtarget.is64Bit()">;

// True when generating 64-bit code. This also implies HasV9.
def Is64Bit : Predicate<"Subtarget.is64Bit()">;

// HasV9 - This predicate is true when the target processor supports V9
// instructions.  Note that the machine may be running in 32-bit mode.
def HasV9   : Predicate<"Subtarget.isV9()">;

// HasNoV9 - This predicate is true when the target doesn't have V9
// instructions.  Use of this is just a hack for the isel not having proper
// costs for V8 instructions that are more expensive than their V9 ones.
def HasNoV9 : Predicate<"!Subtarget.isV9()">;

// HasVIS - This is true when the target processor has VIS extensions.
def HasVIS : Predicate<"Subtarget.isVIS()">;

// HasHardQuad - This is true when the target processor supports quad floating
// point instructions.
def HasHardQuad : Predicate<"Subtarget.hasHardQuad()">;

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
  return isShiftedUInt<22, 10>(N->getZExtValue());
}], HI22>;

// Addressing modes.
def ADDRrr : ComplexPattern<iPTR, 2, "SelectADDRrr", [], []>;
def ADDRri : ComplexPattern<iPTR, 2, "SelectADDRri", [frameindex], []>;

// Address operands
def MEMrr : Operand<iPTR> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ptr_rc, ptr_rc);
}
def MEMri : Operand<iPTR> {
  let PrintMethod = "printMemOperand";
  let MIOperandInfo = (ops ptr_rc, i32imm);
}

def TLSSym : Operand<iPTR>;

// Branch targets have OtherVT type.
def brtarget : Operand<OtherVT>;
def calltarget : Operand<i32>;

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i32>;

def SDTSPcmpicc :
SDTypeProfile<0, 2, [SDTCisInt<0>, SDTCisSameAs<0, 1>]>;
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
def SDTSPFTOX :
SDTypeProfile<1, 1, [SDTCisVT<0, f64>, SDTCisFP<1>]>;
def SDTSPXTOF :
SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisVT<1, f64>]>;

def SDTSPtlsadd :
SDTypeProfile<1, 3, [SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisPtrTy<2>]>;
def SDTSPtlsld :
SDTypeProfile<1, 2, [SDTCisPtrTy<0>, SDTCisPtrTy<1>]>;

def SPcmpicc : SDNode<"SPISD::CMPICC", SDTSPcmpicc, [SDNPOutGlue]>;
def SPcmpfcc : SDNode<"SPISD::CMPFCC", SDTSPcmpfcc, [SDNPOutGlue]>;
def SPbricc : SDNode<"SPISD::BRICC", SDTSPbrcc, [SDNPHasChain, SDNPInGlue]>;
def SPbrxcc : SDNode<"SPISD::BRXCC", SDTSPbrcc, [SDNPHasChain, SDNPInGlue]>;
def SPbrfcc : SDNode<"SPISD::BRFCC", SDTSPbrcc, [SDNPHasChain, SDNPInGlue]>;

def SPhi    : SDNode<"SPISD::Hi", SDTIntUnaryOp>;
def SPlo    : SDNode<"SPISD::Lo", SDTIntUnaryOp>;

def SPftoi  : SDNode<"SPISD::FTOI", SDTSPFTOI>;
def SPitof  : SDNode<"SPISD::ITOF", SDTSPITOF>;
def SPftox  : SDNode<"SPISD::FTOX", SDTSPFTOX>;
def SPxtof  : SDNode<"SPISD::XTOF", SDTSPXTOF>;

def SPselecticc : SDNode<"SPISD::SELECT_ICC", SDTSPselectcc, [SDNPInGlue]>;
def SPselectxcc : SDNode<"SPISD::SELECT_XCC", SDTSPselectcc, [SDNPInGlue]>;
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
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def flushw        : SDNode<"SPISD::FLUSHW", SDTNone,
                           [SDNPHasChain, SDNPSideEffect, SDNPMayStore]>;

def tlsadd        : SDNode<"SPISD::TLS_ADD", SDTSPtlsadd>;
def tlsld         : SDNode<"SPISD::TLS_LD",  SDTSPtlsld>;
def tlscall       : SDNode<"SPISD::TLS_CALL", SDT_SPCall,
                            [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                             SDNPVariadic]>;

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
                 [(set i32:$dst, (OpNode i32:$b, i32:$c))]>;
  def ri  : F3_2<2, Op3Val,
                 (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                 !strconcat(OpcStr, " $b, $c, $dst"),
                 [(set i32:$dst, (OpNode i32:$b, (i32 simm13:$c)))]>;
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

let rd = 0 in
  def UNIMP : F2_1<0b000, (outs), (ins i32imm:$val),
                  "unimp $val", []>;

// SELECT_CC_* - Used to implement the SELECT_CC DAG operation.  Expanded after
// instruction selection into a branch sequence.  This has to handle all
// permutations of selection between i32/f32/f64 on ICC and FCC.
// Expanded after instruction selection.
let Uses = [ICC], usesCustomInserter = 1 in {
  def SELECT_CC_Int_ICC
   : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_ICC PSEUDO!",
            [(set i32:$dst, (SPselecticc i32:$T, i32:$F, imm:$Cond))]>;
  def SELECT_CC_FP_ICC
   : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_ICC PSEUDO!",
            [(set f32:$dst, (SPselecticc f32:$T, f32:$F, imm:$Cond))]>;

  def SELECT_CC_DFP_ICC
   : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_ICC PSEUDO!",
            [(set f64:$dst, (SPselecticc f64:$T, f64:$F, imm:$Cond))]>;

  def SELECT_CC_QFP_ICC
   : Pseudo<(outs QFPRegs:$dst), (ins QFPRegs:$T, QFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_QFP_ICC PSEUDO!",
            [(set f128:$dst, (SPselecticc f128:$T, f128:$F, imm:$Cond))]>;
}

let usesCustomInserter = 1, Uses = [FCC] in {

  def SELECT_CC_Int_FCC
   : Pseudo<(outs IntRegs:$dst), (ins IntRegs:$T, IntRegs:$F, i32imm:$Cond),
            "; SELECT_CC_Int_FCC PSEUDO!",
            [(set i32:$dst, (SPselectfcc i32:$T, i32:$F, imm:$Cond))]>;

  def SELECT_CC_FP_FCC
   : Pseudo<(outs FPRegs:$dst), (ins FPRegs:$T, FPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_FP_FCC PSEUDO!",
            [(set f32:$dst, (SPselectfcc f32:$T, f32:$F, imm:$Cond))]>;
  def SELECT_CC_DFP_FCC
   : Pseudo<(outs DFPRegs:$dst), (ins DFPRegs:$T, DFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_DFP_FCC PSEUDO!",
            [(set f64:$dst, (SPselectfcc f64:$T, f64:$F, imm:$Cond))]>;
  def SELECT_CC_QFP_FCC
   : Pseudo<(outs QFPRegs:$dst), (ins QFPRegs:$T, QFPRegs:$F, i32imm:$Cond),
            "; SELECT_CC_QFP_FCC PSEUDO!",
            [(set f128:$dst, (SPselectfcc f128:$T, f128:$F, imm:$Cond))]>;
}


// Section A.3 - Synthetic Instructions, p. 85
// special cases of JMPL:
let isReturn = 1, isTerminator = 1, hasDelaySlot = 1, isBarrier = 1 in {
  let rd = 0, rs1 = 15 in
    def RETL: F3_2<2, 0b111000, (outs), (ins i32imm:$val),
                   "jmp %o7+$val", [(retflag simm13:$val)]>;

  let rd = 0, rs1 = 31 in
    def RET: F3_2<2, 0b111000, (outs), (ins i32imm:$val),
                  "jmp %i7+$val", []>;
}

// Section B.1 - Load Integer Instructions, p. 90
def LDSBrr : F3_1<3, 0b001001,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldsb [$addr], $dst",
                  [(set i32:$dst, (sextloadi8 ADDRrr:$addr))]>;
def LDSBri : F3_2<3, 0b001001,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldsb [$addr], $dst",
                  [(set i32:$dst, (sextloadi8 ADDRri:$addr))]>;
def LDSHrr : F3_1<3, 0b001010,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldsh [$addr], $dst",
                  [(set i32:$dst, (sextloadi16 ADDRrr:$addr))]>;
def LDSHri : F3_2<3, 0b001010,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldsh [$addr], $dst",
                  [(set i32:$dst, (sextloadi16 ADDRri:$addr))]>;
def LDUBrr : F3_1<3, 0b000001,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ldub [$addr], $dst",
                  [(set i32:$dst, (zextloadi8 ADDRrr:$addr))]>;
def LDUBri : F3_2<3, 0b000001,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ldub [$addr], $dst",
                  [(set i32:$dst, (zextloadi8 ADDRri:$addr))]>;
def LDUHrr : F3_1<3, 0b000010,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "lduh [$addr], $dst",
                  [(set i32:$dst, (zextloadi16 ADDRrr:$addr))]>;
def LDUHri : F3_2<3, 0b000010,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "lduh [$addr], $dst",
                  [(set i32:$dst, (zextloadi16 ADDRri:$addr))]>;
def LDrr   : F3_1<3, 0b000000,
                  (outs IntRegs:$dst), (ins MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set i32:$dst, (load ADDRrr:$addr))]>;
def LDri   : F3_2<3, 0b000000,
                  (outs IntRegs:$dst), (ins MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set i32:$dst, (load ADDRri:$addr))]>;

// Section B.2 - Load Floating-point Instructions, p. 92
def LDFrr  : F3_1<3, 0b100000,
                  (outs FPRegs:$dst), (ins MEMrr:$addr),
                  "ld [$addr], $dst",
                  [(set f32:$dst, (load ADDRrr:$addr))]>;
def LDFri  : F3_2<3, 0b100000,
                  (outs FPRegs:$dst), (ins MEMri:$addr),
                  "ld [$addr], $dst",
                  [(set f32:$dst, (load ADDRri:$addr))]>;
def LDDFrr : F3_1<3, 0b100011,
                  (outs DFPRegs:$dst), (ins MEMrr:$addr),
                  "ldd [$addr], $dst",
                  [(set f64:$dst, (load ADDRrr:$addr))]>;
def LDDFri : F3_2<3, 0b100011,
                  (outs DFPRegs:$dst), (ins MEMri:$addr),
                  "ldd [$addr], $dst",
                  [(set f64:$dst, (load ADDRri:$addr))]>;
def LDQFrr : F3_1<3, 0b100010,
                  (outs QFPRegs:$dst), (ins MEMrr:$addr),
                  "ldq [$addr], $dst",
                  [(set f128:$dst, (load ADDRrr:$addr))]>,
                  Requires<[HasV9, HasHardQuad]>;
def LDQFri : F3_2<3, 0b100010,
                  (outs QFPRegs:$dst), (ins MEMri:$addr),
                  "ldq [$addr], $dst",
                  [(set f128:$dst, (load ADDRri:$addr))]>,
                  Requires<[HasV9, HasHardQuad]>;

// Section B.4 - Store Integer Instructions, p. 95
def STBrr : F3_1<3, 0b000101,
                 (outs), (ins MEMrr:$addr, IntRegs:$rd),
                 "stb $rd, [$addr]",
                 [(truncstorei8 i32:$rd, ADDRrr:$addr)]>;
def STBri : F3_2<3, 0b000101,
                 (outs), (ins MEMri:$addr, IntRegs:$rd),
                 "stb $rd, [$addr]",
                 [(truncstorei8 i32:$rd, ADDRri:$addr)]>;
def STHrr : F3_1<3, 0b000110,
                 (outs), (ins MEMrr:$addr, IntRegs:$rd),
                 "sth $rd, [$addr]",
                 [(truncstorei16 i32:$rd, ADDRrr:$addr)]>;
def STHri : F3_2<3, 0b000110,
                 (outs), (ins MEMri:$addr, IntRegs:$rd),
                 "sth $rd, [$addr]",
                 [(truncstorei16 i32:$rd, ADDRri:$addr)]>;
def STrr  : F3_1<3, 0b000100,
                 (outs), (ins MEMrr:$addr, IntRegs:$rd),
                 "st $rd, [$addr]",
                 [(store i32:$rd, ADDRrr:$addr)]>;
def STri  : F3_2<3, 0b000100,
                 (outs), (ins MEMri:$addr, IntRegs:$rd),
                 "st $rd, [$addr]",
                 [(store i32:$rd, ADDRri:$addr)]>;

// Section B.5 - Store Floating-point Instructions, p. 97
def STFrr   : F3_1<3, 0b100100,
                   (outs), (ins MEMrr:$addr, FPRegs:$rd),
                   "st $rd, [$addr]",
                   [(store f32:$rd, ADDRrr:$addr)]>;
def STFri   : F3_2<3, 0b100100,
                   (outs), (ins MEMri:$addr, FPRegs:$rd),
                   "st $rd, [$addr]",
                   [(store f32:$rd, ADDRri:$addr)]>;
def STDFrr  : F3_1<3, 0b100111,
                   (outs), (ins MEMrr:$addr, DFPRegs:$rd),
                   "std  $rd, [$addr]",
                   [(store f64:$rd, ADDRrr:$addr)]>;
def STDFri  : F3_2<3, 0b100111,
                   (outs), (ins MEMri:$addr, DFPRegs:$rd),
                   "std $rd, [$addr]",
                   [(store f64:$rd, ADDRri:$addr)]>;
def STQFrr  : F3_1<3, 0b100110,
                   (outs), (ins MEMrr:$addr, QFPRegs:$rd),
                   "stq  $rd, [$addr]",
                   [(store f128:$rd, ADDRrr:$addr)]>,
                   Requires<[HasV9, HasHardQuad]>;
def STQFri  : F3_2<3, 0b100110,
                   (outs), (ins MEMri:$addr, QFPRegs:$rd),
                   "stq $rd, [$addr]",
                   [(store f128:$rd, ADDRri:$addr)]>,
                   Requires<[HasV9, HasHardQuad]>;

// Section B.9 - SETHI Instruction, p. 104
def SETHIi: F2_1<0b100,
                 (outs IntRegs:$rd), (ins i32imm:$imm22),
                 "sethi $imm22, $rd",
                 [(set i32:$rd, SETHIimm:$imm22)]>;

// Section B.10 - NOP Instruction, p. 105
// (It's a special case of SETHI)
let rd = 0, imm22 = 0 in
  def NOP : F2_1<0b100, (outs), (ins), "nop", []>;

// Section B.11 - Logical Instructions, p. 106
defm AND    : F3_12<"and", 0b000001, and>;

def ANDNrr  : F3_1<2, 0b000101,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "andn $b, $c, $dst",
                   [(set i32:$dst, (and i32:$b, (not i32:$c)))]>;
def ANDNri  : F3_2<2, 0b000101,
                   (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                   "andn $b, $c, $dst", []>;

defm OR     : F3_12<"or", 0b000010, or>;

def ORNrr   : F3_1<2, 0b000110,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "orn $b, $c, $dst",
                   [(set i32:$dst, (or i32:$b, (not i32:$c)))]>;
def ORNri   : F3_2<2, 0b000110,
                   (outs IntRegs:$dst), (ins IntRegs:$b, i32imm:$c),
                   "orn $b, $c, $dst", []>;
defm XOR    : F3_12<"xor", 0b000011, xor>;

def XNORrr  : F3_1<2, 0b000111,
                   (outs IntRegs:$dst), (ins IntRegs:$b, IntRegs:$c),
                   "xnor $b, $c, $dst",
                   [(set i32:$dst, (not (xor i32:$b, i32:$c)))]>;
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
let Predicates = [Is32Bit] in
  def LEA_ADDri   : F3_2<2, 0b000000,
                     (outs IntRegs:$dst), (ins MEMri:$addr),
                     "add ${addr:arith}, $dst",
                     [(set iPTR:$dst, ADDRri:$addr)]>;

let Defs = [ICC] in
  defm ADDCC  : F3_12<"addcc", 0b010000, addc>;

let Uses = [ICC], Defs = [ICC] in
  defm ADDX  : F3_12<"addxcc", 0b011000, adde>;

// Section B.15 - Subtract Instructions, p. 110
defm SUB    : F3_12  <"sub"  , 0b000100, sub>;
let Uses = [ICC], Defs = [ICC] in
  defm SUBX   : F3_12  <"subxcc" , 0b011100, sube>;

let Defs = [ICC] in
  defm SUBCC  : F3_12  <"subcc", 0b010100, subc>;

let Defs = [ICC], rd = 0 in {
  def CMPrr   : F3_1<2, 0b010100,
                     (outs), (ins IntRegs:$b, IntRegs:$c),
                     "cmp $b, $c",
                     [(SPcmpicc i32:$b, i32:$c)]>;
  def CMPri   : F3_2<2, 0b010100,
                     (outs), (ins IntRegs:$b, i32imm:$c),
                     "cmp $b, $c",
                     [(SPcmpicc i32:$b, (i32 simm13:$c))]>;
}

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

// unconditional branch class.
class BranchAlways<dag ins, string asmstr, list<dag> pattern>
  : F2_2<0b010, (outs), ins, asmstr, pattern> {
  let isBranch     = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
  let isBarrier    = 1;
}

let cond = 8 in
  def BA : BranchAlways<(ins brtarget:$imm22), "ba $imm22", [(br bb:$imm22)]>;

// conditional branch class:
class BranchSP<dag ins, string asmstr, list<dag> pattern>
 : F2_2<0b010, (outs), ins, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

// Indirect branch instructions.
let isTerminator = 1, isBarrier = 1,
     hasDelaySlot = 1, isBranch =1,
     isIndirectBranch = 1, rd = 0 in {
  def BINDrr  : F3_1<2, 0b111000,
                   (outs), (ins MEMrr:$ptr),
                   "jmp $ptr",
                   [(brind ADDRrr:$ptr)]>;
  def BINDri  : F3_2<2, 0b111000,
                   (outs), (ins MEMri:$ptr),
                   "jmp $ptr",
                   [(brind ADDRri:$ptr)]>;
}

let Uses = [ICC] in
  def BCOND : BranchSP<(ins brtarget:$imm22, CCOp:$cond),
                         "b$cond $imm22",
                        [(SPbricc bb:$imm22, imm:$cond)]>;

// Section B.22 - Branch on Floating-point Condition Codes Instructions, p. 121

// floating-point conditional branch class:
class FPBranchSP<dag ins, string asmstr, list<dag> pattern>
 : F2_2<0b110, (outs), ins, asmstr, pattern> {
  let isBranch = 1;
  let isTerminator = 1;
  let hasDelaySlot = 1;
}

let Uses = [FCC] in
  def FBCOND  : FPBranchSP<(ins brtarget:$imm22, CCOp:$cond),
                              "fb$cond $imm22",
                              [(SPbrfcc bb:$imm22, imm:$cond)]>;


// Section B.24 - Call and Link Instruction, p. 125
// This is the only Format 1 instruction
let Uses = [O6],
    hasDelaySlot = 1, isCall = 1 in {
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
                    [(call ADDRrr:$ptr)]> { let rd = 15; }
  def JMPLri : F3_2<2, 0b111000,
                    (outs), (ins MEMri:$ptr, variable_ops),
                    "call $ptr",
                    [(call ADDRri:$ptr)]> { let rd = 15; }
}

// Section B.28 - Read State Register Instructions
let Uses = [Y], rs1 = 0, rs2 = 0 in
  def RDY : F3_1<2, 0b101000,
                 (outs IntRegs:$dst), (ins),
                 "rd %y, $dst", []>;

// Section B.29 - Write State Register Instructions
let Defs = [Y], rd = 0 in {
  def WRYrr : F3_1<2, 0b110000,
                   (outs), (ins IntRegs:$b, IntRegs:$c),
                   "wr $b, $c, %y", []>;
  def WRYri : F3_2<2, 0b110000,
                   (outs), (ins IntRegs:$b, i32imm:$c),
                   "wr $b, $c, %y", []>;
}
// Convert Integer to Floating-point Instructions, p. 141
def FITOS : F3_3u<2, 0b110100, 0b011000100,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fitos $src, $dst",
                 [(set FPRegs:$dst, (SPitof FPRegs:$src))]>;
def FITOD : F3_3u<2, 0b110100, 0b011001000,
                 (outs DFPRegs:$dst), (ins FPRegs:$src),
                 "fitod $src, $dst",
                 [(set DFPRegs:$dst, (SPitof FPRegs:$src))]>;
def FITOQ : F3_3u<2, 0b110100, 0b011001100,
                 (outs QFPRegs:$dst), (ins FPRegs:$src),
                 "fitoq $src, $dst",
                 [(set QFPRegs:$dst, (SPitof FPRegs:$src))]>,
                 Requires<[HasHardQuad]>;

// Convert Floating-point to Integer Instructions, p. 142
def FSTOI : F3_3u<2, 0b110100, 0b011010001,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fstoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi FPRegs:$src))]>;
def FDTOI : F3_3u<2, 0b110100, 0b011010010,
                 (outs FPRegs:$dst), (ins DFPRegs:$src),
                 "fdtoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi DFPRegs:$src))]>;
def FQTOI : F3_3u<2, 0b110100, 0b011010011,
                 (outs FPRegs:$dst), (ins QFPRegs:$src),
                 "fqtoi $src, $dst",
                 [(set FPRegs:$dst, (SPftoi QFPRegs:$src))]>,
                 Requires<[HasHardQuad]>;

// Convert between Floating-point Formats Instructions, p. 143
def FSTOD : F3_3u<2, 0b110100, 0b011001001,
                 (outs DFPRegs:$dst), (ins FPRegs:$src),
                 "fstod $src, $dst",
                 [(set f64:$dst, (fextend f32:$src))]>;
def FSTOQ : F3_3u<2, 0b110100, 0b011001101,
                 (outs QFPRegs:$dst), (ins FPRegs:$src),
                 "fstoq $src, $dst",
                 [(set f128:$dst, (fextend f32:$src))]>,
                 Requires<[HasHardQuad]>;
def FDTOS : F3_3u<2, 0b110100, 0b011000110,
                 (outs FPRegs:$dst), (ins DFPRegs:$src),
                 "fdtos $src, $dst",
                 [(set f32:$dst, (fround f64:$src))]>;
def FDTOQ : F3_3u<2, 0b110100, 0b01101110,
                 (outs QFPRegs:$dst), (ins DFPRegs:$src),
                 "fdtoq $src, $dst",
                 [(set f128:$dst, (fextend f64:$src))]>,
                 Requires<[HasHardQuad]>;
def FQTOS : F3_3u<2, 0b110100, 0b011000111,
                 (outs FPRegs:$dst), (ins QFPRegs:$src),
                 "fqtos $src, $dst",
                 [(set f32:$dst, (fround f128:$src))]>,
                 Requires<[HasHardQuad]>;
def FQTOD : F3_3u<2, 0b110100, 0b011001011,
                 (outs DFPRegs:$dst), (ins QFPRegs:$src),
                 "fqtod $src, $dst",
                 [(set f64:$dst, (fround f128:$src))]>,
                 Requires<[HasHardQuad]>;

// Floating-point Move Instructions, p. 144
def FMOVS : F3_3u<2, 0b110100, 0b000000001,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fmovs $src, $dst", []>;
def FNEGS : F3_3u<2, 0b110100, 0b000000101,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fnegs $src, $dst",
                 [(set f32:$dst, (fneg f32:$src))]>;
def FABSS : F3_3u<2, 0b110100, 0b000001001,
                 (outs FPRegs:$dst), (ins FPRegs:$src),
                 "fabss $src, $dst",
                 [(set f32:$dst, (fabs f32:$src))]>;


// Floating-point Square Root Instructions, p.145
def FSQRTS : F3_3u<2, 0b110100, 0b000101001,
                  (outs FPRegs:$dst), (ins FPRegs:$src),
                  "fsqrts $src, $dst",
                  [(set f32:$dst, (fsqrt f32:$src))]>;
def FSQRTD : F3_3u<2, 0b110100, 0b000101010,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src),
                  "fsqrtd $src, $dst",
                  [(set f64:$dst, (fsqrt f64:$src))]>;
def FSQRTQ : F3_3u<2, 0b110100, 0b000101011,
                  (outs QFPRegs:$dst), (ins QFPRegs:$src),
                  "fsqrtq $src, $dst",
                  [(set f128:$dst, (fsqrt f128:$src))]>,
                  Requires<[HasHardQuad]>;



// Floating-point Add and Subtract Instructions, p. 146
def FADDS  : F3_3<2, 0b110100, 0b001000001,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fadds $src1, $src2, $dst",
                  [(set f32:$dst, (fadd f32:$src1, f32:$src2))]>;
def FADDD  : F3_3<2, 0b110100, 0b001000010,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "faddd $src1, $src2, $dst",
                  [(set f64:$dst, (fadd f64:$src1, f64:$src2))]>;
def FADDQ  : F3_3<2, 0b110100, 0b001000011,
                  (outs QFPRegs:$dst), (ins QFPRegs:$src1, QFPRegs:$src2),
                  "faddq $src1, $src2, $dst",
                  [(set f128:$dst, (fadd f128:$src1, f128:$src2))]>,
                  Requires<[HasHardQuad]>;

def FSUBS  : F3_3<2, 0b110100, 0b001000101,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fsubs $src1, $src2, $dst",
                  [(set f32:$dst, (fsub f32:$src1, f32:$src2))]>;
def FSUBD  : F3_3<2, 0b110100, 0b001000110,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "fsubd $src1, $src2, $dst",
                  [(set f64:$dst, (fsub f64:$src1, f64:$src2))]>;
def FSUBQ  : F3_3<2, 0b110100, 0b001000111,
                  (outs QFPRegs:$dst), (ins QFPRegs:$src1, QFPRegs:$src2),
                  "fsubq $src1, $src2, $dst",
                  [(set f128:$dst, (fsub f128:$src1, f128:$src2))]>,
                  Requires<[HasHardQuad]>;


// Floating-point Multiply and Divide Instructions, p. 147
def FMULS  : F3_3<2, 0b110100, 0b001001001,
                  (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fmuls $src1, $src2, $dst",
                  [(set f32:$dst, (fmul f32:$src1, f32:$src2))]>;
def FMULD  : F3_3<2, 0b110100, 0b001001010,
                  (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "fmuld $src1, $src2, $dst",
                  [(set f64:$dst, (fmul f64:$src1, f64:$src2))]>;
def FMULQ  : F3_3<2, 0b110100, 0b001001011,
                  (outs QFPRegs:$dst), (ins QFPRegs:$src1, QFPRegs:$src2),
                  "fmulq $src1, $src2, $dst",
                  [(set f128:$dst, (fmul f128:$src1, f128:$src2))]>,
                  Requires<[HasHardQuad]>;

def FSMULD : F3_3<2, 0b110100, 0b001101001,
                  (outs DFPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                  "fsmuld $src1, $src2, $dst",
                  [(set f64:$dst, (fmul (fextend f32:$src1),
                                        (fextend f32:$src2)))]>;
def FDMULQ : F3_3<2, 0b110100, 0b001101110,
                  (outs QFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                  "fdmulq $src1, $src2, $dst",
                  [(set f128:$dst, (fmul (fextend f64:$src1),
                                         (fextend f64:$src2)))]>,
                  Requires<[HasHardQuad]>;

def FDIVS  : F3_3<2, 0b110100, 0b001001101,
                 (outs FPRegs:$dst), (ins FPRegs:$src1, FPRegs:$src2),
                 "fdivs $src1, $src2, $dst",
                 [(set f32:$dst, (fdiv f32:$src1, f32:$src2))]>;
def FDIVD  : F3_3<2, 0b110100, 0b001001110,
                 (outs DFPRegs:$dst), (ins DFPRegs:$src1, DFPRegs:$src2),
                 "fdivd $src1, $src2, $dst",
                 [(set f64:$dst, (fdiv f64:$src1, f64:$src2))]>;
def FDIVQ  : F3_3<2, 0b110100, 0b001001111,
                 (outs QFPRegs:$dst), (ins QFPRegs:$src1, QFPRegs:$src2),
                 "fdivq $src1, $src2, $dst",
                 [(set f128:$dst, (fdiv f128:$src1, f128:$src2))]>,
                 Requires<[HasHardQuad]>;

// Floating-point Compare Instructions, p. 148
// Note: the 2nd template arg is different for these guys.
// Note 2: the result of a FCMP is not available until the 2nd cycle
// after the instr is retired, but there is no interlock in Sparc V8.
// This behavior is modeled with a forced noop after the instruction in
// DelaySlotFiller.

let Defs = [FCC] in {
  def FCMPS  : F3_3c<2, 0b110101, 0b001010001,
                   (outs), (ins FPRegs:$src1, FPRegs:$src2),
                   "fcmps $src1, $src2",
                   [(SPcmpfcc f32:$src1, f32:$src2)]>;
  def FCMPD  : F3_3c<2, 0b110101, 0b001010010,
                   (outs), (ins DFPRegs:$src1, DFPRegs:$src2),
                   "fcmpd $src1, $src2",
                   [(SPcmpfcc f64:$src1, f64:$src2)]>;
  def FCMPQ  : F3_3c<2, 0b110101, 0b001010011,
                   (outs), (ins QFPRegs:$src1, QFPRegs:$src2),
                   "fcmpq $src1, $src2",
                   [(SPcmpfcc f128:$src1, f128:$src2)]>,
                   Requires<[HasHardQuad]>;
}

//===----------------------------------------------------------------------===//
// Instructions for Thread Local Storage(TLS).
//===----------------------------------------------------------------------===//

def TLS_ADDrr : F3_1<2, 0b000000,
                    (outs IntRegs:$rd),
                    (ins IntRegs:$rs1, IntRegs:$rs2, TLSSym:$sym),
                    "add $rs1, $rs2, $rd, $sym",
                    [(set i32:$rd,
                        (tlsadd i32:$rs1, i32:$rs2, tglobaltlsaddr:$sym))]>;

let mayLoad = 1 in
  def TLS_LDrr : F3_1<3, 0b000000,
                      (outs IntRegs:$dst), (ins MEMrr:$addr, TLSSym:$sym),
                      "ld [$addr], $dst, $sym",
                      [(set i32:$dst,
                          (tlsld ADDRrr:$addr, tglobaltlsaddr:$sym))]>;

let Uses = [O6], isCall = 1, hasDelaySlot = 1 in
  def TLS_CALL : InstSP<(outs),
                        (ins calltarget:$disp, TLSSym:$sym, variable_ops),
                        "call $disp, $sym",
                        [(tlscall texternalsym:$disp, tglobaltlsaddr:$sym)]> {
  bits<30> disp;
  let op = 1;
  let Inst{29-0} = disp;
}

//===----------------------------------------------------------------------===//
// V9 Instructions
//===----------------------------------------------------------------------===//

// V9 Conditional Moves.
let Predicates = [HasV9], Constraints = "$f = $rd" in {
  // Move Integer Register on Condition (MOVcc) p. 194 of the V9 manual.
  let Uses = [ICC], cc = 0b100 in {
    def MOVICCrr
      : F4_1<0b101100, (outs IntRegs:$rd),
             (ins IntRegs:$rs2, IntRegs:$f, CCOp:$cond),
             "mov$cond %icc, $rs2, $rd",
             [(set i32:$rd, (SPselecticc i32:$rs2, i32:$f, imm:$cond))]>;

    def MOVICCri
      : F4_2<0b101100, (outs IntRegs:$rd),
             (ins i32imm:$simm11, IntRegs:$f, CCOp:$cond),
             "mov$cond %icc, $simm11, $rd",
             [(set i32:$rd,
                    (SPselecticc simm11:$simm11, i32:$f, imm:$cond))]>;
  }

  let Uses = [FCC], cc = 0b000 in {
    def MOVFCCrr
      : F4_1<0b101100, (outs IntRegs:$rd),
             (ins IntRegs:$rs2, IntRegs:$f, CCOp:$cond),
             "mov$cond %fcc0, $rs2, $rd",
             [(set i32:$rd, (SPselectfcc i32:$rs2, i32:$f, imm:$cond))]>;
    def MOVFCCri
      : F4_2<0b101100, (outs IntRegs:$rd),
             (ins i32imm:$simm11, IntRegs:$f, CCOp:$cond),
             "mov$cond %fcc0, $simm11, $rd",
             [(set i32:$rd,
                    (SPselectfcc simm11:$simm11, i32:$f, imm:$cond))]>;
  }

  let Uses = [ICC], opf_cc = 0b100 in {
    def FMOVS_ICC
      : F4_3<0b110101, 0b000001, (outs FPRegs:$rd),
             (ins FPRegs:$rs2, FPRegs:$f, CCOp:$cond),
             "fmovs$cond %icc, $rs2, $rd",
             [(set f32:$rd, (SPselecticc f32:$rs2, f32:$f, imm:$cond))]>;
    def FMOVD_ICC
      : F4_3<0b110101, 0b000010, (outs DFPRegs:$rd),
               (ins DFPRegs:$rs2, DFPRegs:$f, CCOp:$cond),
               "fmovd$cond %icc, $rs2, $rd",
               [(set f64:$rd, (SPselecticc f64:$rs2, f64:$f, imm:$cond))]>;
    def FMOVQ_ICC
      : F4_3<0b110101, 0b000011, (outs QFPRegs:$rd),
               (ins QFPRegs:$rs2, QFPRegs:$f, CCOp:$cond),
               "fmovd$cond %icc, $rs2, $rd",
               [(set f128:$rd, (SPselecticc f128:$rs2, f128:$f, imm:$cond))]>;
  }

  let Uses = [FCC], opf_cc = 0b000 in {
    def FMOVS_FCC
      : F4_3<0b110101, 0b000001, (outs FPRegs:$rd),
             (ins FPRegs:$rs2, FPRegs:$f, CCOp:$cond),
             "fmovs$cond %fcc0, $rs2, $rd",
             [(set f32:$rd, (SPselectfcc f32:$rs2, f32:$f, imm:$cond))]>;
    def FMOVD_FCC
      : F4_3<0b110101, 0b000010, (outs DFPRegs:$rd),
             (ins DFPRegs:$rs2, DFPRegs:$f, CCOp:$cond),
             "fmovd$cond %fcc0, $rs2, $rd",
             [(set f64:$rd, (SPselectfcc f64:$rs2, f64:$f, imm:$cond))]>;
    def FMOVQ_FCC
      : F4_3<0b110101, 0b000011, (outs QFPRegs:$rd),
             (ins QFPRegs:$rs2, QFPRegs:$f, CCOp:$cond),
             "fmovd$cond %fcc0, $rs2, $rd",
             [(set f128:$rd, (SPselectfcc f128:$rs2, f128:$f, imm:$cond))]>;
  }

}

// Floating-Point Move Instructions, p. 164 of the V9 manual.
let Predicates = [HasV9] in {
  def FMOVD : F3_3u<2, 0b110100, 0b000000010,
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fmovd $src, $dst", []>;
  def FMOVQ : F3_3u<2, 0b110100, 0b000000011,
                   (outs QFPRegs:$dst), (ins QFPRegs:$src),
                   "fmovq $src, $dst", []>,
                   Requires<[HasHardQuad]>;
  def FNEGD : F3_3u<2, 0b110100, 0b000000110,
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fnegd $src, $dst",
                   [(set f64:$dst, (fneg f64:$src))]>;
  def FNEGQ : F3_3u<2, 0b110100, 0b000000111,
                   (outs QFPRegs:$dst), (ins QFPRegs:$src),
                   "fnegq $src, $dst",
                   [(set f128:$dst, (fneg f128:$src))]>,
                   Requires<[HasHardQuad]>;
  def FABSD : F3_3u<2, 0b110100, 0b000001010,
                   (outs DFPRegs:$dst), (ins DFPRegs:$src),
                   "fabsd $src, $dst",
                   [(set f64:$dst, (fabs f64:$src))]>;
  def FABSQ : F3_3u<2, 0b110100, 0b000001011,
                   (outs QFPRegs:$dst), (ins QFPRegs:$src),
                   "fabsq $src, $dst",
                   [(set f128:$dst, (fabs f128:$src))]>,
                   Requires<[HasHardQuad]>;
}

// POPCrr - This does a ctpop of a 64-bit register.  As such, we have to clear
// the top 32-bits before using it.  To do this clearing, we use a SLLri X,0.
let rs1 = 0 in
  def POPCrr : F3_1<2, 0b101110,
                    (outs IntRegs:$dst), (ins IntRegs:$src),
                    "popc $src, $dst", []>, Requires<[HasV9]>;
def : Pat<(ctpop i32:$src),
          (POPCrr (SLLri $src, 0))>;

//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// Small immediates.
def : Pat<(i32 simm13:$val),
          (ORri (i32 G0), imm:$val)>;
// Arbitrary immediates.
def : Pat<(i32 imm:$val),
          (ORri (SETHIi (HI22 imm:$val)), (LO10 imm:$val))>;


// Global addresses, constant pool entries
def : Pat<(SPhi tglobaladdr:$in), (SETHIi tglobaladdr:$in)>;
def : Pat<(SPlo tglobaladdr:$in), (ORri (i32 G0), tglobaladdr:$in)>;
def : Pat<(SPhi tconstpool:$in), (SETHIi tconstpool:$in)>;
def : Pat<(SPlo tconstpool:$in), (ORri (i32 G0), tconstpool:$in)>;

// GlobalTLS addresses
def : Pat<(SPhi tglobaltlsaddr:$in), (SETHIi tglobaltlsaddr:$in)>;
def : Pat<(SPlo tglobaltlsaddr:$in), (ORri (i32 G0), tglobaltlsaddr:$in)>;
def : Pat<(add (SPhi tglobaltlsaddr:$in1), (SPlo tglobaltlsaddr:$in2)),
          (ADDri (SETHIi tglobaltlsaddr:$in1), (tglobaltlsaddr:$in2))>;
def : Pat<(xor (SPhi tglobaltlsaddr:$in1), (SPlo tglobaltlsaddr:$in2)),
          (XORri (SETHIi tglobaltlsaddr:$in1), (tglobaltlsaddr:$in2))>;

// Blockaddress
def : Pat<(SPhi tblockaddress:$in), (SETHIi tblockaddress:$in)>;
def : Pat<(SPlo tblockaddress:$in), (ORri (i32 G0), tblockaddress:$in)>;

// Add reg, lo.  This is used when taking the addr of a global/constpool entry.
def : Pat<(add iPTR:$r, (SPlo tglobaladdr:$in)), (ADDri $r, tglobaladdr:$in)>;
def : Pat<(add iPTR:$r, (SPlo tconstpool:$in)),  (ADDri $r, tconstpool:$in)>;
def : Pat<(add iPTR:$r, (SPlo tblockaddress:$in)),
                        (ADDri $r, tblockaddress:$in)>;

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

// store 0, addr -> store %g0, addr
def : Pat<(store (i32 0), ADDRrr:$dst), (STrr ADDRrr:$dst, (i32 G0))>;
def : Pat<(store (i32 0), ADDRri:$dst), (STri ADDRri:$dst, (i32 G0))>;

include "SparcInstr64Bit.td"
