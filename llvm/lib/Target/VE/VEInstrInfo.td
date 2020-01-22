//===-- VEInstrInfo.td - Target Description for VE Target -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the VE instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction format superclass
//===----------------------------------------------------------------------===//

include "VEInstrFormats.td"

//===----------------------------------------------------------------------===//
// Feature predicates.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Pattern Stuff
//===----------------------------------------------------------------------===//

def simm7       : PatLeaf<(imm), [{ return isInt<7>(N->getSExtValue()); }]>;
def simm32      : PatLeaf<(imm), [{ return isInt<32>(N->getSExtValue()); }]>;
def uimm32      : PatLeaf<(imm), [{ return isUInt<32>(N->getZExtValue()); }]>;
def uimm6       : PatLeaf<(imm), [{ return isUInt<6>(N->getZExtValue()); }]>;
def lomsbzero   : PatLeaf<(imm), [{ return (N->getZExtValue() & 0x80000000)
                                      == 0; }]>;
def lozero      : PatLeaf<(imm), [{ return (N->getZExtValue() & 0xffffffff)
                                      == 0; }]>;
def fplomsbzero : PatLeaf<(fpimm), [{ return (N->getValueAPF().bitcastToAPInt()
                                      .getZExtValue() & 0x80000000) == 0; }]>;
def fplozero    : PatLeaf<(fpimm), [{ return (N->getValueAPF().bitcastToAPInt()
                                      .getZExtValue() & 0xffffffff) == 0; }]>;

def CCSIOp : PatLeaf<(cond), [{
  switch (N->get()) {
  default:          return true;
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETUGT:
  case ISD::SETUGE: return false;
  }
}]>;

def CCUIOp : PatLeaf<(cond), [{
  switch (N->get()) {
  default:         return true;
  case ISD::SETLT:
  case ISD::SETLE:
  case ISD::SETGT:
  case ISD::SETGE: return false;
  }
}]>;

def LOFP32 : SDNodeXForm<fpimm, [{
  // Get a integer immediate from fpimm
  const APInt& imm = N->getValueAPF().bitcastToAPInt();
  return CurDAG->getTargetConstant(Lo_32(imm.getZExtValue() & 0xffffffff),
                                   SDLoc(N), MVT::i64);
}]>;

def HIFP32 : SDNodeXForm<fpimm, [{
  // Get a integer immediate from fpimm
  const APInt& imm = N->getValueAPF().bitcastToAPInt();
  return CurDAG->getTargetConstant(Hi_32(imm.getZExtValue()),
                                   SDLoc(N), MVT::i64);
}]>;

def LO32 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(Lo_32(N->getZExtValue()),
                                   SDLoc(N), MVT::i64);
}]>;

def HI32 : SDNodeXForm<imm, [{
  // Transformation function: shift the immediate value down into the low bits.
  return CurDAG->getTargetConstant(Hi_32(N->getZExtValue()),
                                   SDLoc(N), MVT::i32);
}]>;

def icond2cc : SDNodeXForm<cond, [{
  VECC::CondCodes cc;
  switch (N->get()) {
  default:          llvm_unreachable("Unknown integer condition code!");
  case ISD::SETEQ:  cc = VECC::CC_IEQ; break;
  case ISD::SETNE:  cc = VECC::CC_INE; break;
  case ISD::SETLT:  cc = VECC::CC_IL;  break;
  case ISD::SETGT:  cc = VECC::CC_IG;  break;
  case ISD::SETLE:  cc = VECC::CC_ILE; break;
  case ISD::SETGE:  cc = VECC::CC_IGE; break;
  case ISD::SETULT: cc = VECC::CC_IL;  break;
  case ISD::SETULE: cc = VECC::CC_ILE; break;
  case ISD::SETUGT: cc = VECC::CC_IG;  break;
  case ISD::SETUGE: cc = VECC::CC_IGE; break;
  }
  return CurDAG->getTargetConstant(cc, SDLoc(N), MVT::i32);
}]>;

def fcond2cc : SDNodeXForm<cond, [{
  VECC::CondCodes cc;
  switch (N->get()) {
  default:          llvm_unreachable("Unknown float condition code!");
  case ISD::SETFALSE: cc = VECC::CC_AF;    break;
  case ISD::SETEQ:
  case ISD::SETOEQ:   cc = VECC::CC_EQ;    break;
  case ISD::SETNE:
  case ISD::SETONE:   cc = VECC::CC_NE;    break;
  case ISD::SETLT:
  case ISD::SETOLT:   cc = VECC::CC_L;     break;
  case ISD::SETGT:
  case ISD::SETOGT:   cc = VECC::CC_G;     break;
  case ISD::SETLE:
  case ISD::SETOLE:   cc = VECC::CC_LE;    break;
  case ISD::SETGE:
  case ISD::SETOGE:   cc = VECC::CC_GE;    break;
  case ISD::SETO:     cc = VECC::CC_NUM;   break;
  case ISD::SETUO:    cc = VECC::CC_NAN;   break;
  case ISD::SETUEQ:   cc = VECC::CC_EQNAN; break;
  case ISD::SETUNE:   cc = VECC::CC_NENAN; break;
  case ISD::SETULT:   cc = VECC::CC_LNAN;  break;
  case ISD::SETUGT:   cc = VECC::CC_GNAN;  break;
  case ISD::SETULE:   cc = VECC::CC_LENAN; break;
  case ISD::SETUGE:   cc = VECC::CC_GENAN; break;
  case ISD::SETTRUE:  cc = VECC::CC_AT;    break;
  }
  return CurDAG->getTargetConstant(cc, SDLoc(N), MVT::i32);
}]>;

// ASX format of memory address
def MEMri : Operand<iPTR> {
  let PrintMethod = "printMemASXOperand";
  let MIOperandInfo = (ops ptr_rc, i64imm);
}

// AS format of memory address
def MEMASri : Operand<iPTR> {
  let PrintMethod = "printMemASOperand";
  let MIOperandInfo = (ops ptr_rc, i64imm);
}

// Branch targets have OtherVT type.
def brtarget32 : Operand<OtherVT> {
  let EncoderMethod = "getBranchTarget32OpValue";
}

def simm7Op32 : Operand<i32> {
  let DecoderMethod = "DecodeSIMM7";
}

def simm7Op64 : Operand<i64> {
  let DecoderMethod = "DecodeSIMM7";
}

def simm32Op32 : Operand<i32> {
  let DecoderMethod = "DecodeSIMM32";
}

def simm32Op64 : Operand<i64> {
  let DecoderMethod = "DecodeSIMM32";
}

def uimm6Op32 : Operand<i32> {
  let DecoderMethod = "DecodeUIMM6";
}

def uimm6Op64 : Operand<i64> {
  let DecoderMethod = "DecodeUIMM6";
}

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i32>;

//  These are target-independent nodes, but have target-specific formats.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i64>,
                                          SDTCisVT<1, i64> ]>;
def SDT_SPCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i64>,
                                        SDTCisVT<1, i64> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

// def SDT_SPCall    : SDTypeProfile<0, -1, [SDTCisVT<0, i64>]>;

def retflag       : SDNode<"VEISD::RET_FLAG", SDTNone,
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
//===----------------------------------------------------------------------===//
// VE Flag Conditions
//===----------------------------------------------------------------------===//

// Note that these values must be kept in sync with the CCOp::CondCode enum
// values.
class CC_VAL<int N> : PatLeaf<(i32 N)>;
def CC_IG    : CC_VAL< 0>;  // Greater
def CC_IL    : CC_VAL< 1>;  // Less
def CC_INE   : CC_VAL< 2>;  // Not Equal
def CC_IEQ   : CC_VAL< 3>;  // Equal
def CC_IGE   : CC_VAL< 4>;  // Greater or Equal
def CC_ILE   : CC_VAL< 5>;  // Less or Equal
def CC_AF    : CC_VAL< 6>;  // Always false
def CC_G     : CC_VAL< 7>;  // Greater
def CC_L     : CC_VAL< 8>;  // Less
def CC_NE    : CC_VAL< 9>;  // Not Equal
def CC_EQ    : CC_VAL<10>;  // Equal
def CC_GE    : CC_VAL<11>;  // Greater or Equal
def CC_LE    : CC_VAL<12>;  // Less or Equal
def CC_NUM   : CC_VAL<13>;  // Number
def CC_NAN   : CC_VAL<14>;  // NaN
def CC_GNAN  : CC_VAL<15>;  // Greater or NaN
def CC_LNAN  : CC_VAL<16>;  // Less or NaN
def CC_NENAN : CC_VAL<17>;  // Not Equal or NaN
def CC_EQNAN : CC_VAL<18>;  // Equal or NaN
def CC_GENAN : CC_VAL<19>;  // Greater or Equal or NaN
def CC_LENAN : CC_VAL<20>;  // Less or Equal or NaN
def CC_AT    : CC_VAL<21>;  // Always true

//===----------------------------------------------------------------------===//
// VE Multiclasses for common instruction formats
//===----------------------------------------------------------------------===//

multiclass RMm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> {
  def rri : RM<
    opc, (outs RC:$sx), (ins RC:$sy, RC:$sz, immOp2:$imm32),
    !strconcat(opcStr, " $sx, ${imm32}($sy, ${sz})")> {
    let cy = 1;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def rzi : RM<
    opc, (outs RC:$sx), (ins RC:$sz, immOp2:$imm32),
    !strconcat(opcStr, " $sx, ${imm32}(${sz})")> {
    let cy = 0;
    let sy = 0;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def zzi : RM<
    opc, (outs RC:$sx), (ins immOp2:$imm32),
    !strconcat(opcStr, " $sx, $imm32")> {
    let cy = 0;
    let sy = 0;
    let cz = 0;
    let sz = 0;
    let hasSideEffects = 0;
  }
}

// Multiclass for RR type instructions

multiclass RRmrr<string opcStr, bits<8>opc, SDNode OpNode,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi> {
  def rr : RR<opc, (outs RCo:$sx), (ins RCi:$sy, RCi:$sz),
              !strconcat(opcStr, " $sx, $sy, $sz"),
              [(set Tyo:$sx, (OpNode Tyi:$sy, Tyi:$sz))]>
           { let cy = 1; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRNDmrr<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi> {
  def rr : RR<opc, (outs RCo:$sx), (ins RCi:$sy, RCi:$sz),
              !strconcat(opcStr, " $sx, $sy, $sz")>
           { let cy = 1; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmri<string opcStr, bits<8>opc, SDNode OpNode,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp> {
  // VE calculates (OpNode $sy, $sz), but llvm requires to have immediate
  // in RHS, so we use following definition.
  def ri : RR<opc, (outs RCo:$sx), (ins RCi:$sz, immOp:$sy),
              !strconcat(opcStr, " $sx, $sy, $sz"),
              [(set Tyo:$sx, (OpNode Tyi:$sz, (Tyi simm7:$sy)))]>
           { let cy = 0; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmiz<string opcStr, bits<8>opc, SDNode OpNode,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp> {
  def zi : RR<opc, (outs RCo:$sx), (ins immOp:$sy),
              !strconcat(opcStr, " $sx, $sy"),
              [(set Tyo:$sx, (OpNode (Tyi simm7:$sy), 0))]>
           { let cy = 0; let cz = 0; let sz = 0; let hasSideEffects = 0; }
}

multiclass RRNDmrm<string opcStr, bits<8>opc,
                   RegisterClass RCo, ValueType Tyo,
                   RegisterClass RCi, ValueType Tyi, Operand immOp2> {
  def rm0 : RR<opc, (outs RCo:$sx), (ins RCi:$sy, immOp2:$sz),
               !strconcat(opcStr, " $sx, $sy, (${sz})0")> {
              let cy = 1;
              let cz = 0;
              let sz{6} = 1;
              // (guess) tblgen conservatively assumes hasSideEffects when
              // it fails to infer from a pattern.
              let hasSideEffects = 0;
            }
  def rm1 : RR<opc, (outs RCo:$sx), (ins RCi:$sy, immOp2:$sz),
               !strconcat(opcStr, " $sx, $sy, (${sz})1")> {
              let cy = 1;
              let cz = 0;
              let hasSideEffects = 0;
            }
}

multiclass RRNDmim<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi,
                 Operand immOp, Operand immOp2> {
  def im1 : RR<opc, (outs RCo:$sx), (ins immOp:$sy, immOp2:$sz),
               !strconcat(opcStr, " $sx, $sy, (${sz})1")> {
              let cy = 0;
              let cz = 0;
              let hasSideEffects = 0;
            }
}

// Used by add, mul, div, and similar commutative instructions
//   The order of operands are "$sx, $sy, $sz"

multiclass RRm<string opcStr, bits<8>opc, SDNode OpNode,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> :
  RRmrr<opcStr, opc, OpNode, RC, Ty, RC, Ty>,
  RRmri<opcStr, opc, OpNode, RC, Ty, RC, Ty, immOp>,
  RRmiz<opcStr, opc, OpNode, RC, Ty, RC, Ty, immOp>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>,
  RRNDmim<opcStr, opc, RC, Ty, RC, Ty, immOp, immOp2>;

// Used by cmp instruction
//   The order of operands are "$sx, $sy, $sz"

multiclass RRNDm<string opcStr, bits<8>opc,
                 RegisterClass RC, ValueType Ty,
                 Operand immOp, Operand immOp2> :
  RRNDmrr<opcStr, opc, RC, Ty, RC, Ty>,
  //RRNDmir<opcStr, opc, RC, Ty, RC, Ty, immOp>,
  //RRNDmiz<opcStr, opc, RC, Ty, RC, Ty, immOp>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>,
  RRNDmim<opcStr, opc, RC, Ty, RC, Ty, immOp, immOp2>;

// Multiclass for RR type instructions
//   Used by sra, sla, sll, and similar instructions
//   The order of operands are "$sx, $sz, $sy"

multiclass RRIm<string opcStr, bits<8>opc, SDNode OpNode,
                RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> {
  def ri : RR<
    opc, (outs RC:$sx), (ins RC:$sz, immOp:$sy),
    !strconcat(opcStr, " $sx, $sz, $sy"),
    [(set Ty:$sx, (OpNode Ty:$sz, (i32 simm7:$sy)))]> {
    let cy = 0;
    let cz = 1;
    let hasSideEffects = 0;
  }
}

// Multiclass for RR type instructions
//   Used by cmov instruction

let Constraints = "$sx = $sd", DisableEncoding = "$sd" in
multiclass RRCMOVm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> {
  def rm0 : RR<
    opc, (outs I64:$sx), (ins CCOp:$cf, RC:$sy, immOp2:$sz, I64:$sd),
    !strconcat(opcStr, " $sx, (${sz})0, $sy")> {
    let cy = 1;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
}

// Branch multiclass
let isBranch = 1, isTerminator = 1, hasDelaySlot = 1 in
multiclass BCRm<string opcStr, string opcStrAt, bits<8> opc,
                RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> {
  def rr : CF<
    opc, (outs),
    (ins CCOp:$cf, RC:$sy, RC:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, $sz, $imm32")> {
    let cy = 1;
    let cz = 1;
    let hasSideEffects = 0;
  }
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

// CMOV instructions
let cx = 0, cw = 0, cw2 = 0 in
defm CMOVL : RRCMOVm<"cmov.l.${cf}", 0x3B, I64, i64, simm7Op64, uimm6Op64>;

let cx = 0, cw = 1, cw2 = 0 in
defm CMOVW : RRCMOVm<"cmov.w.${cf}", 0x3B, I32, i32, simm7Op64, uimm6Op32>;

let cx = 0, cw = 0, cw2 = 1 in
defm CMOVD : RRCMOVm<"cmov.d.${cf}", 0x3B, I64, f64, simm7Op64, uimm6Op64>;

let cx = 0, cw = 1, cw2 = 1 in
defm CMOVS : RRCMOVm<"cmov.s.${cf}", 0x3B, F32, f32, simm7Op64, uimm6Op32>;


// LEA and LEASL instruction (load 32 bit imm to low or high part)
let cx = 0 in
defm LEA : RMm<"lea", 0x06, I64, i64, simm7Op64, simm32Op64>;
let cx = 1 in
defm LEASL : RMm<"lea.sl", 0x06, I64, i64, simm7Op64, simm32Op64>;
let isCodeGenOnly = 1 in {
let cx = 0 in
defm LEA32 : RMm<"lea", 0x06, I32, i32, simm7Op32, simm32Op32>;
}


// 5.3.2.2. Fixed-Point Arithmetic Operation Instructions

// ADS instruction
let cx = 0 in
defm ADS : RRm<"adds.w.sx", 0x4A, add, I32, i32, simm7Op32, uimm6Op32>;
let cx = 1 in
defm ADSU : RRm<"adds.w.zx", 0x4A, add, I32, i32, simm7Op32, uimm6Op32>;

// ADX instruction
let cx = 0 in
defm ADX : RRm<"adds.l", 0x59, add, I64, i64, simm7Op64, uimm6Op64>;

// CMP instruction
let cx = 0 in
defm CMP : RRNDm<"cmpu.l", 0x55, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm CMPUW : RRNDm<"cmpu.w", 0x55, I32, i32, simm7Op32, uimm6Op32>;

// CPS instruction
let cx = 0 in
defm CPS : RRNDm<"cmps.w.sx", 0x7A, I32, i32, simm7Op32, uimm6Op32>;
let cx = 1 in
defm CPSU : RRNDm<"cmps.w.zx", 0x7A, I32, i32, simm7Op32, uimm6Op32>;

// CPX instruction
let cx = 0 in
defm CPX : RRNDm<"cmps.l", 0x6A, I64, i64, simm7Op64, uimm6Op64>;


// 5.3.2.3. Logical Arithmetic Operation Instructions

let cx = 0 in {
  defm AND : RRm<"and", 0x44, and, I64, i64, simm7Op64, uimm6Op64>;
  defm OR : RRm<"or", 0x45, or, I64, i64, simm7Op64, uimm6Op64>;
  let isCodeGenOnly = 1 in {
    defm AND32 : RRm<"and", 0x44, and, I32, i32, simm7Op32, uimm6Op32>;
    defm OR32 : RRm<"or", 0x45, or, I32, i32, simm7Op32, uimm6Op32>;
    defm XOR32 : RRm<"xor", 0x46, xor, I32, i32, simm7Op32, uimm6Op32>;
  }
}


// 5.3.2.4 Shift Instructions

let cx = 0 in
defm SRAX : RRIm<"sra.l", 0x77, sra, I64, i64, simm7Op32, uimm6Op64>;
let cx = 0 in
defm SRA : RRIm<"sra.w.sx", 0x76, sra, I32, i32, simm7Op32, uimm6Op32>;

let cx = 0 in
defm SLL : RRIm<"sll", 0x65, shl, I64, i64, simm7Op32, uimm6Op64>;
let cx = 0 in
defm SLA : RRIm<"sla.w.sx", 0x66, shl, I32, i32, simm7Op32, uimm6Op32>;

// FCP instruction
let cx = 0 in
defm FCP : RRNDm<"fcmp.d", 0x7E, I64, f64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm FCPS : RRNDm<"fcmp.s", 0x7E, F32, f32, simm7Op32, uimm6Op32>;

// Load and Store instructions
// As 1st step, only uses sz and imm32 to represent $addr
let mayLoad = 1, hasSideEffects = 0 in {
let cy = 0, sy = 0, cz = 1 in {
let cx = 0 in
def LDSri : RM<
    0x01, (outs I64:$sx), (ins MEMri:$addr),
    "ld $sx, $addr">;
}
}

let mayStore = 1, hasSideEffects = 0 in {
let cx = 0, cy = 0, sy = 0, cz = 1 in {
def STSri : RM<
    0x11, (outs), (ins MEMri:$addr, I64:$sx),
    "st $sx, $addr">;
}
}

// Return instruction is also a special case of jump.
let cx = 0, cx2 = 0, bpf = 0 /* NONE */, cf = 15 /* AT */, cy = 0, sy = 0,
    cz = 1, sz = 0x10 /* SX10 */, imm32 = 0, Uses = [SX10],
    isReturn = 1, isTerminator = 1, hasDelaySlot = 1, isBarrier = 1,
    isCodeGenOnly = 1, hasSideEffects = 0 in
def RET : CF<
    0x19, (outs), (ins),
    "b.l (,%lr)",
    [(retflag)]>;

// Branch instruction
let cx = 0, cx2 = 0, bpf = 0 /* NONE */ in
defm BCRL : BCRm<"br${cf}.l", "br.l", 0x18, I64, i64, simm7Op64, uimm6Op64>;

let cx = 0, cy = 0, cz = 1, hasSideEffects = 0 in {
let sy = 3 in
def SHMri : RM<
    0x31, (outs), (ins MEMASri:$addr, I64:$sx),
    "shm.l $sx, $addr">;
}

let cx = 0, sx = 0, cy = 0, sy = 0, cz = 0, sz = 0, hasSideEffects = 0 in
def MONC : RR<
    0x3F, (outs), (ins),
    "monc">;

//===----------------------------------------------------------------------===//
// Pattern Matchings
//===----------------------------------------------------------------------===//

// Small immediates.
def : Pat<(i32 simm7:$val), (OR32im1 imm:$val, 0)>;
def : Pat<(i64 simm7:$val), (ORim1 imm:$val, 0)>;
// Medium immediates.
def : Pat<(i32 simm32:$val), (LEA32zzi imm:$val)>;
def : Pat<(i64 simm32:$val), (LEAzzi imm:$val)>;
def : Pat<(i64 uimm32:$val), (ANDrm0 (LEAzzi imm:$val), 32)>;
// Arbitrary immediates.
def : Pat<(i64 lozero:$val),
          (LEASLzzi (HI32 imm:$val))>;
def : Pat<(i64 lomsbzero:$val),
          (LEASLrzi (LEAzzi (LO32 imm:$val)), (HI32 imm:$val))>;
def : Pat<(i64 imm:$val),
          (LEASLrzi (ANDrm0 (LEAzzi (LO32 imm:$val)), 32),
                    (HI32 imm:$val))>;

// floating point
def : Pat<(f32 fpimm:$val),
          (COPY_TO_REGCLASS (LEASLzzi (LOFP32 $val)), F32)>;
def : Pat<(f64 fplozero:$val),
          (LEASLzzi (HIFP32 $val))>;
def : Pat<(f64 fplomsbzero:$val),
          (LEASLrzi (LEAzzi (LOFP32 $val)), (HIFP32 $val))>;
def : Pat<(f64 fpimm:$val),
          (LEASLrzi (ANDrm0 (LEAzzi (LOFP32 $val)), 32),
                    (HIFP32 $val))>;

// The same integer registers are used for i32 and i64 values.
// When registers hold i32 values, the high bits are unused. 

// TODO Use standard expansion for shift-based lowering of sext_inreg

// Cast to i1
def : Pat<(sext_inreg I32:$src, i1),
          (SRAri (SLAri $src, 31), 31)>;
def : Pat<(sext_inreg I64:$src, i1),
          (SRAXri (SLLri $src, 63), 63)>;

// Cast to i8
def : Pat<(sext_inreg I32:$src, i8),
          (SRAri (SLAri $src, 24), 24)>;
def : Pat<(sext_inreg I64:$src, i8),
          (SRAXri (SLLri $src, 56), 56)>;
def : Pat<(sext_inreg (i32 (trunc i64:$src)), i8),
          (EXTRACT_SUBREG (SRAXri (SLLri $src, 56), 56), sub_i32)>;
def : Pat<(and (trunc i64:$src), 0xff),
          (AND32rm0 (EXTRACT_SUBREG $src, sub_i32), 56)>;

// Cast to i16
def : Pat<(sext_inreg I32:$src, i16),
          (SRAri (SLAri $src, 16), 16)>;
def : Pat<(sext_inreg I64:$src, i16),
          (SRAXri (SLLri $src, 48), 48)>;
def : Pat<(sext_inreg (i32 (trunc i64:$src)), i16),
          (EXTRACT_SUBREG (SRAXri (SLLri $src, 48), 48), sub_i32)>;
def : Pat<(and (trunc i64:$src), 0xffff),
          (AND32rm0 (EXTRACT_SUBREG $src, sub_i32), 48)>;

// Cast to i32
def : Pat<(i32 (trunc i64:$src)),
          (ADSrm1 (EXTRACT_SUBREG $src, sub_i32), 0)>;

// Cast to i64
def : Pat<(sext_inreg I64:$src, i32),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
            (ADSrm1 (EXTRACT_SUBREG $src, sub_i32), 0), sub_i32)>;
def : Pat<(i64 (sext i32:$sy)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (ADSrm1 $sy, 0), sub_i32)>;
def : Pat<(i64 (zext i32:$sy)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (ADSUrm1 $sy, 0), sub_i32)>;

def : Pat<(i64 (anyext i32:$sy)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $sy, sub_i32)>;



//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

let Defs = [SX11], Uses = [SX11], hasSideEffects = 0 in {
def ADJCALLSTACKDOWN : Pseudo<(outs), (ins i64imm:$amt, i64imm:$amt2),
                              "# ADJCALLSTACKDOWN $amt, $amt2",
                              [(callseq_start timm:$amt, timm:$amt2)]>;
def ADJCALLSTACKUP : Pseudo<(outs), (ins i64imm:$amt1, i64imm:$amt2),
                            "# ADJCALLSTACKUP $amt1",
                            [(callseq_end timm:$amt1, timm:$amt2)]>;
}

let Defs = [SX8], Uses = [SX8, SX11], hasSideEffects = 0 in
def EXTEND_STACK : Pseudo<(outs), (ins),
                          "# EXTEND STACK",
                          []>;
let  hasSideEffects = 0 in
def EXTEND_STACK_GUARD : Pseudo<(outs), (ins),
                                "# EXTEND STACK GUARD",
                                []>;

// SETCC pattern matches
//
//   CMP  %tmp, lhs, rhs     ; compare lhs and rhs
//   or   %res, 0, (0)1      ; initialize by 0
//   CMOV %res, (63)0, %tmp  ; set 1 if %tmp is true

def : Pat<(i32 (setcc i64:$LHS, i64:$RHS, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrm0 (icond2cc $cond),
                        (CPXrr i64:$LHS, i64:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;

def : Pat<(i32 (setcc i64:$LHS, i64:$RHS, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrm0 (icond2cc $cond),
                        (CMPrr i64:$LHS, i64:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;

def : Pat<(i32 (setcc i32:$LHS, i32:$RHS, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrm0 (icond2cc $cond),
                        (CPSrr i32:$LHS, i32:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;

def : Pat<(i32 (setcc i32:$LHS, i32:$RHS, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrm0 (icond2cc $cond),
                        (CMPUWrr i32:$LHS, i32:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;

def : Pat<(i32 (setcc f64:$LHS, f64:$RHS, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVDrm0 (fcond2cc $cond),
                        (FCPrr f64:$LHS, f64:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;

def : Pat<(i32 (setcc f32:$LHS, f32:$RHS, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVSrm0 (fcond2cc $cond),
                        (FCPSrr f32:$LHS, f32:$RHS),
                        63,
                        (ORim1 0, 0)), sub_i32)>;


// Several special pattern matches to optimize code

def : Pat<(i32 (and i32:$lhs, 0xff)),
          (AND32rm0 $lhs, 56)>;
def : Pat<(i32 (and i32:$lhs, 0xffff)),
          (AND32rm0 $lhs, 48)>;
def : Pat<(i32 (and i32:$lhs, 0xffffffff)),
          (AND32rm0 $lhs, 32)>;
