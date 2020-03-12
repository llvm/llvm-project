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

// Addressing modes.
def ADDRrr : ComplexPattern<iPTR, 2, "SelectADDRrr", [], []>;
def ADDRri : ComplexPattern<iPTR, 2, "SelectADDRri", [frameindex], []>;

// ASX format of memory address
def MEMrr : Operand<iPTR> {
  let PrintMethod = "printMemASXOperand";
  let MIOperandInfo = (ops ptr_rc, ptr_rc);
}

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

def calltarget : Operand<i64> {
  let EncoderMethod = "getCallTargetOpValue";
  let DecoderMethod = "DecodeCall";
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

def VEhi    : SDNode<"VEISD::Hi", SDTIntUnaryOp>;
def VElo    : SDNode<"VEISD::Lo", SDTIntUnaryOp>;

//  These are target-independent nodes, but have target-specific formats.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i64>,
                                          SDTCisVT<1, i64> ]>;
def SDT_SPCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i64>,
                                        SDTCisVT<1, i64> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                           [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SDT_SPCall    : SDTypeProfile<0, -1, [SDTCisVT<0, i64>]>;
def call          : SDNode<"VEISD::CALL", SDT_SPCall,
                           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                            SDNPVariadic]>;

def retflag       : SDNode<"VEISD::RET_FLAG", SDTNone,
                           [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;

def getGOT        : Operand<iPTR>;

// GETFUNPLT for PIC
def GetFunPLT : SDNode<"VEISD::GETFUNPLT", SDTIntUnaryOp>;

// GETTLSADDR for TLS
def GetTLSAddr : SDNode<"VEISD::GETTLSADDR", SDT_SPCall,
                        [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
                         SDNPVariadic]>;



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
               RegisterClass RC, ValueType Ty,
               Operand immOp, Operand immOp2,
               SDPatternOperator OpNode=null_frag> {
  def rri : RM<
    opc, (outs RC:$sx), (ins RC:$sy, RC:$sz, immOp2:$imm32),
    !strconcat(opcStr, " $sx, ${imm32}($sy, ${sz})")> {
    let cy = 1;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def rzi : RM<
    opc, (outs RC:$sx), (ins RC:$sz, immOp2:$imm32),
    !strconcat(opcStr, " $sx, ${imm32}(${sz})"),
    [(set Ty:$sx, (OpNode Ty:$sz, (Ty simm32:$imm32)))]> {
    let cy = 0;
    let sy = 0;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def zii : RM<
    opc, (outs RC:$sx), (ins immOp:$sy, immOp2:$imm32),
    !strconcat(opcStr, " $sx, ${imm32}(${sy})"),
    [/* Not define DAG pattern here to avoid llvm uses LEAzii for all add
        instructions.
        (set Ty:$sx, (OpNode (Ty simm7:$sy), (Ty simm32:$imm32))) */]> {
    let cy = 0;
    let cz = 0;
    let sz = 0;
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

multiclass RRmrr<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi,
                 SDPatternOperator OpNode=null_frag> {
  def rr : RR<opc, (outs RCo:$sx), (ins RCi:$sy, RCi:$sz),
              !strconcat(opcStr, " $sx, $sy, $sz"),
              [(set Tyo:$sx, (OpNode Tyi:$sy, Tyi:$sz))]>
           { let cy = 1; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmri<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp,
                 SDPatternOperator OpNode=null_frag> {
  // VE calculates (OpNode $sy, $sz), but llvm requires to have immediate
  // in RHS, so we use following definition.
  def ri : RR<opc, (outs RCo:$sx), (ins RCi:$sz, immOp:$sy),
              !strconcat(opcStr, " $sx, $sy, $sz"),
              [(set Tyo:$sx, (OpNode Tyi:$sz, (Tyi simm7:$sy)))]>
           { let cy = 0; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmir<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp,
                 SDPatternOperator OpNode=null_frag> {
  def ri : RR<opc, (outs RCo:$sx), (ins immOp:$sy, RCi:$sz),
              !strconcat(opcStr, " $sx, $sy, $sz"),
              [(set Tyo:$sx, (OpNode (Tyi simm7:$sy), Tyi:$sz))]>
           { let cy = 0; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmiz<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp,
                 SDPatternOperator OpNode=null_frag> {
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

multiclass RRm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty,
               Operand immOp, Operand immOp2,
               SDPatternOperator OpNode=null_frag> :
  RRmrr<opcStr, opc, RC, Ty, RC, Ty, OpNode>,
  RRmri<opcStr, opc, RC, Ty, RC, Ty, immOp, OpNode>,
  RRmiz<opcStr, opc, RC, Ty, RC, Ty, immOp, OpNode>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>,
  RRNDmim<opcStr, opc, RC, Ty, RC, Ty, immOp, immOp2>;

// Used by sub, and similar not commutative instructions
//   The order of operands are "$sx, $sy, $sz"

multiclass RRNCm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2,
               SDPatternOperator OpNode=null_frag> :
  RRmrr<opcStr, opc, RC, Ty, RC, Ty, OpNode>,
  RRmir<opcStr, opc, RC, Ty, RC, Ty, immOp, OpNode>,
  RRmiz<opcStr, opc, RC, Ty, RC, Ty, immOp, OpNode>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>,
  RRNDmim<opcStr, opc, RC, Ty, RC, Ty, immOp, immOp2>;

// Used by fadd, fsub, and similar floating point instructions
//   The order of operands are "$sx, $sy, $sz"

multiclass RRFm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2,
               SDPatternOperator OpNode=null_frag> :
  RRmrr<opcStr, opc, RC, Ty, RC, Ty, OpNode>,
  RRmir<opcStr, opc, RC, Ty, RC, Ty, immOp, null_frag>,
  RRmiz<opcStr, opc, RC, Ty, RC, Ty, immOp, null_frag>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>,
  RRNDmim<opcStr, opc, RC, Ty, RC, Ty, immOp, immOp2>;

// Multiclass for RR type instructions
//   Used by sra, sla, sll, and similar instructions
//   The order of operands are "$sx, $sz, $sy"

multiclass RRIm<string opcStr, bits<8>opc,
                RegisterClass RC, ValueType Ty,
                Operand immOp, Operand immOp2,
                SDPatternOperator OpNode=null_frag> {
  def rr : RR<
    opc, (outs RC:$sx), (ins RC:$sz, I32:$sy),
    !strconcat(opcStr, " $sx, $sz, $sy"),
    [(set Ty:$sx, (OpNode Ty:$sz, i32:$sy))]> {
    let cy = 1;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def ri : RR<
    opc, (outs RC:$sx), (ins RC:$sz, immOp:$sy),
    !strconcat(opcStr, " $sx, $sz, $sy"),
    [(set Ty:$sx, (OpNode Ty:$sz, (i32 simm7:$sy)))]> {
    let cy = 0;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def rm0 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz, I32:$sy),
    !strconcat(opcStr, " $sx, (${sz})0, $sy")> {
    let cy = 1;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
  def rm1 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz, I32:$sy),
    !strconcat(opcStr, " $sx, (${sz})1, $sy")> {
    let cy = 1;
    let cz = 0;
    let hasSideEffects = 0;
  }
  def im0 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz, immOp:$sy),
    !strconcat(opcStr, " $sx, (${sz})0, $sy")> {
    let cy = 0;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
  def im1 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz, immOp:$sy),
    !strconcat(opcStr, " $sx, (${sz})1, $sy")> {
    let cy = 0;
    let cz = 0;
    let hasSideEffects = 0;
  }
  def zi : RR<
    opc, (outs RC:$sx), (ins immOp:$sy),
    !strconcat(opcStr, " $sx, $sy"),
    [(set Ty:$sx, (OpNode 0, (i32 simm7:$sy)))]> {
    let cy = 0;
    let cz = 0;
    let sz = 0;
    let hasSideEffects = 0;
  }
}

// Multiclass for RR type instructions
//   Used by cmov instruction

let Constraints = "$sx = $sd", DisableEncoding = "$sd" in
multiclass RRCMOVm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> {
  def rr : RR<
    opc, (outs I64:$sx), (ins CCOp:$cf, RC:$sy, I64:$sz, I64:$sd),
    !strconcat(opcStr, " $sx, $sz, $sy")> {
    let cy = 1;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def rm0 : RR<
    opc, (outs I64:$sx), (ins CCOp:$cf, RC:$sy, immOp2:$sz, I64:$sd),
    !strconcat(opcStr, " $sx, (${sz})0, $sy")> {
    let cy = 1;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
}

// Multiclass for RR type instructions with only 2 operands
//   Used by pcnt, brv
let hasSideEffects = 0 in
multiclass RRI2m<string opcStr, bits<8>opc, RegisterClass RC, ValueType Ty,
                 Operand immOp2, SDPatternOperator OpNode=null_frag> {
  def r : RR<
    opc, (outs RC:$sx), (ins RC:$sz),
    !strconcat(opcStr, " $sx, $sz"),
    [(set Ty:$sx, (OpNode Ty:$sz))]> {
    let cy = 1;
    let cz = 1;
  }
  def i : RR<
    opc, (outs RC:$sx), (ins RC:$sz),
    !strconcat(opcStr, " $sx, $sz"),
    [(set Ty:$sx, (OpNode Ty:$sz))]> {
    let cy = 0;
    let cz = 1;
  }
  def m0 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz),
    !strconcat(opcStr, " $sx, (${sz})0")> {
    let cy = 1;
    let cz = 0;
    let sz{6} = 1;
  }
  def m1 : RR<
    opc, (outs RC:$sx), (ins immOp2:$sz),
    !strconcat(opcStr, " $sx, (${sz})1")> {
    let cy = 1;
    let cz = 0;
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
  def ir : CF<
    opc, (outs),
    (ins CCOp:$cf, immOp:$sy, RC:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, $sz, $imm32")> {
    let cy = 0;
    let cz = 1;
    let hasSideEffects = 0;
  }
  def rm0 : CF<
    opc, (outs), (ins CCOp:$cf, RC:$sy, immOp2:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, (${sz})0, $imm32"), []> {
    let cy = 1;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
  def rm1 : CF<
    opc, (outs), (ins CCOp:$cf, RC:$sy, immOp2:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, (${sz})1, $imm32"), []> {
    let cy = 1;
    let cz = 0;
    let hasSideEffects = 0;
  }
  def im0 : CF<
    opc, (outs), (ins CCOp:$cf, immOp:$sy, immOp2:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, (${sz})0, $imm32"), []> {
    let cy = 0;
    let cz = 0;
    let sz{6} = 1;
    let hasSideEffects = 0;
  }
  def im1 : CF<
    opc, (outs), (ins CCOp:$cf, immOp:$sy, immOp2:$sz, brtarget32:$imm32),
    !strconcat(opcStr, " $sy, (${sz})1, $imm32"), []> {
    let cy = 0;
    let cz = 0;
    let hasSideEffects = 0;
  }
  def a : CF<
    opc, (outs), (ins brtarget32:$imm32),
    !strconcat(opcStrAt, " $imm32"), []> {
    let cy = 0;
    let sy = 0;
    let cz = 0;
    let sz = 0;
    let cf = 15;  /* AT */
    let isBarrier = 1;
    let hasSideEffects = 0;
  }
}

// Multiclass for floating point conversion instructions.
//   Used by CVS/CVD/FLT and others
multiclass CVTm<string opcStr, bits<8> opc,
                RegisterClass RCo, ValueType Tyo,
                RegisterClass RCi, ValueType Tyi, Operand immOp,
                SDPatternOperator OpNode=null_frag> {
  def r : RR<opc, (outs RCo:$sx), (ins RCi:$sy),
    !strconcat(opcStr, " $sx, $sy"),
    [(set Tyo:$sx, (OpNode Tyi:$sy))]> {
      let cy = 1;
      let hasSideEffects = 0;
    }
  def i : RR<opc, (outs RCo:$sx), (ins immOp:$sy),
    !strconcat(opcStr, " $sx, $sy"),
    [/* (set Tyo:$sx, (OpNode (Tyi simm7:$sy))) */]> {
      let cy = 0;
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
defm LEA : RMm<"lea", 0x06, I64, i64, simm7Op64, simm32Op64, add>;
let cx = 1 in
defm LEASL : RMm<"lea.sl", 0x06, I64, i64, simm7Op64, simm32Op64>;
let isCodeGenOnly = 1 in {
let cx = 0 in
defm LEA32 : RMm<"lea", 0x06, I32, i32, simm7Op32, simm32Op32, add>;
}

let cx = 0, cy = 1, cz = 0, sz = 0, hasSideEffects = 0 in {
  def LEAasx : RM<
      0x06, (outs I64:$sx), (ins MEMri:$addr),
      "lea $sx,$addr", [(set iPTR:$sx, ADDRri:$addr)]>;
}

// 5.3.2.2. Fixed-Point Arithmetic Operation Instructions

// ADD instruction
let cx = 0 in
defm ADD : RRm<"addu.l", 0x48, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm ADDUW : RRm<"addu.w", 0x48, I32, i32, simm7Op32, uimm6Op32>;

// ADS instruction
let cx = 0 in
defm ADS : RRm<"adds.w.sx", 0x4A, I32, i32, simm7Op32, uimm6Op32, add>;
let cx = 1 in
defm ADSU : RRm<"adds.w.zx", 0x4A, I32, i32, simm7Op32, uimm6Op32>;

// ADX instruction
let cx = 0 in
defm ADX : RRm<"adds.l", 0x59, I64, i64, simm7Op64, uimm6Op64, add>;

// SUB instruction
let cx = 0 in
defm SUB : RRm<"subu.l", 0x58, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm SUBUW : RRm<"subu.w", 0x58, I32, i32, simm7Op32, uimm6Op32>;

// SBS instruction
let cx = 0 in
defm SBS : RRNCm<"subs.w.sx", 0x5A, I32, i32, simm7Op32, uimm6Op32, sub>;
let cx = 1 in
defm SBSU : RRm<"subs.w.zx", 0x5A, I32, i32, simm7Op32, uimm6Op32>;

// SBX instruction
let cx = 0 in
defm SBX : RRNCm<"subs.l", 0x5B, I64, i64, simm7Op64, uimm6Op64, sub>;

// MPY instruction
let cx = 0 in
defm MPY : RRm<"mulu.l", 0x49, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm MPYUW : RRm<"mulu.w", 0x49, I32, i32, simm7Op32, uimm6Op32>;

// MPS instruction
let cx = 0 in
defm MPS : RRm<"muls.w.sx", 0x4B, I32, i32, simm7Op32, uimm6Op32, mul>;
let cx = 1 in
defm MPSU : RRm<"muls.w.zx", 0x4B, I32, i32, simm7Op32, uimm6Op32>;

// MPX instruction
let cx = 0 in
defm MPX : RRm<"muls.l", 0x6E, I64, i64, simm7Op64, uimm6Op64, mul>;

// DIV instruction
let cx = 0 in
defm DIV : RRNCm<"divu.l", 0x6F, I64, i64, simm7Op64, uimm6Op64, udiv>;
let cx = 1 in
defm DIVUW : RRNCm<"divu.w", 0x6F, I32, i32, simm7Op32, uimm6Op32, udiv>;

// DVS instruction
let cx = 0 in
defm DVS : RRNCm<"divs.w.sx", 0x7B, I32, i32, simm7Op32, uimm6Op32, sdiv>;
let cx = 1 in
defm DVSU : RRm<"divs.w.zx", 0x7B, I32, i32, simm7Op32, uimm6Op32>;

// DVX instruction
let cx = 0 in
defm DVX : RRNCm<"divs.l", 0x7F, I64, i64, simm7Op64, uimm6Op64, sdiv>;

// CMP instruction
let cx = 0 in
defm CMP : RRm<"cmpu.l", 0x55, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm CMPUW : RRm<"cmpu.w", 0x55, I32, i32, simm7Op32, uimm6Op32>;

// CPS instruction
let cx = 0 in
defm CPS : RRm<"cmps.w.sx", 0x7A, I32, i32, simm7Op32, uimm6Op32>;
let cx = 1 in
defm CPSU : RRm<"cmps.w.zx", 0x7A, I32, i32, simm7Op32, uimm6Op32>;

// CPX instruction
let cx = 0 in
defm CPX : RRm<"cmps.l", 0x6A, I64, i64, simm7Op64, uimm6Op64>;

// cx: sx/zx, cw: max/min

let cw = 0 in defm CMXa :
  RRm<"maxs.l", 0x68, I64, i64, simm7Op64, uimm6Op64>;

let cx = 0, cw = 0 in defm CMSa :
  RRm<"maxs.w.zx", 0x78, I32, i32, simm7Op32, uimm6Op32>;

let cw = 1 in defm CMXi :
  RRm<"mins.l", 0x68, I64, i64, simm7Op64, uimm6Op64>;

let cx = 1, cw = 0 in defm CMSi :
  RRm<"mins.w.zx", 0x78, I32, i32, simm7Op32, uimm6Op32>;

// 5.3.2.3. Logical Arithmetic Operation Instructions

let cx = 0 in {
  defm AND : RRm<"and", 0x44, I64, i64, simm7Op64, uimm6Op64, and>;
  defm OR : RRm<"or", 0x45, I64, i64, simm7Op64, uimm6Op64, or>;
  defm XOR : RRm<"xor", 0x46, I64, i64, simm7Op64, uimm6Op64, xor>;
  let isCodeGenOnly = 1 in {
    defm AND32 : RRm<"and", 0x44, I32, i32, simm7Op32, uimm6Op32, and>;
    defm OR32 : RRm<"or", 0x45, I32, i32, simm7Op32, uimm6Op32, or>;
    defm XOR32 : RRm<"xor", 0x46, I32, i32, simm7Op32, uimm6Op32, xor>;
  }
}

// Bits operations

let cx = 0 in {
  defm PCNT : RRI2m<"pcnt", 0x38, I64, i64, uimm6Op64, ctpop>;
  defm BRV : RRI2m<"brv", 0x39, I64, i64, uimm6Op64, bitreverse>;
  defm LDZ : RRI2m<"ldz", 0x67, I64, i64, uimm6Op64, ctlz>;
  defm BSWP : RRIm<"bswp", 0x2B, I64, i64, simm7Op64, uimm6Op64>;
}



// 5.3.2.4 Shift Instructions

let cx = 0 in
defm SRAX : RRIm<"sra.l", 0x77, I64, i64, simm7Op32, uimm6Op64, sra>;
let cx = 0 in
defm SRA : RRIm<"sra.w.sx", 0x76, I32, i32, simm7Op32, uimm6Op32, sra>;
let cx = 1 in
defm SRAU : RRIm<"sra.w.zx", 0x76, I32, i32, simm7Op32, uimm6Op32>;

let cx = 0 in
defm SLL : RRIm<"sll", 0x65, I64, i64, simm7Op32, uimm6Op64, shl>;
let cx = 0 in
defm SLA : RRIm<"sla.w.sx", 0x66, I32, i32, simm7Op32, uimm6Op32, shl>;
let cx = 1 in
defm SLAU : RRIm<"sla.w.zx", 0x66, I32, i32, simm7Op32, uimm6Op32>;
let cx = 0 in
defm SRL : RRIm<"srl", 0x75, I64, i64, simm7Op32, uimm6Op64, srl>;

def : Pat<(i32 (srl i32:$src, (i32 simm7:$val))),
          (EXTRACT_SUBREG (SRLri (ANDrm0 (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
            $src, sub_i32), 32), imm:$val), sub_i32)>;
def : Pat<(i32 (srl i32:$src, i32:$val)),
          (EXTRACT_SUBREG (SRLrr (ANDrm0 (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
            $src, sub_i32), 32), $val), sub_i32)>;

// 5.3.2.5. Floating-point Arithmetic Operation Instructions
let cx = 0 in
defm FAD : RRFm<"fadd.d", 0x4C, I64, f64, simm7Op64, uimm6Op64, fadd>;
let cx = 1 in
defm FADS : RRFm<"fadd.s", 0x4C, F32, f32, simm7Op32, uimm6Op32, fadd>;

let cx = 0 in
defm FSB : RRFm<"fsub.d", 0x5C, I64, f64, simm7Op64, uimm6Op64, fsub>;
let cx = 1 in
defm FSBS : RRFm<"fsub.s", 0x5C, F32, f32, simm7Op32, uimm6Op32, fsub>;

let cx = 0 in
defm FMP : RRFm<"fmul.d", 0x4D, I64, f64, simm7Op64, uimm6Op64, fmul>;
let cx = 1 in
defm FMPS : RRFm<"fmul.s", 0x4D, F32, f32, simm7Op32, uimm6Op32, fmul>;

let cx = 0 in
defm FDV : RRFm<"fdiv.d", 0x5D, I64, f64, simm7Op64, uimm6Op64, fdiv>;
let cx = 1 in
defm FDVS : RRFm<"fdiv.s", 0x5D, F32, f32, simm7Op32, uimm6Op32, fdiv>;

// FCP instruction
let cx = 0 in
defm FCP : RRm<"fcmp.d", 0x7E, I64, f64, simm7Op64, uimm6Op64>;
let cx = 1 in
defm FCPS : RRm<"fcmp.s", 0x7E, F32, f32, simm7Op32, uimm6Op32>;

// FCM
let cw = 0 in {
  let cx = 0 in
  defm FCMA : RRm<"fmax.d", 0x3E, I64, f64, simm7Op64, uimm6Op64>;
  let cx = 1 in
  defm FCMAS : RRm<"fmax.s", 0x3E, F32, f32, simm7Op32, uimm6Op32>;
}
let cw = 1 in {
  let cx = 0 in
  defm FCMI : RRm<"fmin.d", 0x3E, I64, f64, simm7Op64, uimm6Op64>;
  let cx = 1 in
  defm FCMIS : RRm<"fmin.s", 0x3E, F32, f32, simm7Op32, uimm6Op32>;
}

let cx = 0, cw = 0 /* sign extend */, cz = 1, sz = 0 /* round toward zero */ in
defm FIX : CVTm<"cvt.w.d.sx.rz", 0x4E, I32, i32, I64, f64, simm7Op32, fp_to_sint>;
let cx = 1, cw = 0 /* sign extend */, cz = 1, sz = 0 /* round toward zero */ in
defm FIXS : CVTm<"cvt.w.s.sx.rz", 0x4E, I32, i32, F32, f32, simm7Op32, fp_to_sint>;
let cx = 0, cz = 1, sz = 0 /* round toward zero */ in
defm FIXX : CVTm<"cvt.l.d.rz", 0x4F, I64, i64, I64, f64, simm7Op64, fp_to_sint>;
let cz = 0, sz = 0 in {
  let cx = 0 in
  defm FLT : CVTm<"cvt.d.w", 0x5E, I64, f64, I32, i32, simm7Op32, sint_to_fp>;
  let cx = 1 in
  defm FLTS : CVTm<"cvt.s.w", 0x5E, F32, f32, I32, i32, simm7Op32, sint_to_fp>;
  let cx = 0 in
  defm FLTX : CVTm<"cvt.d.l", 0x5F, I64, f64, I64, i64, simm7Op64, sint_to_fp>;
  let cx = 0 in
  defm CVS : CVTm<"cvt.s.d", 0x1F, F32, f32, I64, f64, simm7Op64, fpround>;
  let cx = 0 in
  defm CVD : CVTm<"cvt.d.s", 0x0F, I64, f64, F32, f32, simm7Op32, fpextend>;
}

// Load and Store instructions
// As 1st step, only uses sz and imm32 to represent $addr
let mayLoad = 1, hasSideEffects = 0 in {
let cy = 0, sy = 0, cz = 1 in {
let cx = 0 in
def LDSri : RM<
    0x01, (outs I64:$sx), (ins MEMri:$addr),
    "ld $sx, $addr",
    [(set i64:$sx, (load ADDRri:$addr))]>;
let cx = 0 in
def LDUri : RM<
    0x02, (outs F32:$sx), (ins MEMri:$addr),
    "ldu $sx, $addr",
    [(set f32:$sx, (load ADDRri:$addr))]>;
let cx = 0 in
def LDLri : RM<
    0x03, (outs I32:$sx), (ins MEMri:$addr),
    "ldl.sx $sx, $addr",
    [(set i32:$sx, (load ADDRri:$addr))]>;
let cx = 1 in
def LDLUri : RM<
    0x03, (outs I32:$sx), (ins MEMri:$addr),
    "ldl.zx $sx, $addr",
    [(set i32:$sx, (load ADDRri:$addr))]>;
let cx = 0 in
def LD2Bri : RM<
    0x04, (outs I32:$sx), (ins MEMri:$addr),
    "ld2b.sx $sx, $addr",
    [(set i32:$sx, (sextloadi16 ADDRri:$addr))]>;
let cx = 1 in
def LD2BUri : RM<
    0x04, (outs I32:$sx), (ins MEMri:$addr),
    "ld2b.zx $sx, $addr",
    [(set i32:$sx, (zextloadi16 ADDRri:$addr))]>;
let cx = 0 in
def LD1Bri : RM<
    0x05, (outs I32:$sx), (ins MEMri:$addr),
    "ld1b.sx $sx, $addr",
    [(set i32:$sx, (sextloadi8 ADDRri:$addr))]>;
let cx = 1 in
def LD1BUri : RM<
    0x05, (outs I32:$sx), (ins MEMri:$addr),
    "ld1b.zx $sx, $addr",
    [(set i32:$sx, (zextloadi8 ADDRri:$addr))]>;
}
}

let mayStore = 1, hasSideEffects = 0 in {
let cx = 0, cy = 0, sy = 0, cz = 1 in {
def STSri : RM<
    0x11, (outs), (ins MEMri:$addr, I64:$sx),
    "st $sx, $addr",
    [(store i64:$sx, ADDRri:$addr)]>;
def STUri : RM<
    0x12, (outs), (ins MEMri:$addr, F32:$sx),
    "stu $sx, $addr",
    [(store f32:$sx, ADDRri:$addr)]>;
def STLri : RM<
    0x13, (outs), (ins MEMri:$addr, I32:$sx),
    "stl $sx, $addr",
    [(store i32:$sx, ADDRri:$addr)]>;
def ST2Bri : RM<
    0x14, (outs), (ins MEMri:$addr, I32:$sx),
    "st2b $sx, $addr",
    [(truncstorei16 i32:$sx, ADDRri:$addr)]>;
def ST1Bri : RM<
    0x15, (outs), (ins MEMri:$addr, I32:$sx),
    "st1b $sx, $addr",
    [(truncstorei8 i32:$sx, ADDRri:$addr)]>;
}
}

def : Pat<(f64 (load ADDRri:$addr)), (LDSri ADDRri:$addr)>;
def : Pat<(store f64:$sx, ADDRri:$addr), (STSri ADDRri:$addr, $sx)>;

// Control-flow

// Jump instruction
let cx = 0, cx2 = 0, bpf = 0 /* NONE */, cy = 1, cz = 1,
    isBranch = 1, isTerminator = 1, hasDelaySlot = 1, hasSideEffects = 0 in
def BC : CF<
    0x19, (outs), (ins CCOp:$cf, I64:$sy, brtarget32:$imm32),
    "b.${cf}.l $sy, $imm32">;

// Jump always instruction is treated as a special case of jump in order
// to make finding unconditional jump easy.
let cx = 0, cx2 = 0, bpf = 0 /* NONE */, cf = 15 /* AT */, cy = 0, sy = 0,
    cz = 1,
    isBranch = 1, isTerminator = 1, isBarrier = 1, isIndirectBranch = 1,
    hasDelaySlot = 1, isCodeGenOnly = 1, hasSideEffects = 0 in {
def BArr : CF<
    0x19, (outs), (ins MEMrr:$addr),
    "b.l $addr",
    [(brind ADDRrr:$addr)]>;
def BAri : CF<
    0x19, (outs), (ins MEMri:$addr),
    "b.l $addr",
    [(brind ADDRri:$addr)]>;
}

// Jump never instruction is also a special case of jump.
let cx = 0, cx2 = 0, bpf = 0 /* NONE */, cf = 0 /* AF */, cy = 1, sy = 0,
    cz = 1,
    isBranch = 1, isTerminator = 1, hasDelaySlot = 1, hasSideEffects = 0 in
def BN : CF<
    0x19, (outs), (ins brtarget32:$imm32),
    "b.af.l $imm32">;

// Return instruction is also a special case of jump.
let cx = 0, cx2 = 0, bpf = 0 /* NONE */, cf = 15 /* AT */, cy = 0, sy = 0,
    cz = 1, sz = 0x10 /* SX10 */, imm32 = 0, Uses = [SX10],
    isReturn = 1, isTerminator = 1, hasDelaySlot = 1, isBarrier = 1,
    isCodeGenOnly = 1, hasSideEffects = 0 in
def RET : CF<
    0x19, (outs), (ins),
    "b.l (,%lr)",
    [(retflag)]>;

// Branch and Save IC

let cx = 0, cy = 0, cy = 0, cz = 1, hasSideEffects = 0 /* , Uses = [IC] */ in
def BSIC : RM<0x08, (outs), (ins I64:$sx, I64:$sz), "bsic $sx, (, ${sz})">;

// Branch instruction
let cx = 0, cx2 = 0, bpf = 0 /* NONE */ in
defm BCRL : BCRm<"br${cf}.l", "br.l", 0x18, I64, i64, simm7Op64, uimm6Op64>;
let cx = 1, cx2 = 0, bpf = 0 /* NONE */ in
defm BCRW : BCRm<"br${cf}.w", "br.w", 0x18, I32, i32, simm7Op32, uimm6Op32>;
let cx = 0, cx2 = 1, bpf = 0 /* NONE */ in
defm BCRD : BCRm<"br${cf}.d", "br.d", 0x18, I64, f64, simm7Op64, uimm6Op64>;
let cx = 1, cx2 = 1, bpf = 0 /* NONE */ in
defm BCRS : BCRm<"br${cf}.s", "br.s", 0x18, F32, f32, simm7Op32, uimm6Op32>;

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

// Save Instruction Counter

let cx = 0, cy = 0, sy = 0, cz = 0, sz = 0, hasSideEffects = 0 /* , Uses = [IC] */ in
def SIC : RR<0x28, (outs I32:$sx), (ins), "sic $sx">;

//===----------------------------------------------------------------------===//
// Instructions for CodeGenOnly
//===----------------------------------------------------------------------===//

let isCodeGenOnly = 1 in {

// Call instruction
let Defs = [SX10], Uses = [SX11], hasDelaySlot = 1, isCall = 1, hasSideEffects = 0 in {
let cx = 0, sx = 10, cy = 0, sy = 0, cz = 0, sz = 0 in
def CALL : RM<
    0x08, (outs), (ins calltarget:$imm32, variable_ops),
    "bsic %lr, $imm32">;
// use sz to represent a register
let cx = 0, sx = 10, cy = 0, sy = 0, cz = 1, imm32 = 0 in
def CALLr : RM<
    0x08, (outs), (ins I64:$sz, variable_ops),
    "bsic %lr, (,$sz)">;
}

}

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
def : Pat<(i64 (fp_to_sint f32:$sy)), (FIXXr (CVDr $sy))>;

// Cast to f32
def : Pat<(f32 (sint_to_fp i64:$sy)), (CVSr (FLTXr i64:$sy))>;

def : Pat<(i64 (anyext i32:$sy)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $sy, sub_i32)>;


// extload, sextload and zextload stuff
def : Pat<(i64 (sextloadi8 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD1Bri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (zextloadi8 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD1BUri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (sextloadi16 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD2Bri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (zextloadi16 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD2BUri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (sextloadi32 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LDLri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (zextloadi32 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LDLUri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (extloadi8 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD1BUri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (extloadi16 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LD2BUri MEMri:$addr), sub_i32)>;
def : Pat<(i64 (extloadi32 ADDRri:$addr)),
          (INSERT_SUBREG (i64 (IMPLICIT_DEF)), (LDLri MEMri:$addr), sub_i32)>;

// anyextload
def : Pat<(extloadi8  ADDRri:$addr), (LD1BUri MEMri:$addr)>;
def : Pat<(extloadi16 ADDRri:$addr), (LD2BUri MEMri:$addr)>;

// truncstore
def : Pat<(truncstorei8 i64:$src, ADDRri:$addr),
          (ST1Bri MEMri:$addr, (EXTRACT_SUBREG $src, sub_i32))>;
def : Pat<(truncstorei16 i64:$src, ADDRri:$addr),
          (ST2Bri MEMri:$addr, (EXTRACT_SUBREG $src, sub_i32))>;
def : Pat<(truncstorei32 i64:$src, ADDRri:$addr),
          (STLri MEMri:$addr, (EXTRACT_SUBREG $src, sub_i32))>;

// Address calculation and its optimization
def : Pat<(VEhi tglobaladdr:$in), (LEASLzzi tglobaladdr:$in)>;
def : Pat<(VElo tglobaladdr:$in), (ANDrm0 (LEAzzi tglobaladdr:$in), 32)>;
def : Pat<(add (VEhi tglobaladdr:$in1), (VElo tglobaladdr:$in2)),
          (LEASLrzi (ANDrm0 (LEAzzi tglobaladdr:$in2), 32),
                    (tglobaladdr:$in1))>;

// GlobalTLS address calculation and its optimization
def : Pat<(VEhi tglobaltlsaddr:$in), (LEASLzzi tglobaltlsaddr:$in)>;
def : Pat<(VElo tglobaltlsaddr:$in), (ANDrm0 (LEAzzi tglobaltlsaddr:$in), 32)>;
def : Pat<(add (VEhi tglobaltlsaddr:$in1), (VElo tglobaltlsaddr:$in2)),
          (LEASLrzi (ANDrm0 (LEAzzi tglobaltlsaddr:$in2), 32),
                    (tglobaltlsaddr:$in1))>;

// Address calculation and its optimization
def : Pat<(VEhi texternalsym:$in), (LEASLzzi texternalsym:$in)>;
def : Pat<(VElo texternalsym:$in), (ANDrm0 (LEAzzi texternalsym:$in), 32)>;
def : Pat<(add (VEhi texternalsym:$in1), (VElo texternalsym:$in2)),
          (LEASLrzi (ANDrm0 (LEAzzi texternalsym:$in2), 32),
                    (texternalsym:$in1))>;

// Calls
def : Pat<(call tglobaladdr:$dst),
          (CALL tglobaladdr:$dst)>;
def : Pat<(call i64:$dst),
          (CALLr i64:$dst)>;

// Branches
def : Pat<(br bb:$addr), (BCRLa bb:$addr)>;

// brcc
def : Pat<(brcc CCSIOp:$cond, i32:$l, i32:$r, bb:$addr),
          (BCRWrr (icond2cc $cond), $l, $r, bb:$addr)>;
def : Pat<(brcc CCUIOp:$cond, i32:$l, i32:$r, bb:$addr),
          (BCRWir (icond2cc $cond), 0, (CMPUWrr $r, $l), bb:$addr)>;
def : Pat<(brcc CCSIOp:$cond, i64:$l, i64:$r, bb:$addr),
          (BCRLrr (icond2cc $cond), $l, $r, bb:$addr)>;
def : Pat<(brcc CCUIOp:$cond, i64:$l, i64:$r, bb:$addr),
          (BCRLir (icond2cc $cond), 0, (CMPrr $r, $l), bb:$addr)>;
def : Pat<(brcc cond:$cond, f32:$l, f32:$r, bb:$addr),
          (BCRSrr (fcond2cc $cond), $l, $r, bb:$addr)>;
def : Pat<(brcc cond:$cond, f64:$l, f64:$r, bb:$addr),
          (BCRDrr (fcond2cc $cond), $l, $r, bb:$addr)>;

//===----------------------------------------------------------------------===//
// Pseudo Instructions
//===----------------------------------------------------------------------===//

// GETGOT for PIC
let Defs = [SX15 /* %got */, SX16 /* %plt */], hasSideEffects = 0 in {
  def GETGOT : Pseudo<(outs getGOT:$getpcseq), (ins), "$getpcseq">;
}

// GETFUNPLT for PIC
let hasSideEffects = 0 in
def GETFUNPLT : Pseudo<(outs I64:$dst), (ins i64imm:$addr),
                       "$dst, $addr",
                       [(set iPTR:$dst, (GetFunPLT tglobaladdr:$addr))] >;

def : Pat<(GetFunPLT tglobaladdr:$dst),
          (GETFUNPLT tglobaladdr:$dst)>;
def : Pat<(GetFunPLT texternalsym:$dst),
          (GETFUNPLT texternalsym:$dst)>;

// GETTLSADDR for TLS
let Defs = [SX0, SX10, SX12], hasSideEffects = 0 in
def GETTLSADDR : Pseudo<(outs), (ins i64imm:$addr),
                        "# GETTLSADDR $addr",
                        [(GetTLSAddr tglobaltlsaddr:$addr)] >;

def : Pat<(GetTLSAddr tglobaltlsaddr:$dst),
          (GETTLSADDR tglobaltlsaddr:$dst)>;

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

// Special SELECTCC pattern matches
// Use min/max for better performance.
//
//   MAX/MIN  %res, %lhs, %rhs

def : Pat<(f64 (selectcc f64:$LHS, f64:$RHS, f64:$LHS, f64:$RHS, SETOGT)),
          (FCMArr $LHS, $RHS)>;
def : Pat<(f32 (selectcc f32:$LHS, f32:$RHS, f32:$LHS, f32:$RHS, SETOGT)),
          (FCMASrr $LHS, $RHS)>;
def : Pat<(i64 (selectcc i64:$LHS, i64:$RHS, i64:$LHS, i64:$RHS, SETGT)),
          (CMXarr $LHS, $RHS)>;
def : Pat<(i32 (selectcc i32:$LHS, i32:$RHS, i32:$LHS, i32:$RHS, SETGT)),
          (CMSarr $LHS, $RHS)>;
def : Pat<(f64 (selectcc f64:$LHS, f64:$RHS, f64:$LHS, f64:$RHS, SETOGE)),
          (FCMArr $LHS, $RHS)>;
def : Pat<(f32 (selectcc f32:$LHS, f32:$RHS, f32:$LHS, f32:$RHS, SETOGE)),
          (FCMASrr $LHS, $RHS)>;
def : Pat<(i64 (selectcc i64:$LHS, i64:$RHS, i64:$LHS, i64:$RHS, SETGE)),
          (CMXarr $LHS, $RHS)>;
def : Pat<(i32 (selectcc i32:$LHS, i32:$RHS, i32:$LHS, i32:$RHS, SETGE)),
          (CMSarr $LHS, $RHS)>;

def : Pat<(f64 (selectcc f64:$LHS, f64:$RHS, f64:$LHS, f64:$RHS, SETOLT)),
          (FCMIrr $LHS, $RHS)>;
def : Pat<(f32 (selectcc f32:$LHS, f32:$RHS, f32:$LHS, f32:$RHS, SETOLT)),
          (FCMISrr $LHS, $RHS)>;
def : Pat<(i64 (selectcc i64:$LHS, i64:$RHS, i64:$LHS, i64:$RHS, SETLT)),
          (CMXirr $LHS, $RHS)>;
def : Pat<(i32 (selectcc i32:$LHS, i32:$RHS, i32:$LHS, i32:$RHS, SETLT)),
          (CMSirr $LHS, $RHS)>;
def : Pat<(f64 (selectcc f64:$LHS, f64:$RHS, f64:$LHS, f64:$RHS, SETOLE)),
          (FCMIrr $LHS, $RHS)>;
def : Pat<(f32 (selectcc f32:$LHS, f32:$RHS, f32:$LHS, f32:$RHS, SETOLE)),
          (FCMISrr $LHS, $RHS)>;
def : Pat<(i64 (selectcc i64:$LHS, i64:$RHS, i64:$LHS, i64:$RHS, SETLE)),
          (CMXirr $LHS, $RHS)>;
def : Pat<(i32 (selectcc i32:$LHS, i32:$RHS, i32:$LHS, i32:$RHS, SETLE)),
          (CMSirr $LHS, $RHS)>;

// Generic SELECTCC pattern matches
//
//   CMP  %tmp, %l, %r       ; compare %l and %r
//   or   %res, %f, (0)1     ; initialize by %f
//   CMOV %res, %t, %tmp     ; set %t if %tmp is true

// selectcc for i64 result
def : Pat<(i64 (selectcc i32:$l, i32:$r, i64:$t, i64:$f, CCSIOp:$cond)),
          (CMOVWrr (icond2cc $cond), (CPSrr $l, $r), $t, $f)>;
def : Pat<(i64 (selectcc i32:$l, i32:$r, i64:$t, i64:$f, CCUIOp:$cond)),
          (CMOVWrr (icond2cc $cond), (CMPUWrr $l, $r), $t, $f)>;
def : Pat<(i64 (selectcc i64:$l, i64:$r, i64:$t, i64:$f, CCSIOp:$cond)),
          (CMOVLrr (icond2cc $cond), (CPXrr $l, $r), $t, $f)>;
def : Pat<(i64 (selectcc i64:$l, i64:$r, i64:$t, i64:$f, CCUIOp:$cond)),
          (CMOVLrr (icond2cc $cond), (CMPrr $l, $r), $t, $f)>;
def : Pat<(i64 (selectcc f32:$l, f32:$r, i64:$t, i64:$f, cond:$cond)),
          (CMOVSrr (fcond2cc $cond), (FCPSrr $l, $r), $t, $f)>;
def : Pat<(i64 (selectcc f64:$l, f64:$r, i64:$t, i64:$f, cond:$cond)),
          (CMOVDrr (fcond2cc $cond), (FCPrr $l, $r), $t, $f)>;

// selectcc for i32 result
def : Pat<(i32 (selectcc i32:$l, i32:$r, i32:$t, i32:$f, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrr (icond2cc $cond),
                       (CPSrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;
def : Pat<(i32 (selectcc i32:$l, i32:$r, i32:$t, i32:$f, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrr (icond2cc $cond),
                       (CMPUWrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;
def : Pat<(i32 (selectcc i64:$l, i64:$r, i32:$t, i32:$f, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrr (icond2cc $cond),
                       (CPXrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;
def : Pat<(i32 (selectcc i64:$l, i64:$r, i32:$t, i32:$f, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrr (icond2cc $cond),
                       (CMPrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;
def : Pat<(i32 (selectcc f32:$l, f32:$r, i32:$t, i32:$f, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVSrr (fcond2cc $cond),
                       (FCPSrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;
def : Pat<(i32 (selectcc f64:$l, f64:$r, i32:$t, i32:$f, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVDrr (fcond2cc $cond),
                       (FCPrr $l, $r),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;

// selectcc for f64 result
def : Pat<(f64 (selectcc i32:$l, i32:$r, f64:$t, f64:$f, CCSIOp:$cond)),
          (CMOVWrr (icond2cc $cond), (CPSrr $l, $r), $t, $f)>;
def : Pat<(f64 (selectcc i32:$l, i32:$r, f64:$t, f64:$f, CCUIOp:$cond)),
          (CMOVWrr (icond2cc $cond), (CMPUWrr $l, $r), $t, $f)>;
def : Pat<(f64 (selectcc i64:$l, i64:$r, f64:$t, f64:$f, CCSIOp:$cond)),
          (CMOVLrr (icond2cc $cond), (CPXrr $l, $r), $t, $f)>;
def : Pat<(f64 (selectcc i64:$l, i64:$r, f64:$t, f64:$f, CCUIOp:$cond)),
          (CMOVLrr (icond2cc $cond), (CMPrr $l, $r), $t, $f)>;
def : Pat<(f64 (selectcc f32:$l, f32:$r, f64:$t, f64:$f, cond:$cond)),
          (CMOVSrr (fcond2cc $cond), (FCPSrr $l, $r), $t, $f)>;
def : Pat<(f64 (selectcc f64:$l, f64:$r, f64:$t, f64:$f, cond:$cond)),
          (CMOVDrr (fcond2cc $cond), (FCPrr $l, $r), $t, $f)>;

// selectcc for f32 result
def : Pat<(f32 (selectcc i32:$l, i32:$r, f32:$t, f32:$f, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrr (icond2cc $cond),
                       (CPSrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;
def : Pat<(f32 (selectcc i32:$l, i32:$r, f32:$t, f32:$f, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVWrr (icond2cc $cond),
                       (CMPUWrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;
def : Pat<(f32 (selectcc i64:$l, i64:$r, f32:$t, f32:$f, CCSIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrr (icond2cc $cond),
                       (CPXrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;
def : Pat<(f32 (selectcc i64:$l, i64:$r, f32:$t, f32:$f, CCUIOp:$cond)),
          (EXTRACT_SUBREG
              (CMOVLrr (icond2cc $cond),
                       (CMPrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;
def : Pat<(f32 (selectcc f32:$l, f32:$r, f32:$t, f32:$f, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVSrr (fcond2cc $cond),
                       (FCPSrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;
def : Pat<(f32 (selectcc f64:$l, f64:$r, f32:$t, f32:$f, cond:$cond)),
          (EXTRACT_SUBREG
              (CMOVDrr (fcond2cc $cond),
                       (FCPrr $l, $r),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $t, sub_f32),
                       (INSERT_SUBREG (f64 (IMPLICIT_DEF)), $f, sub_f32)),
              sub_f32)>;

// Generic SELECT pattern matches
// Use cmov.w for all cases since %pred holds i32.
//
//   CMOV.w.ne %res, %tval, %tmp  ; set tval if %tmp is true

def : Pat<(i64 (select i32:$pred, i64:$t, i64:$f)),
          (CMOVWrr CC_INE, $pred, $t, $f)>;

def : Pat<(i32 (select i32:$pred, i32:$t, i32:$f)),
          (EXTRACT_SUBREG
              (CMOVWrr CC_INE, $pred,
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_i32),
                       (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_i32)),
              sub_i32)>;

def : Pat<(f64 (select i32:$pred, f64:$t, f64:$f)),
          (CMOVWrr CC_INE, $pred, $t, $f)>;

def : Pat<(f32 (select i32:$pred, f32:$t, f32:$f)),
          (EXTRACT_SUBREG
            (CMOVWrr CC_INE, $pred,
                     (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $t, sub_f32),
                     (INSERT_SUBREG (i64 (IMPLICIT_DEF)), $f, sub_f32)),
            sub_f32)>;

// bitconvert
def : Pat<(f64 (bitconvert i64:$src)), (COPY_TO_REGCLASS $src, I64)>;
def : Pat<(i64 (bitconvert f64:$src)), (COPY_TO_REGCLASS $src, I64)>;

def : Pat<(i32 (bitconvert f32:$op)),
          (EXTRACT_SUBREG (SRAXri (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
            $op, sub_f32), 32), sub_i32)>;
def : Pat<(f32 (bitconvert i32:$op)),
          (EXTRACT_SUBREG (SLLri (INSERT_SUBREG (i64 (IMPLICIT_DEF)),
            $op, sub_i32), 32), sub_f32)>;

// Bits operations pattern matchings.
def : Pat<(i32 (ctpop i32:$src)),
          (EXTRACT_SUBREG (PCNTr (ANDrm0 (INSERT_SUBREG
            (i64 (IMPLICIT_DEF)), $src, sub_i32), 32)), sub_i32)>;
def : Pat<(i32 (ctlz i32:$src)),
          (EXTRACT_SUBREG (LDZr (SLLri (INSERT_SUBREG
            (i64 (IMPLICIT_DEF)), $src, sub_i32), 32)), sub_i32)>;
def : Pat<(i64 (bswap i64:$src)),
          (BSWPri $src, 0)>;
def : Pat<(i32 (bswap i32:$src)),
          (EXTRACT_SUBREG (BSWPri (INSERT_SUBREG
            (i64 (IMPLICIT_DEF)), $src, sub_i32), 1), sub_i32)>;

// Several special pattern matches to optimize code

def : Pat<(i32 (and i32:$lhs, 0xff)),
          (AND32rm0 $lhs, 56)>;
def : Pat<(i32 (and i32:$lhs, 0xffff)),
          (AND32rm0 $lhs, 48)>;
def : Pat<(i32 (and i32:$lhs, 0xffffffff)),
          (AND32rm0 $lhs, 32)>;
