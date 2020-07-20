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
def uimm6       : PatLeaf<(imm), [{ return isUInt<6>(N->getZExtValue()); }]>;

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

def simm7Op64 : Operand<i64> {
  let DecoderMethod = "DecodeSIMM7";
}

def simm32Op64 : Operand<i64> {
  let DecoderMethod = "DecodeSIMM32";
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

multiclass RRmrr<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi> {
  def rr : RR<opc, (outs RCo:$sx), (ins RCi:$sy, RCi:$sz),
              !strconcat(opcStr, " $sx, $sy, $sz")>
           { let cy = 1; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmri<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp> {
  // VE calculates (OpNode $sy, $sz), but llvm requires to have immediate
  // in RHS, so we use following definition.
  def ri : RR<opc, (outs RCo:$sx), (ins RCi:$sz, immOp:$sy),
              !strconcat(opcStr, " $sx, $sy, $sz")>
           { let cy = 0; let cz = 1; let hasSideEffects = 0; }
}

multiclass RRmiz<string opcStr, bits<8>opc,
                 RegisterClass RCo, ValueType Tyo,
                 RegisterClass RCi, ValueType Tyi, Operand immOp> {
  def zi : RR<opc, (outs RCo:$sx), (ins immOp:$sy),
              !strconcat(opcStr, " $sx, $sy")>
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
}

// Used by add, mul, div, and similar commutative instructions
//   The order of operands are "$sx, $sy, $sz"

multiclass RRm<string opcStr, bits<8>opc,
               RegisterClass RC, ValueType Ty, Operand immOp, Operand immOp2> :
  RRmrr<opcStr, opc, RC, Ty, RC, Ty>,
  RRmri<opcStr, opc, RC, Ty, RC, Ty, immOp>,
  RRmiz<opcStr, opc, RC, Ty, RC, Ty, immOp>,
  RRNDmrm<opcStr, opc, RC, Ty, RC, Ty, immOp2>;

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

// LEA and LEASL instruction (load 32 bit imm to low or high part)
let cx = 0 in
defm LEA : RMm<"lea", 0x06, I64, i64, simm7Op64, simm32Op64>;
let cx = 1 in
defm LEASL : RMm<"lea.sl", 0x06, I64, i64, simm7Op64, simm32Op64>;

// 5.3.2.2. Fixed-Point Arithmetic Operation Instructions

// ADX instruction
let cx = 0 in
defm ADX : RRm<"adds.l", 0x59, I64, i64, simm7Op64, uimm6Op64>;

// 5.3.2.3. Logical Arithmetic Operation Instructions

let cx = 0 in {
  defm AND : RRm<"and", 0x44, I64, i64, simm7Op64, uimm6Op64>;
  defm OR : RRm<"or", 0x45, I64, i64, simm7Op64, uimm6Op64>;
}

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
