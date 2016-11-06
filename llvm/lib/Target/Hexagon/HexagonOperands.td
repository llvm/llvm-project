//===- HexagonImmediates.td - Hexagon immediate processing -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illnois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def s32_0ImmOperand : AsmOperandClass { let Name = "s32_0Imm"; }
def s23_2ImmOperand : AsmOperandClass { let Name = "s23_2Imm"; }
def s8_0ImmOperand : AsmOperandClass { let Name = "s8_0Imm"; }
def s8_0Imm64Operand : AsmOperandClass { let Name = "s8_0Imm64"; }
def s6_0ImmOperand : AsmOperandClass { let Name = "s6_0Imm"; }
def s4_0ImmOperand : AsmOperandClass { let Name = "s4_0Imm"; }
def s4_1ImmOperand : AsmOperandClass { let Name = "s4_1Imm"; }
def s4_2ImmOperand : AsmOperandClass { let Name = "s4_2Imm"; }
def s4_3ImmOperand : AsmOperandClass { let Name = "s4_3Imm"; }
def s4_6ImmOperand : AsmOperandClass { let Name = "s4_6Imm"; }
def s3_6ImmOperand : AsmOperandClass { let Name = "s3_6Imm"; }
def u64_0ImmOperand : AsmOperandClass { let Name = "u64_0Imm"; }
def u32_0ImmOperand : AsmOperandClass { let Name = "u32_0Imm"; }
def u26_6ImmOperand : AsmOperandClass { let Name = "u26_6Imm"; }
def u16_0ImmOperand : AsmOperandClass { let Name = "u16_0Imm"; }
def u16_1ImmOperand : AsmOperandClass { let Name = "u16_1Imm"; }
def u16_2ImmOperand : AsmOperandClass { let Name = "u16_2Imm"; }
def u16_3ImmOperand : AsmOperandClass { let Name = "u16_3Imm"; }
def u11_3ImmOperand : AsmOperandClass { let Name = "u11_3Imm"; }
def u10_0ImmOperand : AsmOperandClass { let Name = "u10_0Imm"; }
def u9_0ImmOperand : AsmOperandClass { let Name = "u9_0Imm"; }
def u8_0ImmOperand : AsmOperandClass { let Name = "u8_0Imm"; }
def u7_0ImmOperand : AsmOperandClass { let Name = "u7_0Imm"; }
def u6_0ImmOperand : AsmOperandClass { let Name = "u6_0Imm"; }
def u6_1ImmOperand : AsmOperandClass { let Name = "u6_1Imm"; }
def u6_2ImmOperand : AsmOperandClass { let Name = "u6_2Imm"; }
def u6_3ImmOperand : AsmOperandClass { let Name = "u6_3Imm"; }
def u5_0ImmOperand : AsmOperandClass { let Name = "u5_0Imm"; }
def u4_0ImmOperand : AsmOperandClass { let Name = "u4_0Imm"; }
def u3_0ImmOperand : AsmOperandClass { let Name = "u3_0Imm"; }
def u2_0ImmOperand : AsmOperandClass { let Name = "u2_0Imm"; }
def u1_0ImmOperand : AsmOperandClass { let Name = "u1_0Imm"; }
def n8_0ImmOperand : AsmOperandClass { let Name = "n8_0Imm"; }
// Immediate operands.

let OperandType = "OPERAND_IMMEDIATE",
    DecoderMethod = "unsignedImmDecoder" in {
  def s32_0Imm : Operand<i32> { let ParserMatchClass = s32_0ImmOperand;
                                let DecoderMethod = "s32_0ImmDecoder"; }
  def s23_2Imm : Operand<i32> { let ParserMatchClass = s23_2ImmOperand; }
  def s8_0Imm : Operand<i32> { let ParserMatchClass = s8_0ImmOperand;
                               let DecoderMethod = "s8_0ImmDecoder"; }
  def s8_0Imm64 : Operand<i64>  { let ParserMatchClass = s8_0Imm64Operand;
                                  let DecoderMethod = "s8_0ImmDecoder"; }
  def s6_0Imm : Operand<i32> { let ParserMatchClass = s6_0ImmOperand;
                             let DecoderMethod = "s6_0ImmDecoder"; }
  def s6_3Imm : Operand<i32>;
  def s4_0Imm : Operand<i32> { let ParserMatchClass = s4_0ImmOperand;
                               let DecoderMethod = "s4_0ImmDecoder"; }
  def s4_1Imm : Operand<i32> { let ParserMatchClass = s4_1ImmOperand;
                               let DecoderMethod = "s4_1ImmDecoder"; }
  def s4_2Imm : Operand<i32> { let ParserMatchClass = s4_2ImmOperand;
                               let DecoderMethod = "s4_2ImmDecoder"; }
  def s4_3Imm : Operand<i32> { let ParserMatchClass = s4_3ImmOperand;
                               let DecoderMethod = "s4_3ImmDecoder"; }
  def u64_0Imm : Operand<i64> { let ParserMatchClass = u64_0ImmOperand; }
  def u32_0Imm : Operand<i32> { let ParserMatchClass = u32_0ImmOperand; }
  def u26_6Imm : Operand<i32> { let ParserMatchClass = u26_6ImmOperand; }
  def u16_0Imm : Operand<i32> { let ParserMatchClass = u16_0ImmOperand; }
  def u16_1Imm : Operand<i32> { let ParserMatchClass = u16_1ImmOperand; }
  def u16_2Imm : Operand<i32> { let ParserMatchClass = u16_2ImmOperand; }
  def u16_3Imm : Operand<i32> { let ParserMatchClass = u16_3ImmOperand; }
  def u11_3Imm : Operand<i32> { let ParserMatchClass = u11_3ImmOperand; }
  def u10_0Imm : Operand<i32> { let ParserMatchClass = u10_0ImmOperand; }
  def u9_0Imm : Operand<i32> { let ParserMatchClass = u9_0ImmOperand; }
  def u8_0Imm : Operand<i32> { let ParserMatchClass = u8_0ImmOperand; }
  def u7_0Imm : Operand<i32> { let ParserMatchClass = u7_0ImmOperand; }
  def u6_0Imm : Operand<i32> { let ParserMatchClass = u6_0ImmOperand; }
  def u6_1Imm : Operand<i32> { let ParserMatchClass = u6_1ImmOperand; }
  def u6_2Imm : Operand<i32> { let ParserMatchClass = u6_2ImmOperand; }
  def u6_3Imm : Operand<i32> { let ParserMatchClass = u6_3ImmOperand; }
  def u5_0Imm : Operand<i32> { let ParserMatchClass = u5_0ImmOperand; }
  def u5_1Imm : Operand<i32>;
  def u5_2Imm : Operand<i32>;
  def u5_3Imm : Operand<i32>;
  def u4_0Imm : Operand<i32> { let ParserMatchClass = u4_0ImmOperand; }
  def u4_1Imm : Operand<i32>;
  def u4_2Imm : Operand<i32>;
  def u4_3Imm : Operand<i32>;
  def u3_0Imm : Operand<i32> { let ParserMatchClass = u3_0ImmOperand; }
  def u3_1Imm : Operand<i32>;
  def u3_2Imm : Operand<i32>;
  def u3_3Imm : Operand<i32>;
  def u2_0Imm : Operand<i32> { let ParserMatchClass = u2_0ImmOperand; }
  def u1_0Imm : Operand<i32> { let ParserMatchClass = u1_0ImmOperand; }
  def n8_0Imm : Operand<i32> { let ParserMatchClass = n8_0ImmOperand; }
}

let OperandType = "OPERAND_IMMEDIATE" in {
  def s4_6Imm : Operand<i32> { let ParserMatchClass = s4_6ImmOperand;
                               let PrintMethod = "prints4_6ImmOperand";
                               let DecoderMethod = "s4_6ImmDecoder";}
  def s4_7Imm : Operand<i32> { let PrintMethod = "prints4_7ImmOperand";
                               let DecoderMethod = "s4_6ImmDecoder";}
  def s3_6Imm : Operand<i32> { let ParserMatchClass = s3_6ImmOperand;
                               let PrintMethod = "prints3_6ImmOperand";
                               let DecoderMethod = "s3_6ImmDecoder";}
  def s3_7Imm : Operand<i32> { let PrintMethod = "prints3_7ImmOperand";
                               let DecoderMethod = "s3_6ImmDecoder";}
}

//
// Immediate predicates
//
def s32_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<32>(v);
}]>;

def s31_1ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<31,1>(v);
}]>;

def s30_2ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<30,2>(v);
}]>;

def s29_3ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<29,3>(v);
}]>;

def s10_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<10>(v);
}]>;

def s8_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<8>(v);
}]>;

def s8_0Imm64Pred  : PatLeaf<(i64 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<8>(v);
}]>;

def s6_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<6>(v);
}]>;

def s4_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isInt<4>(v);
}]>;

def s4_1ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,1>(v);
}]>;

def s4_2ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,2>(v);
}]>;

def s4_3ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<4,3>(v);
}]>;

def u32_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<32>(v);
}]>;

def u16_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<16>(v);
}]>;

def u11_3ImmPred : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<11,3>(v);
}]>;

def u9_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<9>(v);
}]>;

def u8_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<8>(v);
}]>;

def u7_0StrictPosImmPred : ImmLeaf<i32, [{
  // u7_0StrictPosImmPred predicate - True if the immediate fits in an 7-bit
  // unsigned field and is strictly greater than 0.
  return isUInt<7>(Imm) && Imm > 0;
}]>;

def u6_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<6>(v);
}]>;

def u6_1ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<6,1>(v);
}]>;

def u6_2ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedUInt<6,2>(v);
}]>;

def u5_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<5>(v);
}]>;

def u4_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<4>(v);
}]>;

def u3_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<3>(v);
}]>;

def u2_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isUInt<2>(v);
}]>;

def m5_0ImmPred  : PatLeaf<(i32 imm), [{
  // m5_0ImmPred predicate - True if the number is in range -1 .. -31
  // and will fit in a 5 bit field when made positive, for use in memops.
  int64_t v = (int64_t)N->getSExtValue();
  return (-31 <= v && v <= -1);
}]>;

//InN means negative integers in [-(2^N - 1), 0]
def n8_0ImmPred  : PatLeaf<(i32 imm), [{
  // n8_0ImmPred predicate - True if the immediate fits in a 8-bit signed
  // field.
  int64_t v = (int64_t)N->getSExtValue();
  return (-255 <= v && v <= 0);
}]>;

def nOneImmPred  : PatLeaf<(i32 imm), [{
  // nOneImmPred predicate - True if the immediate is -1.
  int64_t v = (int64_t)N->getSExtValue();
  return (-1 == v);
}]>;


// Extendable immediate operands.
def f32ExtOperand : AsmOperandClass { let Name = "f32Ext"; }
def s16_0ExtOperand : AsmOperandClass { let Name = "s16_0Ext"; }
def s12_0ExtOperand : AsmOperandClass { let Name = "s12_0Ext"; }
def s10_0ExtOperand : AsmOperandClass { let Name = "s10_0Ext"; }
def s9_0ExtOperand : AsmOperandClass { let Name = "s9_0Ext"; }
def s8_0ExtOperand : AsmOperandClass { let Name = "s8_0Ext"; }
def s7_0ExtOperand : AsmOperandClass { let Name = "s7_0Ext"; }
def s6_0ExtOperand : AsmOperandClass { let Name = "s6_0Ext"; }
def s11_0ExtOperand : AsmOperandClass { let Name = "s11_0Ext"; }
def s11_1ExtOperand : AsmOperandClass { let Name = "s11_1Ext"; }
def s11_2ExtOperand : AsmOperandClass { let Name = "s11_2Ext"; }
def s11_3ExtOperand : AsmOperandClass { let Name = "s11_3Ext"; }
def u6_0ExtOperand : AsmOperandClass { let Name = "u6_0Ext"; }
def u7_0ExtOperand : AsmOperandClass { let Name = "u7_0Ext"; }
def u8_0ExtOperand : AsmOperandClass { let Name = "u8_0Ext"; }
def u9_0ExtOperand : AsmOperandClass { let Name = "u9_0Ext"; }
def u10_0ExtOperand : AsmOperandClass { let Name = "u10_0Ext"; }
def u6_1ExtOperand : AsmOperandClass { let Name = "u6_1Ext"; }
def u6_2ExtOperand : AsmOperandClass { let Name = "u6_2Ext"; }
def u6_3ExtOperand : AsmOperandClass { let Name = "u6_3Ext"; }
def u32_0MustExtOperand : AsmOperandClass { let Name = "u32_0MustExt"; }



let OperandType = "OPERAND_IMMEDIATE", PrintMethod = "printExtOperand",
    DecoderMethod = "unsignedImmDecoder" in {
  def f32Ext : Operand<f32> { let ParserMatchClass = f32ExtOperand; }
  def s16_0Ext : Operand<i32> { let ParserMatchClass = s16_0ExtOperand;
                                let DecoderMethod = "s16_0ImmDecoder"; }
  def s12_0Ext : Operand<i32> { let ParserMatchClass = s12_0ExtOperand;
                                let DecoderMethod = "s12_0ImmDecoder"; }
  def s11_0Ext : Operand<i32> { let ParserMatchClass = s11_0ExtOperand;
                                let DecoderMethod = "s11_0ImmDecoder"; }
  def s11_1Ext : Operand<i32> { let ParserMatchClass = s11_1ExtOperand;
                                let DecoderMethod = "s11_1ImmDecoder"; }
  def s11_2Ext : Operand<i32> { let ParserMatchClass = s11_2ExtOperand;
                                let DecoderMethod = "s11_2ImmDecoder"; }
  def s11_3Ext : Operand<i32> { let ParserMatchClass = s11_3ExtOperand;
                                let DecoderMethod = "s11_3ImmDecoder"; }
  def s10_0Ext : Operand<i32> { let ParserMatchClass = s10_0ExtOperand;
                                let DecoderMethod = "s10_0ImmDecoder"; }
  def s9_0Ext : Operand<i32> { let ParserMatchClass = s9_0ExtOperand;
                               let DecoderMethod = "s9_0ImmDecoder"; }
  def s8_0Ext : Operand<i32> { let ParserMatchClass = s8_0ExtOperand;
                               let DecoderMethod = "s8_0ImmDecoder"; }
  def s7_0Ext : Operand<i32> { let ParserMatchClass = s7_0ExtOperand; }
  def s6_0Ext : Operand<i32> { let ParserMatchClass = s6_0ExtOperand;
                               let DecoderMethod = "s6_0ImmDecoder"; }
  def u7_0Ext : Operand<i32> { let ParserMatchClass = u7_0ExtOperand; }
  def u8_0Ext : Operand<i32> { let ParserMatchClass = u8_0ExtOperand; }
  def u9_0Ext : Operand<i32> { let ParserMatchClass = u9_0ExtOperand; }
  def u10_0Ext : Operand<i32> { let ParserMatchClass = u10_0ExtOperand; }
  def u6_0Ext : Operand<i32> { let ParserMatchClass = u6_0ExtOperand; }
  def u6_1Ext : Operand<i32> { let ParserMatchClass = u6_1ExtOperand; }
  def u6_2Ext : Operand<i32> { let ParserMatchClass = u6_2ExtOperand; }
  def u6_3Ext : Operand<i32> { let ParserMatchClass = u6_3ExtOperand; }
  def u32_0MustExt : Operand<i32> { let ParserMatchClass = u32_0MustExtOperand; }
}


def s4_7ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (HST->hasV60TOps())
    // Return true if the immediate can fit in a 10-bit sign extended field and
    // is 128-byte aligned.
    return isShiftedInt<4,7>(v);
  return false;
}]>;

def s4_6ImmPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  if (HST->hasV60TOps())
    // Return true if the immediate can fit in a 10-bit sign extended field and
    // is 64-byte aligned.
    return isShiftedInt<4,6>(v);
  return false;
}]>;


// This complex pattern exists only to create a machine instruction operand
// of type "frame index". There doesn't seem to be a way to do that directly
// in the patterns.
def AddrFI : ComplexPattern<i32, 1, "SelectAddrFI", [frameindex], []>;

// These complex patterns are not strictly necessary, since global address
// folding will happen during DAG combining. For distinguishing between GA
// and GP, pat frags with HexagonCONST32 and HexagonCONST32_GP can be used.
def AddrGA : ComplexPattern<i32, 1, "SelectAddrGA", [], []>;
def AddrGP : ComplexPattern<i32, 1, "SelectAddrGP", [], []>;

// Address operands.

let PrintMethod = "printGlobalOperand" in {
  def globaladdress : Operand<i32>;
  def globaladdressExt : Operand<i32>;
}

let PrintMethod = "printJumpTable" in
def jumptablebase : Operand<i32>;

def brtarget : Operand<OtherVT> {
  let DecoderMethod = "brtargetDecoder";
  let PrintMethod = "printBrtarget";
}
def brtargetExt : Operand<OtherVT> {
  let DecoderMethod = "brtargetDecoder";
  let PrintMethod = "printBrtarget";
}
def calltarget : Operand<i32> {
  let DecoderMethod = "brtargetDecoder";
  let PrintMethod = "printBrtarget";
}

def bblabel : Operand<i32>;
def bbl     : SDNode<"ISD::BasicBlock", SDTPtrLeaf, [], "BasicBlockSDNode">;
