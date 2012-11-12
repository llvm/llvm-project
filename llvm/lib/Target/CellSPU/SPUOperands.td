//===-- SPUOperands.td - Cell SPU Instruction Operands -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Cell SPU Instruction Operands:
//===----------------------------------------------------------------------===//

// TO_IMM32 - Convert an i8/i16 to i32.
def TO_IMM32 : SDNodeXForm<imm, [{
  return getI32Imm(N->getZExtValue());
}]>;

// TO_IMM16 - Convert an i8/i32 to i16.
def TO_IMM16 : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), MVT::i16);
}]>;


def LO16 : SDNodeXForm<imm, [{
  unsigned val = N->getZExtValue();
  // Transformation function: get the low 16 bits.
  return getI32Imm(val & 0xffff);
}]>;

def LO16_vec : SDNodeXForm<scalar_to_vector, [{
  SDValue OpVal(0, 0);

  // Transformation function: get the low 16 bit immediate from a build_vector
  // node.
  assert(N->getOpcode() == ISD::BUILD_VECTOR
         && "LO16_vec got something other than a BUILD_VECTOR");

  // Get first constant operand...
  for (unsigned i = 0, e = N->getNumOperands();
       OpVal.getNode() == 0 && i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.getNode() == 0)
      OpVal = N->getOperand(i);
  }
  
  assert(OpVal.getNode() != 0 && "LO16_vec did not locate a <defined> node");
  ConstantSDNode *CN = cast<ConstantSDNode>(OpVal);
  return getI32Imm((unsigned)CN->getZExtValue() & 0xffff);
}]>;

// Transform an immediate, returning the high 16 bits shifted down:
def HI16 : SDNodeXForm<imm, [{
  return getI32Imm((unsigned)N->getZExtValue() >> 16);
}]>;

// Transformation function: shift the high 16 bit immediate from a build_vector
// node into the low 16 bits, and return a 16-bit constant.
def HI16_vec : SDNodeXForm<scalar_to_vector, [{
  SDValue OpVal(0, 0);

  assert(N->getOpcode() == ISD::BUILD_VECTOR
         && "HI16_vec got something other than a BUILD_VECTOR");
  
  // Get first constant operand...
  for (unsigned i = 0, e = N->getNumOperands();
       OpVal.getNode() == 0 && i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.getNode() == 0)
      OpVal = N->getOperand(i);
  }
  
  assert(OpVal.getNode() != 0 && "HI16_vec did not locate a <defined> node");
  ConstantSDNode *CN = cast<ConstantSDNode>(OpVal);
  return getI32Imm((unsigned)CN->getZExtValue() >> 16);
}]>;

// simm7 predicate - True if the immediate fits in an 7-bit signed
// field.
def simm7: PatLeaf<(imm), [{
  int sextVal = int(N->getSExtValue());
  return (sextVal >= -64 && sextVal <= 63);
}]>;

// uimm7 predicate - True if the immediate fits in an 7-bit unsigned
// field.
def uimm7: PatLeaf<(imm), [{
  return (N->getZExtValue() <= 0x7f);
}]>;

// immSExt8 predicate - True if the immediate fits in an 8-bit sign extended
// field.
def immSExt8  : PatLeaf<(imm), [{
  int Value = int(N->getSExtValue());
  return (Value >= -(1 << 8) && Value <= (1 << 8) - 1);
}]>;

// immU8: immediate, unsigned 8-bit quantity
def immU8 : PatLeaf<(imm), [{
  return (N->getZExtValue() <= 0xff);
}]>;

// i32ImmSExt10 predicate - True if the i32 immediate fits in a 10-bit sign
// extended field.  Used by RI10Form instructions like 'ldq'.
def i32ImmSExt10  : PatLeaf<(imm), [{
  return isI32IntS10Immediate(N);
}]>;

// i32ImmUns10 predicate - True if the i32 immediate fits in a 10-bit unsigned
// field.  Used by RI10Form instructions like 'ldq'.
def i32ImmUns10  : PatLeaf<(imm), [{
  return isI32IntU10Immediate(N);
}]>;

// i16ImmSExt10 predicate - True if the i16 immediate fits in a 10-bit sign
// extended field.  Used by RI10Form instructions like 'ldq'.
def i16ImmSExt10  : PatLeaf<(imm), [{
  return isI16IntS10Immediate(N);
}]>;

// i16ImmUns10 predicate - True if the i16 immediate fits into a 10-bit unsigned
// value. Used by RI10Form instructions.
def i16ImmUns10 : PatLeaf<(imm), [{
  return isI16IntU10Immediate(N);
}]>;

def immSExt16  : PatLeaf<(imm), [{
  // immSExt16 predicate - True if the immediate fits in a 16-bit sign extended
  // field.
  short Ignored;
  return isIntS16Immediate(N, Ignored);
}]>;

def immZExt16  : PatLeaf<(imm), [{
  // immZExt16 predicate - True if the immediate fits in a 16-bit zero extended
  // field.
  return (uint64_t)N->getZExtValue() == (unsigned short)N->getZExtValue();
}], LO16>;

def immU16 : PatLeaf<(imm), [{
  // immU16 predicate- True if the immediate fits into a 16-bit unsigned field.
  return (uint64_t)N->getZExtValue() == (N->getZExtValue() & 0xffff);
}]>;

def imm18  : PatLeaf<(imm), [{
  // imm18 predicate: True if the immediate fits into an 18-bit unsigned field.
  int Value = (int) N->getZExtValue();
  return isUInt<18>(Value); 
}]>;

def lo16 : PatLeaf<(imm), [{
  // lo16 predicate - returns true if the immediate has all zeros in the
  // low order bits and is a 32-bit constant:
  if (N->getValueType(0) == MVT::i32) {
    uint32_t val = N->getZExtValue();
    return ((val & 0x0000ffff) == val);
  }

  return false;
}], LO16>;

def hi16 : PatLeaf<(imm), [{
  // hi16 predicate - returns true if the immediate has all zeros in the
  // low order bits and is a 32-bit constant:
  if (N->getValueType(0) == MVT::i32) {
    uint32_t val = uint32_t(N->getZExtValue());
    return ((val & 0xffff0000) == val);
  } else if (N->getValueType(0) == MVT::i64) {
    uint64_t val = N->getZExtValue();
    return ((val & 0xffff0000ULL) == val);
  }

  return false;
}], HI16>;

def bitshift : PatLeaf<(imm), [{
  // bitshift predicate - returns true if 0 < imm <= 7 for SHLQBII
  // (shift left quadword by bits immediate)
  int64_t Val = N->getZExtValue();
  return (Val > 0 && Val <= 7);
}]>;

//===----------------------------------------------------------------------===//
// Floating point operands:
//===----------------------------------------------------------------------===//

// Transform a float, returning the high 16 bits shifted down, as if
// the float was really an unsigned integer:
def HI16_f32 : SDNodeXForm<fpimm, [{
  float fval = N->getValueAPF().convertToFloat();
  return getI32Imm(FloatToBits(fval) >> 16);
}]>;

// Transformation function on floats: get the low 16 bits as if the float was
// an unsigned integer.
def LO16_f32 : SDNodeXForm<fpimm, [{
  float fval = N->getValueAPF().convertToFloat();
  return getI32Imm(FloatToBits(fval) & 0xffff);
}]>;

def FPimm_sext16 : SDNodeXForm<fpimm, [{
  float fval = N->getValueAPF().convertToFloat();
  return getI32Imm((int) ((FloatToBits(fval) << 16) >> 16));
}]>;

def FPimm_u18 : SDNodeXForm<fpimm, [{
  float fval = N->getValueAPF().convertToFloat();
  return getI32Imm(FloatToBits(fval) & ((1 << 18) - 1));
}]>;

def fpimmSExt16 : PatLeaf<(fpimm), [{
  short Ignored;
  return isFPS16Immediate(N, Ignored);  
}], FPimm_sext16>;

// Does the SFP constant only have upp 16 bits set?
def hi16_f32 : PatLeaf<(fpimm), [{
  if (N->getValueType(0) == MVT::f32) {
    uint32_t val = FloatToBits(N->getValueAPF().convertToFloat());
    return ((val & 0xffff0000) == val);
  }

  return false;
}], HI16_f32>;

// Does the SFP constant fit into 18 bits?
def fpimm18  : PatLeaf<(fpimm), [{
  if (N->getValueType(0) == MVT::f32) {
    uint32_t Value = FloatToBits(N->getValueAPF().convertToFloat());
    return isUInt<18>(Value);
  }

  return false;
}], FPimm_u18>;

//===----------------------------------------------------------------------===//
// 64-bit operands (TODO):
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// build_vector operands:
//===----------------------------------------------------------------------===//

// v16i8SExt8Imm_xform function: convert build_vector to 8-bit sign extended
// immediate constant load for v16i8 vectors. N.B.: The incoming constant has
// to be a 16-bit quantity with the upper and lower bytes equal (e.g., 0x2a2a).
def v16i8SExt8Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i8);
}]>;

// v16i8SExt8Imm: Predicate test for 8-bit sign extended immediate constant
// load, works in conjunction with its transform function. N.B.: This relies the
// incoming constant being a 16-bit quantity, where the upper and lower bytes
// are EXACTLY the same (e.g., 0x2a2a)
def v16i8SExt8Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i8).getNode() != 0;
}], v16i8SExt8Imm_xform>;

// v16i8U8Imm_xform function: convert build_vector to unsigned 8-bit
// immediate constant load for v16i8 vectors. N.B.: The incoming constant has
// to be a 16-bit quantity with the upper and lower bytes equal (e.g., 0x2a2a).
def v16i8U8Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i8);
}]>;

// v16i8U8Imm: Predicate test for unsigned 8-bit immediate constant
// load, works in conjunction with its transform function. N.B.: This relies the
// incoming constant being a 16-bit quantity, where the upper and lower bytes
// are EXACTLY the same (e.g., 0x2a2a)
def v16i8U8Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i8).getNode() != 0;
}], v16i8U8Imm_xform>;

// v8i16SExt8Imm_xform function: convert build_vector to 8-bit sign extended
// immediate constant load for v8i16 vectors.
def v8i16SExt8Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i16);
}]>;

// v8i16SExt8Imm: Predicate test for 8-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v8i16SExt8Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i8imm(N, *CurDAG, MVT::i16).getNode() != 0;
}], v8i16SExt8Imm_xform>;

// v8i16SExt10Imm_xform function: convert build_vector to 16-bit sign extended
// immediate constant load for v8i16 vectors.
def v8i16SExt10Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i16);
}]>;

// v8i16SExt10Imm: Predicate test for 16-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v8i16SExt10Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i16).getNode() != 0;
}], v8i16SExt10Imm_xform>;

// v8i16Uns10Imm_xform function: convert build_vector to 16-bit unsigned
// immediate constant load for v8i16 vectors.
def v8i16Uns10Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i16);
}]>;

// v8i16Uns10Imm: Predicate test for 16-bit unsigned immediate constant
// load, works in conjunction with its transform function.
def v8i16Uns10Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i16).getNode() != 0;
}], v8i16Uns10Imm_xform>;

// v8i16SExt16Imm_xform function: convert build_vector to 16-bit sign extended
// immediate constant load for v8i16 vectors.
def v8i16Uns16Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i16);
}]>;

// v8i16SExt16Imm: Predicate test for 16-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v8i16SExt16Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i16).getNode() != 0;
}], v8i16Uns16Imm_xform>;

// v4i32SExt10Imm_xform function: convert build_vector to 10-bit sign extended
// immediate constant load for v4i32 vectors.
def v4i32SExt10Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i32);
}]>;

// v4i32SExt10Imm: Predicate test for 10-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v4i32SExt10Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i32).getNode() != 0;
}], v4i32SExt10Imm_xform>;

// v4i32Uns10Imm_xform function: convert build_vector to 10-bit unsigned
// immediate constant load for v4i32 vectors.
def v4i32Uns10Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i32);
}]>;

// v4i32Uns10Imm: Predicate test for 10-bit unsigned immediate constant
// load, works in conjunction with its transform function.
def v4i32Uns10Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i32).getNode() != 0;
}], v4i32Uns10Imm_xform>;

// v4i32SExt16Imm_xform function: convert build_vector to 16-bit sign extended
// immediate constant load for v4i32 vectors.
def v4i32SExt16Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i32);
}]>;

// v4i32SExt16Imm: Predicate test for 16-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v4i32SExt16Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i32).getNode() != 0;
}], v4i32SExt16Imm_xform>;

// v4i32Uns18Imm_xform function: convert build_vector to 18-bit unsigned
// immediate constant load for v4i32 vectors.
def v4i32Uns18Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_u18imm(N, *CurDAG, MVT::i32);
}]>;

// v4i32Uns18Imm: Predicate test for 18-bit unsigned immediate constant load,
// works in conjunction with its transform function.
def v4i32Uns18Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_u18imm(N, *CurDAG, MVT::i32).getNode() != 0;
}], v4i32Uns18Imm_xform>;

// ILHUvec_get_imm xform function: convert build_vector to ILHUvec imm constant
// load.
def ILHUvec_get_imm: SDNodeXForm<build_vector, [{
  return SPU::get_ILHUvec_imm(N, *CurDAG, MVT::i32);
}]>;

/// immILHUvec: Predicate test for a ILHU constant vector.
def immILHUvec: PatLeaf<(build_vector), [{
  return SPU::get_ILHUvec_imm(N, *CurDAG, MVT::i32).getNode() != 0;
}], ILHUvec_get_imm>;

// Catch-all for any other i32 vector constants
def v4i32_get_imm: SDNodeXForm<build_vector, [{
  return SPU::get_v4i32_imm(N, *CurDAG);
}]>;

def v4i32Imm: PatLeaf<(build_vector), [{
  return SPU::get_v4i32_imm(N, *CurDAG).getNode() != 0;
}], v4i32_get_imm>;

// v2i64SExt10Imm_xform function: convert build_vector to 10-bit sign extended
// immediate constant load for v2i64 vectors.
def v2i64SExt10Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i64);
}]>;

// v2i64SExt10Imm: Predicate test for 10-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v2i64SExt10Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i10imm(N, *CurDAG, MVT::i64).getNode() != 0;
}], v2i64SExt10Imm_xform>;

// v2i64SExt16Imm_xform function: convert build_vector to 16-bit sign extended
// immediate constant load for v2i64 vectors.
def v2i64SExt16Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i64);
}]>;

// v2i64SExt16Imm: Predicate test for 16-bit sign extended immediate constant
// load, works in conjunction with its transform function.
def v2i64SExt16Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_i16imm(N, *CurDAG, MVT::i64).getNode() != 0;
}], v2i64SExt16Imm_xform>;

// v2i64Uns18Imm_xform function: convert build_vector to 18-bit unsigned
// immediate constant load for v2i64 vectors.
def v2i64Uns18Imm_xform: SDNodeXForm<build_vector, [{
  return SPU::get_vec_u18imm(N, *CurDAG, MVT::i64);
}]>;

// v2i64Uns18Imm: Predicate test for 18-bit unsigned immediate constant load,
// works in conjunction with its transform function.
def v2i64Uns18Imm: PatLeaf<(build_vector), [{
  return SPU::get_vec_u18imm(N, *CurDAG, MVT::i64).getNode() != 0;
}], v2i64Uns18Imm_xform>;

/// immILHUvec: Predicate test for a ILHU constant vector.
def immILHUvec_i64: PatLeaf<(build_vector), [{
  return SPU::get_ILHUvec_imm(N, *CurDAG, MVT::i64).getNode() != 0;
}], ILHUvec_get_imm>;

// Catch-all for any other i32 vector constants
def v2i64_get_imm: SDNodeXForm<build_vector, [{
  return SPU::get_v2i64_imm(N, *CurDAG);
}]>;

def v2i64Imm: PatLeaf<(build_vector), [{
  return SPU::get_v2i64_imm(N, *CurDAG).getNode() != 0;
}], v2i64_get_imm>;

//===----------------------------------------------------------------------===//
// Operand Definitions.

def s7imm: Operand<i8> {
  let PrintMethod = "printS7ImmOperand";
}

def s7imm_i8: Operand<i8> {
  let PrintMethod = "printS7ImmOperand";
}

def u7imm: Operand<i16> {
  let PrintMethod = "printU7ImmOperand";
}

def u7imm_i8: Operand<i8> {
  let PrintMethod = "printU7ImmOperand";
}

def u7imm_i32: Operand<i32> {
  let PrintMethod = "printU7ImmOperand";
}

// Halfword, signed 10-bit constant
def s10imm : Operand<i16> {
  let PrintMethod = "printS10ImmOperand";
}

def s10imm_i8: Operand<i8> {
  let PrintMethod = "printS10ImmOperand";
}

def s10imm_i32: Operand<i32> {
  let PrintMethod = "printS10ImmOperand";
}

def s10imm_i64: Operand<i64> {
  let PrintMethod = "printS10ImmOperand";
}

// Unsigned 10-bit integers:
def u10imm: Operand<i16> {
  let PrintMethod = "printU10ImmOperand";
}

def u10imm_i8: Operand<i8> {
  let PrintMethod = "printU10ImmOperand";
}

def u10imm_i32: Operand<i32> {
  let PrintMethod = "printU10ImmOperand";
}

def s16imm  : Operand<i16> {
  let PrintMethod = "printS16ImmOperand";
}

def s16imm_i8: Operand<i8> {
  let PrintMethod = "printS16ImmOperand";
}

def s16imm_i32: Operand<i32> {
  let PrintMethod = "printS16ImmOperand";
}

def s16imm_i64: Operand<i64> {
  let PrintMethod = "printS16ImmOperand";
}

def s16imm_f32: Operand<f32> {
  let PrintMethod = "printS16ImmOperand";
}

def s16imm_f64: Operand<f64> {
  let PrintMethod = "printS16ImmOperand";
}

def u16imm_i64 : Operand<i64> {
  let PrintMethod = "printU16ImmOperand";
}

def u16imm_i32 : Operand<i32> {
  let PrintMethod = "printU16ImmOperand";
}

def u16imm : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
}

def f16imm : Operand<f32> {
  let PrintMethod = "printU16ImmOperand";
}

def s18imm  : Operand<i32> {
  let PrintMethod = "printS18ImmOperand";
}

def u18imm : Operand<i32> {
  let PrintMethod = "printU18ImmOperand";
}

def u18imm_i64 : Operand<i64> {
  let PrintMethod = "printU18ImmOperand";
}

def f18imm : Operand<f32> {
  let PrintMethod = "printU18ImmOperand";
}

def f18imm_f64 : Operand<f64> {
  let PrintMethod = "printU18ImmOperand";
}

// Negated 7-bit halfword rotate immediate operands
def rothNeg7imm : Operand<i32> {
  let PrintMethod = "printROTHNeg7Imm";
}

def rothNeg7imm_i16 : Operand<i16> {
  let PrintMethod = "printROTHNeg7Imm";
}

// Negated 7-bit word rotate immediate operands
def rotNeg7imm : Operand<i32> {
  let PrintMethod = "printROTNeg7Imm";
}

def rotNeg7imm_i16 : Operand<i16> {
  let PrintMethod = "printROTNeg7Imm";
}

def rotNeg7imm_i8 : Operand<i8> {
  let PrintMethod = "printROTNeg7Imm";
}

def target : Operand<OtherVT> {
  let PrintMethod = "printBranchOperand";
}

// Absolute address call target
def calltarget : Operand<iPTR> {
  let PrintMethod = "printCallOperand";
  let MIOperandInfo = (ops u18imm:$calldest);
}

// PC relative call target
def relcalltarget : Operand<iPTR> {
  let PrintMethod = "printPCRelativeOperand";
  let MIOperandInfo = (ops s16imm:$calldest);
}

// Branch targets:
def brtarget : Operand<OtherVT> {
  let PrintMethod = "printPCRelativeOperand";
}

// Hint for branch target
def hbrtarget : Operand<OtherVT> {
  let PrintMethod = "printHBROperand";
}

// Indirect call target
def indcalltarget : Operand<iPTR> {
  let PrintMethod = "printCallOperand";
  let MIOperandInfo = (ops ptr_rc:$calldest);
}

def symbolHi: Operand<i32> {
  let PrintMethod = "printSymbolHi";
}

def symbolLo: Operand<i32> {
  let PrintMethod = "printSymbolLo";
}

def symbolLSA: Operand<i32> {
  let PrintMethod = "printSymbolLSA";
}

// Shuffle address memory operaand [s7imm(reg) d-format]
def shufaddr : Operand<iPTR> {
  let PrintMethod = "printShufAddr";
  let MIOperandInfo = (ops s7imm:$imm, ptr_rc:$reg);
}

// memory s10imm(reg) operand
def dformaddr : Operand<iPTR> {
  let PrintMethod = "printDFormAddr";
  let MIOperandInfo = (ops s10imm:$imm, ptr_rc:$reg);
}

// 256K local store address
// N.B.: The tblgen code generator expects to have two operands, an offset
// and a pointer. Of these, only the immediate is actually used.
def addr256k : Operand<iPTR> {
  let PrintMethod = "printAddr256K";
  let MIOperandInfo = (ops s16imm:$imm, ptr_rc:$reg);
}

// memory s18imm(reg) operand
def memri18 : Operand<iPTR> {
  let PrintMethod = "printMemRegImmS18";
  let MIOperandInfo = (ops s18imm:$imm, ptr_rc:$reg);
}

// memory register + register operand
def memrr : Operand<iPTR> {
  let PrintMethod = "printMemRegReg";
  let MIOperandInfo = (ops ptr_rc:$reg_a, ptr_rc:$reg_b);
}

// Define SPU-specific addressing modes: These come in three basic
// flavors:
//
// D-form   : [r+I10] (10-bit signed offset + reg)
// X-form   : [r+r]   (reg+reg)
// A-form   : abs     (256K LSA offset)
// D-form(2): [r+I7]  (7-bit signed offset + reg)

def dform_addr   : ComplexPattern<iPTR, 2, "SelectDFormAddr",
                                  [], [SDNPWantRoot]>;
def xform_addr   : ComplexPattern<iPTR, 2, "SelectXFormAddr",
                                  [], [SDNPWantRoot]>;
def aform_addr   : ComplexPattern<iPTR, 2, "SelectAFormAddr",
                                  [], [SDNPWantRoot]>;
def dform2_addr  : ComplexPattern<iPTR, 2, "SelectDForm2Addr",
                                  [], [SDNPWantRoot]>;
