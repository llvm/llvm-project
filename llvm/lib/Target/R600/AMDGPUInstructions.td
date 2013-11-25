//===-- AMDGPUInstructions.td - Common instruction defs ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains instruction defs that are common to all hw codegen
// targets.
//
//===----------------------------------------------------------------------===//

class AMDGPUInst <dag outs, dag ins, string asm, list<dag> pattern> : Instruction {
  field bit isRegisterLoad = 0;
  field bit isRegisterStore = 0;

  let Namespace = "AMDGPU";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
  let Itinerary = NullALU;

  let TSFlags{63} = isRegisterLoad;
  let TSFlags{62} = isRegisterStore;
}

class AMDGPUShaderInst <dag outs, dag ins, string asm, list<dag> pattern>
    : AMDGPUInst<outs, ins, asm, pattern> {

  field bits<32> Inst = 0xffffffff;

}

def InstFlag : OperandWithDefaultOps <i32, (ops (i32 0))>;
def ADDRIndirect : ComplexPattern<iPTR, 2, "SelectADDRIndirect", [], []>;

//===----------------------------------------------------------------------===//
// PatLeafs for floating-point comparisons
//===----------------------------------------------------------------------===//

def COND_OEQ : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOEQ || N->get() == ISD::SETEQ;}]
>;

def COND_OGT : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOGT || N->get() == ISD::SETGT;}]
>;

def COND_OGE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOGE || N->get() == ISD::SETGE;}]
>;

def COND_OLT : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOLT || N->get() == ISD::SETLT;}]
>;

def COND_OLE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETOLE || N->get() == ISD::SETLE;}]
>;

def COND_UNE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETUNE || N->get() == ISD::SETNE;}]
>;

def COND_O : PatLeaf <(cond), [{return N->get() == ISD::SETO;}]>;
def COND_UO : PatLeaf <(cond), [{return N->get() == ISD::SETUO;}]>;

//===----------------------------------------------------------------------===//
// PatLeafs for unsigned comparisons
//===----------------------------------------------------------------------===//

def COND_UGT : PatLeaf <(cond), [{return N->get() == ISD::SETUGT;}]>;
def COND_UGE : PatLeaf <(cond), [{return N->get() == ISD::SETUGE;}]>;
def COND_ULT : PatLeaf <(cond), [{return N->get() == ISD::SETULT;}]>;
def COND_ULE : PatLeaf <(cond), [{return N->get() == ISD::SETULE;}]>;

//===----------------------------------------------------------------------===//
// PatLeafs for signed comparisons
//===----------------------------------------------------------------------===//

def COND_SGT : PatLeaf <(cond), [{return N->get() == ISD::SETGT;}]>;
def COND_SGE : PatLeaf <(cond), [{return N->get() == ISD::SETGE;}]>;
def COND_SLT : PatLeaf <(cond), [{return N->get() == ISD::SETLT;}]>;
def COND_SLE : PatLeaf <(cond), [{return N->get() == ISD::SETLE;}]>;

//===----------------------------------------------------------------------===//
// PatLeafs for integer equality
//===----------------------------------------------------------------------===//

def COND_EQ : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETEQ || N->get() == ISD::SETUEQ;}]
>;

def COND_NE : PatLeaf <
  (cond),
  [{return N->get() == ISD::SETNE || N->get() == ISD::SETUNE;}]
>;

def COND_NULL : PatLeaf <
  (cond),
  [{return false;}]
>;

//===----------------------------------------------------------------------===//
// Load/Store Pattern Fragments
//===----------------------------------------------------------------------===//

def az_extload : PatFrag<(ops node:$ptr), (unindexedload node:$ptr), [{
  LoadSDNode *L = cast<LoadSDNode>(N);
  return L->getExtensionType() == ISD::ZEXTLOAD ||
         L->getExtensionType() == ISD::EXTLOAD;
}]>;

def az_extloadi8 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i8;
}]>;

def az_extloadi8_global : PatFrag<(ops node:$ptr), (az_extloadi8 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def sextloadi8_global : PatFrag<(ops node:$ptr), (sextloadi8 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def az_extloadi8_constant : PatFrag<(ops node:$ptr), (az_extloadi8 node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;

def sextloadi8_constant : PatFrag<(ops node:$ptr), (sextloadi8 node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;

def az_extloadi8_local : PatFrag<(ops node:$ptr), (az_extloadi8 node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def sextloadi8_local : PatFrag<(ops node:$ptr), (sextloadi8 node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def az_extloadi16 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i16;
}]>;

def az_extloadi16_global : PatFrag<(ops node:$ptr), (az_extloadi16 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def sextloadi16_global : PatFrag<(ops node:$ptr), (sextloadi16 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def az_extloadi16_constant : PatFrag<(ops node:$ptr), (az_extloadi16 node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;

def sextloadi16_constant : PatFrag<(ops node:$ptr), (sextloadi16 node:$ptr), [{
    return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;

def az_extloadi16_local : PatFrag<(ops node:$ptr), (az_extloadi16 node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def sextloadi16_local : PatFrag<(ops node:$ptr), (sextloadi16 node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def az_extloadi32 : PatFrag<(ops node:$ptr), (az_extload node:$ptr), [{
  return cast<LoadSDNode>(N)->getMemoryVT() == MVT::i32;
}]>;

def az_extloadi32_global : PatFrag<(ops node:$ptr),
                                   (az_extloadi32 node:$ptr), [{
  return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def az_extloadi32_constant : PatFrag<(ops node:$ptr),
                                     (az_extloadi32 node:$ptr), [{
  return isConstantLoad(dyn_cast<LoadSDNode>(N), -1);
}]>;

def truncstorei8_global : PatFrag<(ops node:$val, node:$ptr),
                                  (truncstorei8 node:$val, node:$ptr), [{
  return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;

def truncstorei16_global : PatFrag<(ops node:$val, node:$ptr),
                                  (truncstorei16 node:$val, node:$ptr), [{
  return isGlobalStore(dyn_cast<StoreSDNode>(N));
}]>;

def local_store : PatFrag<(ops node:$val, node:$ptr),
                             (store node:$val, node:$ptr), [{
  return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;

def truncstorei8_local : PatFrag<(ops node:$val, node:$ptr),
                                  (truncstorei8 node:$val, node:$ptr), [{
  return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;

def truncstorei16_local : PatFrag<(ops node:$val, node:$ptr),
                                  (truncstorei16 node:$val, node:$ptr), [{
  return isLocalStore(dyn_cast<StoreSDNode>(N));
}]>;

def local_load : PatFrag<(ops node:$ptr), (load node:$ptr), [{
    return isLocalLoad(dyn_cast<LoadSDNode>(N));
}]>;

def atomic_load_add_local : PatFrag<(ops node:$ptr, node:$value),
                                    (atomic_load_add node:$ptr, node:$value), [{
  return dyn_cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}]>;

def atomic_load_sub_local : PatFrag<(ops node:$ptr, node:$value),
                                    (atomic_load_sub node:$ptr, node:$value), [{
  return dyn_cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}]>;

def mskor_global : PatFrag<(ops node:$val, node:$ptr),
                            (AMDGPUstore_mskor node:$val, node:$ptr), [{
  return dyn_cast<MemSDNode>(N)->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}]>;

class Constants {
int TWO_PI = 0x40c90fdb;
int PI = 0x40490fdb;
int TWO_PI_INV = 0x3e22f983;
int FP_UINT_MAX_PLUS_1 = 0x4f800000;    // 1 << 32 in floating point encoding
}
def CONST : Constants;

def FP_ZERO : PatLeaf <
  (fpimm),
  [{return N->getValueAPF().isZero();}]
>;

def FP_ONE : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(1.0);}]
>;

def U24 : ComplexPattern<i32, 1, "SelectU24", [], []>;
def I24 : ComplexPattern<i32, 1, "SelectI24", [], []>;

let isCodeGenOnly = 1, isPseudo = 1 in {

let usesCustomInserter = 1  in {

class CLAMP <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "CLAMP $dst, $src0",
  [(set f32:$dst, (int_AMDIL_clamp f32:$src0, (f32 FP_ZERO), (f32 FP_ONE)))]
>;

class FABS <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FABS $dst, $src0",
  [(set f32:$dst, (fabs f32:$src0))]
>;

class FNEG <RegisterClass rc> : AMDGPUShaderInst <
  (outs rc:$dst),
  (ins rc:$src0),
  "FNEG $dst, $src0",
  [(set f32:$dst, (fneg f32:$src0))]
>;

} // usesCustomInserter = 1

multiclass RegisterLoadStore <RegisterClass dstClass, Operand addrClass,
                    ComplexPattern addrPat> {
let UseNamedOperandTable = 1 in {

  def RegisterLoad : AMDGPUShaderInst <
    (outs dstClass:$dst),
    (ins addrClass:$addr, i32imm:$chan),
    "RegisterLoad $dst, $addr",
    [(set i32:$dst, (AMDGPUregister_load addrPat:$addr, (i32 timm:$chan)))]
  > {
    let isRegisterLoad = 1;
  }

  def RegisterStore : AMDGPUShaderInst <
    (outs),
    (ins dstClass:$val, addrClass:$addr, i32imm:$chan),
    "RegisterStore $val, $addr",
    [(AMDGPUregister_store i32:$val, addrPat:$addr, (i32 timm:$chan))]
  > {
    let isRegisterStore = 1;
  }
}
}

} // End isCodeGenOnly = 1, isPseudo = 1

/* Generic helper patterns for intrinsics */
/* -------------------------------------- */

class POW_Common <AMDGPUInst log_ieee, AMDGPUInst exp_ieee, AMDGPUInst mul>
  : Pat <
  (fpow f32:$src0, f32:$src1),
  (exp_ieee (mul f32:$src1, (log_ieee f32:$src0)))
>;

/* Other helper patterns */
/* --------------------- */

/* Extract element pattern */
class Extract_Element <ValueType sub_type, ValueType vec_type, int sub_idx, 
                       SubRegIndex sub_reg>
  : Pat<
  (sub_type (vector_extract vec_type:$src, sub_idx)),
  (EXTRACT_SUBREG $src, sub_reg)
>;

/* Insert element pattern */
class Insert_Element <ValueType elem_type, ValueType vec_type,
                      int sub_idx, SubRegIndex sub_reg>
  : Pat <
  (vector_insert vec_type:$vec, elem_type:$elem, sub_idx),
  (INSERT_SUBREG $vec, $elem, sub_reg)
>;

class Vector4_Build <ValueType vecType, ValueType elemType> : Pat <
  (vecType (build_vector elemType:$x, elemType:$y, elemType:$z, elemType:$w)),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (vecType (IMPLICIT_DEF)), $x, sub0), $y, sub1), $z, sub2), $w, sub3)
>;

// XXX: Convert to new syntax and use COPY_TO_REG, once the DFAPacketizer
// can handle COPY instructions.
// bitconvert pattern
class BitConvert <ValueType dt, ValueType st, RegisterClass rc> : Pat <
  (dt (bitconvert (st rc:$src0))),
  (dt rc:$src0)
>;

// XXX: Convert to new syntax and use COPY_TO_REG, once the DFAPacketizer
// can handle COPY instructions.
class DwordAddrPat<ValueType vt, RegisterClass rc> : Pat <
  (vt (AMDGPUdwordaddr (vt rc:$addr))),
  (vt rc:$addr)
>;

// BFI_INT patterns

multiclass BFIPatterns <Instruction BFI_INT> {

  // Definition from ISA doc:
  // (y & x) | (z & ~x)
  def : Pat <
    (or (and i32:$y, i32:$x), (and i32:$z, (not i32:$x))),
    (BFI_INT $x, $y, $z)
  >;

  // SHA-256 Ch function
  // z ^ (x & (y ^ z))
  def : Pat <
    (xor i32:$z, (and i32:$x, (xor i32:$y, i32:$z))),
    (BFI_INT $x, $y, $z)
  >;

}

// SHA-256 Ma patterns

// ((x & z) | (y & (x | z))) -> BFI_INT (XOR x, y), z, y
class SHA256MaPattern <Instruction BFI_INT, Instruction XOR> : Pat <
  (or (and i32:$x, i32:$z), (and i32:$y, (or i32:$x, i32:$z))),
  (BFI_INT (XOR i32:$x, i32:$y), i32:$z, i32:$y)
>;

// Bitfield extract patterns

def legalshift32 : ImmLeaf <i32, [{return Imm >=0 && Imm < 32;}]>;
def bfemask : PatLeaf <(imm), [{return isMask_32(N->getZExtValue());}],
                            SDNodeXForm<imm, [{ return CurDAG->getTargetConstant(CountTrailingOnes_32(N->getZExtValue()), MVT::i32);}]>>;

class BFEPattern <Instruction BFE> : Pat <
  (and (srl i32:$x, legalshift32:$y), bfemask:$z),
  (BFE $x, $y, $z)
>;

// rotr pattern
class ROTRPattern <Instruction BIT_ALIGN> : Pat <
  (rotr i32:$src0, i32:$src1),
  (BIT_ALIGN $src0, $src0, $src1)
>;

// 24-bit arithmetic patterns
def umul24 : PatFrag <(ops node:$x, node:$y), (mul node:$x, node:$y)>;

/*
class UMUL24Pattern <Instruction UMUL24> : Pat <
  (mul U24:$x, U24:$y),
  (UMUL24 $x, $y)
>;
*/

include "R600Instructions.td"

include "SIInstrInfo.td"

