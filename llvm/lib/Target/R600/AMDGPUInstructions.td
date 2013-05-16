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

def COND_EQ : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOEQ: case ISD::SETUEQ:
                     case ISD::SETEQ: return true;}}}]
>;

def COND_NE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETONE: case ISD::SETUNE:
                     case ISD::SETNE: return true;}}}]
>;
def COND_GT : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOGT: case ISD::SETUGT:
                     case ISD::SETGT: return true;}}}]
>;

def COND_GE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOGE: case ISD::SETUGE:
                     case ISD::SETGE: return true;}}}]
>;

def COND_LT : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOLT: case ISD::SETULT:
                     case ISD::SETLT: return true;}}}]
>;

def COND_LE : PatLeaf <
  (cond),
  [{switch(N->get()){{default: return false;
                     case ISD::SETOLE: case ISD::SETULE:
                     case ISD::SETLE: return true;}}}]
>;

def COND_NULL : PatLeaf <
  (cond),
  [{return false;}]
>;

//===----------------------------------------------------------------------===//
// Load/Store Pattern Fragments
//===----------------------------------------------------------------------===//

def zextloadi8_global : PatFrag<(ops node:$ptr), (zextloadi8 node:$ptr), [{
    return isGlobalLoad(dyn_cast<LoadSDNode>(N));
}]>;

class Constants {
int TWO_PI = 0x40c90fdb;
int PI = 0x40490fdb;
int TWO_PI_INV = 0x3e22f983;
int FP_UINT_MAX_PLUS_1 = 0x4f800000;	// 1 << 32 in floating point encoding
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

// Vector Build pattern
class Vector1_Build <ValueType vecType, ValueType elemType,
                     RegisterClass rc> : Pat <
  (vecType (build_vector elemType:$src)),
  (vecType (COPY_TO_REGCLASS $src, rc))
>;

class Vector2_Build <ValueType vecType, ValueType elemType> : Pat <
  (vecType (build_vector elemType:$sub0, elemType:$sub1)),
  (INSERT_SUBREG (INSERT_SUBREG
    (vecType (IMPLICIT_DEF)), $sub0, sub0), $sub1, sub1)
>;

class Vector4_Build <ValueType vecType, ValueType elemType> : Pat <
  (vecType (build_vector elemType:$x, elemType:$y, elemType:$z, elemType:$w)),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (vecType (IMPLICIT_DEF)), $x, sub0), $y, sub1), $z, sub2), $w, sub3)
>;

class Vector8_Build <ValueType vecType, ValueType elemType> : Pat <
  (vecType (build_vector elemType:$sub0, elemType:$sub1,
                         elemType:$sub2, elemType:$sub3,
                         elemType:$sub4, elemType:$sub5,
                         elemType:$sub6, elemType:$sub7)),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (vecType (IMPLICIT_DEF)), $sub0, sub0), $sub1, sub1),
                              $sub2, sub2), $sub3, sub3),
                              $sub4, sub4), $sub5, sub5),
                              $sub6, sub6), $sub7, sub7)
>;

class Vector16_Build <ValueType vecType, ValueType elemType> : Pat <
  (vecType (build_vector elemType:$sub0, elemType:$sub1,
                         elemType:$sub2, elemType:$sub3,
                         elemType:$sub4, elemType:$sub5,
                         elemType:$sub6, elemType:$sub7,
                         elemType:$sub8, elemType:$sub9,
                         elemType:$sub10, elemType:$sub11,
                         elemType:$sub12, elemType:$sub13,
                         elemType:$sub14, elemType:$sub15)),
  (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG (INSERT_SUBREG
    (vecType (IMPLICIT_DEF)), $sub0, sub0), $sub1, sub1),
                            $sub2, sub2), $sub3, sub3),
                            $sub4, sub4), $sub5, sub5),
                            $sub6, sub6), $sub7, sub7),
                            $sub8, sub8), $sub9, sub9),
                            $sub10, sub10), $sub11, sub11),
                            $sub12, sub12), $sub13, sub13),
                            $sub14, sub14), $sub15, sub15)
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

include "R600Instructions.td"

include "SIInstrInfo.td"

