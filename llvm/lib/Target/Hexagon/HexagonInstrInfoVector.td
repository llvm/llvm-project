//===- HexagonInstrInfoVector.td - Hexagon Vector Patterns -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon Vector instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

def V2I1:  PatLeaf<(v2i1  PredRegs:$R)>;
def V4I1:  PatLeaf<(v4i1  PredRegs:$R)>;
def V8I1:  PatLeaf<(v8i1  PredRegs:$R)>;
def V4I8:  PatLeaf<(v4i8  IntRegs:$R)>;
def V2I16: PatLeaf<(v2i16 IntRegs:$R)>;
def V8I8:  PatLeaf<(v8i8  DoubleRegs:$R)>;
def V4I16: PatLeaf<(v4i16 DoubleRegs:$R)>;
def V2I32: PatLeaf<(v2i32 DoubleRegs:$R)>;


multiclass bitconvert_32<ValueType a, ValueType b> {
  def : Pat <(b (bitconvert (a IntRegs:$src))),
             (b IntRegs:$src)>;
  def : Pat <(a (bitconvert (b IntRegs:$src))),
             (a IntRegs:$src)>;
}

multiclass bitconvert_64<ValueType a, ValueType b> {
  def : Pat <(b (bitconvert (a DoubleRegs:$src))),
             (b DoubleRegs:$src)>;
  def : Pat <(a (bitconvert (b DoubleRegs:$src))),
             (a DoubleRegs:$src)>;
}

// Bit convert vector types.
defm : bitconvert_32<v4i8, i32>;
defm : bitconvert_32<v2i16, i32>;
defm : bitconvert_32<v2i16, v4i8>;

defm : bitconvert_64<v8i8, i64>;
defm : bitconvert_64<v4i16, i64>;
defm : bitconvert_64<v2i32, i64>;
defm : bitconvert_64<v8i8, v4i16>;
defm : bitconvert_64<v8i8, v2i32>;
defm : bitconvert_64<v4i16, v2i32>;


// Vector shift support. Vector shifting in Hexagon is rather different
// from internal representation of LLVM.
// LLVM assumes all shifts (in vector case) will have the form
// <VT> = SHL/SRA/SRL <VT> by <VT>
// while Hexagon has the following format:
// <VT> = SHL/SRA/SRL <VT> by <IT/i32>
// As a result, special care is needed to guarantee correctness and
// performance.
class vshift_v4i16<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u4Imm,
      [(set (v4i16 DoubleRegs:$dst),
            (Op (v4i16 DoubleRegs:$src1), u4ImmPred:$src2))]> {
  bits<4> src2;
  let Inst{11-8} = src2;
}

class vshift_v2i32<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u5Imm,
      [(set (v2i32 DoubleRegs:$dst),
            (Op (v2i32 DoubleRegs:$src1), u5ImmPred:$src2))]> {
  bits<5> src2;
  let Inst{12-8} = src2;
}

def : Pat<(v2i16 (add (v2i16 IntRegs:$src1), (v2i16 IntRegs:$src2))),
          (A2_svaddh IntRegs:$src1, IntRegs:$src2)>;

def : Pat<(v2i16 (sub (v2i16 IntRegs:$src1), (v2i16 IntRegs:$src2))),
          (A2_svsubh IntRegs:$src1, IntRegs:$src2)>;

def S2_asr_i_vw : vshift_v2i32<sra, "vasrw", 0b010, 0b000>;
def S2_lsr_i_vw : vshift_v2i32<srl, "vlsrw", 0b010, 0b001>;
def S2_asl_i_vw : vshift_v2i32<shl, "vaslw", 0b010, 0b010>;

def S2_asr_i_vh : vshift_v4i16<sra, "vasrh", 0b100, 0b000>;
def S2_lsr_i_vh : vshift_v4i16<srl, "vlsrh", 0b100, 0b001>;
def S2_asl_i_vh : vshift_v4i16<shl, "vaslh", 0b100, 0b010>;


def HexagonVSPLATB: SDNode<"HexagonISD::VSPLATB", SDTUnaryOp>;
def HexagonVSPLATH: SDNode<"HexagonISD::VSPLATH", SDTUnaryOp>;

// Replicate the low 8-bits from 32-bits input register into each of the
// four bytes of 32-bits destination register.
def: Pat<(v4i8  (HexagonVSPLATB I32:$Rs)), (S2_vsplatrb I32:$Rs)>;

// Replicate the low 16-bits from 32-bits input register into each of the
// four halfwords of 64-bits destination register.
def: Pat<(v4i16 (HexagonVSPLATH I32:$Rs)), (S2_vsplatrh I32:$Rs)>;


class VArith_pat <InstHexagon MI, SDNode Op, PatFrag Type>
  : Pat <(Op Type:$Rss, Type:$Rtt),
         (MI Type:$Rss, Type:$Rtt)>;

def: VArith_pat <A2_vaddub, add, V8I8>;
def: VArith_pat <A2_vaddh,  add, V4I16>;
def: VArith_pat <A2_vaddw,  add, V2I32>;
def: VArith_pat <A2_vsubub, sub, V8I8>;
def: VArith_pat <A2_vsubh,  sub, V4I16>;
def: VArith_pat <A2_vsubw,  sub, V2I32>;

def: VArith_pat <A2_and,    and, V2I16>;
def: VArith_pat <A2_xor,    xor, V2I16>;
def: VArith_pat <A2_or,     or,  V2I16>;

def: VArith_pat <A2_andp,   and, V8I8>;
def: VArith_pat <A2_andp,   and, V4I16>;
def: VArith_pat <A2_andp,   and, V2I32>;
def: VArith_pat <A2_orp,    or,  V8I8>;
def: VArith_pat <A2_orp,    or,  V4I16>;
def: VArith_pat <A2_orp,    or,  V2I32>;
def: VArith_pat <A2_xorp,   xor, V8I8>;
def: VArith_pat <A2_xorp,   xor, V4I16>;
def: VArith_pat <A2_xorp,   xor, V2I32>;

def: Pat<(v2i32 (sra V2I32:$b, (i64 (HexagonCOMBINE (i32 u5ImmPred:$c),
                                                    (i32 u5ImmPred:$c))))),
         (S2_asr_i_vw V2I32:$b, imm:$c)>;
def: Pat<(v2i32 (srl V2I32:$b, (i64 (HexagonCOMBINE (i32 u5ImmPred:$c),
                                                    (i32 u5ImmPred:$c))))),
         (S2_lsr_i_vw V2I32:$b, imm:$c)>;
def: Pat<(v2i32 (shl V2I32:$b, (i64 (HexagonCOMBINE (i32 u5ImmPred:$c),
                                                    (i32 u5ImmPred:$c))))),
         (S2_asl_i_vw V2I32:$b, imm:$c)>;

def: Pat<(v4i16 (sra V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4ImmPred:$c)))))),
         (S2_asr_i_vh V4I16:$b, imm:$c)>;
def: Pat<(v4i16 (srl V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4ImmPred:$c)))))),
         (S2_lsr_i_vh V4I16:$b, imm:$c)>;
def: Pat<(v4i16 (shl V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4ImmPred:$c)))))),
         (S2_asl_i_vh V4I16:$b, imm:$c)>;


def SDTHexagon_v2i32_v2i32_i32 : SDTypeProfile<1, 2,
  [SDTCisSameAs<0, 1>, SDTCisVT<0, v2i32>, SDTCisInt<2>]>;
def SDTHexagon_v4i16_v4i16_i32 : SDTypeProfile<1, 2,
  [SDTCisSameAs<0, 1>, SDTCisVT<0, v4i16>, SDTCisInt<2>]>;

def HexagonVSRAW: SDNode<"HexagonISD::VSRAW", SDTHexagon_v2i32_v2i32_i32>;
def HexagonVSRAH: SDNode<"HexagonISD::VSRAH", SDTHexagon_v4i16_v4i16_i32>;
def HexagonVSRLW: SDNode<"HexagonISD::VSRLW", SDTHexagon_v2i32_v2i32_i32>;
def HexagonVSRLH: SDNode<"HexagonISD::VSRLH", SDTHexagon_v4i16_v4i16_i32>;
def HexagonVSHLW: SDNode<"HexagonISD::VSHLW", SDTHexagon_v2i32_v2i32_i32>;
def HexagonVSHLH: SDNode<"HexagonISD::VSHLH", SDTHexagon_v4i16_v4i16_i32>;

def: Pat<(v2i32 (HexagonVSRAW V2I32:$Rs, u5ImmPred:$u5)),
         (S2_asr_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSRAH V4I16:$Rs, u4ImmPred:$u4)),
         (S2_asr_i_vh V4I16:$Rs, imm:$u4)>;
def: Pat<(v2i32 (HexagonVSRLW V2I32:$Rs, u5ImmPred:$u5)),
         (S2_lsr_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSRLH V4I16:$Rs, u4ImmPred:$u4)),
         (S2_lsr_i_vh V4I16:$Rs, imm:$u4)>;
def: Pat<(v2i32 (HexagonVSHLW V2I32:$Rs, u5ImmPred:$u5)),
         (S2_asl_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSHLH V4I16:$Rs, u4ImmPred:$u4)),
         (S2_asl_i_vh V4I16:$Rs, imm:$u4)>;

// Vector shift words by register
def S2_asr_r_vw : T_S3op_shiftVect < "vasrw", 0b00, 0b00>;
def S2_lsr_r_vw : T_S3op_shiftVect < "vlsrw", 0b00, 0b01>;
def S2_asl_r_vw : T_S3op_shiftVect < "vaslw", 0b00, 0b10>;
def S2_lsl_r_vw : T_S3op_shiftVect < "vlslw", 0b00, 0b11>;

// Vector shift halfwords by register
def S2_asr_r_vh : T_S3op_shiftVect < "vasrh", 0b01, 0b00>;
def S2_lsr_r_vh : T_S3op_shiftVect < "vlsrh", 0b01, 0b01>;
def S2_asl_r_vh : T_S3op_shiftVect < "vaslh", 0b01, 0b10>;
def S2_lsl_r_vh : T_S3op_shiftVect < "vlslh", 0b01, 0b11>;

class vshift_rr_pat<InstHexagon MI, SDNode Op, PatFrag Value>
  : Pat <(Op Value:$Rs, I32:$Rt),
         (MI Value:$Rs, I32:$Rt)>;

def: vshift_rr_pat <S2_asr_r_vw, HexagonVSRAW, V2I32>;
def: vshift_rr_pat <S2_asr_r_vh, HexagonVSRAH, V4I16>;
def: vshift_rr_pat <S2_lsr_r_vw, HexagonVSRLW, V2I32>;
def: vshift_rr_pat <S2_lsr_r_vh, HexagonVSRLH, V4I16>;
def: vshift_rr_pat <S2_asl_r_vw, HexagonVSHLW, V2I32>;
def: vshift_rr_pat <S2_asl_r_vh, HexagonVSHLH, V4I16>;


def SDTHexagonVecCompare_v8i8 : SDTypeProfile<1, 2,
  [SDTCisSameAs<1, 2>, SDTCisVT<0, i1>, SDTCisVT<1, v8i8>]>;
def SDTHexagonVecCompare_v4i16 : SDTypeProfile<1, 2,
  [SDTCisSameAs<1, 2>, SDTCisVT<0, i1>, SDTCisVT<1, v4i16>]>;
def SDTHexagonVecCompare_v2i32 : SDTypeProfile<1, 2,
  [SDTCisSameAs<1, 2>, SDTCisVT<0, i1>, SDTCisVT<1, v2i32>]>;

def HexagonVCMPBEQ:  SDNode<"HexagonISD::VCMPBEQ",  SDTHexagonVecCompare_v8i8>;
def HexagonVCMPBGT:  SDNode<"HexagonISD::VCMPBGT",  SDTHexagonVecCompare_v8i8>;
def HexagonVCMPBGTU: SDNode<"HexagonISD::VCMPBGTU", SDTHexagonVecCompare_v8i8>;
def HexagonVCMPHEQ:  SDNode<"HexagonISD::VCMPHEQ",  SDTHexagonVecCompare_v4i16>;
def HexagonVCMPHGT:  SDNode<"HexagonISD::VCMPHGT",  SDTHexagonVecCompare_v4i16>;
def HexagonVCMPHGTU: SDNode<"HexagonISD::VCMPHGTU", SDTHexagonVecCompare_v4i16>;
def HexagonVCMPWEQ:  SDNode<"HexagonISD::VCMPWEQ",  SDTHexagonVecCompare_v2i32>;
def HexagonVCMPWGT:  SDNode<"HexagonISD::VCMPWGT",  SDTHexagonVecCompare_v2i32>;
def HexagonVCMPWGTU: SDNode<"HexagonISD::VCMPWGTU", SDTHexagonVecCompare_v2i32>;


class vcmp_i1_pat<InstHexagon MI, SDNode Op, PatFrag Value>
  : Pat <(i1 (Op Value:$Rs, Value:$Rt)),
         (MI Value:$Rs, Value:$Rt)>;

def: vcmp_i1_pat<A2_vcmpbeq,  HexagonVCMPBEQ,  V8I8>;
def: vcmp_i1_pat<A4_vcmpbgt,  HexagonVCMPBGT,  V8I8>;
def: vcmp_i1_pat<A2_vcmpbgtu, HexagonVCMPBGTU, V8I8>;

def: vcmp_i1_pat<A2_vcmpheq,  HexagonVCMPHEQ,  V4I16>;
def: vcmp_i1_pat<A2_vcmphgt,  HexagonVCMPHGT,  V4I16>;
def: vcmp_i1_pat<A2_vcmphgtu, HexagonVCMPHGTU, V4I16>;

def: vcmp_i1_pat<A2_vcmpweq,  HexagonVCMPWEQ,  V2I32>;
def: vcmp_i1_pat<A2_vcmpwgt,  HexagonVCMPWGT,  V2I32>;
def: vcmp_i1_pat<A2_vcmpwgtu, HexagonVCMPWGTU, V2I32>;


class vcmp_vi1_pat<InstHexagon MI, PatFrag Op, PatFrag InVal, ValueType OutTy>
  : Pat <(OutTy (Op InVal:$Rs, InVal:$Rt)),
         (MI InVal:$Rs, InVal:$Rt)>;

def: vcmp_vi1_pat<A2_vcmpweq,  seteq,  V2I32, v2i1>;
def: vcmp_vi1_pat<A2_vcmpwgt,  setgt,  V2I32, v2i1>;
def: vcmp_vi1_pat<A2_vcmpwgtu, setugt, V2I32, v2i1>;

def: vcmp_vi1_pat<A2_vcmpheq,  seteq,  V4I16, v4i1>;
def: vcmp_vi1_pat<A2_vcmphgt,  setgt,  V4I16, v4i1>;
def: vcmp_vi1_pat<A2_vcmphgtu, setugt, V4I16, v4i1>;


// Hexagon doesn't have a vector multiply with C semantics.
// Instead, generate a pseudo instruction that gets expaneded into two
// scalar MPYI instructions.
// This is expanded by ExpandPostRAPseudos.
let isPseudo = 1 in
def VMULW : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rs, DoubleRegs:$Rt),
      ".error \"Should never try to emit VMULW\"",
      [(set V2I32:$Rd, (mul V2I32:$Rs, V2I32:$Rt))]>;

let isPseudo = 1 in
def VMULW_ACC : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rx, DoubleRegs:$Rs, DoubleRegs:$Rt),
      ".error \"Should never try to emit VMULW_ACC\"",
      [(set V2I32:$Rd, (add V2I32:$Rx, (mul V2I32:$Rs, V2I32:$Rt)))],
      "$Rd = $Rx">;

// Adds two v4i8: Hexagon does not have an insn for this one, so we
// use the double add v8i8, and use only the low part of the result.
def: Pat<(v4i8 (add (v4i8 IntRegs:$Rs), (v4i8 IntRegs:$Rt))),
         (LoReg (A2_vaddub (Zext64 $Rs), (Zext64 $Rt)))>;

// Subtract two v4i8: Hexagon does not have an insn for this one, so we
// use the double sub v8i8, and use only the low part of the result.
def: Pat<(v4i8 (sub (v4i8 IntRegs:$Rs), (v4i8 IntRegs:$Rt))),
         (LoReg (A2_vsubub (Zext64 $Rs), (Zext64 $Rt)))>;

//
// No 32 bit vector mux.
//
def: Pat<(v4i8 (select I1:$Pu, V4I8:$Rs, V4I8:$Rt)),
         (LoReg (C2_vmux I1:$Pu, (Zext64 $Rs), (Zext64 $Rt)))>;
def: Pat<(v2i16 (select I1:$Pu, V2I16:$Rs, V2I16:$Rt)),
         (LoReg (C2_vmux I1:$Pu, (Zext64 $Rs), (Zext64 $Rt)))>;

//
// 64-bit vector mux.
//
def: Pat<(v8i8 (vselect V8I1:$Pu, V8I8:$Rs, V8I8:$Rt)),
         (C2_vmux V8I1:$Pu, V8I8:$Rs, V8I8:$Rt)>;
def: Pat<(v4i16 (vselect V4I1:$Pu, V4I16:$Rs, V4I16:$Rt)),
         (C2_vmux V4I1:$Pu, V4I16:$Rs, V4I16:$Rt)>;
def: Pat<(v2i32 (vselect V2I1:$Pu, V2I32:$Rs, V2I32:$Rt)),
         (C2_vmux V2I1:$Pu, V2I32:$Rs, V2I32:$Rt)>;

//
// No 32 bit vector compare.
//
def: Pat<(i1 (seteq V4I8:$Rs, V4I8:$Rt)),
         (A2_vcmpbeq (Zext64 $Rs), (Zext64 $Rt))>;
def: Pat<(i1 (setgt V4I8:$Rs, V4I8:$Rt)),
         (A4_vcmpbgt (Zext64 $Rs), (Zext64 $Rt))>;
def: Pat<(i1 (setugt V4I8:$Rs, V4I8:$Rt)),
         (A2_vcmpbgtu (Zext64 $Rs), (Zext64 $Rt))>;

def: Pat<(i1 (seteq V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmpheq (Zext64 $Rs), (Zext64 $Rt))>;
def: Pat<(i1 (setgt V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmphgt (Zext64 $Rs), (Zext64 $Rt))>;
def: Pat<(i1 (setugt V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmphgtu (Zext64 $Rs), (Zext64 $Rt))>;


class InvertCmp_pat<InstHexagon InvMI, PatFrag CmpOp, PatFrag Value,
                    ValueType CmpTy>
  : Pat<(CmpTy (CmpOp Value:$Rs, Value:$Rt)),
        (InvMI Value:$Rt, Value:$Rs)>;

// Map from a compare operation to the corresponding instruction with the
// order of operands reversed, e.g.  x > y --> cmp.lt(y,x).
def: InvertCmp_pat<A4_vcmpbgt,  setlt,  V8I8,  i1>;
def: InvertCmp_pat<A4_vcmpbgt,  setlt,  V8I8,  v8i1>;
def: InvertCmp_pat<A2_vcmphgt,  setlt,  V4I16, i1>;
def: InvertCmp_pat<A2_vcmphgt,  setlt,  V4I16, v4i1>;
def: InvertCmp_pat<A2_vcmpwgt,  setlt,  V2I32, i1>;
def: InvertCmp_pat<A2_vcmpwgt,  setlt,  V2I32, v2i1>;

def: InvertCmp_pat<A2_vcmpbgtu, setult, V8I8,  i1>;
def: InvertCmp_pat<A2_vcmpbgtu, setult, V8I8,  v8i1>;
def: InvertCmp_pat<A2_vcmphgtu, setult, V4I16, i1>;
def: InvertCmp_pat<A2_vcmphgtu, setult, V4I16, v4i1>;
def: InvertCmp_pat<A2_vcmpwgtu, setult, V2I32, i1>;
def: InvertCmp_pat<A2_vcmpwgtu, setult, V2I32, v2i1>;

// Map from vcmpne(Rss) -> !vcmpew(Rss).
// rs != rt -> !(rs == rt).
def: Pat<(v2i1 (setne V2I32:$Rs, V2I32:$Rt)),
         (C2_not (v2i1 (A2_vcmpbeq V2I32:$Rs, V2I32:$Rt)))>;


// Truncate: from vector B copy all 'E'ven 'B'yte elements:
// A[0] = B[0];  A[1] = B[2];  A[2] = B[4];  A[3] = B[6];
def: Pat<(v4i8 (trunc V4I16:$Rs)),
         (S2_vtrunehb V4I16:$Rs)>;

// Truncate: from vector B copy all 'O'dd 'B'yte elements:
// A[0] = B[1];  A[1] = B[3];  A[2] = B[5];  A[3] = B[7];
// S2_vtrunohb

// Truncate: from vectors B and C copy all 'E'ven 'H'alf-word elements:
// A[0] = B[0];  A[1] = B[2];  A[2] = C[0];  A[3] = C[2];
// S2_vtruneh

def: Pat<(v2i16 (trunc V2I32:$Rs)),
         (LoReg (S2_packhl (HiReg $Rs), (LoReg $Rs)))>;


def HexagonVSXTBH : SDNode<"HexagonISD::VSXTBH", SDTUnaryOp>;
def HexagonVSXTBW : SDNode<"HexagonISD::VSXTBW", SDTUnaryOp>;

def: Pat<(i64 (HexagonVSXTBH I32:$Rs)), (S2_vsxtbh I32:$Rs)>;
def: Pat<(i64 (HexagonVSXTBW I32:$Rs)), (S2_vsxthw I32:$Rs)>;

def: Pat<(v4i16 (zext   V4I8:$Rs)),  (S2_vzxtbh V4I8:$Rs)>;
def: Pat<(v2i32 (zext   V2I16:$Rs)), (S2_vzxthw V2I16:$Rs)>;
def: Pat<(v4i16 (anyext V4I8:$Rs)),  (S2_vzxtbh V4I8:$Rs)>;
def: Pat<(v2i32 (anyext V2I16:$Rs)), (S2_vzxthw V2I16:$Rs)>;
def: Pat<(v4i16 (sext   V4I8:$Rs)),  (S2_vsxtbh V4I8:$Rs)>;
def: Pat<(v2i32 (sext   V2I16:$Rs)), (S2_vsxthw V2I16:$Rs)>;

// Sign extends a v2i8 into a v2i32.
def: Pat<(v2i32 (sext_inreg V2I32:$Rs, v2i8)),
         (A2_combinew (A2_sxtb (HiReg $Rs)), (A2_sxtb (LoReg $Rs)))>;

// Sign extends a v2i16 into a v2i32.
def: Pat<(v2i32 (sext_inreg V2I32:$Rs, v2i16)),
         (A2_combinew (A2_sxth (HiReg $Rs)), (A2_sxth (LoReg $Rs)))>;


// Multiplies two v2i16 and returns a v2i32.  We are using here the
// saturating multiply, as hexagon does not provide a non saturating
// vector multiply, and saturation does not impact the result that is
// in double precision of the operands.

// Multiplies two v2i16 vectors: as Hexagon does not have a multiply
// with the C semantics for this one, this pattern uses the half word
// multiply vmpyh that takes two v2i16 and returns a v2i32.  This is
// then truncated to fit this back into a v2i16 and to simulate the
// wrap around semantics for unsigned in C.
def vmpyh: OutPatFrag<(ops node:$Rs, node:$Rt),
                      (M2_vmpy2s_s0 (i32 $Rs), (i32 $Rt))>;

def: Pat<(v2i16 (mul V2I16:$Rs, V2I16:$Rt)),
         (LoReg (S2_vtrunewh (v2i32 (A2_combineii 0, 0)),
                             (v2i32 (vmpyh V2I16:$Rs, V2I16:$Rt))))>;

// Multiplies two v4i16 vectors.
def: Pat<(v4i16 (mul V4I16:$Rs, V4I16:$Rt)),
         (S2_vtrunewh (vmpyh (HiReg $Rs), (HiReg $Rt)),
                      (vmpyh (LoReg $Rs), (LoReg $Rt)))>;

def VMPYB_no_V5: OutPatFrag<(ops node:$Rs, node:$Rt),
  (S2_vtrunewh (vmpyh (HiReg (S2_vsxtbh $Rs)), (HiReg (S2_vsxtbh $Rt))),
               (vmpyh (LoReg (S2_vsxtbh $Rs)), (LoReg (S2_vsxtbh $Rt))))>;

// Multiplies two v4i8 vectors.
def: Pat<(v4i8 (mul V4I8:$Rs, V4I8:$Rt)),
         (S2_vtrunehb (M5_vmpybsu V4I8:$Rs, V4I8:$Rt))>,
     Requires<[HasV5T]>;

def: Pat<(v4i8 (mul V4I8:$Rs, V4I8:$Rt)),
         (S2_vtrunehb (VMPYB_no_V5 V4I8:$Rs, V4I8:$Rt))>;

// Multiplies two v8i8 vectors.
def: Pat<(v8i8 (mul V8I8:$Rs, V8I8:$Rt)),
         (A2_combinew (S2_vtrunehb (M5_vmpybsu (HiReg $Rs), (HiReg $Rt))),
                      (S2_vtrunehb (M5_vmpybsu (LoReg $Rs), (LoReg $Rt))))>,
     Requires<[HasV5T]>;

def: Pat<(v8i8 (mul V8I8:$Rs, V8I8:$Rt)),
         (A2_combinew (S2_vtrunehb (VMPYB_no_V5 (HiReg $Rs), (HiReg $Rt))),
                      (S2_vtrunehb (VMPYB_no_V5 (LoReg $Rs), (LoReg $Rt))))>;


class shuffler<SDNode Op, string Str>
  : SInst<(outs DoubleRegs:$a), (ins DoubleRegs:$b, DoubleRegs:$c),
      "$a = " # Str # "($b, $c)",
      [(set (i64 DoubleRegs:$a),
            (i64 (Op (i64 DoubleRegs:$b), (i64 DoubleRegs:$c))))],
      "", S_3op_tc_1_SLOT23>;

def SDTHexagonBinOp64 : SDTypeProfile<1, 2,
  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisVT<0, i64>]>;

def HexagonSHUFFEB: SDNode<"HexagonISD::SHUFFEB", SDTHexagonBinOp64>;
def HexagonSHUFFEH: SDNode<"HexagonISD::SHUFFEH", SDTHexagonBinOp64>;
def HexagonSHUFFOB: SDNode<"HexagonISD::SHUFFOB", SDTHexagonBinOp64>;
def HexagonSHUFFOH: SDNode<"HexagonISD::SHUFFOH", SDTHexagonBinOp64>;

class ShufflePat<InstHexagon MI, SDNode Op>
  : Pat<(i64 (Op DoubleRegs:$src1, DoubleRegs:$src2)),
        (i64 (MI DoubleRegs:$src1, DoubleRegs:$src2))>;

// Shuffles even bytes for i=0..3: A[2*i].b = C[2*i].b; A[2*i+1].b = B[2*i].b
def: ShufflePat<S2_shuffeb, HexagonSHUFFEB>;

// Shuffles odd bytes for i=0..3: A[2*i].b = C[2*i+1].b; A[2*i+1].b = B[2*i+1].b
def: ShufflePat<S2_shuffob, HexagonSHUFFOB>;

// Shuffles even half for i=0,1: A[2*i].h = C[2*i].h; A[2*i+1].h = B[2*i].h
def: ShufflePat<S2_shuffeh, HexagonSHUFFEH>;

// Shuffles odd half for i=0,1: A[2*i].h = C[2*i+1].h; A[2*i+1].h = B[2*i+1].h
def: ShufflePat<S2_shuffoh, HexagonSHUFFOH>;


// Truncated store from v4i16 to v4i8.
def truncstorev4i8: PatFrag<(ops node:$val, node:$ptr),
                            (truncstore node:$val, node:$ptr),
    [{ return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v4i8; }]>;

// Truncated store from v2i32 to v2i16.
def truncstorev2i16: PatFrag<(ops node:$val, node:$ptr),
                             (truncstore node:$val, node:$ptr),
    [{ return cast<StoreSDNode>(N)->getMemoryVT() == MVT::v2i16; }]>;

def: Pat<(truncstorev2i16 V2I32:$Rs, I32:$Rt),
         (S2_storeri_io I32:$Rt, 0, (LoReg (S2_packhl (HiReg $Rs),
                                                      (LoReg $Rs))))>;

def: Pat<(truncstorev4i8 V4I16:$Rs, I32:$Rt),
         (S2_storeri_io I32:$Rt, 0, (S2_vtrunehb V4I16:$Rs))>;


// Zero and sign extended load from v2i8 into v2i16.
def zextloadv2i8: PatFrag<(ops node:$ptr), (zextload node:$ptr),
    [{ return cast<LoadSDNode>(N)->getMemoryVT() == MVT::v2i8; }]>;

def sextloadv2i8: PatFrag<(ops node:$ptr), (sextload node:$ptr),
    [{ return cast<LoadSDNode>(N)->getMemoryVT() == MVT::v2i8; }]>;

def: Pat<(v2i16 (zextloadv2i8 I32:$Rs)),
         (LoReg (v4i16 (S2_vzxtbh (L2_loadruh_io I32:$Rs, 0))))>;

def: Pat<(v2i16 (sextloadv2i8 I32:$Rs)),
         (LoReg (v4i16 (S2_vsxtbh (L2_loadrh_io I32:$Rs, 0))))>;

def: Pat<(v2i32 (zextloadv2i8 I32:$Rs)),
         (S2_vzxthw (LoReg (v4i16 (S2_vzxtbh (L2_loadruh_io I32:$Rs, 0)))))>;

def: Pat<(v2i32 (sextloadv2i8 I32:$Rs)),
         (S2_vsxthw (LoReg (v4i16 (S2_vsxtbh (L2_loadrh_io I32:$Rs, 0)))))>;
