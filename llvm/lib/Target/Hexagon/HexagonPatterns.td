//==- HexagonPatterns.td - Target Description for Hexagon -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Pattern fragment that combines the value type and the register class
// into a single parameter.

// Pattern fragments to extract the low and high subregisters from a
// 64-bit value.
def LoReg: OutPatFrag<(ops node:$Rs), (EXTRACT_SUBREG (i64 $Rs), isub_lo)>;
def HiReg: OutPatFrag<(ops node:$Rs), (EXTRACT_SUBREG (i64 $Rs), isub_hi)>;

def IsOrAdd: PatFrag<(ops node:$Addr, node:$off),
    (or node:$Addr, node:$off), [{ return isOrEquivalentToAdd(N); }]>;

def Iss4_6 : PatLeaf<(i32 imm), [{
  int32_t V = N->getSExtValue();
  return isShiftedInt<4,6>(V);
}]>;

def Iss4_7 : PatLeaf<(i32 imm), [{
  int32_t V = N->getSExtValue();
  return isShiftedInt<4,7>(V);
}]>;

def IsPow2_32 : PatLeaf<(i32 imm), [{
  uint32_t V = N->getZExtValue();
  return isPowerOf2_32(V);
}]>;

def IsPow2_64 : PatLeaf<(i64 imm), [{
  uint64_t V = N->getZExtValue();
  return isPowerOf2_64(V);
}]>;

def IsNPow2_32 : PatLeaf<(i32 imm), [{
  uint32_t NV = ~N->getZExtValue();
  return isPowerOf2_32(NV);
}]>;

def IsPow2_64L : PatLeaf<(i64 imm), [{
  uint64_t V = N->getZExtValue();
  return isPowerOf2_64(V) && Log2_64(V) < 32;
}]>;

def IsPow2_64H : PatLeaf<(i64 imm), [{
  uint64_t V = N->getZExtValue();
  return isPowerOf2_64(V) && Log2_64(V) >= 32;
}]>;

def IsNPow2_64L : PatLeaf<(i64 imm), [{
  uint64_t NV = ~N->getZExtValue();
  return isPowerOf2_64(NV) && Log2_64(NV) < 32;
}]>;

def IsNPow2_64H : PatLeaf<(i64 imm), [{
  uint64_t NV = ~N->getZExtValue();
  return isPowerOf2_64(NV) && Log2_64(NV) >= 32;
}]>;

def SDEC1 : SDNodeXForm<imm, [{
  int32_t V = N->getSExtValue();
  return CurDAG->getTargetConstant(V-1, SDLoc(N), MVT::i32);
}]>;

def UDEC1 : SDNodeXForm<imm, [{
  uint32_t V = N->getZExtValue();
  assert(V >= 1);
  return CurDAG->getTargetConstant(V-1, SDLoc(N), MVT::i32);
}]>;

def UDEC32 : SDNodeXForm<imm, [{
  uint32_t V = N->getZExtValue();
  assert(V >= 32);
  return CurDAG->getTargetConstant(V-32, SDLoc(N), MVT::i32);
}]>;

def Log2_32 : SDNodeXForm<imm, [{
  uint32_t V = N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(V), SDLoc(N), MVT::i32);
}]>;

def Log2_64 : SDNodeXForm<imm, [{
  uint64_t V = N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_64(V), SDLoc(N), MVT::i32);
}]>;

def LogN2_32 : SDNodeXForm<imm, [{
  uint32_t NV = ~N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(NV), SDLoc(N), MVT::i32);
}]>;

def LogN2_64 : SDNodeXForm<imm, [{
  uint64_t NV = ~N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_64(NV), SDLoc(N), MVT::i32);
}]>;

def ToZext64: OutPatFrag<(ops node:$Rs),
  (i64 (A4_combineir 0, (i32 $Rs)))>;
def ToSext64: OutPatFrag<(ops node:$Rs),
  (i64 (A2_sxtw (i32 $Rs)))>;


class T_CMP_pat <InstHexagon MI, PatFrag OpNode, PatLeaf ImmPred>
  : Pat<(i1 (OpNode I32:$src1, ImmPred:$src2)),
        (MI IntRegs:$src1, ImmPred:$src2)>;

def : T_CMP_pat <C2_cmpeqi,  seteq,  s10_0ImmPred>;
def : T_CMP_pat <C2_cmpgti,  setgt,  s10_0ImmPred>;
def : T_CMP_pat <C2_cmpgtui, setugt, u9_0ImmPred>;

def SDTHexagonI64I32I32 : SDTypeProfile<1, 2,
  [SDTCisVT<0, i64>, SDTCisVT<1, i32>, SDTCisSameAs<1, 2>]>;

def HexagonCOMBINE : SDNode<"HexagonISD::COMBINE", SDTHexagonI64I32I32>;
def HexagonPACKHL  : SDNode<"HexagonISD::PACKHL",  SDTHexagonI64I32I32>;

// Pats for instruction selection.
class BinOp32_pat<SDNode Op, InstHexagon MI, ValueType ResT>
  : Pat<(ResT (Op I32:$Rs, I32:$Rt)),
        (ResT (MI IntRegs:$Rs, IntRegs:$Rt))>;

def: BinOp32_pat<add, A2_add, i32>;
def: BinOp32_pat<and, A2_and, i32>;
def: BinOp32_pat<or,  A2_or,  i32>;
def: BinOp32_pat<sub, A2_sub, i32>;
def: BinOp32_pat<xor, A2_xor, i32>;

def: BinOp32_pat<HexagonCOMBINE, A2_combinew, i64>;
def: BinOp32_pat<HexagonPACKHL,  S2_packhl,   i64>;

// Patfrag to convert the usual comparison patfrags (e.g. setlt) to ones
// that reverse the order of the operands.
class RevCmp<PatFrag F> : PatFrag<(ops node:$rhs, node:$lhs), F.Fragment>;

// Pats for compares. They use PatFrags as operands, not SDNodes,
// since seteq/setgt/etc. are defined as ParFrags.
class T_cmp32_rr_pat<InstHexagon MI, PatFrag Op, ValueType VT>
  : Pat<(VT (Op I32:$Rs, I32:$Rt)),
        (MI IntRegs:$Rs, IntRegs:$Rt)>;

def: T_cmp32_rr_pat<C2_cmpeq,  seteq,  i1>;
def: T_cmp32_rr_pat<C2_cmpgt,  setgt,  i1>;
def: T_cmp32_rr_pat<C2_cmpgtu, setugt, i1>;

def: T_cmp32_rr_pat<C2_cmpgt,  RevCmp<setlt>,  i1>;
def: T_cmp32_rr_pat<C2_cmpgtu, RevCmp<setult>, i1>;

def: Pat<(select I1:$Pu, I32:$Rs, I32:$Rt),
         (C2_mux PredRegs:$Pu, IntRegs:$Rs, IntRegs:$Rt)>;

def: Pat<(add I32:$Rs, s32_0ImmPred:$s16),
         (A2_addi I32:$Rs, imm:$s16)>;

def: Pat<(or I32:$Rs, s32_0ImmPred:$s10),
         (A2_orir IntRegs:$Rs, imm:$s10)>;
def: Pat<(and I32:$Rs, s32_0ImmPred:$s10),
         (A2_andir IntRegs:$Rs, imm:$s10)>;

def: Pat<(sub s32_0ImmPred:$s10, IntRegs:$Rs),
         (A2_subri imm:$s10, IntRegs:$Rs)>;

// Rd = not(Rs) gets mapped to Rd=sub(#-1, Rs).
def: Pat<(not I32:$src1),
         (A2_subri -1, IntRegs:$src1)>;

def TruncI64ToI32: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def: Pat<(s32_0ImmPred:$s16), (A2_tfrsi imm:$s16)>;
def: Pat<(s8_0Imm64Pred:$s8), (A2_tfrpi (TruncI64ToI32 $s8))>;

def : Pat<(select I1:$Pu, s32_0ImmPred:$s8, I32:$Rs),
          (C2_muxri I1:$Pu, imm:$s8, I32:$Rs)>;

def : Pat<(select I1:$Pu, I32:$Rs, s32_0ImmPred:$s8),
          (C2_muxir I1:$Pu, I32:$Rs, imm:$s8)>;

def : Pat<(select I1:$Pu, s32_0ImmPred:$s8, s8_0ImmPred:$S8),
          (C2_muxii I1:$Pu, imm:$s8, imm:$S8)>;

def: Pat<(shl I32:$src1, (i32 16)),   (A2_aslh I32:$src1)>;
def: Pat<(sra I32:$src1, (i32 16)),   (A2_asrh I32:$src1)>;
def: Pat<(sext_inreg I32:$src1, i8),  (A2_sxtb I32:$src1)>;
def: Pat<(sext_inreg I32:$src1, i16), (A2_sxth I32:$src1)>;

class T_vcmp_pat<InstHexagon MI, PatFrag Op, ValueType T>
  : Pat<(i1 (Op (T DoubleRegs:$Rss), (T DoubleRegs:$Rtt))),
        (i1 (MI DoubleRegs:$Rss, DoubleRegs:$Rtt))>;

def: T_vcmp_pat<A2_vcmpbeq,  seteq,  v8i8>;
def: T_vcmp_pat<A2_vcmpbgtu, setugt, v8i8>;
def: T_vcmp_pat<A2_vcmpheq,  seteq,  v4i16>;
def: T_vcmp_pat<A2_vcmphgt,  setgt,  v4i16>;
def: T_vcmp_pat<A2_vcmphgtu, setugt, v4i16>;
def: T_vcmp_pat<A2_vcmpweq,  seteq,  v2i32>;
def: T_vcmp_pat<A2_vcmpwgt,  setgt,  v2i32>;
def: T_vcmp_pat<A2_vcmpwgtu, setugt, v2i32>;

// Add halfword.
def: Pat<(sext_inreg (add I32:$src1, I32:$src2), i16),
         (A2_addh_l16_ll I32:$src1, I32:$src2)>;

def: Pat<(sra (add (shl I32:$src1, (i32 16)), I32:$src2), (i32 16)),
         (A2_addh_l16_hl I32:$src1, I32:$src2)>;

def: Pat<(shl (add I32:$src1, I32:$src2), (i32 16)),
         (A2_addh_h16_ll I32:$src1, I32:$src2)>;

// Subtract halfword.
def: Pat<(sext_inreg (sub I32:$src1, I32:$src2), i16),
         (A2_subh_l16_ll I32:$src1, I32:$src2)>;

def: Pat<(shl (sub I32:$src1, I32:$src2), (i32 16)),
         (A2_subh_h16_ll I32:$src1, I32:$src2)>;

// Here, depending on  the operand being selected, we'll either generate a
// min or max instruction.
// Ex:
// (a>b)?a:b --> max(a,b) => Here check performed is '>' and the value selected
// is the larger of two. So, the corresponding HexagonInst is passed in 'Inst'.
// (a>b)?b:a --> min(a,b) => Here check performed is '>' but the smaller value
// is selected and the corresponding HexagonInst is passed in 'SwapInst'.

multiclass T_MinMax_pats <PatFrag Op, PatLeaf Val,
                          InstHexagon Inst, InstHexagon SwapInst> {
  def: Pat<(select (i1 (Op Val:$src1, Val:$src2)), Val:$src1, Val:$src2),
           (Inst Val:$src1, Val:$src2)>;
  def: Pat<(select (i1 (Op Val:$src1, Val:$src2)), Val:$src2, Val:$src1),
           (SwapInst Val:$src1, Val:$src2)>;
}

def IsPosHalf : PatLeaf<(i32 IntRegs:$a), [{
  return isPositiveHalfWord(N);
}]>;

multiclass MinMax_pats <PatFrag Op, InstHexagon Inst, InstHexagon SwapInst> {
  defm: T_MinMax_pats<Op, I32, Inst, SwapInst>;

  def: Pat<(sext_inreg (select (i1 (Op IsPosHalf:$src1, IsPosHalf:$src2)),
                               IsPosHalf:$src1, IsPosHalf:$src2),
                       i16),
           (Inst IntRegs:$src1, IntRegs:$src2)>;

  def: Pat<(sext_inreg (select (i1 (Op IsPosHalf:$src1, IsPosHalf:$src2)),
                               IsPosHalf:$src2, IsPosHalf:$src1),
                       i16),
           (SwapInst IntRegs:$src1, IntRegs:$src2)>;
}

let AddedComplexity = 200 in {
  defm: MinMax_pats<setge,  A2_max,  A2_min>;
  defm: MinMax_pats<setgt,  A2_max,  A2_min>;
  defm: MinMax_pats<setle,  A2_min,  A2_max>;
  defm: MinMax_pats<setlt,  A2_min,  A2_max>;
  defm: MinMax_pats<setuge, A2_maxu, A2_minu>;
  defm: MinMax_pats<setugt, A2_maxu, A2_minu>;
  defm: MinMax_pats<setule, A2_minu, A2_maxu>;
  defm: MinMax_pats<setult, A2_minu, A2_maxu>;
}

class T_cmp64_rr_pat<InstHexagon MI, PatFrag CmpOp>
  : Pat<(i1 (CmpOp I64:$Rs, I64:$Rt)),
        (i1 (MI DoubleRegs:$Rs, DoubleRegs:$Rt))>;

def: T_cmp64_rr_pat<C2_cmpeqp,  seteq>;
def: T_cmp64_rr_pat<C2_cmpgtp,  setgt>;
def: T_cmp64_rr_pat<C2_cmpgtup, setugt>;
def: T_cmp64_rr_pat<C2_cmpgtp,  RevCmp<setlt>>;
def: T_cmp64_rr_pat<C2_cmpgtup, RevCmp<setult>>;

def: Pat<(i64 (add I64:$Rs, I64:$Rt)), (A2_addp I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (sub I64:$Rs, I64:$Rt)), (A2_subp I64:$Rs, I64:$Rt)>;

def: Pat<(i64 (and I64:$Rs, I64:$Rt)), (A2_andp I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (or  I64:$Rs, I64:$Rt)), (A2_orp  I64:$Rs, I64:$Rt)>;
def: Pat<(i64 (xor I64:$Rs, I64:$Rt)), (A2_xorp I64:$Rs, I64:$Rt)>;

def: Pat<(i1 (not I1:$Ps)), (C2_not PredRegs:$Ps)>;

def: Pat<(i1 (and I1:$Ps, I1:$Pt)),       (C2_and  I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (or  I1:$Ps, I1:$Pt)),       (C2_or   I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (xor I1:$Ps, I1:$Pt)),       (C2_xor  I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (and I1:$Ps, (not I1:$Pt))), (C2_andn I1:$Ps, I1:$Pt)>;
def: Pat<(i1 (or  I1:$Ps, (not I1:$Pt))), (C2_orn  I1:$Ps, I1:$Pt)>;

def retflag : SDNode<"HexagonISD::RET_FLAG", SDTNone,
                     [SDNPHasChain, SDNPOptInGlue, SDNPVariadic]>;
def eh_return: SDNode<"HexagonISD::EH_RETURN", SDTNone, [SDNPHasChain]>;

def: Pat<(br bb:$dst),                  (J2_jump b30_2Imm:$dst)>;
def: Pat<(brcond I1:$src1, bb:$block),  (J2_jumpt PredRegs:$src1, bb:$block)>;
def: Pat<(brind I32:$dst),              (J2_jumpr IntRegs:$dst)>;

def: Pat<(retflag),   (PS_jmpret (i32 R31))>;
def: Pat<(eh_return), (EH_RETURN_JMPR (i32 R31))>;

// Patterns to select load-indexed (i.e. load from base+offset).
multiclass Loadx_pat<PatFrag Load, ValueType VT, PatLeaf ImmPred,
                     InstHexagon MI> {
  def: Pat<(VT (Load AddrFI:$fi)), (VT (MI AddrFI:$fi, 0))>;
  def: Pat<(VT (Load (add (i32 AddrFI:$fi), ImmPred:$Off))),
           (VT (MI AddrFI:$fi, imm:$Off))>;
  def: Pat<(VT (Load (IsOrAdd (i32 AddrFI:$fi), ImmPred:$Off))),
           (VT (MI AddrFI:$fi, imm:$Off))>;
  def: Pat<(VT (Load (add I32:$Rs, ImmPred:$Off))),
           (VT (MI IntRegs:$Rs, imm:$Off))>;
  def: Pat<(VT (Load I32:$Rs)), (VT (MI IntRegs:$Rs, 0))>;
}

let AddedComplexity = 20 in {
  defm: Loadx_pat<load,           i32, s30_2ImmPred, L2_loadri_io>;
  defm: Loadx_pat<load,           i64, s29_3ImmPred, L2_loadrd_io>;
  defm: Loadx_pat<atomic_load_8 , i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<atomic_load_16, i32, s31_1ImmPred, L2_loadruh_io>;
  defm: Loadx_pat<atomic_load_32, i32, s30_2ImmPred, L2_loadri_io>;
  defm: Loadx_pat<atomic_load_64, i64, s29_3ImmPred, L2_loadrd_io>;

  defm: Loadx_pat<extloadi1,      i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<extloadi8,      i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<extloadi16,     i32, s31_1ImmPred, L2_loadruh_io>;
  defm: Loadx_pat<sextloadi8,     i32, s32_0ImmPred, L2_loadrb_io>;
  defm: Loadx_pat<sextloadi16,    i32, s31_1ImmPred, L2_loadrh_io>;
  defm: Loadx_pat<zextloadi1,     i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<zextloadi8,     i32, s32_0ImmPred, L2_loadrub_io>;
  defm: Loadx_pat<zextloadi16,    i32, s31_1ImmPred, L2_loadruh_io>;
  // No sextloadi1.
}

// Sign-extending loads of i1 need to replicate the lowest bit throughout
// the 32-bit value. Since the loaded value can only be 0 or 1, 0-v should
// do the trick.
let AddedComplexity = 20 in
def: Pat<(i32 (sextloadi1 I32:$Rs)),
         (A2_subri 0, (L2_loadrub_io IntRegs:$Rs, 0))>;

def: Pat<(i32 (mul   I32:$src1, I32:$src2)), (M2_mpyi    I32:$src1, I32:$src2)>;
def: Pat<(i32 (mulhs I32:$src1, I32:$src2)), (M2_mpy_up  I32:$src1, I32:$src2)>;
def: Pat<(i32 (mulhu I32:$src1, I32:$src2)), (M2_mpyu_up I32:$src1, I32:$src2)>;

def: Pat<(mul IntRegs:$Rs, u32_0ImmPred:$u8),
         (M2_mpysip IntRegs:$Rs, imm:$u8)>;
def: Pat<(ineg (mul IntRegs:$Rs, u8_0ImmPred:$u8)),
         (M2_mpysin IntRegs:$Rs, imm:$u8)>;
def: Pat<(mul IntRegs:$src1, s32_0ImmPred:$src2),
         (M2_mpysmi IntRegs:$src1, imm:$src2)>;
def: Pat<(add (mul IntRegs:$src2, u32_0ImmPred:$src3), IntRegs:$src1),
         (M2_macsip IntRegs:$src1, IntRegs:$src2, imm:$src3)>;
def: Pat<(add (mul I32:$src2, I32:$src3), I32:$src1),
         (M2_maci IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;
def: Pat<(add (add IntRegs:$src2, s32_0ImmPred:$src3), IntRegs:$src1),
         (M2_accii IntRegs:$src1, IntRegs:$src2, imm:$src3)>;
def: Pat<(add (add I32:$src2, I32:$src3), I32:$src1),
         (M2_acci IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

class T_MType_acc_pat1 <InstHexagon MI, SDNode firstOp, SDNode secOp,
                        PatLeaf ImmPred>
  : Pat <(secOp IntRegs:$src1, (firstOp IntRegs:$src2, ImmPred:$src3)),
         (MI IntRegs:$src1, IntRegs:$src2, ImmPred:$src3)>;

class T_MType_acc_pat2 <InstHexagon MI, SDNode firstOp, SDNode secOp>
  : Pat <(i32 (secOp IntRegs:$src1, (firstOp IntRegs:$src2, IntRegs:$src3))),
         (MI IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def : T_MType_acc_pat2 <M2_xor_xacc, xor, xor>;
def : T_MType_acc_pat1 <M2_macsin, mul, sub, u32_0ImmPred>;

def : T_MType_acc_pat1 <M2_naccii, add, sub, s32_0ImmPred>;
def : T_MType_acc_pat2 <M2_nacci, add, sub>;

def: T_MType_acc_pat2 <M4_or_xor, xor, or>;
def: T_MType_acc_pat2 <M4_and_xor, xor, and>;
def: T_MType_acc_pat2 <M4_or_and, and, or>;
def: T_MType_acc_pat2 <M4_and_and, and, and>;
def: T_MType_acc_pat2 <M4_xor_and, and, xor>;
def: T_MType_acc_pat2 <M4_or_or, or, or>;
def: T_MType_acc_pat2 <M4_and_or, or, and>;
def: T_MType_acc_pat2 <M4_xor_or, or, xor>;

class T_MType_acc_pat3 <InstHexagon MI, SDNode firstOp, SDNode secOp>
  : Pat <(secOp I32:$src1, (firstOp I32:$src2, (not I32:$src3))),
         (MI IntRegs:$src1, IntRegs:$src2, IntRegs:$src3)>;

def: T_MType_acc_pat3 <M4_or_andn, and, or>;
def: T_MType_acc_pat3 <M4_and_andn, and, and>;
def: T_MType_acc_pat3 <M4_xor_andn, and, xor>;

// This complex pattern is really only to detect various forms of
// sign-extension i32->i64. The selected value will be of type i64
// whose low word is the value being extended. The high word is
// unspecified.
def Usxtw : ComplexPattern<i64, 1, "DetectUseSxtw", [], []>;

def Aext64: PatFrag<(ops node:$Rs), (i64 (anyext node:$Rs))>;
def Zext64: PatFrag<(ops node:$Rs), (i64 (zext node:$Rs))>;
def Sext64: PatLeaf<(i64 Usxtw:$Rs)>;

def: Pat<(i32 (trunc (sra (mul Sext64:$Rs, Sext64:$Rt), (i32 32)))),
         (M2_mpy_up (LoReg Sext64:$Rs), (LoReg Sext64:$Rt))>;
def: Pat<(i32 (trunc (srl (mul Sext64:$Rs, Sext64:$Rt), (i32 32)))),
         (M2_mpy_up (LoReg Sext64:$Rs), (LoReg Sext64:$Rt))>;

def: Pat<(mul (Aext64 I32:$Rs), (Aext64 I32:$Rt)),
         (M2_dpmpyuu_s0 I32:$Rs, I32:$Rt)>;

def: Pat<(mul Sext64:$Rs, Sext64:$Rt),
         (M2_dpmpyss_s0 (LoReg Sext64:$Rs), (LoReg Sext64:$Rt))>;

// Multiply and accumulate, use full result.
// Rxx[+-]=mpy(Rs,Rt)

def: Pat<(add I64:$Rx, (mul Sext64:$Rs, Sext64:$Rt)),
         (M2_dpmpyss_acc_s0 I64:$Rx, (LoReg Sext64:$Rs), (LoReg Sext64:$Rt))>;

def: Pat<(sub I64:$Rx, (mul Sext64:$Rs, Sext64:$Rt)),
         (M2_dpmpyss_nac_s0 I64:$Rx, (LoReg Sext64:$Rs), (LoReg Sext64:$Rt))>;

def: Pat<(add I64:$Rx, (mul (Aext64 I32:$Rs), (Aext64 I32:$Rt))),
         (M2_dpmpyuu_acc_s0 I64:$Rx, I32:$Rs, I32:$Rt)>;

def: Pat<(add I64:$Rx, (mul (Zext64 I32:$Rs), (Zext64 I32:$Rt))),
         (M2_dpmpyuu_acc_s0 I64:$Rx, I32:$Rs, I32:$Rt)>;

def: Pat<(sub I64:$Rx, (mul (Aext64 I32:$Rs), (Aext64 I32:$Rt))),
         (M2_dpmpyuu_nac_s0 I64:$Rx, I32:$Rs, I32:$Rt)>;

def: Pat<(sub I64:$Rx, (mul (Zext64 I32:$Rs), (Zext64 I32:$Rt))),
         (M2_dpmpyuu_nac_s0 I64:$Rx, I32:$Rs, I32:$Rt)>;

class Storepi_pat<PatFrag Store, PatFrag Value, PatFrag Offset,
                  InstHexagon MI>
  : Pat<(Store Value:$src1, I32:$src2, Offset:$offset),
        (MI I32:$src2, imm:$offset, Value:$src1)>;

def: Storepi_pat<post_truncsti8,  I32, s4_0ImmPred, S2_storerb_pi>;
def: Storepi_pat<post_truncsti16, I32, s4_1ImmPred, S2_storerh_pi>;
def: Storepi_pat<post_store,      I32, s4_2ImmPred, S2_storeri_pi>;
def: Storepi_pat<post_store,      I64, s4_3ImmPred, S2_storerd_pi>;

// Patterns for generating stores, where the address takes different forms:
// - frameindex,
// - frameindex + offset,
// - base + offset,
// - simple (base address without offset).
// These would usually be used together (via Storex_pat defined below), but
// in some cases one may want to apply different properties (such as
// AddedComplexity) to the individual patterns.
class Storex_fi_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Rs, AddrFI:$fi), (MI AddrFI:$fi, 0, Value:$Rs)>;
multiclass Storex_fi_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                             InstHexagon MI> {
  def: Pat<(Store Value:$Rs, (add (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, Value:$Rs)>;
  def: Pat<(Store Value:$Rs, (IsOrAdd (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, Value:$Rs)>;
}
multiclass Storex_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                          InstHexagon MI> {
  def: Pat<(Store Value:$Rt, (add I32:$Rs, ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, Value:$Rt)>;
  def: Pat<(Store Value:$Rt, (IsOrAdd I32:$Rs, ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, Value:$Rt)>;
}
class Storex_simple_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Rt, I32:$Rs),
        (MI IntRegs:$Rs, 0, Value:$Rt)>;

// Patterns for generating stores, where the address takes different forms,
// and where the value being stored is transformed through the value modifier
// ValueMod.  The address forms are same as above.
class Storexm_fi_pat<PatFrag Store, PatFrag Value, PatFrag ValueMod,
                     InstHexagon MI>
  : Pat<(Store Value:$Rs, AddrFI:$fi),
        (MI AddrFI:$fi, 0, (ValueMod Value:$Rs))>;
multiclass Storexm_fi_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                              PatFrag ValueMod, InstHexagon MI> {
  def: Pat<(Store Value:$Rs, (add (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, (ValueMod Value:$Rs))>;
  def: Pat<(Store Value:$Rs, (IsOrAdd (i32 AddrFI:$fi), ImmPred:$Off)),
           (MI AddrFI:$fi, imm:$Off, (ValueMod Value:$Rs))>;
}
multiclass Storexm_add_pat<PatFrag Store, PatFrag Value, PatFrag ImmPred,
                           PatFrag ValueMod, InstHexagon MI> {
  def: Pat<(Store Value:$Rt, (add I32:$Rs, ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, (ValueMod Value:$Rt))>;
  def: Pat<(Store Value:$Rt, (IsOrAdd I32:$Rs, ImmPred:$Off)),
           (MI IntRegs:$Rs, imm:$Off, (ValueMod Value:$Rt))>;
}
class Storexm_simple_pat<PatFrag Store, PatFrag Value, PatFrag ValueMod,
                         InstHexagon MI>
  : Pat<(Store Value:$Rt, I32:$Rs),
        (MI IntRegs:$Rs, 0, (ValueMod Value:$Rt))>;

multiclass Storex_pat<PatFrag Store, PatFrag Value, PatLeaf ImmPred,
                      InstHexagon MI> {
  def:  Storex_fi_pat     <Store, Value,          MI>;
  defm: Storex_fi_add_pat <Store, Value, ImmPred, MI>;
  defm: Storex_add_pat    <Store, Value, ImmPred, MI>;
}

multiclass Storexm_pat<PatFrag Store, PatFrag Value, PatLeaf ImmPred,
                       PatFrag ValueMod, InstHexagon MI> {
  def:  Storexm_fi_pat     <Store, Value,          ValueMod, MI>;
  defm: Storexm_fi_add_pat <Store, Value, ImmPred, ValueMod, MI>;
  defm: Storexm_add_pat    <Store, Value, ImmPred, ValueMod, MI>;
}

// Regular stores in the DAG have two operands: value and address.
// Atomic stores also have two, but they are reversed: address, value.
// To use atomic stores with the patterns, they need to have their operands
// swapped. This relies on the knowledge that the F.Fragment uses names
// "ptr" and "val".
class SwapSt<PatFrag F>
  : PatFrag<(ops node:$val, node:$ptr), F.Fragment, F.PredicateCode,
            F.OperandTransform>;

let AddedComplexity = 20 in {
  defm: Storex_pat<truncstorei8,    I32, s32_0ImmPred, S2_storerb_io>;
  defm: Storex_pat<truncstorei16,   I32, s31_1ImmPred, S2_storerh_io>;
  defm: Storex_pat<store,           I32, s30_2ImmPred, S2_storeri_io>;
  defm: Storex_pat<store,           I64, s29_3ImmPred, S2_storerd_io>;

  defm: Storex_pat<SwapSt<atomic_store_8>,  I32, s32_0ImmPred, S2_storerb_io>;
  defm: Storex_pat<SwapSt<atomic_store_16>, I32, s31_1ImmPred, S2_storerh_io>;
  defm: Storex_pat<SwapSt<atomic_store_32>, I32, s30_2ImmPred, S2_storeri_io>;
  defm: Storex_pat<SwapSt<atomic_store_64>, I64, s29_3ImmPred, S2_storerd_io>;
}

// Simple patterns should be tried with the least priority.
def: Storex_simple_pat<truncstorei8,    I32, S2_storerb_io>;
def: Storex_simple_pat<truncstorei16,   I32, S2_storerh_io>;
def: Storex_simple_pat<store,           I32, S2_storeri_io>;
def: Storex_simple_pat<store,           I64, S2_storerd_io>;

def: Storex_simple_pat<SwapSt<atomic_store_8>,  I32, S2_storerb_io>;
def: Storex_simple_pat<SwapSt<atomic_store_16>, I32, S2_storerh_io>;
def: Storex_simple_pat<SwapSt<atomic_store_32>, I32, S2_storeri_io>;
def: Storex_simple_pat<SwapSt<atomic_store_64>, I64, S2_storerd_io>;

let AddedComplexity = 20 in {
  defm: Storexm_pat<truncstorei8,  I64, s32_0ImmPred, LoReg, S2_storerb_io>;
  defm: Storexm_pat<truncstorei16, I64, s31_1ImmPred, LoReg, S2_storerh_io>;
  defm: Storexm_pat<truncstorei32, I64, s30_2ImmPred, LoReg, S2_storeri_io>;
}

def: Storexm_simple_pat<truncstorei8,  I64, LoReg, S2_storerb_io>;
def: Storexm_simple_pat<truncstorei16, I64, LoReg, S2_storerh_io>;
def: Storexm_simple_pat<truncstorei32, I64, LoReg, S2_storeri_io>;

def: Pat <(i64 (sext I32:$src)),            (A2_sxtw I32:$src)>;
def: Pat <(i64 (sext_inreg I64:$src, i32)), (A2_sxtw (LoReg I64:$src))>;

def: Pat<(select (i1 (setlt I32:$src, 0)), (sub 0, I32:$src), I32:$src),
         (A2_abs IntRegs:$src)>;

let AddedComplexity = 50 in
def: Pat<(xor (add (sra I32:$src, (i32 31)),
                   I32:$src),
              (sra I32:$src, (i32 31))),
         (A2_abs IntRegs:$src)>;

def: Pat<(sra I32:$src, u5_0ImmPred:$u5),
         (S2_asr_i_r IntRegs:$src, imm:$u5)>;
def: Pat<(srl I32:$src, u5_0ImmPred:$u5),
         (S2_lsr_i_r IntRegs:$src, imm:$u5)>;
def: Pat<(shl I32:$src, u5_0ImmPred:$u5),
         (S2_asl_i_r IntRegs:$src, imm:$u5)>;

def: Pat<(sra (add (sra I32:$src1, u5_0ImmPred:$src2), 1), (i32 1)),
         (S2_asr_i_r_rnd IntRegs:$src1, u5_0ImmPred:$src2)>;

def : Pat<(not I64:$src1),
          (A2_notp DoubleRegs:$src1)>;

// Count leading zeros.
def: Pat<(ctlz I32:$Rs), (S2_cl0 I32:$Rs)>;
def: Pat<(i32 (trunc (ctlz I64:$Rss))), (S2_cl0p I64:$Rss)>;

// Count trailing zeros: 32-bit.
def: Pat<(cttz I32:$Rs), (S2_ct0 I32:$Rs)>;

// Count leading ones.
def: Pat<(ctlz (not I32:$Rs)), (S2_cl1 I32:$Rs)>;
def: Pat<(i32 (trunc (ctlz (not I64:$Rss)))), (S2_cl1p I64:$Rss)>;

// Count trailing ones: 32-bit.
def: Pat<(cttz (not I32:$Rs)), (S2_ct1 I32:$Rs)>;

let AddedComplexity = 20 in { // Complexity greater than and/or/xor
  def: Pat<(and I32:$Rs, IsNPow2_32:$V),
           (S2_clrbit_i IntRegs:$Rs, (LogN2_32 $V))>;
  def: Pat<(or I32:$Rs, IsPow2_32:$V),
           (S2_setbit_i IntRegs:$Rs, (Log2_32 $V))>;
  def: Pat<(xor I32:$Rs, IsPow2_32:$V),
           (S2_togglebit_i IntRegs:$Rs, (Log2_32 $V))>;

  def: Pat<(and I32:$Rs, (not (shl 1, I32:$Rt))),
           (S2_clrbit_r IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(or I32:$Rs, (shl 1, I32:$Rt)),
           (S2_setbit_r IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(xor I32:$Rs, (shl 1, I32:$Rt)),
           (S2_togglebit_r IntRegs:$Rs, IntRegs:$Rt)>;
}

// Clr/set/toggle bit for 64-bit values with immediate bit index.
let AddedComplexity = 20 in { // Complexity greater than and/or/xor
  def: Pat<(and I64:$Rss, IsNPow2_64L:$V),
           (REG_SEQUENCE DoubleRegs,
                (i32 (HiReg $Rss)), isub_hi,
                (S2_clrbit_i (LoReg $Rss), (LogN2_64 $V)), isub_lo)>;
  def: Pat<(and I64:$Rss, IsNPow2_64H:$V),
           (REG_SEQUENCE DoubleRegs,
                (S2_clrbit_i (HiReg $Rss), (UDEC32 (i32 (LogN2_64 $V)))),
                isub_hi,
                (i32 (LoReg $Rss)), isub_lo)>;

  def: Pat<(or I64:$Rss, IsPow2_64L:$V),
           (REG_SEQUENCE DoubleRegs,
                (i32 (HiReg $Rss)), isub_hi,
                (S2_setbit_i (LoReg $Rss), (Log2_64 $V)), isub_lo)>;
  def: Pat<(or I64:$Rss, IsPow2_64H:$V),
           (REG_SEQUENCE DoubleRegs,
                (S2_setbit_i (HiReg $Rss), (UDEC32 (i32 (Log2_64 $V)))),
                isub_hi,
                (i32 (LoReg $Rss)), isub_lo)>;

  def: Pat<(xor I64:$Rss, IsPow2_64L:$V),
           (REG_SEQUENCE DoubleRegs,
                (i32 (HiReg $Rss)), isub_hi,
                (S2_togglebit_i (LoReg $Rss), (Log2_64 $V)), isub_lo)>;
  def: Pat<(xor I64:$Rss, IsPow2_64H:$V),
           (REG_SEQUENCE DoubleRegs,
                (S2_togglebit_i (HiReg $Rss), (UDEC32 (i32 (Log2_64 $V)))),
                isub_hi,
                (i32 (LoReg $Rss)), isub_lo)>;
}

let AddedComplexity = 20 in { // Complexity greater than cmp reg-imm.
  def: Pat<(i1 (setne (and (shl 1, u5_0ImmPred:$u5), I32:$Rs), 0)),
           (S2_tstbit_i IntRegs:$Rs, u5_0ImmPred:$u5)>;
  def: Pat<(i1 (setne (and (shl 1, I32:$Rt), I32:$Rs), 0)),
           (S2_tstbit_r IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(i1 (trunc I32:$Rs)),
           (S2_tstbit_i IntRegs:$Rs, 0)>;
  def: Pat<(i1 (trunc I64:$Rs)),
           (S2_tstbit_i (LoReg DoubleRegs:$Rs), 0)>;
}

let AddedComplexity = 20 in { // Complexity greater than compare reg-imm.
  def: Pat<(i1 (seteq (and I32:$Rs, u6_0ImmPred:$u6), 0)),
           (C2_bitsclri IntRegs:$Rs, u6_0ImmPred:$u6)>;
  def: Pat<(i1 (seteq (and I32:$Rs, I32:$Rt), 0)),
           (C2_bitsclr IntRegs:$Rs, IntRegs:$Rt)>;
}

let AddedComplexity = 10 in   // Complexity greater than compare reg-reg.
def: Pat<(i1 (seteq (and I32:$Rs, I32:$Rt), IntRegs:$Rt)),
         (C2_bitsset IntRegs:$Rs, IntRegs:$Rt)>;

def: Pat<(or (or (shl (or (shl (i32 (extloadi8 (add I32:$b, 3))),
                               (i32 8)),
                          (i32 (zextloadi8 (add I32:$b, 2)))),
                      (i32 16)),
                 (shl (i32 (zextloadi8 (add I32:$b, 1))), (i32 8))),
             (zextloadi8 I32:$b)),
         (A2_swiz (L2_loadri_io IntRegs:$b, 0))>;

// Patterns for loads of i1:
def: Pat<(i1 (load AddrFI:$fi)),
         (C2_tfrrp (L2_loadrub_io AddrFI:$fi, 0))>;
def: Pat<(i1 (load (add I32:$Rs, s32_0ImmPred:$Off))),
         (C2_tfrrp (L2_loadrub_io IntRegs:$Rs, imm:$Off))>;
def: Pat<(i1 (load I32:$Rs)),
         (C2_tfrrp (L2_loadrub_io IntRegs:$Rs, 0))>;

def I1toI32: OutPatFrag<(ops node:$Rs),
                        (C2_muxii (i1 $Rs), 1, 0)>;

def I32toI1: OutPatFrag<(ops node:$Rs),
                        (i1 (C2_tfrrp (i32 $Rs)))>;

defm: Storexm_pat<store, I1, s32_0ImmPred, I1toI32, S2_storerb_io>;
def: Storexm_simple_pat<store, I1, I1toI32, S2_storerb_io>;

def: Pat<(sra (add (sra I64:$src, u6_0ImmPred:$u6), 1), (i32 1)),
         (S2_asr_i_p_rnd DoubleRegs:$src, imm:$u6)>, Requires<[HasV5T]>;
def: Pat<(sra I64:$src, u6_0ImmPred:$u6),
         (S2_asr_i_p DoubleRegs:$src, imm:$u6)>;
def: Pat<(srl I64:$src, u6_0ImmPred:$u6),
         (S2_lsr_i_p DoubleRegs:$src, imm:$u6)>;
def: Pat<(shl I64:$src, u6_0ImmPred:$u6),
         (S2_asl_i_p DoubleRegs:$src, imm:$u6)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$Rt, (shl I32:$Rs, u3_0ImmPred:$u3)),
         (S2_addasl_rrri IntRegs:$Rt, IntRegs:$Rs, imm:$u3)>;

def HexagonBARRIER: SDNode<"HexagonISD::BARRIER", SDTNone, [SDNPHasChain]>;
def: Pat<(HexagonBARRIER), (Y2_barrier)>;

def: Pat<(IsOrAdd (i32 AddrFI:$Rs), s32_0ImmPred:$off),
         (PS_fi (i32 AddrFI:$Rs), s32_0ImmPred:$off)>;


// Support for generating global address.
// Taken from X86InstrInfo.td.
def SDTHexagonCONST32 : SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                             SDTCisVT<1, i32>,
                                             SDTCisPtrTy<0>]>;
def HexagonCONST32    : SDNode<"HexagonISD::CONST32",    SDTHexagonCONST32>;
def HexagonCONST32_GP : SDNode<"HexagonISD::CONST32_GP", SDTHexagonCONST32>;

// Map TLS addressses to A2_tfrsi.
def: Pat<(HexagonCONST32 tglobaltlsaddr:$addr), (A2_tfrsi s32_0Imm:$addr)>;
def: Pat<(HexagonCONST32 bbl:$label),           (A2_tfrsi s32_0Imm:$label)>;

def: Pat<(i64 imm:$v), (CONST64 imm:$v)>;
def: Pat<(i1 0), (PS_false)>;
def: Pat<(i1 1), (PS_true)>;

// Pseudo instructions.
def SDT_SPCallSeqStart : SDCallSeqStart<[ SDTCisVT<0, i32>,
                                          SDTCisVT<1, i32> ]>;
def SDT_SPCallSeqEnd   : SDCallSeqEnd<[ SDTCisVT<0, i32>,
                                        SDTCisVT<1, i32> ]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START", SDT_SPCallSeqStart,
                    [SDNPHasChain, SDNPOutGlue]>;
def callseq_end   : SDNode<"ISD::CALLSEQ_END",   SDT_SPCallSeqEnd,
                    [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;

def SDT_SPCall  : SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>;

// For tailcalls a HexagonTCRet SDNode has 3 SDNode Properties - a chain,
// Optional Flag and Variable Arguments.
// Its 1 Operand has pointer type.
def HexagonTCRet : SDNode<"HexagonISD::TC_RETURN", SDT_SPCall,
                          [SDNPHasChain,  SDNPOptInGlue, SDNPVariadic]>;


def: Pat<(callseq_start timm:$amt, timm:$amt2),
          (ADJCALLSTACKDOWN imm:$amt, imm:$amt2)>;
def: Pat<(callseq_end timm:$amt1, timm:$amt2),
         (ADJCALLSTACKUP imm:$amt1, imm:$amt2)>;

//Tail calls.
def: Pat<(HexagonTCRet tglobaladdr:$dst),
         (PS_tailcall_i tglobaladdr:$dst)>;
def: Pat<(HexagonTCRet texternalsym:$dst),
         (PS_tailcall_i texternalsym:$dst)>;
def: Pat<(HexagonTCRet I32:$dst),
         (PS_tailcall_r I32:$dst)>;

// Map from r0 = and(r1, 65535) to r0 = zxth(r1)
def: Pat<(and I32:$src1, 65535),
         (A2_zxth IntRegs:$src1)>;

// Map from r0 = and(r1, 255) to r0 = zxtb(r1).
def: Pat<(and I32:$src1, 255),
         (A2_zxtb IntRegs:$src1)>;

// Map Add(p1, true) to p1 = not(p1).
//     Add(p1, false) should never be produced,
//     if it does, it got to be mapped to NOOP.
def: Pat<(add I1:$src1, -1),
         (C2_not PredRegs:$src1)>;

// Map from p0 = pnot(p0); r0 = mux(p0, #i, #j) => r0 = mux(p0, #j, #i).
def: Pat<(select (not I1:$src1), s8_0ImmPred:$src2, s32_0ImmPred:$src3),
         (C2_muxii PredRegs:$src1, s32_0ImmPred:$src3, s8_0ImmPred:$src2)>;

// Map from p0 = pnot(p0); r0 = select(p0, #i, r1)
// => r0 = C2_muxir(p0, r1, #i)
def: Pat<(select (not I1:$src1), s32_0ImmPred:$src2,
                 I32:$src3),
         (C2_muxir PredRegs:$src1, IntRegs:$src3, s32_0ImmPred:$src2)>;

// Map from p0 = pnot(p0); r0 = mux(p0, r1, #i)
// => r0 = C2_muxri (p0, #i, r1)
def: Pat<(select (not I1:$src1), IntRegs:$src2, s32_0ImmPred:$src3),
         (C2_muxri PredRegs:$src1, s32_0ImmPred:$src3, IntRegs:$src2)>;

// Map from p0 = pnot(p0); if (p0) jump => if (!p0) jump.
def: Pat<(brcond (not I1:$src1), bb:$offset),
         (J2_jumpf PredRegs:$src1, bb:$offset)>;

// Map from Rdd = sign_extend_inreg(Rss, i32) -> Rdd = A2_sxtw(Rss.lo).
def: Pat<(i64 (sext_inreg I64:$src1, i32)),
         (A2_sxtw (LoReg DoubleRegs:$src1))>;

// Map from Rdd = sign_extend_inreg(Rss, i16) -> Rdd = A2_sxtw(A2_sxth(Rss.lo)).
def: Pat<(i64 (sext_inreg I64:$src1, i16)),
         (A2_sxtw (A2_sxth (LoReg DoubleRegs:$src1)))>;

// Map from Rdd = sign_extend_inreg(Rss, i8) -> Rdd = A2_sxtw(A2_sxtb(Rss.lo)).
def: Pat<(i64 (sext_inreg I64:$src1, i8)),
         (A2_sxtw (A2_sxtb (LoReg DoubleRegs:$src1)))>;

def: Pat<(brcond (i1 (setne I32:$Rs, I32:$Rt)), bb:$offset),
         (J2_jumpf (C2_cmpeq I32:$Rs, I32:$Rt), bb:$offset)>;
def: Pat<(brcond (i1 (setne I32:$Rs, s10_0ImmPred:$s10)), bb:$offset),
         (J2_jumpf (C2_cmpeqi I32:$Rs, imm:$s10), bb:$offset)>;
def: Pat<(brcond (i1 (setne I1:$Pu, (i1 -1))), bb:$offset),
         (J2_jumpf PredRegs:$Pu, bb:$offset)>;
def: Pat<(brcond (i1 (setne I1:$Pu, (i1 0))), bb:$offset),
         (J2_jumpt PredRegs:$Pu, bb:$offset)>;

// cmp.lt(Rs, Imm) -> !cmp.ge(Rs, Imm) -> !cmp.gt(Rs, Imm-1)
def: Pat<(brcond (i1 (setlt I32:$Rs, s8_0ImmPred:$s8)), bb:$offset),
         (J2_jumpf (C2_cmpgti IntRegs:$Rs, (SDEC1 imm:$s8)), bb:$offset)>;


// Map from a 64-bit select to an emulated 64-bit mux.
// Hexagon does not support 64-bit MUXes; so emulate with combines.
def: Pat<(select I1:$src1, I64:$src2,
                 I64:$src3),
         (A2_combinew (C2_mux PredRegs:$src1, (HiReg DoubleRegs:$src2),
                                              (HiReg DoubleRegs:$src3)),
                      (C2_mux PredRegs:$src1, (LoReg DoubleRegs:$src2),
                                              (LoReg DoubleRegs:$src3)))>;

// Map from a 1-bit select to logical ops.
// From LegalizeDAG.cpp: (B1 ? B2 : B3) <=> (B1 & B2)|(!B1&B3).
def: Pat<(select I1:$src1, I1:$src2, I1:$src3),
         (C2_or (C2_and PredRegs:$src1, PredRegs:$src2),
                (C2_and (C2_not PredRegs:$src1), PredRegs:$src3))>;

// Map for truncating from 64 immediates to 32 bit immediates.
def: Pat<(i32 (trunc I64:$src)),
         (LoReg DoubleRegs:$src)>;

// Map for truncating from i64 immediates to i1 bit immediates.
def: Pat<(i1 (trunc I64:$src)),
         (C2_tfrrp (LoReg DoubleRegs:$src))>;

// rs <= rt -> !(rs > rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setle I32:$src1, s32_0ImmPred:$src2)),
         (C2_not (C2_cmpgti IntRegs:$src1, s32_0ImmPred:$src2))>;

// rs <= rt -> !(rs > rt).
def : Pat<(i1 (setle I32:$src1, I32:$src2)),
      (i1 (C2_not (C2_cmpgt I32:$src1, I32:$src2)))>;

// Rss <= Rtt -> !(Rss > Rtt).
def: Pat<(i1 (setle I64:$src1, I64:$src2)),
         (C2_not (C2_cmpgtp DoubleRegs:$src1, DoubleRegs:$src2))>;

// Map cmpne -> cmpeq.
// Hexagon_TODO: We should improve on this.
// rs != rt -> !(rs == rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setne I32:$src1, s32_0ImmPred:$src2)),
         (C2_not (C2_cmpeqi IntRegs:$src1, s32_0ImmPred:$src2))>;

// Convert setne back to xor for hexagon since we compute w/ pred registers.
def: Pat<(i1 (setne I1:$src1, I1:$src2)),
         (C2_xor PredRegs:$src1, PredRegs:$src2)>;

// Map cmpne(Rss) -> !cmpew(Rss).
// rs != rt -> !(rs == rt).
def: Pat<(i1 (setne I64:$src1, I64:$src2)),
         (C2_not (C2_cmpeqp DoubleRegs:$src1, DoubleRegs:$src2))>;

// rs >= rt -> rt <= rs
def: Pat<(i1 (setge I32:$Rs, I32:$Rt)),
         (C4_cmplte I32:$Rt, I32:$Rs)>;

let AddedComplexity = 30 in
def: Pat<(i1 (setge I32:$Rs, s32_0ImmPred:$s10)),
         (C2_cmpgti IntRegs:$Rs, (SDEC1 imm:$s10))>;

// Map cmpge(Rss, Rtt) -> !cmpgt(Rtt, Rss).
// rss >= rtt -> !(rtt > rss).
def: Pat<(i1 (setge I64:$src1, I64:$src2)),
         (C2_not (C2_cmpgtp DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map cmplt(Rs, Imm) -> !cmpge(Rs, Imm).
// !cmpge(Rs, Imm) -> !cmpgt(Rs, Imm-1).
// rs < rt -> !(rs >= rt).
let AddedComplexity = 30 in
def: Pat<(i1 (setlt I32:$src1, s32_0ImmPred:$src2)),
         (C2_not (C2_cmpgti IntRegs:$src1, (SDEC1 s32_0ImmPred:$src2)))>;

// Generate cmpgeu(Rs, #0) -> cmpeq(Rs, Rs)
def: Pat<(i1 (setuge I32:$src1, 0)),
         (C2_cmpeq IntRegs:$src1, IntRegs:$src1)>;

// Generate cmpgeu(Rs, #u8) -> cmpgtu(Rs, #u8 -1)
def: Pat<(i1 (setuge I32:$src1, u32_0ImmPred:$src2)),
         (C2_cmpgtui IntRegs:$src1, (UDEC1 u32_0ImmPred:$src2))>;

// Generate cmpgtu(Rs, #u9)
def: Pat<(i1 (setugt I32:$src1, u32_0ImmPred:$src2)),
         (C2_cmpgtui IntRegs:$src1, u32_0ImmPred:$src2)>;

// Map from Rs >= Rt -> !(Rt > Rs).
// rs >= rt -> !(rt > rs).
def: Pat<(i1 (setuge I64:$src1, I64:$src2)),
         (C2_not (C2_cmpgtup DoubleRegs:$src2, DoubleRegs:$src1))>;

// Map from cmpleu(Rss, Rtt) -> !cmpgtu(Rss, Rtt-1).
// Map from (Rs <= Rt) -> !(Rs > Rt).
def: Pat<(i1 (setule I64:$src1, I64:$src2)),
         (C2_not (C2_cmpgtup DoubleRegs:$src1, DoubleRegs:$src2))>;

// Sign extends.
// sext i1->i32
def: Pat<(i32 (sext I1:$Pu)),
         (C2_muxii I1:$Pu, -1, 0)>;

// sext i1->i64
def: Pat<(i64 (sext I1:$Pu)),
         (A2_combinew (C2_muxii PredRegs:$Pu, -1, 0),
                      (C2_muxii PredRegs:$Pu, -1, 0))>;

// Zero extends.
// zext i1->i32
def: Pat<(i32 (zext I1:$Pu)),
         (C2_muxii PredRegs:$Pu, 1, 0)>;

// zext i1->i64
def: Pat<(i64 (zext I1:$Pu)),
         (ToZext64 (C2_muxii PredRegs:$Pu, 1, 0))>;

// zext i32->i64
def: Pat<(Zext64 I32:$Rs),
         (ToZext64 IntRegs:$Rs)>;

// Map from Rs = Pd to Pd = mux(Pd, #1, #0)
def: Pat<(i32 (anyext I1:$Pu)),
         (C2_muxii PredRegs:$Pu, 1, 0)>;

// Map from Rss = Pd to Rdd = combine(#0, (mux(Pd, #1, #0)))
def: Pat<(i64 (anyext I1:$Pu)),
         (ToZext64 (C2_muxii PredRegs:$Pu, 1, 0))>;

// Clear the sign bit in a 64-bit register.
def ClearSign : OutPatFrag<(ops node:$Rss),
  (A2_combinew (S2_clrbit_i (HiReg $Rss), 31), (LoReg $Rss))>;

def MulHU : OutPatFrag<(ops node:$Rss, node:$Rtt),
  (A2_addp
    (M2_dpmpyuu_acc_s0
      (S2_lsr_i_p
        (A2_addp
          (M2_dpmpyuu_acc_s0
            (S2_lsr_i_p (M2_dpmpyuu_s0 (LoReg $Rss), (LoReg $Rtt)), 32),
            (HiReg $Rss),
            (LoReg $Rtt)),
          (A2_combinew (A2_tfrsi 0),
                       (LoReg (M2_dpmpyuu_s0 (LoReg $Rss), (HiReg $Rtt))))),
        32),
      (HiReg $Rss),
      (HiReg $Rtt)),
    (S2_lsr_i_p (M2_dpmpyuu_s0 (LoReg $Rss), (HiReg $Rtt)), 32))>;

// Multiply 64-bit unsigned and use upper result.
def : Pat <(mulhu I64:$Rss, I64:$Rtt), (MulHU $Rss, $Rtt)>;

// Multiply 64-bit signed and use upper result.
//
// For two signed 64-bit integers A and B, let A' and B' denote A and B
// with the sign bit cleared. Then A = -2^63*s(A) + A', where s(A) is the
// sign bit of A (and identically for B). With this notation, the signed
// product A*B can be written as:
//   AB = (-2^63 s(A) + A') * (-2^63 s(B) + B')
//      = 2^126 s(A)s(B) - 2^63 [s(A)B'+s(B)A'] + A'B'
//      = 2^126 s(A)s(B) + 2^63 [s(A)B'+s(B)A'] + A'B' - 2*2^63 [s(A)B'+s(B)A']
//      = (unsigned product AB) - 2^64 [s(A)B'+s(B)A']

def : Pat <(mulhs I64:$Rss, I64:$Rtt),
  (A2_subp
    (MulHU $Rss, $Rtt),
    (A2_addp
      (A2_andp (S2_asr_i_p $Rss, 63), (ClearSign $Rtt)),
      (A2_andp (S2_asr_i_p $Rtt, 63), (ClearSign $Rss))))>;

// Hexagon specific ISD nodes.
def SDTHexagonALLOCA : SDTypeProfile<1, 2,
      [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;
def HexagonALLOCA : SDNode<"HexagonISD::ALLOCA", SDTHexagonALLOCA,
      [SDNPHasChain]>;


def: Pat<(HexagonALLOCA I32:$Rs, (i32 imm:$A)),
         (PS_alloca IntRegs:$Rs, imm:$A)>;

def HexagonJT:     SDNode<"HexagonISD::JT", SDTIntUnaryOp>;
def HexagonCP:     SDNode<"HexagonISD::CP", SDTIntUnaryOp>;

def: Pat<(HexagonJT tjumptable:$dst), (A2_tfrsi imm:$dst)>;
def: Pat<(HexagonCP tconstpool:$dst), (A2_tfrsi imm:$dst)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (sra I32:$Rs, u5_0ImmPred:$u5)), (S2_asr_i_r_acc IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(sub I32:$src1, (sra I32:$Rs, u5_0ImmPred:$u5)), (S2_asr_i_r_nac IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(and I32:$src1, (sra I32:$Rs, u5_0ImmPred:$u5)), (S2_asr_i_r_and IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(or I32:$src1, (sra I32:$Rs, u5_0ImmPred:$u5)), (S2_asr_i_r_or IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (sra I64:$Rs, u6_0ImmPred:$u5)), (S2_asr_i_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(sub I64:$src1, (sra I64:$Rs, u6_0ImmPred:$u5)), (S2_asr_i_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(and I64:$src1, (sra I64:$Rs, u6_0ImmPred:$u5)), (S2_asr_i_p_and DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(or I64:$src1, (sra I64:$Rs, u6_0ImmPred:$u5)), (S2_asr_i_p_or DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (srl I32:$Rs, u5_0ImmPred:$u5)), (S2_lsr_i_r_acc IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(sub I32:$src1, (srl I32:$Rs, u5_0ImmPred:$u5)), (S2_lsr_i_r_nac IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(and I32:$src1, (srl I32:$Rs, u5_0ImmPred:$u5)), (S2_lsr_i_r_and IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(or I32:$src1, (srl I32:$Rs, u5_0ImmPred:$u5)), (S2_lsr_i_r_or IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
let AddedComplexity = 100 in
def: Pat<(xor I32:$src1, (srl I32:$Rs, u5_0ImmPred:$u5)), (S2_lsr_i_r_xacc IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (srl I64:$Rs, u6_0ImmPred:$u5)), (S2_lsr_i_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(sub I64:$src1, (srl I64:$Rs, u6_0ImmPred:$u5)), (S2_lsr_i_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(and I64:$src1, (srl I64:$Rs, u6_0ImmPred:$u5)), (S2_lsr_i_p_and DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(or I64:$src1, (srl I64:$Rs, u6_0ImmPred:$u5)), (S2_lsr_i_p_or DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
let AddedComplexity = 100 in
def: Pat<(xor I64:$src1, (srl I64:$Rs, u6_0ImmPred:$u5)), (S2_lsr_i_p_xacc DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (shl I32:$Rs, u5_0ImmPred:$u5)), (S2_asl_i_r_acc IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(sub I32:$src1, (shl I32:$Rs, u5_0ImmPred:$u5)), (S2_asl_i_r_nac IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(and I32:$src1, (shl I32:$Rs, u5_0ImmPred:$u5)), (S2_asl_i_r_and IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
def: Pat<(or I32:$src1, (shl I32:$Rs, u5_0ImmPred:$u5)), (S2_asl_i_r_or IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;
let AddedComplexity = 100 in
def: Pat<(xor I32:$src1, (shl I32:$Rs, u5_0ImmPred:$u5)), (S2_asl_i_r_xacc IntRegs:$src1, IntRegs:$Rs, u5_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (shl I64:$Rs, u6_0ImmPred:$u5)), (S2_asl_i_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(sub I64:$src1, (shl I64:$Rs, u6_0ImmPred:$u5)), (S2_asl_i_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(and I64:$src1, (shl I64:$Rs, u6_0ImmPred:$u5)), (S2_asl_i_p_and DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
def: Pat<(or I64:$src1, (shl I64:$Rs, u6_0ImmPred:$u5)), (S2_asl_i_p_or DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;
let AddedComplexity = 100 in
def: Pat<(xor I64:$src1, (shl I64:$Rs, u6_0ImmPred:$u5)), (S2_asl_i_p_xacc DoubleRegs:$src1, DoubleRegs:$Rs, u6_0ImmPred:$u5)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_asl_r_r_acc IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_asl_r_r_nac IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_asl_r_r_and IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_asl_r_r_or IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_asl_r_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_asl_r_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_asl_r_p_and DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_asl_r_p_or DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(xor I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_asl_r_p_xor DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (sra I32:$Rs, I32:$Rt)), (S2_asr_r_r_acc IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I32:$src1, (sra I32:$Rs, I32:$Rt)), (S2_asr_r_r_nac IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I32:$src1, (sra I32:$Rs, I32:$Rt)), (S2_asr_r_r_and IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I32:$src1, (sra I32:$Rs, I32:$Rt)), (S2_asr_r_r_or IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (sra I64:$Rs, I32:$Rt)), (S2_asr_r_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I64:$src1, (sra I64:$Rs, I32:$Rt)), (S2_asr_r_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I64:$src1, (sra I64:$Rs, I32:$Rt)), (S2_asr_r_p_and DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I64:$src1, (sra I64:$Rs, I32:$Rt)), (S2_asr_r_p_or DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(xor I64:$src1, (sra I64:$Rs, I32:$Rt)), (S2_asr_r_p_xor DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (srl I32:$Rs, I32:$Rt)), (S2_lsr_r_r_acc IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I32:$src1, (srl I32:$Rs, I32:$Rt)), (S2_lsr_r_r_nac IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I32:$src1, (srl I32:$Rs, I32:$Rt)), (S2_lsr_r_r_and IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I32:$src1, (srl I32:$Rs, I32:$Rt)), (S2_lsr_r_r_or IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (srl I64:$Rs, I32:$Rt)), (S2_lsr_r_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I64:$src1, (srl I64:$Rs, I32:$Rt)), (S2_lsr_r_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I64:$src1, (srl I64:$Rs, I32:$Rt)), (S2_lsr_r_p_and DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I64:$src1, (srl I64:$Rs, I32:$Rt)), (S2_lsr_r_p_or DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(xor I64:$src1, (srl I64:$Rs, I32:$Rt)), (S2_lsr_r_p_xor DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;

let AddedComplexity = 100 in
def: Pat<(add I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_lsl_r_r_acc IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_lsl_r_r_nac IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_lsl_r_r_and IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I32:$src1, (shl I32:$Rs, I32:$Rt)), (S2_lsl_r_r_or IntRegs:$src1, IntRegs:$Rs, IntRegs:$Rt)>;
let AddedComplexity = 100 in
def: Pat<(add I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_lsl_r_p_acc DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(sub I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_lsl_r_p_nac DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(and I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_lsl_r_p_and DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(or I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_lsl_r_p_or DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(xor I64:$src1, (shl I64:$Rs, I32:$Rt)), (S2_lsl_r_p_xor DoubleRegs:$src1, DoubleRegs:$Rs, IntRegs:$Rt)>;

def: Pat<(sra I64:$src1, I32:$src2), (S2_asr_r_p DoubleRegs:$src1, IntRegs:$src2)>;
def: Pat<(srl I64:$src1, I32:$src2), (S2_lsr_r_p DoubleRegs:$src1, IntRegs:$src2)>;
def: Pat<(shl I64:$src1, I32:$src2), (S2_asl_r_p DoubleRegs:$src1, IntRegs:$src2)>;
def: Pat<(shl I64:$src1, I32:$src2), (S2_lsl_r_p DoubleRegs:$src1, IntRegs:$src2)>;

def: Pat<(sra I32:$src1, I32:$src2), (S2_asr_r_r IntRegs:$src1, IntRegs:$src2)>;
def: Pat<(srl I32:$src1, I32:$src2), (S2_lsr_r_r IntRegs:$src1, IntRegs:$src2)>;
def: Pat<(shl I32:$src1, I32:$src2), (S2_asl_r_r IntRegs:$src1, IntRegs:$src2)>;
def: Pat<(shl I32:$src1, I32:$src2), (S2_lsl_r_r IntRegs:$src1, IntRegs:$src2)>;

def SDTHexagonINSERT:
  SDTypeProfile<1, 4, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                       SDTCisInt<0>, SDTCisVT<3, i32>, SDTCisVT<4, i32>]>;
def SDTHexagonINSERTRP:
  SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                       SDTCisInt<0>, SDTCisVT<3, i64>]>;

def HexagonINSERT   : SDNode<"HexagonISD::INSERT",   SDTHexagonINSERT>;
def HexagonINSERTRP : SDNode<"HexagonISD::INSERTRP", SDTHexagonINSERTRP>;

def: Pat<(HexagonINSERT I32:$Rs, I32:$Rt, u5_0ImmPred:$u1, u5_0ImmPred:$u2),
         (S2_insert I32:$Rs, I32:$Rt, u5_0ImmPred:$u1, u5_0ImmPred:$u2)>;
def: Pat<(HexagonINSERT I64:$Rs, I64:$Rt, u6_0ImmPred:$u1, u6_0ImmPred:$u2),
         (S2_insertp I64:$Rs, I64:$Rt, u6_0ImmPred:$u1, u6_0ImmPred:$u2)>;
def: Pat<(HexagonINSERTRP I32:$Rs, I32:$Rt, I64:$Ru),
         (S2_insert_rp I32:$Rs, I32:$Rt, I64:$Ru)>;
def: Pat<(HexagonINSERTRP I64:$Rs, I64:$Rt, I64:$Ru),
         (S2_insertp_rp I64:$Rs, I64:$Rt, I64:$Ru)>;

let AddedComplexity = 100 in
def: Pat<(or (or (shl (HexagonINSERT (i32 (zextloadi8 (add I32:$b, 2))),
                                     (i32 (extloadi8  (add I32:$b, 3))),
                                     24, 8),
                      (i32 16)),
                 (shl (i32 (zextloadi8 (add I32:$b, 1))), (i32 8))),
             (zextloadi8 I32:$b)),
         (A2_swiz (L2_loadri_io I32:$b, 0))>;

def SDTHexagonEXTRACTU:
  SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<1>,
                       SDTCisVT<2, i32>, SDTCisVT<3, i32>]>;
def SDTHexagonEXTRACTURP:
  SDTypeProfile<1, 2, [SDTCisSameAs<0, 1>, SDTCisInt<0>, SDTCisInt<1>,
                       SDTCisVT<2, i64>]>;

def HexagonEXTRACTU   : SDNode<"HexagonISD::EXTRACTU",   SDTHexagonEXTRACTU>;
def HexagonEXTRACTURP : SDNode<"HexagonISD::EXTRACTURP", SDTHexagonEXTRACTURP>;

def: Pat<(HexagonEXTRACTU I32:$src1, u5_0ImmPred:$src2, u5_0ImmPred:$src3),
         (S2_extractu I32:$src1, u5_0ImmPred:$src2, u5_0ImmPred:$src3)>;
def: Pat<(HexagonEXTRACTU I64:$src1, u6_0ImmPred:$src2, u6_0ImmPred:$src3),
         (S2_extractup I64:$src1, u6_0ImmPred:$src2, u6_0ImmPred:$src3)>;
def: Pat<(HexagonEXTRACTURP I32:$src1, I64:$src2),
         (S2_extractu_rp I32:$src1, I64:$src2)>;
def: Pat<(HexagonEXTRACTURP I64:$src1, I64:$src2),
         (S2_extractup_rp I64:$src1, I64:$src2)>;

def n8_0ImmPred: PatLeaf<(i32 imm), [{
  int64_t V = N->getSExtValue();
  return -255 <= V && V <= 0;
}]>;

// Change the sign of the immediate for Rd=-mpyi(Rs,#u8)
def: Pat<(mul I32:$src1, (ineg n8_0ImmPred:$src2)),
         (M2_mpysin IntRegs:$src1, u8_0ImmPred:$src2)>;

multiclass MinMax_pats_p<PatFrag Op, InstHexagon Inst, InstHexagon SwapInst> {
  defm: T_MinMax_pats<Op, I64, Inst, SwapInst>;
}

def: Pat<(add Sext64:$Rs, I64:$Rt),
         (A2_addsp (LoReg Sext64:$Rs), DoubleRegs:$Rt)>;

let AddedComplexity = 200 in {
  defm: MinMax_pats_p<setge,  A2_maxp,  A2_minp>;
  defm: MinMax_pats_p<setgt,  A2_maxp,  A2_minp>;
  defm: MinMax_pats_p<setle,  A2_minp,  A2_maxp>;
  defm: MinMax_pats_p<setlt,  A2_minp,  A2_maxp>;
  defm: MinMax_pats_p<setuge, A2_maxup, A2_minup>;
  defm: MinMax_pats_p<setugt, A2_maxup, A2_minup>;
  defm: MinMax_pats_p<setule, A2_minup, A2_maxup>;
  defm: MinMax_pats_p<setult, A2_minup, A2_maxup>;
}

def callv3 : SDNode<"HexagonISD::CALL", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;

def callv3nr : SDNode<"HexagonISD::CALLnr", SDT_SPCall,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue, SDNPVariadic]>;


// Map call instruction
def : Pat<(callv3 I32:$dst),
          (J2_callr I32:$dst)>;
def : Pat<(callv3 tglobaladdr:$dst),
          (J2_call tglobaladdr:$dst)>;
def : Pat<(callv3 texternalsym:$dst),
          (J2_call texternalsym:$dst)>;
def : Pat<(callv3 tglobaltlsaddr:$dst),
          (J2_call tglobaltlsaddr:$dst)>;

def : Pat<(callv3nr I32:$dst),
          (PS_callr_nr I32:$dst)>;
def : Pat<(callv3nr tglobaladdr:$dst),
          (PS_call_nr tglobaladdr:$dst)>;
def : Pat<(callv3nr texternalsym:$dst),
          (PS_call_nr texternalsym:$dst)>;


def addrga: PatLeaf<(i32 AddrGA:$Addr)>;
def addrgp: PatLeaf<(i32 AddrGP:$Addr)>;


// Pats for instruction selection.

// A class to embed the usual comparison patfrags within a zext to i32.
// The seteq/setne frags use "lhs" and "rhs" as operands, so use the same
// names, or else the frag's "body" won't match the operands.
class CmpInReg<PatFrag Op>
  : PatFrag<(ops node:$lhs, node:$rhs),(i32 (zext (i1 Op.Fragment)))>;

def: T_cmp32_rr_pat<A4_rcmpeq,  CmpInReg<seteq>, i32>;
def: T_cmp32_rr_pat<A4_rcmpneq, CmpInReg<setne>, i32>;

def: T_cmp32_rr_pat<C4_cmpneq,  setne,  i1>;
def: T_cmp32_rr_pat<C4_cmplte,  setle,  i1>;
def: T_cmp32_rr_pat<C4_cmplteu, setule, i1>;

def: T_cmp32_rr_pat<C4_cmplte,  RevCmp<setge>,  i1>;
def: T_cmp32_rr_pat<C4_cmplteu, RevCmp<setuge>, i1>;

let AddedComplexity = 100 in {
  def: Pat<(i1 (seteq (and (xor I32:$Rs, I32:$Rt),
                       255), 0)),
           (A4_cmpbeq IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(i1 (setne (and (xor I32:$Rs, I32:$Rt),
                       255), 0)),
           (C2_not (A4_cmpbeq IntRegs:$Rs, IntRegs:$Rt))>;
  def: Pat<(i1 (seteq (and (xor I32:$Rs, I32:$Rt),
                           65535), 0)),
           (A4_cmpheq IntRegs:$Rs, IntRegs:$Rt)>;
  def: Pat<(i1 (setne (and (xor I32:$Rs, I32:$Rt),
                           65535), 0)),
           (C2_not (A4_cmpheq IntRegs:$Rs, IntRegs:$Rt))>;
}

def: Pat<(i32 (zext (i1 (seteq I32:$Rs, s32_0ImmPred:$s8)))),
         (A4_rcmpeqi IntRegs:$Rs, s32_0ImmPred:$s8)>;
def: Pat<(i32 (zext (i1 (setne I32:$Rs, s32_0ImmPred:$s8)))),
         (A4_rcmpneqi IntRegs:$Rs, s32_0ImmPred:$s8)>;

// Preserve the S2_tstbit_r generation
def: Pat<(i32 (zext (i1 (setne (i32 (and (i32 (shl 1, I32:$src2)),
                                         I32:$src1)), 0)))),
         (C2_muxii (S2_tstbit_r IntRegs:$src1, IntRegs:$src2), 1, 0)>;

// The complexity of the combines involving immediates should be greater
// than the complexity of the combine with two registers.
let AddedComplexity = 50 in {
def: Pat<(HexagonCOMBINE IntRegs:$r, s32_0ImmPred:$i),
         (A4_combineri IntRegs:$r, s32_0ImmPred:$i)>;

def: Pat<(HexagonCOMBINE s32_0ImmPred:$i, IntRegs:$r),
         (A4_combineir s32_0ImmPred:$i, IntRegs:$r)>;
}

// The complexity of the combine with two immediates should be greater than
// the complexity of a combine involving a register.
let AddedComplexity = 75 in {
def: Pat<(HexagonCOMBINE s8_0ImmPred:$s8, u32_0ImmPred:$u6),
         (A4_combineii imm:$s8, imm:$u6)>;
def: Pat<(HexagonCOMBINE s32_0ImmPred:$s8, s8_0ImmPred:$S8),
         (A2_combineii imm:$s8, imm:$S8)>;
}


// Patterns to generate indexed loads with different forms of the address:
// - frameindex,
// - base + offset,
// - base (without offset).
multiclass Loadxm_pat<PatFrag Load, ValueType VT, PatFrag ValueMod,
                      PatLeaf ImmPred, InstHexagon MI> {
  def: Pat<(VT (Load AddrFI:$fi)),
           (VT (ValueMod (MI AddrFI:$fi, 0)))>;
  def: Pat<(VT (Load (add AddrFI:$fi, ImmPred:$Off))),
           (VT (ValueMod (MI AddrFI:$fi, imm:$Off)))>;
  def: Pat<(VT (Load (add IntRegs:$Rs, ImmPred:$Off))),
           (VT (ValueMod (MI IntRegs:$Rs, imm:$Off)))>;
  def: Pat<(VT (Load I32:$Rs)),
           (VT (ValueMod (MI IntRegs:$Rs, 0)))>;
}

defm: Loadxm_pat<extloadi1,   i64, ToZext64, s32_0ImmPred, L2_loadrub_io>;
defm: Loadxm_pat<extloadi8,   i64, ToZext64, s32_0ImmPred, L2_loadrub_io>;
defm: Loadxm_pat<extloadi16,  i64, ToZext64, s31_1ImmPred, L2_loadruh_io>;
defm: Loadxm_pat<zextloadi1,  i64, ToZext64, s32_0ImmPred, L2_loadrub_io>;
defm: Loadxm_pat<zextloadi8,  i64, ToZext64, s32_0ImmPred, L2_loadrub_io>;
defm: Loadxm_pat<zextloadi16, i64, ToZext64, s31_1ImmPred, L2_loadruh_io>;
defm: Loadxm_pat<sextloadi8,  i64, ToSext64, s32_0ImmPred, L2_loadrb_io>;
defm: Loadxm_pat<sextloadi16, i64, ToSext64, s31_1ImmPred, L2_loadrh_io>;

// Map Rdd = anyext(Rs) -> Rdd = combine(#0, Rs).
def: Pat<(Aext64 I32:$src1), (ToZext64 IntRegs:$src1)>;

multiclass T_LoadAbsReg_Pat <PatFrag ldOp, InstHexagon MI, ValueType VT = i32> {
  def  : Pat <(VT (ldOp (add (shl IntRegs:$src1, u2_0ImmPred:$src2),
                             (HexagonCONST32 tglobaladdr:$src3)))),
              (MI IntRegs:$src1, u2_0ImmPred:$src2, tglobaladdr:$src3)>;
  def  : Pat <(VT (ldOp (add IntRegs:$src1,
                             (HexagonCONST32 tglobaladdr:$src2)))),
              (MI IntRegs:$src1, 0, tglobaladdr:$src2)>;

  def  : Pat <(VT (ldOp (add (shl IntRegs:$src1, u2_0ImmPred:$src2),
                             (HexagonCONST32 tconstpool:$src3)))),
              (MI IntRegs:$src1, u2_0ImmPred:$src2, tconstpool:$src3)>;
  def  : Pat <(VT (ldOp (add IntRegs:$src1,
                             (HexagonCONST32 tconstpool:$src2)))),
              (MI IntRegs:$src1, 0, tconstpool:$src2)>;

  def  : Pat <(VT (ldOp (add (shl IntRegs:$src1, u2_0ImmPred:$src2),
                             (HexagonCONST32 tjumptable:$src3)))),
              (MI IntRegs:$src1, u2_0ImmPred:$src2, tjumptable:$src3)>;
  def  : Pat <(VT (ldOp (add IntRegs:$src1,
                             (HexagonCONST32 tjumptable:$src2)))),
              (MI IntRegs:$src1, 0, tjumptable:$src2)>;
}

let AddedComplexity  = 60 in {
defm : T_LoadAbsReg_Pat <sextloadi8, L4_loadrb_ur>;
defm : T_LoadAbsReg_Pat <zextloadi8, L4_loadrub_ur>;
defm : T_LoadAbsReg_Pat <extloadi8,  L4_loadrub_ur>;

defm : T_LoadAbsReg_Pat <sextloadi16, L4_loadrh_ur>;
defm : T_LoadAbsReg_Pat <zextloadi16, L4_loadruh_ur>;
defm : T_LoadAbsReg_Pat <extloadi16,  L4_loadruh_ur>;

defm : T_LoadAbsReg_Pat <load, L4_loadri_ur>;
defm : T_LoadAbsReg_Pat <load, L4_loadrd_ur, i64>;
}

// 'def pats' for load instructions with base + register offset and non-zero
// immediate value. Immediate value is used to left-shift the second
// register operand.
class Loadxs_pat<PatFrag Load, ValueType VT, InstHexagon MI>
  : Pat<(VT (Load (add I32:$Rs,
                       (i32 (shl I32:$Rt, u2_0ImmPred:$u2))))),
        (VT (MI IntRegs:$Rs, IntRegs:$Rt, imm:$u2))>;

let AddedComplexity = 40 in {
  def: Loadxs_pat<extloadi8,   i32, L4_loadrub_rr>;
  def: Loadxs_pat<zextloadi8,  i32, L4_loadrub_rr>;
  def: Loadxs_pat<sextloadi8,  i32, L4_loadrb_rr>;
  def: Loadxs_pat<extloadi16,  i32, L4_loadruh_rr>;
  def: Loadxs_pat<zextloadi16, i32, L4_loadruh_rr>;
  def: Loadxs_pat<sextloadi16, i32, L4_loadrh_rr>;
  def: Loadxs_pat<load,        i32, L4_loadri_rr>;
  def: Loadxs_pat<load,        i64, L4_loadrd_rr>;
}

// 'def pats' for load instruction base + register offset and
// zero immediate value.
class Loadxs_simple_pat<PatFrag Load, ValueType VT, InstHexagon MI>
  : Pat<(VT (Load (add I32:$Rs, I32:$Rt))),
        (VT (MI IntRegs:$Rs, IntRegs:$Rt, 0))>;

let AddedComplexity = 20 in {
  def: Loadxs_simple_pat<extloadi8,   i32, L4_loadrub_rr>;
  def: Loadxs_simple_pat<zextloadi8,  i32, L4_loadrub_rr>;
  def: Loadxs_simple_pat<sextloadi8,  i32, L4_loadrb_rr>;
  def: Loadxs_simple_pat<extloadi16,  i32, L4_loadruh_rr>;
  def: Loadxs_simple_pat<zextloadi16, i32, L4_loadruh_rr>;
  def: Loadxs_simple_pat<sextloadi16, i32, L4_loadrh_rr>;
  def: Loadxs_simple_pat<load,        i32, L4_loadri_rr>;
  def: Loadxs_simple_pat<load,        i64, L4_loadrd_rr>;
}

let AddedComplexity = 40 in
multiclass T_StoreAbsReg_Pats <InstHexagon MI, RegisterClass RC, ValueType VT,
                           PatFrag stOp> {
 def : Pat<(stOp (VT RC:$src4),
                 (add (shl I32:$src1, u2_0ImmPred:$src2),
                      u32_0ImmPred:$src3)),
          (MI IntRegs:$src1, u2_0ImmPred:$src2, u32_0ImmPred:$src3, RC:$src4)>;

 def : Pat<(stOp (VT RC:$src4),
                 (add (shl IntRegs:$src1, u2_0ImmPred:$src2),
                      (HexagonCONST32 tglobaladdr:$src3))),
           (MI IntRegs:$src1, u2_0ImmPred:$src2, tglobaladdr:$src3, RC:$src4)>;

 def : Pat<(stOp (VT RC:$src4),
                 (add IntRegs:$src1, (HexagonCONST32 tglobaladdr:$src3))),
           (MI IntRegs:$src1, 0, tglobaladdr:$src3, RC:$src4)>;
}

defm : T_StoreAbsReg_Pats <S4_storerd_ur, DoubleRegs, i64, store>;
defm : T_StoreAbsReg_Pats <S4_storeri_ur, IntRegs, i32, store>;
defm : T_StoreAbsReg_Pats <S4_storerb_ur, IntRegs, i32, truncstorei8>;
defm : T_StoreAbsReg_Pats <S4_storerh_ur, IntRegs, i32, truncstorei16>;

class Storexs_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Ru, (add I32:$Rs,
                               (i32 (shl I32:$Rt, u2_0ImmPred:$u2)))),
        (MI IntRegs:$Rs, IntRegs:$Rt, imm:$u2, Value:$Ru)>;

let AddedComplexity = 40 in {
  def: Storexs_pat<truncstorei8,  I32, S4_storerb_rr>;
  def: Storexs_pat<truncstorei16, I32, S4_storerh_rr>;
  def: Storexs_pat<store,         I32, S4_storeri_rr>;
  def: Storexs_pat<store,         I64, S4_storerd_rr>;
}

def s30_2ProperPred  : PatLeaf<(i32 imm), [{
  int64_t v = (int64_t)N->getSExtValue();
  return isShiftedInt<30,2>(v) && !isShiftedInt<29,3>(v);
}]>;
def RoundTo8 : SDNodeXForm<imm, [{
  int32_t Imm = N->getSExtValue();
  return CurDAG->getTargetConstant(Imm & -8, SDLoc(N), MVT::i32);
}]>;

let AddedComplexity = 40 in
def: Pat<(store I64:$Ru, (add I32:$Rs, s30_2ProperPred:$Off)),
         (S2_storerd_io (A2_addi I32:$Rs, 4), (RoundTo8 $Off), I64:$Ru)>;

class Store_rr_pat<PatFrag Store, PatFrag Value, InstHexagon MI>
  : Pat<(Store Value:$Ru, (add I32:$Rs, I32:$Rt)),
        (MI IntRegs:$Rs, IntRegs:$Rt, 0, Value:$Ru)>;

let AddedComplexity = 20 in {
  def: Store_rr_pat<truncstorei8,  I32, S4_storerb_rr>;
  def: Store_rr_pat<truncstorei16, I32, S4_storerh_rr>;
  def: Store_rr_pat<store,         I32, S4_storeri_rr>;
  def: Store_rr_pat<store,         I64, S4_storerd_rr>;
}


def IMM_BYTE : SDNodeXForm<imm, [{
  // -1 etc is  represented as 255 etc
  // assigning to a byte restores our desired signed value.
  int8_t imm = N->getSExtValue();
  return CurDAG->getTargetConstant(imm, SDLoc(N), MVT::i32);
}]>;

def IMM_HALF : SDNodeXForm<imm, [{
  // -1 etc is  represented as 65535 etc
  // assigning to a short restores our desired signed value.
  int16_t imm = N->getSExtValue();
  return CurDAG->getTargetConstant(imm, SDLoc(N), MVT::i32);
}]>;

def IMM_WORD : SDNodeXForm<imm, [{
  // -1 etc can be represented as 4294967295 etc
  // Currently, it's not doing this. But some optimization
  // might convert -1 to a large +ve number.
  // assigning to a word restores our desired signed value.
  int32_t imm = N->getSExtValue();
  return CurDAG->getTargetConstant(imm, SDLoc(N), MVT::i32);
}]>;

def ToImmByte : OutPatFrag<(ops node:$R), (IMM_BYTE $R)>;
def ToImmHalf : OutPatFrag<(ops node:$R), (IMM_HALF $R)>;
def ToImmWord : OutPatFrag<(ops node:$R), (IMM_WORD $R)>;

// Emit store-immediate, but only when the stored value will not be constant-
// extended. The reason for that is that there is no pass that can optimize
// constant extenders in store-immediate instructions. In some cases we can
// end up will a number of such stores, all of which store the same extended
// value (e.g. after unrolling a loop that initializes floating point array).

// Predicates to determine if the 16-bit immediate is expressible as a sign-
// extended 8-bit immediate. Store-immediate-halfword will ignore any bits
// beyond 0..15, so we don't care what is in there.

def i16in8ImmPred: PatLeaf<(i32 imm), [{
  int64_t v = (int16_t)N->getSExtValue();
  return v == (int64_t)(int8_t)v;
}]>;

// Predicates to determine if the 32-bit immediate is expressible as a sign-
// extended 8-bit immediate.
def i32in8ImmPred: PatLeaf<(i32 imm), [{
  int64_t v = (int32_t)N->getSExtValue();
  return v == (int64_t)(int8_t)v;
}]>;

class SmallStackStore<PatFrag Store>
  : PatFrag<(ops node:$Val, node:$Addr), (Store node:$Val, node:$Addr), [{
  return isSmallStackStore(cast<StoreSDNode>(N));
}]>;

let AddedComplexity = 40 in {
  // Even though the offset is not extendable in the store-immediate, we
  // can still generate the fi# in the base address. If the final offset
  // is not valid for the instruction, we will replace it with a scratch
  // register.
  def: Storexm_fi_pat <SmallStackStore<truncstorei8>, s32_0ImmPred,
                       ToImmByte, S4_storeirb_io>;
  def: Storexm_fi_pat <SmallStackStore<truncstorei16>, i16in8ImmPred,
                       ToImmHalf, S4_storeirh_io>;
  def: Storexm_fi_pat <SmallStackStore<store>, i32in8ImmPred,
                       ToImmWord, S4_storeiri_io>;

//  defm: Storexm_fi_add_pat <truncstorei8, s32_0ImmPred, u6_0ImmPred, ToImmByte,
//                            S4_storeirb_io>;
//  defm: Storexm_fi_add_pat <truncstorei16, i16in8ImmPred, u6_1ImmPred,
//                            ToImmHalf, S4_storeirh_io>;
//  defm: Storexm_fi_add_pat <store, i32in8ImmPred, u6_2ImmPred, ToImmWord,
//                            S4_storeiri_io>;

  defm: Storexm_add_pat<truncstorei8, s32_0ImmPred, u6_0ImmPred, ToImmByte,
                        S4_storeirb_io>;
  defm: Storexm_add_pat<truncstorei16, i16in8ImmPred, u6_1ImmPred, ToImmHalf,
                        S4_storeirh_io>;
  defm: Storexm_add_pat<store, i32in8ImmPred, u6_2ImmPred, ToImmWord,
                        S4_storeiri_io>;
}

def: Storexm_simple_pat<truncstorei8,  s32_0ImmPred, ToImmByte, S4_storeirb_io>;
def: Storexm_simple_pat<truncstorei16, s32_0ImmPred, ToImmHalf, S4_storeirh_io>;
def: Storexm_simple_pat<store,         s32_0ImmPred, ToImmWord, S4_storeiri_io>;

// op(Ps, op(Pt, Pu))
class LogLog_pat<SDNode Op1, SDNode Op2, InstHexagon MI>
  : Pat<(i1 (Op1 I1:$Ps, (Op2 I1:$Pt, I1:$Pu))),
        (MI I1:$Ps, I1:$Pt, I1:$Pu)>;

// op(Ps, op(Pt, ~Pu))
class LogLogNot_pat<SDNode Op1, SDNode Op2, InstHexagon MI>
  : Pat<(i1 (Op1 I1:$Ps, (Op2 I1:$Pt, (not I1:$Pu)))),
        (MI I1:$Ps, I1:$Pt, I1:$Pu)>;

def: LogLog_pat<and, and, C4_and_and>;
def: LogLog_pat<and, or,  C4_and_or>;
def: LogLog_pat<or,  and, C4_or_and>;
def: LogLog_pat<or,  or,  C4_or_or>;

def: LogLogNot_pat<and, and, C4_and_andn>;
def: LogLogNot_pat<and, or,  C4_and_orn>;
def: LogLogNot_pat<or,  and, C4_or_andn>;
def: LogLogNot_pat<or,  or,  C4_or_orn>;

//===----------------------------------------------------------------------===//
// PIC: Support for PIC compilations. The patterns and SD nodes defined
// below are needed to support code generation for PIC
//===----------------------------------------------------------------------===//

def SDT_HexagonAtGot
  : SDTypeProfile<1, 3, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i32>]>;
def SDT_HexagonAtPcrel
  : SDTypeProfile<1, 1, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>;

// AT_GOT address-of-GOT, address-of-global, offset-in-global
def HexagonAtGot       : SDNode<"HexagonISD::AT_GOT", SDT_HexagonAtGot>;
// AT_PCREL address-of-global
def HexagonAtPcrel     : SDNode<"HexagonISD::AT_PCREL", SDT_HexagonAtPcrel>;

def: Pat<(HexagonAtGot I32:$got, I32:$addr, (i32 0)),
         (L2_loadri_io I32:$got, imm:$addr)>;
def: Pat<(HexagonAtGot I32:$got, I32:$addr, s30_2ImmPred:$off),
         (A2_addi (L2_loadri_io I32:$got, imm:$addr), imm:$off)>;
def: Pat<(HexagonAtPcrel I32:$addr),
         (C4_addipc imm:$addr)>;

def: Pat<(i64 (and I64:$Rs, (i64 (not I64:$Rt)))),
         (A4_andnp DoubleRegs:$Rs, DoubleRegs:$Rt)>;
def: Pat<(i64 (or  I64:$Rs, (i64 (not I64:$Rt)))),
         (A4_ornp DoubleRegs:$Rs, DoubleRegs:$Rt)>;

def: Pat<(add I32:$Rs, (add I32:$Ru, s32_0ImmPred:$s6)),
         (S4_addaddi IntRegs:$Rs, IntRegs:$Ru, imm:$s6)>;

// Rd=add(Rs,sub(#s6,Ru))
def: Pat<(add I32:$src1, (sub s32_0ImmPred:$src2,
                                        I32:$src3)),
         (S4_subaddi IntRegs:$src1, s32_0ImmPred:$src2, IntRegs:$src3)>;

// Rd=sub(add(Rs,#s6),Ru)
def: Pat<(sub (add I32:$src1, s32_0ImmPred:$src2),
                   I32:$src3),
         (S4_subaddi IntRegs:$src1, s32_0ImmPred:$src2, IntRegs:$src3)>;

// Rd=add(sub(Rs,Ru),#s6)
def: Pat<(add (sub I32:$src1, I32:$src3),
                   (s32_0ImmPred:$src2)),
         (S4_subaddi IntRegs:$src1, s32_0ImmPred:$src2, IntRegs:$src3)>;

def: Pat<(xor I64:$dst2,
              (xor I64:$Rss, I64:$Rtt)),
         (M4_xor_xacc DoubleRegs:$dst2, DoubleRegs:$Rss, DoubleRegs:$Rtt)>;
def: Pat<(or I32:$Ru, (and (i32 IntRegs:$_src_), s32_0ImmPred:$s10)),
         (S4_or_andix IntRegs:$Ru, IntRegs:$_src_, imm:$s10)>;

def: Pat<(or I32:$src1, (and I32:$Rs, s32_0ImmPred:$s10)),
         (S4_or_andi IntRegs:$src1, IntRegs:$Rs, imm:$s10)>;

def: Pat<(or I32:$src1, (or I32:$Rs, s32_0ImmPred:$s10)),
         (S4_or_ori IntRegs:$src1, IntRegs:$Rs, imm:$s10)>;



// Count trailing zeros: 64-bit.
def: Pat<(i32 (trunc (cttz I64:$Rss))), (S2_ct0p I64:$Rss)>;

// Count trailing ones: 64-bit.
def: Pat<(i32 (trunc (cttz (not I64:$Rss)))), (S2_ct1p I64:$Rss)>;

// Define leading/trailing patterns that require zero-extensions to 64 bits.
def: Pat<(i64 (ctlz I64:$Rss)), (ToZext64 (S2_cl0p I64:$Rss))>;
def: Pat<(i64 (cttz I64:$Rss)), (ToZext64 (S2_ct0p I64:$Rss))>;
def: Pat<(i64 (ctlz (not I64:$Rss))), (ToZext64 (S2_cl1p I64:$Rss))>;
def: Pat<(i64 (cttz (not I64:$Rss))), (ToZext64 (S2_ct1p I64:$Rss))>;

def: Pat<(i64 (ctpop I64:$Rss)), (ToZext64 (S5_popcountp I64:$Rss))>;
def: Pat<(i32 (ctpop I32:$Rs)), (S5_popcountp (A4_combineir 0, I32:$Rs))>;

def: Pat<(bitreverse I32:$Rs), (S2_brev I32:$Rs)>;
def: Pat<(bitreverse I64:$Rss), (S2_brevp I64:$Rss)>;

def: Pat<(bswap I32:$Rs), (A2_swiz I32:$Rs)>;
def: Pat<(bswap I64:$Rss), (A2_combinew (A2_swiz (LoReg $Rss)),
                                        (A2_swiz (HiReg $Rss)))>;

let AddedComplexity = 20 in {   // Complexity greater than cmp reg-imm.
  def: Pat<(i1 (seteq (and (shl 1, u5_0ImmPred:$u5), I32:$Rs), 0)),
           (S4_ntstbit_i I32:$Rs, u5_0ImmPred:$u5)>;
  def: Pat<(i1 (seteq (and (shl 1, I32:$Rt), I32:$Rs), 0)),
           (S4_ntstbit_r I32:$Rs, I32:$Rt)>;
}

// Add extra complexity to prefer these instructions over bitsset/bitsclr.
// The reason is that tstbit/ntstbit can be folded into a compound instruction:
//   if ([!]tstbit(...)) jump ...
let AddedComplexity = 100 in
def: Pat<(i1 (setne (and I32:$Rs, (i32 IsPow2_32:$u5)), (i32 0))),
         (S2_tstbit_i I32:$Rs, (Log2_32 imm:$u5))>;

let AddedComplexity = 100 in
def: Pat<(i1 (seteq (and I32:$Rs, (i32 IsPow2_32:$u5)), (i32 0))),
         (S4_ntstbit_i I32:$Rs, (Log2_32 imm:$u5))>;

// Do not increase complexity of these patterns. In the DAG, "cmp i8" may be
// represented as a compare against "value & 0xFF", which is an exact match
// for cmpb (same for cmph). The patterns below do not contain any additional
// complexity that would make them preferable, and if they were actually used
// instead of cmpb/cmph, they would result in a compare against register that
// is loaded with the byte/half mask (i.e. 0xFF or 0xFFFF).
def: Pat<(i1 (setne (and I32:$Rs, u6_0ImmPred:$u6), 0)),
         (C4_nbitsclri I32:$Rs, u6_0ImmPred:$u6)>;
def: Pat<(i1 (setne (and I32:$Rs, I32:$Rt), 0)),
         (C4_nbitsclr I32:$Rs, I32:$Rt)>;
def: Pat<(i1 (setne (and I32:$Rs, I32:$Rt), I32:$Rt)),
         (C4_nbitsset I32:$Rs, I32:$Rt)>;


def: Pat<(add (mul I32:$Rs, u6_0ImmPred:$U6), u32_0ImmPred:$u6),
         (M4_mpyri_addi imm:$u6, IntRegs:$Rs, imm:$U6)>;
def: Pat<(add (mul I32:$Rs, u6_0ImmPred:$U6),
              (HexagonCONST32 tglobaladdr:$global)),
         (M4_mpyri_addi tglobaladdr:$global, IntRegs:$Rs, imm:$U6)>;
def: Pat<(add (mul I32:$Rs, I32:$Rt), u32_0ImmPred:$u6),
         (M4_mpyrr_addi imm:$u6, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(add (mul I32:$Rs, I32:$Rt),
              (HexagonCONST32 tglobaladdr:$global)),
         (M4_mpyrr_addi tglobaladdr:$global, IntRegs:$Rs, IntRegs:$Rt)>;
def: Pat<(add I32:$src1, (mul I32:$src3, u6_2ImmPred:$src2)),
         (M4_mpyri_addr_u2 IntRegs:$src1, imm:$src2, IntRegs:$src3)>;
def: Pat<(add I32:$src1, (mul I32:$src3, u32_0ImmPred:$src2)),
         (M4_mpyri_addr IntRegs:$src1, IntRegs:$src3, imm:$src2)>;

def: Pat<(add I32:$Ru, (mul (i32 IntRegs:$_src_), I32:$Rs)),
         (M4_mpyrr_addr IntRegs:$Ru, IntRegs:$_src_, IntRegs:$Rs)>;

def: T_vcmp_pat<A4_vcmpbgt, setgt, v8i8>;

class T_Shift_CommOp_pat<InstHexagon MI, SDNode Op, SDNode ShOp>
  : Pat<(Op (ShOp IntRegs:$Rx, u5_0ImmPred:$U5), u32_0ImmPred:$u8),
        (MI u32_0ImmPred:$u8, IntRegs:$Rx, u5_0ImmPred:$U5)>;

let AddedComplexity = 200 in {
  def : T_Shift_CommOp_pat <S4_addi_asl_ri, add, shl>;
  def : T_Shift_CommOp_pat <S4_addi_lsr_ri, add, srl>;
  def : T_Shift_CommOp_pat <S4_andi_asl_ri, and, shl>;
  def : T_Shift_CommOp_pat <S4_andi_lsr_ri, and, srl>;
}

let AddedComplexity = 30 in {
  def : T_Shift_CommOp_pat <S4_ori_asl_ri,  or,  shl>;
  def : T_Shift_CommOp_pat <S4_ori_lsr_ri,  or,  srl>;
}

class T_Shift_Op_pat<InstHexagon MI, SDNode Op, SDNode ShOp>
  : Pat<(Op u32_0ImmPred:$u8, (ShOp IntRegs:$Rx, u5_0ImmPred:$U5)),
        (MI u32_0ImmPred:$u8, IntRegs:$Rx, u5_0ImmPred:$U5)>;

def : T_Shift_Op_pat <S4_subi_asl_ri, sub, shl>;
def : T_Shift_Op_pat <S4_subi_lsr_ri, sub, srl>;

let AddedComplexity = 200 in {
  def: Pat<(add addrga:$addr, (shl I32:$src2, u5_0ImmPred:$src3)),
           (S4_addi_asl_ri addrga:$addr, IntRegs:$src2, u5_0ImmPred:$src3)>;
  def: Pat<(add addrga:$addr, (srl I32:$src2, u5_0ImmPred:$src3)),
           (S4_addi_lsr_ri addrga:$addr, IntRegs:$src2, u5_0ImmPred:$src3)>;
  def: Pat<(sub addrga:$addr, (shl I32:$src2, u5_0ImmPred:$src3)),
           (S4_subi_asl_ri addrga:$addr, IntRegs:$src2, u5_0ImmPred:$src3)>;
  def: Pat<(sub addrga:$addr, (srl I32:$src2, u5_0ImmPred:$src3)),
           (S4_subi_lsr_ri addrga:$addr, IntRegs:$src2, u5_0ImmPred:$src3)>;
}

def: Pat<(shl s6_0ImmPred:$s6, I32:$Rt),
         (S4_lsli imm:$s6, IntRegs:$Rt)>;


//===----------------------------------------------------------------------===//
// MEMOP
//===----------------------------------------------------------------------===//

def m5_0Imm8Pred : PatLeaf<(i32 imm), [{
  int8_t V = N->getSExtValue();
  return -32 < V && V <= -1;
}]>;

def m5_0Imm16Pred : PatLeaf<(i32 imm), [{
  int16_t V = N->getSExtValue();
  return -32 < V && V <= -1;
}]>;

def m5_0ImmPred  : PatLeaf<(i32 imm), [{
  int64_t V = N->getSExtValue();
  return -31 <= V && V <= -1;
}]>;

def IsNPow2_8 : PatLeaf<(i32 imm), [{
  uint8_t NV = ~N->getZExtValue();
  return isPowerOf2_32(NV);
}]>;

def IsNPow2_16 : PatLeaf<(i32 imm), [{
  uint16_t NV = ~N->getZExtValue();
  return isPowerOf2_32(NV);
}]>;

def Log2_8 : SDNodeXForm<imm, [{
  uint8_t V = N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(V), SDLoc(N), MVT::i32);
}]>;

def Log2_16 : SDNodeXForm<imm, [{
  uint16_t V = N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(V), SDLoc(N), MVT::i32);
}]>;

def LogN2_8 : SDNodeXForm<imm, [{
  uint8_t NV = ~N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(NV), SDLoc(N), MVT::i32);
}]>;

def LogN2_16 : SDNodeXForm<imm, [{
  uint16_t NV = ~N->getZExtValue();
  return CurDAG->getTargetConstant(Log2_32(NV), SDLoc(N), MVT::i32);
}]>;

def NegImm8 : SDNodeXForm<imm, [{
  int8_t NV = -N->getSExtValue();
  return CurDAG->getTargetConstant(NV, SDLoc(N), MVT::i32);
}]>;

def NegImm16 : SDNodeXForm<imm, [{
  int16_t NV = -N->getSExtValue();
  return CurDAG->getTargetConstant(NV, SDLoc(N), MVT::i32);
}]>;

def NegImm32 : SDNodeXForm<imm, [{
  int32_t NV = -N->getSExtValue();
  return CurDAG->getTargetConstant(NV, SDLoc(N), MVT::i32);
}]>;

def IdImm : SDNodeXForm<imm, [{ return SDValue(N, 0); }]>;

multiclass Memopxr_simple_pat<PatFrag Load, PatFrag Store, SDNode Oper,
                              InstHexagon MI> {
  // Addr: i32
  def: Pat<(Store (Oper (Load I32:$Rs), I32:$A), I32:$Rs),
           (MI I32:$Rs, 0, I32:$A)>;
  // Addr: fi
  def: Pat<(Store (Oper (Load AddrFI:$Rs), I32:$A), AddrFI:$Rs),
           (MI AddrFI:$Rs, 0, I32:$A)>;
}

multiclass Memopxr_add_pat<PatFrag Load, PatFrag Store, PatFrag ImmPred,
                           SDNode Oper, InstHexagon MI> {
  // Addr: i32
  def: Pat<(Store (Oper (Load (add I32:$Rs, ImmPred:$Off)), I32:$A),
                  (add I32:$Rs, ImmPred:$Off)),
           (MI I32:$Rs, imm:$Off, I32:$A)>;
  def: Pat<(Store (Oper (Load (IsOrAdd I32:$Rs, ImmPred:$Off)), I32:$A),
                  (IsOrAdd I32:$Rs, ImmPred:$Off)),
           (MI I32:$Rs, imm:$Off, I32:$A)>;
  // Addr: fi
  def: Pat<(Store (Oper (Load (add AddrFI:$Rs, ImmPred:$Off)), I32:$A),
                  (add AddrFI:$Rs, ImmPred:$Off)),
           (MI AddrFI:$Rs, imm:$Off, I32:$A)>;
  def: Pat<(Store (Oper (Load (IsOrAdd AddrFI:$Rs, ImmPred:$Off)), I32:$A),
                  (IsOrAdd AddrFI:$Rs, ImmPred:$Off)),
           (MI AddrFI:$Rs, imm:$Off, I32:$A)>;
}

multiclass Memopxr_pat<PatFrag Load, PatFrag Store, PatFrag ImmPred,
                       SDNode Oper, InstHexagon MI> {
  defm: Memopxr_simple_pat <Load, Store,          Oper, MI>;
  defm: Memopxr_add_pat    <Load, Store, ImmPred, Oper, MI>;
}

let AddedComplexity = 180 in {
  // add reg
  defm: Memopxr_pat<extloadi8, truncstorei8, u6_0ImmPred, add,
        /*anyext*/  L4_add_memopb_io>;
  defm: Memopxr_pat<sextloadi8, truncstorei8, u6_0ImmPred, add,
        /*sext*/    L4_add_memopb_io>;
  defm: Memopxr_pat<zextloadi8, truncstorei8, u6_0ImmPred, add,
        /*zext*/    L4_add_memopb_io>;
  defm: Memopxr_pat<extloadi16, truncstorei16, u6_1ImmPred, add,
        /*anyext*/  L4_add_memoph_io>;
  defm: Memopxr_pat<sextloadi16, truncstorei16, u6_1ImmPred, add,
        /*sext*/    L4_add_memoph_io>;
  defm: Memopxr_pat<zextloadi16, truncstorei16, u6_1ImmPred, add,
        /*zext*/    L4_add_memoph_io>;
  defm: Memopxr_pat<load, store, u6_2ImmPred, add, L4_add_memopw_io>;

  // sub reg
  defm: Memopxr_pat<extloadi8, truncstorei8, u6_0ImmPred, sub,
        /*anyext*/  L4_sub_memopb_io>;
  defm: Memopxr_pat<sextloadi8, truncstorei8, u6_0ImmPred, sub,
        /*sext*/    L4_sub_memopb_io>;
  defm: Memopxr_pat<zextloadi8, truncstorei8, u6_0ImmPred, sub,
        /*zext*/    L4_sub_memopb_io>;
  defm: Memopxr_pat<extloadi16, truncstorei16, u6_1ImmPred, sub,
        /*anyext*/  L4_sub_memoph_io>;
  defm: Memopxr_pat<sextloadi16, truncstorei16, u6_1ImmPred, sub,
        /*sext*/    L4_sub_memoph_io>;
  defm: Memopxr_pat<zextloadi16, truncstorei16, u6_1ImmPred, sub,
        /*zext*/    L4_sub_memoph_io>;
  defm: Memopxr_pat<load, store, u6_2ImmPred, sub, L4_sub_memopw_io>;

  // and reg
  defm: Memopxr_pat<extloadi8, truncstorei8, u6_0ImmPred, and,
        /*anyext*/  L4_and_memopb_io>;
  defm: Memopxr_pat<sextloadi8, truncstorei8, u6_0ImmPred, and,
        /*sext*/    L4_and_memopb_io>;
  defm: Memopxr_pat<zextloadi8, truncstorei8, u6_0ImmPred, and,
        /*zext*/    L4_and_memopb_io>;
  defm: Memopxr_pat<extloadi16, truncstorei16, u6_1ImmPred, and,
        /*anyext*/  L4_and_memoph_io>;
  defm: Memopxr_pat<sextloadi16, truncstorei16, u6_1ImmPred, and,
        /*sext*/    L4_and_memoph_io>;
  defm: Memopxr_pat<zextloadi16, truncstorei16, u6_1ImmPred, and,
        /*zext*/    L4_and_memoph_io>;
  defm: Memopxr_pat<load, store, u6_2ImmPred, and, L4_and_memopw_io>;

  // or reg
  defm: Memopxr_pat<extloadi8, truncstorei8, u6_0ImmPred, or,
        /*anyext*/  L4_or_memopb_io>;
  defm: Memopxr_pat<sextloadi8, truncstorei8, u6_0ImmPred, or,
        /*sext*/    L4_or_memopb_io>;
  defm: Memopxr_pat<zextloadi8, truncstorei8, u6_0ImmPred, or,
        /*zext*/    L4_or_memopb_io>;
  defm: Memopxr_pat<extloadi16, truncstorei16, u6_1ImmPred, or,
        /*anyext*/  L4_or_memoph_io>;
  defm: Memopxr_pat<sextloadi16, truncstorei16, u6_1ImmPred, or,
        /*sext*/    L4_or_memoph_io>;
  defm: Memopxr_pat<zextloadi16, truncstorei16, u6_1ImmPred, or,
        /*zext*/    L4_or_memoph_io>;
  defm: Memopxr_pat<load, store, u6_2ImmPred, or, L4_or_memopw_io>;
}


multiclass Memopxi_simple_pat<PatFrag Load, PatFrag Store, SDNode Oper,
                              PatFrag Arg, SDNodeXForm ArgMod,
                              InstHexagon MI> {
  // Addr: i32
  def: Pat<(Store (Oper (Load I32:$Rs), Arg:$A), I32:$Rs),
           (MI I32:$Rs, 0, (ArgMod Arg:$A))>;
  // Addr: fi
  def: Pat<(Store (Oper (Load AddrFI:$Rs), Arg:$A), AddrFI:$Rs),
           (MI AddrFI:$Rs, 0, (ArgMod Arg:$A))>;
}

multiclass Memopxi_add_pat<PatFrag Load, PatFrag Store, PatFrag ImmPred,
                           SDNode Oper, PatFrag Arg, SDNodeXForm ArgMod,
                           InstHexagon MI> {
  // Addr: i32
  def: Pat<(Store (Oper (Load (add I32:$Rs, ImmPred:$Off)), Arg:$A),
                  (add I32:$Rs, ImmPred:$Off)),
           (MI I32:$Rs, imm:$Off, (ArgMod Arg:$A))>;
  def: Pat<(Store (Oper (Load (IsOrAdd I32:$Rs, ImmPred:$Off)), Arg:$A),
                  (IsOrAdd I32:$Rs, ImmPred:$Off)),
           (MI I32:$Rs, imm:$Off, (ArgMod Arg:$A))>;
  // Addr: fi
  def: Pat<(Store (Oper (Load (add AddrFI:$Rs, ImmPred:$Off)), Arg:$A),
                  (add AddrFI:$Rs, ImmPred:$Off)),
           (MI AddrFI:$Rs, imm:$Off, (ArgMod Arg:$A))>;
  def: Pat<(Store (Oper (Load (IsOrAdd AddrFI:$Rs, ImmPred:$Off)), Arg:$A),
                  (IsOrAdd AddrFI:$Rs, ImmPred:$Off)),
           (MI AddrFI:$Rs, imm:$Off, (ArgMod Arg:$A))>;
}

multiclass Memopxi_pat<PatFrag Load, PatFrag Store, PatFrag ImmPred,
                       SDNode Oper, PatFrag Arg, SDNodeXForm ArgMod,
                       InstHexagon MI> {
  defm: Memopxi_simple_pat <Load, Store,          Oper, Arg, ArgMod, MI>;
  defm: Memopxi_add_pat    <Load, Store, ImmPred, Oper, Arg, ArgMod, MI>;
}


let AddedComplexity = 200 in {
  // add imm
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, add, u5_0ImmPred,
        /*anyext*/  IdImm, L4_iadd_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, add, u5_0ImmPred,
        /*sext*/    IdImm, L4_iadd_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, add, u5_0ImmPred,
        /*zext*/    IdImm, L4_iadd_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, add, u5_0ImmPred,
        /*anyext*/  IdImm, L4_iadd_memoph_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, add, u5_0ImmPred,
        /*sext*/    IdImm, L4_iadd_memoph_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, add, u5_0ImmPred,
        /*zext*/    IdImm, L4_iadd_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, add, u5_0ImmPred, IdImm,
                    L4_iadd_memopw_io>;
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, sub, m5_0Imm8Pred,
        /*anyext*/  NegImm8, L4_iadd_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, sub, m5_0Imm8Pred,
        /*sext*/    NegImm8, L4_iadd_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, sub, m5_0Imm8Pred,
        /*zext*/    NegImm8, L4_iadd_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, sub, m5_0Imm16Pred,
        /*anyext*/  NegImm16, L4_iadd_memoph_io>;
  defm: Memopxi_pat<sextloadi16, truncstorei16, u6_1ImmPred, sub, m5_0Imm16Pred,
        /*sext*/    NegImm16, L4_iadd_memoph_io>;
  defm: Memopxi_pat<zextloadi16, truncstorei16, u6_1ImmPred, sub, m5_0Imm16Pred,
        /*zext*/    NegImm16, L4_iadd_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, sub, m5_0ImmPred, NegImm32,
                    L4_iadd_memopw_io>;

  // sub imm
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, sub, u5_0ImmPred,
        /*anyext*/  IdImm, L4_isub_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, sub, u5_0ImmPred,
        /*sext*/    IdImm, L4_isub_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, sub, u5_0ImmPred,
        /*zext*/    IdImm, L4_isub_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, sub, u5_0ImmPred,
        /*anyext*/  IdImm, L4_isub_memoph_io>;
  defm: Memopxi_pat<sextloadi16, truncstorei16, u6_1ImmPred, sub, u5_0ImmPred,
        /*sext*/    IdImm, L4_isub_memoph_io>;
  defm: Memopxi_pat<zextloadi16, truncstorei16, u6_1ImmPred, sub, u5_0ImmPred,
        /*zext*/    IdImm, L4_isub_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, sub, u5_0ImmPred, IdImm,
                    L4_isub_memopw_io>;
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, add, m5_0Imm8Pred,
        /*anyext*/  NegImm8, L4_isub_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, add, m5_0Imm8Pred,
        /*sext*/    NegImm8, L4_isub_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, add, m5_0Imm8Pred,
        /*zext*/    NegImm8, L4_isub_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, add, m5_0Imm16Pred,
        /*anyext*/  NegImm16, L4_isub_memoph_io>;
  defm: Memopxi_pat<sextloadi16, truncstorei16, u6_1ImmPred, add, m5_0Imm16Pred,
        /*sext*/    NegImm16, L4_isub_memoph_io>;
  defm: Memopxi_pat<zextloadi16, truncstorei16, u6_1ImmPred, add, m5_0Imm16Pred,
        /*zext*/    NegImm16, L4_isub_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, add, m5_0ImmPred, NegImm32,
                    L4_isub_memopw_io>;

  // clrbit imm
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, and, IsNPow2_8,
        /*anyext*/  LogN2_8, L4_iand_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, and, IsNPow2_8,
        /*sext*/    LogN2_8, L4_iand_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, and, IsNPow2_8,
        /*zext*/    LogN2_8, L4_iand_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, and, IsNPow2_16,
        /*anyext*/  LogN2_16, L4_iand_memoph_io>;
  defm: Memopxi_pat<sextloadi16, truncstorei16, u6_1ImmPred, and, IsNPow2_16,
        /*sext*/    LogN2_16, L4_iand_memoph_io>;
  defm: Memopxi_pat<zextloadi16, truncstorei16, u6_1ImmPred, and, IsNPow2_16,
        /*zext*/    LogN2_16, L4_iand_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, and, IsNPow2_32,
		    LogN2_32, L4_iand_memopw_io>;

  // setbit imm
  defm: Memopxi_pat<extloadi8, truncstorei8, u6_0ImmPred, or, IsPow2_32,
        /*anyext*/  Log2_8, L4_ior_memopb_io>;
  defm: Memopxi_pat<sextloadi8, truncstorei8, u6_0ImmPred, or, IsPow2_32,
        /*sext*/    Log2_8, L4_ior_memopb_io>;
  defm: Memopxi_pat<zextloadi8, truncstorei8, u6_0ImmPred, or, IsPow2_32,
        /*zext*/    Log2_8, L4_ior_memopb_io>;
  defm: Memopxi_pat<extloadi16, truncstorei16, u6_1ImmPred, or, IsPow2_32,
        /*anyext*/  Log2_16, L4_ior_memoph_io>;
  defm: Memopxi_pat<sextloadi16, truncstorei16, u6_1ImmPred, or, IsPow2_32,
        /*sext*/    Log2_16, L4_ior_memoph_io>;
  defm: Memopxi_pat<zextloadi16, truncstorei16, u6_1ImmPred, or, IsPow2_32,
        /*zext*/    Log2_16, L4_ior_memoph_io>;
  defm: Memopxi_pat<load, store, u6_2ImmPred, or, IsPow2_32,
		    Log2_32, L4_ior_memopw_io>;
}

def : T_CMP_pat <C4_cmpneqi,  setne,  s32_0ImmPred>;
def : T_CMP_pat <C4_cmpltei,  setle,  s32_0ImmPred>;
def : T_CMP_pat <C4_cmplteui, setule, u9_0ImmPred>;

// Map cmplt(Rs, Imm) -> !cmpgt(Rs, Imm-1).
def: Pat<(i1 (setlt I32:$src1, s32_0ImmPred:$src2)),
         (C4_cmpltei IntRegs:$src1, (SDEC1 s32_0ImmPred:$src2))>;

// rs != rt -> !(rs == rt).
def: Pat<(i1 (setne I32:$src1, s32_0ImmPred:$src2)),
         (C4_cmpneqi IntRegs:$src1, s32_0ImmPred:$src2)>;

// For the sequence
//   zext( setult ( and(Rs, 255), u8))
// Use the isdigit transformation below


def u7_0PosImmPred : ImmLeaf<i32, [{
  // True if the immediate fits in an 7-bit unsigned field and
  // is strictly greater than 0.
  return Imm > 0 && isUInt<7>(Imm);
}]>;


// Generate code of the form 'C2_muxii(cmpbgtui(Rdd, C-1),0,1)'
// for C code of the form r = ((c>='0') & (c<='9')) ? 1 : 0;.
// The isdigit transformation relies on two 'clever' aspects:
// 1) The data type is unsigned which allows us to eliminate a zero test after
//    biasing the expression by 48. We are depending on the representation of
//    the unsigned types, and semantics.
// 2) The front end has converted <= 9 into < 10 on entry to LLVM
//
// For the C code:
//   retval = ((c>='0') & (c<='9')) ? 1 : 0;
// The code is transformed upstream of llvm into
//   retval = (c-48) < 10 ? 1 : 0;

let AddedComplexity = 139 in
def: Pat<(i32 (zext (i1 (setult (and I32:$src1, 255), u7_0PosImmPred:$src2)))),
         (C2_muxii (A4_cmpbgtui IntRegs:$src1, (UDEC1 imm:$src2)), 0, 1)>;

class Loada_pat<PatFrag Load, ValueType VT, PatFrag Addr, InstHexagon MI>
  : Pat<(VT (Load Addr:$addr)), (MI Addr:$addr)>;

class Loadam_pat<PatFrag Load, ValueType VT, PatFrag Addr, PatFrag ValueMod,
                 InstHexagon MI>
  : Pat<(VT (Load Addr:$addr)), (ValueMod (MI Addr:$addr))>;

class Storea_pat<PatFrag Store, PatFrag Value, PatFrag Addr, InstHexagon MI>
  : Pat<(Store Value:$val, Addr:$addr), (MI Addr:$addr, Value:$val)>;

class Stoream_pat<PatFrag Store, PatFrag Value, PatFrag Addr, PatFrag ValueMod,
                  InstHexagon MI>
  : Pat<(Store Value:$val, Addr:$addr),
        (MI Addr:$addr, (ValueMod Value:$val))>;

let AddedComplexity = 30 in {
  def: Storea_pat<truncstorei8,  I32, addrga, PS_storerbabs>;
  def: Storea_pat<truncstorei16, I32, addrga, PS_storerhabs>;
  def: Storea_pat<store,         I32, addrga, PS_storeriabs>;
  def: Storea_pat<store,         I64, addrga, PS_storerdabs>;

  def: Stoream_pat<truncstorei8,  I64, addrga, LoReg, PS_storerbabs>;
  def: Stoream_pat<truncstorei16, I64, addrga, LoReg, PS_storerhabs>;
  def: Stoream_pat<truncstorei32, I64, addrga, LoReg, PS_storeriabs>;
}

def: Storea_pat<SwapSt<atomic_store_8>,  I32, addrgp, S2_storerbgp>;
def: Storea_pat<SwapSt<atomic_store_16>, I32, addrgp, S2_storerhgp>;
def: Storea_pat<SwapSt<atomic_store_32>, I32, addrgp, S2_storerigp>;
def: Storea_pat<SwapSt<atomic_store_64>, I64, addrgp, S2_storerdgp>;

let AddedComplexity = 100 in {
  def: Storea_pat<truncstorei8,  I32, addrgp, S2_storerbgp>;
  def: Storea_pat<truncstorei16, I32, addrgp, S2_storerhgp>;
  def: Storea_pat<store,         I32, addrgp, S2_storerigp>;
  def: Storea_pat<store,         I64, addrgp, S2_storerdgp>;

  // Map from "i1 = constant<-1>; memw(CONST32(#foo)) = i1"
  //       to "r0 = 1; memw(#foo) = r0"
  let AddedComplexity = 100 in
  def: Pat<(store (i1 -1), (HexagonCONST32_GP tglobaladdr:$global)),
           (S2_storerbgp tglobaladdr:$global, (A2_tfrsi 1))>;
}

class LoadAbs_pats <PatFrag ldOp, InstHexagon MI, ValueType VT = i32>
  : Pat <(VT (ldOp (HexagonCONST32 tglobaladdr:$absaddr))),
         (VT (MI tglobaladdr:$absaddr))>;

let AddedComplexity  = 30 in {
  def: LoadAbs_pats <load,        PS_loadriabs>;
  def: LoadAbs_pats <zextloadi1,  PS_loadrubabs>;
  def: LoadAbs_pats <sextloadi8,  PS_loadrbabs>;
  def: LoadAbs_pats <extloadi8,   PS_loadrubabs>;
  def: LoadAbs_pats <zextloadi8,  PS_loadrubabs>;
  def: LoadAbs_pats <sextloadi16, PS_loadrhabs>;
  def: LoadAbs_pats <extloadi16,  PS_loadruhabs>;
  def: LoadAbs_pats <zextloadi16, PS_loadruhabs>;
  def: LoadAbs_pats <load,        PS_loadrdabs, i64>;
}

let AddedComplexity  = 30 in
def: Pat<(i64 (zextloadi1 (HexagonCONST32 tglobaladdr:$absaddr))),
         (ToZext64 (PS_loadrubabs tglobaladdr:$absaddr))>;

def: Loada_pat<atomic_load_8,  i32, addrgp, L2_loadrubgp>;
def: Loada_pat<atomic_load_16, i32, addrgp, L2_loadruhgp>;
def: Loada_pat<atomic_load_32, i32, addrgp, L2_loadrigp>;
def: Loada_pat<atomic_load_64, i64, addrgp, L2_loadrdgp>;

def: Loadam_pat<load, i1, addrga, I32toI1, PS_loadrubabs>;
def: Loadam_pat<load, i1, addrgp, I32toI1, L2_loadrubgp>;

def: Stoream_pat<store, I1, addrga, I1toI32, PS_storerbabs>;
def: Stoream_pat<store, I1, addrgp, I1toI32, S2_storerbgp>;

// Map from load(globaladdress) -> mem[u][bhwd](#foo)
class LoadGP_pats <PatFrag ldOp, InstHexagon MI, ValueType VT = i32>
  : Pat <(VT (ldOp (HexagonCONST32_GP tglobaladdr:$global))),
         (VT (MI tglobaladdr:$global))>;

let AddedComplexity = 100 in {
  def: LoadGP_pats <extloadi8,   L2_loadrubgp>;
  def: LoadGP_pats <sextloadi8,  L2_loadrbgp>;
  def: LoadGP_pats <zextloadi8,  L2_loadrubgp>;
  def: LoadGP_pats <extloadi16,  L2_loadruhgp>;
  def: LoadGP_pats <sextloadi16, L2_loadrhgp>;
  def: LoadGP_pats <zextloadi16, L2_loadruhgp>;
  def: LoadGP_pats <load,        L2_loadrigp>;
  def: LoadGP_pats <load,        L2_loadrdgp, i64>;
}

// When the Interprocedural Global Variable optimizer realizes that a certain
// global variable takes only two constant values, it shrinks the global to
// a boolean. Catch those loads here in the following 3 patterns.
let AddedComplexity = 100 in {
  def: LoadGP_pats <extloadi1, L2_loadrubgp>;
  def: LoadGP_pats <zextloadi1, L2_loadrubgp>;
}

// Transfer global address into a register
def: Pat<(HexagonCONST32 tglobaladdr:$Rs),      (A2_tfrsi imm:$Rs)>;
def: Pat<(HexagonCONST32_GP tblockaddress:$Rs), (A2_tfrsi imm:$Rs)>;
def: Pat<(HexagonCONST32_GP tglobaladdr:$Rs),   (A2_tfrsi imm:$Rs)>;

let AddedComplexity  = 30 in {
  def: Storea_pat<truncstorei8,  I32, u32_0ImmPred, PS_storerbabs>;
  def: Storea_pat<truncstorei16, I32, u32_0ImmPred, PS_storerhabs>;
  def: Storea_pat<store,         I32, u32_0ImmPred, PS_storeriabs>;
  def: Storea_pat<store,         I64, u32_0ImmPred, PS_storerdabs>;

  def: Stoream_pat<truncstorei8,  I64, u32_0ImmPred, LoReg, PS_storerbabs>;
  def: Stoream_pat<truncstorei16, I64, u32_0ImmPred, LoReg, PS_storerhabs>;
  def: Stoream_pat<truncstorei32, I64, u32_0ImmPred, LoReg, PS_storeriabs>;
}

let AddedComplexity  = 30 in {
  def: Loada_pat<load,        i32, u32_0ImmPred, PS_loadriabs>;
  def: Loada_pat<sextloadi8,  i32, u32_0ImmPred, PS_loadrbabs>;
  def: Loada_pat<zextloadi8,  i32, u32_0ImmPred, PS_loadrubabs>;
  def: Loada_pat<sextloadi16, i32, u32_0ImmPred, PS_loadrhabs>;
  def: Loada_pat<zextloadi16, i32, u32_0ImmPred, PS_loadruhabs>;
  def: Loada_pat<load,        i64, u32_0ImmPred, PS_loadrdabs>;

  def: Loadam_pat<extloadi8,   i64, u32_0ImmPred, ToZext64, PS_loadrubabs>;
  def: Loadam_pat<sextloadi8,  i64, u32_0ImmPred, ToSext64, PS_loadrbabs>;
  def: Loadam_pat<zextloadi8,  i64, u32_0ImmPred, ToZext64, PS_loadrubabs>;

  def: Loadam_pat<extloadi16,  i64, u32_0ImmPred, ToZext64, PS_loadruhabs>;
  def: Loadam_pat<sextloadi16, i64, u32_0ImmPred, ToSext64, PS_loadrhabs>;
  def: Loadam_pat<zextloadi16, i64, u32_0ImmPred, ToZext64, PS_loadruhabs>;

  def: Loadam_pat<extloadi32,  i64, u32_0ImmPred, ToZext64, PS_loadriabs>;
  def: Loadam_pat<sextloadi32, i64, u32_0ImmPred, ToSext64, PS_loadriabs>;
  def: Loadam_pat<zextloadi32, i64, u32_0ImmPred, ToZext64, PS_loadriabs>;
}

// Indexed store word - global address.
// memw(Rs+#u6:2)=#S8
let AddedComplexity = 100 in
defm: Storex_add_pat<store, addrga, u6_2ImmPred, S4_storeiri_io>;

// Load from a global address that has only one use in the current basic block.
let AddedComplexity = 100 in {
  def: Loada_pat<extloadi8,   i32, addrga, PS_loadrubabs>;
  def: Loada_pat<sextloadi8,  i32, addrga, PS_loadrbabs>;
  def: Loada_pat<zextloadi8,  i32, addrga, PS_loadrubabs>;

  def: Loada_pat<extloadi16,  i32, addrga, PS_loadruhabs>;
  def: Loada_pat<sextloadi16, i32, addrga, PS_loadrhabs>;
  def: Loada_pat<zextloadi16, i32, addrga, PS_loadruhabs>;

  def: Loada_pat<load,        i32, addrga, PS_loadriabs>;
  def: Loada_pat<load,        i64, addrga, PS_loadrdabs>;
}

// Store to a global address that has only one use in the current basic block.
let AddedComplexity = 100 in {
  def: Storea_pat<truncstorei8,  I32, addrga, PS_storerbabs>;
  def: Storea_pat<truncstorei16, I32, addrga, PS_storerhabs>;
  def: Storea_pat<store,         I32, addrga, PS_storeriabs>;
  def: Storea_pat<store,         I64, addrga, PS_storerdabs>;

  def: Stoream_pat<truncstorei32, I64, addrga, LoReg, PS_storeriabs>;
}

// i8/i16/i32 -> i64 loads
// We need a complexity of 120 here to override preceding handling of
// zextload.
let AddedComplexity = 120 in {
  def: Loadam_pat<extloadi8,   i64, addrga, ToZext64, PS_loadrubabs>;
  def: Loadam_pat<sextloadi8,  i64, addrga, ToSext64, PS_loadrbabs>;
  def: Loadam_pat<zextloadi8,  i64, addrga, ToZext64, PS_loadrubabs>;

  def: Loadam_pat<extloadi16,  i64, addrga, ToZext64, PS_loadruhabs>;
  def: Loadam_pat<sextloadi16, i64, addrga, ToSext64, PS_loadrhabs>;
  def: Loadam_pat<zextloadi16, i64, addrga, ToZext64, PS_loadruhabs>;

  def: Loadam_pat<extloadi32,  i64, addrga, ToZext64, PS_loadriabs>;
  def: Loadam_pat<sextloadi32, i64, addrga, ToSext64, PS_loadriabs>;
  def: Loadam_pat<zextloadi32, i64, addrga, ToZext64, PS_loadriabs>;
}

let AddedComplexity = 100 in {
  def: Loada_pat<extloadi8,   i32, addrgp, PS_loadrubabs>;
  def: Loada_pat<sextloadi8,  i32, addrgp, PS_loadrbabs>;
  def: Loada_pat<zextloadi8,  i32, addrgp, PS_loadrubabs>;

  def: Loada_pat<extloadi16,  i32, addrgp, PS_loadruhabs>;
  def: Loada_pat<sextloadi16, i32, addrgp, PS_loadrhabs>;
  def: Loada_pat<zextloadi16, i32, addrgp, PS_loadruhabs>;

  def: Loada_pat<load,        i32, addrgp, PS_loadriabs>;
  def: Loada_pat<load,        i64, addrgp, PS_loadrdabs>;
}

let AddedComplexity = 100 in {
  def: Storea_pat<truncstorei8,  I32, addrgp, PS_storerbabs>;
  def: Storea_pat<truncstorei16, I32, addrgp, PS_storerhabs>;
  def: Storea_pat<store,         I32, addrgp, PS_storeriabs>;
  def: Storea_pat<store,         I64, addrgp, PS_storerdabs>;
}

def: Loada_pat<atomic_load_8,  i32, addrgp, PS_loadrubabs>;
def: Loada_pat<atomic_load_16, i32, addrgp, PS_loadruhabs>;
def: Loada_pat<atomic_load_32, i32, addrgp, PS_loadriabs>;
def: Loada_pat<atomic_load_64, i64, addrgp, PS_loadrdabs>;

def: Storea_pat<SwapSt<atomic_store_8>,  I32, addrgp, PS_storerbabs>;
def: Storea_pat<SwapSt<atomic_store_16>, I32, addrgp, PS_storerhabs>;
def: Storea_pat<SwapSt<atomic_store_32>, I32, addrgp, PS_storeriabs>;
def: Storea_pat<SwapSt<atomic_store_64>, I64, addrgp, PS_storerdabs>;

def: Pat<(or (or (or (shl (i64 (zext (and I32:$b, (i32 65535)))), (i32 16)),
                     (i64 (zext (i32 (and I32:$a, (i32 65535)))))),
                 (shl (i64 (anyext (and I32:$c, (i32 65535)))), (i32 32))),
             (shl (Aext64 I32:$d), (i32 48))),
         (A2_combinew (A2_combine_ll I32:$d, I32:$c),
                      (A2_combine_ll I32:$b, I32:$a))>;

// We need custom lowering of ISD::PREFETCH into HexagonISD::DCFETCH
// because the SDNode ISD::PREFETCH has properties MayLoad and MayStore.
// We don't really want either one here.
def SDTHexagonDCFETCH : SDTypeProfile<0, 2, [SDTCisPtrTy<0>,SDTCisInt<1>]>;
def HexagonDCFETCH : SDNode<"HexagonISD::DCFETCH", SDTHexagonDCFETCH,
                            [SDNPHasChain]>;

def: Pat<(HexagonDCFETCH IntRegs:$Rs, u11_3ImmPred:$u11_3),
         (Y2_dcfetchbo IntRegs:$Rs, imm:$u11_3)>;
def: Pat<(HexagonDCFETCH (i32 (add IntRegs:$Rs, u11_3ImmPred:$u11_3)), (i32 0)),
         (Y2_dcfetchbo IntRegs:$Rs, imm:$u11_3)>;

def f32ImmPred : PatLeaf<(f32 fpimm:$F)>;
def f64ImmPred : PatLeaf<(f64 fpimm:$F)>;

def ftoi : SDNodeXForm<fpimm, [{
  APInt I = N->getValueAPF().bitcastToAPInt();
  return CurDAG->getTargetConstant(I.getZExtValue(), SDLoc(N),
                                   MVT::getIntegerVT(I.getBitWidth()));
}]>;


def: Pat<(sra (i64 (add (sra I64:$src1, u6_0ImmPred:$src2), 1)), (i32 1)),
         (S2_asr_i_p_rnd I64:$src1, imm:$src2)>;

let AddedComplexity = 20 in {
  defm: Loadx_pat<load, f32, s30_2ImmPred, L2_loadri_io>;
  defm: Loadx_pat<load, f64, s29_3ImmPred, L2_loadrd_io>;
}

let AddedComplexity = 60 in {
  defm : T_LoadAbsReg_Pat <load, L4_loadri_ur, f32>;
  defm : T_LoadAbsReg_Pat <load, L4_loadrd_ur, f64>;
}

let AddedComplexity = 40 in {
  def: Loadxs_pat<load, f32, L4_loadri_rr>;
  def: Loadxs_pat<load, f64, L4_loadrd_rr>;
}

let AddedComplexity = 20 in {
  def: Loadxs_simple_pat<load, f32, L4_loadri_rr>;
  def: Loadxs_simple_pat<load, f64, L4_loadrd_rr>;
}

let AddedComplexity  = 80 in {
  def: Loada_pat<load, f32, u32_0ImmPred, PS_loadriabs>;
  def: Loada_pat<load, f32, addrga, PS_loadriabs>;
  def: Loada_pat<load, f64, addrga, PS_loadrdabs>;
}

let AddedComplexity = 100 in {
  def: LoadGP_pats <load, L2_loadrigp, f32>;
  def: LoadGP_pats <load, L2_loadrdgp, f64>;
}

let AddedComplexity = 20 in {
  defm: Storex_pat<store, F32, s30_2ImmPred, S2_storeri_io>;
  defm: Storex_pat<store, F64, s29_3ImmPred, S2_storerd_io>;
}

// Simple patterns should be tried with the least priority.
def: Storex_simple_pat<store, F32, S2_storeri_io>;
def: Storex_simple_pat<store, F64, S2_storerd_io>;

let AddedComplexity = 60 in {
  defm : T_StoreAbsReg_Pats <S4_storeri_ur, IntRegs, f32, store>;
  defm : T_StoreAbsReg_Pats <S4_storerd_ur, DoubleRegs, f64, store>;
}

let AddedComplexity = 40 in {
  def: Storexs_pat<store, F32, S4_storeri_rr>;
  def: Storexs_pat<store, F64, S4_storerd_rr>;
}

let AddedComplexity = 20 in {
  def: Store_rr_pat<store, F32, S4_storeri_rr>;
  def: Store_rr_pat<store, F64, S4_storerd_rr>;
}

let AddedComplexity = 80 in {
  def: Storea_pat<store, F32, addrga, PS_storeriabs>;
  def: Storea_pat<store, F64, addrga, PS_storerdabs>;
}

let AddedComplexity = 100 in {
  def: Storea_pat<store, F32, addrgp, S2_storerigp>;
  def: Storea_pat<store, F64, addrgp, S2_storerdgp>;
}

defm: Storex_pat<store, F32, s30_2ImmPred, S2_storeri_io>;
defm: Storex_pat<store, F64, s29_3ImmPred, S2_storerd_io>;
def: Storex_simple_pat<store, F32, S2_storeri_io>;
def: Storex_simple_pat<store, F64, S2_storerd_io>;

def: Pat<(fadd F32:$src1, F32:$src2),
         (F2_sfadd F32:$src1, F32:$src2)>;

def: Pat<(fsub F32:$src1, F32:$src2),
         (F2_sfsub F32:$src1, F32:$src2)>;

def: Pat<(fmul F32:$src1, F32:$src2),
         (F2_sfmpy F32:$src1, F32:$src2)>;

let Predicates = [HasV5T] in {
  def: Pat<(f32 (fminnum F32:$Rs, F32:$Rt)), (F2_sfmin F32:$Rs, F32:$Rt)>;
  def: Pat<(f32 (fmaxnum F32:$Rs, F32:$Rt)), (F2_sfmax F32:$Rs, F32:$Rt)>;
}

let AddedComplexity = 100, Predicates = [HasV5T] in {
  class SfSel12<PatFrag Cmp, InstHexagon MI>
    : Pat<(select (i1 (Cmp F32:$Rs, F32:$Rt)), F32:$Rs, F32:$Rt),
          (MI F32:$Rs, F32:$Rt)>;
  class SfSel21<PatFrag Cmp, InstHexagon MI>
    : Pat<(select (i1 (Cmp F32:$Rs, F32:$Rt)), F32:$Rt, F32:$Rs),
          (MI F32:$Rs, F32:$Rt)>;

  def: SfSel12<setolt, F2_sfmin>;
  def: SfSel12<setole, F2_sfmin>;
  def: SfSel12<setogt, F2_sfmax>;
  def: SfSel12<setoge, F2_sfmax>;
  def: SfSel21<setolt, F2_sfmax>;
  def: SfSel21<setole, F2_sfmax>;
  def: SfSel21<setogt, F2_sfmin>;
  def: SfSel21<setoge, F2_sfmin>;
}

class T_fcmp32_pat<PatFrag OpNode, InstHexagon MI>
  : Pat<(i1 (OpNode F32:$src1, F32:$src2)),
        (MI F32:$src1, F32:$src2)>;
class T_fcmp64_pat<PatFrag OpNode, InstHexagon MI>
  : Pat<(i1 (OpNode F64:$src1, F64:$src2)),
        (MI F64:$src1, F64:$src2)>;

def: T_fcmp32_pat<setoge, F2_sfcmpge>;
def: T_fcmp32_pat<setuo,  F2_sfcmpuo>;
def: T_fcmp32_pat<setoeq, F2_sfcmpeq>;
def: T_fcmp32_pat<setogt, F2_sfcmpgt>;

def: T_fcmp64_pat<setoge, F2_dfcmpge>;
def: T_fcmp64_pat<setuo,  F2_dfcmpuo>;
def: T_fcmp64_pat<setoeq, F2_dfcmpeq>;
def: T_fcmp64_pat<setogt, F2_dfcmpgt>;

let Predicates = [HasV5T] in
multiclass T_fcmp_pats<PatFrag cmpOp, InstHexagon IntMI, InstHexagon DoubleMI> {
  // IntRegs
  def: Pat<(i1 (cmpOp F32:$src1, F32:$src2)),
           (IntMI F32:$src1, F32:$src2)>;
  // DoubleRegs
  def: Pat<(i1 (cmpOp F64:$src1, F64:$src2)),
           (DoubleMI F64:$src1, F64:$src2)>;
}

defm : T_fcmp_pats <seteq, F2_sfcmpeq, F2_dfcmpeq>;
defm : T_fcmp_pats <setgt, F2_sfcmpgt, F2_dfcmpgt>;
defm : T_fcmp_pats <setge, F2_sfcmpge, F2_dfcmpge>;

//===----------------------------------------------------------------------===//
// Multiclass to define 'Def Pats' for unordered gt, ge, eq operations.
//===----------------------------------------------------------------------===//
let Predicates = [HasV5T] in
multiclass unord_Pats <PatFrag cmpOp, InstHexagon IntMI, InstHexagon DoubleMI> {
  // IntRegs
  def: Pat<(i1 (cmpOp F32:$src1, F32:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, F32:$src2),
                  (IntMI F32:$src1, F32:$src2))>;

  // DoubleRegs
  def: Pat<(i1 (cmpOp F64:$src1, F64:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, F64:$src2),
                  (DoubleMI F64:$src1, F64:$src2))>;
}

defm : unord_Pats <setuge, F2_sfcmpge, F2_dfcmpge>;
defm : unord_Pats <setugt, F2_sfcmpgt, F2_dfcmpgt>;
defm : unord_Pats <setueq, F2_sfcmpeq, F2_dfcmpeq>;

//===----------------------------------------------------------------------===//
// Multiclass to define 'Def Pats' for the following dags:
// seteq(setoeq(op1, op2), 0) -> not(setoeq(op1, op2))
// seteq(setoeq(op1, op2), 1) -> setoeq(op1, op2)
// setne(setoeq(op1, op2), 0) -> setoeq(op1, op2)
// setne(setoeq(op1, op2), 1) -> not(setoeq(op1, op2))
//===----------------------------------------------------------------------===//
let Predicates = [HasV5T] in
multiclass eq_ordgePats <PatFrag cmpOp, InstHexagon IntMI,
                         InstHexagon DoubleMI> {
  // IntRegs
  def: Pat<(i1 (seteq (i1 (cmpOp F32:$src1, F32:$src2)), 0)),
           (C2_not (IntMI F32:$src1, F32:$src2))>;
  def: Pat<(i1 (seteq (i1 (cmpOp F32:$src1, F32:$src2)), 1)),
           (IntMI F32:$src1, F32:$src2)>;
  def: Pat<(i1 (setne (i1 (cmpOp F32:$src1, F32:$src2)), 0)),
           (IntMI F32:$src1, F32:$src2)>;
  def: Pat<(i1 (setne (i1 (cmpOp F32:$src1, F32:$src2)), 1)),
           (C2_not (IntMI F32:$src1, F32:$src2))>;

  // DoubleRegs
  def : Pat<(i1 (seteq (i1 (cmpOp F64:$src1, F64:$src2)), 0)),
            (C2_not (DoubleMI F64:$src1, F64:$src2))>;
  def : Pat<(i1 (seteq (i1 (cmpOp F64:$src1, F64:$src2)), 1)),
            (DoubleMI F64:$src1, F64:$src2)>;
  def : Pat<(i1 (setne (i1 (cmpOp F64:$src1, F64:$src2)), 0)),
            (DoubleMI F64:$src1, F64:$src2)>;
  def : Pat<(i1 (setne (i1 (cmpOp F64:$src1, F64:$src2)), 1)),
            (C2_not (DoubleMI F64:$src1, F64:$src2))>;
}

defm : eq_ordgePats<setoeq, F2_sfcmpeq, F2_dfcmpeq>;
defm : eq_ordgePats<setoge, F2_sfcmpge, F2_dfcmpge>;
defm : eq_ordgePats<setogt, F2_sfcmpgt, F2_dfcmpgt>;

//===----------------------------------------------------------------------===//
// Multiclass to define 'Def Pats' for the following dags:
// seteq(setolt(op1, op2), 0) -> not(setogt(op2, op1))
// seteq(setolt(op1, op2), 1) -> setogt(op2, op1)
// setne(setolt(op1, op2), 0) -> setogt(op2, op1)
// setne(setolt(op1, op2), 1) -> not(setogt(op2, op1))
//===----------------------------------------------------------------------===//
let Predicates = [HasV5T] in
multiclass eq_ordltPats <PatFrag cmpOp, InstHexagon IntMI,
                         InstHexagon DoubleMI> {
  // IntRegs
  def: Pat<(i1 (seteq (i1 (cmpOp F32:$src1, F32:$src2)), 0)),
           (C2_not (IntMI F32:$src2, F32:$src1))>;
  def: Pat<(i1 (seteq (i1 (cmpOp F32:$src1, F32:$src2)), 1)),
           (IntMI F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setne (i1 (cmpOp F32:$src1, F32:$src2)), 0)),
           (IntMI F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setne (i1 (cmpOp F32:$src1, F32:$src2)), 1)),
           (C2_not (IntMI F32:$src2, F32:$src1))>;

  // DoubleRegs
  def: Pat<(i1 (seteq (i1 (cmpOp F64:$src1, F64:$src2)), 0)),
           (C2_not (DoubleMI F64:$src2, F64:$src1))>;
  def: Pat<(i1 (seteq (i1 (cmpOp F64:$src1, F64:$src2)), 1)),
           (DoubleMI F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setne (i1 (cmpOp F64:$src1, F64:$src2)), 0)),
           (DoubleMI F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setne (i1 (cmpOp F64:$src1, F64:$src2)), 0)),
           (C2_not (DoubleMI F64:$src2, F64:$src1))>;
}

defm : eq_ordltPats<setole, F2_sfcmpge, F2_dfcmpge>;
defm : eq_ordltPats<setolt, F2_sfcmpgt, F2_dfcmpgt>;


// o. seto inverse of setuo. http://llvm.org/docs/LangRef.html#i_fcmp
let Predicates = [HasV5T] in {
  def: Pat<(i1 (seto F32:$src1, F32:$src2)),
           (C2_not (F2_sfcmpuo F32:$src2, F32:$src1))>;
  def: Pat<(i1 (seto F32:$src1, f32ImmPred:$src2)),
           (C2_not (F2_sfcmpuo (f32 (A2_tfrsi (ftoi $src2))), F32:$src1))>;
  def: Pat<(i1 (seto F64:$src1, F64:$src2)),
           (C2_not (F2_dfcmpuo F64:$src2, F64:$src1))>;
  def: Pat<(i1 (seto F64:$src1, f64ImmPred:$src2)),
           (C2_not (F2_dfcmpuo (CONST64 (ftoi $src2)), F64:$src1))>;
}

// Ordered lt.
let Predicates = [HasV5T] in {
  def: Pat<(i1 (setolt F32:$src1, F32:$src2)),
           (F2_sfcmpgt F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setolt F32:$src1, f32ImmPred:$src2)),
           (F2_sfcmpgt (f32 (A2_tfrsi (ftoi $src2))), F32:$src1)>;
  def: Pat<(i1 (setolt F64:$src1, F64:$src2)),
           (F2_dfcmpgt F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setolt F64:$src1, f64ImmPred:$src2)),
           (F2_dfcmpgt (CONST64 (ftoi $src2)), F64:$src1)>;
}

// Unordered lt.
let Predicates = [HasV5T] in {
  def: Pat<(i1 (setult F32:$src1, F32:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, F32:$src2),
                  (F2_sfcmpgt F32:$src2, F32:$src1))>;
  def: Pat<(i1 (setult F32:$src1, f32ImmPred:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, (f32 (A2_tfrsi (ftoi $src2)))),
                  (F2_sfcmpgt (f32 (A2_tfrsi (ftoi $src2))), F32:$src1))>;
  def: Pat<(i1 (setult F64:$src1, F64:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, F64:$src2),
                  (F2_dfcmpgt F64:$src2, F64:$src1))>;
  def: Pat<(i1 (setult F64:$src1, f64ImmPred:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, (CONST64 (ftoi $src2))),
                  (F2_dfcmpgt (CONST64 (ftoi $src2)), F64:$src1))>;
}

// Ordered le.
let Predicates = [HasV5T] in {
  // rs <= rt -> rt >= rs.
  def: Pat<(i1 (setole F32:$src1, F32:$src2)),
           (F2_sfcmpge F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setole F32:$src1, f32ImmPred:$src2)),
           (F2_sfcmpge (f32 (A2_tfrsi (ftoi $src2))), F32:$src1)>;

  // Rss <= Rtt -> Rtt >= Rss.
  def: Pat<(i1 (setole F64:$src1, F64:$src2)),
           (F2_dfcmpge F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setole F64:$src1, f64ImmPred:$src2)),
           (F2_dfcmpge (CONST64 (ftoi $src2)), F64:$src1)>;
}

// Unordered le.
let Predicates = [HasV5T] in {
// rs <= rt -> rt >= rs.
  def: Pat<(i1 (setule F32:$src1, F32:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, F32:$src2),
                  (F2_sfcmpge F32:$src2, F32:$src1))>;
  def: Pat<(i1 (setule F32:$src1, f32ImmPred:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, (f32 (A2_tfrsi (ftoi $src2)))),
                  (F2_sfcmpge (f32 (A2_tfrsi (ftoi $src2))), F32:$src1))>;
  def: Pat<(i1 (setule F64:$src1, F64:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, F64:$src2),
                  (F2_dfcmpge F64:$src2, F64:$src1))>;
  def: Pat<(i1 (setule F64:$src1, f64ImmPred:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, (CONST64 (ftoi $src2))),
                  (F2_dfcmpge (CONST64 (ftoi $src2)), F64:$src1))>;
}

// Ordered ne.
let Predicates = [HasV5T] in {
  def: Pat<(i1 (setone F32:$src1, F32:$src2)),
           (C2_not (F2_sfcmpeq F32:$src1, F32:$src2))>;
  def: Pat<(i1 (setone F64:$src1, F64:$src2)),
           (C2_not (F2_dfcmpeq F64:$src1, F64:$src2))>;
  def: Pat<(i1 (setone F32:$src1, f32ImmPred:$src2)),
           (C2_not (F2_sfcmpeq F32:$src1, (f32 (A2_tfrsi (ftoi $src2)))))>;
  def: Pat<(i1 (setone F64:$src1, f64ImmPred:$src2)),
           (C2_not (F2_dfcmpeq F64:$src1, (CONST64 (ftoi $src2))))>;
}

// Unordered ne.
let Predicates = [HasV5T] in {
  def: Pat<(i1 (setune F32:$src1, F32:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, F32:$src2),
                  (C2_not (F2_sfcmpeq F32:$src1, F32:$src2)))>;
  def: Pat<(i1 (setune F64:$src1, F64:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, F64:$src2),
                  (C2_not (F2_dfcmpeq F64:$src1, F64:$src2)))>;
  def: Pat<(i1 (setune F32:$src1, f32ImmPred:$src2)),
           (C2_or (F2_sfcmpuo F32:$src1, (f32 (A2_tfrsi (ftoi $src2)))),
                  (C2_not (F2_sfcmpeq F32:$src1,
                                      (f32 (A2_tfrsi (ftoi $src2))))))>;
  def: Pat<(i1 (setune F64:$src1, f64ImmPred:$src2)),
           (C2_or (F2_dfcmpuo F64:$src1, (CONST64 (ftoi $src2))),
                  (C2_not (F2_dfcmpeq F64:$src1,
                                      (CONST64 (ftoi $src2)))))>;
}

// Besides set[o|u][comparions], we also need set[comparisons].
let Predicates = [HasV5T] in {
  // lt.
  def: Pat<(i1 (setlt F32:$src1, F32:$src2)),
           (F2_sfcmpgt F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setlt F32:$src1, f32ImmPred:$src2)),
           (F2_sfcmpgt (f32 (A2_tfrsi (ftoi $src2))), F32:$src1)>;
  def: Pat<(i1 (setlt F64:$src1, F64:$src2)),
           (F2_dfcmpgt F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setlt F64:$src1, f64ImmPred:$src2)),
           (F2_dfcmpgt (CONST64 (ftoi $src2)), F64:$src1)>;

  // le.
  // rs <= rt -> rt >= rs.
  def: Pat<(i1 (setle F32:$src1, F32:$src2)),
           (F2_sfcmpge F32:$src2, F32:$src1)>;
  def: Pat<(i1 (setle F32:$src1, f32ImmPred:$src2)),
           (F2_sfcmpge (f32 (A2_tfrsi (ftoi $src2))), F32:$src1)>;

  // Rss <= Rtt -> Rtt >= Rss.
  def: Pat<(i1 (setle F64:$src1, F64:$src2)),
           (F2_dfcmpge F64:$src2, F64:$src1)>;
  def: Pat<(i1 (setle F64:$src1, f64ImmPred:$src2)),
           (F2_dfcmpge (CONST64 (ftoi $src2)), F64:$src1)>;

  // ne.
  def: Pat<(i1 (setne F32:$src1, F32:$src2)),
           (C2_not (F2_sfcmpeq F32:$src1, F32:$src2))>;
  def: Pat<(i1 (setne F64:$src1, F64:$src2)),
           (C2_not (F2_dfcmpeq F64:$src1, F64:$src2))>;
  def: Pat<(i1 (setne F32:$src1, f32ImmPred:$src2)),
           (C2_not (F2_sfcmpeq F32:$src1, (f32 (A2_tfrsi (ftoi $src2)))))>;
  def: Pat<(i1 (setne F64:$src1, f64ImmPred:$src2)),
           (C2_not (F2_dfcmpeq F64:$src1, (CONST64 (ftoi $src2))))>;
}


def: Pat<(f64 (fpextend F32:$Rs)), (F2_conv_sf2df F32:$Rs)>;
def: Pat<(f32 (fpround F64:$Rs)), (F2_conv_df2sf F64:$Rs)>;

def: Pat<(f32 (sint_to_fp I32:$Rs)), (F2_conv_w2sf I32:$Rs)>;
def: Pat<(f32 (sint_to_fp I64:$Rs)), (F2_conv_d2sf I64:$Rs)>;
def: Pat<(f64 (sint_to_fp I32:$Rs)), (F2_conv_w2df I32:$Rs)>;
def: Pat<(f64 (sint_to_fp I64:$Rs)), (F2_conv_d2df I64:$Rs)>;

def: Pat<(f32 (uint_to_fp I32:$Rs)), (F2_conv_uw2sf I32:$Rs)>;
def: Pat<(f32 (uint_to_fp I64:$Rs)), (F2_conv_ud2sf I64:$Rs)>;
def: Pat<(f64 (uint_to_fp I32:$Rs)), (F2_conv_uw2df I32:$Rs)>;
def: Pat<(f64 (uint_to_fp I64:$Rs)), (F2_conv_ud2df I64:$Rs)>;

def: Pat<(i32 (fp_to_sint F32:$Rs)), (F2_conv_sf2w_chop F32:$Rs)>;
def: Pat<(i32 (fp_to_sint F64:$Rs)), (F2_conv_df2w_chop F64:$Rs)>;
def: Pat<(i64 (fp_to_sint F32:$Rs)), (F2_conv_sf2d_chop F32:$Rs)>;
def: Pat<(i64 (fp_to_sint F64:$Rs)), (F2_conv_df2d_chop F64:$Rs)>;

def: Pat<(i32 (fp_to_uint F32:$Rs)), (F2_conv_sf2uw_chop F32:$Rs)>;
def: Pat<(i32 (fp_to_uint F64:$Rs)), (F2_conv_df2uw_chop F64:$Rs)>;
def: Pat<(i64 (fp_to_uint F32:$Rs)), (F2_conv_sf2ud_chop F32:$Rs)>;
def: Pat<(i64 (fp_to_uint F64:$Rs)), (F2_conv_df2ud_chop F64:$Rs)>;

// Bitcast is different than [fp|sint|uint]_to_[sint|uint|fp].
let Predicates = [HasV5T] in {
  def: Pat <(i32 (bitconvert F32:$src)), (I32:$src)>;
  def: Pat <(f32 (bitconvert I32:$src)), (F32:$src)>;
  def: Pat <(i64 (bitconvert F64:$src)), (I64:$src)>;
  def: Pat <(f64 (bitconvert I64:$src)), (F64:$src)>;
}

def : Pat <(fma F32:$src2, F32:$src3, F32:$src1),
           (F2_sffma F32:$src1, F32:$src2, F32:$src3)>;

def : Pat <(fma (fneg F32:$src2), F32:$src3, F32:$src1),
           (F2_sffms F32:$src1, F32:$src2, F32:$src3)>;

def : Pat <(fma F32:$src2, (fneg F32:$src3), F32:$src1),
           (F2_sffms F32:$src1, F32:$src2, F32:$src3)>;

def: Pat<(select I1:$Pu, F32:$Rs, f32ImmPred:$imm),
         (C2_muxir I1:$Pu, F32:$Rs, (ftoi $imm))>,
    Requires<[HasV5T]>;

def: Pat<(select I1:$Pu, f32ImmPred:$imm, F32:$Rt),
         (C2_muxri I1:$Pu, (ftoi $imm), F32:$Rt)>,
    Requires<[HasV5T]>;

def: Pat<(select I1:$src1, F32:$src2, F32:$src3),
         (C2_mux I1:$src1, F32:$src2, F32:$src3)>,
     Requires<[HasV5T]>;

def: Pat<(select (i1 (setult F32:$src1, F32:$src2)), F32:$src3, F32:$src4),
         (C2_mux (F2_sfcmpgt F32:$src2, F32:$src1), F32:$src4, F32:$src3)>,
     Requires<[HasV5T]>;

def: Pat<(select I1:$src1, F64:$src2, F64:$src3),
         (C2_vmux I1:$src1, F64:$src2, F64:$src3)>,
    Requires<[HasV5T]>;

def: Pat<(select (i1 (setult F64:$src1, F64:$src2)), F64:$src3, F64:$src4),
         (C2_vmux (F2_dfcmpgt F64:$src2, F64:$src1), F64:$src3, F64:$src4)>,
     Requires<[HasV5T]>;

// Map from p0 = pnot(p0); r0 = select(p0, #i, r1)
// => r0 = mux(p0, #i, r1)
def: Pat<(select (not I1:$src1), f32ImmPred:$src2, F32:$src3),
         (C2_muxir I1:$src1, F32:$src3, (ftoi $src2))>,
     Requires<[HasV5T]>;

// Map from p0 = pnot(p0); r0 = mux(p0, r1, #i)
// => r0 = mux(p0, r1, #i)
def: Pat<(select (not I1:$src1), F32:$src2, f32ImmPred:$src3),
         (C2_muxri I1:$src1, (ftoi $src3), F32:$src2)>,
     Requires<[HasV5T]>;

def: Pat<(i32 (fp_to_sint F64:$src1)),
         (LoReg (F2_conv_df2d_chop F64:$src1))>,
     Requires<[HasV5T]>;

def : Pat <(fabs F32:$src1),
           (S2_clrbit_i F32:$src1, 31)>,
          Requires<[HasV5T]>;

def : Pat <(fneg F32:$src1),
           (S2_togglebit_i F32:$src1, 31)>,
          Requires<[HasV5T]>;

def: Pat<(fabs F64:$Rs),
         (REG_SEQUENCE DoubleRegs,
              (S2_clrbit_i (HiReg $Rs), 31), isub_hi,
              (i32 (LoReg $Rs)), isub_lo)>;

def: Pat<(fneg F64:$Rs),
         (REG_SEQUENCE DoubleRegs,
              (S2_togglebit_i (HiReg $Rs), 31), isub_hi,
              (i32 (LoReg $Rs)), isub_lo)>;

def: Pat<(mul I64:$Rss, I64:$Rtt),
         (A2_combinew
           (M2_maci (M2_maci (HiReg (M2_dpmpyuu_s0 (LoReg $Rss), (LoReg $Rtt))),
                             (LoReg $Rss),
                             (HiReg $Rtt)),
                    (LoReg $Rtt),
                    (HiReg $Rss)),
           (LoReg (M2_dpmpyuu_s0 (LoReg $Rss), (LoReg $Rtt))))>;

def alignedload : PatFrag<(ops node:$addr), (load $addr), [{
  return isAlignedMemNode(dyn_cast<MemSDNode>(N));
}]>;

def unalignedload : PatFrag<(ops node:$addr), (load $addr), [{
  return !isAlignedMemNode(dyn_cast<MemSDNode>(N));
}]>;

def alignedstore : PatFrag<(ops node:$val, node:$addr), (store $val, $addr), [{
  return isAlignedMemNode(dyn_cast<MemSDNode>(N));
}]>;

def unalignedstore : PatFrag<(ops node:$val, node:$addr), (store $val, $addr), [{
  return !isAlignedMemNode(dyn_cast<MemSDNode>(N));
}]>;


multiclass vS32b_ai_pats <ValueType VTSgl, ValueType VTDbl> {
  // Aligned stores
  def : Pat<(alignedstore (VTSgl VectorRegs:$src1), IntRegs:$addr),
            (V6_vS32b_ai IntRegs:$addr, 0, (VTSgl VectorRegs:$src1))>,
            Requires<[UseHVXSgl]>;
  def : Pat<(unalignedstore (VTSgl VectorRegs:$src1), IntRegs:$addr),
            (V6_vS32Ub_ai IntRegs:$addr, 0, (VTSgl VectorRegs:$src1))>,
            Requires<[UseHVXSgl]>;

  // 128B Aligned stores
  def : Pat<(alignedstore (VTDbl VectorRegs128B:$src1), IntRegs:$addr),
            (V6_vS32b_ai_128B IntRegs:$addr, 0, (VTDbl VectorRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;
  def : Pat<(unalignedstore (VTDbl VectorRegs128B:$src1), IntRegs:$addr),
            (V6_vS32Ub_ai_128B IntRegs:$addr, 0, (VTDbl VectorRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;

  // Fold Add R+OFF into vector store.
  let AddedComplexity = 10 in {
    def : Pat<(alignedstore (VTSgl VectorRegs:$src1),
                     (add IntRegs:$src2, Iss4_6:$offset)),
              (V6_vS32b_ai IntRegs:$src2, Iss4_6:$offset,
                           (VTSgl VectorRegs:$src1))>,
              Requires<[UseHVXSgl]>;
    def : Pat<(unalignedstore (VTSgl VectorRegs:$src1),
                     (add IntRegs:$src2, Iss4_6:$offset)),
              (V6_vS32Ub_ai IntRegs:$src2, Iss4_6:$offset,
                           (VTSgl VectorRegs:$src1))>,
              Requires<[UseHVXSgl]>;

    // Fold Add R+OFF into vector store 128B.
    def : Pat<(alignedstore (VTDbl VectorRegs128B:$src1),
                     (add IntRegs:$src2, Iss4_7:$offset)),
              (V6_vS32b_ai_128B IntRegs:$src2, Iss4_7:$offset,
                                (VTDbl VectorRegs128B:$src1))>,
              Requires<[UseHVXDbl]>;
    def : Pat<(unalignedstore (VTDbl VectorRegs128B:$src1),
                     (add IntRegs:$src2, Iss4_7:$offset)),
              (V6_vS32Ub_ai_128B IntRegs:$src2, Iss4_7:$offset,
                                (VTDbl VectorRegs128B:$src1))>,
              Requires<[UseHVXDbl]>;
  }
}

defm : vS32b_ai_pats <v64i8,  v128i8>;
defm : vS32b_ai_pats <v32i16, v64i16>;
defm : vS32b_ai_pats <v16i32, v32i32>;
defm : vS32b_ai_pats <v8i64,  v16i64>;


multiclass vL32b_ai_pats <ValueType VTSgl, ValueType VTDbl> {
  // Aligned loads
  def : Pat < (VTSgl (alignedload IntRegs:$addr)),
              (V6_vL32b_ai IntRegs:$addr, 0) >,
              Requires<[UseHVXSgl]>;
  def : Pat < (VTSgl (unalignedload IntRegs:$addr)),
              (V6_vL32Ub_ai IntRegs:$addr, 0) >,
              Requires<[UseHVXSgl]>;

  // 128B Load
  def : Pat < (VTDbl (alignedload IntRegs:$addr)),
              (V6_vL32b_ai_128B IntRegs:$addr, 0) >,
              Requires<[UseHVXDbl]>;
  def : Pat < (VTDbl (unalignedload IntRegs:$addr)),
              (V6_vL32Ub_ai_128B IntRegs:$addr, 0) >,
              Requires<[UseHVXDbl]>;

  // Fold Add R+OFF into vector load.
  let AddedComplexity = 10 in {
    def : Pat<(VTDbl (alignedload (add IntRegs:$src2, Iss4_7:$offset))),
              (V6_vL32b_ai_128B IntRegs:$src2, Iss4_7:$offset)>,
               Requires<[UseHVXDbl]>;
    def : Pat<(VTDbl (unalignedload (add IntRegs:$src2, Iss4_7:$offset))),
              (V6_vL32Ub_ai_128B IntRegs:$src2, Iss4_7:$offset)>,
               Requires<[UseHVXDbl]>;

    def : Pat<(VTSgl (alignedload (add IntRegs:$src2, Iss4_6:$offset))),
              (V6_vL32b_ai IntRegs:$src2, Iss4_6:$offset)>,
              Requires<[UseHVXSgl]>;
    def : Pat<(VTSgl (unalignedload (add IntRegs:$src2, Iss4_6:$offset))),
              (V6_vL32Ub_ai IntRegs:$src2, Iss4_6:$offset)>,
              Requires<[UseHVXSgl]>;
  }
}

defm : vL32b_ai_pats <v64i8,  v128i8>;
defm : vL32b_ai_pats <v32i16, v64i16>;
defm : vL32b_ai_pats <v16i32, v32i32>;
defm : vL32b_ai_pats <v8i64,  v16i64>;

multiclass STrivv_pats <ValueType VTSgl, ValueType VTDbl> {
  def : Pat<(alignedstore (VTSgl VecDblRegs:$src1), IntRegs:$addr),
            (PS_vstorerw_ai IntRegs:$addr, 0, (VTSgl VecDblRegs:$src1))>,
           Requires<[UseHVXSgl]>;
  def : Pat<(unalignedstore (VTSgl VecDblRegs:$src1), IntRegs:$addr),
            (PS_vstorerwu_ai IntRegs:$addr, 0, (VTSgl VecDblRegs:$src1))>,
           Requires<[UseHVXSgl]>;

  def : Pat<(alignedstore (VTDbl VecDblRegs128B:$src1), IntRegs:$addr),
            (PS_vstorerw_ai_128B IntRegs:$addr, 0,
                  (VTDbl VecDblRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;
  def : Pat<(unalignedstore (VTDbl VecDblRegs128B:$src1), IntRegs:$addr),
            (PS_vstorerwu_ai_128B IntRegs:$addr, 0,
                  (VTDbl VecDblRegs128B:$src1))>,
            Requires<[UseHVXDbl]>;
}

defm : STrivv_pats <v128i8, v256i8>;
defm : STrivv_pats <v64i16, v128i16>;
defm : STrivv_pats <v32i32, v64i32>;
defm : STrivv_pats <v16i64, v32i64>;

multiclass LDrivv_pats <ValueType VTSgl, ValueType VTDbl> {
  def : Pat<(VTSgl (alignedload I32:$addr)),
            (PS_vloadrw_ai I32:$addr, 0)>,
           Requires<[UseHVXSgl]>;
  def : Pat<(VTSgl (unalignedload I32:$addr)),
            (PS_vloadrwu_ai I32:$addr, 0)>,
           Requires<[UseHVXSgl]>;

  def : Pat<(VTDbl (alignedload I32:$addr)),
            (PS_vloadrw_ai_128B I32:$addr, 0)>,
           Requires<[UseHVXDbl]>;
  def : Pat<(VTDbl (unalignedload I32:$addr)),
            (PS_vloadrwu_ai_128B I32:$addr, 0)>,
           Requires<[UseHVXDbl]>;
}

defm : LDrivv_pats <v128i8, v256i8>;
defm : LDrivv_pats <v64i16, v128i16>;
defm : LDrivv_pats <v32i32, v64i32>;
defm : LDrivv_pats <v16i64, v32i64>;

let Predicates = [HasV60T,UseHVXSgl] in {
  def: Pat<(select I1:$Pu, (v16i32 VectorRegs:$Vs), VectorRegs:$Vt),
           (PS_vselect I1:$Pu, VectorRegs:$Vs, VectorRegs:$Vt)>;
  def: Pat<(select I1:$Pu, (v32i32 VecDblRegs:$Vs), VecDblRegs:$Vt),
           (PS_wselect I1:$Pu, VecDblRegs:$Vs, VecDblRegs:$Vt)>;
}
let Predicates = [HasV60T,UseHVXDbl] in {
  def: Pat<(select I1:$Pu, (v32i32 VectorRegs128B:$Vs), VectorRegs128B:$Vt),
           (PS_vselect_128B I1:$Pu, VectorRegs128B:$Vs, VectorRegs128B:$Vt)>;
  def: Pat<(select I1:$Pu, (v64i32 VecDblRegs128B:$Vs), VecDblRegs128B:$Vt),
           (PS_wselect_128B I1:$Pu, VecDblRegs128B:$Vs, VecDblRegs128B:$Vt)>;
}


def SDTHexagonVCOMBINE: SDTypeProfile<1, 2, [SDTCisSameAs<1, 2>,
      SDTCisSubVecOfVec<1, 0>]>;

def HexagonVCOMBINE: SDNode<"HexagonISD::VCOMBINE", SDTHexagonVCOMBINE>;

def: Pat<(v32i32 (HexagonVCOMBINE (v16i32 VectorRegs:$Vs),
                                  (v16i32 VectorRegs:$Vt))),
         (V6_vcombine VectorRegs:$Vs, VectorRegs:$Vt)>,
         Requires<[UseHVXSgl]>;
def: Pat<(v64i32 (HexagonVCOMBINE (v32i32 VecDblRegs:$Vs),
                                  (v32i32 VecDblRegs:$Vt))),
         (V6_vcombine_128B VecDblRegs:$Vs, VecDblRegs:$Vt)>,
         Requires<[UseHVXDbl]>;

def SDTHexagonVPACK: SDTypeProfile<1, 3, [SDTCisSameAs<1, 2>,
                                          SDTCisInt<3>]>;

def HexagonVPACK: SDNode<"HexagonISD::VPACK", SDTHexagonVPACK>;

// 0 as the last argument denotes vpacke. 1 denotes vpacko
def: Pat<(v64i8 (HexagonVPACK (v64i8 VectorRegs:$Vs),
                              (v64i8 VectorRegs:$Vt), (i32 0))),
         (V6_vpackeb VectorRegs:$Vs, VectorRegs:$Vt)>,
         Requires<[UseHVXSgl]>;
def: Pat<(v64i8 (HexagonVPACK (v64i8 VectorRegs:$Vs),
                              (v64i8 VectorRegs:$Vt), (i32 1))),
         (V6_vpackob VectorRegs:$Vs, VectorRegs:$Vt)>,
         Requires<[UseHVXSgl]>;
def: Pat<(v32i16 (HexagonVPACK (v32i16 VectorRegs:$Vs),
                               (v32i16 VectorRegs:$Vt), (i32 0))),
         (V6_vpackeh VectorRegs:$Vs, VectorRegs:$Vt)>,
         Requires<[UseHVXSgl]>;
def: Pat<(v32i16 (HexagonVPACK (v32i16 VectorRegs:$Vs),
                             (v32i16 VectorRegs:$Vt), (i32 1))),
         (V6_vpackoh VectorRegs:$Vs, VectorRegs:$Vt)>,
         Requires<[UseHVXSgl]>;

def: Pat<(v128i8 (HexagonVPACK (v128i8 VecDblRegs:$Vs),
                             (v128i8 VecDblRegs:$Vt), (i32 0))),
         (V6_vpackeb_128B VecDblRegs:$Vs, VecDblRegs:$Vt)>,
         Requires<[UseHVXDbl]>;
def: Pat<(v128i8 (HexagonVPACK (v128i8 VecDblRegs:$Vs),
                             (v128i8 VecDblRegs:$Vt), (i32 1))),
         (V6_vpackob_128B VecDblRegs:$Vs, VecDblRegs:$Vt)>,
         Requires<[UseHVXDbl]>;
def: Pat<(v64i16 (HexagonVPACK (v64i16 VecDblRegs:$Vs),
                             (v64i16 VecDblRegs:$Vt), (i32 0))),
         (V6_vpackeh_128B VecDblRegs:$Vs, VecDblRegs:$Vt)>,
         Requires<[UseHVXDbl]>;
def: Pat<(v64i16 (HexagonVPACK (v64i16 VecDblRegs:$Vs),
                            (v64i16 VecDblRegs:$Vt), (i32 1))),
        (V6_vpackoh_128B VecDblRegs:$Vs, VecDblRegs:$Vt)>,
        Requires<[UseHVXDbl]>;

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

// Bit convert vector types to integers.
defm : bitconvert_32<v4i8,  i32>;
defm : bitconvert_32<v2i16, i32>;
defm : bitconvert_64<v8i8,  i64>;
defm : bitconvert_64<v4i16, i64>;
defm : bitconvert_64<v2i32, i64>;

def: Pat<(sra (v4i16 DoubleRegs:$src1), u4_0ImmPred:$src2),
         (S2_asr_i_vh DoubleRegs:$src1, imm:$src2)>;
def: Pat<(srl (v4i16 DoubleRegs:$src1), u4_0ImmPred:$src2),
         (S2_lsr_i_vh DoubleRegs:$src1, imm:$src2)>;
def: Pat<(shl (v4i16 DoubleRegs:$src1), u4_0ImmPred:$src2),
         (S2_asl_i_vh DoubleRegs:$src1, imm:$src2)>;

def: Pat<(sra (v2i32 DoubleRegs:$src1), u5_0ImmPred:$src2),
         (S2_asr_i_vw DoubleRegs:$src1, imm:$src2)>;
def: Pat<(srl (v2i32 DoubleRegs:$src1), u5_0ImmPred:$src2),
         (S2_lsr_i_vw DoubleRegs:$src1, imm:$src2)>;
def: Pat<(shl (v2i32 DoubleRegs:$src1), u5_0ImmPred:$src2),
         (S2_asl_i_vw DoubleRegs:$src1, imm:$src2)>;

def : Pat<(v2i16 (add (v2i16 IntRegs:$src1), (v2i16 IntRegs:$src2))),
          (A2_svaddh IntRegs:$src1, IntRegs:$src2)>;

def : Pat<(v2i16 (sub (v2i16 IntRegs:$src1), (v2i16 IntRegs:$src2))),
          (A2_svsubh IntRegs:$src1, IntRegs:$src2)>;

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

def: Pat<(v2i32 (sra V2I32:$b, (i64 (HexagonCOMBINE (i32 u5_0ImmPred:$c),
                                                    (i32 u5_0ImmPred:$c))))),
         (S2_asr_i_vw V2I32:$b, imm:$c)>;
def: Pat<(v2i32 (srl V2I32:$b, (i64 (HexagonCOMBINE (i32 u5_0ImmPred:$c),
                                                    (i32 u5_0ImmPred:$c))))),
         (S2_lsr_i_vw V2I32:$b, imm:$c)>;
def: Pat<(v2i32 (shl V2I32:$b, (i64 (HexagonCOMBINE (i32 u5_0ImmPred:$c),
                                                    (i32 u5_0ImmPred:$c))))),
         (S2_asl_i_vw V2I32:$b, imm:$c)>;

def: Pat<(v4i16 (sra V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4_0ImmPred:$c)))))),
         (S2_asr_i_vh V4I16:$b, imm:$c)>;
def: Pat<(v4i16 (srl V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4_0ImmPred:$c)))))),
         (S2_lsr_i_vh V4I16:$b, imm:$c)>;
def: Pat<(v4i16 (shl V4I16:$b, (v4i16 (HexagonVSPLATH (i32 (u4_0ImmPred:$c)))))),
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

def: Pat<(v2i32 (HexagonVSRAW V2I32:$Rs, u5_0ImmPred:$u5)),
         (S2_asr_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSRAH V4I16:$Rs, u4_0ImmPred:$u4)),
         (S2_asr_i_vh V4I16:$Rs, imm:$u4)>;
def: Pat<(v2i32 (HexagonVSRLW V2I32:$Rs, u5_0ImmPred:$u5)),
         (S2_lsr_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSRLH V4I16:$Rs, u4_0ImmPred:$u4)),
         (S2_lsr_i_vh V4I16:$Rs, imm:$u4)>;
def: Pat<(v2i32 (HexagonVSHLW V2I32:$Rs, u5_0ImmPred:$u5)),
         (S2_asl_i_vw V2I32:$Rs, imm:$u5)>;
def: Pat<(v4i16 (HexagonVSHLH V4I16:$Rs, u4_0ImmPred:$u4)),
         (S2_asl_i_vh V4I16:$Rs, imm:$u4)>;

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

def: Pat<(mul V2I32:$Rs, V2I32:$Rt),
         (PS_vmulw DoubleRegs:$Rs, DoubleRegs:$Rt)>;
def: Pat<(add V2I32:$Rx, (mul V2I32:$Rs, V2I32:$Rt)),
         (PS_vmulw_acc DoubleRegs:$Rx, DoubleRegs:$Rs, DoubleRegs:$Rt)>;


// Adds two v4i8: Hexagon does not have an insn for this one, so we
// use the double add v8i8, and use only the low part of the result.
def: Pat<(v4i8 (add (v4i8 IntRegs:$Rs), (v4i8 IntRegs:$Rt))),
         (LoReg (A2_vaddub (ToZext64 $Rs), (ToZext64 $Rt)))>;

// Subtract two v4i8: Hexagon does not have an insn for this one, so we
// use the double sub v8i8, and use only the low part of the result.
def: Pat<(v4i8 (sub (v4i8 IntRegs:$Rs), (v4i8 IntRegs:$Rt))),
         (LoReg (A2_vsubub (ToZext64 $Rs), (ToZext64 $Rt)))>;

//
// No 32 bit vector mux.
//
def: Pat<(v4i8 (select I1:$Pu, V4I8:$Rs, V4I8:$Rt)),
         (LoReg (C2_vmux I1:$Pu, (ToZext64 $Rs), (ToZext64 $Rt)))>;
def: Pat<(v2i16 (select I1:$Pu, V2I16:$Rs, V2I16:$Rt)),
         (LoReg (C2_vmux I1:$Pu, (ToZext64 $Rs), (ToZext64 $Rt)))>;

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
         (A2_vcmpbeq (ToZext64 $Rs), (ToZext64 $Rt))>;
def: Pat<(i1 (setgt V4I8:$Rs, V4I8:$Rt)),
         (A4_vcmpbgt (ToZext64 $Rs), (ToZext64 $Rt))>;
def: Pat<(i1 (setugt V4I8:$Rs, V4I8:$Rt)),
         (A2_vcmpbgtu (ToZext64 $Rs), (ToZext64 $Rt))>;

def: Pat<(i1 (seteq V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmpheq (ToZext64 $Rs), (ToZext64 $Rt))>;
def: Pat<(i1 (setgt V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmphgt (ToZext64 $Rs), (ToZext64 $Rt))>;
def: Pat<(i1 (setugt V2I16:$Rs, V2I16:$Rt)),
         (A2_vcmphgtu (ToZext64 $Rs), (ToZext64 $Rt))>;


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
         (LoReg (S2_vtrunewh (A2_combineii 0, 0),
                             (vmpyh V2I16:$Rs, V2I16:$Rt)))>;

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


// Read cycle counter.
//
def SDTInt64Leaf: SDTypeProfile<1, 0, [SDTCisVT<0, i64>]>;
def HexagonREADCYCLE: SDNode<"HexagonISD::READCYCLE", SDTInt64Leaf,
  [SDNPHasChain]>;

def: Pat<(HexagonREADCYCLE), (A4_tfrcpp UPCYCLE)>;
