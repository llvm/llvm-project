//=- HexagonInstrInfoV5.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V5 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

def f32ImmPred : PatLeaf<(f32 fpimm:$F)>;
def f64ImmPred : PatLeaf<(f64 fpimm:$F)>;

def ftoi : SDNodeXForm<fpimm, [{
  APInt I = N->getValueAPF().bitcastToAPInt();
  return CurDAG->getTargetConstant(I.getZExtValue(), SDLoc(N),
                                   MVT::getIntegerVT(I.getBitWidth()));
}]>;

//===----------------------------------------------------------------------===//
// XTYPE/MPY
//===----------------------------------------------------------------------===//

  //Rdd[+]=vrmpybsu(Rss,Rtt)
let Predicates = [HasV5T] in {
  def M5_vrmpybsu: T_XTYPE_Vect<"vrmpybsu", 0b110, 0b001, 0>;
  def M5_vrmacbsu: T_XTYPE_Vect_acc<"vrmpybsu", 0b110, 0b001, 0>;

  //Rdd[+]=vrmpybu(Rss,Rtt)
  def M5_vrmpybuu: T_XTYPE_Vect<"vrmpybu", 0b100, 0b001, 0>;
  def M5_vrmacbuu: T_XTYPE_Vect_acc<"vrmpybu", 0b100, 0b001, 0>;

  def M5_vdmpybsu: T_M2_vmpy<"vdmpybsu", 0b101, 0b001, 0, 0, 1>;
  def M5_vdmacbsu: T_M2_vmpy_acc_sat <"vdmpybsu", 0b001, 0b001, 0, 0>;
}

// Vector multiply bytes
// Rdd=vmpyb[s]u(Rs,Rt)
let Predicates = [HasV5T] in {
  def M5_vmpybsu: T_XTYPE_mpy64 <"vmpybsu", 0b010, 0b001, 0, 0, 0>;
  def M5_vmpybuu: T_XTYPE_mpy64 <"vmpybu",  0b100, 0b001, 0, 0, 0>;

  // Rxx+=vmpyb[s]u(Rs,Rt)
  def M5_vmacbsu: T_XTYPE_mpy64_acc <"vmpybsu", "+", 0b110, 0b001, 0, 0, 0>;
  def M5_vmacbuu: T_XTYPE_mpy64_acc <"vmpybu", "+", 0b100, 0b001, 0, 0, 0>;

  // Rd=vaddhub(Rss,Rtt):sat
  let hasNewValue = 1, opNewValue = 0 in
    def A5_vaddhubs: T_S3op_1 <"vaddhub", IntRegs, 0b01, 0b001, 0, 1>;
}

def S2_asr_i_p_rnd : S_2OpInstImm<"asr", 0b110, 0b111, u6Imm,
      [(set I64:$dst,
            (sra (i64 (add (i64 (sra I64:$src1, u6ImmPred:$src2)), 1)),
                 (i32 1)))], 1>,
      Requires<[HasV5T]> {
  bits<6> src2;
  let Inst{13-8} = src2;
}

let isAsmParserOnly = 1 in
def S2_asr_i_p_rnd_goodsyntax
  : MInst<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1, u6Imm:$src2),
    "$dst = asrrnd($src1, #$src2)">;

def C4_fastcorner9 : T_LOGICAL_2OP<"fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{13,7,4} = 0b111;
}

def C4_fastcorner9_not : T_LOGICAL_2OP<"!fastcorner9", 0b000, 0, 0>,
  Requires<[HasV5T]> {
  let Inst{20,13,7,4} = 0b1111;
}

def SDTHexagonI32I64: SDTypeProfile<1, 1, [SDTCisVT<0, i32>,
                                           SDTCisVT<1, i64>]>;

def HexagonPOPCOUNT: SDNode<"HexagonISD::POPCOUNT", SDTHexagonI32I64>;

let hasNewValue = 1, validSubTargets = HasV5SubT in
def S5_popcountp : ALU64_rr<(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
  "$Rd = popcount($Rss)",
  [(set I32:$Rd, (HexagonPOPCOUNT I64:$Rss))], "", S_2op_tc_2_SLOT23>,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rss;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1000011;
    let Inst{7-5} = 0b011;
    let Inst{4-0} = Rd;
    let Inst{20-16} = Rss;
  }

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
  def: Loada_pat<load, f32, u32ImmPred, L4_loadri_abs>;
  def: Loada_pat<load, f32, addrga, L4_loadri_abs>;
  def: Loada_pat<load, f64, addrga, L4_loadrd_abs>;
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
  def: Storea_pat<store, F32, addrga, S2_storeriabs>;
  def: Storea_pat<store, F64, addrga, S2_storerdabs>;
}

let AddedComplexity = 100 in {
  def: Storea_pat<store, F32, addrgp, S2_storerigp>;
  def: Storea_pat<store, F64, addrgp, S2_storerdgp>;
}

defm: Storex_pat<store, F32, s30_2ImmPred, S2_storeri_io>;
defm: Storex_pat<store, F64, s29_3ImmPred, S2_storerd_io>;
def: Storex_simple_pat<store, F32, S2_storeri_io>;
def: Storex_simple_pat<store, F64, S2_storerd_io>;

let isFP = 1, hasNewValue = 1, opNewValue = 0 in
class T_MInstFloat <string mnemonic, bits<3> MajOp, bits<3> MinOp>
  : MInst<(outs IntRegs:$Rd),
          (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd = "#mnemonic#"($Rs, $Rt)", [],
  "" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-24} = 0b1011;
    let Inst{23-21} = MajOp;
    let Inst{20-16} = Rs;
    let Inst{13} = 0b0;
    let Inst{12-8} = Rt;
    let Inst{7-5} = MinOp;
    let Inst{4-0} = Rd;
  }

let isCommutable = 1 in {
  def F2_sfadd : T_MInstFloat < "sfadd", 0b000, 0b000>;
  def F2_sfmpy : T_MInstFloat < "sfmpy", 0b010, 0b000>;
}

def F2_sfsub : T_MInstFloat < "sfsub", 0b000, 0b001>;

def: Pat<(fadd F32:$src1, F32:$src2),
         (F2_sfadd F32:$src1, F32:$src2)>;

def: Pat<(fsub F32:$src1, F32:$src2),
         (F2_sfsub F32:$src1, F32:$src2)>;

def: Pat<(fmul F32:$src1, F32:$src2),
         (F2_sfmpy F32:$src1, F32:$src2)>;

let Itinerary = M_tc_3x_SLOT23 in {
  def F2_sfmax : T_MInstFloat < "sfmax", 0b100, 0b000>;
  def F2_sfmin : T_MInstFloat < "sfmin", 0b100, 0b001>;
}

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

let Itinerary = M_tc_3or4x_SLOT23 in {
def F2_sffixupn : T_MInstFloat < "sffixupn", 0b110, 0b000>;
def F2_sffixupd : T_MInstFloat < "sffixupd", 0b110, 0b001>;
}

// F2_sfrecipa: Reciprocal approximation for division.
let Uses = [USR], isPredicateLate = 1, isFP = 1,
    hasSideEffects = 0, hasNewValue = 1, Itinerary = M_tc_3or4x_SLOT23 in
def F2_sfrecipa: MInst <
  (outs IntRegs:$Rd, PredRegs:$Pe),
  (ins IntRegs:$Rs, IntRegs:$Rt),
  "$Rd, $Pe = sfrecipa($Rs, $Rt)">,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<2> Pe;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;
    let Inst{27-21} = 0b1011111;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pe;
    let Inst{4-0}   = Rd;
  }

// F2_dfcmpeq: Floating point compare for equal.
let Uses = [USR], isCompare = 1, isFP = 1 in
class T_fcmp <string mnemonic, RegisterClass RC, bits<3> MinOp,
              list<dag> pattern = [] >
  : ALU64Inst <(outs PredRegs:$dst), (ins RC:$src1, RC:$src2),
  "$dst = "#mnemonic#"($src1, $src2)", pattern,
  "" , ALU64_tc_2early_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<2> dst;
    bits<5> src1;
    bits<5> src2;

    let IClass = 0b1101;

    let Inst{27-21} = 0b0010111;
    let Inst{20-16} = src1;
    let Inst{12-8}  = src2;
    let Inst{7-5}   = MinOp;
    let Inst{1-0}   = dst;
  }

class T_fcmp64 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, DoubleRegs, MinOp,
  [(set  I1:$dst, (OpNode F64:$src1, F64:$src2))]> {
  let IClass = 0b1101;
  let Inst{27-21} = 0b0010111;
}

class T_fcmp32 <string mnemonic, PatFrag OpNode, bits<3> MinOp>
  : T_fcmp <mnemonic, IntRegs, MinOp,
  [(set  I1:$dst, (OpNode F32:$src1, F32:$src2))]> {
  let IClass = 0b1100;
  let Inst{27-21} = 0b0111111;
}

def F2_dfcmpeq : T_fcmp64<"dfcmp.eq", setoeq, 0b000>;
def F2_dfcmpgt : T_fcmp64<"dfcmp.gt", setogt, 0b001>;
def F2_dfcmpge : T_fcmp64<"dfcmp.ge", setoge, 0b010>;
def F2_dfcmpuo : T_fcmp64<"dfcmp.uo", setuo,  0b011>;

def F2_sfcmpge : T_fcmp32<"sfcmp.ge", setoge, 0b000>;
def F2_sfcmpuo : T_fcmp32<"sfcmp.uo", setuo,  0b001>;
def F2_sfcmpeq : T_fcmp32<"sfcmp.eq", setoeq, 0b011>;
def F2_sfcmpgt : T_fcmp32<"sfcmp.gt", setogt, 0b100>;

//===----------------------------------------------------------------------===//
// Multiclass to define 'Def Pats' for ordered gt, ge, eq operations.
//===----------------------------------------------------------------------===//

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

// F2 convert template classes:
let Uses = [USR], isFP = 1 in
class F2_RDD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                         SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                         string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss),
   "$Rdd = "#mnemonic#"($Rss)"#chop,
   [(set RCOut:$Rdd, (Op RCIn:$Rss))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0000111;
     let Inst{20-16} = Rss;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let Uses = [USR], isFP = 1 in
class F2_RDD_RS_CONVERT<string mnemonic, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs DoubleRegs:$Rdd), (ins IntRegs:$Rs),
   "$Rdd = "#mnemonic#"($Rs)"#chop,
   [(set RCOut:$Rdd, (Op RCIn:$Rs))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rdd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-21} = 0b0100100;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rdd;
  }

let Uses = [USR], isFP = 1, hasNewValue = 1 in
class F2_RD_RSS_CONVERT<string mnemonic, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss),
   "$Rd = "#mnemonic#"($Rss)"#chop,
   [(set RCOut:$Rd, (Op RCIn:$Rss))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rss;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1000;
     let Inst{23-21} = MinOp;
     let Inst{20-16} = Rss;
     let Inst{7-5} = 0b001;
     let Inst{4-0} = Rd;
  }

let Uses = [USR], isFP = 1, hasNewValue = 1 in
class F2_RD_RS_CONVERT<string mnemonic, bits<3> MajOp, bits<3> MinOp,
                        SDNode Op, PatLeaf RCOut, PatLeaf RCIn,
                        string chop ="">
  : SInst <(outs IntRegs:$Rd), (ins IntRegs:$Rs),
   "$Rd = "#mnemonic#"($Rs)"#chop,
   [(set RCOut:$Rd, (Op RCIn:$Rs))], "",
   S_2op_tc_3or4x_SLOT23> {
     bits<5> Rd;
     bits<5> Rs;

     let IClass = 0b1000;

     let Inst{27-24} = 0b1011;
     let Inst{23-21} = MajOp;
     let Inst{20-16} = Rs;
     let Inst{7-5} = MinOp;
     let Inst{4-0} = Rd;
  }

// Convert single precision to double precision and vice-versa.
def F2_conv_sf2df : F2_RDD_RS_CONVERT <"convert_sf2df", 0b000,
                                       fpextend, F64, F32>;

def F2_conv_df2sf : F2_RD_RSS_CONVERT <"convert_df2sf", 0b000,
                                       fpround, F32, F64>;

// Convert Integer to Floating Point.
def F2_conv_d2sf : F2_RD_RSS_CONVERT <"convert_d2sf", 0b010,
                                       sint_to_fp, F32, I64>;
def F2_conv_ud2sf : F2_RD_RSS_CONVERT <"convert_ud2sf", 0b001,
                                       uint_to_fp, F32, I64>;
def F2_conv_uw2sf : F2_RD_RS_CONVERT <"convert_uw2sf", 0b001, 0b000,
                                       uint_to_fp, F32, I32>;
def F2_conv_w2sf : F2_RD_RS_CONVERT <"convert_w2sf", 0b010, 0b000,
                                       sint_to_fp, F32, I32>;
def F2_conv_d2df : F2_RDD_RSS_CONVERT <"convert_d2df", 0b011,
                                       sint_to_fp, F64, I64>;
def F2_conv_ud2df : F2_RDD_RSS_CONVERT <"convert_ud2df", 0b010,
                                        uint_to_fp, F64, I64>;
def F2_conv_uw2df : F2_RDD_RS_CONVERT <"convert_uw2df", 0b001,
                                       uint_to_fp, F64, I32>;
def F2_conv_w2df : F2_RDD_RS_CONVERT <"convert_w2df", 0b010,
                                       sint_to_fp, F64, I32>;

// Convert Floating Point to Integer - default.
def F2_conv_df2uw_chop : F2_RD_RSS_CONVERT <"convert_df2uw", 0b101,
                                            fp_to_uint, I32, F64, ":chop">;
def F2_conv_df2w_chop : F2_RD_RSS_CONVERT <"convert_df2w", 0b111,
                                            fp_to_sint, I32, F64, ":chop">;
def F2_conv_sf2uw_chop : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b001,
                                       fp_to_uint, I32, F32, ":chop">;
def F2_conv_sf2w_chop : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b001,
                                       fp_to_sint, I32, F32, ":chop">;
def F2_conv_df2d_chop : F2_RDD_RSS_CONVERT <"convert_df2d", 0b110,
                                            fp_to_sint, I64, F64, ":chop">;
def F2_conv_df2ud_chop : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b111,
                                             fp_to_uint, I64, F64, ":chop">;
def F2_conv_sf2d_chop : F2_RDD_RS_CONVERT <"convert_sf2d", 0b110,
                                       fp_to_sint, I64, F32, ":chop">;
def F2_conv_sf2ud_chop : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b101,
                                            fp_to_uint, I64, F32, ":chop">;

// Convert Floating Point to Integer: non-chopped.
let AddedComplexity = 20, Predicates = [HasV5T, IEEERndNearV5T] in {
  def F2_conv_df2d : F2_RDD_RSS_CONVERT <"convert_df2d", 0b000,
                                         fp_to_sint, I64, F64>;
  def F2_conv_df2ud : F2_RDD_RSS_CONVERT <"convert_df2ud", 0b001,
                                          fp_to_uint, I64, F64>;
  def F2_conv_sf2ud : F2_RDD_RS_CONVERT <"convert_sf2ud", 0b011,
                                         fp_to_uint, I64, F32>;
  def F2_conv_sf2d : F2_RDD_RS_CONVERT <"convert_sf2d", 0b100,
                                         fp_to_sint, I64, F32>;
  def F2_conv_df2uw : F2_RD_RSS_CONVERT <"convert_df2uw", 0b011,
                                         fp_to_uint, I32, F64>;
  def F2_conv_df2w : F2_RD_RSS_CONVERT <"convert_df2w", 0b100,
                                         fp_to_sint, I32, F64>;
  def F2_conv_sf2uw : F2_RD_RS_CONVERT <"convert_sf2uw", 0b011, 0b000,
                                         fp_to_uint, I32, F32>;
  def F2_conv_sf2w : F2_RD_RS_CONVERT <"convert_sf2w", 0b100, 0b000,
                                         fp_to_sint, I32, F32>;
}

// Fix up radicand.
let Uses = [USR], isFP = 1, hasNewValue = 1 in
def F2_sffixupr: SInst<(outs IntRegs:$Rd), (ins IntRegs:$Rs),
  "$Rd = sffixupr($Rs)",
  [], "" , S_2op_tc_3or4x_SLOT23>, Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rs;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1011101;
    let Inst{20-16} = Rs;
    let Inst{7-5}   = 0b000;
    let Inst{4-0}   = Rd;
  }

// Bitcast is different than [fp|sint|uint]_to_[sint|uint|fp].
let Predicates = [HasV5T] in {
  def: Pat <(i32 (bitconvert F32:$src)), (I32:$src)>;
  def: Pat <(f32 (bitconvert I32:$src)), (F32:$src)>;
  def: Pat <(i64 (bitconvert F64:$src)), (I64:$src)>;
  def: Pat <(f64 (bitconvert I64:$src)), (F64:$src)>;
}

// F2_sffma: Floating-point fused multiply add.
let Uses = [USR], isFP = 1, hasNewValue = 1 in
class T_sfmpy_acc <bit isSub, bit isLib>
  : MInst<(outs IntRegs:$Rx),
          (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt),
  "$Rx "#!if(isSub, "-=","+=")#" sfmpy($Rs, $Rt)"#!if(isLib, ":lib",""),
  [], "$dst2 = $Rx" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111000;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6}     = isLib;
    let Inst{5}     = isSub;
    let Inst{4-0}   = Rx;
  }

def F2_sffma: T_sfmpy_acc <0, 0>;
def F2_sffms: T_sfmpy_acc <1, 0>;
def F2_sffma_lib: T_sfmpy_acc <0, 1>;
def F2_sffms_lib: T_sfmpy_acc <1, 1>;

def : Pat <(fma F32:$src2, F32:$src3, F32:$src1),
           (F2_sffma F32:$src1, F32:$src2, F32:$src3)>;

def : Pat <(fma (fneg F32:$src2), F32:$src3, F32:$src1),
           (F2_sffms F32:$src1, F32:$src2, F32:$src3)>;

def : Pat <(fma F32:$src2, (fneg F32:$src3), F32:$src1),
           (F2_sffms F32:$src1, F32:$src2, F32:$src3)>;

// Floating-point fused multiply add w/ additional scaling (2**pu).
let Uses = [USR], isFP = 1, hasNewValue = 1 in
def F2_sffma_sc: MInst <
  (outs IntRegs:$Rx),
  (ins IntRegs:$dst2, IntRegs:$Rs, IntRegs:$Rt, PredRegs:$Pu),
  "$Rx += sfmpy($Rs, $Rt, $Pu):scale" ,
  [], "$dst2 = $Rx" , M_tc_3or4x_SLOT23 > ,
  Requires<[HasV5T]> {
    bits<5> Rx;
    bits<5> Rs;
    bits<5> Rt;
    bits<2> Pu;

    let IClass = 0b1110;

    let Inst{27-21} = 0b1111011;
    let Inst{20-16} = Rs;
    let Inst{13}    = 0b0;
    let Inst{12-8}  = Rt;
    let Inst{7}     = 0b1;
    let Inst{6-5}   = Pu;
    let Inst{4-0}   = Rx;
  }

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

//===----------------------------------------------------------------------===//
// :natural forms of vasrh and vasrhub insns
//===----------------------------------------------------------------------===//
// S5_asrhub_rnd_sat: Vector arithmetic shift right by immediate with round,
// saturate, and pack.
let Defs = [USR_OVF], hasSideEffects = 0, hasNewValue = 1, opNewValue = 0 in
class T_ASRHUB<bit isSat>
  : SInst <(outs IntRegs:$Rd),
  (ins DoubleRegs:$Rss, u4Imm:$u4),
  "$Rd = vasrhub($Rss, #$u4):"#!if(isSat, "sat", "raw"),
  [], "", S_2op_tc_2_SLOT23>,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<5> Rss;
    bits<4> u4;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1000011;
    let Inst{20-16} = Rss;
    let Inst{13-12} = 0b00;
    let Inst{11-8} = u4;
    let Inst{7-6} = 0b10;
    let Inst{5} = isSat;
    let Inst{4-0} = Rd;
  }

def S5_asrhub_rnd_sat : T_ASRHUB <0>;
def S5_asrhub_sat : T_ASRHUB <1>;

let isAsmParserOnly = 1 in
def S5_asrhub_rnd_sat_goodsyntax
  : SInst <(outs IntRegs:$Rd), (ins DoubleRegs:$Rss, u4Imm:$u4),
  "$Rd = vasrhub($Rss, #$u4):rnd:sat">, Requires<[HasV5T]>;

// S5_vasrhrnd: Vector arithmetic shift right by immediate with round.
let hasSideEffects = 0 in
def S5_vasrhrnd : SInst <(outs DoubleRegs:$Rdd),
                         (ins DoubleRegs:$Rss, u4Imm:$u4),
  "$Rdd = vasrh($Rss, #$u4):raw">,
  Requires<[HasV5T]> {
    bits<5> Rdd;
    bits<5> Rss;
    bits<4> u4;

    let IClass = 0b1000;

    let Inst{27-21} = 0b0000001;
    let Inst{20-16} = Rss;
    let Inst{13-12} = 0b00;
    let Inst{11-8}  = u4;
    let Inst{7-5}   = 0b000;
    let Inst{4-0}   = Rdd;
  }

let isAsmParserOnly = 1 in
def S5_vasrhrnd_goodsyntax
  : SInst <(outs DoubleRegs:$Rdd), (ins DoubleRegs:$Rss, u4Imm:$u4),
  "$Rdd = vasrh($Rss,#$u4):rnd">, Requires<[HasV5T]>;

// Floating point reciprocal square root approximation
let Uses = [USR], isPredicateLate = 1, isFP = 1,
    hasSideEffects = 0, hasNewValue = 1, opNewValue = 0,
    validSubTargets = HasV5SubT in
def F2_sfinvsqrta: SInst <
  (outs IntRegs:$Rd, PredRegs:$Pe),
  (ins IntRegs:$Rs),
  "$Rd, $Pe = sfinvsqrta($Rs)" > ,
  Requires<[HasV5T]> {
    bits<5> Rd;
    bits<2> Pe;
    bits<5> Rs;

    let IClass = 0b1000;

    let Inst{27-21} = 0b1011111;
    let Inst{20-16} = Rs;
    let Inst{7} = 0b0;
    let Inst{6-5} = Pe;
    let Inst{4-0} = Rd;
  }

// Complex multiply 32x16
let Defs = [USR_OVF], Itinerary = S_3op_tc_3x_SLOT23 in {
  def M4_cmpyi_whc : T_S3op_8<"cmpyiwh", 0b101, 1, 1, 1, 1>;
  def M4_cmpyr_whc : T_S3op_8<"cmpyrwh", 0b111, 1, 1, 1, 1>;
}

// Classify floating-point value
let Uses = [USR], isFP = 1 in
def F2_sfclass : T_TEST_BIT_IMM<"sfclass", 0b111>;

let Uses = [USR], isFP = 1 in
def F2_dfclass: ALU64Inst<(outs PredRegs:$Pd), (ins DoubleRegs:$Rss, u5Imm:$u5),
  "$Pd = dfclass($Rss, #$u5)",
  [], "" , ALU64_tc_2early_SLOT23 > , Requires<[HasV5T]> {
    bits<2> Pd;
    bits<5> Rss;
    bits<5> u5;

    let IClass = 0b1101;
    let Inst{27-21} = 0b1100100;
    let Inst{20-16} = Rss;
    let Inst{12-10} = 0b000;
    let Inst{9-5}   = u5;
    let Inst{4-3}   = 0b10;
    let Inst{1-0}   = Pd;
  }

// Instructions to create floating point constant
class T_fimm <string mnemonic, RegisterClass RC, bits<4> RegType, bit isNeg>
  : ALU64Inst<(outs RC:$dst), (ins u10Imm:$src),
  "$dst = "#mnemonic#"(#$src)"#!if(isNeg, ":neg", ":pos"),
  [], "", ALU64_tc_2_SLOT23>, Requires<[HasV5T]> {
    bits<5> dst;
    bits<10> src;

    let IClass = 0b1101;
    let Inst{27-24} = RegType;
    let Inst{23}    = 0b0;
    let Inst{22}    = isNeg;
    let Inst{21}    = src{9};
    let Inst{13-5}  = src{8-0};
    let Inst{4-0}   = dst;
  }

let hasNewValue = 1, opNewValue = 0 in {
  def F2_sfimm_p : T_fimm <"sfmake", IntRegs, 0b0110, 0>;
  def F2_sfimm_n : T_fimm <"sfmake", IntRegs, 0b0110, 1>;
}

def F2_dfimm_p : T_fimm <"dfmake", DoubleRegs, 0b1001, 0>;
def F2_dfimm_n : T_fimm <"dfmake", DoubleRegs, 0b1001, 1>;

def : Pat <(fabs (f32 IntRegs:$src1)),
           (S2_clrbit_i (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;

def : Pat <(fneg (f32 IntRegs:$src1)),
           (S2_togglebit_i (f32 IntRegs:$src1), 31)>,
          Requires<[HasV5T]>;
