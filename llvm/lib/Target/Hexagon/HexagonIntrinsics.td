//===-- HexagonIntrinsics.td - Instruction intrinsics ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This is populated based on the following specs:
// Hexagon V2 Architecture
// Application-Level Specification
// 80-V9418-8 Rev. B
// March 4, 2008
//===----------------------------------------------------------------------===//

class T_I_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$Is),
         (MI imm:$Is)>;

class T_R_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs),
         (MI I32:$Rs)>;

class T_P_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs),
         (MI I64:$Rs)>;

class T_II_pat <InstHexagon MI, Intrinsic IntID, PatFrag Imm1, PatFrag Imm2>
  : Pat<(IntID Imm1:$Is, Imm2:$It),
        (MI Imm1:$Is, Imm2:$It)>;

class T_RI_pat <InstHexagon MI, Intrinsic IntID,
                PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID I32:$Rs, ImmPred:$It),
        (MI I32:$Rs, ImmPred:$It)>;

class T_IR_pat <InstHexagon MI, Intrinsic IntID,
                PatFrag ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID ImmPred:$Is, I32:$Rt),
        (MI ImmPred:$Is, I32:$Rt)>;

class T_PI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID I64:$Rs, imm:$It),
        (MI I64:$Rs, imm:$It)>;

class T_RP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID I32:$Rs, I64:$Rt),
        (MI I32:$Rs, I64:$Rt)>;

class T_RR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt),
         (MI I32:$Rs, I32:$Rt)>;

class T_PP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt),
         (MI I64:$Rs, I64:$Rt)>;

class T_QQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt),
         (MI (C2_tfrrp I32:$Rs), (C2_tfrrp I32:$Rt))>;

class T_QII_pat <InstHexagon MI, Intrinsic IntID, PatFrag Imm1, PatFrag Imm2>
  : Pat <(IntID I32:$Rp, Imm1:$Is, Imm2:$It),
         (MI (C2_tfrrp I32:$Rp), Imm1:$Is, Imm2:$It)>;

class T_QRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rp, I32:$Rs, I32:$Rt),
         (MI (C2_tfrrp I32:$Rp), I32:$Rs, I32:$Rt)>;

class T_QRI_pat <InstHexagon MI, Intrinsic IntID, PatFrag ImmPred>
  : Pat <(IntID I32:$Rp, I32:$Rs, ImmPred:$Is),
         (MI (C2_tfrrp I32:$Rp), I32:$Rs, ImmPred:$Is)>;

class T_QIR_pat <InstHexagon MI, Intrinsic IntID, PatFrag ImmPred>
  : Pat <(IntID I32:$Rp, ImmPred:$Is, I32:$Rs),
         (MI (C2_tfrrp I32:$Rp), ImmPred:$Is, I32:$Rs)>;

class T_QPP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rp, I64:$Rs, I64:$Rt),
         (MI (C2_tfrrp I32:$Rp), I64:$Rs, I64:$Rt)>;

class T_RRI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt, imm:$Iu),
         (MI I32:$Rs, I32:$Rt, imm:$Iu)>;

class T_RII_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, imm:$It, imm:$Iu),
         (MI I32:$Rs, imm:$It, imm:$Iu)>;

class T_IRI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$It, I32:$Rs, imm:$Iu),
         (MI imm:$It, I32:$Rs, imm:$Iu)>;

class T_IRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID imm:$Is, I32:$Rs, I32:$Rt),
         (MI imm:$Is, I32:$Rs, I32:$Rt)>;

class T_RIR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, imm:$Is, I32:$Rt),
         (MI I32:$Rs, imm:$Is, I32:$Rt)>;

class T_RRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt, I32:$Ru),
         (MI I32:$Rs, I32:$Rt, I32:$Ru)>;

class T_PPI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, imm:$Iu),
         (MI I64:$Rs, I64:$Rt, imm:$Iu)>;

class T_PII_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, imm:$It, imm:$Iu),
         (MI I64:$Rs, imm:$It, imm:$Iu)>;

class T_PPP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, I64:$Ru),
         (MI I64:$Rs, I64:$Rt, I64:$Ru)>;

class T_PPR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, I32:$Ru),
         (MI I64:$Rs, I64:$Rt, I32:$Ru)>;

class T_PRR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I32:$Rt, I32:$Ru),
         (MI I64:$Rs, I32:$Rt, I32:$Ru)>;

class T_PPQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt, I32:$Rp),
         (MI I64:$Rs, I64:$Rt, (C2_tfrrp I32:$Rp))>;

class T_PR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I32:$Rt),
         (MI I64:$Rs, I32:$Rt)>;

class T_D_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID (F64:$Rs)),
        (MI (F64:$Rs))>;

class T_DI_pat <InstHexagon MI, Intrinsic IntID,
                PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID F64:$Rs, ImmPred:$It),
        (MI F64:$Rs, ImmPred:$It)>;

class T_F_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs),
        (MI F32:$Rs)>;

class T_FI_pat <InstHexagon MI, Intrinsic IntID,
                PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID F32:$Rs, ImmPred:$It),
        (MI F32:$Rs, ImmPred:$It)>;

class T_FF_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, F32:$Rt),
        (MI F32:$Rs, F32:$Rt)>;

class T_DD_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F64:$Rs, F64:$Rt),
        (MI F64:$Rs, F64:$Rt)>;

class T_FFF_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, F32:$Rt, F32:$Ru),
        (MI F32:$Rs, F32:$Rt, F32:$Ru)>;

class T_FFFQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID F32:$Rs, F32:$Rt, F32:$Ru, I32:$Rp),
         (MI F32:$Rs, F32:$Rt, F32:$Ru, (C2_tfrrp I32:$Rp))>;

class T_Q_RI_pat <InstHexagon MI, Intrinsic IntID,
                  PatLeaf ImmPred = PatLeaf<(i32 imm)>>
  : Pat<(IntID I32:$Rs, ImmPred:$It),
        (C2_tfrpr (MI I32:$Rs, ImmPred:$It))>;

class T_Q_RR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I32:$Rt),
         (C2_tfrpr (MI I32:$Rs, I32:$Rt))>;

class T_Q_RP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rs, I64:$Rt),
         (C2_tfrpr (MI I32:$Rs, I64:$Rt))>;

class T_Q_PR_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I32:$Rt),
         (C2_tfrpr (MI I64:$Rs, I32:$Rt))>;

class T_Q_PI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID I64:$Rs, imm:$It),
        (C2_tfrpr (MI I64:$Rs, imm:$It))>;

class T_Q_PP_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I64:$Rs, I64:$Rt),
         (C2_tfrpr (MI I64:$Rs, I64:$Rt))>;

class T_Q_Q_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rp),
         (C2_tfrpr (MI (C2_tfrrp I32:$Rp)))>;

class T_Q_QQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rp, I32:$Rq),
         (C2_tfrpr (MI (C2_tfrrp I32:$Rp), (C2_tfrrp I32:$Rq)))>;

class T_Q_FF_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, F32:$Rt),
        (C2_tfrpr (MI F32:$Rs, F32:$Rt))>;

class T_Q_DD_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F64:$Rs, F64:$Rt),
        (C2_tfrpr (MI F64:$Rs, F64:$Rt))>;

class T_Q_FI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F32:$Rs, imm:$It),
        (C2_tfrpr (MI F32:$Rs, imm:$It))>;

class T_Q_DI_pat <InstHexagon MI, Intrinsic IntID>
  : Pat<(IntID F64:$Rs, imm:$It),
        (C2_tfrpr (MI F64:$Rs, imm:$It))>;

class T_Q_QQQ_pat <InstHexagon MI, Intrinsic IntID>
  : Pat <(IntID I32:$Rp, I32:$Rq, I32:$Rs),
         (C2_tfrpr (MI (C2_tfrrp I32:$Rp), (C2_tfrrp I32:$Rq),
                       (C2_tfrrp I32:$Rs)))>;

//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords
//Rd=mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:rnd][:sat]
//===----------------------------------------------------------------------===//

def : T_RR_pat <M2_mpy_ll_s1, int_hexagon_M2_mpy_ll_s1>;
def : T_RR_pat <M2_mpy_ll_s0, int_hexagon_M2_mpy_ll_s0>;
def : T_RR_pat <M2_mpy_lh_s1, int_hexagon_M2_mpy_lh_s1>;
def : T_RR_pat <M2_mpy_lh_s0, int_hexagon_M2_mpy_lh_s0>;
def : T_RR_pat <M2_mpy_hl_s1, int_hexagon_M2_mpy_hl_s1>;
def : T_RR_pat <M2_mpy_hl_s0, int_hexagon_M2_mpy_hl_s0>;
def : T_RR_pat <M2_mpy_hh_s1, int_hexagon_M2_mpy_hh_s1>;
def : T_RR_pat <M2_mpy_hh_s0, int_hexagon_M2_mpy_hh_s0>;

def : T_RR_pat <M2_mpyu_ll_s1, int_hexagon_M2_mpyu_ll_s1>;
def : T_RR_pat <M2_mpyu_ll_s0, int_hexagon_M2_mpyu_ll_s0>;
def : T_RR_pat <M2_mpyu_lh_s1, int_hexagon_M2_mpyu_lh_s1>;
def : T_RR_pat <M2_mpyu_lh_s0, int_hexagon_M2_mpyu_lh_s0>;
def : T_RR_pat <M2_mpyu_hl_s1, int_hexagon_M2_mpyu_hl_s1>;
def : T_RR_pat <M2_mpyu_hl_s0, int_hexagon_M2_mpyu_hl_s0>;
def : T_RR_pat <M2_mpyu_hh_s1, int_hexagon_M2_mpyu_hh_s1>;
def : T_RR_pat <M2_mpyu_hh_s0, int_hexagon_M2_mpyu_hh_s0>;

def : T_RR_pat <M2_mpy_sat_ll_s1, int_hexagon_M2_mpy_sat_ll_s1>;
def : T_RR_pat <M2_mpy_sat_ll_s0, int_hexagon_M2_mpy_sat_ll_s0>;
def : T_RR_pat <M2_mpy_sat_lh_s1, int_hexagon_M2_mpy_sat_lh_s1>;
def : T_RR_pat <M2_mpy_sat_lh_s0, int_hexagon_M2_mpy_sat_lh_s0>;
def : T_RR_pat <M2_mpy_sat_hl_s1, int_hexagon_M2_mpy_sat_hl_s1>;
def : T_RR_pat <M2_mpy_sat_hl_s0, int_hexagon_M2_mpy_sat_hl_s0>;
def : T_RR_pat <M2_mpy_sat_hh_s1, int_hexagon_M2_mpy_sat_hh_s1>;
def : T_RR_pat <M2_mpy_sat_hh_s0, int_hexagon_M2_mpy_sat_hh_s0>;

def : T_RR_pat <M2_mpy_rnd_ll_s1, int_hexagon_M2_mpy_rnd_ll_s1>;
def : T_RR_pat <M2_mpy_rnd_ll_s0, int_hexagon_M2_mpy_rnd_ll_s0>;
def : T_RR_pat <M2_mpy_rnd_lh_s1, int_hexagon_M2_mpy_rnd_lh_s1>;
def : T_RR_pat <M2_mpy_rnd_lh_s0, int_hexagon_M2_mpy_rnd_lh_s0>;
def : T_RR_pat <M2_mpy_rnd_hl_s1, int_hexagon_M2_mpy_rnd_hl_s1>;
def : T_RR_pat <M2_mpy_rnd_hl_s0, int_hexagon_M2_mpy_rnd_hl_s0>;
def : T_RR_pat <M2_mpy_rnd_hh_s1, int_hexagon_M2_mpy_rnd_hh_s1>;
def : T_RR_pat <M2_mpy_rnd_hh_s0, int_hexagon_M2_mpy_rnd_hh_s0>;

def : T_RR_pat <M2_mpy_sat_rnd_ll_s1, int_hexagon_M2_mpy_sat_rnd_ll_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_ll_s0, int_hexagon_M2_mpy_sat_rnd_ll_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_lh_s1, int_hexagon_M2_mpy_sat_rnd_lh_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_lh_s0, int_hexagon_M2_mpy_sat_rnd_lh_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_hl_s1, int_hexagon_M2_mpy_sat_rnd_hl_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_hl_s0, int_hexagon_M2_mpy_sat_rnd_hl_s0>;
def : T_RR_pat <M2_mpy_sat_rnd_hh_s1, int_hexagon_M2_mpy_sat_rnd_hh_s1>;
def : T_RR_pat <M2_mpy_sat_rnd_hh_s0, int_hexagon_M2_mpy_sat_rnd_hh_s0>;


//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the accumulator.
//Rx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

def : T_RRR_pat <M2_mpy_acc_ll_s1, int_hexagon_M2_mpy_acc_ll_s1>;
def : T_RRR_pat <M2_mpy_acc_ll_s0, int_hexagon_M2_mpy_acc_ll_s0>;
def : T_RRR_pat <M2_mpy_acc_lh_s1, int_hexagon_M2_mpy_acc_lh_s1>;
def : T_RRR_pat <M2_mpy_acc_lh_s0, int_hexagon_M2_mpy_acc_lh_s0>;
def : T_RRR_pat <M2_mpy_acc_hl_s1, int_hexagon_M2_mpy_acc_hl_s1>;
def : T_RRR_pat <M2_mpy_acc_hl_s0, int_hexagon_M2_mpy_acc_hl_s0>;
def : T_RRR_pat <M2_mpy_acc_hh_s1, int_hexagon_M2_mpy_acc_hh_s1>;
def : T_RRR_pat <M2_mpy_acc_hh_s0, int_hexagon_M2_mpy_acc_hh_s0>;

def : T_RRR_pat <M2_mpyu_acc_ll_s1, int_hexagon_M2_mpyu_acc_ll_s1>;
def : T_RRR_pat <M2_mpyu_acc_ll_s0, int_hexagon_M2_mpyu_acc_ll_s0>;
def : T_RRR_pat <M2_mpyu_acc_lh_s1, int_hexagon_M2_mpyu_acc_lh_s1>;
def : T_RRR_pat <M2_mpyu_acc_lh_s0, int_hexagon_M2_mpyu_acc_lh_s0>;
def : T_RRR_pat <M2_mpyu_acc_hl_s1, int_hexagon_M2_mpyu_acc_hl_s1>;
def : T_RRR_pat <M2_mpyu_acc_hl_s0, int_hexagon_M2_mpyu_acc_hl_s0>;
def : T_RRR_pat <M2_mpyu_acc_hh_s1, int_hexagon_M2_mpyu_acc_hh_s1>;
def : T_RRR_pat <M2_mpyu_acc_hh_s0, int_hexagon_M2_mpyu_acc_hh_s0>;

def : T_RRR_pat <M2_mpy_nac_ll_s1, int_hexagon_M2_mpy_nac_ll_s1>;
def : T_RRR_pat <M2_mpy_nac_ll_s0, int_hexagon_M2_mpy_nac_ll_s0>;
def : T_RRR_pat <M2_mpy_nac_lh_s1, int_hexagon_M2_mpy_nac_lh_s1>;
def : T_RRR_pat <M2_mpy_nac_lh_s0, int_hexagon_M2_mpy_nac_lh_s0>;
def : T_RRR_pat <M2_mpy_nac_hl_s1, int_hexagon_M2_mpy_nac_hl_s1>;
def : T_RRR_pat <M2_mpy_nac_hl_s0, int_hexagon_M2_mpy_nac_hl_s0>;
def : T_RRR_pat <M2_mpy_nac_hh_s1, int_hexagon_M2_mpy_nac_hh_s1>;
def : T_RRR_pat <M2_mpy_nac_hh_s0, int_hexagon_M2_mpy_nac_hh_s0>;

def : T_RRR_pat <M2_mpyu_nac_ll_s1, int_hexagon_M2_mpyu_nac_ll_s1>;
def : T_RRR_pat <M2_mpyu_nac_ll_s0, int_hexagon_M2_mpyu_nac_ll_s0>;
def : T_RRR_pat <M2_mpyu_nac_lh_s1, int_hexagon_M2_mpyu_nac_lh_s1>;
def : T_RRR_pat <M2_mpyu_nac_lh_s0, int_hexagon_M2_mpyu_nac_lh_s0>;
def : T_RRR_pat <M2_mpyu_nac_hl_s1, int_hexagon_M2_mpyu_nac_hl_s1>;
def : T_RRR_pat <M2_mpyu_nac_hl_s0, int_hexagon_M2_mpyu_nac_hl_s0>;
def : T_RRR_pat <M2_mpyu_nac_hh_s1, int_hexagon_M2_mpyu_nac_hh_s1>;
def : T_RRR_pat <M2_mpyu_nac_hh_s0, int_hexagon_M2_mpyu_nac_hh_s0>;

def : T_RRR_pat <M2_mpy_acc_sat_ll_s1, int_hexagon_M2_mpy_acc_sat_ll_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_ll_s0, int_hexagon_M2_mpy_acc_sat_ll_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_lh_s1, int_hexagon_M2_mpy_acc_sat_lh_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_lh_s0, int_hexagon_M2_mpy_acc_sat_lh_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_hl_s1, int_hexagon_M2_mpy_acc_sat_hl_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_hl_s0, int_hexagon_M2_mpy_acc_sat_hl_s0>;
def : T_RRR_pat <M2_mpy_acc_sat_hh_s1, int_hexagon_M2_mpy_acc_sat_hh_s1>;
def : T_RRR_pat <M2_mpy_acc_sat_hh_s0, int_hexagon_M2_mpy_acc_sat_hh_s0>;

def : T_RRR_pat <M2_mpy_nac_sat_ll_s1, int_hexagon_M2_mpy_nac_sat_ll_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_ll_s0, int_hexagon_M2_mpy_nac_sat_ll_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_lh_s1, int_hexagon_M2_mpy_nac_sat_lh_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_lh_s0, int_hexagon_M2_mpy_nac_sat_lh_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_hl_s1, int_hexagon_M2_mpy_nac_sat_hl_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_hl_s0, int_hexagon_M2_mpy_nac_sat_hl_s0>;
def : T_RRR_pat <M2_mpy_nac_sat_hh_s1, int_hexagon_M2_mpy_nac_sat_hh_s1>;
def : T_RRR_pat <M2_mpy_nac_sat_hh_s0, int_hexagon_M2_mpy_nac_sat_hh_s0>;


//===----------------------------------------------------------------------===//
// Multiply signed/unsigned halfwords with and without saturation and rounding
// into a 64-bits destination register.
//===----------------------------------------------------------------------===//

def : T_RR_pat <M2_mpyd_hh_s0, int_hexagon_M2_mpyd_hh_s0>;
def : T_RR_pat <M2_mpyd_hl_s0, int_hexagon_M2_mpyd_hl_s0>;
def : T_RR_pat <M2_mpyd_lh_s0, int_hexagon_M2_mpyd_lh_s0>;
def : T_RR_pat <M2_mpyd_ll_s0, int_hexagon_M2_mpyd_ll_s0>;
def : T_RR_pat <M2_mpyd_hh_s1, int_hexagon_M2_mpyd_hh_s1>;
def : T_RR_pat <M2_mpyd_hl_s1, int_hexagon_M2_mpyd_hl_s1>;
def : T_RR_pat <M2_mpyd_lh_s1, int_hexagon_M2_mpyd_lh_s1>;
def : T_RR_pat <M2_mpyd_ll_s1, int_hexagon_M2_mpyd_ll_s1>;

def : T_RR_pat <M2_mpyd_rnd_hh_s0, int_hexagon_M2_mpyd_rnd_hh_s0>;
def : T_RR_pat <M2_mpyd_rnd_hl_s0, int_hexagon_M2_mpyd_rnd_hl_s0>;
def : T_RR_pat <M2_mpyd_rnd_lh_s0, int_hexagon_M2_mpyd_rnd_lh_s0>;
def : T_RR_pat <M2_mpyd_rnd_ll_s0, int_hexagon_M2_mpyd_rnd_ll_s0>;
def : T_RR_pat <M2_mpyd_rnd_hh_s1, int_hexagon_M2_mpyd_rnd_hh_s1>;
def : T_RR_pat <M2_mpyd_rnd_hl_s1, int_hexagon_M2_mpyd_rnd_hl_s1>;
def : T_RR_pat <M2_mpyd_rnd_lh_s1, int_hexagon_M2_mpyd_rnd_lh_s1>;
def : T_RR_pat <M2_mpyd_rnd_ll_s1, int_hexagon_M2_mpyd_rnd_ll_s1>;

def : T_RR_pat <M2_mpyud_hh_s0, int_hexagon_M2_mpyud_hh_s0>;
def : T_RR_pat <M2_mpyud_hl_s0, int_hexagon_M2_mpyud_hl_s0>;
def : T_RR_pat <M2_mpyud_lh_s0, int_hexagon_M2_mpyud_lh_s0>;
def : T_RR_pat <M2_mpyud_ll_s0, int_hexagon_M2_mpyud_ll_s0>;
def : T_RR_pat <M2_mpyud_hh_s1, int_hexagon_M2_mpyud_hh_s1>;
def : T_RR_pat <M2_mpyud_hl_s1, int_hexagon_M2_mpyud_hl_s1>;
def : T_RR_pat <M2_mpyud_lh_s1, int_hexagon_M2_mpyud_lh_s1>;
def : T_RR_pat <M2_mpyud_ll_s1, int_hexagon_M2_mpyud_ll_s1>;

//===----------------------------------------------------------------------===//
// MPYS / Multipy signed/unsigned halfwords and add/subtract the
// result from the 64-bit destination register.
//Rxx [-+]= mpy[u](Rs.[H|L],Rt.[H|L])[:<<1][:sat]
//===----------------------------------------------------------------------===//

def : T_PRR_pat <M2_mpyd_acc_hh_s0, int_hexagon_M2_mpyd_acc_hh_s0>;
def : T_PRR_pat <M2_mpyd_acc_hl_s0, int_hexagon_M2_mpyd_acc_hl_s0>;
def : T_PRR_pat <M2_mpyd_acc_lh_s0, int_hexagon_M2_mpyd_acc_lh_s0>;
def : T_PRR_pat <M2_mpyd_acc_ll_s0, int_hexagon_M2_mpyd_acc_ll_s0>;

def : T_PRR_pat <M2_mpyd_acc_hh_s1, int_hexagon_M2_mpyd_acc_hh_s1>;
def : T_PRR_pat <M2_mpyd_acc_hl_s1, int_hexagon_M2_mpyd_acc_hl_s1>;
def : T_PRR_pat <M2_mpyd_acc_lh_s1, int_hexagon_M2_mpyd_acc_lh_s1>;
def : T_PRR_pat <M2_mpyd_acc_ll_s1, int_hexagon_M2_mpyd_acc_ll_s1>;

def : T_PRR_pat <M2_mpyd_nac_hh_s0, int_hexagon_M2_mpyd_nac_hh_s0>;
def : T_PRR_pat <M2_mpyd_nac_hl_s0, int_hexagon_M2_mpyd_nac_hl_s0>;
def : T_PRR_pat <M2_mpyd_nac_lh_s0, int_hexagon_M2_mpyd_nac_lh_s0>;
def : T_PRR_pat <M2_mpyd_nac_ll_s0, int_hexagon_M2_mpyd_nac_ll_s0>;

def : T_PRR_pat <M2_mpyd_nac_hh_s1, int_hexagon_M2_mpyd_nac_hh_s1>;
def : T_PRR_pat <M2_mpyd_nac_hl_s1, int_hexagon_M2_mpyd_nac_hl_s1>;
def : T_PRR_pat <M2_mpyd_nac_lh_s1, int_hexagon_M2_mpyd_nac_lh_s1>;
def : T_PRR_pat <M2_mpyd_nac_ll_s1, int_hexagon_M2_mpyd_nac_ll_s1>;

def : T_PRR_pat <M2_mpyud_acc_hh_s0, int_hexagon_M2_mpyud_acc_hh_s0>;
def : T_PRR_pat <M2_mpyud_acc_hl_s0, int_hexagon_M2_mpyud_acc_hl_s0>;
def : T_PRR_pat <M2_mpyud_acc_lh_s0, int_hexagon_M2_mpyud_acc_lh_s0>;
def : T_PRR_pat <M2_mpyud_acc_ll_s0, int_hexagon_M2_mpyud_acc_ll_s0>;

def : T_PRR_pat <M2_mpyud_acc_hh_s1, int_hexagon_M2_mpyud_acc_hh_s1>;
def : T_PRR_pat <M2_mpyud_acc_hl_s1, int_hexagon_M2_mpyud_acc_hl_s1>;
def : T_PRR_pat <M2_mpyud_acc_lh_s1, int_hexagon_M2_mpyud_acc_lh_s1>;
def : T_PRR_pat <M2_mpyud_acc_ll_s1, int_hexagon_M2_mpyud_acc_ll_s1>;

def : T_PRR_pat <M2_mpyud_nac_hh_s0, int_hexagon_M2_mpyud_nac_hh_s0>;
def : T_PRR_pat <M2_mpyud_nac_hl_s0, int_hexagon_M2_mpyud_nac_hl_s0>;
def : T_PRR_pat <M2_mpyud_nac_lh_s0, int_hexagon_M2_mpyud_nac_lh_s0>;
def : T_PRR_pat <M2_mpyud_nac_ll_s0, int_hexagon_M2_mpyud_nac_ll_s0>;

def : T_PRR_pat <M2_mpyud_nac_hh_s1, int_hexagon_M2_mpyud_nac_hh_s1>;
def : T_PRR_pat <M2_mpyud_nac_hl_s1, int_hexagon_M2_mpyud_nac_hl_s1>;
def : T_PRR_pat <M2_mpyud_nac_lh_s1, int_hexagon_M2_mpyud_nac_lh_s1>;
def : T_PRR_pat <M2_mpyud_nac_ll_s1, int_hexagon_M2_mpyud_nac_ll_s1>;

// Vector complex multiply imaginary: Rdd=vcmpyi(Rss,Rtt)[:<<1]:sat
def : T_PP_pat <M2_vcmpy_s1_sat_i, int_hexagon_M2_vcmpy_s1_sat_i>;
def : T_PP_pat <M2_vcmpy_s0_sat_i, int_hexagon_M2_vcmpy_s0_sat_i>;

// Vector complex multiply real: Rdd=vcmpyr(Rss,Rtt)[:<<1]:sat
def : T_PP_pat <M2_vcmpy_s1_sat_r, int_hexagon_M2_vcmpy_s1_sat_r>;
def : T_PP_pat <M2_vcmpy_s0_sat_r, int_hexagon_M2_vcmpy_s0_sat_r>;

// Vector dual multiply: Rdd=vdmpy(Rss,Rtt)[:<<1]:sat
def : T_PP_pat <M2_vdmpys_s1, int_hexagon_M2_vdmpys_s1>;
def : T_PP_pat <M2_vdmpys_s0, int_hexagon_M2_vdmpys_s0>;

// Vector multiply even halfwords: Rdd=vmpyeh(Rss,Rtt)[:<<1]:sat
def : T_PP_pat <M2_vmpy2es_s1, int_hexagon_M2_vmpy2es_s1>;
def : T_PP_pat <M2_vmpy2es_s0, int_hexagon_M2_vmpy2es_s0>;

//Rdd=vmpywoh(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PP_pat <M2_mmpyh_s0,  int_hexagon_M2_mmpyh_s0>;
def : T_PP_pat <M2_mmpyh_s1,  int_hexagon_M2_mmpyh_s1>;
def : T_PP_pat <M2_mmpyh_rs0, int_hexagon_M2_mmpyh_rs0>;
def : T_PP_pat <M2_mmpyh_rs1, int_hexagon_M2_mmpyh_rs1>;

//Rdd=vmpyweh(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PP_pat <M2_mmpyl_s0,  int_hexagon_M2_mmpyl_s0>;
def : T_PP_pat <M2_mmpyl_s1,  int_hexagon_M2_mmpyl_s1>;
def : T_PP_pat <M2_mmpyl_rs0, int_hexagon_M2_mmpyl_rs0>;
def : T_PP_pat <M2_mmpyl_rs1, int_hexagon_M2_mmpyl_rs1>;

//Rdd=vmpywouh(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PP_pat <M2_mmpyuh_s0,  int_hexagon_M2_mmpyuh_s0>;
def : T_PP_pat <M2_mmpyuh_s1,  int_hexagon_M2_mmpyuh_s1>;
def : T_PP_pat <M2_mmpyuh_rs0, int_hexagon_M2_mmpyuh_rs0>;
def : T_PP_pat <M2_mmpyuh_rs1, int_hexagon_M2_mmpyuh_rs1>;

//Rdd=vmpyweuh(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PP_pat <M2_mmpyul_s0,  int_hexagon_M2_mmpyul_s0>;
def : T_PP_pat <M2_mmpyul_s1,  int_hexagon_M2_mmpyul_s1>;
def : T_PP_pat <M2_mmpyul_rs0, int_hexagon_M2_mmpyul_rs0>;
def : T_PP_pat <M2_mmpyul_rs1, int_hexagon_M2_mmpyul_rs1>;

// Vector reduce add unsigned bytes: Rdd32[+]=vrmpybu(Rss32,Rtt32)
def : T_PP_pat  <A2_vraddub,     int_hexagon_A2_vraddub>;
def : T_PPP_pat <A2_vraddub_acc, int_hexagon_A2_vraddub_acc>;

// Vector sum of absolute differences unsigned bytes: Rdd=vrsadub(Rss,Rtt)
def : T_PP_pat  <A2_vrsadub,     int_hexagon_A2_vrsadub>;
def : T_PPP_pat <A2_vrsadub_acc, int_hexagon_A2_vrsadub_acc>;

// Vector absolute difference: Rdd=vabsdiffh(Rtt,Rss)
def : T_PP_pat <M2_vabsdiffh, int_hexagon_M2_vabsdiffh>;

// Vector absolute difference words: Rdd=vabsdiffw(Rtt,Rss)
def : T_PP_pat <M2_vabsdiffw, int_hexagon_M2_vabsdiffw>;

// Vector reduce complex multiply real or imaginary:
// Rdd[+]=vrcmpy[ir](Rss,Rtt[*])
def : T_PP_pat  <M2_vrcmpyi_s0,  int_hexagon_M2_vrcmpyi_s0>;
def : T_PP_pat  <M2_vrcmpyi_s0c, int_hexagon_M2_vrcmpyi_s0c>;
def : T_PPP_pat <M2_vrcmaci_s0,  int_hexagon_M2_vrcmaci_s0>;
def : T_PPP_pat <M2_vrcmaci_s0c, int_hexagon_M2_vrcmaci_s0c>;

def : T_PP_pat  <M2_vrcmpyr_s0,  int_hexagon_M2_vrcmpyr_s0>;
def : T_PP_pat  <M2_vrcmpyr_s0c, int_hexagon_M2_vrcmpyr_s0c>;
def : T_PPP_pat <M2_vrcmacr_s0,  int_hexagon_M2_vrcmacr_s0>;
def : T_PPP_pat <M2_vrcmacr_s0c, int_hexagon_M2_vrcmacr_s0c>;

// Vector reduce halfwords
// Rdd[+]=vrmpyh(Rss,Rtt)
def : T_PP_pat  <M2_vrmpy_s0, int_hexagon_M2_vrmpy_s0>;
def : T_PPP_pat <M2_vrmac_s0, int_hexagon_M2_vrmac_s0>;

//===----------------------------------------------------------------------===//
// Vector Multipy with accumulation
//===----------------------------------------------------------------------===//

// Vector multiply word by signed half with accumulation
// Rxx+=vmpyw[eo]h(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PPP_pat <M2_mmacls_s1, int_hexagon_M2_mmacls_s1>;
def : T_PPP_pat <M2_mmacls_s0, int_hexagon_M2_mmacls_s0>;
def : T_PPP_pat <M2_mmacls_rs1, int_hexagon_M2_mmacls_rs1>;
def : T_PPP_pat <M2_mmacls_rs0, int_hexagon_M2_mmacls_rs0>;
def : T_PPP_pat <M2_mmachs_s1, int_hexagon_M2_mmachs_s1>;
def : T_PPP_pat <M2_mmachs_s0, int_hexagon_M2_mmachs_s0>;
def : T_PPP_pat <M2_mmachs_rs1, int_hexagon_M2_mmachs_rs1>;
def : T_PPP_pat <M2_mmachs_rs0, int_hexagon_M2_mmachs_rs0>;

// Vector multiply word by unsigned half with accumulation
// Rxx+=vmpyw[eo]uh(Rss,Rtt)[:<<1][:rnd]:sat
def : T_PPP_pat <M2_mmaculs_s1, int_hexagon_M2_mmaculs_s1>;
def : T_PPP_pat <M2_mmaculs_s0, int_hexagon_M2_mmaculs_s0>;
def : T_PPP_pat <M2_mmaculs_rs1, int_hexagon_M2_mmaculs_rs1>;
def : T_PPP_pat <M2_mmaculs_rs0, int_hexagon_M2_mmaculs_rs0>;
def : T_PPP_pat <M2_mmacuhs_s1, int_hexagon_M2_mmacuhs_s1>;
def : T_PPP_pat <M2_mmacuhs_s0, int_hexagon_M2_mmacuhs_s0>;
def : T_PPP_pat <M2_mmacuhs_rs1, int_hexagon_M2_mmacuhs_rs1>;
def : T_PPP_pat <M2_mmacuhs_rs0, int_hexagon_M2_mmacuhs_rs0>;

// Vector multiply even halfwords with accumulation
// Rxx+=vmpyeh(Rss,Rtt)[:<<1][:sat]
def : T_PPP_pat <M2_vmac2es, int_hexagon_M2_vmac2es>;
def : T_PPP_pat <M2_vmac2es_s1, int_hexagon_M2_vmac2es_s1>;
def : T_PPP_pat <M2_vmac2es_s0, int_hexagon_M2_vmac2es_s0>;

// Vector dual multiply with accumulation
// Rxx+=vdmpy(Rss,Rtt)[:sat]
def : T_PPP_pat <M2_vdmacs_s1, int_hexagon_M2_vdmacs_s1>;
def : T_PPP_pat <M2_vdmacs_s0, int_hexagon_M2_vdmacs_s0>;

// Vector complex multiply real or imaginary with accumulation
// Rxx+=vcmpy[ir](Rss,Rtt):sat
def : T_PPP_pat <M2_vcmac_s0_sat_r, int_hexagon_M2_vcmac_s0_sat_r>;
def : T_PPP_pat <M2_vcmac_s0_sat_i, int_hexagon_M2_vcmac_s0_sat_i>;

//===----------------------------------------------------------------------===//
// Add/Subtract halfword
// Rd=add(Rt.L,Rs.[HL])[:sat]
// Rd=sub(Rt.L,Rs.[HL])[:sat]
// Rd=add(Rt.[LH],Rs.[HL])[:sat][:<16]
// Rd=sub(Rt.[LH],Rs.[HL])[:sat][:<16]
//===----------------------------------------------------------------------===//

//Rd=add(Rt.L,Rs.[LH])
def : T_RR_pat <A2_addh_l16_ll,     int_hexagon_A2_addh_l16_ll>;
def : T_RR_pat <A2_addh_l16_hl,     int_hexagon_A2_addh_l16_hl>;

//Rd=add(Rt.L,Rs.[LH]):sat
def : T_RR_pat <A2_addh_l16_sat_ll, int_hexagon_A2_addh_l16_sat_ll>;
def : T_RR_pat <A2_addh_l16_sat_hl, int_hexagon_A2_addh_l16_sat_hl>;

//Rd=sub(Rt.L,Rs.[LH])
def : T_RR_pat <A2_subh_l16_ll,     int_hexagon_A2_subh_l16_ll>;
def : T_RR_pat <A2_subh_l16_hl,     int_hexagon_A2_subh_l16_hl>;

//Rd=sub(Rt.L,Rs.[LH]):sat
def : T_RR_pat <A2_subh_l16_sat_ll, int_hexagon_A2_subh_l16_sat_ll>;
def : T_RR_pat <A2_subh_l16_sat_hl, int_hexagon_A2_subh_l16_sat_hl>;

//Rd=add(Rt.[LH],Rs.[LH]):<<16
def : T_RR_pat <A2_addh_h16_ll,     int_hexagon_A2_addh_h16_ll>;
def : T_RR_pat <A2_addh_h16_lh,     int_hexagon_A2_addh_h16_lh>;
def : T_RR_pat <A2_addh_h16_hl,     int_hexagon_A2_addh_h16_hl>;
def : T_RR_pat <A2_addh_h16_hh,     int_hexagon_A2_addh_h16_hh>;

//Rd=sub(Rt.[LH],Rs.[LH]):<<16
def : T_RR_pat <A2_subh_h16_ll,     int_hexagon_A2_subh_h16_ll>;
def : T_RR_pat <A2_subh_h16_lh,     int_hexagon_A2_subh_h16_lh>;
def : T_RR_pat <A2_subh_h16_hl,     int_hexagon_A2_subh_h16_hl>;
def : T_RR_pat <A2_subh_h16_hh,     int_hexagon_A2_subh_h16_hh>;

//Rd=add(Rt.[LH],Rs.[LH]):sat:<<16
def : T_RR_pat <A2_addh_h16_sat_ll, int_hexagon_A2_addh_h16_sat_ll>;
def : T_RR_pat <A2_addh_h16_sat_lh, int_hexagon_A2_addh_h16_sat_lh>;
def : T_RR_pat <A2_addh_h16_sat_hl, int_hexagon_A2_addh_h16_sat_hl>;
def : T_RR_pat <A2_addh_h16_sat_hh, int_hexagon_A2_addh_h16_sat_hh>;

//Rd=sub(Rt.[LH],Rs.[LH]):sat:<<16
def : T_RR_pat <A2_subh_h16_sat_ll, int_hexagon_A2_subh_h16_sat_ll>;
def : T_RR_pat <A2_subh_h16_sat_lh, int_hexagon_A2_subh_h16_sat_lh>;
def : T_RR_pat <A2_subh_h16_sat_hl, int_hexagon_A2_subh_h16_sat_hl>;
def : T_RR_pat <A2_subh_h16_sat_hh, int_hexagon_A2_subh_h16_sat_hh>;

// ALU64 / ALU / min max
def : T_RR_pat<A2_max,  int_hexagon_A2_max>;
def : T_RR_pat<A2_min,  int_hexagon_A2_min>;
def : T_RR_pat<A2_maxu, int_hexagon_A2_maxu>;
def : T_RR_pat<A2_minu, int_hexagon_A2_minu>;

// Shift and accumulate
def : T_RRI_pat <S2_asr_i_r_nac,  int_hexagon_S2_asr_i_r_nac>;
def : T_RRI_pat <S2_lsr_i_r_nac,  int_hexagon_S2_lsr_i_r_nac>;
def : T_RRI_pat <S2_asl_i_r_nac,  int_hexagon_S2_asl_i_r_nac>;
def : T_RRI_pat <S2_asr_i_r_acc,  int_hexagon_S2_asr_i_r_acc>;
def : T_RRI_pat <S2_lsr_i_r_acc,  int_hexagon_S2_lsr_i_r_acc>;
def : T_RRI_pat <S2_asl_i_r_acc,  int_hexagon_S2_asl_i_r_acc>;

def : T_RRI_pat <S2_asr_i_r_and,  int_hexagon_S2_asr_i_r_and>;
def : T_RRI_pat <S2_lsr_i_r_and,  int_hexagon_S2_lsr_i_r_and>;
def : T_RRI_pat <S2_asl_i_r_and,  int_hexagon_S2_asl_i_r_and>;
def : T_RRI_pat <S2_asr_i_r_or,   int_hexagon_S2_asr_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_or,   int_hexagon_S2_lsr_i_r_or>;
def : T_RRI_pat <S2_asl_i_r_or,   int_hexagon_S2_asl_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_xacc, int_hexagon_S2_lsr_i_r_xacc>;
def : T_RRI_pat <S2_asl_i_r_xacc, int_hexagon_S2_asl_i_r_xacc>;

def : T_PPI_pat <S2_asr_i_p_nac,  int_hexagon_S2_asr_i_p_nac>;
def : T_PPI_pat <S2_lsr_i_p_nac,  int_hexagon_S2_lsr_i_p_nac>;
def : T_PPI_pat <S2_asl_i_p_nac,  int_hexagon_S2_asl_i_p_nac>;
def : T_PPI_pat <S2_asr_i_p_acc,  int_hexagon_S2_asr_i_p_acc>;
def : T_PPI_pat <S2_lsr_i_p_acc,  int_hexagon_S2_lsr_i_p_acc>;
def : T_PPI_pat <S2_asl_i_p_acc,  int_hexagon_S2_asl_i_p_acc>;

def : T_PPI_pat <S2_asr_i_p_and,  int_hexagon_S2_asr_i_p_and>;
def : T_PPI_pat <S2_lsr_i_p_and,  int_hexagon_S2_lsr_i_p_and>;
def : T_PPI_pat <S2_asl_i_p_and,  int_hexagon_S2_asl_i_p_and>;
def : T_PPI_pat <S2_asr_i_p_or,   int_hexagon_S2_asr_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_or,   int_hexagon_S2_lsr_i_p_or>;
def : T_PPI_pat <S2_asl_i_p_or,   int_hexagon_S2_asl_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_xacc, int_hexagon_S2_lsr_i_p_xacc>;
def : T_PPI_pat <S2_asl_i_p_xacc, int_hexagon_S2_asl_i_p_xacc>;

def : T_RRR_pat <S2_asr_r_r_nac,  int_hexagon_S2_asr_r_r_nac>;
def : T_RRR_pat <S2_lsr_r_r_nac,  int_hexagon_S2_lsr_r_r_nac>;
def : T_RRR_pat <S2_asl_r_r_nac,  int_hexagon_S2_asl_r_r_nac>;
def : T_RRR_pat <S2_lsl_r_r_nac,  int_hexagon_S2_lsl_r_r_nac>;
def : T_RRR_pat <S2_asr_r_r_acc,  int_hexagon_S2_asr_r_r_acc>;
def : T_RRR_pat <S2_lsr_r_r_acc,  int_hexagon_S2_lsr_r_r_acc>;
def : T_RRR_pat <S2_asl_r_r_acc,  int_hexagon_S2_asl_r_r_acc>;
def : T_RRR_pat <S2_lsl_r_r_acc,  int_hexagon_S2_lsl_r_r_acc>;

def : T_RRR_pat <S2_asr_r_r_and,  int_hexagon_S2_asr_r_r_and>;
def : T_RRR_pat <S2_lsr_r_r_and,  int_hexagon_S2_lsr_r_r_and>;
def : T_RRR_pat <S2_asl_r_r_and,  int_hexagon_S2_asl_r_r_and>;
def : T_RRR_pat <S2_lsl_r_r_and,  int_hexagon_S2_lsl_r_r_and>;
def : T_RRR_pat <S2_asr_r_r_or,   int_hexagon_S2_asr_r_r_or>;
def : T_RRR_pat <S2_lsr_r_r_or,   int_hexagon_S2_lsr_r_r_or>;
def : T_RRR_pat <S2_asl_r_r_or,   int_hexagon_S2_asl_r_r_or>;
def : T_RRR_pat <S2_lsl_r_r_or,   int_hexagon_S2_lsl_r_r_or>;

def : T_PPR_pat <S2_asr_r_p_nac,  int_hexagon_S2_asr_r_p_nac>;
def : T_PPR_pat <S2_lsr_r_p_nac,  int_hexagon_S2_lsr_r_p_nac>;
def : T_PPR_pat <S2_asl_r_p_nac,  int_hexagon_S2_asl_r_p_nac>;
def : T_PPR_pat <S2_lsl_r_p_nac,  int_hexagon_S2_lsl_r_p_nac>;
def : T_PPR_pat <S2_asr_r_p_acc,  int_hexagon_S2_asr_r_p_acc>;
def : T_PPR_pat <S2_lsr_r_p_acc,  int_hexagon_S2_lsr_r_p_acc>;
def : T_PPR_pat <S2_asl_r_p_acc,  int_hexagon_S2_asl_r_p_acc>;
def : T_PPR_pat <S2_lsl_r_p_acc,  int_hexagon_S2_lsl_r_p_acc>;

def : T_PPR_pat <S2_asr_r_p_and,  int_hexagon_S2_asr_r_p_and>;
def : T_PPR_pat <S2_lsr_r_p_and,  int_hexagon_S2_lsr_r_p_and>;
def : T_PPR_pat <S2_asl_r_p_and,  int_hexagon_S2_asl_r_p_and>;
def : T_PPR_pat <S2_lsl_r_p_and,  int_hexagon_S2_lsl_r_p_and>;
def : T_PPR_pat <S2_asr_r_p_or,   int_hexagon_S2_asr_r_p_or>;
def : T_PPR_pat <S2_lsr_r_p_or,   int_hexagon_S2_lsr_r_p_or>;
def : T_PPR_pat <S2_asl_r_p_or,   int_hexagon_S2_asl_r_p_or>;
def : T_PPR_pat <S2_lsl_r_p_or,   int_hexagon_S2_lsl_r_p_or>;

def : T_RRI_pat <S2_asr_i_r_nac,  int_hexagon_S2_asr_i_r_nac>;
def : T_RRI_pat <S2_lsr_i_r_nac,  int_hexagon_S2_lsr_i_r_nac>;
def : T_RRI_pat <S2_asl_i_r_nac,  int_hexagon_S2_asl_i_r_nac>;
def : T_RRI_pat <S2_asr_i_r_acc,  int_hexagon_S2_asr_i_r_acc>;
def : T_RRI_pat <S2_lsr_i_r_acc,  int_hexagon_S2_lsr_i_r_acc>;
def : T_RRI_pat <S2_asl_i_r_acc,  int_hexagon_S2_asl_i_r_acc>;

def : T_RRI_pat <S2_asr_i_r_and,  int_hexagon_S2_asr_i_r_and>;
def : T_RRI_pat <S2_lsr_i_r_and,  int_hexagon_S2_lsr_i_r_and>;
def : T_RRI_pat <S2_asl_i_r_and,  int_hexagon_S2_asl_i_r_and>;
def : T_RRI_pat <S2_asr_i_r_or,   int_hexagon_S2_asr_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_or,   int_hexagon_S2_lsr_i_r_or>;
def : T_RRI_pat <S2_asl_i_r_or,   int_hexagon_S2_asl_i_r_or>;
def : T_RRI_pat <S2_lsr_i_r_xacc, int_hexagon_S2_lsr_i_r_xacc>;
def : T_RRI_pat <S2_asl_i_r_xacc, int_hexagon_S2_asl_i_r_xacc>;

def : T_PPI_pat <S2_asr_i_p_nac,  int_hexagon_S2_asr_i_p_nac>;
def : T_PPI_pat <S2_lsr_i_p_nac,  int_hexagon_S2_lsr_i_p_nac>;
def : T_PPI_pat <S2_asl_i_p_nac,  int_hexagon_S2_asl_i_p_nac>;
def : T_PPI_pat <S2_asr_i_p_acc,  int_hexagon_S2_asr_i_p_acc>;
def : T_PPI_pat <S2_lsr_i_p_acc,  int_hexagon_S2_lsr_i_p_acc>;
def : T_PPI_pat <S2_asl_i_p_acc,  int_hexagon_S2_asl_i_p_acc>;

def : T_PPI_pat <S2_asr_i_p_and,  int_hexagon_S2_asr_i_p_and>;
def : T_PPI_pat <S2_lsr_i_p_and,  int_hexagon_S2_lsr_i_p_and>;
def : T_PPI_pat <S2_asl_i_p_and,  int_hexagon_S2_asl_i_p_and>;
def : T_PPI_pat <S2_asr_i_p_or,   int_hexagon_S2_asr_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_or,   int_hexagon_S2_lsr_i_p_or>;
def : T_PPI_pat <S2_asl_i_p_or,   int_hexagon_S2_asl_i_p_or>;
def : T_PPI_pat <S2_lsr_i_p_xacc, int_hexagon_S2_lsr_i_p_xacc>;
def : T_PPI_pat <S2_asl_i_p_xacc, int_hexagon_S2_asl_i_p_xacc>;

def : T_RRR_pat <S2_asr_r_r_nac,  int_hexagon_S2_asr_r_r_nac>;
def : T_RRR_pat <S2_lsr_r_r_nac,  int_hexagon_S2_lsr_r_r_nac>;
def : T_RRR_pat <S2_asl_r_r_nac,  int_hexagon_S2_asl_r_r_nac>;
def : T_RRR_pat <S2_lsl_r_r_nac,  int_hexagon_S2_lsl_r_r_nac>;
def : T_RRR_pat <S2_asr_r_r_acc,  int_hexagon_S2_asr_r_r_acc>;
def : T_RRR_pat <S2_lsr_r_r_acc,  int_hexagon_S2_lsr_r_r_acc>;
def : T_RRR_pat <S2_asl_r_r_acc,  int_hexagon_S2_asl_r_r_acc>;
def : T_RRR_pat <S2_lsl_r_r_acc,  int_hexagon_S2_lsl_r_r_acc>;

def : T_RRR_pat <S2_asr_r_r_and,  int_hexagon_S2_asr_r_r_and>;
def : T_RRR_pat <S2_lsr_r_r_and,  int_hexagon_S2_lsr_r_r_and>;
def : T_RRR_pat <S2_asl_r_r_and,  int_hexagon_S2_asl_r_r_and>;
def : T_RRR_pat <S2_lsl_r_r_and,  int_hexagon_S2_lsl_r_r_and>;
def : T_RRR_pat <S2_asr_r_r_or,   int_hexagon_S2_asr_r_r_or>;
def : T_RRR_pat <S2_lsr_r_r_or,   int_hexagon_S2_lsr_r_r_or>;
def : T_RRR_pat <S2_asl_r_r_or,   int_hexagon_S2_asl_r_r_or>;
def : T_RRR_pat <S2_lsl_r_r_or,   int_hexagon_S2_lsl_r_r_or>;

def : T_PPR_pat <S2_asr_r_p_nac,  int_hexagon_S2_asr_r_p_nac>;
def : T_PPR_pat <S2_lsr_r_p_nac,  int_hexagon_S2_lsr_r_p_nac>;
def : T_PPR_pat <S2_asl_r_p_nac,  int_hexagon_S2_asl_r_p_nac>;
def : T_PPR_pat <S2_lsl_r_p_nac,  int_hexagon_S2_lsl_r_p_nac>;
def : T_PPR_pat <S2_asr_r_p_acc,  int_hexagon_S2_asr_r_p_acc>;
def : T_PPR_pat <S2_lsr_r_p_acc,  int_hexagon_S2_lsr_r_p_acc>;
def : T_PPR_pat <S2_asl_r_p_acc,  int_hexagon_S2_asl_r_p_acc>;
def : T_PPR_pat <S2_lsl_r_p_acc,  int_hexagon_S2_lsl_r_p_acc>;

def : T_PPR_pat <S2_asr_r_p_and,  int_hexagon_S2_asr_r_p_and>;
def : T_PPR_pat <S2_lsr_r_p_and,  int_hexagon_S2_lsr_r_p_and>;
def : T_PPR_pat <S2_asl_r_p_and,  int_hexagon_S2_asl_r_p_and>;
def : T_PPR_pat <S2_lsl_r_p_and,  int_hexagon_S2_lsl_r_p_and>;
def : T_PPR_pat <S2_asr_r_p_or,   int_hexagon_S2_asr_r_p_or>;
def : T_PPR_pat <S2_lsr_r_p_or,   int_hexagon_S2_lsr_r_p_or>;
def : T_PPR_pat <S2_asl_r_p_or,   int_hexagon_S2_asl_r_p_or>;
def : T_PPR_pat <S2_lsl_r_p_or,   int_hexagon_S2_lsl_r_p_or>;

//*******************************************************************
//           ALU32/ALU
//*******************************************************************
def : T_RR_pat<A2_add,      int_hexagon_A2_add>;
def : T_RI_pat<A2_addi,     int_hexagon_A2_addi>;
def : T_RR_pat<A2_sub,      int_hexagon_A2_sub>;
def : T_IR_pat<A2_subri,    int_hexagon_A2_subri>;
def : T_RR_pat<A2_and,      int_hexagon_A2_and>;
def : T_RI_pat<A2_andir,    int_hexagon_A2_andir>;
def : T_RR_pat<A2_or,       int_hexagon_A2_or>;
def : T_RI_pat<A2_orir,     int_hexagon_A2_orir>;
def : T_RR_pat<A2_xor,      int_hexagon_A2_xor>;
def : T_RR_pat<A2_combinew, int_hexagon_A2_combinew>;

// Assembler mapped from Rd32=not(Rs32) to Rd32=sub(#-1,Rs32)
def : Pat <(int_hexagon_A2_not I32:$Rs),
           (A2_subri -1, I32:$Rs)>;

// Assembler mapped from Rd32=neg(Rs32) to Rd32=sub(#0,Rs32)
def : Pat <(int_hexagon_A2_neg I32:$Rs),
           (A2_subri 0, I32:$Rs)>;

// Transfer immediate
def  : Pat <(int_hexagon_A2_tfril I32:$Rs, u16_0ImmPred:$Is),
            (A2_tfril I32:$Rs, u16_0ImmPred:$Is)>;
def  : Pat <(int_hexagon_A2_tfrih I32:$Rs, u16_0ImmPred:$Is),
            (A2_tfrih I32:$Rs, u16_0ImmPred:$Is)>;

//  Transfer Register/immediate.
def : T_R_pat <A2_tfr, int_hexagon_A2_tfr>;
def : T_I_pat <A2_tfrsi, int_hexagon_A2_tfrsi>;

def ImmExt64: SDNodeXForm<imm, [{
  int64_t V = N->getSExtValue();
  return CurDAG->getTargetConstant(V, SDLoc(N), MVT::i64);
}]>;

// A2_tfrpi has an operand of type i64. This is necessary, since it is
// generated from "(set I64:$Rd, imm)". That pattern would not appear
// in the DAG, if the immediate was not a 64-bit value.
// The builtin for A2_tfrpi, on the other hand, takes a 32-bit value,
// which makes it impossible to simply replace it with the instruction.
// To connect the builtin with the instruction, the builtin's operand
// needs to be extended to the right type.

def : Pat<(int_hexagon_A2_tfrpi imm:$Is),
          (A2_tfrpi (ImmExt64 $Is))>;

// Assembler mapped from Rdd32=Rss32 to Rdd32=combine(Rss.H32,Rss.L32)
def : Pat<(int_hexagon_A2_tfrp I64:$src),
          (A2_combinew (HiReg I64:$src), (LoReg I64:$src))>;

//*******************************************************************
//           ALU32/PERM
//*******************************************************************
// Combine
def: T_RR_pat<A2_combine_hh, int_hexagon_A2_combine_hh>;
def: T_RR_pat<A2_combine_hl, int_hexagon_A2_combine_hl>;
def: T_RR_pat<A2_combine_lh, int_hexagon_A2_combine_lh>;
def: T_RR_pat<A2_combine_ll, int_hexagon_A2_combine_ll>;

def: T_II_pat<A2_combineii, int_hexagon_A2_combineii, s32_0ImmPred, s8_0ImmPred>;

// Mux
def : T_QRR_pat<C2_mux,   int_hexagon_C2_mux>;
def : T_QRI_pat<C2_muxir, int_hexagon_C2_muxir, s32_0ImmPred>;
def : T_QIR_pat<C2_muxri, int_hexagon_C2_muxri, s32_0ImmPred>;
def : T_QII_pat<C2_muxii, int_hexagon_C2_muxii, s32_0ImmPred, s8_0ImmPred>;

// Shift halfword
def : T_R_pat<A2_aslh, int_hexagon_A2_aslh>;
def : T_R_pat<A2_asrh, int_hexagon_A2_asrh>;
def : T_R_pat<A2_asrh, int_hexagon_SI_to_SXTHI_asrh>;

// Sign/zero extend
def : T_R_pat<A2_sxth, int_hexagon_A2_sxth>;
def : T_R_pat<A2_sxtb, int_hexagon_A2_sxtb>;
def : T_R_pat<A2_zxth, int_hexagon_A2_zxth>;
def : T_R_pat<A2_zxtb, int_hexagon_A2_zxtb>;

//*******************************************************************
//           ALU32/PRED
//*******************************************************************
// Compare
def : T_Q_RR_pat<C2_cmpeq,  int_hexagon_C2_cmpeq>;
def : T_Q_RR_pat<C2_cmpgt,  int_hexagon_C2_cmpgt>;
def : T_Q_RR_pat<C2_cmpgtu, int_hexagon_C2_cmpgtu>;

def : T_Q_RI_pat<C2_cmpeqi,  int_hexagon_C2_cmpeqi, s32_0ImmPred>;
def : T_Q_RI_pat<C2_cmpgti,  int_hexagon_C2_cmpgti, s32_0ImmPred>;
def : T_Q_RI_pat<C2_cmpgtui, int_hexagon_C2_cmpgtui, u32_0ImmPred>;

def : Pat <(int_hexagon_C2_cmpgei I32:$src1, s32_0ImmPred:$src2),
           (C2_tfrpr (C2_cmpgti I32:$src1, (SDEC1 s32_0ImmPred:$src2)))>;

def : Pat <(int_hexagon_C2_cmpgeui I32:$src1, u32_0ImmPred:$src2),
           (C2_tfrpr (C2_cmpgtui I32:$src1, (UDEC1 u32_0ImmPred:$src2)))>;

def : Pat <(int_hexagon_C2_cmpgeui I32:$src, 0),
           (C2_tfrpr (C2_cmpeq I32:$src, I32:$src))>;
def : Pat <(int_hexagon_C2_cmplt I32:$src1, I32:$src2),
           (C2_tfrpr (C2_cmpgt I32:$src2, I32:$src1))>;
def : Pat <(int_hexagon_C2_cmpltu I32:$src1, I32:$src2),
           (C2_tfrpr (C2_cmpgtu I32:$src2, I32:$src1))>;

//*******************************************************************
//           ALU32/VH
//*******************************************************************
// Vector add, subtract, average halfwords
def: T_RR_pat<A2_svaddh,   int_hexagon_A2_svaddh>;
def: T_RR_pat<A2_svaddhs,  int_hexagon_A2_svaddhs>;
def: T_RR_pat<A2_svadduhs, int_hexagon_A2_svadduhs>;

def: T_RR_pat<A2_svsubh,   int_hexagon_A2_svsubh>;
def: T_RR_pat<A2_svsubhs,  int_hexagon_A2_svsubhs>;
def: T_RR_pat<A2_svsubuhs, int_hexagon_A2_svsubuhs>;

def: T_RR_pat<A2_svavgh,   int_hexagon_A2_svavgh>;
def: T_RR_pat<A2_svavghs,  int_hexagon_A2_svavghs>;
def: T_RR_pat<A2_svnavgh,  int_hexagon_A2_svnavgh>;

//*******************************************************************
//           ALU64/ALU
//*******************************************************************
def: T_RR_pat<A2_addsat,     int_hexagon_A2_addsat>;
def: T_RR_pat<A2_subsat,     int_hexagon_A2_subsat>;
def: T_PP_pat<A2_addp,       int_hexagon_A2_addp>;
def: T_PP_pat<A2_subp,       int_hexagon_A2_subp>;

def: T_PP_pat<A2_andp,       int_hexagon_A2_andp>;
def: T_PP_pat<A2_orp,        int_hexagon_A2_orp>;
def: T_PP_pat<A2_xorp,       int_hexagon_A2_xorp>;

def: T_Q_PP_pat<C2_cmpeqp,   int_hexagon_C2_cmpeqp>;
def: T_Q_PP_pat<C2_cmpgtp,   int_hexagon_C2_cmpgtp>;
def: T_Q_PP_pat<C2_cmpgtup,  int_hexagon_C2_cmpgtup>;

def: T_PP_pat<S2_parityp,    int_hexagon_S2_parityp>;
def: T_RR_pat<S2_packhl,     int_hexagon_S2_packhl>;

//*******************************************************************
//           ALU64/VB
//*******************************************************************
// ALU64 - Vector add
def : T_PP_pat <A2_vaddub,   int_hexagon_A2_vaddub>;
def : T_PP_pat <A2_vaddubs,  int_hexagon_A2_vaddubs>;
def : T_PP_pat <A2_vaddh,    int_hexagon_A2_vaddh>;
def : T_PP_pat <A2_vaddhs,   int_hexagon_A2_vaddhs>;
def : T_PP_pat <A2_vadduhs,  int_hexagon_A2_vadduhs>;
def : T_PP_pat <A2_vaddw,    int_hexagon_A2_vaddw>;
def : T_PP_pat <A2_vaddws,   int_hexagon_A2_vaddws>;

// ALU64 - Vector average
def : T_PP_pat <A2_vavgub,   int_hexagon_A2_vavgub>;
def : T_PP_pat <A2_vavgubr,  int_hexagon_A2_vavgubr>;
def : T_PP_pat <A2_vavgh,    int_hexagon_A2_vavgh>;
def : T_PP_pat <A2_vavghr,   int_hexagon_A2_vavghr>;
def : T_PP_pat <A2_vavghcr,  int_hexagon_A2_vavghcr>;
def : T_PP_pat <A2_vavguh,   int_hexagon_A2_vavguh>;
def : T_PP_pat <A2_vavguhr,  int_hexagon_A2_vavguhr>;

def : T_PP_pat <A2_vavgw,    int_hexagon_A2_vavgw>;
def : T_PP_pat <A2_vavgwr,   int_hexagon_A2_vavgwr>;
def : T_PP_pat <A2_vavgwcr,  int_hexagon_A2_vavgwcr>;
def : T_PP_pat <A2_vavguw,   int_hexagon_A2_vavguw>;
def : T_PP_pat <A2_vavguwr,  int_hexagon_A2_vavguwr>;

// ALU64 - Vector negative average
def : T_PP_pat <A2_vnavgh,   int_hexagon_A2_vnavgh>;
def : T_PP_pat <A2_vnavghr,  int_hexagon_A2_vnavghr>;
def : T_PP_pat <A2_vnavghcr, int_hexagon_A2_vnavghcr>;
def : T_PP_pat <A2_vnavgw,   int_hexagon_A2_vnavgw>;
def : T_PP_pat <A2_vnavgwr,  int_hexagon_A2_vnavgwr>;
def : T_PP_pat <A2_vnavgwcr, int_hexagon_A2_vnavgwcr>;

// ALU64 - Vector max
def : T_PP_pat <A2_vmaxh,    int_hexagon_A2_vmaxh>;
def : T_PP_pat <A2_vmaxw,    int_hexagon_A2_vmaxw>;
def : T_PP_pat <A2_vmaxub,   int_hexagon_A2_vmaxub>;
def : T_PP_pat <A2_vmaxuh,   int_hexagon_A2_vmaxuh>;
def : T_PP_pat <A2_vmaxuw,   int_hexagon_A2_vmaxuw>;

// ALU64 - Vector min
def : T_PP_pat <A2_vminh,    int_hexagon_A2_vminh>;
def : T_PP_pat <A2_vminw,    int_hexagon_A2_vminw>;
def : T_PP_pat <A2_vminub,   int_hexagon_A2_vminub>;
def : T_PP_pat <A2_vminuh,   int_hexagon_A2_vminuh>;
def : T_PP_pat <A2_vminuw,   int_hexagon_A2_vminuw>;

// ALU64 - Vector sub
def : T_PP_pat <A2_vsubub,   int_hexagon_A2_vsubub>;
def : T_PP_pat <A2_vsububs,  int_hexagon_A2_vsububs>;
def : T_PP_pat <A2_vsubh,    int_hexagon_A2_vsubh>;
def : T_PP_pat <A2_vsubhs,   int_hexagon_A2_vsubhs>;
def : T_PP_pat <A2_vsubuhs,  int_hexagon_A2_vsubuhs>;
def : T_PP_pat <A2_vsubw,    int_hexagon_A2_vsubw>;
def : T_PP_pat <A2_vsubws,   int_hexagon_A2_vsubws>;

// ALU64 - Vector compare bytes
def : T_Q_PP_pat <A2_vcmpbeq,  int_hexagon_A2_vcmpbeq>;
def : T_Q_PP_pat <A4_vcmpbgt,  int_hexagon_A4_vcmpbgt>;
def : T_Q_PP_pat <A2_vcmpbgtu, int_hexagon_A2_vcmpbgtu>;

// ALU64 - Vector compare halfwords
def : T_Q_PP_pat <A2_vcmpheq,  int_hexagon_A2_vcmpheq>;
def : T_Q_PP_pat <A2_vcmphgt,  int_hexagon_A2_vcmphgt>;
def : T_Q_PP_pat <A2_vcmphgtu, int_hexagon_A2_vcmphgtu>;

// ALU64 - Vector compare words
def : T_Q_PP_pat <A2_vcmpweq,  int_hexagon_A2_vcmpweq>;
def : T_Q_PP_pat <A2_vcmpwgt,  int_hexagon_A2_vcmpwgt>;
def : T_Q_PP_pat <A2_vcmpwgtu, int_hexagon_A2_vcmpwgtu>;

// ALU64 / VB / Vector mux.
def : T_QPP_pat <C2_vmux,      int_hexagon_C2_vmux>;

// MPY - Multiply and use full result
// Rdd = mpy[u](Rs, Rt)
def : T_RR_pat <M2_dpmpyss_s0, int_hexagon_M2_dpmpyss_s0>;
def : T_RR_pat <M2_dpmpyuu_s0, int_hexagon_M2_dpmpyuu_s0>;

// Complex multiply real or imaginary
def : T_RR_pat <M2_cmpyi_s0,   int_hexagon_M2_cmpyi_s0>;
def : T_RR_pat <M2_cmpyr_s0,   int_hexagon_M2_cmpyr_s0>;

// Complex multiply
def : T_RR_pat <M2_cmpys_s0,   int_hexagon_M2_cmpys_s0>;
def : T_RR_pat <M2_cmpysc_s0,  int_hexagon_M2_cmpysc_s0>;
def : T_RR_pat <M2_cmpys_s1,   int_hexagon_M2_cmpys_s1>;
def : T_RR_pat <M2_cmpysc_s1,  int_hexagon_M2_cmpysc_s1>;

// Vector multiply halfwords
// Rdd=vmpyh(Rs,Rt)[:<<1]:sat
def : T_RR_pat <M2_vmpy2s_s0,  int_hexagon_M2_vmpy2s_s0>;
def : T_RR_pat <M2_vmpy2s_s1,  int_hexagon_M2_vmpy2s_s1>;

// Rxx[+-]= mpy[u](Rs,Rt)
def : T_PRR_pat <M2_dpmpyss_acc_s0, int_hexagon_M2_dpmpyss_acc_s0>;
def : T_PRR_pat <M2_dpmpyss_nac_s0, int_hexagon_M2_dpmpyss_nac_s0>;
def : T_PRR_pat <M2_dpmpyuu_acc_s0, int_hexagon_M2_dpmpyuu_acc_s0>;
def : T_PRR_pat <M2_dpmpyuu_nac_s0, int_hexagon_M2_dpmpyuu_nac_s0>;

// Rxx[-+]=cmpy(Rs,Rt)[:<<1]:sat
def : T_PRR_pat <M2_cmacs_s0, int_hexagon_M2_cmacs_s0>;
def : T_PRR_pat <M2_cnacs_s0, int_hexagon_M2_cnacs_s0>;
def : T_PRR_pat <M2_cmacs_s1, int_hexagon_M2_cmacs_s1>;
def : T_PRR_pat <M2_cnacs_s1, int_hexagon_M2_cnacs_s1>;

// Rxx[-+]=cmpy(Rs,Rt*)[:<<1]:sat
def : T_PRR_pat <M2_cmacsc_s0, int_hexagon_M2_cmacsc_s0>;
def : T_PRR_pat <M2_cnacsc_s0, int_hexagon_M2_cnacsc_s0>;
def : T_PRR_pat <M2_cmacsc_s1, int_hexagon_M2_cmacsc_s1>;
def : T_PRR_pat <M2_cnacsc_s1, int_hexagon_M2_cnacsc_s1>;

// Rxx+=cmpy[ir](Rs,Rt)
def : T_PRR_pat <M2_cmaci_s0, int_hexagon_M2_cmaci_s0>;
def : T_PRR_pat <M2_cmacr_s0, int_hexagon_M2_cmacr_s0>;

// Rxx+=vmpyh(Rs,Rt)[:<<1][:sat]
def : T_PRR_pat <M2_vmac2, int_hexagon_M2_vmac2>;
def : T_PRR_pat <M2_vmac2s_s0, int_hexagon_M2_vmac2s_s0>;
def : T_PRR_pat <M2_vmac2s_s1, int_hexagon_M2_vmac2s_s1>;

//*******************************************************************
//           CR
//*******************************************************************
def: T_Q_Q_pat<C2_not,       int_hexagon_C2_not>;
def: T_Q_Q_pat<C2_all8,      int_hexagon_C2_all8>;
def: T_Q_Q_pat<C2_any8,      int_hexagon_C2_any8>;
def: T_Q_Q_pat<C2_pxfer_map, int_hexagon_C2_pxfer_map>;

def: T_Q_QQ_pat<C2_and,      int_hexagon_C2_and>;
def: T_Q_QQ_pat<C2_andn,     int_hexagon_C2_andn>;
def: T_Q_QQ_pat<C2_or,       int_hexagon_C2_or>;
def: T_Q_QQ_pat<C2_orn,      int_hexagon_C2_orn>;
def: T_Q_QQ_pat<C2_xor,      int_hexagon_C2_xor>;

// Multiply 32x32 and use lower result
def : T_RRI_pat <M2_macsip, int_hexagon_M2_macsip>;
def : T_RRI_pat <M2_macsin, int_hexagon_M2_macsin>;
def : T_RRR_pat <M2_maci,   int_hexagon_M2_maci>;

// Subtract and accumulate
def : T_RRR_pat <M2_subacc, int_hexagon_M2_subacc>;

// Add and accumulate
def : T_RRR_pat <M2_acci,   int_hexagon_M2_acci>;
def : T_RRR_pat <M2_nacci,  int_hexagon_M2_nacci>;
def : T_RRI_pat <M2_accii,  int_hexagon_M2_accii>;
def : T_RRI_pat <M2_naccii, int_hexagon_M2_naccii>;

// XOR and XOR with destination
def : T_RRR_pat <M2_xor_xacc, int_hexagon_M2_xor_xacc>;

// Vector dual multiply with round and pack
def : T_PP_pat <M2_vdmpyrs_s0, int_hexagon_M2_vdmpyrs_s0>;
def : T_PP_pat <M2_vdmpyrs_s1, int_hexagon_M2_vdmpyrs_s1>;

// Vector multiply halfwords with round and pack
def : T_RR_pat <M2_vmpy2s_s0pack, int_hexagon_M2_vmpy2s_s0pack>;
def : T_RR_pat <M2_vmpy2s_s1pack, int_hexagon_M2_vmpy2s_s1pack>;

// Multiply and use lower result
def : T_RR_pat <M2_mpyi, int_hexagon_M2_mpyi>;
def : T_RI_pat <M2_mpysmi, int_hexagon_M2_mpysmi>;

// Assembler mapped from Rd32=mpyui(Rs32,Rt32) to Rd32=mpyi(Rs32,Rt32)
def : T_RR_pat <M2_mpyi, int_hexagon_M2_mpyui>;

// Multiply and use upper result
def : T_RR_pat <M2_mpy_up, int_hexagon_M2_mpy_up>;
def : T_RR_pat <M2_mpyu_up, int_hexagon_M2_mpyu_up>;
def : T_RR_pat <M2_hmmpyh_rs1, int_hexagon_M2_hmmpyh_rs1>;
def : T_RR_pat <M2_hmmpyl_rs1, int_hexagon_M2_hmmpyl_rs1>;
def : T_RR_pat <M2_dpmpyss_rnd_s0, int_hexagon_M2_dpmpyss_rnd_s0>;

// Complex multiply with round and pack
// Rxx32+=cmpy(Rs32,[*]Rt32:<<1]:rnd:sat
def : T_RR_pat <M2_cmpyrs_s0, int_hexagon_M2_cmpyrs_s0>;
def : T_RR_pat <M2_cmpyrs_s1, int_hexagon_M2_cmpyrs_s1>;
def : T_RR_pat <M2_cmpyrsc_s0, int_hexagon_M2_cmpyrsc_s0>;
def : T_RR_pat <M2_cmpyrsc_s1, int_hexagon_M2_cmpyrsc_s1>;

//*******************************************************************
//           STYPE/ALU
//*******************************************************************
def : T_P_pat <A2_absp, int_hexagon_A2_absp>;
def : T_P_pat <A2_negp, int_hexagon_A2_negp>;
def : T_P_pat <A2_notp, int_hexagon_A2_notp>;

//*******************************************************************
//           STYPE/BIT
//*******************************************************************

// Count leading/trailing
def: T_R_pat<S2_cl0,     int_hexagon_S2_cl0>;
def: T_P_pat<S2_cl0p,    int_hexagon_S2_cl0p>;
def: T_R_pat<S2_cl1,     int_hexagon_S2_cl1>;
def: T_P_pat<S2_cl1p,    int_hexagon_S2_cl1p>;
def: T_R_pat<S2_clb,     int_hexagon_S2_clb>;
def: T_P_pat<S2_clbp,    int_hexagon_S2_clbp>;
def: T_R_pat<S2_clbnorm, int_hexagon_S2_clbnorm>;
def: T_R_pat<S2_ct0,     int_hexagon_S2_ct0>;
def: T_R_pat<S2_ct1,     int_hexagon_S2_ct1>;

// Compare bit mask
def: T_RR_pat<C2_bitsclr,  int_hexagon_C2_bitsclr>;
def: T_RI_pat<C2_bitsclri, int_hexagon_C2_bitsclri>;
def: T_RR_pat<C2_bitsset,  int_hexagon_C2_bitsset>;

// Vector shuffle
def : T_PP_pat <S2_shuffeb, int_hexagon_S2_shuffeb>;
def : T_PP_pat <S2_shuffob, int_hexagon_S2_shuffob>;
def : T_PP_pat <S2_shuffeh, int_hexagon_S2_shuffeh>;
def : T_PP_pat <S2_shuffoh, int_hexagon_S2_shuffoh>;

// Vector truncate
def : T_PP_pat <S2_vtrunewh, int_hexagon_S2_vtrunewh>;
def : T_PP_pat <S2_vtrunowh, int_hexagon_S2_vtrunowh>;

// Linear feedback-shift Iteration.
def : T_PP_pat <S2_lfsp, int_hexagon_S2_lfsp>;

// Vector align
// Need custom lowering
def : T_PPQ_pat <S2_valignrb, int_hexagon_S2_valignrb>;
def : T_PPI_pat <S2_valignib, int_hexagon_S2_valignib>;

// Vector splice
def : T_PPQ_pat <S2_vsplicerb, int_hexagon_S2_vsplicerb>;
def : T_PPI_pat <S2_vspliceib, int_hexagon_S2_vspliceib>;

// Shift by immediate and add
def : T_RRI_pat<S2_addasl_rrri, int_hexagon_S2_addasl_rrri>;

// Extract bitfield
def : T_PII_pat<S2_extractup,    int_hexagon_S2_extractup>;
def : T_RII_pat<S2_extractu,     int_hexagon_S2_extractu>;
def : T_RP_pat <S2_extractu_rp,  int_hexagon_S2_extractu_rp>;
def : T_PP_pat <S2_extractup_rp, int_hexagon_S2_extractup_rp>;

// Insert bitfield
def : Pat <(int_hexagon_S2_insert_rp I32:$src1, I32:$src2, I64:$src3),
           (S2_insert_rp I32:$src1, I32:$src2, I64:$src3)>;

def : Pat<(i64 (int_hexagon_S2_insertp_rp I64:$src1, I64:$src2, I64:$src3)),
          (i64 (S2_insertp_rp I64:$src1, I64:$src2, I64:$src3))>;

def : Pat<(int_hexagon_S2_insert I32:$src1, I32:$src2,
                                 u5_0ImmPred:$src3, u5_0ImmPred:$src4),
          (S2_insert I32:$src1, I32:$src2,
                     u5_0ImmPred:$src3, u5_0ImmPred:$src4)>;

def : Pat<(i64 (int_hexagon_S2_insertp I64:$src1, I64:$src2,
                                       u6_0ImmPred:$src3, u6_0ImmPred:$src4)),
          (i64 (S2_insertp I64:$src1, I64:$src2,
                           u6_0ImmPred:$src3, u6_0ImmPred:$src4))>;

// Innterleave/deinterleave
def : T_P_pat <S2_interleave, int_hexagon_S2_interleave>;
def : T_P_pat <S2_deinterleave, int_hexagon_S2_deinterleave>;

// Set/Clear/Toggle Bit
def: T_RI_pat<S2_setbit_i,    int_hexagon_S2_setbit_i>;
def: T_RI_pat<S2_clrbit_i,    int_hexagon_S2_clrbit_i>;
def: T_RI_pat<S2_togglebit_i, int_hexagon_S2_togglebit_i>;

def: T_RR_pat<S2_setbit_r,    int_hexagon_S2_setbit_r>;
def: T_RR_pat<S2_clrbit_r,    int_hexagon_S2_clrbit_r>;
def: T_RR_pat<S2_togglebit_r, int_hexagon_S2_togglebit_r>;

// Test Bit
def: T_Q_RI_pat<S2_tstbit_i,  int_hexagon_S2_tstbit_i>;
def: T_Q_RR_pat<S2_tstbit_r,  int_hexagon_S2_tstbit_r>;

//*******************************************************************
//           STYPE/COMPLEX
//*******************************************************************
// Vector Complex conjugate
def : T_P_pat <A2_vconj, int_hexagon_A2_vconj>;

// Vector Complex rotate
def : T_PR_pat <S2_vcrotate, int_hexagon_S2_vcrotate>;

//*******************************************************************
//           STYPE/PERM
//*******************************************************************

// Vector saturate without pack
def : T_P_pat <S2_vsathb_nopack, int_hexagon_S2_vsathb_nopack>;
def : T_P_pat <S2_vsathub_nopack, int_hexagon_S2_vsathub_nopack>;
def : T_P_pat <S2_vsatwh_nopack, int_hexagon_S2_vsatwh_nopack>;
def : T_P_pat <S2_vsatwuh_nopack, int_hexagon_S2_vsatwuh_nopack>;

//*******************************************************************
//           STYPE/PRED
//*******************************************************************

// Predicate transfer
def: Pat<(i32 (int_hexagon_C2_tfrpr I32:$Rs)),
         (i32 (C2_tfrpr (C2_tfrrp I32:$Rs)))>;
def: Pat<(i32 (int_hexagon_C2_tfrrp I32:$Rs)),
         (i32 (C2_tfrpr (C2_tfrrp I32:$Rs)))>;

// Mask generate from predicate
def: Pat<(i64 (int_hexagon_C2_mask I32:$Rs)),
         (i64 (C2_mask (C2_tfrrp I32:$Rs)))>;

// Viterbi pack even and odd predicate bits
def: T_QQ_pat<C2_vitpack, int_hexagon_C2_vitpack>;

//*******************************************************************
//           STYPE/SHIFT
//*******************************************************************

def : T_PI_pat <S2_asr_i_p, int_hexagon_S2_asr_i_p>;
def : T_PI_pat <S2_lsr_i_p, int_hexagon_S2_lsr_i_p>;
def : T_PI_pat <S2_asl_i_p, int_hexagon_S2_asl_i_p>;

def : T_PR_pat <S2_asr_r_p, int_hexagon_S2_asr_r_p>;
def : T_PR_pat <S2_lsr_r_p, int_hexagon_S2_lsr_r_p>;
def : T_PR_pat <S2_asl_r_p, int_hexagon_S2_asl_r_p>;
def : T_PR_pat <S2_lsl_r_p, int_hexagon_S2_lsl_r_p>;

def : T_RR_pat <S2_asr_r_r, int_hexagon_S2_asr_r_r>;
def : T_RR_pat <S2_lsr_r_r, int_hexagon_S2_lsr_r_r>;
def : T_RR_pat <S2_asl_r_r, int_hexagon_S2_asl_r_r>;
def : T_RR_pat <S2_lsl_r_r, int_hexagon_S2_lsl_r_r>;

def : T_RR_pat <S2_asr_r_r_sat, int_hexagon_S2_asr_r_r_sat>;
def : T_RR_pat <S2_asl_r_r_sat, int_hexagon_S2_asl_r_r_sat>;

def : T_R_pat <S2_vsxtbh,   int_hexagon_S2_vsxtbh>;
def : T_R_pat <S2_vzxtbh,   int_hexagon_S2_vzxtbh>;
def : T_R_pat <S2_vsxthw,   int_hexagon_S2_vsxthw>;
def : T_R_pat <S2_vzxthw,   int_hexagon_S2_vzxthw>;
def : T_R_pat <S2_vsplatrh, int_hexagon_S2_vsplatrh>;
def : T_R_pat <A2_sxtw,     int_hexagon_A2_sxtw>;

// Vector saturate and pack
def : T_R_pat <S2_svsathb,  int_hexagon_S2_svsathb>;
def : T_R_pat <S2_svsathub, int_hexagon_S2_svsathub>;
def : T_P_pat <S2_vsathub,  int_hexagon_S2_vsathub>;
def : T_P_pat <S2_vsatwh,   int_hexagon_S2_vsatwh>;
def : T_P_pat <S2_vsatwuh,  int_hexagon_S2_vsatwuh>;
def : T_P_pat <S2_vsathb,   int_hexagon_S2_vsathb>;

def : T_P_pat <S2_vtrunohb,    int_hexagon_S2_vtrunohb>;
def : T_P_pat <S2_vtrunehb,    int_hexagon_S2_vtrunehb>;
def : T_P_pat <S2_vrndpackwh,  int_hexagon_S2_vrndpackwh>;
def : T_P_pat <S2_vrndpackwhs, int_hexagon_S2_vrndpackwhs>;
def : T_R_pat <S2_brev,        int_hexagon_S2_brev>;
def : T_R_pat <S2_vsplatrb,    int_hexagon_S2_vsplatrb>;

def : T_R_pat <A2_abs,    int_hexagon_A2_abs>;
def : T_R_pat <A2_abssat, int_hexagon_A2_abssat>;
def : T_R_pat <A2_negsat, int_hexagon_A2_negsat>;

def : T_R_pat <A2_swiz,   int_hexagon_A2_swiz>;

def : T_P_pat <A2_sat,    int_hexagon_A2_sat>;
def : T_R_pat <A2_sath,   int_hexagon_A2_sath>;
def : T_R_pat <A2_satuh,  int_hexagon_A2_satuh>;
def : T_R_pat <A2_satub,  int_hexagon_A2_satub>;
def : T_R_pat <A2_satb,   int_hexagon_A2_satb>;

// Vector arithmetic shift right by immediate with truncate and pack.
def : T_PI_pat<S2_asr_i_svw_trun, int_hexagon_S2_asr_i_svw_trun>;

def : T_RI_pat <S2_asr_i_r,     int_hexagon_S2_asr_i_r>;
def : T_RI_pat <S2_lsr_i_r,     int_hexagon_S2_lsr_i_r>;
def : T_RI_pat <S2_asl_i_r,     int_hexagon_S2_asl_i_r>;
def : T_RI_pat <S2_asr_i_r_rnd, int_hexagon_S2_asr_i_r_rnd>;
def : T_RI_pat <S2_asr_i_r_rnd_goodsyntax,
                int_hexagon_S2_asr_i_r_rnd_goodsyntax>;

// Shift left by immediate with saturation.
def : T_RI_pat <S2_asl_i_r_sat, int_hexagon_S2_asl_i_r_sat>;

//===----------------------------------------------------------------------===//
// Template 'def pat' to map tableidx[bhwd] intrinsics to :raw instructions.
//===----------------------------------------------------------------------===//
class S2op_tableidx_pat <Intrinsic IntID, InstHexagon OutputInst,
                         SDNodeXForm XformImm>
  : Pat <(IntID I32:$src1, I32:$src2, u4_0ImmPred:$src3, u5_0ImmPred:$src4),
         (OutputInst I32:$src1, I32:$src2, u4_0ImmPred:$src3,
                     (XformImm u5_0ImmPred:$src4))>;

def SDEC2 : SDNodeXForm<imm, [{
  int32_t V = N->getSExtValue();
  return CurDAG->getTargetConstant(V-2, SDLoc(N), MVT::i32);
}]>;

def SDEC3 : SDNodeXForm<imm, [{
  int32_t V = N->getSExtValue();
  return CurDAG->getTargetConstant(V-3, SDLoc(N), MVT::i32);
}]>;

// Table Index : Extract and insert bits.
// Map to the real hardware instructions after subtracting appropriate
// values from the 4th input operand. Please note that subtraction is not
// needed for int_hexagon_S2_tableidxb_goodsyntax.

def : Pat <(int_hexagon_S2_tableidxb_goodsyntax I32:$src1, I32:$src2,
                                              u4_0ImmPred:$src3, u5_0ImmPred:$src4),
           (S2_tableidxb I32:$src1, I32:$src2,
                         u4_0ImmPred:$src3, u5_0ImmPred:$src4)>;

def : S2op_tableidx_pat <int_hexagon_S2_tableidxh_goodsyntax, S2_tableidxh,
                         SDEC1>;
def : S2op_tableidx_pat <int_hexagon_S2_tableidxw_goodsyntax, S2_tableidxw,
                         SDEC2>;
def : S2op_tableidx_pat <int_hexagon_S2_tableidxd_goodsyntax, S2_tableidxd,
                         SDEC3>;

//*******************************************************************
//           STYPE/VH
//*******************************************************************

// Vector absolute value halfwords with and without saturation
// Rdd64=vabsh(Rss64)[:sat]
def : T_P_pat <A2_vabsh, int_hexagon_A2_vabsh>;
def : T_P_pat <A2_vabshsat, int_hexagon_A2_vabshsat>;

// Vector shift halfwords by immediate
// Rdd64=[vaslh/vasrh/vlsrh](Rss64,u4)
def : T_PI_pat <S2_asr_i_vh, int_hexagon_S2_asr_i_vh>;
def : T_PI_pat <S2_lsr_i_vh, int_hexagon_S2_lsr_i_vh>;
def : T_PI_pat <S2_asl_i_vh, int_hexagon_S2_asl_i_vh>;

// Vector shift halfwords by register
// Rdd64=[vaslw/vasrw/vlslw/vlsrw](Rss64,Rt32)
def : T_PR_pat <S2_asr_r_vh, int_hexagon_S2_asr_r_vh>;
def : T_PR_pat <S2_lsr_r_vh, int_hexagon_S2_lsr_r_vh>;
def : T_PR_pat <S2_asl_r_vh, int_hexagon_S2_asl_r_vh>;
def : T_PR_pat <S2_lsl_r_vh, int_hexagon_S2_lsl_r_vh>;

//*******************************************************************
//           STYPE/VW
//*******************************************************************

// Vector absolute value words with and without saturation
def : T_P_pat <A2_vabsw, int_hexagon_A2_vabsw>;
def : T_P_pat <A2_vabswsat, int_hexagon_A2_vabswsat>;

// Vector shift words by immediate.
// Rdd64=[vasrw/vlsrw|vaslw](Rss64,u5)
def : T_PI_pat <S2_asr_i_vw, int_hexagon_S2_asr_i_vw>;
def : T_PI_pat <S2_lsr_i_vw, int_hexagon_S2_lsr_i_vw>;
def : T_PI_pat <S2_asl_i_vw, int_hexagon_S2_asl_i_vw>;

// Vector shift words by register.
// Rdd64=[vasrw/vlsrw|vaslw|vlslw](Rss64,Rt32)
def : T_PR_pat <S2_asr_r_vw, int_hexagon_S2_asr_r_vw>;
def : T_PR_pat <S2_lsr_r_vw, int_hexagon_S2_lsr_r_vw>;
def : T_PR_pat <S2_asl_r_vw, int_hexagon_S2_asl_r_vw>;
def : T_PR_pat <S2_lsl_r_vw, int_hexagon_S2_lsl_r_vw>;

// Vector shift words with truncate and pack
def : T_PR_pat <S2_asr_r_svw_trun, int_hexagon_S2_asr_r_svw_trun>;

// Load/store locked.
def : T_R_pat<L2_loadw_locked, int_hexagon_L2_loadw_locked>;
def : T_R_pat<L4_loadd_locked, int_hexagon_L4_loadd_locked>;

def : Pat<(int_hexagon_S2_storew_locked I32:$Rs, I32:$Rt),
          (C2_tfrpr (S2_storew_locked I32:$Rs, I32:$Rt))>;
def : Pat<(int_hexagon_S4_stored_locked I32:$Rs, I64:$Rt),
          (C2_tfrpr (S4_stored_locked I32:$Rs, I64:$Rt))>;

//*******************************************************************
//           ST
//*******************************************************************

class T_stb_pat <InstHexagon MI, Intrinsic IntID, PatLeaf Val>
  : Pat<(IntID I32:$Rs, Val:$Rt, I32:$Ru),
        (MI I32:$Rs, I32:$Ru, Val:$Rt)>;

def : T_stb_pat <S2_storerh_pbr, int_hexagon_brev_sth,   I32>;
def : T_stb_pat <S2_storerb_pbr, int_hexagon_brev_stb,   I32>;
def : T_stb_pat <S2_storeri_pbr, int_hexagon_brev_stw,   I32>;
def : T_stb_pat <S2_storerf_pbr, int_hexagon_brev_sthhi, I32>;
def : T_stb_pat <S2_storerd_pbr, int_hexagon_brev_std,   I64>;

class T_stc_pat <InstHexagon MI, Intrinsic IntID, PatLeaf Imm, PatLeaf Val>
  : Pat<(IntID I32:$Rs, Val:$Rt, I32:$Ru, Imm:$s),
        (MI I32:$Rs, Imm:$s, I32:$Ru, Val:$Rt)>;

def: T_stc_pat<S2_storerb_pci, int_hexagon_circ_stb,   s4_0ImmPred, I32>;
def: T_stc_pat<S2_storerh_pci, int_hexagon_circ_sth,   s4_1ImmPred, I32>;
def: T_stc_pat<S2_storeri_pci, int_hexagon_circ_stw,   s4_2ImmPred, I32>;
def: T_stc_pat<S2_storerd_pci, int_hexagon_circ_std,   s4_3ImmPred, I64>;
def: T_stc_pat<S2_storerf_pci, int_hexagon_circ_sthhi, s4_1ImmPred, I32>;

multiclass MaskedStore <InstHexagon MI, Intrinsic IntID> {
  def : Pat<(IntID HvxQR:$src1, IntRegs:$src2, HvxVR:$src3),
            (MI HvxQR:$src1, IntRegs:$src2, #0, HvxVR:$src3)>;
  def : Pat<(!cast<Intrinsic>(IntID#"_128B") HvxQR:$src1, IntRegs:$src2,
                                             HvxVR:$src3),
            (MI HvxQR:$src1, IntRegs:$src2, #0, HvxVR:$src3)>;
}

defm : MaskedStore <V6_vS32b_qpred_ai, int_hexagon_V6_vmaskedstoreq>;
defm : MaskedStore <V6_vS32b_nqpred_ai, int_hexagon_V6_vmaskedstorenq>;
defm : MaskedStore <V6_vS32b_nt_qpred_ai, int_hexagon_V6_vmaskedstorentq>;
defm : MaskedStore <V6_vS32b_nt_nqpred_ai, int_hexagon_V6_vmaskedstorentnq>;

//*******************************************************************
//           SYSTEM
//*******************************************************************

def: T_R_pat<Y2_dccleana,    int_hexagon_Y2_dccleana>;
def: T_R_pat<Y2_dccleaninva, int_hexagon_Y2_dccleaninva>;
def: T_R_pat<Y2_dcinva,      int_hexagon_Y2_dcinva>;
def: T_R_pat<Y2_dczeroa,     int_hexagon_Y2_dczeroa>;

def: T_RR_pat<Y4_l2fetch,    int_hexagon_Y4_l2fetch>;
def: T_RP_pat<Y5_l2fetch,    int_hexagon_Y5_l2fetch>;

include "HexagonIntrinsicsV3.td"
include "HexagonIntrinsicsV4.td"
include "HexagonIntrinsicsV5.td"
include "HexagonIntrinsicsV60.td"
