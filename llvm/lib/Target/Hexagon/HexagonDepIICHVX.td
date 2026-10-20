//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

def tc_V6_extractw : InstrItinClass;
def tc_V6_lvsplatw : InstrItinClass;
def tc_V6_pred_and : InstrItinClass;
def tc_V6_pred_not : InstrItinClass;
def tc_V6_pred_scalar2 : InstrItinClass;
def tc_V6_v6mpyhubs10 : InstrItinClass;
def tc_V6_v6mpyhubs10_vxx : InstrItinClass;
def tc_V6_vL32Ub_ai : InstrItinClass;
def tc_V6_vL32Ub_pi : InstrItinClass;
def tc_V6_vL32b_ai : InstrItinClass;
def tc_V6_vL32b_cur_npred_ai : InstrItinClass;
def tc_V6_vL32b_cur_npred_pi : InstrItinClass;
def tc_V6_vL32b_cur_pi : InstrItinClass;
def tc_V6_vL32b_nt_tmp_ai : InstrItinClass;
def tc_V6_vL32b_nt_tmp_npred_ai : InstrItinClass;
def tc_V6_vL32b_nt_tmp_npred_pi : InstrItinClass;
def tc_V6_vL32b_nt_tmp_pi : InstrItinClass;
def tc_V6_vS32Ub_ai : InstrItinClass;
def tc_V6_vS32Ub_npred_ai : InstrItinClass;
def tc_V6_vS32Ub_npred_pi : InstrItinClass;
def tc_V6_vS32Ub_pi : InstrItinClass;
def tc_V6_vS32b_ai : InstrItinClass;
def tc_V6_vS32b_new_ai : InstrItinClass;
def tc_V6_vS32b_new_npred_ai : InstrItinClass;
def tc_V6_vS32b_new_npred_pi : InstrItinClass;
def tc_V6_vS32b_new_pi : InstrItinClass;
def tc_V6_vS32b_npred_ai : InstrItinClass;
def tc_V6_vS32b_npred_pi : InstrItinClass;
def tc_V6_vS32b_nqpred_ai : InstrItinClass;
def tc_V6_vS32b_nqpred_pi : InstrItinClass;
def tc_V6_vS32b_nt_pi : InstrItinClass;
def tc_V6_vS32b_srls_ai : InstrItinClass;
def tc_V6_vS32b_srls_pi : InstrItinClass;
def tc_V6_vabs_hf : InstrItinClass;
def tc_V6_vabsdiffh : InstrItinClass;
def tc_V6_vaddb : InstrItinClass;
def tc_V6_vaddbnq : InstrItinClass;
def tc_V6_vaddcarry : InstrItinClass;
def tc_V6_vaddcarryo : InstrItinClass;
def tc_V6_vaddhw : InstrItinClass;
def tc_V6_valign4 : InstrItinClass;
def tc_V6_valignb : InstrItinClass;
def tc_V6_vandqrt : InstrItinClass;
def tc_V6_vandqrt_acc : InstrItinClass;
def tc_V6_vaslh : InstrItinClass;
def tc_V6_vaslh_acc : InstrItinClass;
def tc_V6_vasr_into : InstrItinClass;
def tc_V6_vasrhbrndsat : InstrItinClass;
def tc_V6_vasrvuhubrndsat : InstrItinClass;
def tc_V6_vassign_tmp : InstrItinClass;
def tc_V6_vccombine : InstrItinClass;
def tc_V6_vcmov : InstrItinClass;
def tc_V6_vcombine_tmp : InstrItinClass;
def tc_V6_vcvt_h_hf : InstrItinClass;
def tc_V6_vcvt_hf_b : InstrItinClass;
def tc_V6_vdd0 : InstrItinClass;
def tc_V6_vdeal : InstrItinClass;
def tc_V6_vdealb : InstrItinClass;
def tc_V6_vdealb4w : InstrItinClass;
def tc_V6_vdealvdd : InstrItinClass;
def tc_V6_vdmpy_sf_hf_acc : InstrItinClass;
def tc_V6_vdmpybus : InstrItinClass;
def tc_V6_vdmpybus_acc : InstrItinClass;
def tc_V6_vdmpybus_dv : InstrItinClass;
def tc_V6_vdmpybus_dv_acc : InstrItinClass;
def tc_V6_vdmpyhsat : InstrItinClass;
def tc_V6_vdmpyhsat_acc : InstrItinClass;
def tc_V6_vdmpyhvsat : InstrItinClass;
def tc_V6_vdmpyhvsat_acc : InstrItinClass;
def tc_V6_vfmax_hf : InstrItinClass;
def tc_V6_vgathermh : InstrItinClass;
def tc_V6_vgathermhq : InstrItinClass;
def tc_V6_vgathermhw : InstrItinClass;
def tc_V6_vgathermhwq : InstrItinClass;
def tc_V6_vhist : InstrItinClass;
def tc_V6_vhistq : InstrItinClass;
def tc_V6_vlut4 : InstrItinClass;
def tc_V6_vlutvvb_oracc : InstrItinClass;
def tc_V6_vmpahhsat : InstrItinClass;
def tc_V6_vprefixqb : InstrItinClass;
def tc_V6_vrmpybub_rtt : InstrItinClass;
def tc_V6_vrmpybub_rtt_acc : InstrItinClass;
def tc_V6_vrmpybusi : InstrItinClass;
def tc_V6_vrmpybusi_acc : InstrItinClass;
def tc_V6_vrmpybusv_acc : InstrItinClass;
def tc_V6_vrmpyzbb_rt : InstrItinClass;
def tc_V6_vrmpyzbb_rt_acc : InstrItinClass;
def tc_V6_vrmpyzbb_rx : InstrItinClass;
def tc_V6_vrmpyzbb_rx_acc : InstrItinClass;
def tc_V6_vror : InstrItinClass;
def tc_V6_vsathub : InstrItinClass;
def tc_V6_vsb : InstrItinClass;
def tc_V6_vscattermh : InstrItinClass;
def tc_V6_vscattermhq : InstrItinClass;
def tc_V6_vscattermhw : InstrItinClass;
def tc_V6_vscattermhwq : InstrItinClass;
def tc_V6_vswap : InstrItinClass;
def tc_V6_vunpackb : InstrItinClass;
def tc_V6_vunpackob : InstrItinClass;
def tc_V6_vwhist128m : InstrItinClass;
def tc_V6_vwhist128qm : InstrItinClass;
def tc_V6_zLd_ai : InstrItinClass;
def tc_V6_zLd_pi : InstrItinClass;
def tc_V6_zLd_pred_ai : InstrItinClass;
def tc_V6_zLd_pred_pi : InstrItinClass;

class DepHVXItinV55 {
  list<InstrItinData> DepHVXItinV55_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV60 {
  list<InstrItinData> DepHVXItinV60_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV62 {
  list<InstrItinData> DepHVXItinV62_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV65 {
  list<InstrItinData> DepHVXItinV65_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV66 {
  list<InstrItinData> DepHVXItinV66_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV67 {
  list<InstrItinData> DepHVXItinV67_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT1,LOAD,VA*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT1,LOAD,VA_DV*/
      [InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV68 {
  list<InstrItinData> DepHVXItinV68_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV69 {
  list<InstrItinData> DepHVXItinV69_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV71 {
  list<InstrItinData> DepHVXItinV71_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV73 {
  list<InstrItinData> DepHVXItinV73_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV75 {
  list<InstrItinData> DepHVXItinV75_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV79 {
  list<InstrItinData> DepHVXItinV79_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepHVXItinV81 {
  list<InstrItinData> DepHVXItinV81_list = [
    InstrItinData <tc_V6_extractw, /*SLOT0,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [4, 7, 1],
      [Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_lvsplatw, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_pred_and, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_not, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_pred_scalar2, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_v6mpyhubs10_vxx, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_ai, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32Ub_pi, /*SLOT0,NOSLOT1,LOAD,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_ai, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_npred_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_cur_pi, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_ai, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_npred_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 2, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vL32b_nt_tmp_pi, /*SLOT01,LOAD*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD]>], [9, 3, 1, 2],
      [HVX_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32Ub_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_ai, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_npred_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32Ub_pi, /*SLOT0,NOSLOT1,STORE,VP*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [SLOT1], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_XLANE]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 5],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_npred_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_new_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2, 5],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_npred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 2, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_ai, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nqpred_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 7, 1, 2, 7],
      [Hex_FWD, HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_nt_pi, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [3, 1, 2, 7],
      [Hex_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_ai, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vS32b_srls_pi, /*SLOT0,STORE*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vabs_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vabsdiffh, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddb, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddbnq, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarry, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddcarryo, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vaddhw, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_valign4, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_valignb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vandqrt_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vaslh_acc, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasr_into, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vasrhbrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vasrvuhubrndsat, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vassign_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vccombine, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 2, 7, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcmov, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 2, 7],
      [HVX_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcombine_tmp, /*SLOT0123*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_h_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vcvt_hf_b, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdd0, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vdeal, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdealb, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealb4w, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdealvdd, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpy_sf_hf_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpybus, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpybus_dv_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhsat_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vdmpyhvsat_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vfmax_hf, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermh, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhq, /*SLOT01,LOAD,VA*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhw, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vgathermhwq, /*SLOT01,LOAD,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_LD], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vhist, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [],
      []>,

    InstrItinData <tc_V6_vhistq, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3],
      [HVX_FWD]>,

    InstrItinData <tc_V6_vlut4, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vlutvvb_oracc, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vmpahhsat, /*SLOT2,VX_DV*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vprefixqb, /*SLOT0123,VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_SHIFT]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybub_rtt_acc, /*SLOT23,VS_VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1], 0>,
       InstrStage<1, [CVI_SHIFT, CVI_XLANE]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 5, 2, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusi_acc, /*SLOT23,VX_DV*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01]>], [9, 7, 5, 2, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpybusv_acc, /*SLOT23,VX*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1]>], [9, 7, 5, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rt_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 7, 5, 2],
      [HVX_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vrmpyzbb_rx_acc, /*SLOT23,4SLOT_MPY*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL_NOMEM]>], [9, 3, 7, 5, 2],
      [HVX_FWD, Hex_FWD, HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vror, /*SLOT0123,VP*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLANE]>], [9, 5, 2],
      [HVX_FWD, HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_vsathub, /*SLOT0123,VA*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [9, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vsb, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermh, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhq, /*SLOT0,STORE,VA*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY0, CVI_MPY1, CVI_SHIFT, CVI_XLANE]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhw, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [1, 2, 7, 7],
      [Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vscattermhwq, /*SLOT0,STORE,VA_DV*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [7, 1, 2, 7, 7],
      [HVX_FWD, Hex_FWD, Hex_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vswap, /*SLOT0123,VA_DV*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_MPY01, CVI_XLSHF]>], [9, 7, 7, 7],
      [HVX_FWD, HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackb, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 5],
      [HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vunpackob, /*SLOT0123,VP_VS*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_XLSHF]>], [9, 7, 5],
      [HVX_FWD, HVX_FWD, HVX_FWD]>,

    InstrItinData <tc_V6_vwhist128m, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_V6_vwhist128qm, /*SLOT0123,4SLOT*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ALL]>], [3, 2],
      [HVX_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_ai, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_V6_zLd_pred_pi, /*SLOT01,ZW*/
      [InstrStage<1, [SLOT0, SLOT1], 0>,
       InstrStage<1, [CVI_ZW]>], [3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}
