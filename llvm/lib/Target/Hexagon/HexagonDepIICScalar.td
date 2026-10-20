//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

def tc_A2_abs : InstrItinClass;
def tc_A2_add : InstrItinClass;
def tc_A2_addh_h16_hh : InstrItinClass;
def tc_A2_addh_h16_sat_hh : InstrItinClass;
def tc_A2_addh_l16_hl : InstrItinClass;
def tc_A2_addp : InstrItinClass;
def tc_A2_addsat : InstrItinClass;
def tc_A2_aslh : InstrItinClass;
def tc_A2_negp : InstrItinClass;
def tc_A2_nop : InstrItinClass;
def tc_A2_paddf : InstrItinClass;
def tc_A2_paddfnew : InstrItinClass;
def tc_A2_svavgh : InstrItinClass;
def tc_A2_svavghs : InstrItinClass;
def tc_A2_tfrcrr : InstrItinClass;
def tc_A2_tfrpfnew : InstrItinClass;
def tc_A2_tfrrcr : InstrItinClass;
def tc_A2_vavgh : InstrItinClass;
def tc_A2_vavghcr : InstrItinClass;
def tc_A2_vavghr : InstrItinClass;
def tc_A2_vcmpbeq : InstrItinClass;
def tc_A2_vraddub : InstrItinClass;
def tc_A2_vraddub_acc : InstrItinClass;
def tc_A4_addp_c : InstrItinClass;
def tc_A4_cmpbeqi : InstrItinClass;
def tc_A4_ext : InstrItinClass;
def tc_A4_tlbmatch : InstrItinClass;
def tc_A4_vrmaxh : InstrItinClass;
def tc_A5_ACS : InstrItinClass;
def tc_A6_vminub_RdP : InstrItinClass;
def tc_C2_all8 : InstrItinClass;
def tc_C2_and : InstrItinClass;
def tc_C2_cmpeq : InstrItinClass;
def tc_C2_cmpeqi : InstrItinClass;
def tc_C2_tfrrp : InstrItinClass;
def tc_C2_vmux : InstrItinClass;
def tc_C4_addipc : InstrItinClass;
def tc_C4_and_and : InstrItinClass;
def tc_F2_conv_d2df : InstrItinClass;
def tc_F2_dfadd : InstrItinClass;
def tc_F2_dfimm_n : InstrItinClass;
def tc_F2_dfmpyhh : InstrItinClass;
def tc_F2_sfadd : InstrItinClass;
def tc_F2_sffma : InstrItinClass;
def tc_F2_sffma_sc : InstrItinClass;
def tc_F2_sfinvsqrta : InstrItinClass;
def tc_F2_sfmax : InstrItinClass;
def tc_F2_sfrecipa : InstrItinClass;
def tc_G4_tfrgcpp : InstrItinClass;
def tc_G4_tfrgpcp : InstrItinClass;
def tc_J2_call : InstrItinClass;
def tc_J2_callf : InstrItinClass;
def tc_J2_callr : InstrItinClass;
def tc_J2_callrf : InstrItinClass;
def tc_J2_callrh : InstrItinClass;
def tc_J2_endloop0 : InstrItinClass;
def tc_J2_jump : InstrItinClass;
def tc_J2_jumpf : InstrItinClass;
def tc_J2_jumpfnew : InstrItinClass;
def tc_J2_jumpr : InstrItinClass;
def tc_J2_jumprf : InstrItinClass;
def tc_J2_jumprfnew : InstrItinClass;
def tc_J2_jumprgtez : InstrItinClass;
def tc_J2_jumprh : InstrItinClass;
def tc_J2_loop0i : InstrItinClass;
def tc_J2_loop0r : InstrItinClass;
def tc_J2_pause : InstrItinClass;
def tc_J2_ploop1si : InstrItinClass;
def tc_J2_ploop1sr : InstrItinClass;
def tc_J2_rte : InstrItinClass;
def tc_J2_trap0 : InstrItinClass;
def tc_J2_trap1 : InstrItinClass;
def tc_J2_unpause : InstrItinClass;
def tc_J4_cmpeq_f_jumpnv_nt : InstrItinClass;
def tc_J4_cmpeq_fp0_jump_nt : InstrItinClass;
def tc_J4_cmpeqi_f_jumpnv_nt : InstrItinClass;
def tc_J4_cmpeqi_fp0_jump_nt : InstrItinClass;
def tc_J4_cmplt_f_jumpnv_nt : InstrItinClass;
def tc_J4_hintjumpr : InstrItinClass;
def tc_J4_jumpseti : InstrItinClass;
def tc_J4_jumpsetr : InstrItinClass;
def tc_J4_tstbit0_f_jumpnv_nt : InstrItinClass;
def tc_J4_tstbit0_fp0_jump_nt : InstrItinClass;
def tc_L2_deallocframe : InstrItinClass;
def tc_L2_loadalignb_io : InstrItinClass;
def tc_L2_loadalignb_pbr : InstrItinClass;
def tc_L2_loadalignb_pci : InstrItinClass;
def tc_L2_loadbsw2_io : InstrItinClass;
def tc_L2_loadbsw2_pbr : InstrItinClass;
def tc_L2_loadbsw2_pci : InstrItinClass;
def tc_L2_loadrbgp : InstrItinClass;
def tc_L2_loadw_locked : InstrItinClass;
def tc_L2_ploadrbfnew_pi : InstrItinClass;
def tc_L4_add_memopb_io : InstrItinClass;
def tc_L4_iadd_memopb_io : InstrItinClass;
def tc_L4_loadalignb_ap : InstrItinClass;
def tc_L4_loadalignb_ur : InstrItinClass;
def tc_L4_loadbsw2_ap : InstrItinClass;
def tc_L4_loadbsw2_ur : InstrItinClass;
def tc_L4_loadrb_rr : InstrItinClass;
def tc_L4_loadw_phys : InstrItinClass;
def tc_L4_ploadrbf_abs : InstrItinClass;
def tc_L4_ploadrbf_rr : InstrItinClass;
def tc_L4_ploadrbfnew_rr : InstrItinClass;
def tc_L4_return : InstrItinClass;
def tc_L4_return_f : InstrItinClass;
def tc_L4_return_fnew_pnt : InstrItinClass;
def tc_L4_return_map_to_raw_t : InstrItinClass;
def tc_L4_return_map_to_raw_tnew_pnt : InstrItinClass;
def tc_L6_memcpy : InstrItinClass;
def tc_M2_acci : InstrItinClass;
def tc_M2_macsin : InstrItinClass;
def tc_M2_mpysin : InstrItinClass;
def tc_M2_xor_xacc : InstrItinClass;
def tc_M4_mpyri_addr_u2 : InstrItinClass;
def tc_M7_dcmpyiw : InstrItinClass;
def tc_M7_dcmpyiw_acc : InstrItinClass;
def tc_PS_storerbabs : InstrItinClass;
def tc_PS_storerbnewabs : InstrItinClass;
def tc_S2_allocframe : InstrItinClass;
def tc_S2_brev : InstrItinClass;
def tc_S2_cabacdecbin : InstrItinClass;
def tc_S2_extractu_rp : InstrItinClass;
def tc_S2_insert : InstrItinClass;
def tc_S2_pstorerbf_io : InstrItinClass;
def tc_S2_pstorerbf_pi : InstrItinClass;
def tc_S2_pstorerbf_pi_nt : InstrItinClass;
def tc_S2_pstorerbfnew_pi : InstrItinClass;
def tc_S2_pstorerbnewf_io : InstrItinClass;
def tc_S2_pstorerbnewf_pi : InstrItinClass;
def tc_S2_pstorerbnewfnew_pi : InstrItinClass;
def tc_S2_storerb_io : InstrItinClass;
def tc_S2_storerb_pbr : InstrItinClass;
def tc_S2_storerb_pci : InstrItinClass;
def tc_S2_storerb_pi_nt : InstrItinClass;
def tc_S2_storerbnew_io : InstrItinClass;
def tc_S2_storerbnew_pbr : InstrItinClass;
def tc_S2_storerbnew_pci : InstrItinClass;
def tc_S2_storew_locked : InstrItinClass;
def tc_S2_storew_rl_at_vi : InstrItinClass;
def tc_S2_vrndpackwh : InstrItinClass;
def tc_S4_pstorerbf_abs : InstrItinClass;
def tc_S4_pstorerbf_rr : InstrItinClass;
def tc_S4_pstorerbfnew_abs : InstrItinClass;
def tc_S4_pstorerbfnew_rr : InstrItinClass;
def tc_S4_pstorerbnewf_abs : InstrItinClass;
def tc_S4_pstorerbnewf_rr : InstrItinClass;
def tc_S4_pstorerbnewfnew_abs : InstrItinClass;
def tc_S4_pstorerbnewfnew_rr : InstrItinClass;
def tc_S4_storeirb_io : InstrItinClass;
def tc_S4_storeirbf_io : InstrItinClass;
def tc_S4_storeirbfnew_io : InstrItinClass;
def tc_S4_storerb_rr : InstrItinClass;
def tc_S4_storerb_ur : InstrItinClass;
def tc_S4_storerbnew_rr : InstrItinClass;
def tc_S4_storerbnew_ur : InstrItinClass;
def tc_S4_vrcrotate : InstrItinClass;
def tc_S4_vrcrotate_acc : InstrItinClass;
def tc_S5_popcountp : InstrItinClass;
def tc_SA1_addi : InstrItinClass;
def tc_SA1_addsp : InstrItinClass;
def tc_SA1_clrf : InstrItinClass;
def tc_SA1_clrfnew : InstrItinClass;
def tc_SA1_cmpeqi : InstrItinClass;
def tc_SL2_deallocframe : InstrItinClass;
def tc_SL2_jumpr31 : InstrItinClass;
def tc_SL2_return : InstrItinClass;
def tc_SS2_allocframe : InstrItinClass;
def tc_SS2_storebi0 : InstrItinClass;
def tc_Y2_barrier : InstrItinClass;
def tc_Y2_break : InstrItinClass;
def tc_Y2_ciad : InstrItinClass;
def tc_Y2_crswap0 : InstrItinClass;
def tc_Y2_dccleana : InstrItinClass;
def tc_Y2_dcfetch : InstrItinClass;
def tc_Y2_dcfetchbo : InstrItinClass;
def tc_Y2_dckill : InstrItinClass;
def tc_Y2_dctagr : InstrItinClass;
def tc_Y2_dctagw : InstrItinClass;
def tc_Y2_icdatar : InstrItinClass;
def tc_Y2_icdataw : InstrItinClass;
def tc_Y2_icinva : InstrItinClass;
def tc_Y2_ickill : InstrItinClass;
def tc_Y2_ictagw : InstrItinClass;
def tc_Y2_tlbp : InstrItinClass;
def tc_Y2_tlbpp : InstrItinClass;
def tc_Y2_tlbw : InstrItinClass;
def tc_Y4_crswap10 : InstrItinClass;
def tc_Y4_l2fetch : InstrItinClass;
def tc_Y4_l2tagw : InstrItinClass;
def tc_Y5_ctlbw : InstrItinClass;
def tc_Y5_l2locka : InstrItinClass;
def tc_Y5_tlbasidi : InstrItinClass;
def tc_Y6_diag0 : InstrItinClass;
def tc_Y6_dmpause : InstrItinClass;
def tc_Y6_l2gcleaninvpa : InstrItinClass;
def tc_dup_A2_add : InstrItinClass;
def tc_dup_A2_sxtb : InstrItinClass;
def tc_dup_C2_cmovenewif : InstrItinClass;
def tc_dup_C2_cmpeqi : InstrItinClass;
def tc_dup_L2_deallocframe : InstrItinClass;
def tc_dup_L2_loadrb_io : InstrItinClass;
def tc_dup_S2_allocframe : InstrItinClass;
def tc_dup_S2_storerb_io : InstrItinClass;
def tc_dup_S4_storeirb_io : InstrItinClass;

class DepScalarItinV5 {
  list<InstrItinData> DepScalarItinV5_list = [
  InstrItinData <tc_A2_abs, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_add, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_addh_h16_hh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_addh_h16_sat_hh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_addh_l16_hl, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_addp, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_addsat, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_aslh, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_negp, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_nop, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_paddf, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_paddfnew, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_svavgh, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_svavghs, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_tfrcrr, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_A2_tfrpfnew, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_tfrrcr, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_A2_vavgh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_vavghcr, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_vavghr, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_vcmpbeq, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_vraddub, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A2_vraddub_acc, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A4_addp_c, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A4_cmpbeqi, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A4_ext, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_A4_tlbmatch, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A4_vrmaxh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A5_ACS, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_A6_vminub_RdP, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_all8, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_and, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_cmpeq, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_cmpeqi, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_tfrrp, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_C2_vmux, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_C4_addipc, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_C4_and_and, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_conv_d2df, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_dfadd, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_dfimm_n, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_dfmpyhh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sfadd, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sffma, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sffma_sc, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sfinvsqrta, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sfmax, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_F2_sfrecipa, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_G4_tfrgcpp, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_G4_tfrgpcp, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_call, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_callf, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_callr, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_callrf, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_callrh, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_endloop0, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_jump, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_jumpf, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_jumpfnew, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J2_jumpr, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_jumprf, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_jumprfnew, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_jumprgtez, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_jumprh, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_loop0i, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_loop0r, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_pause, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_ploop1si, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_ploop1sr, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_J2_rte, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_trap0, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_trap1, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J2_unpause, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_J4_cmpeq_fp0_jump_nt, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J4_cmplt_f_jumpnv_nt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_J4_hintjumpr, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_J4_jumpseti, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J4_jumpsetr, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_J4_tstbit0_fp0_jump_nt, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_L2_deallocframe, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadalignb_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadalignb_pbr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadalignb_pci, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadbsw2_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadbsw2_pbr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadbsw2_pci, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadrbgp, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L2_loadw_locked, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L2_ploadrbfnew_pi, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_add_memopb_io, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_iadd_memopb_io, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_loadalignb_ap, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_loadalignb_ur, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_loadbsw2_ap, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_loadbsw2_ur, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_loadrb_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_loadw_phys, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_ploadrbf_abs, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_ploadrbf_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_ploadrbfnew_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_L4_return, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_return_f, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_return_fnew_pnt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_return_map_to_raw_t, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_L6_memcpy, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_M2_acci, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_M2_macsin, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_M2_mpysin, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_M2_xor_xacc, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_M4_mpyri_addr_u2, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_M7_dcmpyiw, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_M7_dcmpyiw_acc, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_PS_storerbabs, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_PS_storerbnewabs, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_allocframe, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_brev, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S2_cabacdecbin, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S2_extractu_rp, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S2_insert, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S2_pstorerbf_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_pstorerbf_pi, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_pstorerbf_pi_nt, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_pstorerbfnew_pi, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_pstorerbnewf_io, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_pstorerbnewf_pi, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_pstorerbnewfnew_pi, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_storerb_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_storerb_pbr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_storerb_pci, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_storerb_pi_nt, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S2_storerbnew_io, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_storerbnew_pbr, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_storerbnew_pci, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_storew_locked, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_storew_rl_at_vi, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S2_vrndpackwh, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S4_pstorerbf_abs, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_pstorerbf_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_pstorerbfnew_abs, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_pstorerbfnew_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_pstorerbnewf_abs, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_pstorerbnewf_rr, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_pstorerbnewfnew_abs, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_pstorerbnewfnew_rr, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_storeirb_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_storeirbf_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_storeirbfnew_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_storerb_rr, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_storerb_ur, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_S4_storerbnew_rr, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_storerbnew_ur, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_S4_vrcrotate, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S4_vrcrotate_acc, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_S5_popcountp, [InstrStage<1, [SLOT2, SLOT3]>]>,
  InstrItinData <tc_SA1_addi, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SA1_addsp, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SA1_clrf, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SA1_clrfnew, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SA1_cmpeqi, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SL2_deallocframe, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_SL2_jumpr31, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_SL2_return, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_SS2_allocframe, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_SS2_storebi0, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_Y2_barrier, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_break, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y2_ciad, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y2_crswap0, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y2_dccleana, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_dcfetch, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_dcfetchbo, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_dckill, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_dctagr, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_dctagw, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y2_icdatar, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_Y2_icdataw, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_Y2_icinva, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_Y2_ickill, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_Y2_ictagw, [InstrStage<1, [SLOT2]>]>,
  InstrItinData <tc_Y2_tlbp, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y2_tlbpp, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y2_tlbw, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y4_crswap10, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y4_l2fetch, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y4_l2tagw, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y5_ctlbw, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y5_l2locka, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y5_tlbasidi, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y6_diag0, [InstrStage<1, [SLOT3]>]>,
  InstrItinData <tc_Y6_dmpause, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_Y6_l2gcleaninvpa, [InstrStage<1, [SLOT0]>]>,
  InstrItinData <tc_dup_A2_add, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_dup_A2_sxtb, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_dup_C2_cmovenewif, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_dup_C2_cmpeqi, [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>]>,
  InstrItinData <tc_dup_L2_deallocframe, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_dup_L2_loadrb_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_dup_S2_allocframe, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_dup_S2_storerb_io, [InstrStage<1, [SLOT0, SLOT1]>]>,
  InstrItinData <tc_dup_S4_storeirb_io, [InstrStage<1, [SLOT0, SLOT1]>]>  ];
}

class DepScalarItinV55 {
  list<InstrItinData> DepScalarItinV55_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV60 {
  list<InstrItinData> DepScalarItinV60_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV60se {
  list<InstrItinData> DepScalarItinV60se_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 1, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_2early*/
      [InstrStage<1, [SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3], 0>,
       InstrStage<1, [CVI_ST]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0], 0>,
       InstrStage<1, [CVI_ST]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV62 {
  list<InstrItinData> DepScalarItinV62_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_2early*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_2early*/
      [InstrStage<1, [SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_2early*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV65 {
  list<InstrItinData> DepScalarItinV65_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV66 {
  list<InstrItinData> DepScalarItinV66_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV67 {
  list<InstrItinData> DepScalarItinV67_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV67T {
  list<InstrItinData> DepScalarItinV67T_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV68 {
  list<InstrItinData> DepScalarItinV68_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV69 {
  list<InstrItinData> DepScalarItinV69_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV71 {
  list<InstrItinData> DepScalarItinV71_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV71T {
  list<InstrItinData> DepScalarItinV71T_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV73 {
  list<InstrItinData> DepScalarItinV73_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV75 {
  list<InstrItinData> DepScalarItinV75_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV79 {
  list<InstrItinData> DepScalarItinV79_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}

class DepScalarItinV81 {
  list<InstrItinData> DepScalarItinV81_list = [
    InstrItinData <tc_A2_abs, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_h16_sat_hh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addh_l16_hl, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_addsat, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_aslh, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_negp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_nop, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_A2_paddf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_paddfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavgh, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_svavghs, /*tc_2*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrcrr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrpfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_tfrrcr, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavgh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghcr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vavghr, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vcmpbeq, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A2_vraddub_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_addp_c, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_cmpbeqi, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_ext, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_A4_tlbmatch, /*tc_3stall*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A4_vrmaxh, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A5_ACS, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_A6_vminub_RdP, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_all8, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeq, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_tfrrp, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C2_vmux, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_addipc, /*tc_2*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_C4_and_and, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_conv_d2df, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfimm_n, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_dfmpyhh, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfadd, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sffma_sc, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfinvsqrta, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfmax, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_F2_sfrecipa, /*tc_4x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [5, 5, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgcpp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_G4_tfrgpcp, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_call, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callf, /*tc_3*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callr, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_callrf, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_callrh, /*tc_3*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_J2_endloop0, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [],
      []>,

    InstrItinData <tc_J2_jump, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumpf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpfnew, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumpr, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_jumprf, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprfnew, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprgtez, /*tc_1*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_jumprh, /*tc_1*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_loop0i, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_loop0r, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_pause, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_ploop1si, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_ploop1sr, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [2, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_rte, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J2_trap0, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J2_trap1, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J2_unpause, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_J4_cmpeq_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeq_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmpeqi_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_cmplt_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_hintjumpr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_J4_jumpseti, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_jumpsetr, /*tc_1*/
      [InstrStage<1, [SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_f_jumpnv_nt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_J4_tstbit0_fp0_jump_nt, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadalignb_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pbr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadbsw2_pci, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadrbgp, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_loadw_locked, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L2_ploadrbfnew_pi, /*tc_latepredldaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 4, 3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_add_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_iadd_memopb_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadalignb_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ap, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadbsw2_ur, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadrb_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_loadw_phys, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_abs, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbf_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_ploadrbfnew_rr, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_f, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_fnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3, 3, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_t, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_L4_return_map_to_raw_tnew_pnt, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_L6_memcpy, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_acci, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_macsin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_mpysin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M2_xor_xacc, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M4_mpyri_addr_u2, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_M7_dcmpyiw_acc, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 2, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbabs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_PS_storerbnewabs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_brev, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_cabacdecbin, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_extractu_rp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_insert, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 2, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbf_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 3, 1, 2, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewf_pi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_pstorerbnewfnew_pi, /*tc_latepredstaia*/
      [InstrStage<1, [SLOT0]>], [4, 3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pci, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerb_pi_nt, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pbr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storerbnew_pci, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_locked, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_storew_rl_at_vi, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S2_vrndpackwh, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewf_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [2, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_abs, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_pstorerbnewfnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3, 1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbf_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storeirbfnew_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_rr, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerb_ur, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_rr, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_storerbnew_ur, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 2, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S4_vrcrotate_acc, /*tc_3x*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2, 1, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_S5_popcountp, /*tc_2*/
      [InstrStage<1, [SLOT2, SLOT3]>], [4, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_addsp, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SA1_clrf, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_clrfnew, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4],
      [Hex_FWD]>,

    InstrItinData <tc_SA1_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1]>], [2, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_SL2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [],
      []>,

    InstrItinData <tc_SL2_jumpr31, /*tc_2early*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SL2_return, /*tc_newvjump*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_SS2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [2],
      [Hex_FWD]>,

    InstrItinData <tc_SS2_storebi0, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_barrier, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_break, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [],
      []>,

    InstrItinData <tc_Y2_ciad, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_crswap0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dccleana, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetch, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_dcfetchbo, /*tc_ld*/
      [InstrStage<1, [SLOT0]>], [1, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dckill, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [],
      []>,

    InstrItinData <tc_Y2_dctagr, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_dctagw, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdatar, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icdataw, /*tc_2early*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_icinva, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y2_ickill, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [],
      []>,

    InstrItinData <tc_Y2_ictagw, /*tc_3stall*/
      [InstrStage<1, [SLOT2]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbpp, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y2_tlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_crswap10, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2fetch, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y4_l2tagw, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [1, 3],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_ctlbw, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [4, 1, 1],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_l2locka, /*tc_3stall*/
      [InstrStage<1, [SLOT0]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y5_tlbasidi, /*tc_3stall*/
      [InstrStage<1, [SLOT3]>], [1],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_diag0, /*tc_3x*/
      [InstrStage<1, [SLOT3]>], [1, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_Y6_dmpause, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_Y6_l2gcleaninvpa, /*tc_st*/
      [InstrStage<1, [SLOT0]>], [3],
      [Hex_FWD]>,

    InstrItinData <tc_dup_A2_add, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_A2_sxtb, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmovenewif, /*tc_2latepred*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [4, 3, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_C2_cmpeqi, /*tc_1*/
      [InstrStage<1, [SLOT0, SLOT1, SLOT2, SLOT3]>], [3, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_deallocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1],
      [Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_L2_loadrb_io, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [4, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_allocframe, /*tc_ld*/
      [InstrStage<1, [SLOT0, SLOT1]>], [3, 1, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S2_storerb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 3],
      [Hex_FWD, Hex_FWD, Hex_FWD]>,

    InstrItinData <tc_dup_S4_storeirb_io, /*tc_st*/
      [InstrStage<1, [SLOT0, SLOT1]>], [1, 2, 2],
      [Hex_FWD, Hex_FWD, Hex_FWD]>
  ];
}
