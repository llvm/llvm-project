//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPTIMINGCLASSES_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPTIMINGCLASSES_H

#include "HexagonInstrInfo.h"

namespace llvm {

inline bool is_TC1(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_A2_add:
  case Hexagon::Sched::tc_A2_addp:
  case Hexagon::Sched::tc_A2_aslh:
  case Hexagon::Sched::tc_A2_negp:
  case Hexagon::Sched::tc_A2_nop:
  case Hexagon::Sched::tc_A2_paddf:
  case Hexagon::Sched::tc_A2_vcmpbeq:
  case Hexagon::Sched::tc_A4_addp_c:
  case Hexagon::Sched::tc_A4_cmpbeqi:
  case Hexagon::Sched::tc_A4_ext:
  case Hexagon::Sched::tc_C2_all8:
  case Hexagon::Sched::tc_C2_and:
  case Hexagon::Sched::tc_C2_cmpeq:
  case Hexagon::Sched::tc_C2_cmpeqi:
  case Hexagon::Sched::tc_C2_tfrrp:
  case Hexagon::Sched::tc_C2_vmux:
  case Hexagon::Sched::tc_C4_and_and:
  case Hexagon::Sched::tc_J2_endloop0:
  case Hexagon::Sched::tc_J2_jump:
  case Hexagon::Sched::tc_J2_jumpf:
  case Hexagon::Sched::tc_J2_jumpfnew:
  case Hexagon::Sched::tc_J2_jumpr:
  case Hexagon::Sched::tc_J2_jumprf:
  case Hexagon::Sched::tc_J2_jumprfnew:
  case Hexagon::Sched::tc_J2_jumprgtez:
  case Hexagon::Sched::tc_J2_jumprh:
  case Hexagon::Sched::tc_J4_cmpeq_fp0_jump_nt:
  case Hexagon::Sched::tc_J4_cmpeqi_fp0_jump_nt:
  case Hexagon::Sched::tc_J4_hintjumpr:
  case Hexagon::Sched::tc_J4_jumpseti:
  case Hexagon::Sched::tc_J4_jumpsetr:
  case Hexagon::Sched::tc_J4_tstbit0_fp0_jump_nt:
  case Hexagon::Sched::tc_SA1_addi:
  case Hexagon::Sched::tc_SA1_addsp:
  case Hexagon::Sched::tc_SA1_clrf:
  case Hexagon::Sched::tc_SA1_cmpeqi:
  case Hexagon::Sched::tc_dup_A2_add:
  case Hexagon::Sched::tc_dup_A2_sxtb:
  case Hexagon::Sched::tc_dup_C2_cmpeqi:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_A2_abs:
  case Hexagon::Sched::tc_A2_addh_h16_hh:
  case Hexagon::Sched::tc_A2_addh_h16_sat_hh:
  case Hexagon::Sched::tc_A2_addh_l16_hl:
  case Hexagon::Sched::tc_A2_addsat:
  case Hexagon::Sched::tc_A2_svavgh:
  case Hexagon::Sched::tc_A2_svavghs:
  case Hexagon::Sched::tc_A2_vavgh:
  case Hexagon::Sched::tc_A2_vavghcr:
  case Hexagon::Sched::tc_A2_vavghr:
  case Hexagon::Sched::tc_A6_vminub_RdP:
  case Hexagon::Sched::tc_C4_addipc:
  case Hexagon::Sched::tc_F2_dfimm_n:
  case Hexagon::Sched::tc_F2_sfmax:
  case Hexagon::Sched::tc_M2_acci:
  case Hexagon::Sched::tc_M2_xor_xacc:
  case Hexagon::Sched::tc_S2_brev:
  case Hexagon::Sched::tc_S2_extractu_rp:
  case Hexagon::Sched::tc_S2_insert:
  case Hexagon::Sched::tc_S2_vrndpackwh:
  case Hexagon::Sched::tc_S5_popcountp:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2early(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_J2_trap0:
  case Hexagon::Sched::tc_J2_unpause:
  case Hexagon::Sched::tc_SL2_jumpr31:
  case Hexagon::Sched::tc_Y2_icdataw:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2latepred(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_A2_paddfnew:
  case Hexagon::Sched::tc_A2_tfrpfnew:
  case Hexagon::Sched::tc_SA1_clrfnew:
  case Hexagon::Sched::tc_dup_C2_cmovenewif:
    return true;
  default:
    return false;
  }
}

inline bool is_TC3(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_J2_call:
  case Hexagon::Sched::tc_J2_callf:
  case Hexagon::Sched::tc_J2_callr:
  case Hexagon::Sched::tc_J2_callrf:
  case Hexagon::Sched::tc_J2_callrh:
    return true;
  default:
    return false;
  }
}

inline bool is_TC3stall(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_A2_tfrcrr:
  case Hexagon::Sched::tc_A2_tfrrcr:
  case Hexagon::Sched::tc_A4_tlbmatch:
  case Hexagon::Sched::tc_J2_pause:
  case Hexagon::Sched::tc_J2_rte:
  case Hexagon::Sched::tc_J2_trap1:
  case Hexagon::Sched::tc_S2_storew_locked:
  case Hexagon::Sched::tc_Y2_break:
  case Hexagon::Sched::tc_Y2_dckill:
  case Hexagon::Sched::tc_Y2_dctagr:
  case Hexagon::Sched::tc_Y2_dctagw:
  case Hexagon::Sched::tc_Y2_icdatar:
  case Hexagon::Sched::tc_Y2_icinva:
  case Hexagon::Sched::tc_Y2_ickill:
  case Hexagon::Sched::tc_Y2_ictagw:
  case Hexagon::Sched::tc_Y2_tlbp:
  case Hexagon::Sched::tc_Y2_tlbpp:
  case Hexagon::Sched::tc_Y2_tlbw:
  case Hexagon::Sched::tc_Y4_l2fetch:
  case Hexagon::Sched::tc_Y5_ctlbw:
  case Hexagon::Sched::tc_Y5_l2locka:
  case Hexagon::Sched::tc_Y5_tlbasidi:
    return true;
  default:
    return false;
  }
}

inline bool is_TC3x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_A2_vraddub:
  case Hexagon::Sched::tc_A2_vraddub_acc:
  case Hexagon::Sched::tc_A4_vrmaxh:
  case Hexagon::Sched::tc_A5_ACS:
  case Hexagon::Sched::tc_G4_tfrgcpp:
  case Hexagon::Sched::tc_G4_tfrgpcp:
  case Hexagon::Sched::tc_J2_loop0i:
  case Hexagon::Sched::tc_J2_loop0r:
  case Hexagon::Sched::tc_J2_ploop1si:
  case Hexagon::Sched::tc_J2_ploop1sr:
  case Hexagon::Sched::tc_M2_macsin:
  case Hexagon::Sched::tc_M2_mpysin:
  case Hexagon::Sched::tc_M4_mpyri_addr_u2:
  case Hexagon::Sched::tc_M7_dcmpyiw:
  case Hexagon::Sched::tc_M7_dcmpyiw_acc:
  case Hexagon::Sched::tc_S2_cabacdecbin:
  case Hexagon::Sched::tc_S4_vrcrotate:
  case Hexagon::Sched::tc_S4_vrcrotate_acc:
  case Hexagon::Sched::tc_Y2_ciad:
  case Hexagon::Sched::tc_Y2_crswap0:
  case Hexagon::Sched::tc_Y4_crswap10:
  case Hexagon::Sched::tc_Y6_diag0:
    return true;
  default:
    return false;
  }
}

inline bool is_TC4x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_F2_conv_d2df:
  case Hexagon::Sched::tc_F2_dfadd:
  case Hexagon::Sched::tc_F2_dfmpyhh:
  case Hexagon::Sched::tc_F2_sfadd:
  case Hexagon::Sched::tc_F2_sffma:
  case Hexagon::Sched::tc_F2_sffma_sc:
  case Hexagon::Sched::tc_F2_sfinvsqrta:
  case Hexagon::Sched::tc_F2_sfrecipa:
    return true;
  default:
    return false;
  }
}

inline bool is_TClatepredldaia(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_L2_ploadrbfnew_pi:
    return true;
  default:
    return false;
  }
}

inline bool is_TClatepredstaia(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_S2_pstorerbfnew_pi:
  case Hexagon::Sched::tc_S2_pstorerbnewfnew_pi:
    return true;
  default:
    return false;
  }
}

inline bool is_TCld(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_L2_deallocframe:
  case Hexagon::Sched::tc_L2_loadalignb_io:
  case Hexagon::Sched::tc_L2_loadalignb_pbr:
  case Hexagon::Sched::tc_L2_loadalignb_pci:
  case Hexagon::Sched::tc_L2_loadbsw2_io:
  case Hexagon::Sched::tc_L2_loadbsw2_pbr:
  case Hexagon::Sched::tc_L2_loadbsw2_pci:
  case Hexagon::Sched::tc_L2_loadrbgp:
  case Hexagon::Sched::tc_L2_loadw_locked:
  case Hexagon::Sched::tc_L4_loadalignb_ap:
  case Hexagon::Sched::tc_L4_loadalignb_ur:
  case Hexagon::Sched::tc_L4_loadbsw2_ap:
  case Hexagon::Sched::tc_L4_loadbsw2_ur:
  case Hexagon::Sched::tc_L4_loadrb_rr:
  case Hexagon::Sched::tc_L4_loadw_phys:
  case Hexagon::Sched::tc_L4_ploadrbf_abs:
  case Hexagon::Sched::tc_L4_ploadrbf_rr:
  case Hexagon::Sched::tc_L4_ploadrbfnew_rr:
  case Hexagon::Sched::tc_L6_memcpy:
  case Hexagon::Sched::tc_S2_allocframe:
  case Hexagon::Sched::tc_SL2_deallocframe:
  case Hexagon::Sched::tc_SS2_allocframe:
  case Hexagon::Sched::tc_Y2_dcfetch:
  case Hexagon::Sched::tc_Y2_dcfetchbo:
  case Hexagon::Sched::tc_dup_L2_deallocframe:
  case Hexagon::Sched::tc_dup_L2_loadrb_io:
  case Hexagon::Sched::tc_dup_S2_allocframe:
    return true;
  default:
    return false;
  }
}

inline bool is_TCnewvjump(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_J4_cmpeq_f_jumpnv_nt:
  case Hexagon::Sched::tc_J4_cmpeqi_f_jumpnv_nt:
  case Hexagon::Sched::tc_J4_cmplt_f_jumpnv_nt:
  case Hexagon::Sched::tc_J4_tstbit0_f_jumpnv_nt:
  case Hexagon::Sched::tc_L4_return:
  case Hexagon::Sched::tc_L4_return_f:
  case Hexagon::Sched::tc_L4_return_fnew_pnt:
  case Hexagon::Sched::tc_L4_return_map_to_raw_t:
  case Hexagon::Sched::tc_L4_return_map_to_raw_tnew_pnt:
  case Hexagon::Sched::tc_SL2_return:
    return true;
  default:
    return false;
  }
}

inline bool is_TCst(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_L4_add_memopb_io:
  case Hexagon::Sched::tc_L4_iadd_memopb_io:
  case Hexagon::Sched::tc_PS_storerbabs:
  case Hexagon::Sched::tc_PS_storerbnewabs:
  case Hexagon::Sched::tc_S2_pstorerbf_io:
  case Hexagon::Sched::tc_S2_pstorerbf_pi:
  case Hexagon::Sched::tc_S2_pstorerbf_pi_nt:
  case Hexagon::Sched::tc_S2_pstorerbnewf_io:
  case Hexagon::Sched::tc_S2_pstorerbnewf_pi:
  case Hexagon::Sched::tc_S2_storerb_io:
  case Hexagon::Sched::tc_S2_storerb_pbr:
  case Hexagon::Sched::tc_S2_storerb_pci:
  case Hexagon::Sched::tc_S2_storerb_pi_nt:
  case Hexagon::Sched::tc_S2_storerbnew_io:
  case Hexagon::Sched::tc_S2_storerbnew_pbr:
  case Hexagon::Sched::tc_S2_storerbnew_pci:
  case Hexagon::Sched::tc_S2_storew_rl_at_vi:
  case Hexagon::Sched::tc_S4_pstorerbf_abs:
  case Hexagon::Sched::tc_S4_pstorerbf_rr:
  case Hexagon::Sched::tc_S4_pstorerbfnew_abs:
  case Hexagon::Sched::tc_S4_pstorerbfnew_rr:
  case Hexagon::Sched::tc_S4_pstorerbnewf_abs:
  case Hexagon::Sched::tc_S4_pstorerbnewf_rr:
  case Hexagon::Sched::tc_S4_pstorerbnewfnew_abs:
  case Hexagon::Sched::tc_S4_pstorerbnewfnew_rr:
  case Hexagon::Sched::tc_S4_storeirb_io:
  case Hexagon::Sched::tc_S4_storeirbf_io:
  case Hexagon::Sched::tc_S4_storeirbfnew_io:
  case Hexagon::Sched::tc_S4_storerb_rr:
  case Hexagon::Sched::tc_S4_storerb_ur:
  case Hexagon::Sched::tc_S4_storerbnew_rr:
  case Hexagon::Sched::tc_S4_storerbnew_ur:
  case Hexagon::Sched::tc_SS2_storebi0:
  case Hexagon::Sched::tc_Y2_barrier:
  case Hexagon::Sched::tc_Y2_dccleana:
  case Hexagon::Sched::tc_Y4_l2tagw:
  case Hexagon::Sched::tc_Y6_dmpause:
  case Hexagon::Sched::tc_Y6_l2gcleaninvpa:
  case Hexagon::Sched::tc_dup_S2_storerb_io:
  case Hexagon::Sched::tc_dup_S4_storeirb_io:
    return true;
  default:
    return false;
  }
}
} // namespace llvm

#endif  // LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPTIMINGCLASSES_H
