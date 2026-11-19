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
  case Hexagon::Sched::tc_J2_jumprf_rb:
  case Hexagon::Sched::tc_J2_jumprfnew:
  case Hexagon::Sched::tc_J2_jumprfnew_rb:
  case Hexagon::Sched::tc_J2_jumprgtez:
  case Hexagon::Sched::tc_J2_jumprh:
  case Hexagon::Sched::tc_J4_cmpeq_fp0_jump_nt:
  case Hexagon::Sched::tc_J4_cmpeqi_fp0_jump_nt:
  case Hexagon::Sched::tc_J4_hintjumpr:
  case Hexagon::Sched::tc_J4_jumpseti:
  case Hexagon::Sched::tc_J4_jumpsetr:
  case Hexagon::Sched::tc_J4_tstbit0_fp0_jump_nt:
  case Hexagon::Sched::tc_LX_absdiff_h:
  case Hexagon::Sched::tc_LX_mpa_uwuhuw_si:
  case Hexagon::Sched::tc_LX_mpy_uw_uh:
  case Hexagon::Sched::tc_LX_shra_h_h:
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
} // namespace llvm

#endif  // LLVM_LIB_TARGET_HEXAGON_HEXAGONDEPTIMINGCLASSES_H
