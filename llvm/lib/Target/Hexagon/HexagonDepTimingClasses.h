//===- HexagonDepTimingClasses.h ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Automatically generated file, please consult code owner before editing.
//===----------------------------------------------------------------------===//



#ifndef TARGET_HEXAGON_HEXAGON_DEP_TIMING_CLASSES_H
#define TARGET_HEXAGON_HEXAGON_DEP_TIMING_CLASSES_H

#include "HexagonInstrInfo.h"

namespace llvm {

inline bool is_TC3x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_16d0d8d5:
  case Hexagon::Sched::tc_1853ea6d:
  case Hexagon::Sched::tc_60571023:
  case Hexagon::Sched::tc_7934b9df:
  case Hexagon::Sched::tc_8fd5f294:
  case Hexagon::Sched::tc_b9c0b731:
  case Hexagon::Sched::tc_bcc96cee:
  case Hexagon::Sched::tc_c6ce9b3f:
  case Hexagon::Sched::tc_c6ebf8dd:
  case Hexagon::Sched::tc_c82dc1ff:
  case Hexagon::Sched::tc_caaebcba:
  case Hexagon::Sched::tc_cf59f215:
  case Hexagon::Sched::tc_e913dc32:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2early(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_14cd4cfa:
  case Hexagon::Sched::tc_2a160009:
    return true;
  default:
    return false;
  }
}

inline bool is_TC4x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_038a1342:
  case Hexagon::Sched::tc_4d99bca9:
  case Hexagon::Sched::tc_6792d5ff:
  case Hexagon::Sched::tc_9c00ce8d:
  case Hexagon::Sched::tc_d580173f:
  case Hexagon::Sched::tc_f3eaa14b:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_00afc57e:
  case Hexagon::Sched::tc_1b9c9ee5:
  case Hexagon::Sched::tc_234a11a5:
  case Hexagon::Sched::tc_2b6f77c6:
  case Hexagon::Sched::tc_41d5298e:
  case Hexagon::Sched::tc_5ba5997d:
  case Hexagon::Sched::tc_84df2cd3:
  case Hexagon::Sched::tc_87735c3b:
  case Hexagon::Sched::tc_897d1a9d:
  case Hexagon::Sched::tc_976ddc4f:
  case Hexagon::Sched::tc_b44c6e2a:
  case Hexagon::Sched::tc_b9c4623f:
  case Hexagon::Sched::tc_c2f7d806:
  case Hexagon::Sched::tc_c74f796f:
  case Hexagon::Sched::tc_d088982c:
  case Hexagon::Sched::tc_ef84f62f:
  case Hexagon::Sched::tc_f49e76f4:
    return true;
  default:
    return false;
  }
}

inline bool is_TC1(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_181af5d0:
  case Hexagon::Sched::tc_1b82a277:
  case Hexagon::Sched::tc_1e856f58:
  case Hexagon::Sched::tc_351fed2d:
  case Hexagon::Sched::tc_3669266a:
  case Hexagon::Sched::tc_3cb8ea06:
  case Hexagon::Sched::tc_452f85af:
  case Hexagon::Sched::tc_481e5e5c:
  case Hexagon::Sched::tc_49eb22c8:
  case Hexagon::Sched::tc_523fcf30:
  case Hexagon::Sched::tc_52d7bbea:
  case Hexagon::Sched::tc_53bc8a6a:
  case Hexagon::Sched::tc_540fdfbc:
  case Hexagon::Sched::tc_55050d58:
  case Hexagon::Sched::tc_609d2efe:
  case Hexagon::Sched::tc_68cb12ce:
  case Hexagon::Sched::tc_6ebb4a12:
  case Hexagon::Sched::tc_6efc556e:
  case Hexagon::Sched::tc_73043bf4:
  case Hexagon::Sched::tc_7a830544:
  case Hexagon::Sched::tc_855b0b61:
  case Hexagon::Sched::tc_8fe6b782:
  case Hexagon::Sched::tc_90f3e30c:
  case Hexagon::Sched::tc_97743097:
  case Hexagon::Sched::tc_99be14ca:
  case Hexagon::Sched::tc_9faf76ae:
  case Hexagon::Sched::tc_a46f0df5:
  case Hexagon::Sched::tc_a904d137:
  case Hexagon::Sched::tc_b9488031:
  case Hexagon::Sched::tc_be706f30:
  case Hexagon::Sched::tc_c6aa82f7:
  case Hexagon::Sched::tc_cde8b071:
  case Hexagon::Sched::tc_d6bf0472:
  case Hexagon::Sched::tc_dbdffe3d:
  case Hexagon::Sched::tc_e0739b8c:
  case Hexagon::Sched::tc_e1e99bfa:
  case Hexagon::Sched::tc_e9fae2d6:
  case Hexagon::Sched::tc_f2704b9a:
  case Hexagon::Sched::tc_f8eeed7a:
    return true;
  default:
    return false;
  }
}
} // namespace llvm

#endif
