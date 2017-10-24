//===--- HexagonDepTimingClasses.h ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_HEXAGON_HEXAGON_DEP_TIMING_CLASSES_H
#define TARGET_HEXAGON_HEXAGON_DEP_TIMING_CLASSES_H

#include "HexagonInstrInfo.h"

namespace llvm {

inline bool is_TC3x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_1000eb10:
  case Hexagon::Sched::tc_2aaab1e0:
  case Hexagon::Sched::tc_4997da4a:
  case Hexagon::Sched::tc_5d806107:
  case Hexagon::Sched::tc_6264c5e0:
  case Hexagon::Sched::tc_69bb508b:
  case Hexagon::Sched::tc_8c8041e6:
  case Hexagon::Sched::tc_8cb685d9:
  case Hexagon::Sched::tc_a12a5971:
  case Hexagon::Sched::tc_ae0722f7:
  case Hexagon::Sched::tc_ae2c2dc2:
  case Hexagon::Sched::tc_bc5561d8:
  case Hexagon::Sched::tc_d6a805a8:
  case Hexagon::Sched::tc_f055fbb6:
  case Hexagon::Sched::tc_feb4974b:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2early(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_35fb9d13:
  case Hexagon::Sched::tc_cbe45117:
    return true;
  default:
    return false;
  }
}

inline bool is_TC4x(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_09c86199:
  case Hexagon::Sched::tc_2d1e6f5c:
  case Hexagon::Sched::tc_2e55aa16:
  case Hexagon::Sched::tc_3bea1824:
  case Hexagon::Sched::tc_e836c161:
  case Hexagon::Sched::tc_f1aa2cdb:
    return true;
  default:
    return false;
  }
}

inline bool is_TC2(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_090485bb:
  case Hexagon::Sched::tc_1fe8323c:
  case Hexagon::Sched::tc_37326008:
  case Hexagon::Sched::tc_3c10f809:
  case Hexagon::Sched::tc_47ab9233:
  case Hexagon::Sched::tc_485bb57c:
  case Hexagon::Sched::tc_511f28f6:
  case Hexagon::Sched::tc_583510c7:
  case Hexagon::Sched::tc_63cd9d2d:
  case Hexagon::Sched::tc_76c4c5ef:
  case Hexagon::Sched::tc_7ca2ea10:
  case Hexagon::Sched::tc_87601822:
  case Hexagon::Sched::tc_88fa2da6:
  case Hexagon::Sched::tc_94e6ffd9:
  case Hexagon::Sched::tc_ab1b5e74:
  case Hexagon::Sched::tc_b0f50e3c:
  case Hexagon::Sched::tc_bd16579e:
  case Hexagon::Sched::tc_c0cd91a8:
  case Hexagon::Sched::tc_ca280e8b:
  case Hexagon::Sched::tc_cd321066:
  case Hexagon::Sched::tc_d95f4e98:
  case Hexagon::Sched::tc_e17ce9ad:
  case Hexagon::Sched::tc_f1240c08:
  case Hexagon::Sched::tc_faab1248:
    return true;
  default:
    return false;
  }
}

inline bool is_TC1(unsigned SchedClass) {
  switch (SchedClass) {
  case Hexagon::Sched::tc_07ac815d:
  case Hexagon::Sched::tc_1b6011fb:
  case Hexagon::Sched::tc_1b834fe7:
  case Hexagon::Sched::tc_1e062b18:
  case Hexagon::Sched::tc_1f9668cc:
  case Hexagon::Sched::tc_43068634:
  case Hexagon::Sched::tc_47f0b7ad:
  case Hexagon::Sched::tc_537e2013:
  case Hexagon::Sched::tc_548f402d:
  case Hexagon::Sched::tc_5fa2857c:
  case Hexagon::Sched::tc_5fe9fcd0:
  case Hexagon::Sched::tc_78b3c689:
  case Hexagon::Sched::tc_7c2dcd4d:
  case Hexagon::Sched::tc_81a23d44:
  case Hexagon::Sched::tc_821c4233:
  case Hexagon::Sched::tc_92d1833c:
  case Hexagon::Sched::tc_9a13af9d:
  case Hexagon::Sched::tc_9c18c9a5:
  case Hexagon::Sched::tc_9df8b0dc:
  case Hexagon::Sched::tc_9f518242:
  case Hexagon::Sched::tc_a1fb80e1:
  case Hexagon::Sched::tc_a333d2a9:
  case Hexagon::Sched::tc_a87879e8:
  case Hexagon::Sched::tc_aad55963:
  case Hexagon::Sched::tc_b08b653e:
  case Hexagon::Sched::tc_b324366f:
  case Hexagon::Sched::tc_b5bfaa60:
  case Hexagon::Sched::tc_b86c7e8b:
  case Hexagon::Sched::tc_c58f771a:
  case Hexagon::Sched::tc_d108a090:
  case Hexagon::Sched::tc_d1b5a4b6:
  case Hexagon::Sched::tc_d2609065:
  case Hexagon::Sched::tc_d63b71d1:
  case Hexagon::Sched::tc_e2c31426:
  case Hexagon::Sched::tc_e8c7a357:
  case Hexagon::Sched::tc_eb07ef6f:
  case Hexagon::Sched::tc_f16d5b17:
    return true;
  default:
    return false;
  }
}
} // namespace llvm

#endif
