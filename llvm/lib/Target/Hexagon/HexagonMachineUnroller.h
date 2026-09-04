//===- HexagonMachineUnroller.h - Hexagon machine unroller ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hexagon-specific implementation of machine loop unrolling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONMACHINEUNROLLER_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONMACHINEUNROLLER_H

#include "HexagonInstrInfo.h"
#include "llvm/CodeGen/MachineUnroller.h"

namespace llvm {

class HexagonMachineUnroller : public MachineUnroller {
  const HexagonInstrInfo *HII;

public:
  HexagonMachineUnroller(MachineUnrollerContext *C) : MachineUnroller(C) {
    HII = static_cast<const HexagonInstrInfo *>(C->TII);
  }

  unsigned getLoopCount(MachineBasicBlock &MBB) const override;

  /// Add instruction to compute trip count for the unrolled loop.
  unsigned addUnrolledLoopCountMI(MachineBasicBlock &MBB, unsigned LC,
                                  unsigned UnrollFactor) const override;

  /// Add instruction to compute remainder trip count for the unrolled loop.
  unsigned addRemLoopCountMI(MachineBasicBlock &MBB, unsigned LC,
                             unsigned UnrollFactor) const override;

  void changeLoopCount(MachineBasicBlock &BB, MachineBasicBlock &Preheader,
                       MachineBasicBlock &Header, MachineBasicBlock &LoopBB,
                       unsigned LC,
                       SmallVectorImpl<MachineOperand> &Cond) const override;
};
} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_HEXAGONMACHINEUNROLLER_H
