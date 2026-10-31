//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "RISCVExpandPseudoBase.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"

using namespace llvm;

#ifndef NDEBUG
static unsigned getFuncSizeFromInsts(const MachineFunction &MF,
                                     const RISCVInstrInfo *TII) {
  unsigned Size = 0;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      Size += TII->getInstSizeInBytes(MI);
  return Size;
}
#endif

bool RISCVExpandPseudoImplBase::run(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getFuncSizeFromInsts(MF, TII);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getFuncSizeFromInsts(MF, TII);
  assert(OldSize >= NewSize &&
         "Expanding Pseudos should not increase function size estimate");
#endif
  return Modified;
}

bool RISCVExpandPseudoImplBase::expandMBB(MachineBasicBlock &MBB) const {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}
