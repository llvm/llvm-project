//===---------------------- RetireStage.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the retire stage of an instruction pipeline.
/// The RetireStage represents the process logic that interacts with the
/// simulated RetireControlUnit hardware.
///
//===----------------------------------------------------------------------===//

#include "RetireStage.h"
#include "HWEventListener.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "llvm-mca"

namespace mca {

void RetireStage::cycleStart() {
  if (RCU.isEmpty())
    return;

  const unsigned MaxRetirePerCycle = RCU.getMaxRetirePerCycle();
  unsigned NumRetired = 0;
  while (!RCU.isEmpty()) {
    if (MaxRetirePerCycle != 0 && NumRetired == MaxRetirePerCycle)
      break;
    const RetireControlUnit::RUToken &Current = RCU.peekCurrentToken();
    if (!Current.Executed)
      break;
    RCU.consumeCurrentToken();
    notifyInstructionRetired(Current.IR);
    NumRetired++;
  }
}

void RetireStage::notifyInstructionRetired(const InstRef &IR) {
  LLVM_DEBUG(dbgs() << "[E] Instruction Retired: #" << IR << '\n');
  SmallVector<unsigned, 4> FreedRegs(PRF.getNumRegisterFiles());
  const Instruction &Inst = *IR.getInstruction();
  const InstrDesc &Desc = Inst.getDesc();

  bool ShouldFreeRegs = !(Desc.isZeroLatency() && Inst.isDependencyBreaking());
  for (const std::unique_ptr<WriteState> &WS : Inst.getDefs())
    PRF.removeRegisterWrite(*WS.get(), FreedRegs, ShouldFreeRegs);
  notifyEvent<HWInstructionEvent>(HWInstructionRetiredEvent(IR, FreedRegs));
}

} // namespace mca
