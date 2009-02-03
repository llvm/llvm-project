//===-- TargetInstrInfo.cpp - Target Instruction Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Constant.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;

/// findTiedToSrcOperand - Returns the operand that is tied to the specified
/// dest operand. Returns -1 if there isn't one.
int TargetInstrDesc::findTiedToSrcOperand(unsigned OpNum) const {
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i == OpNum)
      continue;
    if (getOperandConstraint(i, TOI::TIED_TO) == (int)OpNum)
      return i;
  }
  return -1;
}


TargetInstrInfo::TargetInstrInfo(const TargetInstrDesc* Desc,
                                 unsigned numOpcodes)
  : Descriptors(Desc), NumOpcodes(numOpcodes) {
}

TargetInstrInfo::~TargetInstrInfo() {
}

bool TargetInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  const TargetInstrDesc &TID = MI->getDesc();
  if (!TID.isTerminator()) return false;
  
  // Conditional branch is a special case.
  if (TID.isBranch() && !TID.isBarrier())
    return true;
  if (!TID.isPredicable())
    return true;
  return !isPredicated(MI);
}
