//===-- ARMRegisterInfo.cpp - ARM Register Information --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the ARM implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARMRegisterInfo.h"
#include "ARM.h"
#include "ARMBaseInstrInfo.h"
using namespace llvm;

void ARMRegisterInfo::anchor() { }

ARMRegisterInfo::ARMRegisterInfo(const ARMBaseInstrInfo &tii,
                                 const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}
