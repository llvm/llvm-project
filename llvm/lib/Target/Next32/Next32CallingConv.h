//=== Next32CallingConv.h - Next32 Custom Calling Convention Routines - C++ ==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the Next32 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_NEXT32CALLINGCONV_H
#define LLVM_LIB_TARGET_NEXT32_NEXT32CALLINGCONV_H

#include "Next32.h"
#include "Next32Subtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

static bool CC_Next32RegisterAllocator(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                       CCValAssign::LocInfo &LocInfo,
                                       ISD::ArgFlagsTy &ArgFlags,
                                       CCState &State) {
  unsigned int RegIdx = ValNo;
  MCRegisterClass GPR32 = Next32MCRegisterClasses[Next32::GPR32RegClassID];
  while (RegIdx < GPR32.getNumRegs() &&
         State.isAllocated(GPR32.getRegister(RegIdx)))
    ++RegIdx;

  if (RegIdx >= GPR32.getNumRegs())
    return false;
  State.addLoc(CCValAssign::getReg(ValNo, ValVT, GPR32.getRegister(ValNo),
                                   LocVT, LocInfo));
  return true;
}
} // namespace llvm

#endif
