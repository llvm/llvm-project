//=== X86CallingConv.h - X86 Custom Calling Convention Routines -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom routines for the X86 Calling Convention that
// aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86CALLINGCONV_H
#define LLVM_LIB_TARGET_X86_X86CALLINGCONV_H

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

inline bool CC_X86_32_VectorCallIndirect(unsigned &ValNo, MVT &ValVT,
                                         MVT &LocVT,
                                         CCValAssign::LocInfo &LocInfo,
                                         ISD::ArgFlagsTy &ArgFlags,
                                         CCState &State) {
  // Similar to CCPassIndirect, with the addition of inreg.
  LocVT = MVT::i32;
  LocInfo = CCValAssign::Indirect;
  ArgFlags.setInReg();
  return false; // Continue the search, but now for i32.
}


inline bool CC_X86_AnyReg_Error(unsigned &, MVT &, MVT &,
                                CCValAssign::LocInfo &, ISD::ArgFlagsTy &,
                                CCState &) {
  llvm_unreachable("The AnyReg calling convention is only supported by the " \
                   "stackmap and patchpoint intrinsics.");
  // gracefully fallback to X86 C calling convention on Release builds.
  return false;
}

inline bool CC_X86_32_MCUInReg(unsigned &ValNo, MVT &ValVT,
                                         MVT &LocVT,
                                         CCValAssign::LocInfo &LocInfo,
                                         ISD::ArgFlagsTy &ArgFlags,
                                         CCState &State) {
  // This is similar to CCAssignToReg<[EAX, EDX, ECX]>, but makes sure
  // not to split i64 and double between a register and stack
  static const MCPhysReg RegList[] = {X86::EAX, X86::EDX, X86::ECX};
  static const unsigned NumRegs = sizeof(RegList)/sizeof(RegList[0]);
  
  SmallVectorImpl<CCValAssign> &PendingMembers = State.getPendingLocs();

  // If this is the first part of an double/i64/i128, or if we're already
  // in the middle of a split, add to the pending list. If this is not
  // the end of the split, return, otherwise go on to process the pending
  // list
  if (ArgFlags.isSplit() || !PendingMembers.empty()) {
    PendingMembers.push_back(
        CCValAssign::getPending(ValNo, ValVT, LocVT, LocInfo));
    if (!ArgFlags.isSplitEnd())
      return true;
  }

  // If there are no pending members, we are not in the middle of a split,
  // so do the usual inreg stuff.
  if (PendingMembers.empty()) {
    if (unsigned Reg = State.AllocateReg(RegList)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT, Reg, LocVT, LocInfo));
      return true;
    }
    return false;
  }

  assert(ArgFlags.isSplitEnd());

  // We now have the entire original argument in PendingMembers, so decide
  // whether to use registers or the stack.
  // Per the MCU ABI:
  // a) To use registers, we need to have enough of them free to contain
  // the entire argument.
  // b) We never want to use more than 2 registers for a single argument.

  unsigned FirstFree = State.getFirstUnallocated(RegList);
  bool UseRegs = PendingMembers.size() <= std::min(2U, NumRegs - FirstFree);

  for (auto &It : PendingMembers) {
    if (UseRegs)
      It.convertToReg(State.AllocateReg(RegList[FirstFree++]));
    else
      It.convertToMem(State.AllocateStack(4, 4));
    State.addLoc(It);
  }

  PendingMembers.clear();

  return true;
}

} // End llvm namespace

#endif

