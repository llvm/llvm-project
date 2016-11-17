//=== X86CallingConv.cpp - X86 Custom Calling Convention Impl   -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of custom routines for the X86
// Calling Convention that aren't done by tablegen.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86MCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/IR/CallingConv.h"

namespace llvm {

bool CC_X86_32_RegCall_Assign2Regs(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                   CCValAssign::LocInfo &LocInfo,
                                   ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  // List of GPR registers that are available to store values in regcall
  // calling convention.
  static const MCPhysReg RegList[] = {X86::EAX, X86::ECX, X86::EDX, X86::EDI,
                                      X86::ESI};

  // The vector will save all the available registers for allocation.
  SmallVector<unsigned, 5> AvailableRegs;

  // searching for the available registers.
  for (auto Reg : RegList) {
    if (!State.isAllocated(Reg))
      AvailableRegs.push_back(Reg);
  }

  const size_t RequiredGprsUponSplit = 2;
  if (AvailableRegs.size() < RequiredGprsUponSplit)
    return false; // Not enough free registers - continue the search.

  // Allocating the available registers
  for (unsigned I = 0; I < RequiredGprsUponSplit; I++) {

    // Marking the register as located
    unsigned Reg = State.AllocateReg(AvailableRegs[I]);

    // Since we previously made sure that 2 registers are available
    // we expect that a real register number will be returned
    assert(Reg && "Expecting a register will be available");

    // Assign the value to the allocated register
    State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, Reg, LocVT, LocInfo));
  }

  // Successful in allocating regsiters - stop scanning next rules.
  return true;
}

} // End llvm namespace
