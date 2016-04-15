//===-- llvm/lib/Target/AArch64/AArch64CallLowering.cpp - Call lowering ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "AArch64CallLowering.h"
#include "AArch64ISelLowering.h"

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

AArch64CallLowering::AArch64CallLowering(const AArch64TargetLowering &TLI)
  : CallLowering(&TLI) {
}

bool AArch64CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                        const Value *Val, unsigned VReg) const {
  MachineInstr *Return = MIRBuilder.buildInstr(AArch64::RET_ReallyLR);
  assert(Return && "Unable to build a return instruction?!");

  assert(((Val && VReg) || (!Val && !VReg)) && "Return value without a vreg");
  if (VReg) {
    assert(Val->getType()->isIntegerTy() && "Type not supported yet");
    unsigned Size = Val->getType()->getPrimitiveSizeInBits();
    assert((Size == 64 || Size == 32) && "Size not supported yet");
    unsigned ResReg = (Size == 32) ? AArch64::W0 : AArch64::X0;
    // Set the insertion point to be right before Return.
    MIRBuilder.setInstr(*Return, /* Before */ true);
    MachineInstr *Copy =
        MIRBuilder.buildInstr(TargetOpcode::COPY, ResReg, VReg);
    (void)Copy;
    assert(Copy->getNextNode() == Return &&
           "The insertion did not happen where we expected");
    MachineInstrBuilder(MIRBuilder.getMF(), Return)
        .addReg(ResReg, RegState::Implicit);
  }
  return true;
}

bool AArch64CallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function::ArgumentListType &Args,
    const SmallVectorImpl<unsigned> &VRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  unsigned NumArgs = Args.size();
  Function::const_arg_iterator CurOrigArg = Args.begin();
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  for (unsigned i = 0; i != NumArgs; ++i, ++CurOrigArg) {
    MVT ValVT = MVT::getVT(CurOrigArg->getType());
    CCAssignFn *AssignFn =
        TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);
    bool Res =
        AssignFn(i, ValVT, ValVT, CCValAssign::Full, ISD::ArgFlagsTy(), CCInfo);
    assert(!Res && "Call operand has unhandled type");
    (void)Res;
  }
  assert(ArgLocs.size() == Args.size() &&
         "We have a different number of location and args?!");
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    assert(VA.isRegLoc() && "Not yet implemented");
    // Transform the arguments in physical registers into virtual ones.
    MIRBuilder.getMBB().addLiveIn(VA.getLocReg());
    MIRBuilder.buildInstr(TargetOpcode::COPY, VRegs[i], VA.getLocReg());

    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      // We don't care about bitcast.
      break;
    case CCValAssign::AExt:
    case CCValAssign::SExt:
    case CCValAssign::ZExt:
      // Zero/Sign extend the register.
      assert(0 && "Not yet implemented");
      break;
    }
  }
  return true;
}
