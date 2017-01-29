//===-- llvm/lib/Target/X86/X86CallLowering.cpp - Call lowering -----------===//
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

#include "X86CallLowering.h"
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "X86TargetMachine.h"
#include "X86CallingConv.h"

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"

using namespace llvm;

#include "X86GenCallingConv.inc"

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

X86CallLowering::X86CallLowering(const X86TargetLowering &TLI)
    : CallLowering(&TLI) {}

bool X86CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, unsigned VReg) const {
  // TODO: handle functions returning non-void values.
  if (Val)
    return false;

  // silence unused-function warning, remove after the function implementation.
  (void)RetCC_X86;

  MIRBuilder.buildInstr(X86::RET).addImm(0);

  return true;
}

namespace {
struct FormalArgHandler : public CallLowering::ValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn, const DataLayout &DL)
      : ValueHandler(MIRBuilder, MRI, AssignFn), DL(DL) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {

    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

    unsigned AddrReg =
        MRI.createGenericVirtualRegister(LLT::pointer(0,
                                         DL.getPointerSizeInBits(0)));
    MIRBuilder.buildFrameIndex(AddrReg, FI);
    return AddrReg;
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {

    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, Size,
        0);
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

  const DataLayout &DL;
};
}

bool X86CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  if (F.arg_empty())
    return true;

  //TODO: handle variadic function
  if (F.isVarArg())
    return false;

  auto DL = MIRBuilder.getMF().getDataLayout();

  SmallVector<ArgInfo, 8> ArgInfos;
  unsigned Idx = 0;
  for (auto &Arg : F.getArgumentList()) {
    ArgInfo AInfo(VRegs[Idx], Arg.getType());
    setArgFlags(AInfo, Idx + 1, DL, F);
    ArgInfos.push_back(AInfo);
    Idx++;
  }

  FormalArgHandler ArgHandler(MIRBuilder, MIRBuilder.getMF().getRegInfo(),
                              CC_X86, DL);
  return handleAssignments(MIRBuilder, ArgInfos, ArgHandler);
}
