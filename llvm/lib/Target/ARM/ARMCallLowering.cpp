//===-- llvm/lib/Target/ARM/ARMCallLowering.cpp - Call lowering -----------===//
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

#include "ARMCallLowering.h"

#include "ARMBaseInstrInfo.h"
#include "ARMISelLowering.h"

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

ARMCallLowering::ARMCallLowering(const ARMTargetLowering &TLI)
    : CallLowering(&TLI) {}

static bool isSupportedType(const DataLayout DL, const ARMTargetLowering &TLI,
                            Type *T) {
  EVT VT = TLI.getValueType(DL, T);
  return VT.isSimple() && VT.isInteger() &&
         VT.getSimpleVT().getSizeInBits() == 32;
}

namespace {
struct FuncReturnHandler : public CallLowering::ValueHandler {
  FuncReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder &MIB)
      : ValueHandler(MIRBuilder, MRI), MIB(MIB) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    llvm_unreachable("Don't know how to get a stack address yet");
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    assert(VA.getValVT().getSizeInBits() == 32 && "Unsupported value size");
    assert(VA.getLocVT().getSizeInBits() == 32 && "Unsupported location size");

    MIRBuilder.buildCopy(PhysReg, ValVReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    llvm_unreachable("Don't know how to assign a value to an address yet");
  }

  MachineInstrBuilder &MIB;
};
} // End anonymous namespace.

/// Lower the return value for the already existing \p Ret. This assumes that
/// \p MIRBuilder's insertion point is correct.
bool ARMCallLowering::lowerReturnVal(MachineIRBuilder &MIRBuilder,
                                     const Value *Val, unsigned VReg,
                                     MachineInstrBuilder &Ret) const {
  if (!Val)
    // Nothing to do here.
    return true;

  auto &MF = MIRBuilder.getMF();
  const auto &F = *MF.getFunction();

  auto DL = MF.getDataLayout();
  auto &TLI = *getTLI<ARMTargetLowering>();
  if (!isSupportedType(DL, TLI, Val->getType()))
    return false;

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForReturn(F.getCallingConv(), F.isVarArg());

  ArgInfo RetInfo(VReg, Val->getType());
  setArgFlags(RetInfo, AttributeSet::ReturnIndex, DL, F);

  FuncReturnHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret);
  return handleAssignments(MIRBuilder, AssignFn, RetInfo, RetHandler);
}

bool ARMCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, unsigned VReg) const {
  assert(!Val == !VReg && "Return value without a vreg");

  auto Ret = AddDefaultPred(MIRBuilder.buildInstrNoInsert(ARM::BX_RET));

  if (!lowerReturnVal(MIRBuilder, Val, VReg, Ret))
    return false;

  MIRBuilder.insertInstr(Ret);
  return true;
}

namespace {
struct FormalArgHandler : public CallLowering::ValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : ValueHandler(MIRBuilder, MRI) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    llvm_unreachable("Don't know how to get a stack address yet");
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    assert(VA.getValVT().getSizeInBits() == 32 && "Unsupported value size");
    assert(VA.getLocVT().getSizeInBits() == 32 && "Unsupported location size");

    MIRBuilder.getMBB().addLiveIn(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    llvm_unreachable("Don't know how to assign a value to an address yet");
  }
};
} // End anonymous namespace

bool ARMCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  // Quick exit if there aren't any args
  if (F.arg_empty())
    return true;

  // Stick to only 4 arguments for now
  if (F.arg_size() > 4)
    return false;

  if (F.isVarArg())
    return false;

  auto DL = MIRBuilder.getMF().getDataLayout();
  auto &TLI = *getTLI<ARMTargetLowering>();

  auto &Args = F.getArgumentList();
  for (auto &Arg : Args)
    if (!isSupportedType(DL, TLI, Arg.getType()))
      return false;

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), F.isVarArg());

  SmallVector<ArgInfo, 8> ArgInfos;
  unsigned Idx = 0;
  for (auto &Arg : Args) {
    ArgInfo AInfo(VRegs[Idx], Arg.getType());
    setArgFlags(AInfo, Idx + 1, DL, F);
    ArgInfos.push_back(AInfo);
    Idx++;
  }

  FormalArgHandler ArgHandler(MIRBuilder, MIRBuilder.getMF().getRegInfo());
  return handleAssignments(MIRBuilder, AssignFn, ArgInfos, ArgHandler);
}
