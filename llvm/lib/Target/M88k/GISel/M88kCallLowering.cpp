//===-- M88kCallLowering.cpp - Call lowering --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
//
//===----------------------------------------------------------------------===//

#include "M88kCallLowering.h"
#include "M88kCallingConv.h"
#include "M88kInstrInfo.h"
#include "M88kSubtarget.h"
#include "M88kTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/TargetCallingConv.h"

using namespace llvm;

// The generated calling convention is included twice.
#include "M88kGenCallingConv.inc"

M88kCallLowering::M88kCallLowering(const M88kTargetLowering &TLI)
    : CallLowering(&TLI) {}

namespace {

struct OutgoingArgHandler : public CallLowering::OutgoingValueHandler {
  OutgoingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     MachineInstrBuilder MIB)
      : OutgoingValueHandler(MIRBuilder, MRI), MIB(MIB) {}

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign VA) override;

  unsigned assignCustomValue(CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs,
                             std::function<void()> *Thunk = nullptr) override;

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    llvm_unreachable("assignValueToAddress not implemented");
  }

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override {
    llvm_unreachable("unimplemented");
  }

  MachineInstrBuilder MIB;
};

struct M88kIncomingValueHandler : public CallLowering::IncomingValueHandler {
  M88kIncomingValueHandler(MachineIRBuilder &MIRBuilder,
                           MachineRegisterInfo &MRI)
      : CallLowering::IncomingValueHandler(MIRBuilder, MRI) {}

  uint64_t StackUsed;

private:
  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign VA) override;

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            MachinePointerInfo &MPO, CCValAssign &VA) override;

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override;

  unsigned assignCustomValue(CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs,
                             std::function<void()> *Thunk = nullptr) override;

  /// Marking a physical register as used is different between formal
  /// parameters, where it's a basic block live-in, and call returns, where it's
  /// an implicit-def of the call instruction.
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;
};

struct FormalArgHandler : public M88kIncomingValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : M88kIncomingValueHandler(MIRBuilder, MRI) {}

  void markPhysRegUsed(unsigned PhysReg) override;
};

} // namespace

void OutgoingArgHandler::assignValueToReg(Register ValVReg, Register PhysReg,
                                          CCValAssign VA) {
  MIB.addUse(PhysReg, RegState::Implicit);
  Register ExtReg = extendRegister(ValVReg, VA);
  MIRBuilder.buildCopy(PhysReg, ExtReg);
}

unsigned OutgoingArgHandler::assignCustomValue(CallLowering::ArgInfo &Arg,
                                               ArrayRef<CCValAssign> VAs,
                                               std::function<void()> *Thunk) {
  assert(Arg.Regs.size() == 1 && "Can't handle multple regs yet");

  CCValAssign VA = VAs[0];
  assert(VA.needsCustom() && "Value doesn't need custom handling");

  // Custom lowering for other types, such as f16, is currently not supported
  if (VA.getValVT() != MVT::f64)
    return 0;

  CCValAssign NextVA = VAs[1];
  assert(NextVA.needsCustom() && "Value doesn't need custom handling");
  assert(NextVA.getValVT() == MVT::f64 && "Unsupported type");

  assert(VA.getValNo() == NextVA.getValNo() &&
         "Values belong to different arguments");

  assert(VA.isRegLoc() && "Value should be in reg");
  assert(NextVA.isRegLoc() && "Value should be in reg");

  Register NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                        MRI.createGenericVirtualRegister(LLT::scalar(32))};
  MIRBuilder.buildUnmerge(NewRegs, Arg.Regs[0]);

  if (Thunk) {
    *Thunk = [=]() {
      assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
      assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);
    };
    return 1;
  }
  assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
  assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);
  return 1;
}

void M88kIncomingValueHandler::assignValueToReg(Register ValVReg,
                                                Register PhysReg,
                                                CCValAssign VA) {
  markPhysRegUsed(PhysReg);
  IncomingValueHandler::assignValueToReg(ValVReg, PhysReg, VA);
}

void M88kIncomingValueHandler::assignValueToAddress(Register ValVReg,
                                                    Register Addr, LLT MemTy,
                                                    MachinePointerInfo &MPO,
                                                    CCValAssign &VA) {
  MachineFunction &MF = MIRBuilder.getMF();
  auto *MMO = MF.getMachineMemOperand(MPO, MachineMemOperand::MOLoad, MemTy,
                                      inferAlignFromPtrInfo(MF, MPO));
  MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
}

Register M88kIncomingValueHandler::getStackAddress(uint64_t Size,
                                                   int64_t Offset,
                                                   MachinePointerInfo &MPO,
                                                   ISD::ArgFlagsTy Flags) {
  auto &MFI = MIRBuilder.getMF().getFrameInfo();
  const bool IsImmutable = !Flags.isByVal();
  int FI = MFI.CreateFixedObject(Size, Offset, IsImmutable);
  MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

  // Build Frame Index
  llvm::LLT FramePtr = LLT::pointer(
      0, MIRBuilder.getMF().getDataLayout().getPointerSizeInBits());
  MachineInstrBuilder AddrReg = MIRBuilder.buildFrameIndex(FramePtr, FI);
  StackUsed = std::max(StackUsed, Size + Offset);
  return AddrReg.getReg(0);
}

unsigned
M88kIncomingValueHandler::assignCustomValue(CallLowering::ArgInfo &Arg,
                                            ArrayRef<CCValAssign> VAs,
                                            std::function<void()> *Thunk) {
  assert(Arg.Regs.size() == 1 && "Can't handle multple regs yet");

  CCValAssign VA = VAs[0];
  assert(VA.needsCustom() && "Value doesn't need custom handling");

  // Custom lowering for other types is currently not supported.
  if (VA.getValVT() != MVT::f64)
    return 0;

  CCValAssign NextVA = VAs[1];
  assert(NextVA.needsCustom() && "Value doesn't need custom handling");
  assert(NextVA.getValVT() == MVT::f64 && "Unsupported type");

  assert(VA.getValNo() == NextVA.getValNo() &&
         "Values belong to different arguments");

  assert(VA.isRegLoc() && "Value should be in reg");
  assert(NextVA.isRegLoc() && "Value should be in reg");

  Register NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                        MRI.createGenericVirtualRegister(LLT::scalar(32))};

  assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
  assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

  MIRBuilder.buildMerge(Arg.Regs[0], NewRegs);

  return 1;
}

void FormalArgHandler::markPhysRegUsed(unsigned PhysReg) {
  MIRBuilder.getMRI()->addLiveIn(PhysReg);
  MIRBuilder.getMBB().addLiveIn(PhysReg);
}

bool M88kCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, ArrayRef<Register> VRegs,
                                   FunctionLoweringInfo &FLI,
                                   Register SwiftErrorVReg) const {
  auto MIB = MIRBuilder.buildInstrNoInsert(M88k::RET);
  bool Success = true;
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();
  if (!VRegs.empty()) {
    SmallVector<ArgInfo, 8> SplitArgs;
    ArgInfo OrigArg{VRegs, Val->getType(), 0};
    setArgFlags(OrigArg, AttributeList::ReturnIndex, DL, F);
    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    OutgoingValueAssigner ArgAssigner(RetCC_M88k);
    OutgoingArgHandler ArgHandler(MIRBuilder, MRI, MIB);
    Success = determineAndHandleAssignments(ArgHandler, ArgAssigner, SplitArgs,
                                            MIRBuilder, F.getCallingConv(),
                                            F.isVarArg());
  }
  MIRBuilder.insertInstr(MIB);
  return Success;
}

bool M88kCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<ArrayRef<Register>> VRegs,
                                            FunctionLoweringInfo &FLI) const {
  // Add register holding return address as live-in.
  MIRBuilder.getMRI()->addLiveIn(M88k::R1);
  MIRBuilder.getMBB().addLiveIn(M88k::R1);

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned I = 0;
  for (const auto &Arg : F.args()) {
    ArgInfo OrigArg{VRegs[I], Arg.getType(), I};
    setArgFlags(OrigArg, I + AttributeList::FirstArgIndex, DL, F);
    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    ++I;
  }

  IncomingValueAssigner ArgAssigner(CC_M88k);
  FormalArgHandler ArgHandler(MIRBuilder, MRI);
  return determineAndHandleAssignments(ArgHandler, ArgAssigner, SplitArgs,
                                       MIRBuilder, F.getCallingConv(),
                                       F.isVarArg());
}

bool M88kCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                 CallLoweringInfo &Info) const {
  return false;
}
