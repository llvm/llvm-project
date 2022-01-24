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
                            MachinePointerInfo &MPO, CCValAssign &VA) override;

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override;

  MachineInstrBuilder MIB;
};

struct M88kIncomingValueHandler : public CallLowering::IncomingValueHandler {
  M88kIncomingValueHandler(MachineIRBuilder &MIRBuilder,
                           MachineRegisterInfo &MRI)
      : CallLowering::IncomingValueHandler(MIRBuilder, MRI) {}

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

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

struct CallReturnHandler : public M88kIncomingValueHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder MIB)
      : M88kIncomingValueHandler(MIRBuilder, MRI), MIB(MIB) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
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

void OutgoingArgHandler::assignValueToAddress(Register ValVReg, Register Addr,
                                              LLT MemTy,
                                              MachinePointerInfo &MPO,
                                              CCValAssign &VA) {
  MachineFunction &MF = MIRBuilder.getMF();
  uint64_t LocMemOffset = VA.getLocMemOffset();

  auto MMO = MF.getMachineMemOperand(
      MPO, MachineMemOperand::MOStore, MemTy,
      commonAlignment(Align(16) /*STI.getStackAlignment()*/, LocMemOffset));

  Register ExtReg = extendRegister(ValVReg, VA);
  MIRBuilder.buildStore(ExtReg, Addr, *MMO);
}

Register OutgoingArgHandler::getStackAddress(uint64_t Size, int64_t Offset,
                                             MachinePointerInfo &MPO,
                                             ISD::ArgFlagsTy Flags) {
  MachineFunction &MF = MIRBuilder.getMF();
  MPO = MachinePointerInfo::getStack(MF, Offset);

  LLT p0 = LLT::pointer(0, 32);
  LLT s32 = LLT::scalar(32);
  auto SPReg = MIRBuilder.buildCopy(p0, Register(M88k::R31));

  auto OffsetReg = MIRBuilder.buildConstant(s32, Offset);
  auto AddrReg = MIRBuilder.buildPtrAdd(p0, SPReg, OffsetReg);
  return AddrReg.getReg(0);
}

void M88kIncomingValueHandler::assignValueToReg(Register ValVReg,
                                                Register PhysReg,
                                                CCValAssign VA) {
  assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
  assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

  uint64_t ValSize = VA.getValVT().getFixedSizeInBits();
  uint64_t LocSize = VA.getLocVT().getFixedSizeInBits();

  assert(ValSize <= 64 && "Unsupported value size");
  assert(LocSize <= 64 && "Unsupported location size");

  markPhysRegUsed(PhysReg);
  if (ValSize == LocSize) {
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  } else {
    assert(ValSize < LocSize && "Extensions not supported");

    // We cannot create a truncating copy, nor a trunc of a physical register.
    // Therefore, we need to copy the content of the physical register into a
    // virtual one and then truncate that.
    auto PhysRegToVReg = MIRBuilder.buildCopy(LLT::scalar(LocSize), PhysReg);
    MIRBuilder.buildTrunc(ValVReg, PhysRegToVReg);
  }
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
  llvm::LLT FramePtr = LLT::pointer(0, 32);
  MachineInstrBuilder AddrReg = MIRBuilder.buildFrameIndex(FramePtr, FI);
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

bool M88kCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, ArrayRef<Register> VRegs,
                                   FunctionLoweringInfo &FLI,
                                   Register SwiftErrorVReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const M88kSubtarget &STI = MF.getSubtarget<M88kSubtarget>();
  const M88kInstrInfo &TII = *STI.getInstrInfo();
  auto &DL = F.getParent()->getDataLayout();

  // Setup virtual register to hold incoming return address register aka %r1.
  Register ReturnAddrVReg = getFunctionLiveInPhysReg(
      MF, TII, M88k::R1, M88k::GPRRCRegClass, MIRBuilder.getDebugLoc());
  MRI.setType(ReturnAddrVReg, LLT::pointer(0, 32));

  auto MIB = MIRBuilder.buildInstrNoInsert(M88k::RET);

  bool Success = true;
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

  // Copy virtual return address register to %r1. It's used by RET.
  MIRBuilder.buildCopy(M88k::R1, ReturnAddrVReg);
  MIRBuilder.insertInstr(MIB);
  return Success;
}

bool M88kCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<ArrayRef<Register>> VRegs,
                                            FunctionLoweringInfo &FLI) const {
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
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> OutArgs;
  for (auto &OrigArg : Info.OrigArgs) {
    splitToValueTypes(OrigArg, OutArgs, DL, Info.CallConv);
  }

  SmallVector<ArgInfo, 8> InArgs;
  if (!Info.OrigRet.Ty->isVoidTy())
    splitToValueTypes(Info.OrigRet, InArgs, DL, Info.CallConv);

  // TODO Handle tail calls.

  MachineInstrBuilder CallSeqStart;
  CallSeqStart = MIRBuilder.buildInstr(M88k::ADJCALLSTACKDOWN);

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  unsigned Opc = Info.Callee.isReg() ? M88k::JSR : M88k::BSR;

  auto MIB = MIRBuilder.buildInstrNoInsert(Opc);
  MIB.add(Info.Callee);

  // Tell the call which registers are clobbered.
  const uint32_t *Mask;
  const M88kSubtarget &Subtarget = MF.getSubtarget<M88kSubtarget>();
  const auto *TRI = Subtarget.getRegisterInfo();

  // Do the actual argument marshalling.
  OutgoingValueAssigner ArgAssigner(CC_M88k);
  OutgoingArgHandler Handler(MIRBuilder, MRI, MIB); //, /*IsReturn*/ false);
  if (!determineAndHandleAssignments(Handler, ArgAssigner, OutArgs, MIRBuilder,
                                     Info.CallConv, Info.IsVarArg))
    return false;

  Mask = TRI->getCallPreservedMask(MF, Info.CallConv);
  MIB.addRegMask(Mask);

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  // If Callee is a reg, since it is used by a target specific
  // instruction, it must have a register class matching the
  // constraint of that instruction.
  if (Info.Callee.isReg())
    constrainOperandRegClass(MF, *TRI, MRI, *Subtarget.getInstrInfo(),
                             *Subtarget.getRegBankInfo(), *MIB, MIB->getDesc(),
                             Info.Callee, 0);

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arguments, the physical register must be an
  // implicit-define of the call instruction.
  if (!Info.OrigRet.Ty->isVoidTy()) {
    OutgoingValueAssigner ArgAssigner(RetCC_M88k);
    CallReturnHandler ReturnedArgHandler(MIRBuilder, MRI, MIB);
    if (!determineAndHandleAssignments(ReturnedArgHandler, ArgAssigner, InArgs,
                                       MIRBuilder, Info.CallConv, Info.IsVarArg,
                                       OutArgs[0].Regs[0]))
      return false;
  }

  CallSeqStart.addImm(ArgAssigner.StackOffset).addImm(0);
  MIRBuilder.buildInstr(M88k::ADJCALLSTACKUP)
      .addImm(ArgAssigner.StackOffset)
      .addImm(0);

  return true;
}
