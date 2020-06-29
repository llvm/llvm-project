//===- llvm/lib/Target/ARM/ARMCallLowering.cpp - Call lowering ------------===//
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

#include "ARMCallLowering.h"
#include "ARMBaseInstrInfo.h"
#include "ARMISelLowering.h"
#include "ARMSubtarget.h"
#include "Utils/ARMBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/MachineValueType.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;

ARMCallLowering::ARMCallLowering(const ARMTargetLowering &TLI)
    : CallLowering(&TLI) {}

static bool isSupportedType(const DataLayout &DL, const ARMTargetLowering &TLI,
                            Type *T) {
  if (T->isArrayTy())
    return isSupportedType(DL, TLI, T->getArrayElementType());

  if (T->isStructTy()) {
    // For now we only allow homogeneous structs that we can manipulate with
    // G_MERGE_VALUES and G_UNMERGE_VALUES
    auto StructT = cast<StructType>(T);
    for (unsigned i = 1, e = StructT->getNumElements(); i != e; ++i)
      if (StructT->getElementType(i) != StructT->getElementType(0))
        return false;
    return isSupportedType(DL, TLI, StructT->getElementType(0));
  }

  EVT VT = TLI.getValueType(DL, T, true);
  if (!VT.isSimple() || VT.isVector() ||
      !(VT.isInteger() || VT.isFloatingPoint()))
    return false;

  unsigned VTSize = VT.getSimpleVT().getSizeInBits();

  if (VTSize == 64)
    // FIXME: Support i64 too
    return VT.isFloatingPoint();

  return VTSize == 1 || VTSize == 8 || VTSize == 16 || VTSize == 32;
}

namespace {

/// Helper class for values going out through an ABI boundary (used for handling
/// function return values and call parameters).
struct OutgoingValueHandler : public CallLowering::ValueHandler {
  OutgoingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       MachineInstrBuilder &MIB, CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  bool isIncomingArgumentHandler() const override { return false; }

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    LLT p0 = LLT::pointer(0, 32);
    LLT s32 = LLT::scalar(32);
    auto SPReg = MIRBuilder.buildCopy(p0, Register(ARM::SP));

    auto OffsetReg = MIRBuilder.buildConstant(s32, Offset);

    auto AddrReg = MIRBuilder.buildPtrAdd(p0, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
    return AddrReg.getReg(0);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    assert(VA.getValVT().getSizeInBits() <= 64 && "Unsupported value size");
    assert(VA.getLocVT().getSizeInBits() <= 64 && "Unsupported location size");

    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    Register ExtReg = extendRegister(ValVReg, VA);
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOStore, VA.getLocVT().getStoreSize(),
        Align(1));
    MIRBuilder.buildStore(ExtReg, Addr, *MMO);
  }

  unsigned assignCustomValue(const CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs) override {
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

    bool IsLittle = MIRBuilder.getMF().getSubtarget<ARMSubtarget>().isLittle();
    if (!IsLittle)
      std::swap(NewRegs[0], NewRegs[1]);

    assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
    assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

    return 1;
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    if (AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State))
      return true;

    StackSize =
        std::max(StackSize, static_cast<uint64_t>(State.getNextStackOffset()));
    return false;
  }

  MachineInstrBuilder &MIB;
  uint64_t StackSize = 0;
};

} // end anonymous namespace

void ARMCallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                        SmallVectorImpl<ArgInfo> &SplitArgs,
                                        MachineFunction &MF) const {
  const ARMTargetLowering &TLI = *getTLI<ARMTargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();
  const DataLayout &DL = MF.getDataLayout();
  const Function &F = MF.getFunction();

  SmallVector<EVT, 4> SplitVTs;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, nullptr, nullptr, 0);
  assert(OrigArg.Regs.size() == SplitVTs.size() && "Regs / types mismatch");

  if (SplitVTs.size() == 1) {
    // Even if there is no splitting to do, we still want to replace the
    // original type (e.g. pointer type -> integer).
    auto Flags = OrigArg.Flags[0];
    Flags.setOrigAlign(DL.getABITypeAlign(OrigArg.Ty));
    SplitArgs.emplace_back(OrigArg.Regs[0], SplitVTs[0].getTypeForEVT(Ctx),
                           Flags, OrigArg.IsFixed);
    return;
  }

  // Create one ArgInfo for each virtual register.
  for (unsigned i = 0, e = SplitVTs.size(); i != e; ++i) {
    EVT SplitVT = SplitVTs[i];
    Type *SplitTy = SplitVT.getTypeForEVT(Ctx);
    auto Flags = OrigArg.Flags[0];

    Flags.setOrigAlign(DL.getABITypeAlign(SplitTy));

    bool NeedsConsecutiveRegisters =
        TLI.functionArgumentNeedsConsecutiveRegisters(
            SplitTy, F.getCallingConv(), F.isVarArg());
    if (NeedsConsecutiveRegisters) {
      Flags.setInConsecutiveRegs();
      if (i == e - 1)
        Flags.setInConsecutiveRegsLast();
    }

    // FIXME: We also want to split SplitTy further.
    Register PartReg = OrigArg.Regs[i];
    SplitArgs.emplace_back(PartReg, SplitTy, Flags, OrigArg.IsFixed);
  }
}

/// Lower the return value for the already existing \p Ret. This assumes that
/// \p MIRBuilder's insertion point is correct.
bool ARMCallLowering::lowerReturnVal(MachineIRBuilder &MIRBuilder,
                                     const Value *Val, ArrayRef<Register> VRegs,
                                     MachineInstrBuilder &Ret) const {
  if (!Val)
    // Nothing to do here.
    return true;

  auto &MF = MIRBuilder.getMF();
  const auto &F = MF.getFunction();

  auto DL = MF.getDataLayout();
  auto &TLI = *getTLI<ARMTargetLowering>();
  if (!isSupportedType(DL, TLI, Val->getType()))
    return false;

  ArgInfo OrigRetInfo(VRegs, Val->getType());
  setArgFlags(OrigRetInfo, AttributeList::ReturnIndex, DL, F);

  SmallVector<ArgInfo, 4> SplitRetInfos;
  splitToValueTypes(OrigRetInfo, SplitRetInfos, MF);

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForReturn(F.getCallingConv(), F.isVarArg());

  OutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret, AssignFn);
  return handleAssignments(MIRBuilder, SplitRetInfos, RetHandler);
}

bool ARMCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val,
                                  ArrayRef<Register> VRegs) const {
  assert(!Val == VRegs.empty() && "Return value without a vreg");

  auto const &ST = MIRBuilder.getMF().getSubtarget<ARMSubtarget>();
  unsigned Opcode = ST.getReturnOpcode();
  auto Ret = MIRBuilder.buildInstrNoInsert(Opcode).add(predOps(ARMCC::AL));

  if (!lowerReturnVal(MIRBuilder, Val, VRegs, Ret))
    return false;

  MIRBuilder.insertInstr(Ret);
  return true;
}

namespace {

/// Helper class for values coming in through an ABI boundary (used for handling
/// formal arguments and call return values).
struct IncomingValueHandler : public CallLowering::ValueHandler {
  IncomingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       CCAssignFn AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn) {}

  bool isIncomingArgumentHandler() const override { return true; }

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    auto &MFI = MIRBuilder.getMF().getFrameInfo();

    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

    return MIRBuilder.buildFrameIndex(LLT::pointer(MPO.getAddrSpace(), 32), FI)
        .getReg(0);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    if (VA.getLocInfo() == CCValAssign::SExt ||
        VA.getLocInfo() == CCValAssign::ZExt) {
      // If the value is zero- or sign-extended, its size becomes 4 bytes, so
      // that's what we should load.
      Size = 4;
      assert(MRI.getType(ValVReg).isScalar() && "Only scalars supported atm");

      auto LoadVReg = buildLoad(LLT::scalar(32), Addr, Size, MPO);
      MIRBuilder.buildTrunc(ValVReg, LoadVReg);
    } else {
      // If the value is not extended, a simple load will suffice.
      buildLoad(ValVReg, Addr, Size, MPO);
    }
  }

  MachineInstrBuilder buildLoad(const DstOp &Res, Register Addr, uint64_t Size,
                                MachinePointerInfo &MPO) {
    MachineFunction &MF = MIRBuilder.getMF();

    auto MMO = MF.getMachineMemOperand(MPO, MachineMemOperand::MOLoad, Size,
                                       inferAlignFromPtrInfo(MF, MPO));
    return MIRBuilder.buildLoad(Res, Addr, *MMO);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    auto ValSize = VA.getValVT().getSizeInBits();
    auto LocSize = VA.getLocVT().getSizeInBits();

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

  unsigned assignCustomValue(const ARMCallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs) override {
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

    assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
    assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

    bool IsLittle = MIRBuilder.getMF().getSubtarget<ARMSubtarget>().isLittle();
    if (!IsLittle)
      std::swap(NewRegs[0], NewRegs[1]);

    MIRBuilder.buildMerge(Arg.Regs[0], NewRegs);

    return 1;
  }

  /// Marking a physical register as used is different between formal
  /// parameters, where it's a basic block live-in, and call returns, where it's
  /// an implicit-def of the call instruction.
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;
};

struct FormalArgHandler : public IncomingValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn AssignFn)
      : IncomingValueHandler(MIRBuilder, MRI, AssignFn) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

} // end anonymous namespace

bool ARMCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  auto &TLI = *getTLI<ARMTargetLowering>();
  auto Subtarget = TLI.getSubtarget();

  if (Subtarget->isThumb1Only())
    return false;

  // Quick exit if there aren't any args
  if (F.arg_empty())
    return true;

  if (F.isVarArg())
    return false;

  auto &MF = MIRBuilder.getMF();
  auto &MBB = MIRBuilder.getMBB();
  auto DL = MF.getDataLayout();

  for (auto &Arg : F.args()) {
    if (!isSupportedType(DL, TLI, Arg.getType()))
      return false;
    if (Arg.hasPassPointeeByValueCopyAttr())
      return false;
  }

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), F.isVarArg());

  FormalArgHandler ArgHandler(MIRBuilder, MIRBuilder.getMF().getRegInfo(),
                              AssignFn);

  SmallVector<ArgInfo, 8> SplitArgInfos;
  unsigned Idx = 0;
  for (auto &Arg : F.args()) {
    ArgInfo OrigArgInfo(VRegs[Idx], Arg.getType());

    setArgFlags(OrigArgInfo, Idx + AttributeList::FirstArgIndex, DL, F);
    splitToValueTypes(OrigArgInfo, SplitArgInfos, MF);

    Idx++;
  }

  if (!MBB.empty())
    MIRBuilder.setInstr(*MBB.begin());

  if (!handleAssignments(MIRBuilder, SplitArgInfos, ArgHandler))
    return false;

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);
  return true;
}

namespace {

struct CallReturnHandler : public IncomingValueHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : IncomingValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
};

// FIXME: This should move to the ARMSubtarget when it supports all the opcodes.
unsigned getCallOpcode(const ARMSubtarget &STI, bool isDirect) {
  if (isDirect)
    return STI.isThumb() ? ARM::tBL : ARM::BL;

  if (STI.isThumb())
    return ARM::tBLXr;

  if (STI.hasV5TOps())
    return ARM::BLX;

  if (STI.hasV4TOps())
    return ARM::BX_CALL;

  return ARM::BMOVPCRX_CALL;
}
} // end anonymous namespace

bool ARMCallLowering::lowerCall(MachineIRBuilder &MIRBuilder, CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const auto &TLI = *getTLI<ARMTargetLowering>();
  const auto &DL = MF.getDataLayout();
  const auto &STI = MF.getSubtarget<ARMSubtarget>();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (STI.genLongCalls())
    return false;

  if (STI.isThumb1Only())
    return false;

  auto CallSeqStart = MIRBuilder.buildInstr(ARM::ADJCALLSTACKDOWN);

  // Create the call instruction so we can add the implicit uses of arg
  // registers, but don't insert it yet.
  bool IsDirect = !Info.Callee.isReg();
  auto CallOpcode = getCallOpcode(STI, IsDirect);
  auto MIB = MIRBuilder.buildInstrNoInsert(CallOpcode);

  bool IsThumb = STI.isThumb();
  if (IsThumb)
    MIB.add(predOps(ARMCC::AL));

  MIB.add(Info.Callee);
  if (!IsDirect) {
    auto CalleeReg = Info.Callee.getReg();
    if (CalleeReg && !Register::isPhysicalRegister(CalleeReg)) {
      unsigned CalleeIdx = IsThumb ? 2 : 0;
      MIB->getOperand(CalleeIdx).setReg(constrainOperandRegClass(
          MF, *TRI, MRI, *STI.getInstrInfo(), *STI.getRegBankInfo(),
          *MIB.getInstr(), MIB->getDesc(), Info.Callee, CalleeIdx));
    }
  }

  MIB.addRegMask(TRI->getCallPreservedMask(MF, Info.CallConv));

  bool IsVarArg = false;
  SmallVector<ArgInfo, 8> ArgInfos;
  for (auto Arg : Info.OrigArgs) {
    if (!isSupportedType(DL, TLI, Arg.Ty))
      return false;

    if (!Arg.IsFixed)
      IsVarArg = true;

    if (Arg.Flags[0].isByVal())
      return false;

    splitToValueTypes(Arg, ArgInfos, MF);
  }

  auto ArgAssignFn = TLI.CCAssignFnForCall(Info.CallConv, IsVarArg);
  OutgoingValueHandler ArgHandler(MIRBuilder, MRI, MIB, ArgAssignFn);
  if (!handleAssignments(MIRBuilder, ArgInfos, ArgHandler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  if (!Info.OrigRet.Ty->isVoidTy()) {
    if (!isSupportedType(DL, TLI, Info.OrigRet.Ty))
      return false;

    ArgInfos.clear();
    splitToValueTypes(Info.OrigRet, ArgInfos, MF);
    auto RetAssignFn = TLI.CCAssignFnForReturn(Info.CallConv, IsVarArg);
    CallReturnHandler RetHandler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, ArgInfos, RetHandler))
      return false;
  }

  // We now know the size of the stack - update the ADJCALLSTACKDOWN
  // accordingly.
  CallSeqStart.addImm(ArgHandler.StackSize).addImm(0).add(predOps(ARMCC::AL));

  MIRBuilder.buildInstr(ARM::ADJCALLSTACKUP)
      .addImm(ArgHandler.StackSize)
      .addImm(0)
      .add(predOps(ARMCC::AL));

  return true;
}
