//===--- AArch64CallLowering.cpp - Call lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/MachineValueType.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

#define DEBUG_TYPE "aarch64-call-lowering"

using namespace llvm;

AArch64CallLowering::AArch64CallLowering(const AArch64TargetLowering &TLI)
  : CallLowering(&TLI) {}

namespace {
struct IncomingArgHandler : public CallLowering::ValueHandler {
  IncomingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), StackUsed(0) {}

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);
    Register AddrReg = MRI.createGenericVirtualRegister(LLT::pointer(0, 64));
    MIRBuilder.buildFrameIndex(AddrReg, FI);
    StackUsed = std::max(StackUsed, Size + Offset);
    return AddrReg;
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    markPhysRegUsed(PhysReg);
    switch (VA.getLocInfo()) {
    default:
      MIRBuilder.buildCopy(ValVReg, PhysReg);
      break;
    case CCValAssign::LocInfo::SExt:
    case CCValAssign::LocInfo::ZExt:
    case CCValAssign::LocInfo::AExt: {
      auto Copy = MIRBuilder.buildCopy(LLT{VA.getLocVT()}, PhysReg);
      MIRBuilder.buildTrunc(ValVReg, Copy);
      break;
    }
    }
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    // FIXME: Get alignment
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, Size,
        1);
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;

  bool isIncomingArgumentHandler() const override { return true; }

  uint64_t StackUsed;
};

struct FormalArgHandler : public IncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn)
    : IncomingArgHandler(MIRBuilder, MRI, AssignFn) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

struct CallReturnHandler : public IncomingArgHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder MIB, CCAssignFn *AssignFn)
    : IncomingArgHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
};

struct OutgoingArgHandler : public CallLowering::ValueHandler {
  OutgoingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     MachineInstrBuilder MIB, CCAssignFn *AssignFn,
                     CCAssignFn *AssignFnVarArg, bool IsTailCall = false)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB),
        AssignFnVarArg(AssignFnVarArg), IsTailCall(IsTailCall), StackSize(0) {}

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    MachineFunction &MF = MIRBuilder.getMF();
    LLT p0 = LLT::pointer(0, 64);
    LLT s64 = LLT::scalar(64);

    if (IsTailCall) {
      // TODO: For -tailcallopt tail calls, Offset will need FPDiff like in
      // ISelLowering.
      int FI = MF.getFrameInfo().CreateFixedObject(Size, Offset, true);
      Register FIReg = MRI.createGenericVirtualRegister(p0);
      MIRBuilder.buildFrameIndex(FIReg, FI);
      MPO = MachinePointerInfo::getFixedStack(MF, FI);
      return FIReg;
    }

    Register SPReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(SPReg, Register(AArch64::SP));

    Register OffsetReg = MRI.createGenericVirtualRegister(s64);
    MIRBuilder.buildConstant(OffsetReg, Offset);

    Register AddrReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildGEP(AddrReg, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MF, Offset);
    return AddrReg;
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    if (VA.getLocInfo() == CCValAssign::LocInfo::AExt) {
      Size = VA.getLocVT().getSizeInBits() / 8;
      ValVReg = MIRBuilder.buildAnyExt(LLT::scalar(Size * 8), ValVReg)
                    ->getOperand(0)
                    .getReg();
    }
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOStore, Size, 1);
    MIRBuilder.buildStore(ValVReg, Addr, *MMO);
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info,
                 ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    bool Res;
    if (Info.IsFixed)
      Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
    else
      Res = AssignFnVarArg(ValNo, ValVT, LocVT, LocInfo, Flags, State);

    StackSize = State.getNextStackOffset();
    return Res;
  }

  MachineInstrBuilder MIB;
  CCAssignFn *AssignFnVarArg;
  bool IsTailCall;
  uint64_t StackSize;
};
} // namespace

void AArch64CallLowering::splitToValueTypes(
    const ArgInfo &OrigArg, SmallVectorImpl<ArgInfo> &SplitArgs,
    const DataLayout &DL, MachineRegisterInfo &MRI, CallingConv::ID CallConv) const {
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  if (OrigArg.Ty->isVoidTy())
    return;

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() == 1) {
    // No splitting to do, but we want to replace the original type (e.g. [1 x
    // double] -> double).
    SplitArgs.emplace_back(OrigArg.Regs[0], SplitVTs[0].getTypeForEVT(Ctx),
                           OrigArg.Flags[0], OrigArg.IsFixed);
    return;
  }

  // Create one ArgInfo for each virtual register in the original ArgInfo.
  assert(OrigArg.Regs.size() == SplitVTs.size() && "Regs / types mismatch");

  bool NeedsRegBlock = TLI.functionArgumentNeedsConsecutiveRegisters(
      OrigArg.Ty, CallConv, false);
  for (unsigned i = 0, e = SplitVTs.size(); i < e; ++i) {
    Type *SplitTy = SplitVTs[i].getTypeForEVT(Ctx);
    SplitArgs.emplace_back(OrigArg.Regs[i], SplitTy, OrigArg.Flags[0],
                           OrigArg.IsFixed);
    if (NeedsRegBlock)
      SplitArgs.back().Flags[0].setInConsecutiveRegs();
  }

  SplitArgs.back().Flags[0].setInConsecutiveRegsLast();
}

bool AArch64CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                      const Value *Val,
                                      ArrayRef<Register> VRegs,
                                      Register SwiftErrorVReg) const {
  auto MIB = MIRBuilder.buildInstrNoInsert(AArch64::RET_ReallyLR);
  assert(((Val && !VRegs.empty()) || (!Val && VRegs.empty())) &&
         "Return value without a vreg");

  bool Success = true;
  if (!VRegs.empty()) {
    MachineFunction &MF = MIRBuilder.getMF();
    const Function &F = MF.getFunction();

    MachineRegisterInfo &MRI = MF.getRegInfo();
    const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
    CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
    auto &DL = F.getParent()->getDataLayout();
    LLVMContext &Ctx = Val->getType()->getContext();

    SmallVector<EVT, 4> SplitEVTs;
    ComputeValueVTs(TLI, DL, Val->getType(), SplitEVTs);
    assert(VRegs.size() == SplitEVTs.size() &&
           "For each split Type there should be exactly one VReg.");

    SmallVector<ArgInfo, 8> SplitArgs;
    CallingConv::ID CC = F.getCallingConv();

    for (unsigned i = 0; i < SplitEVTs.size(); ++i) {
      if (TLI.getNumRegistersForCallingConv(Ctx, CC, SplitEVTs[i]) > 1) {
        LLVM_DEBUG(dbgs() << "Can't handle extended arg types which need split");
        return false;
      }

      Register CurVReg = VRegs[i];
      ArgInfo CurArgInfo = ArgInfo{CurVReg, SplitEVTs[i].getTypeForEVT(Ctx)};
      setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, F);

      // i1 is a special case because SDAG i1 true is naturally zero extended
      // when widened using ANYEXT. We need to do it explicitly here.
      if (MRI.getType(CurVReg).getSizeInBits() == 1) {
        CurVReg = MIRBuilder.buildZExt(LLT::scalar(8), CurVReg).getReg(0);
      } else {
        // Some types will need extending as specified by the CC.
        MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CC, SplitEVTs[i]);
        if (EVT(NewVT) != SplitEVTs[i]) {
          unsigned ExtendOp = TargetOpcode::G_ANYEXT;
          if (F.getAttributes().hasAttribute(AttributeList::ReturnIndex,
                                             Attribute::SExt))
            ExtendOp = TargetOpcode::G_SEXT;
          else if (F.getAttributes().hasAttribute(AttributeList::ReturnIndex,
                                                  Attribute::ZExt))
            ExtendOp = TargetOpcode::G_ZEXT;

          LLT NewLLT(NewVT);
          LLT OldLLT(MVT::getVT(CurArgInfo.Ty));
          CurArgInfo.Ty = EVT(NewVT).getTypeForEVT(Ctx);
          // Instead of an extend, we might have a vector type which needs
          // padding with more elements, e.g. <2 x half> -> <4 x half>.
          if (NewVT.isVector()) {
            if (OldLLT.isVector()) {
              if (NewLLT.getNumElements() > OldLLT.getNumElements()) {
                // We don't handle VA types which are not exactly twice the
                // size, but can easily be done in future.
                if (NewLLT.getNumElements() != OldLLT.getNumElements() * 2) {
                  LLVM_DEBUG(dbgs() << "Outgoing vector ret has too many elts");
                  return false;
                }
                auto Undef = MIRBuilder.buildUndef({OldLLT});
                CurVReg =
                    MIRBuilder.buildMerge({NewLLT}, {CurVReg, Undef.getReg(0)})
                        .getReg(0);
              } else {
                // Just do a vector extend.
                CurVReg = MIRBuilder.buildInstr(ExtendOp, {NewLLT}, {CurVReg})
                              .getReg(0);
              }
            } else if (NewLLT.getNumElements() == 2) {
              // We need to pad a <1 x S> type to <2 x S>. Since we don't have
              // <1 x S> vector types in GISel we use a build_vector instead
              // of a vector merge/concat.
              auto Undef = MIRBuilder.buildUndef({OldLLT});
              CurVReg =
                  MIRBuilder
                      .buildBuildVector({NewLLT}, {CurVReg, Undef.getReg(0)})
                      .getReg(0);
            } else {
              LLVM_DEBUG(dbgs() << "Could not handle ret ty");
              return false;
            }
          } else {
            // A scalar extend.
            CurVReg =
                MIRBuilder.buildInstr(ExtendOp, {NewLLT}, {CurVReg}).getReg(0);
          }
        }
      }
      if (CurVReg != CurArgInfo.Regs[0]) {
        CurArgInfo.Regs[0] = CurVReg;
        // Reset the arg flags after modifying CurVReg.
        setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, F);
      }
     splitToValueTypes(CurArgInfo, SplitArgs, DL, MRI, CC);
    }

    OutgoingArgHandler Handler(MIRBuilder, MRI, MIB, AssignFn, AssignFn);
    Success = handleAssignments(MIRBuilder, SplitArgs, Handler);
  }

  if (SwiftErrorVReg) {
    MIB.addUse(AArch64::X21, RegState::Implicit);
    MIRBuilder.buildCopy(AArch64::X21, SwiftErrorVReg);
  }

  MIRBuilder.insertInstr(MIB);
  return Success;
}

bool AArch64CallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned i = 0;
  for (auto &Arg : F.args()) {
    if (DL.getTypeStoreSize(Arg.getType()) == 0)
      continue;

    ArgInfo OrigArg{VRegs[i], Arg.getType()};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, F);

    splitToValueTypes(OrigArg, SplitArgs, DL, MRI, F.getCallingConv());
    ++i;
  }

  if (!MBB.empty())
    MIRBuilder.setInstr(*MBB.begin());

  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);

  FormalArgHandler Handler(MIRBuilder, MRI, AssignFn);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;

  AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  uint64_t StackOffset = Handler.StackUsed;
  if (F.isVarArg()) {
    auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
    if (!Subtarget.isTargetDarwin()) {
        // FIXME: we need to reimplement saveVarArgsRegisters from
      // AArch64ISelLowering.
      return false;
    }

    // We currently pass all varargs at 8-byte alignment, or 4 in ILP32.
    StackOffset = alignTo(Handler.StackUsed, Subtarget.isTargetILP32() ? 4 : 8);

    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    FuncInfo->setVarArgsStackIndex(MFI.CreateFixedObject(4, StackOffset, true));
  }

  // TODO: Port checks for stack to restore for -tailcallopt from ISelLowering.
  // We need to keep track of the size of function stacks for tail call
  // optimization. When we tail call, we need to check if the callee's arguments
  // will fit on the caller's stack. So, whenever we lower formal arguments,
  // we should keep track of this information, since we might lower a tail call
  // in this function later.
  FuncInfo->setBytesInStackArgArea(StackOffset);

  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  if (Subtarget.hasCustomCallingConv())
    Subtarget.getRegisterInfo()->UpdateCustomCalleeSavedRegs(MF);

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return true;
}

/// Return true if the calling convention is one that we can guarantee TCO for.
static bool canGuaranteeTCO(CallingConv::ID CC) {
  return CC == CallingConv::Fast;
}

/// Return true if we might ever do TCO for calls with this calling convention.
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
  case CallingConv::PreserveMost:
  case CallingConv::Swift:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}

bool AArch64CallLowering::doCallerAndCalleePassArgsTheSameWay(
    CallLoweringInfo &Info, MachineFunction &MF,
    SmallVectorImpl<ArgInfo> &InArgs) const {
  const Function &CallerF = MF.getFunction();
  CallingConv::ID CalleeCC = Info.CallConv;
  CallingConv::ID CallerCC = CallerF.getCallingConv();

  // If the calling conventions match, then everything must be the same.
  if (CalleeCC == CallerCC)
    return true;

  // Check if the caller and callee will handle arguments in the same way.
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *CalleeAssignFn = TLI.CCAssignFnForCall(CalleeCC, Info.IsVarArg);
  CCAssignFn *CallerAssignFn =
      TLI.CCAssignFnForCall(CallerCC, CallerF.isVarArg());

  if (!resultsCompatible(Info, MF, InArgs, *CalleeAssignFn, *CallerAssignFn))
    return false;

  // Make sure that the caller and callee preserve all of the same registers.
  auto TRI = MF.getSubtarget<AArch64Subtarget>().getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);
  const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
  if (MF.getSubtarget<AArch64Subtarget>().hasCustomCallingConv()) {
    TRI->UpdateCustomCallPreservedMask(MF, &CallerPreserved);
    TRI->UpdateCustomCallPreservedMask(MF, &CalleePreserved);
  }

  return TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved);
}

bool AArch64CallLowering::areCalleeOutgoingArgsTailCallable(
    CallLoweringInfo &Info, MachineFunction &MF,
    SmallVectorImpl<ArgInfo> &OutArgs) const {
  // If there are no outgoing arguments, then we are done.
  if (OutArgs.empty())
    return true;

  const Function &CallerF = MF.getFunction();
  CallingConv::ID CalleeCC = Info.CallConv;
  CallingConv::ID CallerCC = CallerF.getCallingConv();
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();

  // We have outgoing arguments. Make sure that we can tail call with them.
  SmallVector<CCValAssign, 16> OutLocs;
  CCState OutInfo(CalleeCC, false, MF, OutLocs, CallerF.getContext());

  if (!analyzeArgInfo(OutInfo, OutArgs,
                      *TLI.CCAssignFnForCall(CalleeCC, Info.IsVarArg))) {
    LLVM_DEBUG(dbgs() << "... Could not analyze call operands.\n");
    return false;
  }

  // Make sure that they can fit on the caller's stack.
  const AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
  if (OutInfo.getNextStackOffset() > FuncInfo->getBytesInStackArgArea()) {
    LLVM_DEBUG(dbgs() << "... Cannot fit call operands on caller's stack.\n");
    return false;
  }

  // Verify that the parameters in callee-saved registers match.
  // TODO: Port this over to CallLowering as general code once swiftself is
  // supported.
  auto TRI = MF.getSubtarget<AArch64Subtarget>().getRegisterInfo();
  const uint32_t *CallerPreservedMask = TRI->getCallPreservedMask(MF, CallerCC);
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned i = 0; i < OutLocs.size(); ++i) {
    auto &ArgLoc = OutLocs[i];
    // If it's not a register, it's fine.
    if (!ArgLoc.isRegLoc()) {
      if (Info.IsVarArg) {
        // Be conservative and disallow variadic memory operands to match SDAG's
        // behaviour.
        // FIXME: If the caller's calling convention is C, then we can
        // potentially use its argument area. However, for cases like fastcc,
        // we can't do anything.
        LLVM_DEBUG(
            dbgs()
            << "... Cannot tail call vararg function with stack arguments\n");
        return false;
      }
      continue;
    }

    Register Reg = ArgLoc.getLocReg();

    // Only look at callee-saved registers.
    if (MachineOperand::clobbersPhysReg(CallerPreservedMask, Reg))
      continue;

    LLVM_DEBUG(
        dbgs()
        << "... Call has an argument passed in a callee-saved register.\n");

    // Check if it was copied from.
    ArgInfo &OutInfo = OutArgs[i];

    if (OutInfo.Regs.size() > 1) {
      LLVM_DEBUG(
          dbgs() << "... Cannot handle arguments in multiple registers.\n");
      return false;
    }

    // Check if we copy the register, walking through copies from virtual
    // registers. Note that getDefIgnoringCopies does not ignore copies from
    // physical registers.
    MachineInstr *RegDef = getDefIgnoringCopies(OutInfo.Regs[0], MRI);
    if (!RegDef || RegDef->getOpcode() != TargetOpcode::COPY) {
      LLVM_DEBUG(
          dbgs()
          << "... Parameter was not copied into a VReg, cannot tail call.\n");
      return false;
    }

    // Got a copy. Verify that it's the same as the register we want.
    Register CopyRHS = RegDef->getOperand(1).getReg();
    if (CopyRHS != Reg) {
      LLVM_DEBUG(dbgs() << "... Callee-saved register was not copied into "
                           "VReg, cannot tail call.\n");
      return false;
    }
  }

  return true;
}

bool AArch64CallLowering::isEligibleForTailCallOptimization(
    MachineIRBuilder &MIRBuilder, CallLoweringInfo &Info,
    SmallVectorImpl<ArgInfo> &InArgs,
    SmallVectorImpl<ArgInfo> &OutArgs) const {
  CallingConv::ID CalleeCC = Info.CallConv;
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &CallerF = MF.getFunction();

  LLVM_DEBUG(dbgs() << "Attempting to lower call as tail call\n");

  if (Info.SwiftErrorVReg) {
    // TODO: We should handle this.
    // Note that this is also handled by the check for no outgoing arguments.
    // Proactively disabling this though, because the swifterror handling in
    // lowerCall inserts a COPY *after* the location of the call.
    LLVM_DEBUG(dbgs() << "... Cannot handle tail calls with swifterror yet.\n");
    return false;
  }

  if (!mayTailCallThisCC(CalleeCC)) {
    LLVM_DEBUG(dbgs() << "... Calling convention cannot be tail called.\n");
    return false;
  }

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible (see
  // X86).
  //
  // FIXME: In AArch64ISelLowering, this isn't worked around. Can/should we try
  // it?
  //
  // On Windows, "inreg" attributes signify non-aggregate indirect returns.
  // In this case, it is necessary to save/restore X0 in the callee. Tail
  // call opt interferes with this. So we disable tail call opt when the
  // caller has an argument with "inreg" attribute.
  //
  // FIXME: Check whether the callee also has an "inreg" argument.
  //
  // When the caller has a swifterror argument, we don't want to tail call
  // because would have to move into the swifterror register before the
  // tail call.
  if (any_of(CallerF.args(), [](const Argument &A) {
        return A.hasByValAttr() || A.hasInRegAttr() || A.hasSwiftErrorAttr();
      })) {
    LLVM_DEBUG(dbgs() << "... Cannot tail call from callers with byval, "
                         "inreg, or swifterror arguments\n");
    return false;
  }

  // Externally-defined functions with weak linkage should not be
  // tail-called on AArch64 when the OS does not support dynamic
  // pre-emption of symbols, as the AAELF spec requires normal calls
  // to undefined weak functions to be replaced with a NOP or jump to the
  // next instruction. The behaviour of branch instructions in this
  // situation (as used for tail calls) is implementation-defined, so we
  // cannot rely on the linker replacing the tail call with a return.
  if (Info.Callee.isGlobal()) {
    const GlobalValue *GV = Info.Callee.getGlobal();
    const Triple &TT = MF.getTarget().getTargetTriple();
    if (GV->hasExternalWeakLinkage() &&
        (!TT.isOSWindows() || TT.isOSBinFormatELF() ||
         TT.isOSBinFormatMachO())) {
      LLVM_DEBUG(dbgs() << "... Cannot tail call externally-defined function "
                           "with weak linkage for this OS.\n");
      return false;
    }
  }

  // If we have -tailcallopt and matching CCs, at this point, we could return
  // true. However, we don't have full tail call support yet. So, continue
  // checking. We want to emit a sibling call.

  // I want anyone implementing a new calling convention to think long and hard
  // about this assert.
  assert((!Info.IsVarArg || CalleeCC == CallingConv::C) &&
         "Unexpected variadic calling convention");

  // Before we can musttail varargs, we need to forward parameters like in
  // r345641. Make sure that we don't enable musttail with varargs without
  // addressing that!
  assert(!(Info.IsVarArg && Info.IsMustTailCall) &&
         "musttail support for varargs not implemented yet!");

  // Verify that the incoming and outgoing arguments from the callee are
  // safe to tail call.
  if (!doCallerAndCalleePassArgsTheSameWay(Info, MF, InArgs)) {
    LLVM_DEBUG(
        dbgs()
        << "... Caller and callee have incompatible calling conventions.\n");
    return false;
  }

  if (!areCalleeOutgoingArgsTailCallable(Info, MF, OutArgs))
    return false;

  LLVM_DEBUG(
      dbgs() << "... Call is eligible for tail call optimization.\n");
  return true;
}

static unsigned getCallOpcode(const Function &CallerF, bool IsIndirect,
                              bool IsTailCall) {
  if (!IsTailCall)
    return IsIndirect ? AArch64::BLR : AArch64::BL;

  if (!IsIndirect)
    return AArch64::TCRETURNdi;

  // When BTI is enabled, we need to use TCRETURNriBTI to make sure that we use
  // x16 or x17.
  if (CallerF.hasFnAttribute("branch-target-enforcement"))
    return AArch64::TCRETURNriBTI;

  return AArch64::TCRETURNri;
}

bool AArch64CallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                    CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();

  if (Info.IsMustTailCall) {
    // TODO: Until we lower all tail calls, we should fall back on this.
    LLVM_DEBUG(dbgs() << "Cannot lower musttail calls yet.\n");
    return false;
  }

  if (Info.IsTailCall && MF.getTarget().Options.GuaranteedTailCallOpt) {
    // TODO: Until we lower all tail calls, we should fall back on this.
    LLVM_DEBUG(dbgs() << "Cannot handle -tailcallopt yet.\n");
    return false;
  }

  SmallVector<ArgInfo, 8> OutArgs;
  for (auto &OrigArg : Info.OrigArgs) {
    splitToValueTypes(OrigArg, OutArgs, DL, MRI, Info.CallConv);
    // AAPCS requires that we zero-extend i1 to 8 bits by the caller.
    if (OrigArg.Ty->isIntegerTy(1))
      OutArgs.back().Flags[0].setZExt();
  }

  SmallVector<ArgInfo, 8> InArgs;
  if (!Info.OrigRet.Ty->isVoidTy())
    splitToValueTypes(Info.OrigRet, InArgs, DL, MRI, F.getCallingConv());

  bool IsSibCall = Info.IsTailCall && isEligibleForTailCallOptimization(
                                          MIRBuilder, Info, InArgs, OutArgs);
  if (IsSibCall)
    MF.getFrameInfo().setHasTailCall();

  // Find out which ABI gets to decide where things go.
  CCAssignFn *AssignFnFixed =
      TLI.CCAssignFnForCall(Info.CallConv, /*IsVarArg=*/false);
  CCAssignFn *AssignFnVarArg =
      TLI.CCAssignFnForCall(Info.CallConv, /*IsVarArg=*/true);

  // If we have a sibling call, then we don't have to adjust the stack.
  // Otherwise, we need to adjust it.
  MachineInstrBuilder CallSeqStart;
  if (!IsSibCall)
    CallSeqStart = MIRBuilder.buildInstr(AArch64::ADJCALLSTACKDOWN);

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  unsigned Opc = getCallOpcode(F, Info.Callee.isReg(), IsSibCall);

  // TODO: Right now, regbankselect doesn't know how to handle the rtcGPR64
  // register class. Until we can do that, we should fall back here.
  if (Opc == AArch64::TCRETURNriBTI) {
    LLVM_DEBUG(
        dbgs() << "Cannot lower indirect tail calls with BTI enabled yet.\n");
    return false;
  }

  auto MIB = MIRBuilder.buildInstrNoInsert(Opc);
  MIB.add(Info.Callee);

  // Add the byte offset for the tail call. We only have sibling calls, so this
  // is always 0.
  // TODO: Handle tail calls where we will have a different value here.
  if (IsSibCall)
    MIB.addImm(0);

  // Tell the call which registers are clobbered.
  auto TRI = MF.getSubtarget<AArch64Subtarget>().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, F.getCallingConv());
  if (MF.getSubtarget<AArch64Subtarget>().hasCustomCallingConv())
    TRI->UpdateCustomCallPreservedMask(MF, &Mask);
  MIB.addRegMask(Mask);

  if (TRI->isAnyArgRegReserved(MF))
    TRI->emitReservedArgRegCallError(MF);

  // Do the actual argument marshalling.
  SmallVector<unsigned, 8> PhysRegs;
  OutgoingArgHandler Handler(MIRBuilder, MRI, MIB, AssignFnFixed,
                             AssignFnVarArg, IsSibCall);
  if (!handleAssignments(MIRBuilder, OutArgs, Handler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  // If Callee is a reg, since it is used by a target specific
  // instruction, it must have a register class matching the
  // constraint of that instruction.
  if (Info.Callee.isReg())
    MIB->getOperand(0).setReg(constrainOperandRegClass(
        MF, *TRI, MRI, *MF.getSubtarget().getInstrInfo(),
        *MF.getSubtarget().getRegBankInfo(), *MIB, MIB->getDesc(), Info.Callee,
        0));

  // If we're tail calling, then we're the return from the block. So, we don't
  // want to copy anything.
  if (IsSibCall)
    return true;

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arugments, the physical register must be an
  // implicit-define of the call instruction.
  if (!Info.OrigRet.Ty->isVoidTy()) {
    CCAssignFn *RetAssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
    CallReturnHandler Handler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, InArgs, Handler))
      return false;
  }

  if (Info.SwiftErrorVReg) {
    MIB.addDef(AArch64::X21, RegState::Implicit);
    MIRBuilder.buildCopy(Info.SwiftErrorVReg, Register(AArch64::X21));
  }

  CallSeqStart.addImm(Handler.StackSize).addImm(0);
  MIRBuilder.buildInstr(AArch64::ADJCALLSTACKUP)
      .addImm(Handler.StackSize)
      .addImm(0);

  return true;
}
