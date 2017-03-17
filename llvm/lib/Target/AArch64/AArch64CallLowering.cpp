//===--- AArch64CallLowering.cpp - Call lowering --------------------------===//
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
#include "llvm/CodeGen/MachineValueType.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

AArch64CallLowering::AArch64CallLowering(const AArch64TargetLowering &TLI)
  : CallLowering(&TLI) {}

struct IncomingArgHandler : public CallLowering::ValueHandler {
  IncomingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), StackUsed(0) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);
    unsigned AddrReg = MRI.createGenericVirtualRegister(LLT::pointer(0, 64));
    MIRBuilder.buildFrameIndex(AddrReg, FI);
    StackUsed = std::max(StackUsed, Size + Offset);
    return AddrReg;
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    markPhysRegUsed(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
    // FIXME: assert extension
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, Size,
        0);
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;

  uint64_t StackUsed;
};

struct FormalArgHandler : public IncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn)
    : IncomingArgHandler(MIRBuilder, MRI, AssignFn) {}

  void markPhysRegUsed(unsigned PhysReg) override {
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
                     CCAssignFn *AssignFnVarArg)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB),
        AssignFnVarArg(AssignFnVarArg), StackSize(0) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    LLT p0 = LLT::pointer(0, 64);
    LLT s64 = LLT::scalar(64);
    unsigned SPReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(SPReg, AArch64::SP);

    unsigned OffsetReg = MRI.createGenericVirtualRegister(s64);
    MIRBuilder.buildConstant(OffsetReg, Offset);

    unsigned AddrReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildGEP(AddrReg, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
    return AddrReg;
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    unsigned ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOStore, Size, 0);
    MIRBuilder.buildStore(ValVReg, Addr, *MMO);
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info,
                 CCState &State) override {
    bool Res;
    if (Info.IsFixed)
      Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State);
    else
      Res = AssignFnVarArg(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State);

    StackSize = State.getNextStackOffset();
    return Res;
  }

  MachineInstrBuilder MIB;
  CCAssignFn *AssignFnVarArg;
  uint64_t StackSize;
};

void AArch64CallLowering::splitToValueTypes(
    const ArgInfo &OrigArg, SmallVectorImpl<ArgInfo> &SplitArgs,
    const DataLayout &DL, MachineRegisterInfo &MRI,
    const SplitArgTy &PerformArgSplit) const {
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() == 1) {
    // No splitting to do, but we want to replace the original type (e.g. [1 x
    // double] -> double).
    SplitArgs.emplace_back(OrigArg.Reg, SplitVTs[0].getTypeForEVT(Ctx),
                           OrigArg.Flags, OrigArg.IsFixed);
    return;
  }

  unsigned FirstRegIdx = SplitArgs.size();
  for (auto SplitVT : SplitVTs) {
    // FIXME: set split flags if they're actually used (e.g. i128 on AAPCS).
    Type *SplitTy = SplitVT.getTypeForEVT(Ctx);
    SplitArgs.push_back(
        ArgInfo{MRI.createGenericVirtualRegister(getLLTForType(*SplitTy, DL)),
                SplitTy, OrigArg.Flags, OrigArg.IsFixed});
  }

  for (unsigned i = 0; i < Offsets.size(); ++i)
    PerformArgSplit(SplitArgs[FirstRegIdx + i].Reg, Offsets[i] * 8);
}

bool AArch64CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                      const Value *Val, unsigned VReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();

  auto MIB = MIRBuilder.buildInstrNoInsert(AArch64::RET_ReallyLR);
  assert(((Val && VReg) || (!Val && !VReg)) && "Return value without a vreg");
  bool Success = true;
  if (VReg) {
    const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
    CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
    MachineRegisterInfo &MRI = MF.getRegInfo();
    auto &DL = F.getParent()->getDataLayout();

    ArgInfo OrigArg{VReg, Val->getType()};
    setArgFlags(OrigArg, AttributeSet::ReturnIndex, DL, F);

    SmallVector<ArgInfo, 8> SplitArgs;
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        MIRBuilder.buildExtract(Reg, VReg, Offset);
                      });

    OutgoingArgHandler Handler(MIRBuilder, MRI, MIB, AssignFn, AssignFn);
    Success = handleAssignments(MIRBuilder, SplitArgs, Handler);
  }

  MIRBuilder.insertInstr(MIB);
  return Success;
}

bool AArch64CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                               const Function &F,
                                               ArrayRef<unsigned> VRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned i = 0;
  for (auto &Arg : F.args()) {
    ArgInfo OrigArg{VRegs[i], Arg.getType()};
    setArgFlags(OrigArg, i + 1, DL, F);
    bool Split = false;
    LLT Ty = MRI.getType(VRegs[i]);
    unsigned Dst = VRegs[i];

    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        if (!Split) {
                          Split = true;
                          Dst = MRI.createGenericVirtualRegister(Ty);
                          MIRBuilder.buildUndef(Dst);
                        }
                        unsigned Tmp = MRI.createGenericVirtualRegister(Ty);
                        MIRBuilder.buildInsert(Tmp, Dst, Reg, Offset);
                        Dst = Tmp;
                      });

    if (Dst != VRegs[i])
      MIRBuilder.buildCopy(VRegs[i], Dst);
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

  if (F.isVarArg()) {
    if (!MF.getSubtarget<AArch64Subtarget>().isTargetDarwin()) {
      // FIXME: we need to reimplement saveVarArgsRegisters from
      // AArch64ISelLowering.
      return false;
    }

    // We currently pass all varargs at 8-byte alignment.
    uint64_t StackOffset = alignTo(Handler.StackUsed, 8);

    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    AArch64FunctionInfo *FuncInfo = MF.getInfo<AArch64FunctionInfo>();
    FuncInfo->setVarArgsStackIndex(MFI.CreateFixedObject(4, StackOffset, true));
  }

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return true;
}

bool AArch64CallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                    const MachineOperand &Callee,
                                    const ArgInfo &OrigRet,
                                    ArrayRef<ArgInfo> OrigArgs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  for (auto &OrigArg : OrigArgs) {
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        MIRBuilder.buildExtract(Reg, OrigArg.Reg, Offset);
                      });
  }

  // Find out which ABI gets to decide where things go.
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *AssignFnFixed =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);
  CCAssignFn *AssignFnVarArg =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/true);

  auto CallSeqStart = MIRBuilder.buildInstr(AArch64::ADJCALLSTACKDOWN);

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  auto MIB = MIRBuilder.buildInstrNoInsert(Callee.isReg() ? AArch64::BLR
                                                          : AArch64::BL);
  MIB.add(Callee);

  // Tell the call which registers are clobbered.
  auto TRI = MF.getSubtarget().getRegisterInfo();
  MIB.addRegMask(TRI->getCallPreservedMask(MF, F.getCallingConv()));

  // Do the actual argument marshalling.
  SmallVector<unsigned, 8> PhysRegs;
  OutgoingArgHandler Handler(MIRBuilder, MRI, MIB, AssignFnFixed,
                             AssignFnVarArg);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  // If Callee is a reg, since it is used by a target specific
  // instruction, it must have a register class matching the
  // constraint of that instruction.
  if (Callee.isReg())
    MIB->getOperand(0).setReg(constrainOperandRegClass(
        MF, *TRI, MRI, *MF.getSubtarget().getInstrInfo(),
        *MF.getSubtarget().getRegBankInfo(), *MIB, MIB->getDesc(),
        Callee.getReg(), 0));

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arugments, the physical register must be an
  // implicit-define of the call instruction.
  CCAssignFn *RetAssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
  if (OrigRet.Reg) {
    SplitArgs.clear();

    SmallVector<uint64_t, 8> RegOffsets;
    SmallVector<unsigned, 8> SplitRegs;
    splitToValueTypes(OrigRet, SplitArgs, DL, MRI,
                      [&](unsigned Reg, uint64_t Offset) {
                        RegOffsets.push_back(Offset);
                        SplitRegs.push_back(Reg);
                      });

    CallReturnHandler Handler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
      return false;

    if (!RegOffsets.empty())
      MIRBuilder.buildSequence(OrigRet.Reg, SplitRegs, RegOffsets);
  }

  CallSeqStart.addImm(Handler.StackSize);
  MIRBuilder.buildInstr(AArch64::ADJCALLSTACKUP)
      .addImm(Handler.StackSize)
      .addImm(0);

  return true;
}
