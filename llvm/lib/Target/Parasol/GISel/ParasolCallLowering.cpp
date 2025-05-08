//===-- ParasolCallLowering.cpp - Call lowering for GlobalISel ------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "ParasolCallLowering.h"
#include "ParasolISelLowering.h"
#include "ParasolRegisterInfo.h"
#include "ParasolSubtarget.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parasol-call-lowering"

using namespace llvm;

#include "ParasolGenCallingConv.inc"

#define NUM_ARGUMENT_REGISTERS 32

namespace {
struct ParasolIncomingValueAssigner
    : public CallLowering::IncomingValueAssigner {
  ParasolIncomingValueAssigner(CCAssignFn *AssignFn_,
                               CCAssignFn *AssignFnVarArg_)
      : IncomingValueAssigner(AssignFn_, AssignFnVarArg_) {}

  bool assignArg(unsigned ValNo, EVT OrigVT, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    return IncomingValueAssigner::assignArg(ValNo, OrigVT, ValVT, LocVT,
                                            LocInfo, Info, Flags, State);
  }
};

struct ParasolOutgoingValueAssigner
    : public CallLowering::OutgoingValueAssigner {
  const ParasolSubtarget &Subtarget;

  /// Track if this is used for a return instead of function argument
  /// passing. We apply a hack to i1/i8/i16 stack passed values, but do not use
  /// stack passed returns for them and cannot apply the type adjustment.
  bool IsReturn;

  ParasolOutgoingValueAssigner(CCAssignFn *AssignFn_,
                               CCAssignFn *AssignFnVarArg_,
                               const ParasolSubtarget &Subtarget_,
                               bool IsReturn)
      : OutgoingValueAssigner(AssignFn_, AssignFnVarArg_),
        Subtarget(Subtarget_), IsReturn(IsReturn) {}

  bool assignArg(unsigned ValNo, EVT OrigVT, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {

    if (State.isVarArg()) {
      llvm_unreachable("VarArg not supported");
    }

    bool Res;

    Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
    return Res;
  }
};

struct IncomingArgHandler : public CallLowering::IncomingValueHandler {
  IncomingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : IncomingValueHandler(MIRBuilder, MRI) {}

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override {

    llvm_unreachable("getStackAddress not supported");
  }

  LLT getStackValueStoreType(const DataLayout &DL, const CCValAssign &VA,
                             ISD::ArgFlagsTy Flags) const override {
    llvm_unreachable("getStackValueStoreType not supported");
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        const CCValAssign &VA) override {
    markPhysRegUsed(PhysReg);
    IncomingValueHandler::assignValueToReg(ValVReg, PhysReg, VA);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            const MachinePointerInfo &MPO,
                            const CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();

    LLT ValTy(VA.getValVT());
    LLT LocTy(VA.getLocVT());

    assert(LocTy.getSizeInBits() == MemTy.getSizeInBits());
    LocTy = MemTy;

    auto MMO = MF.getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, LocTy,
        inferAlignFromPtrInfo(MF, MPO));

    switch (VA.getLocInfo()) {
    case CCValAssign::LocInfo::ZExt:
      MIRBuilder.buildLoadInstr(TargetOpcode::G_ZEXTLOAD, ValVReg, Addr, *MMO);
      return;
    case CCValAssign::LocInfo::SExt:
      MIRBuilder.buildLoadInstr(TargetOpcode::G_SEXTLOAD, ValVReg, Addr, *MMO);
      return;
    default:
      MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
      return;
    }
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(MCRegister PhysReg) = 0;
};

struct FormalArgHandler : public IncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : IncomingArgHandler(MIRBuilder, MRI) {}

  void markPhysRegUsed(MCRegister PhysReg) override {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};
} // namespace

ParasolCallLowering::ParasolCallLowering(const ParasolTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool ParasolCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                      const Value *Val,
                                      ArrayRef<Register> VRegs,
                                      FunctionLoweringInfo &FLI,
                                      Register SwiftErrorVReg) const {
  if (!VRegs.empty())
    return false;

  MIRBuilder.buildInstr(Parasol::PseudoRET);
  return true;
}

bool ParasolCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {

  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  // We don't handle varargs for now. Bail out.
  if (F.isVarArg())
    return false;

  // if VRegs is empty, it means the function has no arguments.
  if (VRegs.empty())
    return true;

  SmallVector<ArgInfo, NUM_ARGUMENT_REGISTERS> SplitArgs;
  unsigned i = 0;

  for (auto &Arg : F.args()) {
    if (DL.getTypeStoreSize(Arg.getType()).isZero())
      continue;

    ArgInfo OrigArg{VRegs[i], Arg, i};
    setArgFlags(OrigArg, i + AttributeList::FirstArgIndex, DL, F);

    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    ++i;
  }

  if (!MBB.empty())
    MIRBuilder.setInstr(*MBB.begin());

  const ParasolTargetLowering &TLI = *getTLI<ParasolTargetLowering>();
  CCAssignFn *AssignFn = CC_Parasol;

  ParasolIncomingValueAssigner Assigner(AssignFn, AssignFn);
  FormalArgHandler Handler(MIRBuilder, MRI);
  SmallVector<CCValAssign, NUM_ARGUMENT_REGISTERS> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());
  if (!determineAssignments(Assigner, SplitArgs, CCInfo) ||
      !handleAssignments(Handler, SplitArgs, CCInfo, ArgLocs, MIRBuilder))
    return false;

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return true;
}

bool ParasolCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                    CallLoweringInfo &Info) const {
  llvm_unreachable("lowerCall not supported");
}
