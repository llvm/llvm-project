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
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

AArch64CallLowering::AArch64CallLowering(const AArch64TargetLowering &TLI)
  : CallLowering(&TLI) {
}


bool AArch64CallLowering::handleAssignments(MachineIRBuilder &MIRBuilder,
                                            CCAssignFn *AssignFn,
                                            ArrayRef<ArgInfo> Args,
                                            AssignFnTy AssignValToReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  unsigned NumArgs = Args.size();
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT CurVT = MVT::getVT(Args[i].Ty);
    if (AssignFn(i, CurVT, CurVT, CCValAssign::Full, Args[i].Flags, CCInfo))
      return false;
  }
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // FIXME: Support non-register argument.
    if (!VA.isRegLoc())
      return false;

    // Everything checks out, tell the caller where we've decided this
    // parameter/return value should go.
    AssignValToReg(MIRBuilder, Args[i].Ty, Args[i].Reg, VA);
  }
  return true;
}

void AArch64CallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                            SmallVectorImpl<ArgInfo> &SplitArgs,
                                            const DataLayout &DL,
                                            MachineRegisterInfo &MRI,
                                            SplitArgTy PerformArgSplit) const {
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() == 1) {
    // No splitting to do, just forward the input directly.
    SplitArgs.push_back(OrigArg);
    return;
  }

  unsigned FirstRegIdx = SplitArgs.size();
  for (auto SplitVT : SplitVTs) {
    // FIXME: set split flags if they're actually used (e.g. i128 on AAPCS).
    Type *SplitTy = SplitVT.getTypeForEVT(Ctx);
    SplitArgs.push_back(
        ArgInfo{MRI.createGenericVirtualRegister(LLT{*SplitTy, DL}), SplitTy,
                OrigArg.Flags});
  }

  SmallVector<uint64_t, 4> BitOffsets;
  for (auto Offset : Offsets)
    BitOffsets.push_back(Offset * 8);

  SmallVector<unsigned, 8> SplitRegs;
  for (auto I = &SplitArgs[FirstRegIdx]; I != SplitArgs.end(); ++I)
    SplitRegs.push_back(I->Reg);

  PerformArgSplit(SplitRegs, BitOffsets);
}

static void copyToPhysReg(MachineIRBuilder &MIRBuilder, unsigned ValReg,
                          CCValAssign &VA, MachineRegisterInfo &MRI) {
  LLT LocTy{VA.getLocVT()};
  switch (VA.getLocInfo()) {
  default: break;
  case CCValAssign::AExt:
    assert(!VA.getLocVT().isVector() && "unexpected vector extend");
    // Otherwise, it's a nop.
    break;
  case CCValAssign::SExt: {
    unsigned NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildSExt(NewReg, ValReg);
    ValReg = NewReg;
    break;
  }
  case CCValAssign::ZExt: {
    unsigned NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildZExt(NewReg, ValReg);
    ValReg = NewReg;
    break;
  }
  }
  MIRBuilder.buildCopy(VA.getLocReg(), ValReg);
}


bool AArch64CallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                      const Value *Val, unsigned VReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();

  MachineInstrBuilder MIB = MIRBuilder.buildInstr(AArch64::RET_ReallyLR);
  assert(MIB.getInstr() && "Unable to build a return instruction?!");

  assert(((Val && VReg) || (!Val && !VReg)) && "Return value without a vreg");
  if (VReg) {
    MIRBuilder.setInstr(*MIB.getInstr(), /* Before */ true);
    const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
    CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
    MachineRegisterInfo &MRI = MF.getRegInfo();
    auto &DL = F.getParent()->getDataLayout();

    ArgInfo OrigArg{VReg, Val->getType()};
    setArgFlags(OrigArg, AttributeSet::ReturnIndex, DL, F);

    SmallVector<ArgInfo, 8> SplitArgs;
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        MIRBuilder.buildExtract(Regs, Offsets, VReg);
                      });

    return handleAssignments(MIRBuilder, AssignFn, SplitArgs,
                             [&](MachineIRBuilder &MIRBuilder, Type *Ty,
                                 unsigned ValReg, CCValAssign &VA) {
                               copyToPhysReg(MIRBuilder, ValReg, VA, MRI);
                               MIB.addUse(VA.getLocReg(), RegState::Implicit);
                             });
  }
  return true;
}

bool AArch64CallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                               const Function &F,
                                               ArrayRef<unsigned> VRegs) const {
  auto &Args = F.getArgumentList();
  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<ArgInfo, 8> SplitArgs;
  unsigned i = 0;
  for (auto &Arg : Args) {
    ArgInfo OrigArg{VRegs[i], Arg.getType()};
    setArgFlags(OrigArg, i + 1, DL, F);
    splitToValueTypes(OrigArg, SplitArgs, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        MIRBuilder.buildSequence(VRegs[i], Regs, Offsets);
                      });
    ++i;
  }

  if (!MBB.empty())
    MIRBuilder.setInstr(*MBB.begin());

  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);

  if (!handleAssignments(MIRBuilder, AssignFn, SplitArgs,
                         [](MachineIRBuilder &MIRBuilder, Type *Ty,
                            unsigned ValReg, CCValAssign &VA) {
                           // FIXME: a sign/zeroext loc actually gives
                           // us an optimization hint. We should use it.
                           MIRBuilder.getMBB().addLiveIn(VA.getLocReg());
                           MIRBuilder.buildCopy(ValReg, VA.getLocReg());
                         }))
    return false;

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
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        MIRBuilder.buildExtract(Regs, Offsets, OrigArg.Reg);
                      });
  }

  // Find out which ABI gets to decide where things go.
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *CallAssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);

  // And finally we can do the actual assignments. For a call we need to keep
  // track of the registers used because they'll be implicit uses of the BL.
  SmallVector<unsigned, 8> PhysRegs;
  if (!handleAssignments(MIRBuilder, CallAssignFn, SplitArgs,
                         [&](MachineIRBuilder &MIRBuilder, Type *Ty,
                             unsigned ValReg, CCValAssign &VA) {
                           copyToPhysReg(MIRBuilder, ValReg, VA, MRI);
                           PhysRegs.push_back(VA.getLocReg());
                         }))
    return false;

  // Now we can build the actual call instruction.
  auto MIB = MIRBuilder.buildInstr(Callee.isReg() ? AArch64::BLR : AArch64::BL);
  MIB.addOperand(Callee);

  // Tell the call which registers are clobbered.
  auto TRI = MF.getSubtarget().getRegisterInfo();
  MIB.addRegMask(TRI->getCallPreservedMask(MF, F.getCallingConv()));

  for (auto Reg : PhysRegs)
    MIB.addUse(Reg, RegState::Implicit);

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arugments, the physical register must be an
  // implicit-define of the call instruction.
  CCAssignFn *RetAssignFn = TLI.CCAssignFnForReturn(F.getCallingConv());
  if (OrigRet.Reg) {
    SplitArgs.clear();

    SmallVector<uint64_t, 8> RegOffsets;
    SmallVector<unsigned, 8> SplitRegs;
    splitToValueTypes(OrigRet, SplitArgs, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        std::copy(Offsets.begin(), Offsets.end(),
                                  std::back_inserter(RegOffsets));
                        std::copy(Regs.begin(), Regs.end(),
                                  std::back_inserter(SplitRegs));
                      });

    if (!handleAssignments(MIRBuilder, RetAssignFn, SplitArgs,
                      [&](MachineIRBuilder &MIRBuilder, Type *Ty,
                          unsigned ValReg, CCValAssign &VA) {
                             // FIXME: a sign/zeroext loc actually gives
                             // us an optimization hint. We should use it.
                             MIRBuilder.buildCopy(ValReg, VA.getLocReg());
                             MIB.addDef(VA.getLocReg(), RegState::Implicit);
                           }))
      return false;

    if (!RegOffsets.empty())
      MIRBuilder.buildSequence(OrigRet.Reg, SplitRegs, RegOffsets);
  }

  return true;
}
