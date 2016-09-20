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
                                            ArrayRef<Type *> ArgTypes,
                                            ArrayRef<unsigned> ArgRegs,
                                            AssignFnTy AssignValToReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  unsigned NumArgs = ArgTypes.size();
  auto CurTy = ArgTypes.begin();
  for (unsigned i = 0; i != NumArgs; ++i, ++CurTy) {
    MVT CurVT = MVT::getVT(*CurTy);
    if (AssignFn(i, CurVT, CurVT, CCValAssign::Full, ISD::ArgFlagsTy(), CCInfo))
      return false;
  }
  assert(ArgLocs.size() == ArgTypes.size() &&
         "We have a different number of location and args?!");
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // FIXME: Support non-register argument.
    if (!VA.isRegLoc())
      return false;

    switch (VA.getLocInfo()) {
    default:
      //  Unknown loc info!
      return false;
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      // We don't care about bitcast.
      break;
    case CCValAssign::AExt:
      // Existing high bits are fine for anyext (whatever they are).
      break;
    case CCValAssign::SExt:
    case CCValAssign::ZExt:
      // Zero/Sign extend the register.
      // FIXME: Not yet implemented
      return false;
    }

    // Everything checks out, tell the caller where we've decided this
    // parameter/return value should go.
    AssignValToReg(MIRBuilder, ArgTypes[i], ArgRegs[i], VA.getLocReg());
  }
  return true;
}

void AArch64CallLowering::splitToValueTypes(
    unsigned Reg, Type *Ty, SmallVectorImpl<unsigned> &SplitRegs,
    SmallVectorImpl<Type *> &SplitTys, const DataLayout &DL,
    MachineRegisterInfo &MRI, SplitArgTy SplitArg) const {
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  LLVMContext &Ctx = Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, Ty, SplitVTs, &Offsets, 0);

  if (SplitVTs.size() == 1) {
    // No splitting to do, just forward the input directly.
    SplitTys.push_back(Ty);
    SplitRegs.push_back(Reg);
    return;
  }

  unsigned FirstRegIdx = SplitRegs.size();
  for (auto SplitVT : SplitVTs) {
    Type *SplitTy = SplitVT.getTypeForEVT(Ctx);
    SplitRegs.push_back(MRI.createGenericVirtualRegister(LLT{*SplitTy, DL}));
    SplitTys.push_back(SplitTy);
  }

  SmallVector<uint64_t, 4> BitOffsets;
  for (auto Offset : Offsets)
    BitOffsets.push_back(Offset * 8);

  SplitArg(ArrayRef<unsigned>(&SplitRegs[FirstRegIdx], SplitRegs.end()),
           BitOffsets);
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

    SmallVector<Type *, 8> SplitTys;
    SmallVector<unsigned, 8> SplitRegs;
    splitToValueTypes(VReg, Val->getType(), SplitRegs, SplitTys, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        MIRBuilder.buildExtract(Regs, Offsets, VReg);
                      });

    return handleAssignments(MIRBuilder, AssignFn, SplitTys, SplitRegs,
                             [&](MachineIRBuilder &MIRBuilder, Type *Ty,
                                 unsigned ValReg, unsigned PhysReg) {
                               MIRBuilder.buildCopy(PhysReg, ValReg);
                               MIB.addUse(PhysReg, RegState::Implicit);
                             });
  }
  return true;
}

bool AArch64CallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function::ArgumentListType &Args,
    ArrayRef<unsigned> VRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineBasicBlock &MBB = MIRBuilder.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = *MF.getFunction();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<MachineInstr *, 8> Seqs;
  SmallVector<Type *, 8> SplitTys;
  SmallVector<unsigned, 8> SplitRegs;
  unsigned i = 0;
  for (auto &Arg : Args) {
    splitToValueTypes(VRegs[i], Arg.getType(), SplitRegs, SplitTys, DL, MRI,
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

  bool Res = handleAssignments(MIRBuilder, AssignFn, SplitTys, SplitRegs,
                               [](MachineIRBuilder &MIRBuilder, Type *Ty,
                                  unsigned ValReg, unsigned PhysReg) {
                                 MIRBuilder.getMBB().addLiveIn(PhysReg);
                                 MIRBuilder.buildCopy(ValReg, PhysReg);
                               });

  // Move back to the end of the basic block.
  MIRBuilder.setMBB(MBB);

  return Res;
}

bool AArch64CallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                    const MachineOperand &Callee, Type *ResTy,
                                    unsigned ResReg, ArrayRef<Type *> ArgTys,
                                    ArrayRef<unsigned> ArgRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &DL = F.getParent()->getDataLayout();

  SmallVector<Type *, 8> SplitTys;
  SmallVector<unsigned, 8> SplitRegs;
  for (unsigned i = 0; i < ArgTys.size(); ++i) {
    splitToValueTypes(ArgRegs[i], ArgTys[i], SplitRegs, SplitTys, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        MIRBuilder.buildExtract(Regs, Offsets, ArgRegs[i]);
                      });
  }

  // Find out which ABI gets to decide where things go.
  const AArch64TargetLowering &TLI = *getTLI<AArch64TargetLowering>();
  CCAssignFn *CallAssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), /*IsVarArg=*/false);

  // And finally we can do the actual assignments. For a call we need to keep
  // track of the registers used because they'll be implicit uses of the BL.
  SmallVector<unsigned, 8> PhysRegs;
  handleAssignments(MIRBuilder, CallAssignFn, SplitTys, SplitRegs,
                    [&](MachineIRBuilder &MIRBuilder, Type *Ty, unsigned ValReg,
                        unsigned PhysReg) {
                      MIRBuilder.buildCopy(PhysReg, ValReg);
                      PhysRegs.push_back(PhysReg);
                    });

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
  if (ResReg) {
    SplitTys.clear();
    SplitRegs.clear();

    SmallVector<uint64_t, 8> RegOffsets;
    splitToValueTypes(ResReg, ResTy, SplitRegs, SplitTys, DL, MRI,
                      [&](ArrayRef<unsigned> Regs, ArrayRef<uint64_t> Offsets) {
                        std::copy(Offsets.begin(), Offsets.end(),
                                  std::back_inserter(RegOffsets));
                      });

    handleAssignments(MIRBuilder, RetAssignFn, SplitTys, SplitRegs,
                      [&](MachineIRBuilder &MIRBuilder, Type *Ty,
                          unsigned ValReg, unsigned PhysReg) {
                        MIRBuilder.buildCopy(ValReg, PhysReg);
                        MIB.addDef(PhysReg, RegState::Implicit);
                      });

    if (!RegOffsets.empty())
      MIRBuilder.buildSequence(ResReg, SplitRegs, RegOffsets);
  }

  return true;
}
