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
                                            ValueHandler &Handler) const {
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

    if (VA.isRegLoc())
      Handler.assignValueToReg(Args[i].Reg, VA.getLocReg(), VA);
    else if (VA.isMemLoc()) {
      unsigned Size = VA.getValVT().getSizeInBits() / 8;
      unsigned Offset = VA.getLocMemOffset();
      MachinePointerInfo MPO;
      unsigned StackAddr = Handler.getStackAddress(Size, Offset, MPO);
      Handler.assignValueToAddress(Args[i].Reg, StackAddr, Size, MPO, VA);
    } else {
      // FIXME: Support byvals and other weirdness
      return false;
    }
  }
  return true;
}

unsigned AArch64CallLowering::ValueHandler::extendRegister(unsigned ValReg,
                                                           CCValAssign &VA) {
  LLT LocTy{VA.getLocVT()};
  switch (VA.getLocInfo()) {
  default: break;
  case CCValAssign::Full:
  case CCValAssign::BCvt:
    // FIXME: bitconverting between vector types may or may not be a
    // nop in big-endian situations.
    return ValReg;
  case CCValAssign::AExt:
    assert(!VA.getLocVT().isVector() && "unexpected vector extend");
    // Otherwise, it's a nop.
    return ValReg;
  case CCValAssign::SExt: {
    unsigned NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildSExt(NewReg, ValReg);
    return NewReg;
  }
  case CCValAssign::ZExt: {
    unsigned NewReg = MRI.createGenericVirtualRegister(LocTy);
    MIRBuilder.buildZExt(NewReg, ValReg);
    return NewReg;
  }
  }
  llvm_unreachable("unable to extend register");
}

struct IncomingArgHandler : public AArch64CallLowering::ValueHandler {
  IncomingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
    : ValueHandler(MIRBuilder, MRI) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);
    unsigned AddrReg = MRI.createGenericVirtualRegister(LLT::pointer(0, 64));
    MIRBuilder.buildFrameIndex(AddrReg, FI);
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
};

struct FormalArgHandler : public IncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : IncomingArgHandler(MIRBuilder, MRI) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

struct CallReturnHandler : public IncomingArgHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       MachineInstrBuilder MIB)
    : IncomingArgHandler(MIRBuilder, MRI), MIB(MIB) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
};

struct OutgoingArgHandler : public AArch64CallLowering::ValueHandler {
  OutgoingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     MachineInstrBuilder MIB)
      : ValueHandler(MIRBuilder, MRI), MIB(MIB) {}

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

  MachineInstrBuilder MIB;
};

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

    OutgoingArgHandler Handler(MIRBuilder, MRI, MIB);
    return handleAssignments(MIRBuilder, AssignFn, SplitArgs, Handler);
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

  FormalArgHandler Handler(MIRBuilder, MRI);
  if (!handleAssignments(MIRBuilder, AssignFn, SplitArgs, Handler))
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

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  auto MIB = MIRBuilder.buildInstrNoInsert(Callee.isReg() ? AArch64::BLR
                                                          : AArch64::BL);
  MIB.addOperand(Callee);

  // Tell the call which registers are clobbered.
  auto TRI = MF.getSubtarget().getRegisterInfo();
  MIB.addRegMask(TRI->getCallPreservedMask(MF, F.getCallingConv()));

  // Do the actual argument marshalling.
  SmallVector<unsigned, 8> PhysRegs;
  OutgoingArgHandler Handler(MIRBuilder, MRI, MIB);
  if (!handleAssignments(MIRBuilder, CallAssignFn, SplitArgs, Handler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

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

    CallReturnHandler Handler(MIRBuilder, MRI, MIB);
    if (!handleAssignments(MIRBuilder, RetAssignFn, SplitArgs, Handler))
      return false;

    if (!RegOffsets.empty())
      MIRBuilder.buildSequence(OrigRet.Reg, SplitRegs, RegOffsets);
  }

  return true;
}
