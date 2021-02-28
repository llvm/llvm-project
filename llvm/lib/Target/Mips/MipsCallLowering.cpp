//===- MipsCallLowering.cpp -------------------------------------*- C++ -*-===//
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

#include "MipsCallLowering.h"
#include "MipsCCState.h"
#include "MipsMachineFunction.h"
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

MipsCallLowering::MipsCallLowering(const MipsTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool MipsCallLowering::MipsHandler::assign(Register VReg, const CCValAssign &VA,
                                           const EVT &VT) {
  if (VA.isRegLoc()) {
    assignValueToReg(VReg, VA, VT);
  } else if (VA.isMemLoc()) {
    assignValueToAddress(VReg, VA);
  } else {
    return false;
  }
  return true;
}

bool MipsCallLowering::MipsHandler::assignVRegs(ArrayRef<Register> VRegs,
                                                ArrayRef<CCValAssign> ArgLocs,
                                                unsigned ArgLocsStartIndex,
                                                const EVT &VT) {
  for (unsigned i = 0; i < VRegs.size(); ++i)
    if (!assign(VRegs[i], ArgLocs[ArgLocsStartIndex + i], VT))
      return false;
  return true;
}

void MipsCallLowering::MipsHandler::setLeastSignificantFirst(
    SmallVectorImpl<Register> &VRegs) {
  if (!MIRBuilder.getMF().getDataLayout().isLittleEndian())
    std::reverse(VRegs.begin(), VRegs.end());
}

bool MipsCallLowering::MipsHandler::handle(
    ArrayRef<CCValAssign> ArgLocs, ArrayRef<CallLowering::ArgInfo> Args) {
  SmallVector<Register, 4> VRegs;
  unsigned SplitLength;
  const Function &F = MIRBuilder.getMF().getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  const MipsTargetLowering &TLI = *static_cast<const MipsTargetLowering *>(
      MIRBuilder.getMF().getSubtarget().getTargetLowering());

  for (unsigned ArgsIndex = 0, ArgLocsIndex = 0; ArgsIndex < Args.size();
       ++ArgsIndex, ArgLocsIndex += SplitLength) {
    EVT VT = TLI.getValueType(DL, Args[ArgsIndex].Ty);
    SplitLength = TLI.getNumRegistersForCallingConv(F.getContext(),
                                                    F.getCallingConv(), VT);
    assert(Args[ArgsIndex].Regs.size() == 1 && "Can't handle multple regs yet");

    if (SplitLength > 1) {
      VRegs.clear();
      MVT RegisterVT = TLI.getRegisterTypeForCallingConv(
          F.getContext(), F.getCallingConv(), VT);
      for (unsigned i = 0; i < SplitLength; ++i)
        VRegs.push_back(MRI.createGenericVirtualRegister(LLT{RegisterVT}));

      if (!handleSplit(VRegs, ArgLocs, ArgLocsIndex, Args[ArgsIndex].Regs[0],
                       VT))
        return false;
    } else {
      if (!assign(Args[ArgsIndex].Regs[0], ArgLocs[ArgLocsIndex], VT))
        return false;
    }
  }
  return true;
}

namespace {
class MipsIncomingValueHandler : public MipsCallLowering::MipsHandler {
public:
  MipsIncomingValueHandler(MachineIRBuilder &MIRBuilder,
                           MachineRegisterInfo &MRI)
      : MipsHandler(MIRBuilder, MRI) {}

private:
  void assignValueToReg(Register ValVReg, const CCValAssign &VA,
                        const EVT &VT) override;

  Register getStackAddress(const CCValAssign &VA,
                           MachineMemOperand *&MMO) override;

  void assignValueToAddress(Register ValVReg, const CCValAssign &VA) override;

  bool handleSplit(SmallVectorImpl<Register> &VRegs,
                   ArrayRef<CCValAssign> ArgLocs, unsigned ArgLocsStartIndex,
                   Register ArgsReg, const EVT &VT) override;

  virtual void markPhysRegUsed(unsigned PhysReg) {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }

  MachineInstrBuilder buildLoad(const DstOp &Res, const CCValAssign &VA) {
    MachineMemOperand *MMO;
    Register Addr = getStackAddress(VA, MMO);
    return MIRBuilder.buildLoad(Res, Addr, *MMO);
  }
};

class CallReturnHandler : public MipsIncomingValueHandler {
public:
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder &MIB)
      : MipsIncomingValueHandler(MIRBuilder, MRI), MIB(MIB) {}

private:
  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder &MIB;
};

} // end anonymous namespace

void MipsIncomingValueHandler::assignValueToReg(Register ValVReg,
                                                const CCValAssign &VA,
                                                const EVT &VT) {
  Register PhysReg = VA.getLocReg();
  if (VT == MVT::f64 && PhysReg >= Mips::A0 && PhysReg <= Mips::A3) {
    const MipsSubtarget &STI =
        static_cast<const MipsSubtarget &>(MIRBuilder.getMF().getSubtarget());
    bool IsEL = STI.isLittle();
    LLT s32 = LLT::scalar(32);
    auto Lo = MIRBuilder.buildCopy(s32, Register(PhysReg + (IsEL ? 0 : 1)));
    auto Hi = MIRBuilder.buildCopy(s32, Register(PhysReg + (IsEL ? 1 : 0)));
    MIRBuilder.buildMerge(ValVReg, {Lo, Hi});
    markPhysRegUsed(PhysReg);
    markPhysRegUsed(PhysReg + 1);
  } else if (VT == MVT::f32 && PhysReg >= Mips::A0 && PhysReg <= Mips::A3) {
    MIRBuilder.buildCopy(ValVReg, PhysReg);
    markPhysRegUsed(PhysReg);
  } else {
    switch (VA.getLocInfo()) {
    case CCValAssign::LocInfo::SExt:
    case CCValAssign::LocInfo::ZExt:
    case CCValAssign::LocInfo::AExt: {
      auto Copy = MIRBuilder.buildCopy(LLT{VA.getLocVT()}, PhysReg);
      MIRBuilder.buildTrunc(ValVReg, Copy);
      break;
    }
    default:
      MIRBuilder.buildCopy(ValVReg, PhysReg);
      break;
    }
    markPhysRegUsed(PhysReg);
  }
}

Register MipsIncomingValueHandler::getStackAddress(const CCValAssign &VA,
                                                   MachineMemOperand *&MMO) {
  MachineFunction &MF = MIRBuilder.getMF();
  unsigned Size = alignTo(VA.getValVT().getSizeInBits(), 8) / 8;
  unsigned Offset = VA.getLocMemOffset();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  int FI = MFI.CreateFixedObject(Size, Offset, true);
  MachinePointerInfo MPO =
      MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
  Align Alignment = commonAlignment(TFL->getStackAlign(), Offset);
  MMO =
      MF.getMachineMemOperand(MPO, MachineMemOperand::MOLoad, Size, Alignment);

  return MIRBuilder.buildFrameIndex(LLT::pointer(0, 32), FI).getReg(0);
}

void MipsIncomingValueHandler::assignValueToAddress(Register ValVReg,
                                                    const CCValAssign &VA) {
  if (VA.getLocInfo() == CCValAssign::SExt ||
      VA.getLocInfo() == CCValAssign::ZExt ||
      VA.getLocInfo() == CCValAssign::AExt) {
    auto Load = buildLoad(LLT::scalar(32), VA);
    MIRBuilder.buildTrunc(ValVReg, Load);
  } else
    buildLoad(ValVReg, VA);
}

bool MipsIncomingValueHandler::handleSplit(SmallVectorImpl<Register> &VRegs,
                                           ArrayRef<CCValAssign> ArgLocs,
                                           unsigned ArgLocsStartIndex,
                                           Register ArgsReg, const EVT &VT) {
  if (!assignVRegs(VRegs, ArgLocs, ArgLocsStartIndex, VT))
    return false;
  setLeastSignificantFirst(VRegs);
  MIRBuilder.buildMerge(ArgsReg, VRegs);
  return true;
}

namespace {
class MipsOutgoingValueHandler : public MipsCallLowering::MipsHandler {
public:
  MipsOutgoingValueHandler(MachineIRBuilder &MIRBuilder,
                           MachineRegisterInfo &MRI, MachineInstrBuilder &MIB)
      : MipsHandler(MIRBuilder, MRI), MIB(MIB) {}

private:
  void assignValueToReg(Register ValVReg, const CCValAssign &VA,
                        const EVT &VT) override;

  Register getStackAddress(const CCValAssign &VA,
                           MachineMemOperand *&MMO) override;

  void assignValueToAddress(Register ValVReg, const CCValAssign &VA) override;

  bool handleSplit(SmallVectorImpl<Register> &VRegs,
                   ArrayRef<CCValAssign> ArgLocs, unsigned ArgLocsStartIndex,
                   Register ArgsReg, const EVT &VT) override;

  Register extendRegister(Register ValReg, const CCValAssign &VA);

  MachineInstrBuilder &MIB;
};
} // end anonymous namespace

void MipsOutgoingValueHandler::assignValueToReg(Register ValVReg,
                                                const CCValAssign &VA,
                                                const EVT &VT) {
  Register PhysReg = VA.getLocReg();
  if (VT == MVT::f64 && PhysReg >= Mips::A0 && PhysReg <= Mips::A3) {
    const MipsSubtarget &STI =
        static_cast<const MipsSubtarget &>(MIRBuilder.getMF().getSubtarget());
    bool IsEL = STI.isLittle();
    auto Unmerge = MIRBuilder.buildUnmerge(LLT::scalar(32), ValVReg);
    MIRBuilder.buildCopy(Register(PhysReg + (IsEL ? 0 : 1)), Unmerge.getReg(0));
    MIRBuilder.buildCopy(Register(PhysReg + (IsEL ? 1 : 0)), Unmerge.getReg(1));
  } else if (VT == MVT::f32 && PhysReg >= Mips::A0 && PhysReg <= Mips::A3) {
    MIRBuilder.buildCopy(PhysReg, ValVReg);
  } else {
    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }
}

Register MipsOutgoingValueHandler::getStackAddress(const CCValAssign &VA,
                                                   MachineMemOperand *&MMO) {
  MachineFunction &MF = MIRBuilder.getMF();
  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();

  LLT p0 = LLT::pointer(0, 32);
  LLT s32 = LLT::scalar(32);
  auto SPReg = MIRBuilder.buildCopy(p0, Register(Mips::SP));

  unsigned Offset = VA.getLocMemOffset();
  auto OffsetReg = MIRBuilder.buildConstant(s32, Offset);

  auto AddrReg = MIRBuilder.buildPtrAdd(p0, SPReg, OffsetReg);

  MachinePointerInfo MPO =
      MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
  unsigned Size = alignTo(VA.getValVT().getSizeInBits(), 8) / 8;
  Align Alignment = commonAlignment(TFL->getStackAlign(), Offset);
  MMO =
      MF.getMachineMemOperand(MPO, MachineMemOperand::MOStore, Size, Alignment);

  return AddrReg.getReg(0);
}

void MipsOutgoingValueHandler::assignValueToAddress(Register ValVReg,
                                                    const CCValAssign &VA) {
  MachineMemOperand *MMO;
  Register Addr = getStackAddress(VA, MMO);
  Register ExtReg = extendRegister(ValVReg, VA);
  MIRBuilder.buildStore(ExtReg, Addr, *MMO);
}

Register MipsOutgoingValueHandler::extendRegister(Register ValReg,
                                                  const CCValAssign &VA) {
  LLT LocTy{VA.getLocVT()};
  switch (VA.getLocInfo()) {
  case CCValAssign::SExt: {
    return MIRBuilder.buildSExt(LocTy, ValReg).getReg(0);
  }
  case CCValAssign::ZExt: {
    return MIRBuilder.buildZExt(LocTy, ValReg).getReg(0);
  }
  case CCValAssign::AExt: {
    return MIRBuilder.buildAnyExt(LocTy, ValReg).getReg(0);
  }
  // TODO : handle upper extends
  case CCValAssign::Full:
    return ValReg;
  default:
    break;
  }
  llvm_unreachable("unable to extend register");
}

bool MipsOutgoingValueHandler::handleSplit(SmallVectorImpl<Register> &VRegs,
                                           ArrayRef<CCValAssign> ArgLocs,
                                           unsigned ArgLocsStartIndex,
                                           Register ArgsReg, const EVT &VT) {
  MIRBuilder.buildUnmerge(VRegs, ArgsReg);
  setLeastSignificantFirst(VRegs);
  if (!assignVRegs(VRegs, ArgLocs, ArgLocsStartIndex, VT))
    return false;

  return true;
}

static bool isSupportedArgumentType(Type *T) {
  if (T->isIntegerTy())
    return true;
  if (T->isPointerTy())
    return true;
  if (T->isFloatingPointTy())
    return true;
  return false;
}

static bool isSupportedReturnType(Type *T) {
  if (T->isIntegerTy())
    return true;
  if (T->isPointerTy())
    return true;
  if (T->isFloatingPointTy())
    return true;
  if (T->isAggregateType())
    return true;
  return false;
}

static CCValAssign::LocInfo determineLocInfo(const MVT RegisterVT, const EVT VT,
                                             const ISD::ArgFlagsTy &Flags) {
  // > does not mean loss of information as type RegisterVT can't hold type VT,
  // it means that type VT is split into multiple registers of type RegisterVT
  if (VT.getFixedSizeInBits() >= RegisterVT.getFixedSizeInBits())
    return CCValAssign::LocInfo::Full;
  if (Flags.isSExt())
    return CCValAssign::LocInfo::SExt;
  if (Flags.isZExt())
    return CCValAssign::LocInfo::ZExt;
  return CCValAssign::LocInfo::AExt;
}

template <typename T>
static void setLocInfo(SmallVectorImpl<CCValAssign> &ArgLocs,
                       const SmallVectorImpl<T> &Arguments) {
  for (unsigned i = 0; i < ArgLocs.size(); ++i) {
    const CCValAssign &VA = ArgLocs[i];
    CCValAssign::LocInfo LocInfo = determineLocInfo(
        Arguments[i].VT, Arguments[i].ArgVT, Arguments[i].Flags);
    if (VA.isMemLoc())
      ArgLocs[i] =
          CCValAssign::getMem(VA.getValNo(), VA.getValVT(),
                              VA.getLocMemOffset(), VA.getLocVT(), LocInfo);
    else
      ArgLocs[i] = CCValAssign::getReg(VA.getValNo(), VA.getValVT(),
                                       VA.getLocReg(), VA.getLocVT(), LocInfo);
  }
}

bool MipsCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, ArrayRef<Register> VRegs,
                                   FunctionLoweringInfo &FLI) const {

  MachineInstrBuilder Ret = MIRBuilder.buildInstrNoInsert(Mips::RetRA);

  if (Val != nullptr && !isSupportedReturnType(Val->getType()))
    return false;

  if (!VRegs.empty()) {
    MachineFunction &MF = MIRBuilder.getMF();
    const Function &F = MF.getFunction();
    const DataLayout &DL = MF.getDataLayout();
    const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();

    SmallVector<ArgInfo, 8> RetInfos;
    SmallVector<unsigned, 8> OrigArgIndices;

    ArgInfo ArgRetInfo(VRegs, Val->getType());
    setArgFlags(ArgRetInfo, AttributeList::ReturnIndex, DL, F);
    splitToValueTypes(DL, ArgRetInfo, 0, RetInfos, OrigArgIndices);

    SmallVector<ISD::OutputArg, 8> Outs;
    subTargetRegTypeForCallingConv(F, RetInfos, OrigArgIndices, Outs);

    SmallVector<CCValAssign, 16> ArgLocs;
    MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                       F.getContext());
    CCInfo.AnalyzeReturn(Outs, TLI.CCAssignFnForReturn());
    setLocInfo(ArgLocs, Outs);

    MipsOutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret);
    if (!RetHandler.handle(ArgLocs, RetInfos)) {
      return false;
    }
  }
  MIRBuilder.insertInstr(Ret);
  return true;
}

bool MipsCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<ArrayRef<Register>> VRegs,
                                            FunctionLoweringInfo &FLI) const {

  // Quick exit if there aren't any args.
  if (F.arg_empty())
    return true;

  for (auto &Arg : F.args()) {
    if (!isSupportedArgumentType(Arg.getType()))
      return false;
  }

  MachineFunction &MF = MIRBuilder.getMF();
  const DataLayout &DL = MF.getDataLayout();
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();

  SmallVector<ArgInfo, 8> ArgInfos;
  SmallVector<unsigned, 8> OrigArgIndices;
  unsigned i = 0;
  for (auto &Arg : F.args()) {
    ArgInfo AInfo(VRegs[i], Arg.getType());
    setArgFlags(AInfo, i + AttributeList::FirstArgIndex, DL, F);
    ArgInfos.push_back(AInfo);
    OrigArgIndices.push_back(i);
    ++i;
  }

  SmallVector<ISD::InputArg, 8> Ins;
  subTargetRegTypeForCallingConv(F, ArgInfos, OrigArgIndices, Ins);

  SmallVector<CCValAssign, 16> ArgLocs;
  MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                     F.getContext());

  const MipsTargetMachine &TM =
      static_cast<const MipsTargetMachine &>(MF.getTarget());
  const MipsABIInfo &ABI = TM.getABI();
  CCInfo.AllocateStack(ABI.GetCalleeAllocdArgSizeInBytes(F.getCallingConv()),
                       Align(1));
  CCInfo.AnalyzeFormalArguments(Ins, TLI.CCAssignFnForCall());
  setLocInfo(ArgLocs, Ins);

  MipsIncomingValueHandler Handler(MIRBuilder, MF.getRegInfo());
  if (!Handler.handle(ArgLocs, ArgInfos))
    return false;

  if (F.isVarArg()) {
    ArrayRef<MCPhysReg> ArgRegs = ABI.GetVarArgRegs();
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);

    int VaArgOffset;
    unsigned RegSize = 4;
    if (ArgRegs.size() == Idx)
      VaArgOffset = alignTo(CCInfo.getNextStackOffset(), RegSize);
    else {
      VaArgOffset =
          (int)ABI.GetCalleeAllocdArgSizeInBytes(CCInfo.getCallingConv()) -
          (int)(RegSize * (ArgRegs.size() - Idx));
    }

    MachineFrameInfo &MFI = MF.getFrameInfo();
    int FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
    MF.getInfo<MipsFunctionInfo>()->setVarArgsFrameIndex(FI);

    for (unsigned I = Idx; I < ArgRegs.size(); ++I, VaArgOffset += RegSize) {
      MIRBuilder.getMBB().addLiveIn(ArgRegs[I]);

      MachineInstrBuilder Copy =
          MIRBuilder.buildCopy(LLT::scalar(RegSize * 8), Register(ArgRegs[I]));
      FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
      MachinePointerInfo MPO = MachinePointerInfo::getFixedStack(MF, FI);
      MachineInstrBuilder FrameIndex =
          MIRBuilder.buildFrameIndex(LLT::pointer(MPO.getAddrSpace(), 32), FI);
      MachineMemOperand *MMO = MF.getMachineMemOperand(
          MPO, MachineMemOperand::MOStore, RegSize, Align(RegSize));
      MIRBuilder.buildStore(Copy, FrameIndex, *MMO);
    }
  }

  return true;
}

bool MipsCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                 CallLoweringInfo &Info) const {

  if (Info.CallConv != CallingConv::C)
    return false;

  for (auto &Arg : Info.OrigArgs) {
    if (!isSupportedArgumentType(Arg.Ty))
      return false;
    if (Arg.Flags[0].isByVal())
      return false;
    if (Arg.Flags[0].isSRet() && !Arg.Ty->isPointerTy())
      return false;
  }

  if (!Info.OrigRet.Ty->isVoidTy() && !isSupportedReturnType(Info.OrigRet.Ty))
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = MF.getDataLayout();
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();
  const MipsTargetMachine &TM =
      static_cast<const MipsTargetMachine &>(MF.getTarget());
  const MipsABIInfo &ABI = TM.getABI();

  MachineInstrBuilder CallSeqStart =
      MIRBuilder.buildInstr(Mips::ADJCALLSTACKDOWN);

  const bool IsCalleeGlobalPIC =
      Info.Callee.isGlobal() && TM.isPositionIndependent();

  MachineInstrBuilder MIB = MIRBuilder.buildInstrNoInsert(
      Info.Callee.isReg() || IsCalleeGlobalPIC ? Mips::JALRPseudo : Mips::JAL);
  MIB.addDef(Mips::SP, RegState::Implicit);
  if (IsCalleeGlobalPIC) {
    Register CalleeReg =
        MF.getRegInfo().createGenericVirtualRegister(LLT::pointer(0, 32));
    MachineInstr *CalleeGlobalValue =
        MIRBuilder.buildGlobalValue(CalleeReg, Info.Callee.getGlobal());
    if (!Info.Callee.getGlobal()->hasLocalLinkage())
      CalleeGlobalValue->getOperand(1).setTargetFlags(MipsII::MO_GOT_CALL);
    MIB.addUse(CalleeReg);
  } else
    MIB.add(Info.Callee);
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MIB.addRegMask(TRI->getCallPreservedMask(MF, F.getCallingConv()));

  TargetLowering::ArgListTy FuncOrigArgs;
  FuncOrigArgs.reserve(Info.OrigArgs.size());

  SmallVector<ArgInfo, 8> ArgInfos;
  SmallVector<unsigned, 8> OrigArgIndices;
  unsigned i = 0;
  for (auto &Arg : Info.OrigArgs) {

    TargetLowering::ArgListEntry Entry;
    Entry.Ty = Arg.Ty;
    FuncOrigArgs.push_back(Entry);

    ArgInfos.push_back(Arg);
    OrigArgIndices.push_back(i);
    ++i;
  }

  SmallVector<ISD::OutputArg, 8> Outs;
  subTargetRegTypeForCallingConv(F, ArgInfos, OrigArgIndices, Outs);

  SmallVector<CCValAssign, 8> ArgLocs;
  bool IsCalleeVarArg = false;
  if (Info.Callee.isGlobal()) {
    const Function *CF = static_cast<const Function *>(Info.Callee.getGlobal());
    IsCalleeVarArg = CF->isVarArg();
  }
  MipsCCState CCInfo(F.getCallingConv(), IsCalleeVarArg, MF, ArgLocs,
                     F.getContext());

  CCInfo.AllocateStack(ABI.GetCalleeAllocdArgSizeInBytes(Info.CallConv),
                       Align(1));
  const char *Call =
      Info.Callee.isSymbol() ? Info.Callee.getSymbolName() : nullptr;
  CCInfo.AnalyzeCallOperands(Outs, TLI.CCAssignFnForCall(), FuncOrigArgs, Call);
  setLocInfo(ArgLocs, Outs);

  MipsOutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), MIB);
  if (!RetHandler.handle(ArgLocs, ArgInfos)) {
    return false;
  }

  unsigned NextStackOffset = CCInfo.getNextStackOffset();
  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
  unsigned StackAlignment = TFL->getStackAlignment();
  NextStackOffset = alignTo(NextStackOffset, StackAlignment);
  CallSeqStart.addImm(NextStackOffset).addImm(0);

  if (IsCalleeGlobalPIC) {
    MIRBuilder.buildCopy(
      Register(Mips::GP),
      MF.getInfo<MipsFunctionInfo>()->getGlobalBaseRegForGlobalISel(MF));
    MIB.addDef(Mips::GP, RegState::Implicit);
  }
  MIRBuilder.insertInstr(MIB);
  if (MIB->getOpcode() == Mips::JALRPseudo) {
    const MipsSubtarget &STI =
        static_cast<const MipsSubtarget &>(MIRBuilder.getMF().getSubtarget());
    MIB.constrainAllUses(MIRBuilder.getTII(), *STI.getRegisterInfo(),
                         *STI.getRegBankInfo());
  }

  if (!Info.OrigRet.Ty->isVoidTy()) {
    ArgInfos.clear();
    SmallVector<unsigned, 8> OrigRetIndices;

    splitToValueTypes(DL, Info.OrigRet, 0, ArgInfos, OrigRetIndices);

    SmallVector<ISD::InputArg, 8> Ins;
    subTargetRegTypeForCallingConv(F, ArgInfos, OrigRetIndices, Ins);

    SmallVector<CCValAssign, 8> ArgLocs;
    MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                       F.getContext());

    CCInfo.AnalyzeCallResult(Ins, TLI.CCAssignFnForReturn(), Info.OrigRet.Ty,
                             Call);
    setLocInfo(ArgLocs, Ins);

    CallReturnHandler Handler(MIRBuilder, MF.getRegInfo(), MIB);
    if (!Handler.handle(ArgLocs, ArgInfos))
      return false;
  }

  MIRBuilder.buildInstr(Mips::ADJCALLSTACKUP).addImm(NextStackOffset).addImm(0);

  return true;
}

template <typename T>
void MipsCallLowering::subTargetRegTypeForCallingConv(
    const Function &F, ArrayRef<ArgInfo> Args,
    ArrayRef<unsigned> OrigArgIndices, SmallVectorImpl<T> &ISDArgs) const {
  const DataLayout &DL = F.getParent()->getDataLayout();
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();

  unsigned ArgNo = 0;
  for (auto &Arg : Args) {

    EVT VT = TLI.getValueType(DL, Arg.Ty);
    MVT RegisterVT = TLI.getRegisterTypeForCallingConv(F.getContext(),
                                                       F.getCallingConv(), VT);
    unsigned NumRegs = TLI.getNumRegistersForCallingConv(
        F.getContext(), F.getCallingConv(), VT);

    for (unsigned i = 0; i < NumRegs; ++i) {
      ISD::ArgFlagsTy Flags = Arg.Flags[0];

      if (i == 0)
        Flags.setOrigAlign(TLI.getABIAlignmentForCallingConv(Arg.Ty, DL));
      else
        Flags.setOrigAlign(Align(1));

      ISDArgs.emplace_back(Flags, RegisterVT, VT, true, OrigArgIndices[ArgNo],
                           0);
    }
    ++ArgNo;
  }
}

// FIXME: This should be removed and the generic version used
void MipsCallLowering::splitToValueTypes(
    const DataLayout &DL, const ArgInfo &OrigArg, unsigned OriginalIndex,
    SmallVectorImpl<ArgInfo> &SplitArgs,
    SmallVectorImpl<unsigned> &SplitArgsOrigIndices) const {

  SmallVector<EVT, 4> SplitEVTs;
  SmallVector<Register, 4> SplitVRegs;
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitEVTs);

  for (unsigned i = 0; i < SplitEVTs.size(); ++i) {
    ArgInfo Info = ArgInfo{OrigArg.Regs[i], SplitEVTs[i].getTypeForEVT(Ctx)};
    Info.Flags = OrigArg.Flags;
    SplitArgs.push_back(Info);
    SplitArgsOrigIndices.push_back(OriginalIndex);
  }
}
