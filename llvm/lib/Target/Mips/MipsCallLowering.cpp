//===- MipsCallLowering.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "MipsTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

using namespace llvm;

MipsCallLowering::MipsCallLowering(const MipsTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool MipsCallLowering::MipsHandler::assign(const CCValAssign &VA,
                                           unsigned vreg) {
  if (VA.isRegLoc()) {
    assignValueToReg(vreg, VA.getLocReg());
  } else if (VA.isMemLoc()) {
    unsigned Size = alignTo(VA.getValVT().getSizeInBits(), 8) / 8;
    unsigned Offset = VA.getLocMemOffset();
    MachinePointerInfo MPO;
    unsigned StackAddr = getStackAddress(Size, Offset, MPO);
    assignValueToAddress(vreg, StackAddr, Size, MPO);
  } else {
    return false;
  }
  return true;
}

namespace {
class IncomingValueHandler : public MipsCallLowering::MipsHandler {
public:
  IncomingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI)
      : MipsHandler(MIRBuilder, MRI) {}

  bool handle(ArrayRef<CCValAssign> ArgLocs,
              ArrayRef<CallLowering::ArgInfo> Args);

private:
  void assignValueToReg(unsigned ValVReg, unsigned PhysReg) override;

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override;

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO) override;

  virtual void markPhysRegUsed(unsigned PhysReg) {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }

  void buildLoad(unsigned Val, unsigned Addr, uint64_t Size, unsigned Alignment,
                 MachinePointerInfo &MPO) {
    MachineMemOperand *MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad, Size, Alignment);
    MIRBuilder.buildLoad(Val, Addr, *MMO);
  }
};

class CallReturnHandler : public IncomingValueHandler {
public:
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder &MIB)
      : IncomingValueHandler(MIRBuilder, MRI), MIB(MIB) {}

private:
  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder &MIB;
};

} // end anonymous namespace

void IncomingValueHandler::assignValueToReg(unsigned ValVReg,
                                            unsigned PhysReg) {
  MIRBuilder.buildCopy(ValVReg, PhysReg);
  markPhysRegUsed(PhysReg);
}

unsigned IncomingValueHandler::getStackAddress(uint64_t Size, int64_t Offset,
                                               MachinePointerInfo &MPO) {
  MachineFrameInfo &MFI = MIRBuilder.getMF().getFrameInfo();

  int FI = MFI.CreateFixedObject(Size, Offset, true);
  MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

  unsigned AddrReg = MRI.createGenericVirtualRegister(LLT::pointer(0, 32));
  MIRBuilder.buildFrameIndex(AddrReg, FI);

  return AddrReg;
}

void IncomingValueHandler::assignValueToAddress(unsigned ValVReg, unsigned Addr,
                                                uint64_t Size,
                                                MachinePointerInfo &MPO) {
  // If the value is not extended, a simple load will suffice.
  buildLoad(ValVReg, Addr, Size, /* Alignment */ 0, MPO);
}

bool IncomingValueHandler::handle(ArrayRef<CCValAssign> ArgLocs,
                                  ArrayRef<CallLowering::ArgInfo> Args) {
  for (unsigned i = 0, ArgsSize = Args.size(); i < ArgsSize; ++i) {
    if (!assign(ArgLocs[i], Args[i].Reg))
      return false;
  }
  return true;
}

namespace {
class OutgoingValueHandler : public MipsCallLowering::MipsHandler {
public:
  OutgoingValueHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                       MachineInstrBuilder &MIB)
      : MipsHandler(MIRBuilder, MRI), MIB(MIB) {}

  bool handle(ArrayRef<CCValAssign> ArgLocs,
              ArrayRef<CallLowering::ArgInfo> Args);

private:
  void assignValueToReg(unsigned ValVReg, unsigned PhysReg) override;

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override;

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO) override;

  MachineInstrBuilder &MIB;
};
} // end anonymous namespace

void OutgoingValueHandler::assignValueToReg(unsigned ValVReg,
                                            unsigned PhysReg) {
  MIRBuilder.buildCopy(PhysReg, ValVReg);
  MIB.addUse(PhysReg, RegState::Implicit);
}

unsigned OutgoingValueHandler::getStackAddress(uint64_t Size, int64_t Offset,
                                               MachinePointerInfo &MPO) {
  LLT p0 = LLT::pointer(0, 32);
  LLT s32 = LLT::scalar(32);
  unsigned SPReg = MRI.createGenericVirtualRegister(p0);
  MIRBuilder.buildCopy(SPReg, Mips::SP);

  unsigned OffsetReg = MRI.createGenericVirtualRegister(s32);
  MIRBuilder.buildConstant(OffsetReg, Offset);

  unsigned AddrReg = MRI.createGenericVirtualRegister(p0);
  MIRBuilder.buildGEP(AddrReg, SPReg, OffsetReg);

  MPO = MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
  return AddrReg;
}

void OutgoingValueHandler::assignValueToAddress(unsigned ValVReg, unsigned Addr,
                                                uint64_t Size,
                                                MachinePointerInfo &MPO) {
  MachineMemOperand *MMO = MIRBuilder.getMF().getMachineMemOperand(
      MPO, MachineMemOperand::MOStore, Size, /* Alignment */ 0);
  MIRBuilder.buildStore(ValVReg, Addr, *MMO);
}

bool OutgoingValueHandler::handle(ArrayRef<CCValAssign> ArgLocs,
                                  ArrayRef<CallLowering::ArgInfo> Args) {
  for (unsigned i = 0; i < Args.size(); ++i) {
    if (!assign(ArgLocs[i], Args[i].Reg))
      return false;
  }
  return true;
}

static bool isSupportedType(Type *T) {
  if (T->isIntegerTy() && T->getScalarSizeInBits() == 32)
    return true;
  if (T->isPointerTy())
    return true;
  return false;
}

bool MipsCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                   const Value *Val, unsigned VReg) const {

  MachineInstrBuilder Ret = MIRBuilder.buildInstrNoInsert(Mips::RetRA);

  if (Val != nullptr) {
    if (!isSupportedType(Val->getType()))
      return false;

    MachineFunction &MF = MIRBuilder.getMF();
    const Function &F = MF.getFunction();
    const DataLayout &DL = MF.getDataLayout();
    const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();

    SmallVector<ArgInfo, 8> RetInfos;
    SmallVector<unsigned, 8> OrigArgIndices;

    ArgInfo ArgRetInfo(VReg, Val->getType());
    setArgFlags(ArgRetInfo, AttributeList::ReturnIndex, DL, F);
    splitToValueTypes(ArgRetInfo, 0, RetInfos, OrigArgIndices);

    SmallVector<ISD::OutputArg, 8> Outs;
    subTargetRegTypeForCallingConv(
        MIRBuilder, RetInfos, OrigArgIndices,
        [&](ISD::ArgFlagsTy flags, EVT vt, EVT argvt, bool used,
            unsigned origIdx, unsigned partOffs) {
          Outs.emplace_back(flags, vt, argvt, used, origIdx, partOffs);
        });

    SmallVector<CCValAssign, 16> ArgLocs;
    MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                       F.getContext());
    CCInfo.AnalyzeReturn(Outs, TLI.CCAssignFnForReturn());

    OutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret);
    if (!RetHandler.handle(ArgLocs, RetInfos)) {
      return false;
    }
  }
  MIRBuilder.insertInstr(Ret);
  return true;
}

bool MipsCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                            const Function &F,
                                            ArrayRef<unsigned> VRegs) const {

  // Quick exit if there aren't any args.
  if (F.arg_empty())
    return true;

  if (F.isVarArg()) {
    return false;
  }

  for (auto &Arg : F.args()) {
    if (!isSupportedType(Arg.getType()))
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
    splitToValueTypes(AInfo, i, ArgInfos, OrigArgIndices);
    ++i;
  }

  SmallVector<ISD::InputArg, 8> Ins;
  subTargetRegTypeForCallingConv(
      MIRBuilder, ArgInfos, OrigArgIndices,
      [&](ISD::ArgFlagsTy flags, EVT vt, EVT argvt, bool used, unsigned origIdx,
          unsigned partOffs) {
        Ins.emplace_back(flags, vt, argvt, used, origIdx, partOffs);
      });

  SmallVector<CCValAssign, 16> ArgLocs;
  MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                     F.getContext());

  const MipsTargetMachine &TM =
      static_cast<const MipsTargetMachine &>(MF.getTarget());
  const MipsABIInfo &ABI = TM.getABI();
  CCInfo.AllocateStack(ABI.GetCalleeAllocdArgSizeInBytes(F.getCallingConv()),
                       1);
  CCInfo.AnalyzeFormalArguments(Ins, TLI.CCAssignFnForCall());

  IncomingValueHandler Handler(MIRBuilder, MF.getRegInfo());
  if (!Handler.handle(ArgLocs, ArgInfos))
    return false;

  return true;
}

bool MipsCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                 CallingConv::ID CallConv,
                                 const MachineOperand &Callee,
                                 const ArgInfo &OrigRet,
                                 ArrayRef<ArgInfo> OrigArgs) const {

  if (CallConv != CallingConv::C)
    return false;

  for (auto &Arg : OrigArgs) {
    if (!isSupportedType(Arg.Ty))
      return false;
    if (Arg.Flags.isByVal() || Arg.Flags.isSRet())
      return false;
  }
  if (OrigRet.Reg && !isSupportedType(OrigRet.Ty))
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();
  const MipsTargetMachine &TM =
      static_cast<const MipsTargetMachine &>(MF.getTarget());
  const MipsABIInfo &ABI = TM.getABI();

  MachineInstrBuilder CallSeqStart =
      MIRBuilder.buildInstr(Mips::ADJCALLSTACKDOWN);

  // FIXME: Add support for pic calling sequences, long call sequences for O32,
  //       N32 and N64. First handle the case when Callee.isReg().
  if (Callee.isReg())
    return false;

  MachineInstrBuilder MIB = MIRBuilder.buildInstrNoInsert(Mips::JAL);
  MIB.addDef(Mips::SP, RegState::Implicit);
  MIB.add(Callee);
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MIB.addRegMask(TRI->getCallPreservedMask(MF, F.getCallingConv()));

  TargetLowering::ArgListTy FuncOrigArgs;
  FuncOrigArgs.reserve(OrigArgs.size());

  SmallVector<ArgInfo, 8> ArgInfos;
  SmallVector<unsigned, 8> OrigArgIndices;
  unsigned i = 0;
  for (auto &Arg : OrigArgs) {

    TargetLowering::ArgListEntry Entry;
    Entry.Ty = Arg.Ty;
    FuncOrigArgs.push_back(Entry);

    splitToValueTypes(Arg, i, ArgInfos, OrigArgIndices);
    ++i;
  }

  SmallVector<ISD::OutputArg, 8> Outs;
  subTargetRegTypeForCallingConv(
      MIRBuilder, ArgInfos, OrigArgIndices,
      [&](ISD::ArgFlagsTy flags, EVT vt, EVT argvt, bool used, unsigned origIdx,
          unsigned partOffs) {
        Outs.emplace_back(flags, vt, argvt, used, origIdx, partOffs);
      });

  SmallVector<CCValAssign, 8> ArgLocs;
  MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                     F.getContext());

  CCInfo.AllocateStack(ABI.GetCalleeAllocdArgSizeInBytes(CallConv), 1);
  const char *Call = Callee.isSymbol() ? Callee.getSymbolName() : nullptr;
  CCInfo.AnalyzeCallOperands(Outs, TLI.CCAssignFnForCall(), FuncOrigArgs, Call);

  OutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), MIB);
  if (!RetHandler.handle(ArgLocs, ArgInfos)) {
    return false;
  }

  unsigned NextStackOffset = CCInfo.getNextStackOffset();
  const TargetFrameLowering *TFL = MF.getSubtarget().getFrameLowering();
  unsigned StackAlignment = TFL->getStackAlignment();
  NextStackOffset = alignTo(NextStackOffset, StackAlignment);
  CallSeqStart.addImm(NextStackOffset).addImm(0);

  MIRBuilder.insertInstr(MIB);

  if (OrigRet.Reg) {

    ArgInfos.clear();
    SmallVector<unsigned, 8> OrigRetIndices;

    splitToValueTypes(OrigRet, 0, ArgInfos, OrigRetIndices);

    SmallVector<ISD::InputArg, 8> Ins;
    subTargetRegTypeForCallingConv(
        MIRBuilder, ArgInfos, OrigRetIndices,
        [&](ISD::ArgFlagsTy flags, EVT vt, EVT argvt, bool used,
            unsigned origIdx, unsigned partOffs) {
          Ins.emplace_back(flags, vt, argvt, used, origIdx, partOffs);
        });

    SmallVector<CCValAssign, 8> ArgLocs;
    MipsCCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
                       F.getContext());

    CCInfo.AnalyzeCallResult(Ins, TLI.CCAssignFnForReturn(), OrigRet.Ty, Call);

    CallReturnHandler Handler(MIRBuilder, MF.getRegInfo(), MIB);
    if (!Handler.handle(ArgLocs, ArgInfos))
      return false;
  }

  MIRBuilder.buildInstr(Mips::ADJCALLSTACKUP).addImm(NextStackOffset).addImm(0);

  return true;
}

void MipsCallLowering::subTargetRegTypeForCallingConv(
    MachineIRBuilder &MIRBuilder, ArrayRef<ArgInfo> Args,
    ArrayRef<unsigned> OrigArgIndices, const FunTy &PushBack) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  const MipsTargetLowering &TLI = *getTLI<MipsTargetLowering>();

  unsigned ArgNo = 0;
  for (auto &Arg : Args) {

    EVT VT = TLI.getValueType(DL, Arg.Ty);
    MVT RegisterVT = TLI.getRegisterTypeForCallingConv(F.getContext(),
                                                       F.getCallingConv(), VT);

    ISD::ArgFlagsTy Flags = Arg.Flags;
    Flags.setOrigAlign(TLI.getABIAlignmentForCallingConv(Arg.Ty, DL));

    PushBack(Flags, RegisterVT, VT, true, OrigArgIndices[ArgNo], 0);

    ++ArgNo;
  }
}

void MipsCallLowering::splitToValueTypes(
    const ArgInfo &OrigArg, unsigned OriginalIndex,
    SmallVectorImpl<ArgInfo> &SplitArgs,
    SmallVectorImpl<unsigned> &SplitArgsOrigIndices) const {

  // TODO : perform structure and array split. For now we only deal with
  // types that pass isSupportedType check.
  SplitArgs.push_back(OrigArg);
  SplitArgsOrigIndices.push_back(OriginalIndex);
}
