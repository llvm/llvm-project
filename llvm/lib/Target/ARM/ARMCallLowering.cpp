//===-- llvm/lib/Target/ARM/ARMCallLowering.cpp - Call lowering -----------===//
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

#include "ARMCallLowering.h"

#include "ARMBaseInstrInfo.h"
#include "ARMISelLowering.h"
#include "ARMSubtarget.h"

#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

ARMCallLowering::ARMCallLowering(const ARMTargetLowering &TLI)
    : CallLowering(&TLI) {}

static bool isSupportedType(const DataLayout &DL, const ARMTargetLowering &TLI,
                            Type *T) {
  EVT VT = TLI.getValueType(DL, T, true);
  if (!VT.isSimple() || VT.isVector())
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
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB), StackSize(0) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    LLT p0 = LLT::pointer(0, 32);
    LLT s32 = LLT::scalar(32);
    unsigned SPReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildCopy(SPReg, ARM::SP);

    unsigned OffsetReg = MRI.createGenericVirtualRegister(s32);
    MIRBuilder.buildConstant(OffsetReg, Offset);

    unsigned AddrReg = MRI.createGenericVirtualRegister(p0);
    MIRBuilder.buildGEP(AddrReg, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MIRBuilder.getMF(), Offset);
    return AddrReg;
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    assert(VA.getValVT().getSizeInBits() <= 64 && "Unsupported value size");
    assert(VA.getLocVT().getSizeInBits() <= 64 && "Unsupported location size");

    unsigned ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    unsigned ExtReg = extendRegister(ValVReg, VA);
    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOStore, VA.getLocVT().getStoreSize(),
        /* Alignment */ 0);
    MIRBuilder.buildStore(ExtReg, Addr, *MMO);
  }

  unsigned assignCustomValue(const CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs) override {
    CCValAssign VA = VAs[0];
    assert(VA.needsCustom() && "Value doesn't need custom handling");
    assert(VA.getValVT() == MVT::f64 && "Unsupported type");

    CCValAssign NextVA = VAs[1];
    assert(NextVA.needsCustom() && "Value doesn't need custom handling");
    assert(NextVA.getValVT() == MVT::f64 && "Unsupported type");

    assert(VA.getValNo() == NextVA.getValNo() &&
           "Values belong to different arguments");

    assert(VA.isRegLoc() && "Value should be in reg");
    assert(NextVA.isRegLoc() && "Value should be in reg");

    unsigned NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                          MRI.createGenericVirtualRegister(LLT::scalar(32))};
    MIRBuilder.buildExtract(NewRegs[0], Arg.Reg, 0);
    MIRBuilder.buildExtract(NewRegs[1], Arg.Reg, 32);

    bool IsLittle = MIRBuilder.getMF().getSubtarget<ARMSubtarget>().isLittle();
    if (!IsLittle)
      std::swap(NewRegs[0], NewRegs[1]);

    assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
    assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

    return 1;
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, CCState &State) override {
    if (AssignFn(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State))
      return true;

    StackSize =
        std::max(StackSize, static_cast<uint64_t>(State.getNextStackOffset()));
    return false;
  }

  MachineInstrBuilder &MIB;
  uint64_t StackSize;
};
} // End anonymous namespace.

void ARMCallLowering::splitToValueTypes(const ArgInfo &OrigArg,
                                        SmallVectorImpl<ArgInfo> &SplitArgs,
                                        const DataLayout &DL,
                                        MachineRegisterInfo &MRI) const {
  const ARMTargetLowering &TLI = *getTLI<ARMTargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  assert(SplitVTs.size() == 1 && "Unsupported type");

  // Even if there is no splitting to do, we still want to replace the original
  // type (e.g. pointer type -> integer).
  SplitArgs.emplace_back(OrigArg.Reg, SplitVTs[0].getTypeForEVT(Ctx),
                         OrigArg.Flags, OrigArg.IsFixed);
}

/// Lower the return value for the already existing \p Ret. This assumes that
/// \p MIRBuilder's insertion point is correct.
bool ARMCallLowering::lowerReturnVal(MachineIRBuilder &MIRBuilder,
                                     const Value *Val, unsigned VReg,
                                     MachineInstrBuilder &Ret) const {
  if (!Val)
    // Nothing to do here.
    return true;

  auto &MF = MIRBuilder.getMF();
  const auto &F = *MF.getFunction();

  auto DL = MF.getDataLayout();
  auto &TLI = *getTLI<ARMTargetLowering>();
  if (!isSupportedType(DL, TLI, Val->getType()))
    return false;

  SmallVector<ArgInfo, 4> SplitVTs;
  ArgInfo RetInfo(VReg, Val->getType());
  setArgFlags(RetInfo, AttributeList::ReturnIndex, DL, F);
  splitToValueTypes(RetInfo, SplitVTs, DL, MF.getRegInfo());

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForReturn(F.getCallingConv(), F.isVarArg());

  OutgoingValueHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret, AssignFn);
  return handleAssignments(MIRBuilder, SplitVTs, RetHandler);
}

bool ARMCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                  const Value *Val, unsigned VReg) const {
  assert(!Val == !VReg && "Return value without a vreg");

  auto Ret = MIRBuilder.buildInstrNoInsert(ARM::BX_RET).add(predOps(ARMCC::AL));

  if (!lowerReturnVal(MIRBuilder, Val, VReg, Ret))
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

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    auto &MFI = MIRBuilder.getMF().getFrameInfo();

    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

    unsigned AddrReg =
        MRI.createGenericVirtualRegister(LLT::pointer(MPO.getAddrSpace(), 32));
    MIRBuilder.buildFrameIndex(AddrReg, FI);

    return AddrReg;
  }

  void assignValueToAddress(unsigned ValVReg, unsigned Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    assert((Size == 1 || Size == 2 || Size == 4 || Size == 8) &&
           "Unsupported size");

    if (VA.getLocInfo() == CCValAssign::SExt ||
        VA.getLocInfo() == CCValAssign::ZExt) {
      // If the value is zero- or sign-extended, its size becomes 4 bytes, so
      // that's what we should load.
      Size = 4;
      assert(MRI.getType(ValVReg).isScalar() && "Only scalars supported atm");
      MRI.setType(ValVReg, LLT::scalar(32));
    }

    auto MMO = MIRBuilder.getMF().getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad, Size, /* Alignment */ 0);
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  void assignValueToReg(unsigned ValVReg, unsigned PhysReg,
                        CCValAssign &VA) override {
    assert(VA.isRegLoc() && "Value shouldn't be assigned to reg");
    assert(VA.getLocReg() == PhysReg && "Assigning to the wrong reg?");

    assert(VA.getValVT().getSizeInBits() <= 64 && "Unsupported value size");
    assert(VA.getLocVT().getSizeInBits() <= 64 && "Unsupported location size");

    // The necesary extensions are handled on the other side of the ABI
    // boundary.
    markPhysRegUsed(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

  unsigned assignCustomValue(const ARMCallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs) override {
    CCValAssign VA = VAs[0];
    assert(VA.needsCustom() && "Value doesn't need custom handling");
    assert(VA.getValVT() == MVT::f64 && "Unsupported type");

    CCValAssign NextVA = VAs[1];
    assert(NextVA.needsCustom() && "Value doesn't need custom handling");
    assert(NextVA.getValVT() == MVT::f64 && "Unsupported type");

    assert(VA.getValNo() == NextVA.getValNo() &&
           "Values belong to different arguments");

    assert(VA.isRegLoc() && "Value should be in reg");
    assert(NextVA.isRegLoc() && "Value should be in reg");

    unsigned NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                          MRI.createGenericVirtualRegister(LLT::scalar(32))};

    assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
    assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

    bool IsLittle = MIRBuilder.getMF().getSubtarget<ARMSubtarget>().isLittle();
    if (!IsLittle)
      std::swap(NewRegs[0], NewRegs[1]);

    MIRBuilder.buildSequence(Arg.Reg, NewRegs, {0, 32});

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
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};
} // End anonymous namespace

bool ARMCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                           const Function &F,
                                           ArrayRef<unsigned> VRegs) const {
  // Quick exit if there aren't any args
  if (F.arg_empty())
    return true;

  if (F.isVarArg())
    return false;

  auto &MF = MIRBuilder.getMF();
  auto DL = MF.getDataLayout();
  auto &TLI = *getTLI<ARMTargetLowering>();

  auto Subtarget = TLI.getSubtarget();

  if (Subtarget->isThumb())
    return false;

  // FIXME: Support soft float (when we're ready to generate libcalls)
  if (Subtarget->useSoftFloat() || !Subtarget->hasVFP2())
    return false;

  for (auto &Arg : F.args())
    if (!isSupportedType(DL, TLI, Arg.getType()))
      return false;

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), F.isVarArg());

  SmallVector<ArgInfo, 8> ArgInfos;
  unsigned Idx = 0;
  for (auto &Arg : F.args()) {
    ArgInfo AInfo(VRegs[Idx], Arg.getType());
    setArgFlags(AInfo, Idx + 1, DL, F);
    splitToValueTypes(AInfo, ArgInfos, DL, MF.getRegInfo());
    Idx++;
  }

  FormalArgHandler ArgHandler(MIRBuilder, MIRBuilder.getMF().getRegInfo(),
                              AssignFn);
  return handleAssignments(MIRBuilder, ArgInfos, ArgHandler);
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
} // End anonymous namespace.

bool ARMCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                CallingConv::ID CallConv,
                                const MachineOperand &Callee,
                                const ArgInfo &OrigRet,
                                ArrayRef<ArgInfo> OrigArgs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const auto &TLI = *getTLI<ARMTargetLowering>();
  const auto &DL = MF.getDataLayout();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (MF.getSubtarget<ARMSubtarget>().genLongCalls())
    return false;

  auto CallSeqStart = MIRBuilder.buildInstr(ARM::ADJCALLSTACKDOWN);

  // Create the call instruction so we can add the implicit uses of arg
  // registers, but don't insert it yet.
  auto MIB = MIRBuilder.buildInstrNoInsert(ARM::BLX).add(Callee).addRegMask(
      TRI->getCallPreservedMask(MF, CallConv));

  SmallVector<ArgInfo, 8> ArgInfos;
  for (auto Arg : OrigArgs) {
    if (!isSupportedType(DL, TLI, Arg.Ty))
      return false;

    if (!Arg.IsFixed)
      return false;

    splitToValueTypes(Arg, ArgInfos, DL, MRI);
  }

  auto ArgAssignFn = TLI.CCAssignFnForCall(CallConv, /*IsVarArg=*/false);
  OutgoingValueHandler ArgHandler(MIRBuilder, MRI, MIB, ArgAssignFn);
  if (!handleAssignments(MIRBuilder, ArgInfos, ArgHandler))
    return false;

  // Now we can add the actual call instruction to the correct basic block.
  MIRBuilder.insertInstr(MIB);

  if (!OrigRet.Ty->isVoidTy()) {
    if (!isSupportedType(DL, TLI, OrigRet.Ty))
      return false;

    ArgInfos.clear();
    splitToValueTypes(OrigRet, ArgInfos, DL, MRI);

    auto RetAssignFn = TLI.CCAssignFnForReturn(CallConv, /*IsVarArg=*/false);
    CallReturnHandler RetHandler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, ArgInfos, RetHandler))
      return false;
  }

  // We now know the size of the stack - update the ADJCALLSTACKDOWN
  // accordingly.
  CallSeqStart.addImm(ArgHandler.StackSize).add(predOps(ARMCC::AL));

  MIRBuilder.buildInstr(ARM::ADJCALLSTACKUP)
      .addImm(ArgHandler.StackSize)
      .addImm(0)
      .add(predOps(ARMCC::AL));

  return true;
}
