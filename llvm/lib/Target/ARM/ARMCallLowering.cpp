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
struct FuncReturnHandler : public CallLowering::ValueHandler {
  FuncReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder &MIB, CCAssignFn *AssignFn)
    : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  unsigned getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    llvm_unreachable("Don't know how to get a stack address yet");
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
    llvm_unreachable("Don't know how to assign a value to an address yet");
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

    MIRBuilder.buildExtract(NewRegs, {0, 32}, Arg.Reg);

    bool IsLittle = MIRBuilder.getMF().getSubtarget<ARMSubtarget>().isLittle();
    if (!IsLittle)
      std::swap(NewRegs[0], NewRegs[1]);

    assignValueToReg(NewRegs[0], VA.getLocReg(), VA);
    assignValueToReg(NewRegs[1], NextVA.getLocReg(), NextVA);

    return 1;
  }

  MachineInstrBuilder &MIB;
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
  setArgFlags(RetInfo, AttributeSet::ReturnIndex, DL, F);
  splitToValueTypes(RetInfo, SplitVTs, DL, MF.getRegInfo());

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForReturn(F.getCallingConv(), F.isVarArg());

  FuncReturnHandler RetHandler(MIRBuilder, MF.getRegInfo(), Ret, AssignFn);
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
struct FormalArgHandler : public CallLowering::ValueHandler {
  FormalArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
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
      // If the argument is zero- or sign-extended by the caller, its size
      // becomes 4 bytes, so that's what we should load.
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

    // The caller should handle all necesary extensions.
    MIRBuilder.getMBB().addLiveIn(PhysReg);
    MIRBuilder.buildCopy(ValVReg, PhysReg);
  }

  unsigned assignCustomValue(const llvm::ARMCallLowering::ArgInfo &Arg,
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

  auto &Args = F.getArgumentList();
  for (auto &Arg : Args)
    if (!isSupportedType(DL, TLI, Arg.getType()))
      return false;

  CCAssignFn *AssignFn =
      TLI.CCAssignFnForCall(F.getCallingConv(), F.isVarArg());

  SmallVector<ArgInfo, 8> ArgInfos;
  unsigned Idx = 0;
  for (auto &Arg : Args) {
    ArgInfo AInfo(VRegs[Idx], Arg.getType());
    setArgFlags(AInfo, Idx + 1, DL, F);
    splitToValueTypes(AInfo, ArgInfos, DL, MF.getRegInfo());
    Idx++;
  }

  FormalArgHandler ArgHandler(MIRBuilder, MIRBuilder.getMF().getRegInfo(),
                              AssignFn);
  return handleAssignments(MIRBuilder, ArgInfos, ArgHandler);
}
