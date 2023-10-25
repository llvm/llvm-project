//===-- RISCVCallLowering.cpp - Call lowering -------------------*- C++ -*-===//
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

#include "RISCVCallLowering.h"
#include "RISCVISelLowering.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;

namespace {

struct RISCVOutgoingValueAssigner : public CallLowering::OutgoingValueAssigner {
private:
  // The function used internally to assign args - we ignore the AssignFn stored
  // by OutgoingValueAssigner since RISC-V implements its CC using a custom
  // function with a different signature.
  RISCVTargetLowering::RISCVCCAssignFn *RISCVAssignFn;

  // Whether this is assigning args for a return.
  bool IsRet;

public:
  RISCVOutgoingValueAssigner(
      RISCVTargetLowering::RISCVCCAssignFn *RISCVAssignFn_, bool IsRet)
      : CallLowering::OutgoingValueAssigner(nullptr),
        RISCVAssignFn(RISCVAssignFn_), IsRet(IsRet) {}

  bool assignArg(unsigned ValNo, EVT OrigVT, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    MachineFunction &MF = State.getMachineFunction();
    const DataLayout &DL = MF.getDataLayout();
    const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();

    return RISCVAssignFn(DL, Subtarget.getTargetABI(), ValNo, ValVT, LocVT,
                         LocInfo, Flags, State, /*IsFixed=*/true, IsRet,
                         Info.Ty, *Subtarget.getTargetLowering(),
                         /*FirstMaskArgument=*/std::nullopt);
  }
};

struct RISCVOutgoingValueHandler : public CallLowering::OutgoingValueHandler {
  RISCVOutgoingValueHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                            MachineInstrBuilder MIB)
      : OutgoingValueHandler(B, MRI), MIB(MIB),
        Subtarget(MIRBuilder.getMF().getSubtarget<RISCVSubtarget>()) {}
  Register getStackAddress(uint64_t MemSize, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override {
    MachineFunction &MF = MIRBuilder.getMF();
    LLT p0 = LLT::pointer(0, Subtarget.getXLen());
    LLT sXLen = LLT::scalar(Subtarget.getXLen());

    if (!SPReg)
      SPReg = MIRBuilder.buildCopy(p0, Register(RISCV::X2)).getReg(0);

    auto OffsetReg = MIRBuilder.buildConstant(sXLen, Offset);

    auto AddrReg = MIRBuilder.buildPtrAdd(p0, SPReg, OffsetReg);

    MPO = MachinePointerInfo::getStack(MF, Offset);
    return AddrReg.getReg(0);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            const MachinePointerInfo &MPO,
                            const CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();
    uint64_t LocMemOffset = VA.getLocMemOffset();

    // TODO: Move StackAlignment to subtarget and share with FrameLowering.
    auto MMO =
        MF.getMachineMemOperand(MPO, MachineMemOperand::MOStore, MemTy,
                                commonAlignment(Align(16), LocMemOffset));

    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildStore(ExtReg, Addr, *MMO);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        const CCValAssign &VA) override {
    // If we're passing an f32 value into an i64, anyextend before copying.
    if (VA.getLocVT() == MVT::i64 && VA.getValVT() == MVT::f32)
      ValVReg = MIRBuilder.buildAnyExt(LLT::scalar(64), ValVReg).getReg(0);

    Register ExtReg = extendRegister(ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }

  unsigned assignCustomValue(CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs,
                             std::function<void()> *Thunk) override {
    assert(VAs.size() >= 2 && "Expected at least 2 VAs.");
    const CCValAssign &VALo = VAs[0];
    const CCValAssign &VAHi = VAs[1];

    assert(VAHi.needsCustom() && "Value doesn't need custom handling");
    assert(VALo.getValNo() == VAHi.getValNo() &&
           "Values belong to different arguments");

    assert(VALo.getLocVT() == MVT::i32 && VAHi.getLocVT() == MVT::i32 &&
           VALo.getValVT() == MVT::f64 && VAHi.getValVT() == MVT::f64 &&
           "unexpected custom value");

    Register NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                          MRI.createGenericVirtualRegister(LLT::scalar(32))};
    MIRBuilder.buildUnmerge(NewRegs, Arg.Regs[0]);

    if (VAHi.isMemLoc()) {
      LLT MemTy(VAHi.getLocVT());

      MachinePointerInfo MPO;
      Register StackAddr = getStackAddress(
          MemTy.getSizeInBytes(), VAHi.getLocMemOffset(), MPO, Arg.Flags[0]);

      assignValueToAddress(NewRegs[1], StackAddr, MemTy, MPO,
                           const_cast<CCValAssign &>(VAHi));
    }

    auto assignFunc = [=]() {
      assignValueToReg(NewRegs[0], VALo.getLocReg(), VALo);
      if (VAHi.isRegLoc())
        assignValueToReg(NewRegs[1], VAHi.getLocReg(), VAHi);
    };

    if (Thunk) {
      *Thunk = assignFunc;
      return 1;
    }

    assignFunc();
    return 1;
  }

private:
  MachineInstrBuilder MIB;

  // Cache the SP register vreg if we need it more than once in this call site.
  Register SPReg;

  const RISCVSubtarget &Subtarget;
};

struct RISCVIncomingValueAssigner : public CallLowering::IncomingValueAssigner {
private:
  // The function used internally to assign args - we ignore the AssignFn stored
  // by IncomingValueAssigner since RISC-V implements its CC using a custom
  // function with a different signature.
  RISCVTargetLowering::RISCVCCAssignFn *RISCVAssignFn;

  // Whether this is assigning args from a return.
  bool IsRet;

public:
  RISCVIncomingValueAssigner(
      RISCVTargetLowering::RISCVCCAssignFn *RISCVAssignFn_, bool IsRet)
      : CallLowering::IncomingValueAssigner(nullptr),
        RISCVAssignFn(RISCVAssignFn_), IsRet(IsRet) {}

  bool assignArg(unsigned ValNo, EVT OrigVT, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info, ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    MachineFunction &MF = State.getMachineFunction();
    const DataLayout &DL = MF.getDataLayout();
    const RISCVSubtarget &Subtarget = MF.getSubtarget<RISCVSubtarget>();

    return RISCVAssignFn(DL, Subtarget.getTargetABI(), ValNo, ValVT, LocVT,
                         LocInfo, Flags, State, /*IsFixed=*/true, IsRet,
                         Info.Ty, *Subtarget.getTargetLowering(),
                         /*FirstMaskArgument=*/std::nullopt);
  }
};

struct RISCVIncomingValueHandler : public CallLowering::IncomingValueHandler {
  RISCVIncomingValueHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI)
      : IncomingValueHandler(B, MRI),
        Subtarget(MIRBuilder.getMF().getSubtarget<RISCVSubtarget>()) {}

  Register getStackAddress(uint64_t MemSize, int64_t Offset,
                           MachinePointerInfo &MPO,
                           ISD::ArgFlagsTy Flags) override {
    MachineFrameInfo &MFI = MIRBuilder.getMF().getFrameInfo();

    int FI = MFI.CreateFixedObject(MemSize, Offset, /*Immutable=*/true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);
    return MIRBuilder.buildFrameIndex(LLT::pointer(0, Subtarget.getXLen()), FI)
        .getReg(0);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, LLT MemTy,
                            const MachinePointerInfo &MPO,
                            const CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();
    auto MMO = MF.getMachineMemOperand(MPO, MachineMemOperand::MOLoad, MemTy,
                                       inferAlignFromPtrInfo(MF, MPO));
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        const CCValAssign &VA) override {
    markPhysRegUsed(PhysReg);
    IncomingValueHandler::assignValueToReg(ValVReg, PhysReg, VA);
  }

  unsigned assignCustomValue(CallLowering::ArgInfo &Arg,
                             ArrayRef<CCValAssign> VAs,
                             std::function<void()> *Thunk) override {
    assert(VAs.size() >= 2 && "Expected at least 2 VAs.");
    const CCValAssign &VALo = VAs[0];
    const CCValAssign &VAHi = VAs[1];

    assert(VAHi.needsCustom() && "Value doesn't need custom handling");
    assert(VALo.getValNo() == VAHi.getValNo() &&
           "Values belong to different arguments");

    assert(VALo.getLocVT() == MVT::i32 && VAHi.getLocVT() == MVT::i32 &&
           VALo.getValVT() == MVT::f64 && VAHi.getValVT() == MVT::f64 &&
           "unexpected custom value");

    Register NewRegs[] = {MRI.createGenericVirtualRegister(LLT::scalar(32)),
                          MRI.createGenericVirtualRegister(LLT::scalar(32))};

    if (VAHi.isMemLoc()) {
      LLT MemTy(VAHi.getLocVT());

      MachinePointerInfo MPO;
      Register StackAddr = getStackAddress(
          MemTy.getSizeInBytes(), VAHi.getLocMemOffset(), MPO, Arg.Flags[0]);

      assignValueToAddress(NewRegs[1], StackAddr, MemTy, MPO,
                           const_cast<CCValAssign &>(VAHi));
    }

    assignValueToReg(NewRegs[0], VALo.getLocReg(), VALo);
    if (VAHi.isRegLoc())
      assignValueToReg(NewRegs[1], VAHi.getLocReg(), VAHi);

    MIRBuilder.buildMergeLikeInstr(Arg.Regs[0], NewRegs);

    return 1;
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(MCRegister PhysReg) = 0;

private:
  const RISCVSubtarget &Subtarget;
};

struct RISCVFormalArgHandler : public RISCVIncomingValueHandler {
  RISCVFormalArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI)
      : RISCVIncomingValueHandler(B, MRI) {}

  void markPhysRegUsed(MCRegister PhysReg) override {
    MIRBuilder.getMRI()->addLiveIn(PhysReg);
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

struct RISCVCallReturnHandler : public RISCVIncomingValueHandler {
  RISCVCallReturnHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                         MachineInstrBuilder &MIB)
      : RISCVIncomingValueHandler(B, MRI), MIB(MIB) {}

  void markPhysRegUsed(MCRegister PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
};

} // namespace

RISCVCallLowering::RISCVCallLowering(const RISCVTargetLowering &TLI)
    : CallLowering(&TLI) {}

// TODO: Support all argument types.
static bool isSupportedArgumentType(Type *T, const RISCVSubtarget &Subtarget) {
  // TODO: Integers larger than 2*XLen are passed indirectly which is not
  // supported yet.
  if (T->isIntegerTy())
    return T->getIntegerBitWidth() <= Subtarget.getXLen() * 2;
  if (T->isFloatTy() || T->isDoubleTy())
    return true;
  if (T->isPointerTy())
    return true;
  return false;
}

// TODO: Only integer, pointer and aggregate types are supported now.
static bool isSupportedReturnType(Type *T, const RISCVSubtarget &Subtarget) {
  // TODO: Integers larger than 2*XLen are passed indirectly which is not
  // supported yet.
  if (T->isIntegerTy())
    return T->getIntegerBitWidth() <= Subtarget.getXLen() * 2;
  if (T->isFloatTy() || T->isDoubleTy())
    return true;
  if (T->isPointerTy())
    return true;

  if (T->isArrayTy())
    return isSupportedReturnType(T->getArrayElementType(), Subtarget);

  if (T->isStructTy()) {
    auto StructT = cast<StructType>(T);
    for (unsigned i = 0, e = StructT->getNumElements(); i != e; ++i)
      if (!isSupportedReturnType(StructT->getElementType(i), Subtarget))
        return false;
    return true;
  }

  return false;
}

bool RISCVCallLowering::lowerReturnVal(MachineIRBuilder &MIRBuilder,
                                       const Value *Val,
                                       ArrayRef<Register> VRegs,
                                       MachineInstrBuilder &Ret) const {
  if (!Val)
    return true;

  const RISCVSubtarget &Subtarget =
      MIRBuilder.getMF().getSubtarget<RISCVSubtarget>();
  if (!isSupportedReturnType(Val->getType(), Subtarget))
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  const DataLayout &DL = MF.getDataLayout();
  const Function &F = MF.getFunction();
  CallingConv::ID CC = F.getCallingConv();

  ArgInfo OrigRetInfo(VRegs, Val->getType(), 0);
  setArgFlags(OrigRetInfo, AttributeList::ReturnIndex, DL, F);

  SmallVector<ArgInfo, 4> SplitRetInfos;
  splitToValueTypes(OrigRetInfo, SplitRetInfos, DL, CC);

  RISCVOutgoingValueAssigner Assigner(
      CC == CallingConv::Fast ? RISCV::CC_RISCV_FastCC : RISCV::CC_RISCV,
      /*IsRet=*/true);
  RISCVOutgoingValueHandler Handler(MIRBuilder, MF.getRegInfo(), Ret);
  return determineAndHandleAssignments(Handler, Assigner, SplitRetInfos,
                                       MIRBuilder, CC, F.isVarArg());
}

bool RISCVCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                    const Value *Val, ArrayRef<Register> VRegs,
                                    FunctionLoweringInfo &FLI) const {
  assert(!Val == VRegs.empty() && "Return value without a vreg");
  MachineInstrBuilder Ret = MIRBuilder.buildInstrNoInsert(RISCV::PseudoRET);

  if (!lowerReturnVal(MIRBuilder, Val, VRegs, Ret))
    return false;

  MIRBuilder.insertInstr(Ret);
  return true;
}

bool RISCVCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                             const Function &F,
                                             ArrayRef<ArrayRef<Register>> VRegs,
                                             FunctionLoweringInfo &FLI) const {
  // Early exit if there are no arguments.
  if (F.arg_empty())
    return true;

  // TODO: Support vararg functions.
  if (F.isVarArg())
    return false;

  const RISCVSubtarget &Subtarget =
      MIRBuilder.getMF().getSubtarget<RISCVSubtarget>();
  for (auto &Arg : F.args()) {
    if (!isSupportedArgumentType(Arg.getType(), Subtarget))
      return false;
  }

  MachineFunction &MF = MIRBuilder.getMF();
  const DataLayout &DL = MF.getDataLayout();
  CallingConv::ID CC = F.getCallingConv();

  SmallVector<ArgInfo, 32> SplitArgInfos;
  unsigned Index = 0;
  for (auto &Arg : F.args()) {
    // Construct the ArgInfo object from destination register and argument type.
    ArgInfo AInfo(VRegs[Index], Arg.getType(), Index);
    setArgFlags(AInfo, Index + AttributeList::FirstArgIndex, DL, F);

    // Handle any required merging from split value types from physical
    // registers into the desired VReg. ArgInfo objects are constructed
    // correspondingly and appended to SplitArgInfos.
    splitToValueTypes(AInfo, SplitArgInfos, DL, CC);

    ++Index;
  }

  RISCVIncomingValueAssigner Assigner(
      CC == CallingConv::Fast ? RISCV::CC_RISCV_FastCC : RISCV::CC_RISCV,
      /*IsRet=*/false);
  RISCVFormalArgHandler Handler(MIRBuilder, MF.getRegInfo());

  return determineAndHandleAssignments(Handler, Assigner, SplitArgInfos,
                                       MIRBuilder, CC, F.isVarArg());
}

bool RISCVCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                  CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const DataLayout &DL = MF.getDataLayout();
  const Function &F = MF.getFunction();
  CallingConv::ID CC = F.getCallingConv();

  const RISCVSubtarget &Subtarget =
      MIRBuilder.getMF().getSubtarget<RISCVSubtarget>();
  for (auto &AInfo : Info.OrigArgs) {
    if (!isSupportedArgumentType(AInfo.Ty, Subtarget))
      return false;
  }

  SmallVector<ArgInfo, 32> SplitArgInfos;
  SmallVector<ISD::OutputArg, 8> Outs;
  for (auto &AInfo : Info.OrigArgs) {
    // Handle any required unmerging of split value types from a given VReg into
    // physical registers. ArgInfo objects are constructed correspondingly and
    // appended to SplitArgInfos.
    splitToValueTypes(AInfo, SplitArgInfos, DL, CC);
  }

  // TODO: Support tail calls.
  Info.IsTailCall = false;

  if (!Info.Callee.isReg())
    Info.Callee.setTargetFlags(RISCVII::MO_CALL);

  MachineInstrBuilder Call =
      MIRBuilder
          .buildInstrNoInsert(Info.Callee.isReg() ? RISCV::PseudoCALLIndirect
                                                  : RISCV::PseudoCALL)
          .add(Info.Callee);

  RISCVOutgoingValueAssigner ArgAssigner(
      CC == CallingConv::Fast ? RISCV::CC_RISCV_FastCC : RISCV::CC_RISCV,
      /*IsRet=*/false);
  RISCVOutgoingValueHandler ArgHandler(MIRBuilder, MF.getRegInfo(), Call);
  if (!determineAndHandleAssignments(ArgHandler, ArgAssigner, SplitArgInfos,
                                     MIRBuilder, CC, Info.IsVarArg))
    return false;

  MIRBuilder.insertInstr(Call);

  if (Info.OrigRet.Ty->isVoidTy())
    return true;

  if (!isSupportedReturnType(Info.OrigRet.Ty, Subtarget))
    return false;

  SmallVector<ArgInfo, 4> SplitRetInfos;
  splitToValueTypes(Info.OrigRet, SplitRetInfos, DL, CC);

  // Assignments should be handled *before* the merging of values takes place.
  // To ensure this, the insert point is temporarily adjusted to just after the
  // call instruction.
  MachineBasicBlock::iterator CallInsertPt = Call;
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), std::next(CallInsertPt));

  RISCVIncomingValueAssigner RetAssigner(
      CC == CallingConv::Fast ? RISCV::CC_RISCV_FastCC : RISCV::CC_RISCV,
      /*IsRet=*/true);
  RISCVCallReturnHandler RetHandler(MIRBuilder, MF.getRegInfo(), Call);
  if (!determineAndHandleAssignments(RetHandler, RetAssigner, SplitRetInfos,
                                     MIRBuilder, CC, Info.IsVarArg))
    return false;

  // Readjust insert point to end of basic block.
  MIRBuilder.setMBB(MIRBuilder.getMBB());

  return true;
}
