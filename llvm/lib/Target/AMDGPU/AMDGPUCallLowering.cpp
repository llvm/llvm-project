//===-- llvm/lib/Target/AMDGPU/AMDGPUCallLowering.cpp - Call lowering -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUCallLowering.h"
#include "AMDGPU.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPUTargetMachine.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#define DEBUG_TYPE "amdgpu-call-lowering"

using namespace llvm;

namespace {

/// Wrapper around extendRegister to ensure we extend to a full 32-bit register.
static Register extendRegisterMin32(CallLowering::ValueHandler &Handler,
                                    Register ValVReg, CCValAssign &VA) {
  if (VA.getLocVT().getSizeInBits() < 32) {
    // 16-bit types are reported as legal for 32-bit registers. We need to
    // extend and do a 32-bit copy to avoid the verifier complaining about it.
    return Handler.MIRBuilder.buildAnyExt(LLT::scalar(32), ValVReg).getReg(0);
  }

  return Handler.extendRegister(ValVReg, VA);
}

struct AMDGPUOutgoingValueHandler : public CallLowering::OutgoingValueHandler {
  AMDGPUOutgoingValueHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                             MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : OutgoingValueHandler(B, MRI, AssignFn), MIB(MIB) {}

  MachineInstrBuilder MIB;

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    llvm_unreachable("not implemented");
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    llvm_unreachable("not implemented");
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    Register ExtReg = extendRegisterMin32(*this, ValVReg, VA);

    // If this is a scalar return, insert a readfirstlane just in case the value
    // ends up in a VGPR.
    // FIXME: Assert this is a shader return.
    const SIRegisterInfo *TRI
      = static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());
    if (TRI->isSGPRReg(MRI, PhysReg)) {
      auto ToSGPR = MIRBuilder.buildIntrinsic(Intrinsic::amdgcn_readfirstlane,
                                              {MRI.getType(ExtReg)}, false)
        .addReg(ExtReg);
      ExtReg = ToSGPR.getReg(0);
    }

    MIRBuilder.buildCopy(PhysReg, ExtReg);
    MIB.addUse(PhysReg, RegState::Implicit);
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info,
                 ISD::ArgFlagsTy Flags,
                 CCState &State) override {
    return AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
  }
};

struct AMDGPUIncomingArgHandler : public CallLowering::IncomingValueHandler {
  uint64_t StackUsed = 0;

  AMDGPUIncomingArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                           CCAssignFn *AssignFn)
      : IncomingValueHandler(B, MRI, AssignFn) {}

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    auto &MFI = MIRBuilder.getMF().getFrameInfo();
    int FI = MFI.CreateFixedObject(Size, Offset, true);
    MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);
    auto AddrReg = MIRBuilder.buildFrameIndex(
        LLT::pointer(AMDGPUAS::PRIVATE_ADDRESS, 32), FI);
    StackUsed = std::max(StackUsed, Size + Offset);
    return AddrReg.getReg(0);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    markPhysRegUsed(PhysReg);

    if (VA.getLocVT().getSizeInBits() < 32) {
      // 16-bit types are reported as legal for 32-bit registers. We need to do
      // a 32-bit copy, and truncate to avoid the verifier complaining about it.
      auto Copy = MIRBuilder.buildCopy(LLT::scalar(32), PhysReg);

      // If we have signext/zeroext, it applies to the whole 32-bit register
      // before truncation.
      auto Extended =
          buildExtensionHint(VA, Copy.getReg(0), LLT(VA.getLocVT()));
      MIRBuilder.buildTrunc(ValVReg, Extended);
      return;
    }

    IncomingValueHandler::assignValueToReg(ValVReg, PhysReg, VA);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t MemSize,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();

    // The reported memory location may be wider than the value.
    const LLT RegTy = MRI.getType(ValVReg);
    MemSize = std::min(static_cast<uint64_t>(RegTy.getSizeInBytes()), MemSize);

    // FIXME: Get alignment
    auto MMO = MF.getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, MemSize,
        inferAlignFromPtrInfo(MF, MPO));
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;
};

struct FormalArgHandler : public AMDGPUIncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn)
      : AMDGPUIncomingArgHandler(B, MRI, AssignFn) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
  }
};

struct CallReturnHandler : public AMDGPUIncomingArgHandler {
  CallReturnHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                    MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : AMDGPUIncomingArgHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIB.addDef(PhysReg, RegState::Implicit);
  }

  MachineInstrBuilder MIB;
};

struct AMDGPUOutgoingArgHandler : public AMDGPUOutgoingValueHandler {
  CCAssignFn *AssignFnVarArg;

  /// For tail calls, the byte offset of the call's argument area from the
  /// callee's. Unused elsewhere.
  int FPDiff;

  // Cache the SP register vreg if we need it more than once in this call site.
  Register SPReg;

  bool IsTailCall;

  AMDGPUOutgoingArgHandler(MachineIRBuilder &MIRBuilder,
                           MachineRegisterInfo &MRI, MachineInstrBuilder MIB,
                           CCAssignFn *AssignFn, CCAssignFn *AssignFnVarArg,
                           bool IsTailCall = false, int FPDiff = 0)
      : AMDGPUOutgoingValueHandler(MIRBuilder, MRI, MIB, AssignFn),
        AssignFnVarArg(AssignFnVarArg), FPDiff(FPDiff), IsTailCall(IsTailCall) {
  }

  Register getStackAddress(uint64_t Size, int64_t Offset,
                           MachinePointerInfo &MPO) override {
    MachineFunction &MF = MIRBuilder.getMF();
    const LLT PtrTy = LLT::pointer(AMDGPUAS::PRIVATE_ADDRESS, 32);
    const LLT S32 = LLT::scalar(32);

    if (IsTailCall) {
      llvm_unreachable("implement me");
    }

    const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    if (!SPReg)
      SPReg = MIRBuilder.buildCopy(PtrTy, MFI->getStackPtrOffsetReg()).getReg(0);

    auto OffsetReg = MIRBuilder.buildConstant(S32, Offset);

    auto AddrReg = MIRBuilder.buildPtrAdd(PtrTy, SPReg, OffsetReg);
    MPO = MachinePointerInfo::getStack(MF, Offset);
    return AddrReg.getReg(0);
  }

  void assignValueToReg(Register ValVReg, Register PhysReg,
                        CCValAssign &VA) override {
    MIB.addUse(PhysReg, RegState::Implicit);
    Register ExtReg = extendRegisterMin32(*this, ValVReg, VA);
    MIRBuilder.buildCopy(PhysReg, ExtReg);
  }

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();
    uint64_t LocMemOffset = VA.getLocMemOffset();
    const auto &ST = MF.getSubtarget<GCNSubtarget>();

    auto MMO = MF.getMachineMemOperand(
      MPO, MachineMemOperand::MOStore, Size,
      commonAlignment(ST.getStackAlignment(), LocMemOffset));
    MIRBuilder.buildStore(ValVReg, Addr, *MMO);
  }

  void assignValueToAddress(const CallLowering::ArgInfo &Arg,
                            unsigned ValRegIndex, Register Addr,
                            uint64_t MemSize, MachinePointerInfo &MPO,
                            CCValAssign &VA) override {
    Register ValVReg = VA.getLocInfo() != CCValAssign::LocInfo::FPExt
                           ? extendRegister(Arg.Regs[ValRegIndex], VA)
                           : Arg.Regs[ValRegIndex];

    // If we extended the value type we might need to adjust the MMO's
    // Size. This happens if ComputeValueVTs widened a small type value to a
    // legal register type (e.g. s8->s16)
    const LLT RegTy = MRI.getType(ValVReg);
    MemSize = std::min(MemSize, (uint64_t)RegTy.getSizeInBytes());
    assignValueToAddress(ValVReg, Addr, MemSize, MPO, VA);
  }
};
}

AMDGPUCallLowering::AMDGPUCallLowering(const AMDGPUTargetLowering &TLI)
  : CallLowering(&TLI) {
}

// FIXME: Compatability shim
static ISD::NodeType extOpcodeToISDExtOpcode(unsigned MIOpc) {
  switch (MIOpc) {
  case TargetOpcode::G_SEXT:
    return ISD::SIGN_EXTEND;
  case TargetOpcode::G_ZEXT:
    return ISD::ZERO_EXTEND;
  case TargetOpcode::G_ANYEXT:
    return ISD::ANY_EXTEND;
  default:
    llvm_unreachable("not an extend opcode");
  }
}

bool AMDGPUCallLowering::canLowerReturn(MachineFunction &MF,
                                        CallingConv::ID CallConv,
                                        SmallVectorImpl<BaseArgInfo> &Outs,
                                        bool IsVarArg) const {
  // For shaders. Vector types should be explicitly handled by CC.
  if (AMDGPU::isEntryFunctionCC(CallConv))
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs,
                 MF.getFunction().getContext());

  return checkReturn(CCInfo, Outs, TLI.CCAssignFnForReturn(CallConv, IsVarArg));
}

/// Lower the return value for the already existing \p Ret. This assumes that
/// \p B's insertion point is correct.
bool AMDGPUCallLowering::lowerReturnVal(MachineIRBuilder &B,
                                        const Value *Val, ArrayRef<Register> VRegs,
                                        MachineInstrBuilder &Ret) const {
  if (!Val)
    return true;

  auto &MF = B.getMF();
  const auto &F = MF.getFunction();
  const DataLayout &DL = MF.getDataLayout();
  MachineRegisterInfo *MRI = B.getMRI();
  LLVMContext &Ctx = F.getContext();

  CallingConv::ID CC = F.getCallingConv();
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();

  SmallVector<EVT, 8> SplitEVTs;
  ComputeValueVTs(TLI, DL, Val->getType(), SplitEVTs);
  assert(VRegs.size() == SplitEVTs.size() &&
         "For each split Type there should be exactly one VReg.");

  SmallVector<ArgInfo, 8> SplitRetInfos;

  for (unsigned i = 0; i < SplitEVTs.size(); ++i) {
    EVT VT = SplitEVTs[i];
    Register Reg = VRegs[i];
    ArgInfo RetInfo(Reg, VT.getTypeForEVT(Ctx));
    setArgFlags(RetInfo, AttributeList::ReturnIndex, DL, F);

    if (VT.isScalarInteger()) {
      unsigned ExtendOp = TargetOpcode::G_ANYEXT;
      if (RetInfo.Flags[0].isSExt()) {
        assert(RetInfo.Regs.size() == 1 && "expect only simple return values");
        ExtendOp = TargetOpcode::G_SEXT;
      } else if (RetInfo.Flags[0].isZExt()) {
        assert(RetInfo.Regs.size() == 1 && "expect only simple return values");
        ExtendOp = TargetOpcode::G_ZEXT;
      }

      EVT ExtVT = TLI.getTypeForExtReturn(Ctx, VT,
                                          extOpcodeToISDExtOpcode(ExtendOp));
      if (ExtVT != VT) {
        RetInfo.Ty = ExtVT.getTypeForEVT(Ctx);
        LLT ExtTy = getLLTForType(*RetInfo.Ty, DL);
        Reg = B.buildInstr(ExtendOp, {ExtTy}, {Reg}).getReg(0);
      }
    }

    if (Reg != RetInfo.Regs[0]) {
      RetInfo.Regs[0] = Reg;
      // Reset the arg flags after modifying Reg.
      setArgFlags(RetInfo, AttributeList::ReturnIndex, DL, F);
    }

    splitToValueTypes(RetInfo, SplitRetInfos, DL, CC);
  }

  CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(CC, F.isVarArg());
  AMDGPUOutgoingValueHandler RetHandler(B, *MRI, Ret, AssignFn);
  return handleAssignments(B, SplitRetInfos, RetHandler, CC, F.isVarArg());
}

bool AMDGPUCallLowering::lowerReturn(MachineIRBuilder &B, const Value *Val,
                                     ArrayRef<Register> VRegs,
                                     FunctionLoweringInfo &FLI) const {

  MachineFunction &MF = B.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MFI->setIfReturnsVoid(!Val);

  assert(!Val == VRegs.empty() && "Return value without a vreg");

  CallingConv::ID CC = B.getMF().getFunction().getCallingConv();
  const bool IsShader = AMDGPU::isShader(CC);
  const bool IsWaveEnd =
      (IsShader && MFI->returnsVoid()) || AMDGPU::isKernel(CC);
  if (IsWaveEnd) {
    B.buildInstr(AMDGPU::S_ENDPGM)
      .addImm(0);
    return true;
  }

  auto const &ST = MF.getSubtarget<GCNSubtarget>();

  unsigned ReturnOpc =
      IsShader ? AMDGPU::SI_RETURN_TO_EPILOG : AMDGPU::S_SETPC_B64_return;

  auto Ret = B.buildInstrNoInsert(ReturnOpc);
  Register ReturnAddrVReg;
  if (ReturnOpc == AMDGPU::S_SETPC_B64_return) {
    ReturnAddrVReg = MRI.createVirtualRegister(&AMDGPU::CCR_SGPR_64RegClass);
    Ret.addUse(ReturnAddrVReg);
  }

  if (!FLI.CanLowerReturn)
    insertSRetStores(B, Val->getType(), VRegs, FLI.DemoteRegister);
  else if (!lowerReturnVal(B, Val, VRegs, Ret))
    return false;

  if (ReturnOpc == AMDGPU::S_SETPC_B64_return) {
    const SIRegisterInfo *TRI = ST.getRegisterInfo();
    Register LiveInReturn = MF.addLiveIn(TRI->getReturnAddressReg(MF),
                                         &AMDGPU::SGPR_64RegClass);
    B.buildCopy(ReturnAddrVReg, LiveInReturn);
  }

  // TODO: Handle CalleeSavedRegsViaCopy.

  B.insertInstr(Ret);
  return true;
}

void AMDGPUCallLowering::lowerParameterPtr(Register DstReg, MachineIRBuilder &B,
                                           Type *ParamTy,
                                           uint64_t Offset) const {
  MachineFunction &MF = B.getMF();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  Register KernArgSegmentPtr =
    MFI->getPreloadedReg(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  Register KernArgSegmentVReg = MRI.getLiveInVirtReg(KernArgSegmentPtr);

  auto OffsetReg = B.buildConstant(LLT::scalar(64), Offset);

  B.buildPtrAdd(DstReg, KernArgSegmentVReg, OffsetReg);
}

void AMDGPUCallLowering::lowerParameter(MachineIRBuilder &B, Type *ParamTy,
                                        uint64_t Offset, Align Alignment,
                                        Register DstReg) const {
  MachineFunction &MF = B.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
  unsigned TypeSize = DL.getTypeStoreSize(ParamTy);

  LLT PtrTy = LLT::pointer(AMDGPUAS::CONSTANT_ADDRESS, 64);
  Register PtrReg = B.getMRI()->createGenericVirtualRegister(PtrTy);
  lowerParameterPtr(PtrReg, B, ParamTy, Offset);

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo,
      MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable |
          MachineMemOperand::MOInvariant,
      TypeSize, Alignment);

  B.buildLoad(DstReg, PtrReg, *MMO);
}

// Allocate special inputs passed in user SGPRs.
static void allocateHSAUserSGPRs(CCState &CCInfo,
                                 MachineIRBuilder &B,
                                 MachineFunction &MF,
                                 const SIRegisterInfo &TRI,
                                 SIMachineFunctionInfo &Info) {
  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info.hasPrivateSegmentBuffer()) {
    Register PrivateSegmentBufferReg = Info.addPrivateSegmentBuffer(TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SGPR_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info.hasDispatchPtr()) {
    Register DispatchPtrReg = Info.addDispatchPtr(TRI);
    MF.addLiveIn(DispatchPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info.hasQueuePtr()) {
    Register QueuePtrReg = Info.addQueuePtr(TRI);
    MF.addLiveIn(QueuePtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info.hasKernargSegmentPtr()) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    Register InputPtrReg = Info.addKernargSegmentPtr(TRI);
    const LLT P4 = LLT::pointer(AMDGPUAS::CONSTANT_ADDRESS, 64);
    Register VReg = MRI.createGenericVirtualRegister(P4);
    MRI.addLiveIn(InputPtrReg, VReg);
    B.getMBB().addLiveIn(InputPtrReg);
    B.buildCopy(VReg, InputPtrReg);
    CCInfo.AllocateReg(InputPtrReg);
  }

  if (Info.hasDispatchID()) {
    Register DispatchIDReg = Info.addDispatchID(TRI);
    MF.addLiveIn(DispatchIDReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info.hasFlatScratchInit()) {
    Register FlatScratchInitReg = Info.addFlatScratchInit(TRI);
    MF.addLiveIn(FlatScratchInitReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  // TODO: Add GridWorkGroupCount user SGPRs when used. For now with HSA we read
  // these from the dispatch pointer.
}

bool AMDGPUCallLowering::lowerFormalArgumentsKernel(
    MachineIRBuilder &B, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  MachineFunction &MF = B.getMF();
  const GCNSubtarget *Subtarget = &MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const SIRegisterInfo *TRI = Subtarget->getRegisterInfo();
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();

  const DataLayout &DL = F.getParent()->getDataLayout();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  allocateHSAUserSGPRs(CCInfo, B, MF, *TRI, *Info);

  unsigned i = 0;
  const Align KernArgBaseAlign(16);
  const unsigned BaseOffset = Subtarget->getExplicitKernelArgOffset(F);
  uint64_t ExplicitArgOffset = 0;

  // TODO: Align down to dword alignment and extract bits for extending loads.
  for (auto &Arg : F.args()) {
    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    unsigned AllocSize = DL.getTypeAllocSize(ArgTy);
    if (AllocSize == 0)
      continue;

    MaybeAlign ABIAlign = IsByRef ? Arg.getParamAlign() : None;
    if (!ABIAlign)
      ABIAlign = DL.getABITypeAlign(ArgTy);

    uint64_t ArgOffset = alignTo(ExplicitArgOffset, ABIAlign) + BaseOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, ABIAlign) + AllocSize;

    if (Arg.use_empty()) {
      ++i;
      continue;
    }

    Align Alignment = commonAlignment(KernArgBaseAlign, ArgOffset);

    if (IsByRef) {
      unsigned ByRefAS = cast<PointerType>(Arg.getType())->getAddressSpace();

      assert(VRegs[i].size() == 1 &&
             "expected only one register for byval pointers");
      if (ByRefAS == AMDGPUAS::CONSTANT_ADDRESS) {
        lowerParameterPtr(VRegs[i][0], B, ArgTy, ArgOffset);
      } else {
        const LLT ConstPtrTy = LLT::pointer(AMDGPUAS::CONSTANT_ADDRESS, 64);
        Register PtrReg = MRI.createGenericVirtualRegister(ConstPtrTy);
        lowerParameterPtr(PtrReg, B, ArgTy, ArgOffset);

        B.buildAddrSpaceCast(VRegs[i][0], PtrReg);
      }
    } else {
      ArrayRef<Register> OrigArgRegs = VRegs[i];
      Register ArgReg =
        OrigArgRegs.size() == 1
        ? OrigArgRegs[0]
        : MRI.createGenericVirtualRegister(getLLTForType(*ArgTy, DL));

      lowerParameter(B, ArgTy, ArgOffset, Alignment, ArgReg);
      if (OrigArgRegs.size() > 1)
        unpackRegs(OrigArgRegs, ArgReg, ArgTy, B);
    }

    ++i;
  }

  TLI.allocateSpecialEntryInputVGPRs(CCInfo, MF, *TRI, *Info);
  TLI.allocateSystemSGPRs(CCInfo, MF, *Info, F.getCallingConv(), false);
  return true;
}

bool AMDGPUCallLowering::lowerFormalArguments(
    MachineIRBuilder &B, const Function &F, ArrayRef<ArrayRef<Register>> VRegs,
    FunctionLoweringInfo &FLI) const {
  CallingConv::ID CC = F.getCallingConv();

  // The infrastructure for normal calling convention lowering is essentially
  // useless for kernels. We want to avoid any kind of legalization or argument
  // splitting.
  if (CC == CallingConv::AMDGPU_KERNEL)
    return lowerFormalArgumentsKernel(B, F, VRegs);

  const bool IsGraphics = AMDGPU::isGraphics(CC);
  const bool IsEntryFunc = AMDGPU::isEntryFunctionCC(CC);

  MachineFunction &MF = B.getMF();
  MachineBasicBlock &MBB = B.getMBB();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &Subtarget = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const DataLayout &DL = F.getParent()->getDataLayout();


  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, F.isVarArg(), MF, ArgLocs, F.getContext());

  if (!IsEntryFunc) {
    Register ReturnAddrReg = TRI->getReturnAddressReg(MF);
    Register LiveInReturn = MF.addLiveIn(ReturnAddrReg,
                                         &AMDGPU::SGPR_64RegClass);
    MBB.addLiveIn(ReturnAddrReg);
    B.buildCopy(LiveInReturn, ReturnAddrReg);
  }

  if (Info->hasImplicitBufferPtr()) {
    Register ImplicitBufferPtrReg = Info->addImplicitBufferPtr(*TRI);
    MF.addLiveIn(ImplicitBufferPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(ImplicitBufferPtrReg);
  }

  SmallVector<ArgInfo, 32> SplitArgs;
  unsigned Idx = 0;
  unsigned PSInputNum = 0;

  // Insert the hidden sret parameter if the return value won't fit in the
  // return registers.
  if (!FLI.CanLowerReturn)
    insertSRetIncomingArgument(F, SplitArgs, FLI.DemoteRegister, MRI, DL);

  for (auto &Arg : F.args()) {
    if (DL.getTypeStoreSize(Arg.getType()) == 0)
      continue;

    const bool InReg = Arg.hasAttribute(Attribute::InReg);

    // SGPR arguments to functions not implemented.
    if (!IsGraphics && InReg)
      return false;

    if (Arg.hasAttribute(Attribute::SwiftSelf) ||
        Arg.hasAttribute(Attribute::SwiftError) ||
        Arg.hasAttribute(Attribute::Nest))
      return false;

    if (CC == CallingConv::AMDGPU_PS && !InReg && PSInputNum <= 15) {
      const bool ArgUsed = !Arg.use_empty();
      bool SkipArg = !ArgUsed && !Info->isPSInputAllocated(PSInputNum);

      if (!SkipArg) {
        Info->markPSInputAllocated(PSInputNum);
        if (ArgUsed)
          Info->markPSInputEnabled(PSInputNum);
      }

      ++PSInputNum;

      if (SkipArg) {
        for (int I = 0, E = VRegs[Idx].size(); I != E; ++I)
          B.buildUndef(VRegs[Idx][I]);

        ++Idx;
        continue;
      }
    }

    ArgInfo OrigArg(VRegs[Idx], Arg.getType());
    const unsigned OrigArgIdx = Idx + AttributeList::FirstArgIndex;
    setArgFlags(OrigArg, OrigArgIdx, DL, F);

    splitToValueTypes(OrigArg, SplitArgs, DL, CC);
    ++Idx;
  }

  // At least one interpolation mode must be enabled or else the GPU will
  // hang.
  //
  // Check PSInputAddr instead of PSInputEnable. The idea is that if the user
  // set PSInputAddr, the user wants to enable some bits after the compilation
  // based on run-time states. Since we can't know what the final PSInputEna
  // will look like, so we shouldn't do anything here and the user should take
  // responsibility for the correct programming.
  //
  // Otherwise, the following restrictions apply:
  // - At least one of PERSP_* (0xF) or LINEAR_* (0x70) must be enabled.
  // - If POS_W_FLOAT (11) is enabled, at least one of PERSP_* must be
  //   enabled too.
  if (CC == CallingConv::AMDGPU_PS) {
    if ((Info->getPSInputAddr() & 0x7F) == 0 ||
        ((Info->getPSInputAddr() & 0xF) == 0 &&
         Info->isPSInputAllocated(11))) {
      CCInfo.AllocateReg(AMDGPU::VGPR0);
      CCInfo.AllocateReg(AMDGPU::VGPR1);
      Info->markPSInputAllocated(0);
      Info->markPSInputEnabled(0);
    }

    if (Subtarget.isAmdPalOS()) {
      // For isAmdPalOS, the user does not enable some bits after compilation
      // based on run-time states; the register values being generated here are
      // the final ones set in hardware. Therefore we need to apply the
      // workaround to PSInputAddr and PSInputEnable together.  (The case where
      // a bit is set in PSInputAddr but not PSInputEnable is where the frontend
      // set up an input arg for a particular interpolation mode, but nothing
      // uses that input arg. Really we should have an earlier pass that removes
      // such an arg.)
      unsigned PsInputBits = Info->getPSInputAddr() & Info->getPSInputEnable();
      if ((PsInputBits & 0x7F) == 0 ||
          ((PsInputBits & 0xF) == 0 &&
           (PsInputBits >> 11 & 1)))
        Info->markPSInputEnabled(
          countTrailingZeros(Info->getPSInputAddr(), ZB_Undefined));
    }
  }

  const SITargetLowering &TLI = *getTLI<SITargetLowering>();
  CCAssignFn *AssignFn = TLI.CCAssignFnForCall(CC, F.isVarArg());

  if (!MBB.empty())
    B.setInstr(*MBB.begin());

  if (!IsEntryFunc) {
    // For the fixed ABI, pass workitem IDs in the last argument register.
    if (AMDGPUTargetMachine::EnableFixedFunctionABI)
      TLI.allocateSpecialInputVGPRsFixed(CCInfo, MF, *TRI, *Info);
  }

  FormalArgHandler Handler(B, MRI, AssignFn);
  if (!handleAssignments(CCInfo, ArgLocs, B, SplitArgs, Handler))
    return false;

  if (!IsEntryFunc && !AMDGPUTargetMachine::EnableFixedFunctionABI) {
    // Special inputs come after user arguments.
    TLI.allocateSpecialInputVGPRs(CCInfo, MF, *TRI, *Info);
  }

  // Start adding system SGPRs.
  if (IsEntryFunc) {
    TLI.allocateSystemSGPRs(CCInfo, MF, *Info, CC, IsGraphics);
  } else {
    if (!Subtarget.enableFlatScratch())
      CCInfo.AllocateReg(Info->getScratchRSrcReg());
    TLI.allocateSpecialInputSGPRs(CCInfo, MF, *TRI, *Info);
  }

  // Move back to the end of the basic block.
  B.setMBB(MBB);

  return true;
}

bool AMDGPUCallLowering::passSpecialInputs(MachineIRBuilder &MIRBuilder,
                                           CCState &CCInfo,
                                           SmallVectorImpl<std::pair<MCRegister, Register>> &ArgRegs,
                                           CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();

  const AMDGPUFunctionArgInfo *CalleeArgInfo
    = &AMDGPUArgumentUsageInfo::FixedABIFunctionInfo;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const AMDGPUFunctionArgInfo &CallerArgInfo = MFI->getArgInfo();


  // TODO: Unify with private memory register handling. This is complicated by
  // the fact that at least in kernels, the input argument is not necessarily
  // in the same location as the input.
  AMDGPUFunctionArgInfo::PreloadedValue InputRegs[] = {
    AMDGPUFunctionArgInfo::DISPATCH_PTR,
    AMDGPUFunctionArgInfo::QUEUE_PTR,
    AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR,
    AMDGPUFunctionArgInfo::DISPATCH_ID,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_X,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_Y,
    AMDGPUFunctionArgInfo::WORKGROUP_ID_Z
  };

  MachineRegisterInfo &MRI = MF.getRegInfo();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const AMDGPULegalizerInfo *LI
    = static_cast<const AMDGPULegalizerInfo*>(ST.getLegalizerInfo());

  for (auto InputID : InputRegs) {
    const ArgDescriptor *OutgoingArg;
    const TargetRegisterClass *ArgRC;
    LLT ArgTy;

    std::tie(OutgoingArg, ArgRC, ArgTy) =
        CalleeArgInfo->getPreloadedValue(InputID);
    if (!OutgoingArg)
      continue;

    const ArgDescriptor *IncomingArg;
    const TargetRegisterClass *IncomingArgRC;
    std::tie(IncomingArg, IncomingArgRC, ArgTy) =
        CallerArgInfo.getPreloadedValue(InputID);
    assert(IncomingArgRC == ArgRC);

    Register InputReg = MRI.createGenericVirtualRegister(ArgTy);

    if (IncomingArg) {
      LI->loadInputValue(InputReg, MIRBuilder, IncomingArg, ArgRC, ArgTy);
    } else {
      assert(InputID == AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR);
      LI->getImplicitArgPtr(InputReg, MRI, MIRBuilder);
    }

    if (OutgoingArg->isRegister()) {
      ArgRegs.emplace_back(OutgoingArg->getRegister(), InputReg);
      if (!CCInfo.AllocateReg(OutgoingArg->getRegister()))
        report_fatal_error("failed to allocate implicit input argument");
    } else {
      LLVM_DEBUG(dbgs() << "Unhandled stack passed implicit input argument\n");
      return false;
    }
  }

  // Pack workitem IDs into a single register or pass it as is if already
  // packed.
  const ArgDescriptor *OutgoingArg;
  const TargetRegisterClass *ArgRC;
  LLT ArgTy;

  std::tie(OutgoingArg, ArgRC, ArgTy) =
      CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_X);
  if (!OutgoingArg)
    std::tie(OutgoingArg, ArgRC, ArgTy) =
        CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Y);
  if (!OutgoingArg)
    std::tie(OutgoingArg, ArgRC, ArgTy) =
        CalleeArgInfo->getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Z);
  if (!OutgoingArg)
    return false;

  auto WorkitemIDX =
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_X);
  auto WorkitemIDY =
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Y);
  auto WorkitemIDZ =
      CallerArgInfo.getPreloadedValue(AMDGPUFunctionArgInfo::WORKITEM_ID_Z);

  const ArgDescriptor *IncomingArgX = std::get<0>(WorkitemIDX);
  const ArgDescriptor *IncomingArgY = std::get<0>(WorkitemIDY);
  const ArgDescriptor *IncomingArgZ = std::get<0>(WorkitemIDZ);
  const LLT S32 = LLT::scalar(32);

  // If incoming ids are not packed we need to pack them.
  // FIXME: Should consider known workgroup size to eliminate known 0 cases.
  Register InputReg;
  if (IncomingArgX && !IncomingArgX->isMasked() && CalleeArgInfo->WorkItemIDX) {
    InputReg = MRI.createGenericVirtualRegister(S32);
    LI->loadInputValue(InputReg, MIRBuilder, IncomingArgX,
                       std::get<1>(WorkitemIDX), std::get<2>(WorkitemIDX));
  }

  if (IncomingArgY && !IncomingArgY->isMasked() && CalleeArgInfo->WorkItemIDY) {
    Register Y = MRI.createGenericVirtualRegister(S32);
    LI->loadInputValue(Y, MIRBuilder, IncomingArgY, std::get<1>(WorkitemIDY),
                       std::get<2>(WorkitemIDY));

    Y = MIRBuilder.buildShl(S32, Y, MIRBuilder.buildConstant(S32, 10)).getReg(0);
    InputReg = InputReg ? MIRBuilder.buildOr(S32, InputReg, Y).getReg(0) : Y;
  }

  if (IncomingArgZ && !IncomingArgZ->isMasked() && CalleeArgInfo->WorkItemIDZ) {
    Register Z = MRI.createGenericVirtualRegister(S32);
    LI->loadInputValue(Z, MIRBuilder, IncomingArgZ, std::get<1>(WorkitemIDZ),
                       std::get<2>(WorkitemIDZ));

    Z = MIRBuilder.buildShl(S32, Z, MIRBuilder.buildConstant(S32, 20)).getReg(0);
    InputReg = InputReg ? MIRBuilder.buildOr(S32, InputReg, Z).getReg(0) : Z;
  }

  if (!InputReg) {
    InputReg = MRI.createGenericVirtualRegister(S32);

    // Workitem ids are already packed, any of present incoming arguments will
    // carry all required fields.
    ArgDescriptor IncomingArg = ArgDescriptor::createArg(
      IncomingArgX ? *IncomingArgX :
        IncomingArgY ? *IncomingArgY : *IncomingArgZ, ~0u);
    LI->loadInputValue(InputReg, MIRBuilder, &IncomingArg,
                       &AMDGPU::VGPR_32RegClass, S32);
  }

  if (OutgoingArg->isRegister()) {
    ArgRegs.emplace_back(OutgoingArg->getRegister(), InputReg);
    if (!CCInfo.AllocateReg(OutgoingArg->getRegister()))
      report_fatal_error("failed to allocate implicit input argument");
  } else {
    LLVM_DEBUG(dbgs() << "Unhandled stack passed implicit input argument\n");
    return false;
  }

  return true;
}

/// Returns a pair containing the fixed CCAssignFn and the vararg CCAssignFn for
/// CC.
static std::pair<CCAssignFn *, CCAssignFn *>
getAssignFnsForCC(CallingConv::ID CC, const SITargetLowering &TLI) {
  return {TLI.CCAssignFnForCall(CC, false), TLI.CCAssignFnForCall(CC, true)};
}

static unsigned getCallOpcode(const MachineFunction &CallerF, bool IsIndirect,
                              bool IsTailCall) {
  return AMDGPU::SI_CALL;
}

// Add operands to call instruction to track the callee.
static bool addCallTargetOperands(MachineInstrBuilder &CallInst,
                                  MachineIRBuilder &MIRBuilder,
                                  AMDGPUCallLowering::CallLoweringInfo &Info) {
  if (Info.Callee.isReg()) {
    CallInst.addReg(Info.Callee.getReg());
    CallInst.addImm(0);
  } else if (Info.Callee.isGlobal() && Info.Callee.getOffset() == 0) {
    // The call lowering lightly assumed we can directly encode a call target in
    // the instruction, which is not the case. Materialize the address here.
    const GlobalValue *GV = Info.Callee.getGlobal();
    auto Ptr = MIRBuilder.buildGlobalValue(
      LLT::pointer(GV->getAddressSpace(), 64), GV);
    CallInst.addReg(Ptr.getReg(0));
    CallInst.add(Info.Callee);
  } else
    return false;

  return true;
}

bool AMDGPUCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                   CallLoweringInfo &Info) const {
  if (Info.IsVarArg) {
    LLVM_DEBUG(dbgs() << "Variadic functions not implemented\n");
    return false;
  }

  MachineFunction &MF = MIRBuilder.getMF();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();
  const DataLayout &DL = F.getParent()->getDataLayout();
  CallingConv::ID CallConv = F.getCallingConv();

  if (!AMDGPUTargetMachine::EnableFixedFunctionABI &&
      CallConv != CallingConv::AMDGPU_Gfx) {
    LLVM_DEBUG(dbgs() << "Variable function ABI not implemented\n");
    return false;
  }

  if (AMDGPU::isShader(CallConv)) {
    LLVM_DEBUG(dbgs() << "Unhandled call from graphics shader\n");
    return false;
  }

  SmallVector<ArgInfo, 8> OutArgs;
  for (auto &OrigArg : Info.OrigArgs)
    splitToValueTypes(OrigArg, OutArgs, DL, Info.CallConv);

  // If we can lower as a tail call, do that instead.
  bool CanTailCallOpt = false;

  // We must emit a tail call if we have musttail.
  if (Info.IsMustTailCall && !CanTailCallOpt) {
    LLVM_DEBUG(dbgs() << "Failed to lower musttail call as tail call\n");
    return false;
  }

  // Find out which ABI gets to decide where things go.
  CCAssignFn *AssignFnFixed;
  CCAssignFn *AssignFnVarArg;
  std::tie(AssignFnFixed, AssignFnVarArg) =
      getAssignFnsForCC(Info.CallConv, TLI);

  MIRBuilder.buildInstr(AMDGPU::ADJCALLSTACKUP)
    .addImm(0)
    .addImm(0);

  // Create a temporarily-floating call instruction so we can add the implicit
  // uses of arg registers.
  unsigned Opc = getCallOpcode(MF, Info.Callee.isReg(), false);

  auto MIB = MIRBuilder.buildInstrNoInsert(Opc);
  MIB.addDef(TRI->getReturnAddressReg(MF));

  if (!addCallTargetOperands(MIB, MIRBuilder, Info))
    return false;

  // Tell the call which registers are clobbered.
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, Info.CallConv);
  MIB.addRegMask(Mask);

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(Info.CallConv, Info.IsVarArg, MF, ArgLocs, F.getContext());

  // We could pass MIB and directly add the implicit uses to the call
  // now. However, as an aesthetic choice, place implicit argument operands
  // after the ordinary user argument registers.
  SmallVector<std::pair<MCRegister, Register>, 12> ImplicitArgRegs;

  if (AMDGPUTargetMachine::EnableFixedFunctionABI &&
      Info.CallConv != CallingConv::AMDGPU_Gfx) {
    // With a fixed ABI, allocate fixed registers before user arguments.
    if (!passSpecialInputs(MIRBuilder, CCInfo, ImplicitArgRegs, Info))
      return false;
  }

  // Do the actual argument marshalling.
  SmallVector<Register, 8> PhysRegs;
  AMDGPUOutgoingArgHandler Handler(MIRBuilder, MRI, MIB, AssignFnFixed,
                                   AssignFnVarArg, false);
  if (!handleAssignments(CCInfo, ArgLocs, MIRBuilder, OutArgs, Handler))
    return false;

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  if (!ST.enableFlatScratch()) {
    // Insert copies for the SRD. In the HSA case, this should be an identity
    // copy.
    auto ScratchRSrcReg = MIRBuilder.buildCopy(LLT::vector(4, 32),
                                               MFI->getScratchRSrcReg());
    MIRBuilder.buildCopy(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, ScratchRSrcReg);
    MIB.addReg(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, RegState::Implicit);
  }

  for (std::pair<MCRegister, Register> ArgReg : ImplicitArgRegs) {
    MIRBuilder.buildCopy((Register)ArgReg.first, ArgReg.second);
    MIB.addReg(ArgReg.first, RegState::Implicit);
  }

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  // If Callee is a reg, since it is used by a target specific
  // instruction, it must have a register class matching the
  // constraint of that instruction.

  // FIXME: We should define regbankselectable call instructions to handle
  // divergent call targets.
  if (MIB->getOperand(1).isReg()) {
    MIB->getOperand(1).setReg(constrainOperandRegClass(
        MF, *TRI, MRI, *ST.getInstrInfo(),
        *ST.getRegBankInfo(), *MIB, MIB->getDesc(), MIB->getOperand(1),
        1));
  }

  auto OrigInsertPt = MIRBuilder.getInsertPt();

  // Now we can add the actual call instruction to the correct position.
  MIRBuilder.insertInstr(MIB);

  // Insert this now to give us an anchor point for managing the insert point.
  MachineInstrBuilder CallSeqEnd =
    MIRBuilder.buildInstr(AMDGPU::ADJCALLSTACKDOWN);

  SmallVector<ArgInfo, 8> InArgs;
  if (!Info.CanLowerReturn) {
    insertSRetLoads(MIRBuilder, Info.OrigRet.Ty, Info.OrigRet.Regs,
                    Info.DemoteRegister, Info.DemoteStackIndex);
  } else if (!Info.OrigRet.Ty->isVoidTy()) {
    splitToValueTypes(Info.OrigRet, InArgs, DL, Info.CallConv);
  }

  // Make sure the raw argument copies are inserted before the marshalling to
  // the original types.
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), CallSeqEnd);

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arguments, the physical register must be an
  // implicit-define of the call instruction.
  if (Info.CanLowerReturn && !Info.OrigRet.Ty->isVoidTy()) {
    CCAssignFn *RetAssignFn = TLI.CCAssignFnForReturn(Info.CallConv,
                                                      Info.IsVarArg);
    CallReturnHandler Handler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, InArgs, Handler, Info.CallConv,
                           Info.IsVarArg))
      return false;
  }

  uint64_t CalleePopBytes = NumBytes;
  CallSeqEnd.addImm(0)
            .addImm(CalleePopBytes);

  // Restore the insert point to after the call sequence.
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), OrigInsertPt);
  return true;
}
