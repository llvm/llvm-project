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
#include "AMDGPUISelLowering.h"
#include "AMDGPULegalizerInfo.h"
#include "AMDGPUSubtarget.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIISelLowering.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/LowLevelTypeImpl.h"

#define DEBUG_TYPE "amdgpu-call-lowering"

using namespace llvm;

namespace {

struct AMDGPUValueHandler : public CallLowering::ValueHandler {
  AMDGPUValueHandler(bool IsIncoming, MachineIRBuilder &B,
                     MachineRegisterInfo &MRI, CCAssignFn *AssignFn)
      : ValueHandler(IsIncoming, B, MRI, AssignFn) {}

  /// Wrapper around extendRegister to ensure we extend to a full 32-bit
  /// register.
  Register extendRegisterMin32(Register ValVReg, CCValAssign &VA) {
    if (VA.getLocVT().getSizeInBits() < 32) {
      // 16-bit types are reported as legal for 32-bit registers. We need to
      // extend and do a 32-bit copy to avoid the verifier complaining about it.
      return MIRBuilder.buildAnyExt(LLT::scalar(32), ValVReg).getReg(0);
    }

    return extendRegister(ValVReg, VA);
  }
};

struct AMDGPUOutgoingValueHandler : public AMDGPUValueHandler {
  AMDGPUOutgoingValueHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                             MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : AMDGPUValueHandler(false, B, MRI, AssignFn), MIB(MIB) {}

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
    Register ExtReg = extendRegisterMin32(ValVReg, VA);

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

struct AMDGPUIncomingArgHandler : public AMDGPUValueHandler {
  uint64_t StackUsed = 0;

  AMDGPUIncomingArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                           CCAssignFn *AssignFn)
      : AMDGPUValueHandler(true, B, MRI, AssignFn) {}

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
      MIRBuilder.buildTrunc(ValVReg, Copy);
      return;
    }

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

struct AMDGPUOutgoingArgHandler : public AMDGPUValueHandler {
  MachineInstrBuilder MIB;
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
      : AMDGPUValueHandler(false, MIRBuilder, MRI, AssignFn), MIB(MIB),
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
    Register ExtReg = extendRegisterMin32(ValVReg, VA);
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

  void assignValueToAddress(const CallLowering::ArgInfo &Arg, Register Addr,
                            uint64_t Size, MachinePointerInfo &MPO,
                            CCValAssign &VA) override {
    Register ValVReg = VA.getLocInfo() != CCValAssign::LocInfo::FPExt
                           ? extendRegister(Arg.Regs[0], VA)
                           : Arg.Regs[0];

    // If we extended we might need to adjust the MMO's Size.
    const LLT RegTy = MRI.getType(ValVReg);
    if (RegTy.getSizeInBytes() > Size)
      Size = RegTy.getSizeInBytes();

    assignValueToAddress(ValVReg, Addr, Size, MPO, VA);
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

void AMDGPUCallLowering::splitToValueTypes(
  MachineIRBuilder &B,
  const ArgInfo &OrigArg,
  SmallVectorImpl<ArgInfo> &SplitArgs,
  const DataLayout &DL, CallingConv::ID CallConv,
  bool IsOutgoing,
  SplitArgTy PerformArgSplit) const {
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();
  LLVMContext &Ctx = OrigArg.Ty->getContext();

  if (OrigArg.Ty->isVoidTy())
    return;

  SmallVector<EVT, 4> SplitVTs;
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs);

  assert(OrigArg.Regs.size() == SplitVTs.size());

  int SplitIdx = 0;
  for (EVT VT : SplitVTs) {
    Register Reg = OrigArg.Regs[SplitIdx];
    Type *Ty = VT.getTypeForEVT(Ctx);
    LLT LLTy = getLLTForType(*Ty, DL);

    if (IsOutgoing && VT.isScalarInteger()) {
      unsigned ExtendOp = TargetOpcode::G_ANYEXT;
      if (OrigArg.Flags[0].isSExt()) {
        assert(OrigArg.Regs.size() == 1 && "expect only simple return values");
        ExtendOp = TargetOpcode::G_SEXT;
      } else if (OrigArg.Flags[0].isZExt()) {
        assert(OrigArg.Regs.size() == 1 && "expect only simple return values");
        ExtendOp = TargetOpcode::G_ZEXT;
      }

      EVT ExtVT = TLI.getTypeForExtReturn(Ctx, VT,
                                          extOpcodeToISDExtOpcode(ExtendOp));
      if (ExtVT.getSizeInBits() != VT.getSizeInBits()) {
        VT = ExtVT;
        Ty = ExtVT.getTypeForEVT(Ctx);
        LLTy = getLLTForType(*Ty, DL);
        Reg = B.buildInstr(ExtendOp, {LLTy}, {Reg}).getReg(0);
      }
    }

    unsigned NumParts = TLI.getNumRegistersForCallingConv(Ctx, CallConv, VT);
    MVT RegVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, VT);

    if (NumParts == 1) {
      // No splitting to do, but we want to replace the original type (e.g. [1 x
      // double] -> double).
      SplitArgs.emplace_back(Reg, Ty, OrigArg.Flags, OrigArg.IsFixed);

      ++SplitIdx;
      continue;
    }

    SmallVector<Register, 8> SplitRegs;
    Type *PartTy = EVT(RegVT).getTypeForEVT(Ctx);
    LLT PartLLT = getLLTForType(*PartTy, DL);
    MachineRegisterInfo &MRI = *B.getMRI();

    // FIXME: Should we be reporting all of the part registers for a single
    // argument, and let handleAssignments take care of the repacking?
    for (unsigned i = 0; i < NumParts; ++i) {
      Register PartReg = MRI.createGenericVirtualRegister(PartLLT);
      SplitRegs.push_back(PartReg);
      SplitArgs.emplace_back(ArrayRef<Register>(PartReg), PartTy, OrigArg.Flags);
    }

    PerformArgSplit(SplitRegs, Reg, LLTy, PartLLT, SplitIdx);

    ++SplitIdx;
  }
}

// TODO: Move to generic code
static void unpackRegsToOrigType(MachineIRBuilder &B,
                                 ArrayRef<Register> DstRegs,
                                 Register SrcReg,
                                 const CallLowering::ArgInfo &Info,
                                 LLT SrcTy,
                                 LLT PartTy) {
  assert(DstRegs.size() > 1 && "Nothing to unpack");

  const unsigned PartSize = PartTy.getSizeInBits();

  if (SrcTy.isVector() && !PartTy.isVector() &&
      PartSize > SrcTy.getElementType().getSizeInBits()) {
    // Vector was scalarized, and the elements extended.
    auto UnmergeToEltTy = B.buildUnmerge(SrcTy.getElementType(), SrcReg);
    for (int i = 0, e = DstRegs.size(); i != e; ++i)
      B.buildAnyExt(DstRegs[i], UnmergeToEltTy.getReg(i));
    return;
  }

  LLT GCDTy = getGCDType(SrcTy, PartTy);
  if (GCDTy == PartTy) {
    // If this already evenly divisible, we can create a simple unmerge.
    B.buildUnmerge(DstRegs, SrcReg);
    return;
  }

  MachineRegisterInfo &MRI = *B.getMRI();
  LLT DstTy = MRI.getType(DstRegs[0]);
  LLT LCMTy = getLCMType(SrcTy, PartTy);

  const unsigned LCMSize = LCMTy.getSizeInBits();
  const unsigned DstSize = DstTy.getSizeInBits();
  const unsigned SrcSize = SrcTy.getSizeInBits();

  Register UnmergeSrc = SrcReg;
  if (LCMSize != SrcSize) {
    // Widen to the common type.
    Register Undef = B.buildUndef(SrcTy).getReg(0);
    SmallVector<Register, 8> MergeParts(1, SrcReg);
    for (unsigned Size = SrcSize; Size != LCMSize; Size += SrcSize)
      MergeParts.push_back(Undef);

    UnmergeSrc = B.buildMerge(LCMTy, MergeParts).getReg(0);
  }

  // Unmerge to the original registers and pad with dead defs.
  SmallVector<Register, 8> UnmergeResults(DstRegs.begin(), DstRegs.end());
  for (unsigned Size = DstSize * DstRegs.size(); Size != LCMSize;
       Size += DstSize) {
    UnmergeResults.push_back(MRI.createGenericVirtualRegister(DstTy));
  }

  B.buildUnmerge(UnmergeResults, UnmergeSrc);
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

  CallingConv::ID CC = F.getCallingConv();
  const SITargetLowering &TLI = *getTLI<SITargetLowering>();

  ArgInfo OrigRetInfo(VRegs, Val->getType());
  setArgFlags(OrigRetInfo, AttributeList::ReturnIndex, DL, F);
  SmallVector<ArgInfo, 4> SplitRetInfos;

  splitToValueTypes(
    B, OrigRetInfo, SplitRetInfos, DL, CC, true,
    [&](ArrayRef<Register> Regs, Register SrcReg, LLT LLTy, LLT PartLLT,
        int VTSplitIdx) {
      unpackRegsToOrigType(B, Regs, SrcReg,
                           SplitRetInfos[VTSplitIdx],
                           LLTy, PartLLT);
    });

  CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(CC, F.isVarArg());
  AMDGPUOutgoingValueHandler RetHandler(B, *MRI, Ret, AssignFn);
  return handleAssignments(B, SplitRetInfos, RetHandler);
}

bool AMDGPUCallLowering::lowerReturn(MachineIRBuilder &B,
                                     const Value *Val,
                                     ArrayRef<Register> VRegs) const {

  MachineFunction &MF = B.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MFI->setIfReturnsVoid(!Val);

  assert(!Val == VRegs.empty() && "Return value without a vreg");

  CallingConv::ID CC = B.getMF().getFunction().getCallingConv();
  const bool IsShader = AMDGPU::isShader(CC);
  const bool IsWaveEnd = (IsShader && MFI->returnsVoid()) ||
                         AMDGPU::isKernel(CC);
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

  if (!lowerReturnVal(B, Val, VRegs, Ret))
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

/// Pack values \p SrcRegs to cover the vector type result \p DstRegs.
static MachineInstrBuilder mergeVectorRegsToResultRegs(
  MachineIRBuilder &B, ArrayRef<Register> DstRegs, ArrayRef<Register> SrcRegs) {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT LLTy = MRI.getType(DstRegs[0]);
  LLT PartLLT = MRI.getType(SrcRegs[0]);

  // Deal with v3s16 split into v2s16
  LLT LCMTy = getLCMType(LLTy, PartLLT);
  if (LCMTy == LLTy) {
    // Common case where no padding is needed.
    assert(DstRegs.size() == 1);
    return B.buildConcatVectors(DstRegs[0], SrcRegs);
  }

  const int NumWide =  LCMTy.getSizeInBits() / PartLLT.getSizeInBits();
  Register Undef = B.buildUndef(PartLLT).getReg(0);

  // Build vector of undefs.
  SmallVector<Register, 8> WidenedSrcs(NumWide, Undef);

  // Replace the first sources with the real registers.
  std::copy(SrcRegs.begin(), SrcRegs.end(), WidenedSrcs.begin());

  auto Widened = B.buildConcatVectors(LCMTy, WidenedSrcs);
  int NumDst = LCMTy.getSizeInBits() / LLTy.getSizeInBits();

  SmallVector<Register, 8> PadDstRegs(NumDst);
  std::copy(DstRegs.begin(), DstRegs.end(), PadDstRegs.begin());

  // Create the excess dead defs for the unmerge.
  for (int I = DstRegs.size(); I != NumDst; ++I)
    PadDstRegs[I] = MRI.createGenericVirtualRegister(LLTy);

  return B.buildUnmerge(PadDstRegs, Widened);
}

// TODO: Move this to generic code
static void packSplitRegsToOrigType(MachineIRBuilder &B,
                                    ArrayRef<Register> OrigRegs,
                                    ArrayRef<Register> Regs,
                                    LLT LLTy,
                                    LLT PartLLT) {
  MachineRegisterInfo &MRI = *B.getMRI();

  if (!LLTy.isVector() && !PartLLT.isVector()) {
    assert(OrigRegs.size() == 1);
    LLT OrigTy = MRI.getType(OrigRegs[0]);

    unsigned SrcSize = PartLLT.getSizeInBits() * Regs.size();
    if (SrcSize == OrigTy.getSizeInBits())
      B.buildMerge(OrigRegs[0], Regs);
    else {
      auto Widened = B.buildMerge(LLT::scalar(SrcSize), Regs);
      B.buildTrunc(OrigRegs[0], Widened);
    }

    return;
  }

  if (LLTy.isVector() && PartLLT.isVector()) {
    assert(OrigRegs.size() == 1);
    assert(LLTy.getElementType() == PartLLT.getElementType());
    mergeVectorRegsToResultRegs(B, OrigRegs, Regs);
    return;
  }

  assert(LLTy.isVector() && !PartLLT.isVector());

  LLT DstEltTy = LLTy.getElementType();

  // Pointer information was discarded. We'll need to coerce some register types
  // to avoid violating type constraints.
  LLT RealDstEltTy = MRI.getType(OrigRegs[0]).getElementType();

  assert(DstEltTy.getSizeInBits() == RealDstEltTy.getSizeInBits());

  if (DstEltTy == PartLLT) {
    // Vector was trivially scalarized.

    if (RealDstEltTy.isPointer()) {
      for (Register Reg : Regs)
        MRI.setType(Reg, RealDstEltTy);
    }

    B.buildBuildVector(OrigRegs[0], Regs);
  } else if (DstEltTy.getSizeInBits() > PartLLT.getSizeInBits()) {
    // Deal with vector with 64-bit elements decomposed to 32-bit
    // registers. Need to create intermediate 64-bit elements.
    SmallVector<Register, 8> EltMerges;
    int PartsPerElt = DstEltTy.getSizeInBits() / PartLLT.getSizeInBits();

    assert(DstEltTy.getSizeInBits() % PartLLT.getSizeInBits() == 0);

    for (int I = 0, NumElts = LLTy.getNumElements(); I != NumElts; ++I)  {
      auto Merge = B.buildMerge(RealDstEltTy, Regs.take_front(PartsPerElt));
      // Fix the type in case this is really a vector of pointers.
      MRI.setType(Merge.getReg(0), RealDstEltTy);
      EltMerges.push_back(Merge.getReg(0));
      Regs = Regs.drop_front(PartsPerElt);
    }

    B.buildBuildVector(OrigRegs[0], EltMerges);
  } else {
    // Vector was split, and elements promoted to a wider type.
    LLT BVType = LLT::vector(LLTy.getNumElements(), PartLLT);
    auto BV = B.buildBuildVector(BVType, Regs);
    B.buildTrunc(OrigRegs[0], BV);
  }
}

bool AMDGPUCallLowering::lowerFormalArguments(
    MachineIRBuilder &B, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  CallingConv::ID CC = F.getCallingConv();

  // The infrastructure for normal calling convention lowering is essentially
  // useless for kernels. We want to avoid any kind of legalization or argument
  // splitting.
  if (CC == CallingConv::AMDGPU_KERNEL)
    return lowerFormalArgumentsKernel(B, F, VRegs);

  const bool IsShader = AMDGPU::isShader(CC);
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

  for (auto &Arg : F.args()) {
    if (DL.getTypeStoreSize(Arg.getType()) == 0)
      continue;

    const bool InReg = Arg.hasAttribute(Attribute::InReg);

    // SGPR arguments to functions not implemented.
    if (!IsShader && InReg)
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

    splitToValueTypes(
      B, OrigArg, SplitArgs, DL, CC, false,
      // FIXME: We should probably be passing multiple registers to
      // handleAssignments to do this
      [&](ArrayRef<Register> Regs, Register DstReg,
          LLT LLTy, LLT PartLLT, int VTSplitIdx) {
        assert(DstReg == VRegs[Idx][VTSplitIdx]);
        packSplitRegsToOrigType(B, VRegs[Idx][VTSplitIdx], Regs,
                                LLTy, PartLLT);
      });

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
    TLI.allocateSystemSGPRs(CCInfo, MF, *Info, CC, IsShader);
  } else {
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
  if (!AMDGPUTargetMachine::EnableFixedFunctionABI) {
    LLVM_DEBUG(dbgs() << "Variable function ABI not implemented\n");
    return false;
  }

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

  if (AMDGPU::isShader(F.getCallingConv())) {
    LLVM_DEBUG(dbgs() << "Unhandled call from graphics shader\n");
    return false;
  }

  SmallVector<ArgInfo, 8> OutArgs;
  SmallVector<ArgInfo, 4> SplitRetInfos;

  for (auto &OrigArg : Info.OrigArgs) {
    splitToValueTypes(
      MIRBuilder, OrigArg, OutArgs, DL, Info.CallConv, true,
      // FIXME: We should probably be passing multiple registers to
      // handleAssignments to do this
      [&](ArrayRef<Register> Regs, Register SrcReg, LLT LLTy, LLT PartLLT,
          int VTSplitIdx) {
        unpackRegsToOrigType(MIRBuilder, Regs, SrcReg, OrigArg, LLTy, PartLLT);
      });
  }

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

  if (AMDGPUTargetMachine::EnableFixedFunctionABI) {
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

  // Insert copies for the SRD. In the HSA case, this should be an identity
  // copy.
  auto ScratchRSrcReg = MIRBuilder.buildCopy(LLT::vector(4, 32),
                                             MFI->getScratchRSrcReg());
  MIRBuilder.buildCopy(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, ScratchRSrcReg);
  MIB.addReg(AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3, RegState::Implicit);

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
  if (!Info.OrigRet.Ty->isVoidTy()) {
    splitToValueTypes(
      MIRBuilder, Info.OrigRet, InArgs, DL, Info.CallConv, false,
      [&](ArrayRef<Register> Regs, Register DstReg,
          LLT LLTy, LLT PartLLT, int VTSplitIdx) {
        assert(DstReg == Info.OrigRet.Regs[VTSplitIdx]);
        packSplitRegsToOrigType(MIRBuilder,  Info.OrigRet.Regs[VTSplitIdx],
                                Regs, LLTy, PartLLT);
      });
  }

  // Make sure the raw argument copies are inserted before the marshalling to
  // the original types.
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), CallSeqEnd);

  // Finally we can copy the returned value back into its virtual-register. In
  // symmetry with the arguments, the physical register must be an
  // implicit-define of the call instruction.
  if (!Info.OrigRet.Ty->isVoidTy()) {
    CCAssignFn *RetAssignFn = TLI.CCAssignFnForReturn(Info.CallConv,
                                                      Info.IsVarArg);
    CallReturnHandler Handler(MIRBuilder, MRI, MIB, RetAssignFn);
    if (!handleAssignments(MIRBuilder, InArgs, Handler))
      return false;
  }

  uint64_t CalleePopBytes = NumBytes;
  CallSeqEnd.addImm(0)
            .addImm(CalleePopBytes);

  // Restore the insert point to after the call sequence.
  MIRBuilder.setInsertPt(MIRBuilder.getMBB(), OrigInsertPt);
  return true;
}
