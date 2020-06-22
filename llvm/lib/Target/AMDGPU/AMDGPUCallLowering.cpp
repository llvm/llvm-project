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
#include "AMDGPUSubtarget.h"
#include "SIISelLowering.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/LowLevelTypeImpl.h"

using namespace llvm;

namespace {

struct OutgoingValueHandler : public CallLowering::ValueHandler {
  OutgoingValueHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                       MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : ValueHandler(B, MRI, AssignFn), MIB(MIB) {}

  MachineInstrBuilder MIB;

  bool isIncomingArgumentHandler() const override { return false; }

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
    Register ExtReg;
    if (VA.getLocVT().getSizeInBits() < 32) {
      // 16-bit types are reported as legal for 32-bit registers. We need to
      // extend and do a 32-bit copy to avoid the verifier complaining about it.
      ExtReg = MIRBuilder.buildAnyExt(LLT::scalar(32), ValVReg).getReg(0);
    } else
      ExtReg = extendRegister(ValVReg, VA);

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

struct IncomingArgHandler : public CallLowering::ValueHandler {
  uint64_t StackUsed = 0;

  IncomingArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                     CCAssignFn *AssignFn)
    : ValueHandler(B, MRI, AssignFn) {}

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

  void assignValueToAddress(Register ValVReg, Register Addr, uint64_t Size,
                            MachinePointerInfo &MPO, CCValAssign &VA) override {
    MachineFunction &MF = MIRBuilder.getMF();

    // FIXME: Get alignment
    auto MMO = MF.getMachineMemOperand(
        MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant, Size,
        inferAlignFromPtrInfo(MF, MPO));
    MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  /// How the physical register gets marked varies between formal
  /// parameters (it's a basic-block live-in), and a call instruction
  /// (it's an implicit-def of the BL).
  virtual void markPhysRegUsed(unsigned PhysReg) = 0;

  // FIXME: What is the point of this being a callback?
  bool isIncomingArgumentHandler() const override { return true; }
};

struct FormalArgHandler : public IncomingArgHandler {
  FormalArgHandler(MachineIRBuilder &B, MachineRegisterInfo &MRI,
                   CCAssignFn *AssignFn)
    : IncomingArgHandler(B, MRI, AssignFn) {}

  void markPhysRegUsed(unsigned PhysReg) override {
    MIRBuilder.getMBB().addLiveIn(PhysReg);
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
  const ArgInfo &OrigArg, unsigned OrigArgIdx,
  SmallVectorImpl<ArgInfo> &SplitArgs,
  const DataLayout &DL, CallingConv::ID CallConv,
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

    if (OrigArgIdx == AttributeList::ReturnIndex && VT.isScalarInteger()) {
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
      if (ExtVT != VT) {
        VT = ExtVT;
        Ty = ExtVT.getTypeForEVT(Ctx);
        LLTy = getLLTForType(*Ty, DL);
        Reg = B.buildInstr(ExtendOp, {LLTy}, {Reg}).getReg(0);
      }
    }

    unsigned NumParts = TLI.getNumRegistersForCallingConv(Ctx, CallConv, VT);
    MVT RegVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, VT);

    if (NumParts == 1) {
      // Fixup EVTs to an MVT.
      //
      // FIXME: This is pretty hacky. Why do we have to split the type
      // legalization logic between here and handleAssignments?
      if (OrigArgIdx != AttributeList::ReturnIndex && VT != RegVT) {
        assert(VT.getSizeInBits() < 32 &&
               "unexpected illegal type");
        Ty = Type::getInt32Ty(Ctx);
        Register OrigReg = Reg;
        Reg = B.getMRI()->createGenericVirtualRegister(LLT::scalar(32));
        B.buildTrunc(OrigReg, Reg);
      }

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

// Get the appropriate type to make \p OrigTy \p Factor times bigger.
static LLT getMultipleType(LLT OrigTy, int Factor) {
  if (OrigTy.isVector()) {
    return LLT::vector(OrigTy.getNumElements() * Factor,
                       OrigTy.getElementType());
  }

  return LLT::scalar(OrigTy.getSizeInBits() * Factor);
}

// TODO: Move to generic code
static void unpackRegsToOrigType(MachineIRBuilder &B,
                                 ArrayRef<Register> DstRegs,
                                 Register SrcReg,
                                 const CallLowering::ArgInfo &Info,
                                 LLT SrcTy,
                                 LLT PartTy) {
  assert(DstRegs.size() > 1 && "Nothing to unpack");

  const unsigned SrcSize = SrcTy.getSizeInBits();
  const unsigned PartSize = PartTy.getSizeInBits();

  if (SrcTy.isVector() && !PartTy.isVector() &&
      PartSize > SrcTy.getElementType().getSizeInBits()) {
    // Vector was scalarized, and the elements extended.
    auto UnmergeToEltTy = B.buildUnmerge(SrcTy.getElementType(),
                                                  SrcReg);
    for (int i = 0, e = DstRegs.size(); i != e; ++i)
      B.buildAnyExt(DstRegs[i], UnmergeToEltTy.getReg(i));
    return;
  }

  if (SrcSize % PartSize == 0) {
    B.buildUnmerge(DstRegs, SrcReg);
    return;
  }

  const int NumRoundedParts = (SrcSize + PartSize - 1) / PartSize;

  LLT BigTy = getMultipleType(PartTy, NumRoundedParts);
  auto ImpDef = B.buildUndef(BigTy);

  auto Big = B.buildInsert(BigTy, ImpDef.getReg(0), SrcReg, 0).getReg(0);

  int64_t Offset = 0;
  for (unsigned i = 0, e = DstRegs.size(); i != e; ++i, Offset += PartSize)
    B.buildExtract(DstRegs[i], Big, Offset);
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
    B, OrigRetInfo, AttributeList::ReturnIndex, SplitRetInfos, DL, CC,
    [&](ArrayRef<Register> Regs, Register SrcReg, LLT LLTy, LLT PartLLT,
        int VTSplitIdx) {
      unpackRegsToOrigType(B, Regs, SrcReg,
                           SplitRetInfos[VTSplitIdx],
                           LLTy, PartLLT);
    });

  CCAssignFn *AssignFn = TLI.CCAssignFnForReturn(CC, F.isVarArg());
  OutgoingValueHandler RetHandler(B, *MRI, Ret, AssignFn);
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

Register AMDGPUCallLowering::lowerParameterPtr(MachineIRBuilder &B,
                                               Type *ParamTy,
                                               uint64_t Offset) const {

  MachineFunction &MF = B.getMF();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PointerType *PtrTy = PointerType::get(ParamTy, AMDGPUAS::CONSTANT_ADDRESS);
  LLT PtrType = getLLTForType(*PtrTy, DL);
  Register KernArgSegmentPtr =
    MFI->getPreloadedReg(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  Register KernArgSegmentVReg = MRI.getLiveInVirtReg(KernArgSegmentPtr);

  auto OffsetReg = B.buildConstant(LLT::scalar(64), Offset);

  return B.buildPtrAdd(PtrType, KernArgSegmentVReg, OffsetReg).getReg(0);
}

void AMDGPUCallLowering::lowerParameter(MachineIRBuilder &B, Type *ParamTy,
                                        uint64_t Offset, Align Alignment,
                                        Register DstReg) const {
  MachineFunction &MF = B.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
  unsigned TypeSize = DL.getTypeStoreSize(ParamTy);
  Register PtrReg = lowerParameterPtr(B, ParamTy, Offset);

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
    Type *ArgTy = Arg.getType();
    unsigned AllocSize = DL.getTypeAllocSize(ArgTy);
    if (AllocSize == 0)
      continue;

    unsigned ABIAlign = DL.getABITypeAlignment(ArgTy);

    uint64_t ArgOffset = alignTo(ExplicitArgOffset, ABIAlign) + BaseOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, ABIAlign) + AllocSize;

    ArrayRef<Register> OrigArgRegs = VRegs[i];
    Register ArgReg =
      OrigArgRegs.size() == 1
      ? OrigArgRegs[0]
      : MRI.createGenericVirtualRegister(getLLTForType(*ArgTy, DL));
    Align Alignment = commonAlignment(KernArgBaseAlign, ArgOffset);
    ArgOffset = alignTo(ArgOffset, DL.getABITypeAlignment(ArgTy));
    lowerParameter(B, ArgTy, ArgOffset, Alignment, ArgReg);
    if (OrigArgRegs.size() > 1)
      unpackRegs(OrigArgRegs, ArgReg, ArgTy, B);
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
      B, OrigArg, OrigArgIdx, SplitArgs, DL, CC,
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

  FormalArgHandler Handler(B, MRI, AssignFn);
  if (!handleAssignments(CCInfo, ArgLocs, B, SplitArgs, Handler))
    return false;

  if (!IsEntryFunc) {
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
