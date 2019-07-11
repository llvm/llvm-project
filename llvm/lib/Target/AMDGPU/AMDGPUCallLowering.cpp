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

struct OutgoingArgHandler : public CallLowering::ValueHandler {
  OutgoingArgHandler(MachineIRBuilder &MIRBuilder, MachineRegisterInfo &MRI,
                     MachineInstrBuilder MIB, CCAssignFn *AssignFn)
      : ValueHandler(MIRBuilder, MRI, AssignFn), MIB(MIB) {}

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
    MIB.addUse(PhysReg);
    MIRBuilder.buildCopy(PhysReg, ValVReg);
  }

  bool assignArg(unsigned ValNo, MVT ValVT, MVT LocVT,
                 CCValAssign::LocInfo LocInfo,
                 const CallLowering::ArgInfo &Info,
                 CCState &State) override {
    return AssignFn(ValNo, ValVT, LocVT, LocInfo, Info.Flags, State);
  }
};

}

AMDGPUCallLowering::AMDGPUCallLowering(const AMDGPUTargetLowering &TLI)
  : CallLowering(&TLI) {
}

bool AMDGPUCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                     const Value *Val,
                                     ArrayRef<Register> VRegs) const {

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MFI->setIfReturnsVoid(!Val);

  if (!Val) {
    MIRBuilder.buildInstr(AMDGPU::S_ENDPGM).addImm(0);
    return true;
  }

  Register VReg = VRegs[0];

  const Function &F = MF.getFunction();
  auto &DL = F.getParent()->getDataLayout();
  if (!AMDGPU::isShader(F.getCallingConv()))
    return false;


  const AMDGPUTargetLowering &TLI = *getTLI<AMDGPUTargetLowering>();
  SmallVector<EVT, 4> SplitVTs;
  SmallVector<uint64_t, 4> Offsets;
  ArgInfo OrigArg{VReg, Val->getType()};
  setArgFlags(OrigArg, AttributeList::ReturnIndex, DL, F);
  ComputeValueVTs(TLI, DL, OrigArg.Ty, SplitVTs, &Offsets, 0);

  SmallVector<ArgInfo, 8> SplitArgs;
  CCAssignFn *AssignFn = CCAssignFnForReturn(F.getCallingConv(), false);
  for (unsigned i = 0, e = Offsets.size(); i != e; ++i) {
    Type *SplitTy = SplitVTs[i].getTypeForEVT(F.getContext());
    SplitArgs.push_back({VRegs[i], SplitTy, OrigArg.Flags, OrigArg.IsFixed});
  }
  auto RetInstr = MIRBuilder.buildInstrNoInsert(AMDGPU::SI_RETURN_TO_EPILOG);
  OutgoingArgHandler Handler(MIRBuilder, MRI, RetInstr, AssignFn);
  if (!handleAssignments(MIRBuilder, SplitArgs, Handler))
    return false;
  MIRBuilder.insertInstr(RetInstr);

  return true;
}

Register AMDGPUCallLowering::lowerParameterPtr(MachineIRBuilder &MIRBuilder,
                                               Type *ParamTy,
                                               uint64_t Offset) const {

  MachineFunction &MF = MIRBuilder.getMF();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PointerType *PtrTy = PointerType::get(ParamTy, AMDGPUAS::CONSTANT_ADDRESS);
  LLT PtrType = getLLTForType(*PtrTy, DL);
  Register DstReg = MRI.createGenericVirtualRegister(PtrType);
  Register KernArgSegmentPtr =
    MFI->getPreloadedReg(AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR);
  Register KernArgSegmentVReg = MRI.getLiveInVirtReg(KernArgSegmentPtr);

  Register OffsetReg = MRI.createGenericVirtualRegister(LLT::scalar(64));
  MIRBuilder.buildConstant(OffsetReg, Offset);

  MIRBuilder.buildGEP(DstReg, KernArgSegmentVReg, OffsetReg);

  return DstReg;
}

void AMDGPUCallLowering::lowerParameter(MachineIRBuilder &MIRBuilder,
                                        Type *ParamTy, uint64_t Offset,
                                        unsigned Align,
                                        Register DstReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PointerType *PtrTy = PointerType::get(ParamTy, AMDGPUAS::CONSTANT_ADDRESS);
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));
  unsigned TypeSize = DL.getTypeStoreSize(ParamTy);
  Register PtrReg = lowerParameterPtr(MIRBuilder, ParamTy, Offset);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(PtrInfo, MachineMemOperand::MOLoad |
                                       MachineMemOperand::MONonTemporal |
                                       MachineMemOperand::MOInvariant,
                                       TypeSize, Align);

  MIRBuilder.buildLoad(DstReg, PtrReg, *MMO);
}

static Register findFirstFreeSGPR(CCState &CCInfo) {
  unsigned NumSGPRs = AMDGPU::SGPR_32RegClass.getNumRegs();
  for (unsigned Reg = 0; Reg < NumSGPRs; ++Reg) {
    if (!CCInfo.isAllocated(AMDGPU::SGPR0 + Reg)) {
      return AMDGPU::SGPR0 + Reg;
    }
  }
  llvm_unreachable("Cannot allocate sgpr");
}

static void allocateSpecialEntryInputVGPRs(CCState &CCInfo,
                                           MachineFunction &MF,
                                           const SIRegisterInfo &TRI,
                                           SIMachineFunctionInfo &Info) {
  const LLT S32 = LLT::scalar(32);
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (Info.hasWorkItemIDX()) {
    Register Reg = AMDGPU::VGPR0;
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDX(ArgDescriptor::createRegister(Reg));
  }

  if (Info.hasWorkItemIDY()) {
    Register Reg = AMDGPU::VGPR1;
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDY(ArgDescriptor::createRegister(Reg));
  }

  if (Info.hasWorkItemIDZ()) {
    Register Reg = AMDGPU::VGPR2;
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::VGPR_32RegClass), S32);

    CCInfo.AllocateReg(Reg);
    Info.setWorkItemIDZ(ArgDescriptor::createRegister(Reg));
  }
}

// Allocate special inputs passed in user SGPRs.
static void allocateHSAUserSGPRs(CCState &CCInfo,
                                 MachineIRBuilder &MIRBuilder,
                                 MachineFunction &MF,
                                 const SIRegisterInfo &TRI,
                                 SIMachineFunctionInfo &Info) {
  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info.hasPrivateSegmentBuffer()) {
    unsigned PrivateSegmentBufferReg = Info.addPrivateSegmentBuffer(TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SGPR_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info.hasDispatchPtr()) {
    unsigned DispatchPtrReg = Info.addDispatchPtr(TRI);
    MF.addLiveIn(DispatchPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info.hasQueuePtr()) {
    unsigned QueuePtrReg = Info.addQueuePtr(TRI);
    MF.addLiveIn(QueuePtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info.hasKernargSegmentPtr()) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    Register InputPtrReg = Info.addKernargSegmentPtr(TRI);
    const LLT P4 = LLT::pointer(AMDGPUAS::CONSTANT_ADDRESS, 64);
    Register VReg = MRI.createGenericVirtualRegister(P4);
    MRI.addLiveIn(InputPtrReg, VReg);
    MIRBuilder.getMBB().addLiveIn(InputPtrReg);
    MIRBuilder.buildCopy(VReg, InputPtrReg);
    CCInfo.AllocateReg(InputPtrReg);
  }

  if (Info.hasDispatchID()) {
    unsigned DispatchIDReg = Info.addDispatchID(TRI);
    MF.addLiveIn(DispatchIDReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info.hasFlatScratchInit()) {
    unsigned FlatScratchInitReg = Info.addFlatScratchInit(TRI);
    MF.addLiveIn(FlatScratchInitReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  // TODO: Add GridWorkGroupCount user SGPRs when used. For now with HSA we read
  // these from the dispatch pointer.
}

static void allocateSystemSGPRs(CCState &CCInfo,
                                MachineFunction &MF,
                                SIMachineFunctionInfo &Info,
                                CallingConv::ID CallConv,
                                bool IsShader) {
  const LLT S32 = LLT::scalar(32);
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (Info.hasWorkGroupIDX()) {
    Register Reg = Info.addWorkGroupIDX();
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass), S32);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDY()) {
    Register Reg = Info.addWorkGroupIDY();
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass), S32);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupIDZ()) {
    unsigned Reg = Info.addWorkGroupIDZ();
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass), S32);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasWorkGroupInfo()) {
    unsigned Reg = Info.addWorkGroupInfo();
    MRI.setType(MF.addLiveIn(Reg, &AMDGPU::SReg_32_XM0RegClass), S32);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.hasPrivateSegmentWaveByteOffset()) {
    // Scratch wave offset passed in system SGPR.
    unsigned PrivateSegmentWaveByteOffsetReg;

    if (IsShader) {
      PrivateSegmentWaveByteOffsetReg =
        Info.getPrivateSegmentWaveByteOffsetSystemSGPR();

      // This is true if the scratch wave byte offset doesn't have a fixed
      // location.
      if (PrivateSegmentWaveByteOffsetReg == AMDGPU::NoRegister) {
        PrivateSegmentWaveByteOffsetReg = findFirstFreeSGPR(CCInfo);
        Info.setPrivateSegmentWaveByteOffset(PrivateSegmentWaveByteOffsetReg);
      }
    } else
      PrivateSegmentWaveByteOffsetReg = Info.addPrivateSegmentWaveByteOffset();

    MF.addLiveIn(PrivateSegmentWaveByteOffsetReg, &AMDGPU::SGPR_32RegClass);
    CCInfo.AllocateReg(PrivateSegmentWaveByteOffsetReg);
  }
}

bool AMDGPUCallLowering::lowerFormalArgumentsKernel(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const GCNSubtarget *Subtarget = &MF.getSubtarget<GCNSubtarget>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  const DataLayout &DL = F.getParent()->getDataLayout();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  allocateHSAUserSGPRs(CCInfo, MIRBuilder, MF, *TRI, *Info);

  unsigned i = 0;
  const unsigned KernArgBaseAlign = 16;
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
    unsigned Align = MinAlign(KernArgBaseAlign, ArgOffset);
    ArgOffset = alignTo(ArgOffset, DL.getABITypeAlignment(ArgTy));
    lowerParameter(MIRBuilder, ArgTy, ArgOffset, Align, ArgReg);
    if (OrigArgRegs.size() > 1)
      unpackRegs(OrigArgRegs, ArgReg, ArgTy, MIRBuilder);
    ++i;
  }

  allocateSpecialEntryInputVGPRs(CCInfo, MF, *TRI, *Info);
  allocateSystemSGPRs(CCInfo, MF, *Info, F.getCallingConv(), false);
  return true;
}

bool AMDGPUCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs) const {
  // The infrastructure for normal calling convention lowering is essentially
  // useless for kernels. We want to avoid any kind of legalization or argument
  // splitting.
  if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL)
    return lowerFormalArgumentsKernel(MIRBuilder, F, VRegs);

  // AMDGPU_GS and AMDGP_HS are not supported yet.
  if (F.getCallingConv() == CallingConv::AMDGPU_GS ||
      F.getCallingConv() == CallingConv::AMDGPU_HS)
    return false;

  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  const DataLayout &DL = F.getParent()->getDataLayout();

  bool IsShader = AMDGPU::isShader(F.getCallingConv());

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  if (Info->hasImplicitBufferPtr()) {
    unsigned ImplicitBufferPtrReg = Info->addImplicitBufferPtr(*TRI);
    MF.addLiveIn(ImplicitBufferPtrReg, &AMDGPU::SGPR_64RegClass);
    CCInfo.AllocateReg(ImplicitBufferPtrReg);
  }

  unsigned NumArgs = F.arg_size();
  Function::const_arg_iterator CurOrigArg = F.arg_begin();
  const AMDGPUTargetLowering &TLI = *getTLI<AMDGPUTargetLowering>();
  unsigned PSInputNum = 0;
  BitVector Skipped(NumArgs);
  for (unsigned i = 0; i != NumArgs; ++i, ++CurOrigArg) {
    EVT ValEVT = TLI.getValueType(DL, CurOrigArg->getType());

    // We can only hanlde simple value types at the moment.
    ISD::ArgFlagsTy Flags;
    assert(VRegs[i].size() == 1 && "Can't lower into more than one register");
    ArgInfo OrigArg{VRegs[i][0], CurOrigArg->getType()};
    setArgFlags(OrigArg, i + 1, DL, F);
    Flags.setOrigAlign(DL.getABITypeAlignment(CurOrigArg->getType()));

    if (F.getCallingConv() == CallingConv::AMDGPU_PS &&
        !OrigArg.Flags.isInReg() && !OrigArg.Flags.isByVal() &&
        PSInputNum <= 15) {
      if (CurOrigArg->use_empty() && !Info->isPSInputAllocated(PSInputNum)) {
        Skipped.set(i);
        ++PSInputNum;
        continue;
      }

      Info->markPSInputAllocated(PSInputNum);
      if (!CurOrigArg->use_empty())
        Info->markPSInputEnabled(PSInputNum);

      ++PSInputNum;
    }

    CCAssignFn *AssignFn = CCAssignFnForCall(F.getCallingConv(),
                                             /*IsVarArg=*/false);

    if (ValEVT.isVector()) {
      EVT ElemVT = ValEVT.getVectorElementType();
      if (!ValEVT.isSimple())
        return false;
      MVT ValVT = ElemVT.getSimpleVT();
      bool Res = AssignFn(i, ValVT, ValVT, CCValAssign::Full,
                          OrigArg.Flags, CCInfo);
      if (!Res)
        return false;
    } else {
      MVT ValVT = ValEVT.getSimpleVT();
      if (!ValEVT.isSimple())
        return false;
      bool Res =
          AssignFn(i, ValVT, ValVT, CCValAssign::Full, OrigArg.Flags, CCInfo);

      // Fail if we don't know how to handle this type.
      if (Res)
        return false;
    }
  }

  Function::const_arg_iterator Arg = F.arg_begin();

  if (F.getCallingConv() == CallingConv::AMDGPU_VS ||
      F.getCallingConv() == CallingConv::AMDGPU_PS) {
    for (unsigned i = 0, OrigArgIdx = 0;
         OrigArgIdx != NumArgs && i != ArgLocs.size(); ++Arg, ++OrigArgIdx) {
       if (Skipped.test(OrigArgIdx))
          continue;
       assert(VRegs[OrigArgIdx].size() == 1 &&
              "Can't lower into more than 1 reg");
       CCValAssign &VA = ArgLocs[i++];
       MRI.addLiveIn(VA.getLocReg(), VRegs[OrigArgIdx][0]);
       MIRBuilder.getMBB().addLiveIn(VA.getLocReg());
       MIRBuilder.buildCopy(VRegs[OrigArgIdx][0], VA.getLocReg());
    }

    allocateSystemSGPRs(CCInfo, MF, *Info, F.getCallingConv(), IsShader);
    return true;
  }

  return false;
}
