//===-- llvm/lib/Target/AMDGPU/AMDGPUCallLowering.cpp - Call lowering -----===//
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

#include "AMDGPUCallLowering.h"
#include "AMDGPU.h"
#include "AMDGPUISelLowering.h"
#include "AMDGPUSubtarget.h"
#include "SIISelLowering.h"
#include "SIRegisterInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#ifndef LLVM_BUILD_GLOBAL_ISEL
#error "This shouldn't be built without GISel"
#endif

AMDGPUCallLowering::AMDGPUCallLowering(const AMDGPUTargetLowering &TLI)
  : CallLowering(&TLI) {
}

bool AMDGPUCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                     const Value *Val, unsigned VReg) const {
  MIRBuilder.buildInstr(AMDGPU::S_ENDPGM);
  return true;
}

unsigned AMDGPUCallLowering::lowerParameterPtr(MachineIRBuilder &MIRBuilder,
                                               Type *ParamTy,
                                               unsigned Offset) const {

  MachineFunction &MF = MIRBuilder.getMF();
  const SIRegisterInfo *TRI = MF.getSubtarget<SISubtarget>().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = *MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PointerType *PtrTy = PointerType::get(ParamTy, AMDGPUAS::CONSTANT_ADDRESS);
  LLT PtrType = getLLTForType(*PtrTy, DL);
  unsigned DstReg = MRI.createGenericVirtualRegister(PtrType);
  unsigned KernArgSegmentPtr =
      TRI->getPreloadedValue(MF, SIRegisterInfo::KERNARG_SEGMENT_PTR);
  unsigned KernArgSegmentVReg = MRI.getLiveInVirtReg(KernArgSegmentPtr);

  unsigned OffsetReg = MRI.createGenericVirtualRegister(LLT::scalar(64));
  MIRBuilder.buildConstant(OffsetReg, Offset);

  MIRBuilder.buildGEP(DstReg, KernArgSegmentVReg, OffsetReg);

  return DstReg;
}

void AMDGPUCallLowering::lowerParameter(MachineIRBuilder &MIRBuilder,
                                        Type *ParamTy, unsigned Offset,
                                        unsigned DstReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = *MF.getFunction();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PointerType *PtrTy = PointerType::get(ParamTy, AMDGPUAS::CONSTANT_ADDRESS);
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));
  unsigned TypeSize = DL.getTypeStoreSize(ParamTy);
  unsigned Align = DL.getABITypeAlignment(ParamTy);
  unsigned PtrReg = lowerParameterPtr(MIRBuilder, ParamTy, Offset);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(PtrInfo, MachineMemOperand::MOLoad |
                                       MachineMemOperand::MONonTemporal |
                                       MachineMemOperand::MOInvariant,
                                       TypeSize, Align);

  MIRBuilder.buildLoad(DstReg, PtrReg, *MMO);
}

bool AMDGPUCallLowering::lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                                              const Function &F,
                                              ArrayRef<unsigned> VRegs) const {

  MachineFunction &MF = MIRBuilder.getMF();
  const SISubtarget *Subtarget = static_cast<const SISubtarget *>(&MF.getSubtarget());
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *Info = MF.getInfo<SIMachineFunctionInfo>();
  const SIRegisterInfo *TRI = MF.getSubtarget<SISubtarget>().getRegisterInfo();
  const DataLayout &DL = F.getParent()->getDataLayout();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs, F.getContext());

  // FIXME: How should these inputs interact with inreg / custom SGPR inputs?
  if (Info->hasPrivateSegmentBuffer()) {
    unsigned PrivateSegmentBufferReg = Info->addPrivateSegmentBuffer(*TRI);
    MF.addLiveIn(PrivateSegmentBufferReg, &AMDGPU::SReg_128RegClass);
    CCInfo.AllocateReg(PrivateSegmentBufferReg);
  }

  if (Info->hasDispatchPtr()) {
    unsigned DispatchPtrReg = Info->addDispatchPtr(*TRI);
    // FIXME: Need to add reg as live-in
    CCInfo.AllocateReg(DispatchPtrReg);
  }

  if (Info->hasQueuePtr()) {
    unsigned QueuePtrReg = Info->addQueuePtr(*TRI);
    // FIXME: Need to add reg as live-in
    CCInfo.AllocateReg(QueuePtrReg);
  }

  if (Info->hasKernargSegmentPtr()) {
    unsigned InputPtrReg = Info->addKernargSegmentPtr(*TRI);
    const LLT P2 = LLT::pointer(2, 64);
    unsigned VReg = MRI.createGenericVirtualRegister(P2);
    MRI.addLiveIn(InputPtrReg, VReg);
    MIRBuilder.getMBB().addLiveIn(InputPtrReg);
    MIRBuilder.buildCopy(VReg, InputPtrReg);
    CCInfo.AllocateReg(InputPtrReg);
  }

  if (Info->hasDispatchID()) {
    unsigned DispatchIDReg = Info->addDispatchID(*TRI);
    // FIXME: Need to add reg as live-in
    CCInfo.AllocateReg(DispatchIDReg);
  }

  if (Info->hasFlatScratchInit()) {
    unsigned FlatScratchInitReg = Info->addFlatScratchInit(*TRI);
    // FIXME: Need to add reg as live-in
    CCInfo.AllocateReg(FlatScratchInitReg);
  }

  unsigned NumArgs = F.arg_size();
  Function::const_arg_iterator CurOrigArg = F.arg_begin();
  const AMDGPUTargetLowering &TLI = *getTLI<AMDGPUTargetLowering>();
  for (unsigned i = 0; i != NumArgs; ++i, ++CurOrigArg) {
    MVT ValVT = TLI.getValueType(DL, CurOrigArg->getType()).getSimpleVT();
    ISD::ArgFlagsTy Flags;
    Flags.setOrigAlign(DL.getABITypeAlignment(CurOrigArg->getType()));
    CCAssignFn *AssignFn = CCAssignFnForCall(F.getCallingConv(),
                                             /*IsVarArg=*/false);
    bool Res =
        AssignFn(i, ValVT, ValVT, CCValAssign::Full, Flags, CCInfo);
    assert(!Res && "Call operand has unhandled type");
    (void)Res;
  }

  Function::const_arg_iterator Arg = F.arg_begin();
  for (unsigned i = 0; i != NumArgs; ++i, ++Arg) {
    // FIXME: We should be getting DebugInfo from the arguments some how.
    CCValAssign &VA = ArgLocs[i];
    lowerParameter(MIRBuilder, Arg->getType(),
                   VA.getLocMemOffset() +
                   Subtarget->getExplicitKernelArgOffset(MF), VRegs[i]);
  }

  return true;
}
