//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUArgumentUsageInfo.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-argument-reg-usage-info"

INITIALIZE_PASS(AMDGPUArgumentUsageInfo, DEBUG_TYPE,
                "Argument Register Usage Information Storage", false, true)

void ArgDescriptor::print(raw_ostream &OS,
                          const TargetRegisterInfo *TRI) const {
  if (!isSet()) {
    OS << "<not set>\n";
    return;
  }

  if (isRegister())
    OS << "Reg " << printReg(getRegister(), TRI);
  else
    OS << "Stack offset " << getStackOffset();

  if (isMasked()) {
    OS << " & ";
    llvm::write_hex(OS, Mask, llvm::HexPrintStyle::PrefixLower);
  }

  OS << '\n';
}

char AMDGPUArgumentUsageInfo::ID = 0;

const AMDGPUFunctionArgInfo AMDGPUArgumentUsageInfo::ExternFunctionInfo{};

// Hardcoded registers from fixed function ABI
const AMDGPUFunctionArgInfo AMDGPUArgumentUsageInfo::FixedABIFunctionInfo
  = AMDGPUFunctionArgInfo::fixedABILayout();

bool AMDGPUArgumentUsageInfo::doInitialization(Module &M) {
  return false;
}

bool AMDGPUArgumentUsageInfo::doFinalization(Module &M) {
  ArgInfoMap.clear();
  return false;
}

void AMDGPUArgumentUsageInfo::print(raw_ostream &OS, const Module *M) const {
  for (const auto &FI : ArgInfoMap) {
    OS << "Arguments for " << FI.first->getName() << '\n'
       << "  PrivateSegmentBuffer: " << FI.second.PrivateSegmentBuffer
       << "  DispatchPtr: " << FI.second.DispatchPtr
       << "  QueuePtr: " << FI.second.QueuePtr
       << "  KernargSegmentPtr: " << FI.second.KernargSegmentPtr
       << "  DispatchID: " << FI.second.DispatchID
       << "  FlatScratchInit: " << FI.second.FlatScratchInit
       << "  PrivateSegmentSize: " << FI.second.PrivateSegmentSize
       << "  WorkGroupIDX: " << FI.second.WorkGroupIDX
       << "  WorkGroupIDY: " << FI.second.WorkGroupIDY
       << "  WorkGroupIDZ: " << FI.second.WorkGroupIDZ
       << "  WorkGroupInfo: " << FI.second.WorkGroupInfo
       << "  PrivateSegmentWaveByteOffset: "
          << FI.second.PrivateSegmentWaveByteOffset
       << "  ImplicitBufferPtr: " << FI.second.ImplicitBufferPtr
       << "  ImplicitArgPtr: " << FI.second.ImplicitArgPtr
       << "  WorkItemIDX " << FI.second.WorkItemIDX
       << "  WorkItemIDY " << FI.second.WorkItemIDY
       << "  WorkItemIDZ " << FI.second.WorkItemIDZ
       << '\n';
  }
}

std::pair<const ArgDescriptor *, const TargetRegisterClass *>
AMDGPUFunctionArgInfo::getPreloadedValue(
  AMDGPUFunctionArgInfo::PreloadedValue Value) const {
  switch (Value) {
  case AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER: {
    return std::make_pair(
      PrivateSegmentBuffer ? &PrivateSegmentBuffer : nullptr,
      &AMDGPU::SGPR_128RegClass);
  }
  case AMDGPUFunctionArgInfo::IMPLICIT_BUFFER_PTR:
    return std::make_pair(ImplicitBufferPtr ? &ImplicitBufferPtr : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::WORKGROUP_ID_X:
    return std::make_pair(WorkGroupIDX ? &WorkGroupIDX : nullptr,
                          &AMDGPU::SGPR_32RegClass);

  case AMDGPUFunctionArgInfo::WORKGROUP_ID_Y:
    return std::make_pair(WorkGroupIDY ? &WorkGroupIDY : nullptr,
                          &AMDGPU::SGPR_32RegClass);
  case AMDGPUFunctionArgInfo::WORKGROUP_ID_Z:
    return std::make_pair(WorkGroupIDZ ? &WorkGroupIDZ : nullptr,
                          &AMDGPU::SGPR_32RegClass);
  case AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:
    return std::make_pair(
      PrivateSegmentWaveByteOffset ? &PrivateSegmentWaveByteOffset : nullptr,
      &AMDGPU::SGPR_32RegClass);
  case AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR:
    return std::make_pair(KernargSegmentPtr ? &KernargSegmentPtr : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::IMPLICIT_ARG_PTR:
    return std::make_pair(ImplicitArgPtr ? &ImplicitArgPtr : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::DISPATCH_ID:
    return std::make_pair(DispatchID ? &DispatchID : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT:
    return std::make_pair(FlatScratchInit ? &FlatScratchInit : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::DISPATCH_PTR:
    return std::make_pair(DispatchPtr ? &DispatchPtr : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::QUEUE_PTR:
    return std::make_pair(QueuePtr ? &QueuePtr : nullptr,
                          &AMDGPU::SGPR_64RegClass);
  case AMDGPUFunctionArgInfo::WORKITEM_ID_X:
    return std::make_pair(WorkItemIDX ? &WorkItemIDX : nullptr,
                          &AMDGPU::VGPR_32RegClass);
  case AMDGPUFunctionArgInfo::WORKITEM_ID_Y:
    return std::make_pair(WorkItemIDY ? &WorkItemIDY : nullptr,
                          &AMDGPU::VGPR_32RegClass);
  case AMDGPUFunctionArgInfo::WORKITEM_ID_Z:
    return std::make_pair(WorkItemIDZ ? &WorkItemIDZ : nullptr,
                          &AMDGPU::VGPR_32RegClass);
  }
  llvm_unreachable("unexpected preloaded value type");
}

constexpr AMDGPUFunctionArgInfo AMDGPUFunctionArgInfo::fixedABILayout() {
  AMDGPUFunctionArgInfo AI;
  AI.PrivateSegmentBuffer = AMDGPU::SGPR0_SGPR1_SGPR2_SGPR3;
  AI.DispatchPtr = AMDGPU::SGPR4_SGPR5;
  AI.QueuePtr = AMDGPU::SGPR6_SGPR7;

  // Do not pass kernarg segment pointer, only pass increment version in its
  // place.
  AI.ImplicitArgPtr = AMDGPU::SGPR8_SGPR9;
  AI.DispatchID = AMDGPU::SGPR10_SGPR11;

  // Skip FlatScratchInit/PrivateSegmentSize
  AI.WorkGroupIDX = AMDGPU::SGPR12;
  AI.WorkGroupIDY = AMDGPU::SGPR13;
  AI.WorkGroupIDZ = AMDGPU::SGPR14;

  const unsigned Mask = 0x3ff;
  AI.WorkItemIDX = ArgDescriptor::createRegister(AMDGPU::VGPR31, Mask);
  AI.WorkItemIDY = ArgDescriptor::createRegister(AMDGPU::VGPR31, Mask << 10);
  AI.WorkItemIDZ = ArgDescriptor::createRegister(AMDGPU::VGPR31, Mask << 20);
  return AI;
}

const AMDGPUFunctionArgInfo &
AMDGPUArgumentUsageInfo::lookupFuncArgInfo(const Function &F) const {
  auto I = ArgInfoMap.find(&F);
  if (I == ArgInfoMap.end()) {
    if (AMDGPUTargetMachine::EnableFixedFunctionABI)
      return FixedABIFunctionInfo;

    // Without the fixed ABI, we assume no function has special inputs.
    assert(F.isDeclaration());
    return ExternFunctionInfo;
  }

  return I->second;
}
