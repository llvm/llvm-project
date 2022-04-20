//===-- AMDGPUMachineFunctionInfo.cpp ---------------------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineFunction.h"
#include "AMDGPUPerfHintAnalysis.h"
#include "AMDGPUSubtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

AMDGPUMachineFunction::AMDGPUMachineFunction(const MachineFunction &MF)
    : Mode(MF.getFunction()), IsEntryFunction(AMDGPU::isEntryFunctionCC(
                                  MF.getFunction().getCallingConv())),
      IsModuleEntryFunction(
          AMDGPU::isModuleEntryFunctionCC(MF.getFunction().getCallingConv())),
      NoSignedZerosFPMath(MF.getTarget().Options.NoSignedZerosFPMath) {
  const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(MF);

  // FIXME: Should initialize KernArgSize based on ExplicitKernelArgOffset,
  // except reserved size is not correctly aligned.
  const Function &F = MF.getFunction();

  Attribute MemBoundAttr = F.getFnAttribute("amdgpu-memory-bound");
  MemoryBound = MemBoundAttr.getValueAsBool();

  Attribute WaveLimitAttr = F.getFnAttribute("amdgpu-wave-limiter");
  WaveLimiter = WaveLimitAttr.getValueAsBool();

  // FIXME: How is this attribute supposed to interact with statically known
  // global sizes?
  StringRef S = F.getFnAttribute("amdgpu-gds-size").getValueAsString();
  if (!S.empty())
    S.consumeInteger(0, GDSSize);

  // Assume the attribute allocates before any known GDS globals.
  StaticGDSSize = GDSSize;

  CallingConv::ID CC = F.getCallingConv();
  if (CC == CallingConv::AMDGPU_KERNEL || CC == CallingConv::SPIR_KERNEL)
    ExplicitKernArgSize = ST.getExplicitKernArgSize(F, MaxKernArgAlign);
}

unsigned AMDGPUMachineFunction::allocateLDSGlobal(const DataLayout &DL,
                                                  const GlobalVariable &GV) {
  auto Entry = LocalMemoryObjects.insert(std::make_pair(&GV, 0));
  if (!Entry.second)
    return Entry.first->second;

  Align Alignment =
      DL.getValueOrABITypeAlignment(GV.getAlign(), GV.getValueType());

  unsigned Offset;
  if (GV.getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
    /// TODO: We should sort these to minimize wasted space due to alignment
    /// padding. Currently the padding is decided by the first encountered use
    /// during lowering.
    Offset = StaticLDSSize = alignTo(StaticLDSSize, Alignment);

    StaticLDSSize += DL.getTypeAllocSize(GV.getValueType());

    // Update the LDS size considering the padding to align the dynamic shared
    // memory.
    LDSSize = alignTo(StaticLDSSize, DynLDSAlign);
  } else {
    assert(GV.getAddressSpace() == AMDGPUAS::REGION_ADDRESS &&
           "expected region address space");

    Offset = StaticGDSSize = alignTo(StaticGDSSize, Alignment);
    StaticGDSSize += DL.getTypeAllocSize(GV.getValueType());

    // FIXME: Apply alignment of dynamic GDS
    GDSSize = StaticGDSSize;
  }

  Entry.first->second = Offset;
  return Offset;
}

void AMDGPUMachineFunction::allocateModuleLDSGlobal(const Module *M) {
  if (isModuleEntryFunction()) {
    const GlobalVariable *GV = M->getNamedGlobal("llvm.amdgcn.module.lds");
    if (GV) {
      unsigned Offset = allocateLDSGlobal(M->getDataLayout(), *GV);
      (void)Offset;
      assert(Offset == 0 &&
             "Module LDS expected to be allocated before other LDS");
    }
  }
}

void AMDGPUMachineFunction::setDynLDSAlign(const DataLayout &DL,
                                           const GlobalVariable &GV) {
  assert(DL.getTypeAllocSize(GV.getValueType()).isZero());

  Align Alignment =
      DL.getValueOrABITypeAlignment(GV.getAlign(), GV.getValueType());
  if (Alignment <= DynLDSAlign)
    return;

  LDSSize = alignTo(StaticLDSSize, Alignment);
  DynLDSAlign = Alignment;
}
