//===-- AMDGPUMachineFunctionInfo.cpp ---------------------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineFunction.h"
#include "AMDGPU.h"
#include "AMDGPUPerfHintAnalysis.h"
#include "AMDGPUSubtarget.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Constants.h"
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
                                                  const GlobalVariable &GV,
                                                  Align Trailing) {
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

    // Align LDS size to trailing, e.g. for aligning dynamic shared memory
    LDSSize = alignTo(StaticLDSSize, Trailing);
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

const GlobalVariable *
AMDGPUMachineFunction::getKernelLDSGlobalFromFunction(const Function &F) {
  const Module *M = F.getParent();
  std::string KernelLDSName = "llvm.amdgcn.kernel.";
  KernelLDSName += F.getName();
  KernelLDSName += ".lds";
  return M->getNamedGlobal(KernelLDSName);
}

// This kernel calls no functions that require the module lds struct
static bool canElideModuleLDS(const Function &F) {
  return F.hasFnAttribute("amdgpu-elide-module-lds");
}

void AMDGPUMachineFunction::allocateKnownAddressLDSGlobal(const Function &F) {
  const Module *M = F.getParent();

  // This function is called before allocating any other LDS so that it can
  // reliably put values at known addresses. Consequently, dynamic LDS, if
  // present, will not yet have been allocated

  assert(getDynLDSAlign() == Align() && "dynamic LDS not yet allocated");

  if (isModuleEntryFunction()) {

    // Pointer values start from zero, memory allocated per-kernel-launch
    // Variables can be grouped into a module level struct and a struct per
    // kernel function by AMDGPULowerModuleLDSPass. If that is done, they
    // are allocated at statically computable addresses here.
    //
    // Address 0
    // {
    //   llvm.amdgcn.module.lds
    // }
    // alignment padding
    // {
    //   llvm.amdgcn.kernel.some-name.lds
    // }
    // other variables, e.g. dynamic lds, allocated after this call

    const GlobalVariable *GV = M->getNamedGlobal("llvm.amdgcn.module.lds");
    const GlobalVariable *KV = getKernelLDSGlobalFromFunction(F);

    if (GV && !canElideModuleLDS(F)) {
      unsigned Offset = allocateLDSGlobal(M->getDataLayout(), *GV, Align());
      (void)Offset;
      assert(Offset == 0 &&
             "Module LDS expected to be allocated before other LDS");
    }

    if (KV) {
      // The per-kernel offset is deterministic because it is allocated
      // before any other non-module LDS variables.
      unsigned Offset = allocateLDSGlobal(M->getDataLayout(), *KV, Align());
      (void)Offset;
    }
  }
}

Optional<uint32_t>
AMDGPUMachineFunction::getLDSKernelIdMetadata(const Function &F) {
  auto MD = F.getMetadata("llvm.amdgcn.lds.kernel.id");
  if (MD && MD->getNumOperands() == 1) {
    ConstantInt *KnownSize = mdconst::extract<ConstantInt>(MD->getOperand(0));
    if (KnownSize) {
      uint64_t V = KnownSize->getZExtValue();
      if (V <= UINT32_MAX) {
        return V;
      }
    }
  }
  return {};
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
