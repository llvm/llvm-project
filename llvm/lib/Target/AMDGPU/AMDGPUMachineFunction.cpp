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
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

AMDGPUMachineFunction::AMDGPUMachineFunction(const Function &F,
                                             const AMDGPUSubtarget &ST)
    : IsEntryFunction(AMDGPU::isEntryFunctionCC(F.getCallingConv())),
      IsModuleEntryFunction(
          AMDGPU::isModuleEntryFunctionCC(F.getCallingConv())),
      NoSignedZerosFPMath(false) {

  // FIXME: Should initialize KernArgSize based on ExplicitKernelArgOffset,
  // except reserved size is not correctly aligned.

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

  // FIXME: Shouldn't be target specific
  Attribute NSZAttr = F.getFnAttribute("no-signed-zeros-fp-math");
  NoSignedZerosFPMath =
      NSZAttr.isStringAttribute() && NSZAttr.getValueAsString() == "true";
}

unsigned AMDGPUMachineFunction::allocateLDSGlobal(const DataLayout &DL,
                                                  const GlobalVariable &GV,
                                                  Align Trailing) {
  auto Entry = LocalMemoryObjects.insert(std::pair(&GV, 0));
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

static constexpr StringLiteral ModuleLDSName = "llvm.amdgcn.module.lds";

static const GlobalVariable *getKernelLDSGlobalFromFunction(const Function &F) {
  const Module *M = F.getParent();
  std::string KernelLDSName = "llvm.amdgcn.kernel.";
  KernelLDSName += F.getName();
  KernelLDSName += ".lds";
  return M->getNamedGlobal(KernelLDSName);
}

static const GlobalVariable *
getKernelDynLDSGlobalFromFunction(const Function &F) {
  const Module *M = F.getParent();
  std::string KernelDynLDSName = "llvm.amdgcn.";
  KernelDynLDSName += F.getName();
  KernelDynLDSName += ".dynlds";
  return M->getNamedGlobal(KernelDynLDSName);
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

    const GlobalVariable *GV = M->getNamedGlobal(ModuleLDSName);
    const GlobalVariable *KV = getKernelLDSGlobalFromFunction(F);
    const GlobalVariable *Dyn = getKernelDynLDSGlobalFromFunction(F);

    if (GV && !canElideModuleLDS(F)) {
      unsigned Offset = allocateLDSGlobal(M->getDataLayout(), *GV, Align());
      std::optional<uint32_t> Expect = getLDSAbsoluteAddress(*GV);
      if (!Expect || (Offset != *Expect)) {
        report_fatal_error("Inconsistent metadata on module LDS variable");
      }
    }

    if (KV) {
      // The per-kernel offset is deterministic because it is allocated
      // before any other non-module LDS variables.
      unsigned Offset = allocateLDSGlobal(M->getDataLayout(), *KV, Align());
      std::optional<uint32_t> Expect = getLDSAbsoluteAddress(*KV);
      if (!Expect || (Offset != *Expect)) {
        report_fatal_error("Inconsistent metadata on kernel LDS variable");
      }
    }

    if (Dyn) {
      // The dynamic LDS is deterministic because the per-kernel one has the
      // maximum alignment of any reachable and all remaining LDS variables,
      // if this is present, are themselves dynamic LDS and will be allocated
      // at the same address.
      setDynLDSAlign(F, *Dyn);
      unsigned Offset = LDSSize;
      std::optional<uint32_t> Expect = getLDSAbsoluteAddress(*Dyn);
      if (!Expect || (Offset != *Expect)) {
        report_fatal_error("Inconsistent metadata on dynamic LDS variable");
      }
    }
  }
}

std::optional<uint32_t>
AMDGPUMachineFunction::getLDSKernelIdMetadata(const Function &F) {
  // TODO: Would be more consistent with the abs symbols to use a range
  MDNode *MD = F.getMetadata("llvm.amdgcn.lds.kernel.id");
  if (MD && MD->getNumOperands() == 1) {
    if (ConstantInt *KnownSize =
            mdconst::extract<ConstantInt>(MD->getOperand(0))) {
      uint64_t ZExt = KnownSize->getZExtValue();
      if (ZExt <= UINT32_MAX) {
        return ZExt;
      }
    }
  }
  return {};
}

std::optional<uint32_t>
AMDGPUMachineFunction::getLDSAbsoluteAddress(const GlobalValue &GV) {
  if (GV.getAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
    return {};

  std::optional<ConstantRange> AbsSymRange = GV.getAbsoluteSymbolRange();
  if (!AbsSymRange)
    return {};

  if (const APInt *V = AbsSymRange->getSingleElement()) {
    std::optional<uint64_t> ZExt = V->tryZExtValue();
    if (ZExt && (*ZExt <= UINT32_MAX)) {
      return *ZExt;
    }
  }

  return {};
}

void AMDGPUMachineFunction::setDynLDSAlign(const Function &F,
                                           const GlobalVariable &GV) {
  const Module *M = F.getParent();
  const DataLayout &DL = M->getDataLayout();
  assert(DL.getTypeAllocSize(GV.getValueType()).isZero());

  Align Alignment =
      DL.getValueOrABITypeAlignment(GV.getAlign(), GV.getValueType());
  if (Alignment <= DynLDSAlign)
    return;

  LDSSize = alignTo(StaticLDSSize, Alignment);
  DynLDSAlign = Alignment;

  // If there is a dynamic LDS variable associated with this function F, every
  // further dynamic LDS instance (allocated by calling setDynLDSAlign) must
  // map to the same address. This holds because no LDS is allocated after the
  // lowering pass if there are dynamic LDS variables present.
  const GlobalVariable *Dyn = getKernelDynLDSGlobalFromFunction(F);
  if (Dyn) {
    unsigned Offset = LDSSize; // return this?
    std::optional<uint32_t> Expect = getLDSAbsoluteAddress(*Dyn);
    if (!Expect || (Offset != *Expect)) {
      report_fatal_error("Inconsistent metadata on dynamic LDS variable");
    }
  }
}
