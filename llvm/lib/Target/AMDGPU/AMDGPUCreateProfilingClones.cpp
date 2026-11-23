//===- AMDGPUCreateProfilingClones.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Create a profiling clone "<name>.prof" of each kernel that accesses LDS.
///
/// This is the last IR pass before instruction selection and it does not
/// change the body of either copy, so both are selected and optimized
/// identically. The clone is instrumented later, on machine IR, by
/// AMDGPULDSProfiling. See AMDGPULDSProfiling.h.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPULDSProfiling.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "amdgpu-create-profiling-clones"

using namespace llvm;
using namespace llvm::AMDGPU::LDSProfiling;

bool AMDGPU::LDSProfiling::isSupportedSubtarget(const GCNSubtarget &ST) {
  // Only the 64 bank, wave32 model is implemented.
  return ST.isAmdHsaOS() && ST.hasGFX1250Insts() && ST.isWave32() &&
         ST.getLDSBankCount() == 64;
}

StringRef AMDGPU::LDSProfiling::getLDSKernelName(const Function &F) {
  Attribute A = F.getFnAttribute(CloneAttr);
  if (A.isValid() && A.isStringAttribute() && !A.getValueAsString().empty())
    return A.getValueAsString();
  return F.getName();
}

/// \returns true if the body of \p F reads or writes LDS.
static bool accessesLDS(const Function &F) {
  for (const Instruction &I : instructions(F)) {
    const Value *Ptr = getLoadStorePointerOperand(&I);
    if (!Ptr) {
      if (const auto *RMW = dyn_cast<AtomicRMWInst>(&I))
        Ptr = RMW->getPointerOperand();
      else if (const auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
        Ptr = CX->getPointerOperand();
      else
        continue;
    }
    if (Ptr->getType()->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
      return true;
  }
  return false;
}

static bool shouldClone(const Function &F, const TargetMachine &TM) {
  if (F.isDeclaration() || F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
    return false;
  if (F.hasFnAttribute(CloneAttr) || F.hasFnAttribute(OriginalAttr) ||
      F.hasFnAttribute(OptOutAttr))
    return false;
  // The runtime finds the clone and its counters by name.
  if (F.hasLocalLinkage() || F.hasAvailableExternallyLinkage())
    return false;
  // The clone and its counters are separate symbols from the kernel. When the
  // kernel may be defined in several translation units, they are deduplicated
  // with a comdat of their own. A weak kernel without a comdat has no such
  // mechanism. Frontends put every HIP kernel in a comdat.
  if (F.isWeakForLinker() && !F.hasComdat())
    return false;
  if (!isSupportedSubtarget(TM.getSubtarget<GCNSubtarget>(F)))
    return false;
  return accessesLDS(F);
}

static bool createProfilingClones(Module &M, const TargetMachine &TM) {
  SmallVector<Function *, 8> Kernels;
  for (Function &F : M)
    if (shouldClone(F, TM))
      Kernels.push_back(&F);

  for (Function *F : Kernels) {
    std::string CloneName = (F->getName() + CloneSuffix).str();
    std::string CountersName = CloneName + CountersSuffix.str();
    if (M.getNamedValue(CloneName) || M.getNamedValue(CountersName))
      continue;

    ValueToValueMapTy VMap;
    Function *Clone = CloneFunction(F, VMap);
    Clone->setName(CloneName);
    Clone->addFnAttr(CloneAttr, F->getName());
    F->addFnAttr(OriginalAttr);

    // The clone must not share the comdat of the kernel: the kernel is the key
    // of that group. Its own group also holds its counters.
    if (const Comdat *C = F->getComdat()) {
      Comdat *CloneComdat = M.getOrInsertComdat(CloneName);
      CloneComdat->setSelectionKind(C->getSelectionKind());
      Clone->setComdat(CloneComdat);
    }

    // The runtime handle of an enqueued block belongs to the original only.
    Clone->setMetadata(LLVMContext::MD_associated, nullptr);

    if (DISubprogram *SP = Clone->getSubprogram()) {
      if (SP != F->getSubprogram())
        SP->replaceLinkageName(MDString::get(M.getContext(), CloneName));
    }

    // The counters are defined when the clone is emitted, once the number of
    // LDS instructions is known.
    auto *Counters = new GlobalVariable(
        M, ArrayType::get(Type::getInt32Ty(M.getContext()), 0),
        /*isConstant=*/false, GlobalValue::ExternalLinkage,
        /*Initializer=*/nullptr, CountersName, /*InsertBefore=*/nullptr,
        GlobalValue::NotThreadLocal, AMDGPUAS::GLOBAL_ADDRESS);
    Counters->setVisibility(GlobalValue::ProtectedVisibility);
    Counters->setDSOLocal(true);
    Counters->setAlignment(Align(4));
  }

  return !Kernels.empty();
}

namespace {

class AMDGPUCreateProfilingClonesLegacy : public ModulePass {
  const TargetMachine *TM;

public:
  static char ID;

  explicit AMDGPUCreateProfilingClonesLegacy(const TargetMachine *TM = nullptr)
      : ModulePass(ID), TM(TM) {}

  StringRef getPassName() const override {
    return "AMDGPU Create Profiling Clones";
  }

private:
  bool runOnModule(Module &M) override {
    return TM && createProfilingClones(M, *TM);
  }
};

} // end anonymous namespace

char AMDGPUCreateProfilingClonesLegacy::ID = 0;

char &llvm::AMDGPUCreateProfilingClonesLegacyID =
    AMDGPUCreateProfilingClonesLegacy::ID;

INITIALIZE_PASS(AMDGPUCreateProfilingClonesLegacy, DEBUG_TYPE,
                "AMDGPU Create Profiling Clones", false, false)

ModulePass *
llvm::createAMDGPUCreateProfilingClonesLegacyPass(const TargetMachine *TM) {
  return new AMDGPUCreateProfilingClonesLegacy(TM);
}

PreservedAnalyses
AMDGPUCreateProfilingClonesPass::run(Module &M, ModuleAnalysisManager &MAM) {
  if (!createProfilingClones(M, TM))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
