//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower LDS global variables with target extension type "amdgpu.named.barrier"
// that require specialized address assignment. It assigns a unique
// barrier identifier to each named-barrier LDS variable and encodes
// this identifier within the !absolute_symbol metadata of that global.
// This encoding ensures that subsequent LDS lowering passes can process these
// barriers correctly without conflicts.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMemoryUtils.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <algorithm>

#define DEBUG_TYPE "amdgpu-lower-exec-sync"

using namespace llvm;
using namespace AMDGPU;

namespace {

// Write the specified address into metadata where it can be retrieved by
// the assembler. Format is a half open range, [Address Address+1)
static void recordLDSAbsoluteAddress(Module *M, GlobalVariable *GV,
                                     uint32_t Address) {
  LLVMContext &Ctx = M->getContext();
  auto *IntTy = M->getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LOCAL_ADDRESS);
  auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
  auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol,
                  MDNode::get(Ctx, {MinC, MaxC}));
}

/// Get next available ID for sync object. The ID allocation is tracked in \p
/// MaxNumGroup groups by \p NextAvailableIDTracker. Each call of the function
/// will ask for \p IDCnt against all the \p Kernels, it will return the
/// maximum of the available ones and update the ID tracker.
template <typename T>
unsigned allocateExecSyncID(T &NextAvailableIDTracker,
                            ArrayRef<Function *> Kernels, unsigned GroupID,
                            unsigned MaxNumGroup, unsigned IDCnt) {
  constexpr unsigned InitialVal = 1;
  unsigned NextID = InitialVal;
  for (Function *F : Kernels) {
    const SmallVectorImpl<unsigned> &NextAvailableID =
        NextAvailableIDTracker.lookup(F);
    unsigned ID = InitialVal;
    if (!NextAvailableID.empty())
      ID = NextAvailableID[GroupID];

    if (ID > NextID)
      NextID = ID;
  }

  // Bump the next available id for the kernels.
  for (Function *F : Kernels) {
    auto Inserted = NextAvailableIDTracker.try_emplace(F);
    // Initialize on first insertion.
    if (Inserted.second)
      Inserted.first->second.assign(MaxNumGroup, InitialVal);
    // Update the available ID.
    Inserted.first->second[GroupID] = NextID + IDCnt;
  }
  return NextID;
}

// Main utility function for special LDS variables lowering.
static bool lowerExecSyncGlobalVariables(Module &M,
                                         LDSUsesInfoTy &LDSUsesInfo) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();

  constexpr unsigned NumBarScopes = 1;
  MapVector<GlobalVariable *, SmallVector<Function *>> AllocationQ;
  DenseMap<Function *, SmallVector<unsigned, NumBarScopes>> KernelBarrierIDs;

  for (auto &[F, GVs] : LDSUsesInfo.indirect_access) {
    for (auto *GV : GVs) {
      if (!isNamedBarrier(*GV) || GV->isAbsoluteSymbolRef())
        continue;
      auto Iter = AllocationQ.find(GV);
      if (Iter == AllocationQ.end())
        AllocationQ.insert({GV, {F}});
      else
        Iter->second.push_back(F);
    }
  }

  for (auto &[F, GVs] : LDSUsesInfo.direct_access) {
    for (auto *GV : GVs) {
      if (!isNamedBarrier(*GV) || GV->isAbsoluteSymbolRef())
        continue;
      auto Iter = AllocationQ.find(GV);
      if (Iter == AllocationQ.end())
        AllocationQ.insert({GV, {F}});
      else
        Iter->second.push_back(F);
    }
  }

  sort(AllocationQ, [](std::pair<GlobalVariable *, SmallVector<Function *>> A,
                       std::pair<GlobalVariable *, SmallVector<Function *>> B) {
    // First order by number of kernels that access the GlobalVariable.
    if (A.second.size() != B.second.size())
      return A.second.size() > B.second.size();

    // Then order by their names so we always get a deterministic order.
    return A.first->getName() < B.first->getName();
  });

  for (auto &[GV, Kernels] : AllocationQ) {
    unsigned Offset;
    if (TargetExtType *ExtTy = isNamedBarrier(*GV)) {
      unsigned BarrierScope = ExtTy->getIntParameter(0);
      unsigned BarCnt = GV->getGlobalSize(DL) / 16;

      unsigned BarID = allocateExecSyncID(KernelBarrierIDs, Kernels,
                                          BarrierScope, NumBarScopes, BarCnt);

      LLVM_DEBUG(GV->printAsOperand(dbgs(), false);
                 dbgs() << " was assigned barrier id: " << BarID
                        << " id-count: " << BarCnt << "\n");
      // 4 bits for alignment, 5 bits for the barrier num,
      // 3 bits for the barrier scope
      Offset = 0x802000u | BarrierScope << 9 | BarID << 4;
    } else {
      llvm_unreachable("Unhandled special variable type.");
    }

    recordLDSAbsoluteAddress(&M, GV, Offset);
  }

  // Also erase those special LDS variables from indirect_access.
  for (auto &K : LDSUsesInfo.indirect_access) {
    assert(isKernel(*K.first));
    K.second.remove_if([](GlobalVariable *GV) { return isNamedBarrier(*GV); });
  }
  return Changed;
}

// With object linking, barrier ID assignment is deferred to the linker.
// Externalize named barrier globals and emit self-contained metadata so the
// AsmPrinter can generate the callgraph entries the linker needs.
static bool handleNamedBarriersForObjectLinking(Module &M) {
  DenseMap<GlobalVariable *, DenseSet<Function *>> BarrierToFuncs;
  for (GlobalVariable &GV : M.globals()) {
    if (!isNamedBarrier(GV) || GV.use_empty())
      continue;
    for (User *U : GV.users()) {
      if (auto *I = dyn_cast<Instruction>(U))
        BarrierToFuncs[&GV].insert(I->getFunction());
    }
  }
  if (BarrierToFuncs.empty())
    return false;

  LLVMContext &Ctx = M.getContext();
  NamedMDNode *BarMD = M.getOrInsertNamedMetadata("amdgpu.named_barrier.uses");

  std::string ModuleId;
  ModuleId = getUniqueModuleId(&M);
  assert(!ModuleId.empty() &&
         "modules with named barriers should have a unique ID");
  for (auto &[V, Funcs] : BarrierToFuncs) {
    if (V->hasLocalLinkage())
      V->setName("__amdgpu_named_barrier." + V->getName() + ModuleId);
    else if (!V->getName().starts_with("__amdgpu_named_barrier"))
      V->setName("__amdgpu_named_barrier." + V->getName());
    V->setInitializer(nullptr);
    V->setLinkage(GlobalValue::ExternalLinkage);

    SmallVector<Metadata *, 4> Ops;
    Ops.push_back(ValueAsMetadata::get(V));
    for (Function *F : Funcs)
      Ops.push_back(ValueAsMetadata::get(F));
    BarMD->addOperand(MDNode::get(Ctx, Ops));
  }
  return true;
}

static bool runLowerExecSyncGlobals(Module &M) {
  if (AMDGPUTargetMachine::EnableObjectLinking)
    return handleNamedBarriersForObjectLinking(M);

  CallGraph CG = CallGraph(M);
  bool Changed = false;
  Changed |= eliminateConstantExprUsesOfLDSFromAllInstructions(M);

  // For each kernel, what variables does it access directly or through
  // callees
  LDSUsesInfoTy LDSUsesInfo = getTransitiveUsesOfLDS(CG, M);

  if (LDSUsesInfo.HasSpecialGVs) {
    // Special LDS variables need special address assignment
    Changed |= lowerExecSyncGlobalVariables(M, LDSUsesInfo);
  }
  return Changed;
}

class AMDGPULowerExecSyncLegacy : public ModulePass {
public:
  static char ID;
  AMDGPULowerExecSyncLegacy() : ModulePass(ID) {}
  bool runOnModule(Module &M) override;
};

} // namespace

char AMDGPULowerExecSyncLegacy::ID = 0;
char &llvm::AMDGPULowerExecSyncLegacyPassID = AMDGPULowerExecSyncLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPULowerExecSyncLegacy, DEBUG_TYPE,
                      "AMDGPU lowering of execution synchronization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPULowerExecSyncLegacy, DEBUG_TYPE,
                    "AMDGPU lowering of execution synchronization", false,
                    false)

bool AMDGPULowerExecSyncLegacy::runOnModule(Module &M) {
  return runLowerExecSyncGlobals(M);
}

ModulePass *llvm::createAMDGPULowerExecSyncLegacyPass() {
  return new AMDGPULowerExecSyncLegacy();
}

PreservedAnalyses AMDGPULowerExecSyncPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  return runLowerExecSyncGlobals(M) ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
}
