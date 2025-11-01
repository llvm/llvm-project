//===-- AMDGPUMemoryUtils.cpp - -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMemoryUtils.h"
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/ReplaceConstant.h"

#define DEBUG_TYPE "amdgpu-memory-utils"

using namespace llvm;

namespace llvm::AMDGPU {

Align getAlign(const DataLayout &DL, const GlobalVariable *GV) {
  return DL.getValueOrABITypeAlignment(GV->getPointerAlignment(DL),
                                       GV->getValueType());
}

// Returns the target extension type of a global variable,
// which can only be a TargetExtType, an array or single-element struct of it,
// or their nesting combination.
// TODO: allow struct of multiple TargetExtType elements of the same type.
// TODO: Disallow other uses of target("amdgcn.named.barrier") including:
// - Structs containing barriers in different scope/rank
// - Structs containing a mixture of barriers and other data.
// - Globals in other address spaces.
// - Allocas.
static TargetExtType *getTargetExtType(const GlobalVariable &GV) {
  Type *Ty = GV.getValueType();
  while (true) {
    if (auto *TTy = dyn_cast<TargetExtType>(Ty))
      return TTy;
    if (auto *STy = dyn_cast<StructType>(Ty)) {
      if (STy->getNumElements() != 1)
        return nullptr;
      Ty = STy->getElementType(0);
      continue;
    }
    if (auto *ATy = dyn_cast<ArrayType>(Ty)) {
      Ty = ATy->getElementType();
      continue;
    }
    return nullptr;
  }
}

TargetExtType *isNamedBarrier(const GlobalVariable &GV) {
  if (TargetExtType *Ty = getTargetExtType(GV))
    return Ty->getName() == "amdgcn.named.barrier" ? Ty : nullptr;
  return nullptr;
}

bool isDynamicLDS(const GlobalVariable &GV) {
  // external zero size addrspace(3) without initializer is dynlds.
  const Module *M = GV.getParent();
  const DataLayout &DL = M->getDataLayout();
  if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
    return false;
  return DL.getTypeAllocSize(GV.getValueType()) == 0;
}

bool isLDSVariableToLower(const GlobalVariable &GV) {
  if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS) {
    return false;
  }
  if (isDynamicLDS(GV)) {
    return true;
  }
  if (GV.isConstant()) {
    // A constant undef variable can't be written to, and any load is
    // undef, so it should be eliminated by the optimizer. It could be
    // dropped by the back end if not. This pass skips over it.
    return false;
  }
  if (GV.hasInitializer() && !isa<UndefValue>(GV.getInitializer())) {
    // Initializers are unimplemented for LDS address space.
    // Leave such variables in place for consistent error reporting.
    return false;
  }
  return true;
}

bool eliminateConstantExprUsesOfLDSFromAllInstructions(Module &M) {
  // Constants are uniqued within LLVM. A ConstantExpr referring to a LDS
  // global may have uses from multiple different functions as a result.
  // This pass specialises LDS variables with respect to the kernel that
  // allocates them.

  // This is semantically equivalent to (the unimplemented as slow):
  // for (auto &F : M.functions())
  //   for (auto &BB : F)
  //     for (auto &I : BB)
  //       for (Use &Op : I.operands())
  //         if (constantExprUsesLDS(Op))
  //           replaceConstantExprInFunction(I, Op);

  SmallVector<Constant *> LDSGlobals;
  for (auto &GV : M.globals())
    if (AMDGPU::isLDSVariableToLower(GV))
      LDSGlobals.push_back(&GV);
  return convertUsersOfConstantsToInstructions(LDSGlobals);
}

void getUsesOfLDSByFunction(const CallGraph &CG, Module &M,
                            FunctionVariableMap &kernels,
                            FunctionVariableMap &Functions) {
  // Get uses from the current function, excluding uses by called Functions
  // Two output variables to avoid walking the globals list twice
  for (auto &GV : M.globals()) {
    if (!AMDGPU::isLDSVariableToLower(GV))
      continue;
    for (User *V : GV.users()) {
      if (auto *I = dyn_cast<Instruction>(V)) {
        Function *F = I->getFunction();
        if (isKernelLDS(F))
          kernels[F].insert(&GV);
        else
          Functions[F].insert(&GV);
      }
    }
  }
}

bool isKernelLDS(const Function *F) {
  return AMDGPU::isKernel(F->getCallingConv());
}

LDSUsesInfoTy getTransitiveUsesOfLDS(const CallGraph &CG, Module &M) {

  FunctionVariableMap DirectMapKernel;
  FunctionVariableMap DirectMapFunction;
  getUsesOfLDSByFunction(CG, M, DirectMapKernel, DirectMapFunction);

  // Collect functions whose address has escaped
  DenseSet<Function *> AddressTakenFuncs;
  for (Function &F : M.functions()) {
    if (!isKernelLDS(&F))
      if (F.hasAddressTaken(nullptr,
                            /* IgnoreCallbackUses */ false,
                            /* IgnoreAssumeLikeCalls */ false,
                            /* IgnoreLLVMUsed */ true,
                            /* IgnoreArcAttachedCall */ false)) {
        AddressTakenFuncs.insert(&F);
      }
  }

  // Collect variables that are used by functions whose address has escaped
  DenseSet<GlobalVariable *> VariablesReachableThroughFunctionPointer;
  for (Function *F : AddressTakenFuncs) {
    set_union(VariablesReachableThroughFunctionPointer, DirectMapFunction[F]);
  }

  auto FunctionMakesUnknownCall = [&](const Function *F) -> bool {
    assert(!F->isDeclaration());
    for (const CallGraphNode::CallRecord &R : *CG[F]) {
      if (!R.second->getFunction())
        return true;
    }
    return false;
  };

  // Work out which variables are reachable through function calls
  FunctionVariableMap TransitiveMapFunction = DirectMapFunction;

  // If the function makes any unknown call, assume the worst case that it can
  // access all variables accessed by functions whose address escaped
  for (Function &F : M.functions()) {
    if (!F.isDeclaration() && FunctionMakesUnknownCall(&F)) {
      if (!isKernelLDS(&F)) {
        set_union(TransitiveMapFunction[&F],
                  VariablesReachableThroughFunctionPointer);
      }
    }
  }

  // Direct implementation of collecting all variables reachable from each
  // function
  for (Function &Func : M.functions()) {
    if (Func.isDeclaration() || isKernelLDS(&Func))
      continue;

    DenseSet<Function *> seen; // catches cycles
    SmallVector<Function *, 4> wip = {&Func};

    while (!wip.empty()) {
      Function *F = wip.pop_back_val();

      // Can accelerate this by referring to transitive map for functions that
      // have already been computed, with more care than this
      set_union(TransitiveMapFunction[&Func], DirectMapFunction[F]);

      for (const CallGraphNode::CallRecord &R : *CG[F]) {
        Function *Ith = R.second->getFunction();
        if (Ith) {
          if (!seen.contains(Ith)) {
            seen.insert(Ith);
            wip.push_back(Ith);
          }
        }
      }
    }
  }

  // Collect variables that are transitively used by functions whose address has
  // escaped
  for (Function *F : AddressTakenFuncs) {
    set_union(VariablesReachableThroughFunctionPointer,
              TransitiveMapFunction[F]);
  }

  // DirectMapKernel lists which variables are used by the kernel
  // find the variables which are used through a function call
  FunctionVariableMap IndirectMapKernel;

  for (Function &Func : M.functions()) {
    if (Func.isDeclaration() || !isKernelLDS(&Func))
      continue;

    for (const CallGraphNode::CallRecord &R : *CG[&Func]) {
      Function *Ith = R.second->getFunction();
      if (Ith) {
        set_union(IndirectMapKernel[&Func], TransitiveMapFunction[Ith]);
      }
    }

    // Check if the kernel encounters unknows calls, wheher directly or
    // indirectly.
    bool SeesUnknownCalls = [&]() {
      SmallVector<Function *> WorkList = {CG[&Func]->getFunction()};
      SmallPtrSet<Function *, 8> Visited;

      while (!WorkList.empty()) {
        Function *F = WorkList.pop_back_val();

        for (const CallGraphNode::CallRecord &CallRecord : *CG[F]) {
          if (!CallRecord.second)
            continue;

          Function *Callee = CallRecord.second->getFunction();
          if (!Callee)
            return true;

          if (Visited.insert(Callee).second)
            WorkList.push_back(Callee);
        }
      }
      return false;
    }();

    if (SeesUnknownCalls) {
      set_union(IndirectMapKernel[&Func],
                VariablesReachableThroughFunctionPointer);
    }
  }

  // Verify that we fall into one of 2 cases:
  //    - All variables are either absolute
  //      or direct mapped dynamic LDS that is not lowered.
  //      this is a re-run of the pass
  //      so we don't have anything to do.
  //    - No variables are absolute.
  std::optional<bool> HasAbsoluteGVs;
  bool HasSpecialGVs = false;
  for (auto &Map : {DirectMapKernel, IndirectMapKernel}) {
    for (auto &[Fn, GVs] : Map) {
      for (auto *GV : GVs) {
        bool IsAbsolute = GV->isAbsoluteSymbolRef();
        bool IsDirectMapDynLDSGV =
            AMDGPU::isDynamicLDS(*GV) && DirectMapKernel.contains(Fn);
        if (IsDirectMapDynLDSGV)
          continue;
        if (isNamedBarrier(*GV)) {
          HasSpecialGVs = true;
          continue;
        }
        if (HasAbsoluteGVs.has_value()) {
          if (*HasAbsoluteGVs != IsAbsolute) {
            reportFatalUsageError(
                "module cannot mix absolute and non-absolute LDS GVs");
          }
        } else
          HasAbsoluteGVs = IsAbsolute;
      }
    }
  }

  // If we only had absolute GVs, we have nothing to do, return an empty
  // result.
  if (HasAbsoluteGVs && *HasAbsoluteGVs)
    return {FunctionVariableMap(), FunctionVariableMap(), false};

  return {std::move(DirectMapKernel), std::move(IndirectMapKernel),
          HasSpecialGVs};
}

void removeFnAttrFromReachable(CallGraph &CG, Function *KernelRoot,
                               ArrayRef<StringRef> FnAttrs) {
  for (StringRef Attr : FnAttrs)
    KernelRoot->removeFnAttr(Attr);

  SmallVector<Function *> WorkList = {CG[KernelRoot]->getFunction()};
  SmallPtrSet<Function *, 8> Visited;
  bool SeenUnknownCall = false;

  while (!WorkList.empty()) {
    Function *F = WorkList.pop_back_val();

    for (auto &CallRecord : *CG[F]) {
      if (!CallRecord.second)
        continue;

      Function *Callee = CallRecord.second->getFunction();
      if (!Callee) {
        if (!SeenUnknownCall) {
          SeenUnknownCall = true;

          // If we see any indirect calls, assume nothing about potential
          // targets.
          // TODO: This could be refined to possible LDS global users.
          for (auto &ExternalCallRecord : *CG.getExternalCallingNode()) {
            Function *PotentialCallee =
                ExternalCallRecord.second->getFunction();
            assert(PotentialCallee);
            if (!isKernelLDS(PotentialCallee)) {
              for (StringRef Attr : FnAttrs)
                PotentialCallee->removeFnAttr(Attr);
            }
          }
        }
      } else {
        for (StringRef Attr : FnAttrs)
          Callee->removeFnAttr(Attr);
        if (Visited.insert(Callee).second)
          WorkList.push_back(Callee);
      }
    }
  }
}

bool isReallyAClobber(const Value *Ptr, MemoryDef *Def, AAResults *AA) {
  Instruction *DefInst = Def->getMemoryInst();

  if (isa<FenceInst>(DefInst))
    return false;

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(DefInst)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::amdgcn_s_cluster_barrier:
    case Intrinsic::amdgcn_s_barrier_signal:
    case Intrinsic::amdgcn_s_barrier_signal_var:
    case Intrinsic::amdgcn_s_barrier_signal_isfirst:
    case Intrinsic::amdgcn_s_barrier_init:
    case Intrinsic::amdgcn_s_barrier_join:
    case Intrinsic::amdgcn_s_barrier_wait:
    case Intrinsic::amdgcn_s_barrier_leave:
    case Intrinsic::amdgcn_s_get_barrier_state:
    case Intrinsic::amdgcn_wave_barrier:
    case Intrinsic::amdgcn_sched_barrier:
    case Intrinsic::amdgcn_sched_group_barrier:
    case Intrinsic::amdgcn_iglp_opt:
      return false;
    default:
      break;
    }
  }

  // Ignore atomics not aliasing with the original load, any atomic is a
  // universal MemoryDef from MSSA's point of view too, just like a fence.
  const auto checkNoAlias = [AA, Ptr](auto I) -> bool {
    return I && AA->isNoAlias(I->getPointerOperand(), Ptr);
  };

  if (checkNoAlias(dyn_cast<AtomicCmpXchgInst>(DefInst)) ||
      checkNoAlias(dyn_cast<AtomicRMWInst>(DefInst)))
    return false;

  return true;
}

bool isClobberedInFunction(const LoadInst *Load, MemorySSA *MSSA,
                           AAResults *AA) {
  MemorySSAWalker *Walker = MSSA->getWalker();
  SmallVector<MemoryAccess *> WorkList{Walker->getClobberingMemoryAccess(Load)};
  SmallPtrSet<MemoryAccess *, 8> Visited;
  MemoryLocation Loc(MemoryLocation::get(Load));

  LLVM_DEBUG(dbgs() << "Checking clobbering of: " << *Load << '\n');

  // Start with a nearest dominating clobbering access, it will be either
  // live on entry (nothing to do, load is not clobbered), MemoryDef, or
  // MemoryPhi if several MemoryDefs can define this memory state. In that
  // case add all Defs to WorkList and continue going up and checking all
  // the definitions of this memory location until the root. When all the
  // defs are exhausted and came to the entry state we have no clobber.
  // Along the scan ignore barriers and fences which are considered clobbers
  // by the MemorySSA, but not really writing anything into the memory.
  while (!WorkList.empty()) {
    MemoryAccess *MA = WorkList.pop_back_val();
    if (!Visited.insert(MA).second)
      continue;

    if (MSSA->isLiveOnEntryDef(MA))
      continue;

    if (MemoryDef *Def = dyn_cast<MemoryDef>(MA)) {
      LLVM_DEBUG(dbgs() << "  Def: " << *Def->getMemoryInst() << '\n');

      if (isReallyAClobber(Load->getPointerOperand(), Def, AA)) {
        LLVM_DEBUG(dbgs() << "      -> load is clobbered\n");
        return true;
      }

      WorkList.push_back(
          Walker->getClobberingMemoryAccess(Def->getDefiningAccess(), Loc));
      continue;
    }

    const MemoryPhi *Phi = cast<MemoryPhi>(MA);
    for (const auto &Use : Phi->incoming_values())
      WorkList.push_back(cast<MemoryAccess>(&Use));
  }

  LLVM_DEBUG(dbgs() << "      -> no clobber\n");
  return false;
}

GlobalVariable *uniquifyGVPerKernel(Module &M, GlobalVariable *GV,
                                    Function *KF) {
  bool NeedsReplacement = false;
  for (Use &U : GV->uses()) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      Function *F = I->getFunction();
      if (isKernelLDS(F) && F != KF) {
        NeedsReplacement = true;
        break;
      }
    }
  }
  if (!NeedsReplacement)
    return GV;
  // Create a new GV used only by this kernel and its function
  GlobalVariable *NewGV = new GlobalVariable(
      M, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
      GV->getInitializer(), GV->getName() + "." + KF->getName(), nullptr,
      GV->getThreadLocalMode(), GV->getType()->getAddressSpace());
  NewGV->copyAttributesFrom(GV);
  for (Use &U : make_early_inc_range(GV->uses())) {
    if (auto *I = dyn_cast<Instruction>(U.getUser())) {
      Function *F = I->getFunction();
      if (!isKernelLDS(F) || F == KF) {
        U.getUser()->replaceUsesOfWith(GV, NewGV);
      }
    }
  }
  return NewGV;
}

template <typename T> std::vector<T> sortByName(std::vector<T> &&V) {
  llvm::sort(V, [](const auto *L, const auto *R) {
    return L->getName() < R->getName();
  });
  return {std::move(V)};
}

void recordLDSAbsoluteAddress(Module *M, GlobalVariable *GV, uint32_t Address) {
  // Write the specified address into metadata where it can be retrieved by
  // the assembler. Format is a half open range, [Address Address+1)
  LLVMContext &Ctx = M->getContext();
  auto *IntTy = M->getDataLayout().getIntPtrType(Ctx, AMDGPUAS::LOCAL_ADDRESS);
  auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address));
  auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntTy, Address + 1));
  GV->setMetadata(LLVMContext::MD_absolute_symbol,
                  MDNode::get(Ctx, {MinC, MaxC}));
}

bool lowerSpecialLDSVariables(
    Module &M, LDSUsesInfoTy &LDSUsesInfo,
    VariableFunctionMap &LDSToKernelsThatNeedToAccessItIndirectly) {
  bool Changed = false;
  const DataLayout &DL = M.getDataLayout();
  // The 1st round: give module-absolute assignments
  int NumAbsolutes = 0;
  std::vector<GlobalVariable *> OrderedGVs;
  for (auto &K : LDSToKernelsThatNeedToAccessItIndirectly) {
    GlobalVariable *GV = K.first;
    if (!isNamedBarrier(*GV))
      continue;
    // give a module-absolute assignment if it is indirectly accessed by
    // multiple kernels. This is not precise, but we don't want to duplicate
    // a function when it is called by multiple kernels.
    if (LDSToKernelsThatNeedToAccessItIndirectly[GV].size() > 1) {
      OrderedGVs.push_back(GV);
    } else {
      // leave it to the 2nd round, which will give a kernel-relative
      // assignment if it is only indirectly accessed by one kernel
      LDSUsesInfo.direct_access[*K.second.begin()].insert(GV);
    }
    LDSToKernelsThatNeedToAccessItIndirectly.erase(GV);
  }
  OrderedGVs = sortByName(std::move(OrderedGVs));
  for (GlobalVariable *GV : OrderedGVs) {
    unsigned BarrierScope = llvm::AMDGPU::Barrier::BARRIER_SCOPE_WORKGROUP;
    unsigned BarId = NumAbsolutes + 1;
    unsigned BarCnt = DL.getTypeAllocSize(GV->getValueType()) / 16;
    NumAbsolutes += BarCnt;

    // 4 bits for alignment, 5 bits for the barrier num,
    // 3 bits for the barrier scope
    unsigned Offset = 0x802000u | BarrierScope << 9 | BarId << 4;
    recordLDSAbsoluteAddress(&M, GV, Offset);
  }
  OrderedGVs.clear();

  // The 2nd round: give a kernel-relative assignment for GV that
  // either only indirectly accessed by single kernel or only directly
  // accessed by multiple kernels.
  std::vector<Function *> OrderedKernels;
  for (auto &K : LDSUsesInfo.direct_access) {
    Function *F = K.first;
    assert(isKernelLDS(F));
    OrderedKernels.push_back(F);
  }
  OrderedKernels = sortByName(std::move(OrderedKernels));

  llvm::DenseMap<Function *, uint32_t> Kernel2BarId;
  for (Function *F : OrderedKernels) {
    for (GlobalVariable *GV : LDSUsesInfo.direct_access[F]) {
      if (!isNamedBarrier(*GV))
        continue;

      LDSUsesInfo.direct_access[F].erase(GV);
      if (GV->isAbsoluteSymbolRef()) {
        // already assigned
        continue;
      }
      OrderedGVs.push_back(GV);
    }
    OrderedGVs = sortByName(std::move(OrderedGVs));
    for (GlobalVariable *GV : OrderedGVs) {
      // GV could also be used directly by other kernels. If so, we need to
      // create a new GV used only by this kernel and its function.
      auto NewGV = uniquifyGVPerKernel(M, GV, F);
      Changed |= (NewGV != GV);
      unsigned BarrierScope = llvm::AMDGPU::Barrier::BARRIER_SCOPE_WORKGROUP;
      unsigned BarId = Kernel2BarId[F];
      BarId += NumAbsolutes + 1;
      unsigned BarCnt = DL.getTypeAllocSize(GV->getValueType()) / 16;
      Kernel2BarId[F] += BarCnt;
      unsigned Offset = 0x802000u | BarrierScope << 9 | BarId << 4;
      recordLDSAbsoluteAddress(&M, NewGV, Offset);
    }
    OrderedGVs.clear();
  }
  // Also erase those special LDS variables from indirect_access.
  for (auto &K : LDSUsesInfo.indirect_access) {
    assert(isKernelLDS(K.first));
    for (GlobalVariable *GV : K.second) {
      if (isNamedBarrier(*GV))
        K.second.erase(GV);
    }
  }
  return Changed;
}

} // end namespace llvm::AMDGPU
