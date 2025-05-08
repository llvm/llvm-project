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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/MemorySSA.h"
#if LLPC_BUILD_NPI
#include "llvm/Analysis/ValueTracking.h"
#endif /* LLPC_BUILD_NPI */
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#if LLPC_BUILD_NPI
#include "llvm/IR/PatternMatch.h"
#endif /* LLPC_BUILD_NPI */
#include "llvm/IR/ReplaceConstant.h"

#define DEBUG_TYPE "amdgpu-memory-utils"

using namespace llvm;

namespace llvm::AMDGPU {

Align getAlign(const DataLayout &DL, const GlobalVariable *GV) {
  return DL.getValueOrABITypeAlignment(GV->getPointerAlignment(DL),
                                       GV->getValueType());
}

#if LLPC_BUILD_NPI
// Returns the target extension type of a global variable,
// which can only be a TargetExtType, an array or single-element struct of it,
// or their nesting combination.
// TODO: allow struct of multiple TargetExtType elements of the same type.
// TODO: Disallow other uses of target("amdgcn.named.barrier") or
// target("amdgcn.semaphore") including:
// - Structs containing barriers/semaphore in different scope/rank
// - Structs containing a mixture of barriers/semaphore and other data.
// - Globals in other address spaces.
// - Allocas.
static TargetExtType *getTargetExtType(const GlobalVariable &GV) {
#else /* LLPC_BUILD_NPI */
TargetExtType *isNamedBarrier(const GlobalVariable &GV) {
  // TODO: Allow arrays and structs, if all members are barriers
  // in the same scope.
  // TODO: Disallow other uses of target("amdgcn.named.barrier") including:
  // - Structs containing barriers in different scope.
  // - Structs containing a mixture of barriers and other data.
  // - Globals in other address spaces.
  // - Allocas.
#endif /* LLPC_BUILD_NPI */
  Type *Ty = GV.getValueType();
  while (true) {
    if (auto *TTy = dyn_cast<TargetExtType>(Ty))
#if LLPC_BUILD_NPI
      return TTy;
#else /* LLPC_BUILD_NPI */
      return TTy->getName() == "amdgcn.named.barrier" ? TTy : nullptr;
#endif /* LLPC_BUILD_NPI */
    if (auto *STy = dyn_cast<StructType>(Ty)) {
#if LLPC_BUILD_NPI
      if (STy->getNumElements() != 1)
#else /* LLPC_BUILD_NPI */
      if (STy->getNumElements() == 0)
#endif /* LLPC_BUILD_NPI */
        return nullptr;
      Ty = STy->getElementType(0);
      continue;
    }
#if LLPC_BUILD_NPI
    if (auto *ATy = dyn_cast<ArrayType>(Ty)) {
      Ty = ATy->getElementType();
      continue;
    }
#endif /* LLPC_BUILD_NPI */
    return nullptr;
  }
}

#if LLPC_BUILD_NPI
TargetExtType *isNamedBarrier(const GlobalVariable &GV) {
  if (TargetExtType *Ty = getTargetExtType(GV))
    return Ty->getName() == "amdgcn.named.barrier" ? Ty : nullptr;
  return nullptr;
}

TargetExtType *isLDSSemaphore(const GlobalVariable &GV) {
  if (TargetExtType *Ty = getTargetExtType(GV))
    return Ty->getName() == "amdgcn.semaphore" ? Ty : nullptr;
  return nullptr;
}

#endif /* LLPC_BUILD_NPI */
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
#if LLPC_BUILD_NPI
  if (isLDSSemaphore(GV))
    return false;
#endif /* LLPC_BUILD_NPI */
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
  // Some weirdness here. AMDGPU::isKernelCC does not call into
  // AMDGPU::isKernel with the calling conv, it instead calls into
  // isModuleEntryFunction which returns true for more calling conventions
  // than AMDGPU::isKernel does. There's a FIXME on AMDGPU::isKernel.
  // There's also a test that checks that the LDS lowering does not hit on
  // a graphics shader, denoted amdgpu_ps, so stay with the limited case.
  // Putting LDS in the name of the function to draw attention to this.
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
            report_fatal_error(
                "Module cannot mix absolute and non-absolute LDS GVs");
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
#if LLPC_BUILD_NPI
    case Intrinsic::amdgcn_s_cluster_barrier:
#endif /* LLPC_BUILD_NPI */
    case Intrinsic::amdgcn_s_barrier_signal:
    case Intrinsic::amdgcn_s_barrier_signal_var:
    case Intrinsic::amdgcn_s_barrier_signal_isfirst:
#if LLPC_BUILD_NPI
    case Intrinsic::amdgcn_s_barrier_init:
    case Intrinsic::amdgcn_s_barrier_join:
#endif /* LLPC_BUILD_NPI */
    case Intrinsic::amdgcn_s_barrier_wait:
#if LLPC_BUILD_NPI
    case Intrinsic::amdgcn_s_barrier_leave:
#endif /* LLPC_BUILD_NPI */
    case Intrinsic::amdgcn_s_get_barrier_state:
#if LLPC_BUILD_NPI
    case Intrinsic::amdgcn_s_wakeup_barrier:
#endif /* LLPC_BUILD_NPI */
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
  SmallSet<MemoryAccess *, 8> Visited;
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
#if LLPC_BUILD_NPI
}

static void collectUses(const Value &V, SmallVectorImpl<const Use *> &Uses) {
  SmallVector<const User *> WorkList;
  SmallPtrSet<const User *, 8> Visited;

  auto extendWorkList = [&](const Use &U) {
    auto User = U.getUser();
    if (Visited.count(User))
      return;
    Visited.insert(User);
    if (isa<ConstantExpr>(User) && isa<GEPOperator>(User))
      WorkList.push_back(User);
    else if (isa<GetElementPtrInst, PHINode, SelectInst>(User))
      WorkList.push_back(User);
  };

  for (auto &U : V.uses()) {
    Uses.push_back(&U);
    extendWorkList(U);
  }

  while (!WorkList.empty()) {
    auto *Cur = WorkList.pop_back_val();
    for (auto &U : Cur->uses()) {
      Uses.push_back(&U);
      extendWorkList(U);
    }
  }
}

static bool allPtrInputsInSameClass(const Value &V, Instruction *Inst) {
  unsigned i = isa<SelectInst>(Inst) ? 1 : 0;
  for (; i < Inst->getNumOperands(); ++i) {
    Value *Op = Inst->getOperand(i);

    if (isa<ConstantPointerNull>(Op))
      continue;

    // TODO-GFX13: if pointers are derived from two different
    // global lane-shared or private objects, it should still work. The
    // important part is both must be promotable into vgpr at
    // the end. It will require one more iteration of processing
    const Value *Obj = getUnderlyingObjectAggressive(Op);
    if (Obj != &V) {
      LLVM_DEBUG(dbgs() << "Found a select/phi with ptrs derived from two "
                           "different objects\n");
      return false;
    }
  }
  return true;
}

// Checks if the instruction I is a memset user of the global variable that we
// can deal with. Currently, only non-volatile memsets that affect the whole
// global variable are handled.
static bool isSupportedMemset(MemSetInst *I, const Value &V, Type *ValueType,
                              const DataLayout &DL) {
  using namespace PatternMatch;
  // For now we only care about non-volatile memsets that affect the whole
  // type (start at index 0 and fill the whole global variable).
  const unsigned Size = DL.getTypeStoreSize(ValueType);
  return I->getOperand(0) == &V &&
         match(I->getOperand(2), m_SpecificInt(Size)) && !I->isVolatile();
}

bool IsPromotableToVGPR(const Value &V, const DataLayout &DL) {
  const auto RejectUser = [&](Instruction *Inst, Twine Msg) {
    LLVM_DEBUG(dbgs() << "  Cannot promote to vgpr: " << Msg << "\n"
                      << "    " << *Inst << "\n");
    return false;
  };

  Type *ValueType;
  if (auto *GV = dyn_cast<GlobalVariable>(&V)) {
    assert(GV->getAddressSpace() == AMDGPUAS::LANE_SHARED);
    ValueType = GV->getValueType();
  } else if (auto *AI = dyn_cast<AllocaInst>(&V)) {
    if (!AI->isStaticAlloca() ||
        AI->getAddressSpace() != AMDGPUAS::PRIVATE_ADDRESS)
      return false;
    ValueType = AI->getAllocatedType();
  } else {
    llvm_unreachable("Unexpected promotion candidate!");
  }

  // TODO-GFX13: Do a proper allocation check across _all_ allocatable objects.
  if (DL.getTypeStoreSize(ValueType) > 4 * (1024 - 64)) {
    LLVM_DEBUG(dbgs() << "  Cannot promote to vgpr: too large\n");
    return false;
  }

  SmallVector<const Use *, 8> Uses;
  collectUses(V, Uses);

  for (auto *U : Uses) {
    Instruction *Inst = dyn_cast<Instruction>(U->getUser());
    if (!Inst)
      continue;

    if (getLoadStorePointerOperand(Inst)) {
      // This is a store of the pointer, not to the pointer.
      if (isa<StoreInst>(Inst) &&
          U->getOperandNo() != StoreInst::getPointerOperandIndex())
        return RejectUser(Inst, "pointer is being stored");

      // Check that this is a simple access of a vector element.
      bool IsSimple = isa<LoadInst>(Inst) ? cast<LoadInst>(Inst)->isSimple()
                                          : cast<StoreInst>(Inst)->isSimple();
      if (!IsSimple)
        return RejectUser(Inst, "not a simple load or store");

      auto Align = isa<LoadInst>(Inst) ? cast<LoadInst>(Inst)->getAlign()
                                       : cast<StoreInst>(Inst)->getAlign();
      if (Align < 4u)
        return RejectUser(Inst, "address is less than dword-aligned");

      Type *AccessTy = getLoadStoreType(Inst);
      auto DataSize = DL.getTypeAllocSize(AccessTy);
      if (DataSize % 4)
        return RejectUser(Inst, "data-size is not supported");

      continue;
    }

    if (isa<GetElementPtrInst>(Inst)) {
      continue;
    }

    if (isa<PHINode>(Inst)) {
      if (allPtrInputsInSameClass(V, Inst)) {
        continue;
      }
      return RejectUser(Inst, "phi on ptrs from two different objects");
    }
    if (isa<SelectInst>(Inst)) {
      if (allPtrInputsInSameClass(V, Inst)) {
        continue;
      }
      return RejectUser(Inst, "select on ptrs from two different objects");
    }

    if (MemSetInst *MSI = dyn_cast<MemSetInst>(Inst)) {
      if (isSupportedMemset(MSI, V, ValueType, DL)) {
        continue;
      }
      return RejectUser(Inst, "cannot handle partial memset inst yet");
    }

    if (isa<MemTransferInst>(Inst))
      return RejectUser(Inst, "cannot handle mem transfer inst yet");

    if (auto *Intr = dyn_cast<IntrinsicInst>(Inst)) {
      if (Intr->getIntrinsicID() == Intrinsic::objectsize) {
        continue;
      }
    }

    // Ignore assume-like intrinsics and comparisons used in assumes.
    if (isAssumeLikeIntrinsic(Inst)) {
      assert(Inst->use_empty() &&
             "does not expect assume-like intrinsic with any user");
      continue;
    }

    if (isa<ICmpInst>(Inst)) {
      if (!all_of(Inst->users(), [](User *U) {
            return isAssumeLikeIntrinsic(cast<Instruction>(U));
          }))
        return RejectUser(Inst, "used in icmp with non-assume-like uses");
      continue;
    }

    return RejectUser(Inst, "unhandled global-variable user");
  }
  return true;
#endif /* LLPC_BUILD_NPI */
}

} // end namespace llvm::AMDGPU
