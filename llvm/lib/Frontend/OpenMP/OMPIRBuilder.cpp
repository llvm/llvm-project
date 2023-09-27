//===- OpenMPIRBuilder.cpp - Builder for LLVM-IR for OpenMP directives ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the OpenMPIRBuilder class, which is used as a
/// convenient way to create LLVM instructions for OpenMP directives.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <cstdint>
#include <optional>

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;

static cl::opt<bool>
    OptimisticAttributes("openmp-ir-builder-optimistic-attributes", cl::Hidden,
                         cl::desc("Use optimistic attributes describing "
                                  "'as-if' properties of runtime calls."),
                         cl::init(false));

static cl::opt<double> UnrollThresholdFactor(
    "openmp-ir-builder-unroll-threshold-factor", cl::Hidden,
    cl::desc("Factor for the unroll threshold to account for code "
             "simplifications still taking place"),
    cl::init(1.5));

#ifndef NDEBUG
/// Return whether IP1 and IP2 are ambiguous, i.e. that inserting instructions
/// at position IP1 may change the meaning of IP2 or vice-versa. This is because
/// an InsertPoint stores the instruction before something is inserted. For
/// instance, if both point to the same instruction, two IRBuilders alternating
/// creating instruction will cause the instructions to be interleaved.
static bool isConflictIP(IRBuilder<>::InsertPoint IP1,
                         IRBuilder<>::InsertPoint IP2) {
  if (!IP1.isSet() || !IP2.isSet())
    return false;
  return IP1.getBlock() == IP2.getBlock() && IP1.getPoint() == IP2.getPoint();
}

static bool isValidWorkshareLoopScheduleType(OMPScheduleType SchedType) {
  // Valid ordered/unordered and base algorithm combinations.
  switch (SchedType & ~OMPScheduleType::MonotonicityMask) {
  case OMPScheduleType::UnorderedStaticChunked:
  case OMPScheduleType::UnorderedStatic:
  case OMPScheduleType::UnorderedDynamicChunked:
  case OMPScheduleType::UnorderedGuidedChunked:
  case OMPScheduleType::UnorderedRuntime:
  case OMPScheduleType::UnorderedAuto:
  case OMPScheduleType::UnorderedTrapezoidal:
  case OMPScheduleType::UnorderedGreedy:
  case OMPScheduleType::UnorderedBalanced:
  case OMPScheduleType::UnorderedGuidedIterativeChunked:
  case OMPScheduleType::UnorderedGuidedAnalyticalChunked:
  case OMPScheduleType::UnorderedSteal:
  case OMPScheduleType::UnorderedStaticBalancedChunked:
  case OMPScheduleType::UnorderedGuidedSimd:
  case OMPScheduleType::UnorderedRuntimeSimd:
  case OMPScheduleType::OrderedStaticChunked:
  case OMPScheduleType::OrderedStatic:
  case OMPScheduleType::OrderedDynamicChunked:
  case OMPScheduleType::OrderedGuidedChunked:
  case OMPScheduleType::OrderedRuntime:
  case OMPScheduleType::OrderedAuto:
  case OMPScheduleType::OrderdTrapezoidal:
  case OMPScheduleType::NomergeUnorderedStaticChunked:
  case OMPScheduleType::NomergeUnorderedStatic:
  case OMPScheduleType::NomergeUnorderedDynamicChunked:
  case OMPScheduleType::NomergeUnorderedGuidedChunked:
  case OMPScheduleType::NomergeUnorderedRuntime:
  case OMPScheduleType::NomergeUnorderedAuto:
  case OMPScheduleType::NomergeUnorderedTrapezoidal:
  case OMPScheduleType::NomergeUnorderedGreedy:
  case OMPScheduleType::NomergeUnorderedBalanced:
  case OMPScheduleType::NomergeUnorderedGuidedIterativeChunked:
  case OMPScheduleType::NomergeUnorderedGuidedAnalyticalChunked:
  case OMPScheduleType::NomergeUnorderedSteal:
  case OMPScheduleType::NomergeOrderedStaticChunked:
  case OMPScheduleType::NomergeOrderedStatic:
  case OMPScheduleType::NomergeOrderedDynamicChunked:
  case OMPScheduleType::NomergeOrderedGuidedChunked:
  case OMPScheduleType::NomergeOrderedRuntime:
  case OMPScheduleType::NomergeOrderedAuto:
  case OMPScheduleType::NomergeOrderedTrapezoidal:
    break;
  default:
    return false;
  }

  // Must not set both monotonicity modifiers at the same time.
  OMPScheduleType MonotonicityFlags =
      SchedType & OMPScheduleType::MonotonicityMask;
  if (MonotonicityFlags == OMPScheduleType::MonotonicityMask)
    return false;

  return true;
}
#endif

/// Determine which scheduling algorithm to use, determined from schedule clause
/// arguments.
static OMPScheduleType
getOpenMPBaseScheduleType(llvm::omp::ScheduleKind ClauseKind, bool HasChunks,
                          bool HasSimdModifier) {
  // Currently, the default schedule it static.
  switch (ClauseKind) {
  case OMP_SCHEDULE_Default:
  case OMP_SCHEDULE_Static:
    return HasChunks ? OMPScheduleType::BaseStaticChunked
                     : OMPScheduleType::BaseStatic;
  case OMP_SCHEDULE_Dynamic:
    return OMPScheduleType::BaseDynamicChunked;
  case OMP_SCHEDULE_Guided:
    return HasSimdModifier ? OMPScheduleType::BaseGuidedSimd
                           : OMPScheduleType::BaseGuidedChunked;
  case OMP_SCHEDULE_Auto:
    return llvm::omp::OMPScheduleType::BaseAuto;
  case OMP_SCHEDULE_Runtime:
    return HasSimdModifier ? OMPScheduleType::BaseRuntimeSimd
                           : OMPScheduleType::BaseRuntime;
  }
  llvm_unreachable("unhandled schedule clause argument");
}

/// Adds ordering modifier flags to schedule type.
static OMPScheduleType
getOpenMPOrderingScheduleType(OMPScheduleType BaseScheduleType,
                              bool HasOrderedClause) {
  assert((BaseScheduleType & OMPScheduleType::ModifierMask) ==
             OMPScheduleType::None &&
         "Must not have ordering nor monotonicity flags already set");

  OMPScheduleType OrderingModifier = HasOrderedClause
                                         ? OMPScheduleType::ModifierOrdered
                                         : OMPScheduleType::ModifierUnordered;
  OMPScheduleType OrderingScheduleType = BaseScheduleType | OrderingModifier;

  // Unsupported combinations
  if (OrderingScheduleType ==
      (OMPScheduleType::BaseGuidedSimd | OMPScheduleType::ModifierOrdered))
    return OMPScheduleType::OrderedGuidedChunked;
  else if (OrderingScheduleType == (OMPScheduleType::BaseRuntimeSimd |
                                    OMPScheduleType::ModifierOrdered))
    return OMPScheduleType::OrderedRuntime;

  return OrderingScheduleType;
}

/// Adds monotonicity modifier flags to schedule type.
static OMPScheduleType
getOpenMPMonotonicityScheduleType(OMPScheduleType ScheduleType,
                                  bool HasSimdModifier, bool HasMonotonic,
                                  bool HasNonmonotonic, bool HasOrderedClause) {
  assert((ScheduleType & OMPScheduleType::MonotonicityMask) ==
             OMPScheduleType::None &&
         "Must not have monotonicity flags already set");
  assert((!HasMonotonic || !HasNonmonotonic) &&
         "Monotonic and Nonmonotonic are contradicting each other");

  if (HasMonotonic) {
    return ScheduleType | OMPScheduleType::ModifierMonotonic;
  } else if (HasNonmonotonic) {
    return ScheduleType | OMPScheduleType::ModifierNonmonotonic;
  } else {
    // OpenMP 5.1, 2.11.4 Worksharing-Loop Construct, Description.
    // If the static schedule kind is specified or if the ordered clause is
    // specified, and if the nonmonotonic modifier is not specified, the
    // effect is as if the monotonic modifier is specified. Otherwise, unless
    // the monotonic modifier is specified, the effect is as if the
    // nonmonotonic modifier is specified.
    OMPScheduleType BaseScheduleType =
        ScheduleType & ~OMPScheduleType::ModifierMask;
    if ((BaseScheduleType == OMPScheduleType::BaseStatic) ||
        (BaseScheduleType == OMPScheduleType::BaseStaticChunked) ||
        HasOrderedClause) {
      // The monotonic is used by default in openmp runtime library, so no need
      // to set it.
      return ScheduleType;
    } else {
      return ScheduleType | OMPScheduleType::ModifierNonmonotonic;
    }
  }
}

/// Determine the schedule type using schedule and ordering clause arguments.
static OMPScheduleType
computeOpenMPScheduleType(ScheduleKind ClauseKind, bool HasChunks,
                          bool HasSimdModifier, bool HasMonotonicModifier,
                          bool HasNonmonotonicModifier, bool HasOrderedClause) {
  OMPScheduleType BaseSchedule =
      getOpenMPBaseScheduleType(ClauseKind, HasChunks, HasSimdModifier);
  OMPScheduleType OrderedSchedule =
      getOpenMPOrderingScheduleType(BaseSchedule, HasOrderedClause);
  OMPScheduleType Result = getOpenMPMonotonicityScheduleType(
      OrderedSchedule, HasSimdModifier, HasMonotonicModifier,
      HasNonmonotonicModifier, HasOrderedClause);

  assert(isValidWorkshareLoopScheduleType(Result));
  return Result;
}

/// Make \p Source branch to \p Target.
///
/// Handles two situations:
/// * \p Source already has an unconditional branch.
/// * \p Source is a degenerate block (no terminator because the BB is
///             the current head of the IR construction).
static void redirectTo(BasicBlock *Source, BasicBlock *Target, DebugLoc DL) {
  if (Instruction *Term = Source->getTerminator()) {
    auto *Br = cast<BranchInst>(Term);
    assert(!Br->isConditional() &&
           "BB's terminator must be an unconditional branch (or degenerate)");
    BasicBlock *Succ = Br->getSuccessor(0);
    Succ->removePredecessor(Source, /*KeepOneInputPHIs=*/true);
    Br->setSuccessor(0, Target);
    return;
  }

  auto *NewBr = BranchInst::Create(Target, Source);
  NewBr->setDebugLoc(DL);
}

void llvm::spliceBB(IRBuilderBase::InsertPoint IP, BasicBlock *New,
                    bool CreateBranch) {
  assert(New->getFirstInsertionPt() == New->begin() &&
         "Target BB must not have PHI nodes");

  // Move instructions to new block.
  BasicBlock *Old = IP.getBlock();
  New->splice(New->begin(), Old, IP.getPoint(), Old->end());

  if (CreateBranch)
    BranchInst::Create(New, Old);
}

void llvm::spliceBB(IRBuilder<> &Builder, BasicBlock *New, bool CreateBranch) {
  DebugLoc DebugLoc = Builder.getCurrentDebugLocation();
  BasicBlock *Old = Builder.GetInsertBlock();

  spliceBB(Builder.saveIP(), New, CreateBranch);
  if (CreateBranch)
    Builder.SetInsertPoint(Old->getTerminator());
  else
    Builder.SetInsertPoint(Old);

  // SetInsertPoint also updates the Builder's debug location, but we want to
  // keep the one the Builder was configured to use.
  Builder.SetCurrentDebugLocation(DebugLoc);
}

BasicBlock *llvm::splitBB(IRBuilderBase::InsertPoint IP, bool CreateBranch,
                          llvm::Twine Name) {
  BasicBlock *Old = IP.getBlock();
  BasicBlock *New = BasicBlock::Create(
      Old->getContext(), Name.isTriviallyEmpty() ? Old->getName() : Name,
      Old->getParent(), Old->getNextNode());
  spliceBB(IP, New, CreateBranch);
  New->replaceSuccessorsPhiUsesWith(Old, New);
  return New;
}

BasicBlock *llvm::splitBB(IRBuilderBase &Builder, bool CreateBranch,
                          llvm::Twine Name) {
  DebugLoc DebugLoc = Builder.getCurrentDebugLocation();
  BasicBlock *New = splitBB(Builder.saveIP(), CreateBranch, Name);
  if (CreateBranch)
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
  else
    Builder.SetInsertPoint(Builder.GetInsertBlock());
  // SetInsertPoint also updates the Builder's debug location, but we want to
  // keep the one the Builder was configured to use.
  Builder.SetCurrentDebugLocation(DebugLoc);
  return New;
}

BasicBlock *llvm::splitBB(IRBuilder<> &Builder, bool CreateBranch,
                          llvm::Twine Name) {
  DebugLoc DebugLoc = Builder.getCurrentDebugLocation();
  BasicBlock *New = splitBB(Builder.saveIP(), CreateBranch, Name);
  if (CreateBranch)
    Builder.SetInsertPoint(Builder.GetInsertBlock()->getTerminator());
  else
    Builder.SetInsertPoint(Builder.GetInsertBlock());
  // SetInsertPoint also updates the Builder's debug location, but we want to
  // keep the one the Builder was configured to use.
  Builder.SetCurrentDebugLocation(DebugLoc);
  return New;
}

BasicBlock *llvm::splitBBWithSuffix(IRBuilderBase &Builder, bool CreateBranch,
                                    llvm::Twine Suffix) {
  BasicBlock *Old = Builder.GetInsertBlock();
  return splitBB(Builder, CreateBranch, Old->getName() + Suffix);
}

//===----------------------------------------------------------------------===//
// OpenMPIRBuilderConfig
//===----------------------------------------------------------------------===//

namespace {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
/// Values for bit flags for marking which requires clauses have been used.
enum OpenMPOffloadingRequiresDirFlags {
  /// flag undefined.
  OMP_REQ_UNDEFINED = 0x000,
  /// no requires directive present.
  OMP_REQ_NONE = 0x001,
  /// reverse_offload clause.
  OMP_REQ_REVERSE_OFFLOAD = 0x002,
  /// unified_address clause.
  OMP_REQ_UNIFIED_ADDRESS = 0x004,
  /// unified_shared_memory clause.
  OMP_REQ_UNIFIED_SHARED_MEMORY = 0x008,
  /// dynamic_allocators clause.
  OMP_REQ_DYNAMIC_ALLOCATORS = 0x010,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/OMP_REQ_DYNAMIC_ALLOCATORS)
};

} // anonymous namespace

OpenMPIRBuilderConfig::OpenMPIRBuilderConfig()
    : RequiresFlags(OMP_REQ_UNDEFINED) {}

OpenMPIRBuilderConfig::OpenMPIRBuilderConfig(
    bool IsTargetDevice, bool IsGPU, bool OpenMPOffloadMandatory,
    bool HasRequiresReverseOffload, bool HasRequiresUnifiedAddress,
    bool HasRequiresUnifiedSharedMemory, bool HasRequiresDynamicAllocators)
    : IsTargetDevice(IsTargetDevice), IsGPU(IsGPU),
      OpenMPOffloadMandatory(OpenMPOffloadMandatory),
      RequiresFlags(OMP_REQ_UNDEFINED) {
  if (HasRequiresReverseOffload)
    RequiresFlags |= OMP_REQ_REVERSE_OFFLOAD;
  if (HasRequiresUnifiedAddress)
    RequiresFlags |= OMP_REQ_UNIFIED_ADDRESS;
  if (HasRequiresUnifiedSharedMemory)
    RequiresFlags |= OMP_REQ_UNIFIED_SHARED_MEMORY;
  if (HasRequiresDynamicAllocators)
    RequiresFlags |= OMP_REQ_DYNAMIC_ALLOCATORS;
}

bool OpenMPIRBuilderConfig::hasRequiresReverseOffload() const {
  return RequiresFlags & OMP_REQ_REVERSE_OFFLOAD;
}

bool OpenMPIRBuilderConfig::hasRequiresUnifiedAddress() const {
  return RequiresFlags & OMP_REQ_UNIFIED_ADDRESS;
}

bool OpenMPIRBuilderConfig::hasRequiresUnifiedSharedMemory() const {
  return RequiresFlags & OMP_REQ_UNIFIED_SHARED_MEMORY;
}

bool OpenMPIRBuilderConfig::hasRequiresDynamicAllocators() const {
  return RequiresFlags & OMP_REQ_DYNAMIC_ALLOCATORS;
}

int64_t OpenMPIRBuilderConfig::getRequiresFlags() const {
  return hasRequiresFlags() ? RequiresFlags
                            : static_cast<int64_t>(OMP_REQ_NONE);
}

void OpenMPIRBuilderConfig::setHasRequiresReverseOffload(bool Value) {
  if (Value)
    RequiresFlags |= OMP_REQ_REVERSE_OFFLOAD;
  else
    RequiresFlags &= ~OMP_REQ_REVERSE_OFFLOAD;
}

void OpenMPIRBuilderConfig::setHasRequiresUnifiedAddress(bool Value) {
  if (Value)
    RequiresFlags |= OMP_REQ_UNIFIED_ADDRESS;
  else
    RequiresFlags &= ~OMP_REQ_UNIFIED_ADDRESS;
}

void OpenMPIRBuilderConfig::setHasRequiresUnifiedSharedMemory(bool Value) {
  if (Value)
    RequiresFlags |= OMP_REQ_UNIFIED_SHARED_MEMORY;
  else
    RequiresFlags &= ~OMP_REQ_UNIFIED_SHARED_MEMORY;
}

void OpenMPIRBuilderConfig::setHasRequiresDynamicAllocators(bool Value) {
  if (Value)
    RequiresFlags |= OMP_REQ_DYNAMIC_ALLOCATORS;
  else
    RequiresFlags &= ~OMP_REQ_DYNAMIC_ALLOCATORS;
}

//===----------------------------------------------------------------------===//
// OpenMPIRBuilder
//===----------------------------------------------------------------------===//

void OpenMPIRBuilder::getKernelArgsVector(TargetKernelArgs &KernelArgs,
                                          IRBuilderBase &Builder,
                                          SmallVector<Value *> &ArgsVector) {
  Value *Version = Builder.getInt32(OMP_KERNEL_ARG_VERSION);
  Value *PointerNum = Builder.getInt32(KernelArgs.NumTargetItems);
  auto Int32Ty = Type::getInt32Ty(Builder.getContext());
  Value *ZeroArray = Constant::getNullValue(ArrayType::get(Int32Ty, 3));
  Value *Flags = Builder.getInt64(KernelArgs.HasNoWait);

  Value *NumTeams3D =
      Builder.CreateInsertValue(ZeroArray, KernelArgs.NumTeams, {0});
  Value *NumThreads3D =
      Builder.CreateInsertValue(ZeroArray, KernelArgs.NumThreads, {0});

  ArgsVector = {Version,
                PointerNum,
                KernelArgs.RTArgs.BasePointersArray,
                KernelArgs.RTArgs.PointersArray,
                KernelArgs.RTArgs.SizesArray,
                KernelArgs.RTArgs.MapTypesArray,
                KernelArgs.RTArgs.MapNamesArray,
                KernelArgs.RTArgs.MappersArray,
                KernelArgs.NumIterations,
                Flags,
                NumTeams3D,
                NumThreads3D,
                KernelArgs.DynCGGroupMem};
}

void OpenMPIRBuilder::addAttributes(omp::RuntimeFunction FnID, Function &Fn) {
  LLVMContext &Ctx = Fn.getContext();
  Triple T(M.getTargetTriple());

  // Get the function's current attributes.
  auto Attrs = Fn.getAttributes();
  auto FnAttrs = Attrs.getFnAttrs();
  auto RetAttrs = Attrs.getRetAttrs();
  SmallVector<AttributeSet, 4> ArgAttrs;
  for (size_t ArgNo = 0; ArgNo < Fn.arg_size(); ++ArgNo)
    ArgAttrs.emplace_back(Attrs.getParamAttrs(ArgNo));

  // Add AS to FnAS while taking special care with integer extensions.
  auto addAttrSet = [&](AttributeSet &FnAS, const AttributeSet &AS,
                        bool Param = true) -> void {
    bool HasSignExt = AS.hasAttribute(Attribute::SExt);
    bool HasZeroExt = AS.hasAttribute(Attribute::ZExt);
    if (HasSignExt || HasZeroExt) {
      assert(AS.getNumAttributes() == 1 &&
             "Currently not handling extension attr combined with others.");
      if (Param) {
        if (auto AK = TargetLibraryInfo::getExtAttrForI32Param(T, HasSignExt))
          FnAS = FnAS.addAttribute(Ctx, AK);
      } else if (auto AK =
                     TargetLibraryInfo::getExtAttrForI32Return(T, HasSignExt))
        FnAS = FnAS.addAttribute(Ctx, AK);
    } else {
      FnAS = FnAS.addAttributes(Ctx, AS);
    }
  };

#define OMP_ATTRS_SET(VarName, AttrSet) AttributeSet VarName = AttrSet;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // Add attributes to the function declaration.
  switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    FnAttrs = FnAttrs.addAttributes(Ctx, FnAttrSet);                           \
    addAttrSet(RetAttrs, RetAttrSet, /*Param*/ false);                         \
    for (size_t ArgNo = 0; ArgNo < ArgAttrSets.size(); ++ArgNo)                \
      addAttrSet(ArgAttrs[ArgNo], ArgAttrSets[ArgNo]);                         \
    Fn.setAttributes(AttributeList::get(Ctx, FnAttrs, RetAttrs, ArgAttrs));    \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    // Attributes are optional.
    break;
  }
}

FunctionCallee
OpenMPIRBuilder::getOrCreateRuntimeFunction(Module &M, RuntimeFunction FnID) {
  FunctionType *FnTy = nullptr;
  Function *Fn = nullptr;

  // Try to find the declation in the module first.
  switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    FnTy = FunctionType::get(ReturnType, ArrayRef<Type *>{__VA_ARGS__},        \
                             IsVarArg);                                        \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }

  if (!Fn) {
    // Create a new declaration if we need one.
    switch (FnID) {
#define OMP_RTL(Enum, Str, ...)                                                \
  case Enum:                                                                   \
    Fn = Function::Create(FnTy, GlobalValue::ExternalLinkage, Str, M);         \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
    }

    // Add information if the runtime function takes a callback function
    if (FnID == OMPRTL___kmpc_fork_call || FnID == OMPRTL___kmpc_fork_teams) {
      if (!Fn->hasMetadata(LLVMContext::MD_callback)) {
        LLVMContext &Ctx = Fn->getContext();
        MDBuilder MDB(Ctx);
        // Annotate the callback behavior of the runtime function:
        //  - The callback callee is argument number 2 (microtask).
        //  - The first two arguments of the callback callee are unknown (-1).
        //  - All variadic arguments to the runtime function are passed to the
        //    callback callee.
        Fn->addMetadata(
            LLVMContext::MD_callback,
            *MDNode::get(Ctx, {MDB.createCallbackEncoding(
                                  2, {-1, -1}, /* VarArgsArePassed */ true)}));
      }
    }

    LLVM_DEBUG(dbgs() << "Created OpenMP runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
    addAttributes(FnID, *Fn);

  } else {
    LLVM_DEBUG(dbgs() << "Found OpenMP runtime function " << Fn->getName()
                      << " with type " << *Fn->getFunctionType() << "\n");
  }

  assert(Fn && "Failed to create OpenMP runtime function");

  return {FnTy, Fn};
}

Function *OpenMPIRBuilder::getOrCreateRuntimeFunctionPtr(RuntimeFunction FnID) {
  FunctionCallee RTLFn = getOrCreateRuntimeFunction(M, FnID);
  auto *Fn = dyn_cast<llvm::Function>(RTLFn.getCallee());
  assert(Fn && "Failed to create OpenMP runtime function pointer");
  return Fn;
}

void OpenMPIRBuilder::initialize() { initializeTypes(M); }

void OpenMPIRBuilder::finalize(Function *Fn) {
  SmallPtrSet<BasicBlock *, 32> ParallelRegionBlockSet;
  SmallVector<BasicBlock *, 32> Blocks;
  SmallVector<OutlineInfo, 16> DeferredOutlines;
  for (OutlineInfo &OI : OutlineInfos) {
    // Skip functions that have not finalized yet; may happen with nested
    // function generation.
    if (Fn && OI.getFunction() != Fn) {
      DeferredOutlines.push_back(OI);
      continue;
    }

    ParallelRegionBlockSet.clear();
    Blocks.clear();
    OI.collectBlocks(ParallelRegionBlockSet, Blocks);

    Function *OuterFn = OI.getFunction();
    CodeExtractorAnalysisCache CEAC(*OuterFn);
    CodeExtractor Extractor(Blocks, /* DominatorTree */ nullptr,
                            /* AggregateArgs */ true,
                            /* BlockFrequencyInfo */ nullptr,
                            /* BranchProbabilityInfo */ nullptr,
                            /* AssumptionCache */ nullptr,
                            /* AllowVarArgs */ true,
                            /* AllowAlloca */ true,
                            /* AllocaBlock*/ OI.OuterAllocaBB,
                            /* Suffix */ ".omp_par");

    LLVM_DEBUG(dbgs() << "Before     outlining: " << *OuterFn << "\n");
    LLVM_DEBUG(dbgs() << "Entry " << OI.EntryBB->getName()
                      << " Exit: " << OI.ExitBB->getName() << "\n");
    assert(Extractor.isEligible() &&
           "Expected OpenMP outlining to be possible!");

    for (auto *V : OI.ExcludeArgsFromAggregate)
      Extractor.excludeArgFromAggregate(V);

    Function *OutlinedFn = Extractor.extractCodeRegion(CEAC);

    LLVM_DEBUG(dbgs() << "After      outlining: " << *OuterFn << "\n");
    LLVM_DEBUG(dbgs() << "   Outlined function: " << *OutlinedFn << "\n");
    assert(OutlinedFn->getReturnType()->isVoidTy() &&
           "OpenMP outlined functions should not return a value!");

    // For compability with the clang CG we move the outlined function after the
    // one with the parallel region.
    OutlinedFn->removeFromParent();
    M.getFunctionList().insertAfter(OuterFn->getIterator(), OutlinedFn);

    // Remove the artificial entry introduced by the extractor right away, we
    // made our own entry block after all.
    {
      BasicBlock &ArtificialEntry = OutlinedFn->getEntryBlock();
      assert(ArtificialEntry.getUniqueSuccessor() == OI.EntryBB);
      assert(OI.EntryBB->getUniquePredecessor() == &ArtificialEntry);
      // Move instructions from the to-be-deleted ArtificialEntry to the entry
      // basic block of the parallel region. CodeExtractor generates
      // instructions to unwrap the aggregate argument and may sink
      // allocas/bitcasts for values that are solely used in the outlined region
      // and do not escape.
      assert(!ArtificialEntry.empty() &&
             "Expected instructions to add in the outlined region entry");
      for (BasicBlock::reverse_iterator It = ArtificialEntry.rbegin(),
                                        End = ArtificialEntry.rend();
           It != End;) {
        Instruction &I = *It;
        It++;

        if (I.isTerminator())
          continue;

        I.moveBeforePreserving(*OI.EntryBB, OI.EntryBB->getFirstInsertionPt());
      }

      OI.EntryBB->moveBefore(&ArtificialEntry);
      ArtificialEntry.eraseFromParent();
    }
    assert(&OutlinedFn->getEntryBlock() == OI.EntryBB);
    assert(OutlinedFn && OutlinedFn->getNumUses() == 1);

    // Run a user callback, e.g. to add attributes.
    if (OI.PostOutlineCB)
      OI.PostOutlineCB(*OutlinedFn);
  }

  // Remove work items that have been completed.
  OutlineInfos = std::move(DeferredOutlines);

  EmitMetadataErrorReportFunctionTy &&ErrorReportFn =
      [](EmitMetadataErrorKind Kind,
         const TargetRegionEntryInfo &EntryInfo) -> void {
    errs() << "Error of kind: " << Kind
           << " when emitting offload entries and metadata during "
              "OMPIRBuilder finalization \n";
  };

  if (!OffloadInfoManager.empty())
    createOffloadEntriesAndInfoMetadata(ErrorReportFn);
}

OpenMPIRBuilder::~OpenMPIRBuilder() {
  assert(OutlineInfos.empty() && "There must be no outstanding outlinings");
}

GlobalValue *OpenMPIRBuilder::createGlobalFlag(unsigned Value, StringRef Name) {
  IntegerType *I32Ty = Type::getInt32Ty(M.getContext());
  auto *GV =
      new GlobalVariable(M, I32Ty,
                         /* isConstant = */ true, GlobalValue::WeakODRLinkage,
                         ConstantInt::get(I32Ty, Value), Name);
  GV->setVisibility(GlobalValue::HiddenVisibility);

  return GV;
}

Constant *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                            uint32_t SrcLocStrSize,
                                            IdentFlag LocFlags,
                                            unsigned Reserve2Flags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  Constant *&Ident =
      IdentMap[{SrcLocStr, uint64_t(LocFlags) << 31 | Reserve2Flags}];
  if (!Ident) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {I32Null,
                             ConstantInt::get(Int32, uint32_t(LocFlags)),
                             ConstantInt::get(Int32, Reserve2Flags),
                             ConstantInt::get(Int32, SrcLocStrSize), SrcLocStr};
    Constant *Initializer =
        ConstantStruct::get(OpenMPIRBuilder::Ident, IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.globals())
      if (GV.getValueType() == OpenMPIRBuilder::Ident && GV.hasInitializer())
        if (GV.getInitializer() == Initializer)
          Ident = &GV;

    if (!Ident) {
      auto *GV = new GlobalVariable(
          M, OpenMPIRBuilder::Ident,
          /* isConstant = */ true, GlobalValue::PrivateLinkage, Initializer, "",
          nullptr, GlobalValue::NotThreadLocal,
          M.getDataLayout().getDefaultGlobalsAddressSpace());
      GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      GV->setAlignment(Align(8));
      Ident = GV;
    }
  }

  return ConstantExpr::getPointerBitCastOrAddrSpaceCast(Ident, IdentPtr);
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef LocStr,
                                                uint32_t &SrcLocStrSize) {
  SrcLocStrSize = LocStr.size();
  Constant *&SrcLocStr = SrcLocStrMap[LocStr];
  if (!SrcLocStr) {
    Constant *Initializer =
        ConstantDataArray::getString(M.getContext(), LocStr);

    // Look for existing encoding of the location, not needed but minimizes the
    // difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.globals())
      if (GV.isConstant() && GV.hasInitializer() &&
          GV.getInitializer() == Initializer)
        return SrcLocStr = ConstantExpr::getPointerCast(&GV, Int8Ptr);

    SrcLocStr = Builder.CreateGlobalStringPtr(LocStr, /* Name */ "",
                                              /* AddressSpace */ 0, &M);
  }
  return SrcLocStr;
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef FunctionName,
                                                StringRef FileName,
                                                unsigned Line, unsigned Column,
                                                uint32_t &SrcLocStrSize) {
  SmallString<128> Buffer;
  Buffer.push_back(';');
  Buffer.append(FileName);
  Buffer.push_back(';');
  Buffer.append(FunctionName);
  Buffer.push_back(';');
  Buffer.append(std::to_string(Line));
  Buffer.push_back(';');
  Buffer.append(std::to_string(Column));
  Buffer.push_back(';');
  Buffer.push_back(';');
  return getOrCreateSrcLocStr(Buffer.str(), SrcLocStrSize);
}

Constant *
OpenMPIRBuilder::getOrCreateDefaultSrcLocStr(uint32_t &SrcLocStrSize) {
  StringRef UnknownLoc = ";unknown;unknown;0;0;;";
  return getOrCreateSrcLocStr(UnknownLoc, SrcLocStrSize);
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(DebugLoc DL,
                                                uint32_t &SrcLocStrSize,
                                                Function *F) {
  DILocation *DIL = DL.get();
  if (!DIL)
    return getOrCreateDefaultSrcLocStr(SrcLocStrSize);
  StringRef FileName = M.getName();
  if (DIFile *DIF = DIL->getFile())
    if (std::optional<StringRef> Source = DIF->getSource())
      FileName = *Source;
  StringRef Function = DIL->getScope()->getSubprogram()->getName();
  if (Function.empty() && F)
    Function = F->getName();
  return getOrCreateSrcLocStr(Function, FileName, DIL->getLine(),
                              DIL->getColumn(), SrcLocStrSize);
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc,
                                                uint32_t &SrcLocStrSize) {
  return getOrCreateSrcLocStr(Loc.DL, SrcLocStrSize,
                              Loc.IP.getBlock()->getParent());
}

Value *OpenMPIRBuilder::getOrCreateThreadID(Value *Ident) {
  return Builder.CreateCall(
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_global_thread_num), Ident,
      "omp_global_thread_num");
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createBarrier(const LocationDescription &Loc, Directive DK,
                               bool ForceSimpleCall, bool CheckCancelFlag) {
  if (!updateToLocation(Loc))
    return Loc.IP;
  return emitBarrierImpl(Loc, DK, ForceSimpleCall, CheckCancelFlag);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::emitBarrierImpl(const LocationDescription &Loc, Directive Kind,
                                 bool ForceSimpleCall, bool CheckCancelFlag) {
  // Build call __kmpc_cancel_barrier(loc, thread_id) or
  //            __kmpc_barrier(loc, thread_id);

  IdentFlag BarrierLocFlags;
  switch (Kind) {
  case OMPD_for:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_FOR;
    break;
  case OMPD_sections:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SECTIONS;
    break;
  case OMPD_single:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL_SINGLE;
    break;
  case OMPD_barrier:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_EXPL;
    break;
  default:
    BarrierLocFlags = OMP_IDENT_FLAG_BARRIER_IMPL;
    break;
  }

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Args[] = {
      getOrCreateIdent(SrcLocStr, SrcLocStrSize, BarrierLocFlags),
      getOrCreateThreadID(getOrCreateIdent(SrcLocStr, SrcLocStrSize))};

  // If we are in a cancellable parallel region, barriers are cancellation
  // points.
  // TODO: Check why we would force simple calls or to ignore the cancel flag.
  bool UseCancelBarrier =
      !ForceSimpleCall && isLastFinalizationInfoCancellable(OMPD_parallel);

  Value *Result =
      Builder.CreateCall(getOrCreateRuntimeFunctionPtr(
                             UseCancelBarrier ? OMPRTL___kmpc_cancel_barrier
                                              : OMPRTL___kmpc_barrier),
                         Args);

  if (UseCancelBarrier && CheckCancelFlag)
    emitCancelationCheckImpl(Result, OMPD_parallel);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createCancel(const LocationDescription &Loc,
                              Value *IfCondition,
                              omp::Directive CanceledDirective) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  // LLVM utilities like blocks with terminators.
  auto *UI = Builder.CreateUnreachable();

  Instruction *ThenTI = UI, *ElseTI = nullptr;
  if (IfCondition)
    SplitBlockAndInsertIfThenElse(IfCondition, UI, &ThenTI, &ElseTI);
  Builder.SetInsertPoint(ThenTI);

  Value *CancelKind = nullptr;
  switch (CanceledDirective) {
#define OMP_CANCEL_KIND(Enum, Str, DirectiveEnum, Value)                       \
  case DirectiveEnum:                                                          \
    CancelKind = Builder.getInt32(Value);                                      \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    llvm_unreachable("Unknown cancel kind!");
  }

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident), CancelKind};
  Value *Result = Builder.CreateCall(
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_cancel), Args);
  auto ExitCB = [this, CanceledDirective, Loc](InsertPointTy IP) {
    if (CanceledDirective == OMPD_parallel) {
      IRBuilder<>::InsertPointGuard IPG(Builder);
      Builder.restoreIP(IP);
      createBarrier(LocationDescription(Builder.saveIP(), Loc.DL),
                    omp::Directive::OMPD_unknown, /* ForceSimpleCall */ false,
                    /* CheckCancelFlag */ false);
    }
  };

  // The actual cancel logic is shared with others, e.g., cancel_barriers.
  emitCancelationCheckImpl(Result, CanceledDirective, ExitCB);

  // Update the insertion point and remove the terminator we introduced.
  Builder.SetInsertPoint(UI->getParent());
  UI->eraseFromParent();

  return Builder.saveIP();
}

void OpenMPIRBuilder::emitOffloadingEntry(Constant *Addr, StringRef Name,
                                          uint64_t Size, int32_t Flags,
                                          StringRef SectionName) {
  Type *Int8PtrTy = Type::getInt8PtrTy(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *SizeTy = M.getDataLayout().getIntPtrType(M.getContext());

  Constant *AddrName = ConstantDataArray::getString(M.getContext(), Name);

  // Create the constant string used to look up the symbol in the device.
  auto *Str =
      new llvm::GlobalVariable(M, AddrName->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::InternalLinkage, AddrName,
                               ".omp_offloading.entry_name");
  Str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  // Construct the offloading entry.
  Constant *EntryData[] = {
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Addr, Int8PtrTy),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Str, Int8PtrTy),
      ConstantInt::get(SizeTy, Size),
      ConstantInt::get(Int32Ty, Flags),
      ConstantInt::get(Int32Ty, 0),
  };
  Constant *EntryInitializer =
      ConstantStruct::get(OpenMPIRBuilder::OffloadEntry, EntryData);

  auto *Entry = new GlobalVariable(
      M, OpenMPIRBuilder::OffloadEntry,
      /* isConstant = */ true, GlobalValue::WeakAnyLinkage, EntryInitializer,
      ".omp_offloading.entry." + Name, nullptr, GlobalValue::NotThreadLocal,
      M.getDataLayout().getDefaultGlobalsAddressSpace());

  // The entry has to be created in the section the linker expects it to be.
  Entry->setSection(SectionName);
  Entry->setAlignment(Align(1));
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitTargetKernel(
    const LocationDescription &Loc, InsertPointTy AllocaIP, Value *&Return,
    Value *Ident, Value *DeviceID, Value *NumTeams, Value *NumThreads,
    Value *HostPtr, ArrayRef<Value *> KernelArgs) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Builder.restoreIP(AllocaIP);
  auto *KernelArgsPtr =
      Builder.CreateAlloca(OpenMPIRBuilder::KernelArgs, nullptr, "kernel_args");
  Builder.restoreIP(Loc.IP);

  for (unsigned I = 0, Size = KernelArgs.size(); I != Size; ++I) {
    llvm::Value *Arg =
        Builder.CreateStructGEP(OpenMPIRBuilder::KernelArgs, KernelArgsPtr, I);
    Builder.CreateAlignedStore(
        KernelArgs[I], Arg,
        M.getDataLayout().getPrefTypeAlign(KernelArgs[I]->getType()));
  }

  SmallVector<Value *> OffloadingArgs{Ident,      DeviceID, NumTeams,
                                      NumThreads, HostPtr,  KernelArgsPtr};

  Return = Builder.CreateCall(
      getOrCreateRuntimeFunction(M, OMPRTL___tgt_target_kernel),
      OffloadingArgs);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitKernelLaunch(
    const LocationDescription &Loc, Function *OutlinedFn, Value *OutlinedFnID,
    EmitFallbackCallbackTy emitTargetCallFallbackCB, TargetKernelArgs &Args,
    Value *DeviceID, Value *RTLoc, InsertPointTy AllocaIP) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Builder.restoreIP(Loc.IP);
  // On top of the arrays that were filled up, the target offloading call
  // takes as arguments the device id as well as the host pointer. The host
  // pointer is used by the runtime library to identify the current target
  // region, so it only has to be unique and not necessarily point to
  // anything. It could be the pointer to the outlined function that
  // implements the target region, but we aren't using that so that the
  // compiler doesn't need to keep that, and could therefore inline the host
  // function if proven worthwhile during optimization.

  // From this point on, we need to have an ID of the target region defined.
  assert(OutlinedFnID && "Invalid outlined function ID!");
  (void)OutlinedFnID;

  // Return value of the runtime offloading call.
  Value *Return = nullptr;

  // Arguments for the target kernel.
  SmallVector<Value *> ArgsVector;
  getKernelArgsVector(Args, Builder, ArgsVector);

  // The target region is an outlined function launched by the runtime
  // via calls to __tgt_target_kernel().
  //
  // Note that on the host and CPU targets, the runtime implementation of
  // these calls simply call the outlined function without forking threads.
  // The outlined functions themselves have runtime calls to
  // __kmpc_fork_teams() and __kmpc_fork() for this purpose, codegen'd by
  // the compiler in emitTeamsCall() and emitParallelCall().
  //
  // In contrast, on the NVPTX target, the implementation of
  // __tgt_target_teams() launches a GPU kernel with the requested number
  // of teams and threads so no additional calls to the runtime are required.
  // Check the error code and execute the host version if required.
  Builder.restoreIP(emitTargetKernel(Builder, AllocaIP, Return, RTLoc, DeviceID,
                                     Args.NumTeams, Args.NumThreads,
                                     OutlinedFnID, ArgsVector));

  BasicBlock *OffloadFailedBlock =
      BasicBlock::Create(Builder.getContext(), "omp_offload.failed");
  BasicBlock *OffloadContBlock =
      BasicBlock::Create(Builder.getContext(), "omp_offload.cont");
  Value *Failed = Builder.CreateIsNotNull(Return);
  Builder.CreateCondBr(Failed, OffloadFailedBlock, OffloadContBlock);

  auto CurFn = Builder.GetInsertBlock()->getParent();
  emitBlock(OffloadFailedBlock, CurFn);
  Builder.restoreIP(emitTargetCallFallbackCB(Builder.saveIP()));
  emitBranch(OffloadContBlock);
  emitBlock(OffloadContBlock, CurFn, /*IsFinished=*/true);
  return Builder.saveIP();
}

void OpenMPIRBuilder::emitCancelationCheckImpl(Value *CancelFlag,
                                               omp::Directive CanceledDirective,
                                               FinalizeCallbackTy ExitCB) {
  assert(isLastFinalizationInfoCancellable(CanceledDirective) &&
         "Unexpected cancellation!");

  // For a cancel barrier we create two new blocks.
  BasicBlock *BB = Builder.GetInsertBlock();
  BasicBlock *NonCancellationBlock;
  if (Builder.GetInsertPoint() == BB->end()) {
    // TODO: This branch will not be needed once we moved to the
    // OpenMPIRBuilder codegen completely.
    NonCancellationBlock = BasicBlock::Create(
        BB->getContext(), BB->getName() + ".cont", BB->getParent());
  } else {
    NonCancellationBlock = SplitBlock(BB, &*Builder.GetInsertPoint());
    BB->getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(BB);
  }
  BasicBlock *CancellationBlock = BasicBlock::Create(
      BB->getContext(), BB->getName() + ".cncl", BB->getParent());

  // Jump to them based on the return value.
  Value *Cmp = Builder.CreateIsNull(CancelFlag);
  Builder.CreateCondBr(Cmp, NonCancellationBlock, CancellationBlock,
                       /* TODO weight */ nullptr, nullptr);

  // From the cancellation block we finalize all variables and go to the
  // post finalization block that is known to the FiniCB callback.
  Builder.SetInsertPoint(CancellationBlock);
  if (ExitCB)
    ExitCB(Builder.saveIP());
  auto &FI = FinalizationStack.back();
  FI.FiniCB(Builder.saveIP());

  // The continuation block is where code generation continues.
  Builder.SetInsertPoint(NonCancellationBlock, NonCancellationBlock->begin());
}

IRBuilder<>::InsertPoint OpenMPIRBuilder::createParallel(
    const LocationDescription &Loc, InsertPointTy OuterAllocaIP,
    BodyGenCallbackTy BodyGenCB, PrivatizeCallbackTy PrivCB,
    FinalizeCallbackTy FiniCB, Value *IfCondition, Value *NumThreads,
    omp::ProcBindKind ProcBind, bool IsCancellable) {
  assert(!isConflictIP(Loc.IP, OuterAllocaIP) && "IPs must not be ambiguous");

  if (!updateToLocation(Loc))
    return Loc.IP;

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = getOrCreateThreadID(Ident);

  if (NumThreads) {
    // Build call __kmpc_push_num_threads(&Ident, global_tid, num_threads)
    Value *Args[] = {
        Ident, ThreadID,
        Builder.CreateIntCast(NumThreads, Int32, /*isSigned*/ false)};
    Builder.CreateCall(
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_push_num_threads), Args);
  }

  if (ProcBind != OMP_PROC_BIND_default) {
    // Build call __kmpc_push_proc_bind(&Ident, global_tid, proc_bind)
    Value *Args[] = {
        Ident, ThreadID,
        ConstantInt::get(Int32, unsigned(ProcBind), /*isSigned=*/true)};
    Builder.CreateCall(
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_push_proc_bind), Args);
  }

  BasicBlock *InsertBB = Builder.GetInsertBlock();
  Function *OuterFn = InsertBB->getParent();

  // Save the outer alloca block because the insertion iterator may get
  // invalidated and we still need this later.
  BasicBlock *OuterAllocaBlock = OuterAllocaIP.getBlock();

  // Vector to remember instructions we used only during the modeling but which
  // we want to delete at the end.
  SmallVector<Instruction *, 4> ToBeDeleted;

  // Change the location to the outer alloca insertion point to create and
  // initialize the allocas we pass into the parallel region.
  Builder.restoreIP(OuterAllocaIP);
  AllocaInst *TIDAddr = Builder.CreateAlloca(Int32, nullptr, "tid.addr");
  AllocaInst *ZeroAddr = Builder.CreateAlloca(Int32, nullptr, "zero.addr");

  // We only need TIDAddr and ZeroAddr for modeling purposes to get the
  // associated arguments in the outlined function, so we delete them later.
  ToBeDeleted.push_back(TIDAddr);
  ToBeDeleted.push_back(ZeroAddr);

  // Create an artificial insertion point that will also ensure the blocks we
  // are about to split are not degenerated.
  auto *UI = new UnreachableInst(Builder.getContext(), InsertBB);

  BasicBlock *EntryBB = UI->getParent();
  BasicBlock *PRegEntryBB = EntryBB->splitBasicBlock(UI, "omp.par.entry");
  BasicBlock *PRegBodyBB = PRegEntryBB->splitBasicBlock(UI, "omp.par.region");
  BasicBlock *PRegPreFiniBB =
      PRegBodyBB->splitBasicBlock(UI, "omp.par.pre_finalize");
  BasicBlock *PRegExitBB = PRegPreFiniBB->splitBasicBlock(UI, "omp.par.exit");

  auto FiniCBWrapper = [&](InsertPointTy IP) {
    // Hide "open-ended" blocks from the given FiniCB by setting the right jump
    // target to the region exit block.
    if (IP.getBlock()->end() == IP.getPoint()) {
      IRBuilder<>::InsertPointGuard IPG(Builder);
      Builder.restoreIP(IP);
      Instruction *I = Builder.CreateBr(PRegExitBB);
      IP = InsertPointTy(I->getParent(), I->getIterator());
    }
    assert(IP.getBlock()->getTerminator()->getNumSuccessors() == 1 &&
           IP.getBlock()->getTerminator()->getSuccessor(0) == PRegExitBB &&
           "Unexpected insertion point for finalization call!");
    return FiniCB(IP);
  };

  FinalizationStack.push_back({FiniCBWrapper, OMPD_parallel, IsCancellable});

  // Generate the privatization allocas in the block that will become the entry
  // of the outlined function.
  Builder.SetInsertPoint(PRegEntryBB->getTerminator());
  InsertPointTy InnerAllocaIP = Builder.saveIP();

  AllocaInst *PrivTIDAddr =
      Builder.CreateAlloca(Int32, nullptr, "tid.addr.local");
  Instruction *PrivTID = Builder.CreateLoad(Int32, PrivTIDAddr, "tid");

  // Add some fake uses for OpenMP provided arguments.
  ToBeDeleted.push_back(Builder.CreateLoad(Int32, TIDAddr, "tid.addr.use"));
  Instruction *ZeroAddrUse =
      Builder.CreateLoad(Int32, ZeroAddr, "zero.addr.use");
  ToBeDeleted.push_back(ZeroAddrUse);

  // EntryBB
  //   |
  //   V
  // PRegionEntryBB         <- Privatization allocas are placed here.
  //   |
  //   V
  // PRegionBodyBB          <- BodeGen is invoked here.
  //   |
  //   V
  // PRegPreFiniBB          <- The block we will start finalization from.
  //   |
  //   V
  // PRegionExitBB          <- A common exit to simplify block collection.
  //

  LLVM_DEBUG(dbgs() << "Before body codegen: " << *OuterFn << "\n");

  // Let the caller create the body.
  assert(BodyGenCB && "Expected body generation callback!");
  InsertPointTy CodeGenIP(PRegBodyBB, PRegBodyBB->begin());
  BodyGenCB(InnerAllocaIP, CodeGenIP);

  LLVM_DEBUG(dbgs() << "After  body codegen: " << *OuterFn << "\n");
  FunctionCallee RTLFn;
  if (IfCondition)
    RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_fork_call_if);
  else
    RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_fork_call);

  if (auto *F = dyn_cast<llvm::Function>(RTLFn.getCallee())) {
    if (!F->hasMetadata(llvm::LLVMContext::MD_callback)) {
      llvm::LLVMContext &Ctx = F->getContext();
      MDBuilder MDB(Ctx);
      // Annotate the callback behavior of the __kmpc_fork_call:
      //  - The callback callee is argument number 2 (microtask).
      //  - The first two arguments of the callback callee are unknown (-1).
      //  - All variadic arguments to the __kmpc_fork_call are passed to the
      //    callback callee.
      F->addMetadata(
          llvm::LLVMContext::MD_callback,
          *llvm::MDNode::get(
              Ctx, {MDB.createCallbackEncoding(2, {-1, -1},
                                               /* VarArgsArePassed */ true)}));
    }
  }

  OutlineInfo OI;
  OI.PostOutlineCB = [=](Function &OutlinedFn) {
    // Add some known attributes.
    OutlinedFn.addParamAttr(0, Attribute::NoAlias);
    OutlinedFn.addParamAttr(1, Attribute::NoAlias);
    OutlinedFn.addFnAttr(Attribute::NoUnwind);
    OutlinedFn.addFnAttr(Attribute::NoRecurse);

    assert(OutlinedFn.arg_size() >= 2 &&
           "Expected at least tid and bounded tid as arguments");
    unsigned NumCapturedVars =
        OutlinedFn.arg_size() - /* tid & bounded tid */ 2;

    CallInst *CI = cast<CallInst>(OutlinedFn.user_back());
    CI->getParent()->setName("omp_parallel");
    Builder.SetInsertPoint(CI);

    // Build call __kmpc_fork_call[_if](Ident, n, microtask, var1, .., varn);
    Value *ForkCallArgs[] = {
        Ident, Builder.getInt32(NumCapturedVars),
        Builder.CreateBitCast(&OutlinedFn, ParallelTaskPtr)};

    SmallVector<Value *, 16> RealArgs;
    RealArgs.append(std::begin(ForkCallArgs), std::end(ForkCallArgs));
    if (IfCondition) {
      Value *Cond = Builder.CreateSExtOrTrunc(IfCondition,
                                              Type::getInt32Ty(M.getContext()));
      RealArgs.push_back(Cond);
    }
    RealArgs.append(CI->arg_begin() + /* tid & bound tid */ 2, CI->arg_end());

    // __kmpc_fork_call_if always expects a void ptr as the last argument
    // If there are no arguments, pass a null pointer.
    auto PtrTy = Type::getInt8PtrTy(M.getContext());
    if (IfCondition && NumCapturedVars == 0) {
      llvm::Value *Void = ConstantPointerNull::get(PtrTy);
      RealArgs.push_back(Void);
    }
    if (IfCondition && RealArgs.back()->getType() != PtrTy)
      RealArgs.back() = Builder.CreateBitCast(RealArgs.back(), PtrTy);

    Builder.CreateCall(RTLFn, RealArgs);

    LLVM_DEBUG(dbgs() << "With fork_call placed: "
                      << *Builder.GetInsertBlock()->getParent() << "\n");

    InsertPointTy ExitIP(PRegExitBB, PRegExitBB->end());

    // Initialize the local TID stack location with the argument value.
    Builder.SetInsertPoint(PrivTID);
    Function::arg_iterator OutlinedAI = OutlinedFn.arg_begin();
    Builder.CreateStore(Builder.CreateLoad(Int32, OutlinedAI), PrivTIDAddr);

    CI->eraseFromParent();

    for (Instruction *I : ToBeDeleted)
      I->eraseFromParent();
  };

  // Adjust the finalization stack, verify the adjustment, and call the
  // finalize function a last time to finalize values between the pre-fini
  // block and the exit block if we left the parallel "the normal way".
  auto FiniInfo = FinalizationStack.pop_back_val();
  (void)FiniInfo;
  assert(FiniInfo.DK == OMPD_parallel &&
         "Unexpected finalization stack state!");

  Instruction *PRegPreFiniTI = PRegPreFiniBB->getTerminator();

  InsertPointTy PreFiniIP(PRegPreFiniBB, PRegPreFiniTI->getIterator());
  FiniCB(PreFiniIP);

  OI.OuterAllocaBB = OuterAllocaBlock;
  OI.EntryBB = PRegEntryBB;
  OI.ExitBB = PRegExitBB;

  SmallPtrSet<BasicBlock *, 32> ParallelRegionBlockSet;
  SmallVector<BasicBlock *, 32> Blocks;
  OI.collectBlocks(ParallelRegionBlockSet, Blocks);

  // Ensure a single exit node for the outlined region by creating one.
  // We might have multiple incoming edges to the exit now due to finalizations,
  // e.g., cancel calls that cause the control flow to leave the region.
  BasicBlock *PRegOutlinedExitBB = PRegExitBB;
  PRegExitBB = SplitBlock(PRegExitBB, &*PRegExitBB->getFirstInsertionPt());
  PRegOutlinedExitBB->setName("omp.par.outlined.exit");
  Blocks.push_back(PRegOutlinedExitBB);

  CodeExtractorAnalysisCache CEAC(*OuterFn);
  CodeExtractor Extractor(Blocks, /* DominatorTree */ nullptr,
                          /* AggregateArgs */ false,
                          /* BlockFrequencyInfo */ nullptr,
                          /* BranchProbabilityInfo */ nullptr,
                          /* AssumptionCache */ nullptr,
                          /* AllowVarArgs */ true,
                          /* AllowAlloca */ true,
                          /* AllocationBlock */ OuterAllocaBlock,
                          /* Suffix */ ".omp_par");

  // Find inputs to, outputs from the code region.
  BasicBlock *CommonExit = nullptr;
  SetVector<Value *> Inputs, Outputs, SinkingCands, HoistingCands;
  Extractor.findAllocas(CEAC, SinkingCands, HoistingCands, CommonExit);
  Extractor.findInputsOutputs(Inputs, Outputs, SinkingCands);

  LLVM_DEBUG(dbgs() << "Before privatization: " << *OuterFn << "\n");

  FunctionCallee TIDRTLFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_global_thread_num);

  auto PrivHelper = [&](Value &V) {
    if (&V == TIDAddr || &V == ZeroAddr) {
      OI.ExcludeArgsFromAggregate.push_back(&V);
      return;
    }

    SetVector<Use *> Uses;
    for (Use &U : V.uses())
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (ParallelRegionBlockSet.count(UserI->getParent()))
          Uses.insert(&U);

    // __kmpc_fork_call expects extra arguments as pointers. If the input
    // already has a pointer type, everything is fine. Otherwise, store the
    // value onto stack and load it back inside the to-be-outlined region. This
    // will ensure only the pointer will be passed to the function.
    // FIXME: if there are more than 15 trailing arguments, they must be
    // additionally packed in a struct.
    Value *Inner = &V;
    if (!V.getType()->isPointerTy()) {
      IRBuilder<>::InsertPointGuard Guard(Builder);
      LLVM_DEBUG(llvm::dbgs() << "Forwarding input as pointer: " << V << "\n");

      Builder.restoreIP(OuterAllocaIP);
      Value *Ptr =
          Builder.CreateAlloca(V.getType(), nullptr, V.getName() + ".reloaded");

      // Store to stack at end of the block that currently branches to the entry
      // block of the to-be-outlined region.
      Builder.SetInsertPoint(InsertBB,
                             InsertBB->getTerminator()->getIterator());
      Builder.CreateStore(&V, Ptr);

      // Load back next to allocations in the to-be-outlined region.
      Builder.restoreIP(InnerAllocaIP);
      Inner = Builder.CreateLoad(V.getType(), Ptr);
    }

    Value *ReplacementValue = nullptr;
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && CI->getCalledFunction() == TIDRTLFn.getCallee()) {
      ReplacementValue = PrivTID;
    } else {
      Builder.restoreIP(
          PrivCB(InnerAllocaIP, Builder.saveIP(), V, *Inner, ReplacementValue));
      assert(ReplacementValue &&
             "Expected copy/create callback to set replacement value!");
      if (ReplacementValue == &V)
        return;
    }

    for (Use *UPtr : Uses)
      UPtr->set(ReplacementValue);
  };

  // Reset the inner alloca insertion as it will be used for loading the values
  // wrapped into pointers before passing them into the to-be-outlined region.
  // Configure it to insert immediately after the fake use of zero address so
  // that they are available in the generated body and so that the
  // OpenMP-related values (thread ID and zero address pointers) remain leading
  // in the argument list.
  InnerAllocaIP = IRBuilder<>::InsertPoint(
      ZeroAddrUse->getParent(), ZeroAddrUse->getNextNode()->getIterator());

  // Reset the outer alloca insertion point to the entry of the relevant block
  // in case it was invalidated.
  OuterAllocaIP = IRBuilder<>::InsertPoint(
      OuterAllocaBlock, OuterAllocaBlock->getFirstInsertionPt());

  for (Value *Input : Inputs) {
    LLVM_DEBUG(dbgs() << "Captured input: " << *Input << "\n");
    PrivHelper(*Input);
  }
  LLVM_DEBUG({
    for (Value *Output : Outputs)
      LLVM_DEBUG(dbgs() << "Captured output: " << *Output << "\n");
  });
  assert(Outputs.empty() &&
         "OpenMP outlining should not produce live-out values!");

  LLVM_DEBUG(dbgs() << "After  privatization: " << *OuterFn << "\n");
  LLVM_DEBUG({
    for (auto *BB : Blocks)
      dbgs() << " PBR: " << BB->getName() << "\n";
  });

  // Register the outlined info.
  addOutlineInfo(std::move(OI));

  InsertPointTy AfterIP(UI->getParent(), UI->getParent()->end());
  UI->eraseFromParent();

  return AfterIP;
}

void OpenMPIRBuilder::emitFlush(const LocationDescription &Loc) {
  // Build call void __kmpc_flush(ident_t *loc)
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Args[] = {getOrCreateIdent(SrcLocStr, SrcLocStrSize)};

  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_flush), Args);
}

void OpenMPIRBuilder::createFlush(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitFlush(Loc);
}

void OpenMPIRBuilder::emitTaskwaitImpl(const LocationDescription &Loc) {
  // Build call kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32
  // global_tid);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident)};

  // Ignore return result until untied tasks are supported.
  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_taskwait),
                     Args);
}

void OpenMPIRBuilder::createTaskwait(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitTaskwaitImpl(Loc);
}

void OpenMPIRBuilder::emitTaskyieldImpl(const LocationDescription &Loc) {
  // Build call __kmpc_omp_taskyield(loc, thread_id, 0);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Constant *I32Null = ConstantInt::getNullValue(Int32);
  Value *Args[] = {Ident, getOrCreateThreadID(Ident), I32Null};

  Builder.CreateCall(getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_taskyield),
                     Args);
}

void OpenMPIRBuilder::createTaskyield(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;
  emitTaskyieldImpl(Loc);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createTask(const LocationDescription &Loc,
                            InsertPointTy AllocaIP, BodyGenCallbackTy BodyGenCB,
                            bool Tied, Value *Final, Value *IfCondition,
                            SmallVector<DependData> Dependencies) {
  if (!updateToLocation(Loc))
    return InsertPointTy();

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  // The current basic block is split into four basic blocks. After outlining,
  // they will be mapped as follows:
  // ```
  // def current_fn() {
  //   current_basic_block:
  //     br label %task.exit
  //   task.exit:
  //     ; instructions after task
  // }
  // def outlined_fn() {
  //   task.alloca:
  //     br label %task.body
  //   task.body:
  //     ret void
  // }
  // ```
  BasicBlock *TaskExitBB = splitBB(Builder, /*CreateBranch=*/true, "task.exit");
  BasicBlock *TaskBodyBB = splitBB(Builder, /*CreateBranch=*/true, "task.body");
  BasicBlock *TaskAllocaBB =
      splitBB(Builder, /*CreateBranch=*/true, "task.alloca");

  OutlineInfo OI;
  OI.EntryBB = TaskAllocaBB;
  OI.OuterAllocaBB = AllocaIP.getBlock();
  OI.ExitBB = TaskExitBB;
  OI.PostOutlineCB = [this, Ident, Tied, Final, IfCondition,
                      Dependencies](Function &OutlinedFn) {
    // The input IR here looks like the following-
    // ```
    // func @current_fn() {
    //   outlined_fn(%args)
    // }
    // func @outlined_fn(%args) { ... }
    // ```
    //
    // This is changed to the following-
    //
    // ```
    // func @current_fn() {
    //   runtime_call(..., wrapper_fn, ...)
    // }
    // func @wrapper_fn(..., %args) {
    //   outlined_fn(%args)
    // }
    // func @outlined_fn(%args) { ... }
    // ```

    // The stale call instruction will be replaced with a new call instruction
    // for runtime call with a wrapper function.
    assert(OutlinedFn.getNumUses() == 1 &&
           "there must be a single user for the outlined function");
    CallInst *StaleCI = cast<CallInst>(OutlinedFn.user_back());

    // HasShareds is true if any variables are captured in the outlined region,
    // false otherwise.
    bool HasShareds = StaleCI->arg_size() > 0;
    Builder.SetInsertPoint(StaleCI);

    // Gather the arguments for emitting the runtime call for
    // @__kmpc_omp_task_alloc
    Function *TaskAllocFn =
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_alloc);

    // Arguments - `loc_ref` (Ident) and `gtid` (ThreadID)
    // call.
    Value *ThreadID = getOrCreateThreadID(Ident);

    // Argument - `flags`
    // Task is tied iff (Flags & 1) == 1.
    // Task is untied iff (Flags & 1) == 0.
    // Task is final iff (Flags & 2) == 2.
    // Task is not final iff (Flags & 2) == 0.
    // TODO: Handle the other flags.
    Value *Flags = Builder.getInt32(Tied);
    if (Final) {
      Value *FinalFlag =
          Builder.CreateSelect(Final, Builder.getInt32(2), Builder.getInt32(0));
      Flags = Builder.CreateOr(FinalFlag, Flags);
    }

    // Argument - `sizeof_kmp_task_t` (TaskSize)
    // Tasksize refers to the size in bytes of kmp_task_t data structure
    // including private vars accessed in task.
    // TODO: add kmp_task_t_with_privates (privates)
    Value *TaskSize = Builder.getInt64(
        divideCeil(M.getDataLayout().getTypeSizeInBits(Task), 8));

    // Argument - `sizeof_shareds` (SharedsSize)
    // SharedsSize refers to the shareds array size in the kmp_task_t data
    // structure.
    Value *SharedsSize = Builder.getInt64(0);
    if (HasShareds) {
      AllocaInst *ArgStructAlloca =
          dyn_cast<AllocaInst>(StaleCI->getArgOperand(0));
      assert(ArgStructAlloca &&
             "Unable to find the alloca instruction corresponding to arguments "
             "for extracted function");
      StructType *ArgStructType =
          dyn_cast<StructType>(ArgStructAlloca->getAllocatedType());
      assert(ArgStructType && "Unable to find struct type corresponding to "
                              "arguments for extracted function");
      SharedsSize =
          Builder.getInt64(M.getDataLayout().getTypeStoreSize(ArgStructType));
    }

    // Argument - task_entry (the wrapper function)
    // If the outlined function has some captured variables (i.e. HasShareds is
    // true), then the wrapper function will have an additional argument (the
    // struct containing captured variables). Otherwise, no such argument will
    // be present.
    SmallVector<Type *> WrapperArgTys{Builder.getInt32Ty()};
    if (HasShareds)
      WrapperArgTys.push_back(OutlinedFn.getArg(0)->getType());
    FunctionCallee WrapperFuncVal = M.getOrInsertFunction(
        (Twine(OutlinedFn.getName()) + ".wrapper").str(),
        FunctionType::get(Builder.getInt32Ty(), WrapperArgTys, false));
    Function *WrapperFunc = dyn_cast<Function>(WrapperFuncVal.getCallee());

    // Emit the @__kmpc_omp_task_alloc runtime call
    // The runtime call returns a pointer to an area where the task captured
    // variables must be copied before the task is run (TaskData)
    CallInst *TaskData = Builder.CreateCall(
        TaskAllocFn, {/*loc_ref=*/Ident, /*gtid=*/ThreadID, /*flags=*/Flags,
                      /*sizeof_task=*/TaskSize, /*sizeof_shared=*/SharedsSize,
                      /*task_func=*/WrapperFunc});

    // Copy the arguments for outlined function
    if (HasShareds) {
      Value *Shareds = StaleCI->getArgOperand(0);
      Align Alignment = TaskData->getPointerAlignment(M.getDataLayout());
      Value *TaskShareds = Builder.CreateLoad(VoidPtr, TaskData);
      Builder.CreateMemCpy(TaskShareds, Alignment, Shareds, Alignment,
                           SharedsSize);
    }

    Value *DepArrayPtr = nullptr;
    if (Dependencies.size()) {
      InsertPointTy OldIP = Builder.saveIP();
      Builder.SetInsertPoint(
          &OldIP.getBlock()->getParent()->getEntryBlock().back());

      Type *DepArrayTy = ArrayType::get(DependInfo, Dependencies.size());
      Value *DepArray =
          Builder.CreateAlloca(DepArrayTy, nullptr, ".dep.arr.addr");

      unsigned P = 0;
      for (const DependData &Dep : Dependencies) {
        Value *Base =
            Builder.CreateConstInBoundsGEP2_64(DepArrayTy, DepArray, 0, P);
        // Store the pointer to the variable
        Value *Addr = Builder.CreateStructGEP(
            DependInfo, Base,
            static_cast<unsigned int>(RTLDependInfoFields::BaseAddr));
        Value *DepValPtr =
            Builder.CreatePtrToInt(Dep.DepVal, Builder.getInt64Ty());
        Builder.CreateStore(DepValPtr, Addr);
        // Store the size of the variable
        Value *Size = Builder.CreateStructGEP(
            DependInfo, Base,
            static_cast<unsigned int>(RTLDependInfoFields::Len));
        Builder.CreateStore(Builder.getInt64(M.getDataLayout().getTypeStoreSize(
                                Dep.DepValueType)),
                            Size);
        // Store the dependency kind
        Value *Flags = Builder.CreateStructGEP(
            DependInfo, Base,
            static_cast<unsigned int>(RTLDependInfoFields::Flags));
        Builder.CreateStore(
            ConstantInt::get(Builder.getInt8Ty(),
                             static_cast<unsigned int>(Dep.DepKind)),
            Flags);
        ++P;
      }

      DepArrayPtr = Builder.CreateBitCast(DepArray, Builder.getInt8PtrTy());
      Builder.restoreIP(OldIP);
    }

    // In the presence of the `if` clause, the following IR is generated:
    //    ...
    //    %data = call @__kmpc_omp_task_alloc(...)
    //    br i1 %if_condition, label %then, label %else
    //  then:
    //    call @__kmpc_omp_task(...)
    //    br label %exit
    //  else:
    //    call @__kmpc_omp_task_begin_if0(...)
    //    call @wrapper_fn(...)
    //    call @__kmpc_omp_task_complete_if0(...)
    //    br label %exit
    //  exit:
    //    ...
    if (IfCondition) {
      // `SplitBlockAndInsertIfThenElse` requires the block to have a
      // terminator.
      BasicBlock *NewBasicBlock =
          splitBB(Builder, /*CreateBranch=*/true, "if.end");
      Instruction *IfTerminator =
          NewBasicBlock->getSinglePredecessor()->getTerminator();
      Instruction *ThenTI = IfTerminator, *ElseTI = nullptr;
      Builder.SetInsertPoint(IfTerminator);
      SplitBlockAndInsertIfThenElse(IfCondition, IfTerminator, &ThenTI,
                                    &ElseTI);
      Builder.SetInsertPoint(ElseTI);
      Function *TaskBeginFn =
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_begin_if0);
      Function *TaskCompleteFn =
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_complete_if0);
      Builder.CreateCall(TaskBeginFn, {Ident, ThreadID, TaskData});
      if (HasShareds)
        Builder.CreateCall(WrapperFunc, {ThreadID, TaskData});
      else
        Builder.CreateCall(WrapperFunc, {ThreadID});
      Builder.CreateCall(TaskCompleteFn, {Ident, ThreadID, TaskData});
      Builder.SetInsertPoint(ThenTI);
    }

    if (Dependencies.size()) {
      Function *TaskFn =
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task_with_deps);
      Builder.CreateCall(
          TaskFn,
          {Ident, ThreadID, TaskData, Builder.getInt32(Dependencies.size()),
           DepArrayPtr, ConstantInt::get(Builder.getInt32Ty(), 0),
           ConstantPointerNull::get(Type::getInt8PtrTy(M.getContext()))});

    } else {
      // Emit the @__kmpc_omp_task runtime call to spawn the task
      Function *TaskFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_omp_task);
      Builder.CreateCall(TaskFn, {Ident, ThreadID, TaskData});
    }

    StaleCI->eraseFromParent();

    // Emit the body for wrapper function
    BasicBlock *WrapperEntryBB =
        BasicBlock::Create(M.getContext(), "", WrapperFunc);
    Builder.SetInsertPoint(WrapperEntryBB);
    if (HasShareds) {
      llvm::Value *Shareds =
          Builder.CreateLoad(VoidPtr, WrapperFunc->getArg(1));
      Builder.CreateCall(&OutlinedFn, {Shareds});
    } else {
      Builder.CreateCall(&OutlinedFn);
    }
    Builder.CreateRet(Builder.getInt32(0));
  };

  addOutlineInfo(std::move(OI));

  InsertPointTy TaskAllocaIP =
      InsertPointTy(TaskAllocaBB, TaskAllocaBB->begin());
  InsertPointTy TaskBodyIP = InsertPointTy(TaskBodyBB, TaskBodyBB->begin());
  BodyGenCB(TaskAllocaIP, TaskBodyIP);
  Builder.SetInsertPoint(TaskExitBB, TaskExitBB->begin());

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createTaskgroup(const LocationDescription &Loc,
                                 InsertPointTy AllocaIP,
                                 BodyGenCallbackTy BodyGenCB) {
  if (!updateToLocation(Loc))
    return InsertPointTy();

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadID = getOrCreateThreadID(Ident);

  // Emit the @__kmpc_taskgroup runtime call to start the taskgroup
  Function *TaskgroupFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_taskgroup);
  Builder.CreateCall(TaskgroupFn, {Ident, ThreadID});

  BasicBlock *TaskgroupExitBB = splitBB(Builder, true, "taskgroup.exit");
  BodyGenCB(AllocaIP, Builder.saveIP());

  Builder.SetInsertPoint(TaskgroupExitBB);
  // Emit the @__kmpc_end_taskgroup runtime call to end the taskgroup
  Function *EndTaskgroupFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_taskgroup);
  Builder.CreateCall(EndTaskgroupFn, {Ident, ThreadID});

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createSections(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<StorableBodyGenCallbackTy> SectionCBs, PrivatizeCallbackTy PrivCB,
    FinalizeCallbackTy FiniCB, bool IsCancellable, bool IsNowait) {
  assert(!isConflictIP(AllocaIP, Loc.IP) && "Dedicated IP allocas required");

  if (!updateToLocation(Loc))
    return Loc.IP;

  auto FiniCBWrapper = [&](InsertPointTy IP) {
    if (IP.getBlock()->end() != IP.getPoint())
      return FiniCB(IP);
    // This must be done otherwise any nested constructs using FinalizeOMPRegion
    // will fail because that function requires the Finalization Basic Block to
    // have a terminator, which is already removed by EmitOMPRegionBody.
    // IP is currently at cancelation block.
    // We need to backtrack to the condition block to fetch
    // the exit block and create a branch from cancelation
    // to exit block.
    IRBuilder<>::InsertPointGuard IPG(Builder);
    Builder.restoreIP(IP);
    auto *CaseBB = IP.getBlock()->getSinglePredecessor();
    auto *CondBB = CaseBB->getSinglePredecessor()->getSinglePredecessor();
    auto *ExitBB = CondBB->getTerminator()->getSuccessor(1);
    Instruction *I = Builder.CreateBr(ExitBB);
    IP = InsertPointTy(I->getParent(), I->getIterator());
    return FiniCB(IP);
  };

  FinalizationStack.push_back({FiniCBWrapper, OMPD_sections, IsCancellable});

  // Each section is emitted as a switch case
  // Each finalization callback is handled from clang.EmitOMPSectionDirective()
  // -> OMP.createSection() which generates the IR for each section
  // Iterate through all sections and emit a switch construct:
  // switch (IV) {
  //   case 0:
  //     <SectionStmt[0]>;
  //     break;
  // ...
  //   case <NumSection> - 1:
  //     <SectionStmt[<NumSection> - 1]>;
  //     break;
  // }
  // ...
  // section_loop.after:
  // <FiniCB>;
  auto LoopBodyGenCB = [&](InsertPointTy CodeGenIP, Value *IndVar) {
    Builder.restoreIP(CodeGenIP);
    BasicBlock *Continue =
        splitBBWithSuffix(Builder, /*CreateBranch=*/false, ".sections.after");
    Function *CurFn = Continue->getParent();
    SwitchInst *SwitchStmt = Builder.CreateSwitch(IndVar, Continue);

    unsigned CaseNumber = 0;
    for (auto SectionCB : SectionCBs) {
      BasicBlock *CaseBB = BasicBlock::Create(
          M.getContext(), "omp_section_loop.body.case", CurFn, Continue);
      SwitchStmt->addCase(Builder.getInt32(CaseNumber), CaseBB);
      Builder.SetInsertPoint(CaseBB);
      BranchInst *CaseEndBr = Builder.CreateBr(Continue);
      SectionCB(InsertPointTy(),
                {CaseEndBr->getParent(), CaseEndBr->getIterator()});
      CaseNumber++;
    }
    // remove the existing terminator from body BB since there can be no
    // terminators after switch/case
  };
  // Loop body ends here
  // LowerBound, UpperBound, and STride for createCanonicalLoop
  Type *I32Ty = Type::getInt32Ty(M.getContext());
  Value *LB = ConstantInt::get(I32Ty, 0);
  Value *UB = ConstantInt::get(I32Ty, SectionCBs.size());
  Value *ST = ConstantInt::get(I32Ty, 1);
  llvm::CanonicalLoopInfo *LoopInfo = createCanonicalLoop(
      Loc, LoopBodyGenCB, LB, UB, ST, true, false, AllocaIP, "section_loop");
  InsertPointTy AfterIP =
      applyStaticWorkshareLoop(Loc.DL, LoopInfo, AllocaIP, !IsNowait);

  // Apply the finalization callback in LoopAfterBB
  auto FiniInfo = FinalizationStack.pop_back_val();
  assert(FiniInfo.DK == OMPD_sections &&
         "Unexpected finalization stack state!");
  if (FinalizeCallbackTy &CB = FiniInfo.FiniCB) {
    Builder.restoreIP(AfterIP);
    BasicBlock *FiniBB =
        splitBBWithSuffix(Builder, /*CreateBranch=*/true, "sections.fini");
    CB(Builder.saveIP());
    AfterIP = {FiniBB, FiniBB->begin()};
  }

  return AfterIP;
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createSection(const LocationDescription &Loc,
                               BodyGenCallbackTy BodyGenCB,
                               FinalizeCallbackTy FiniCB) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  auto FiniCBWrapper = [&](InsertPointTy IP) {
    if (IP.getBlock()->end() != IP.getPoint())
      return FiniCB(IP);
    // This must be done otherwise any nested constructs using FinalizeOMPRegion
    // will fail because that function requires the Finalization Basic Block to
    // have a terminator, which is already removed by EmitOMPRegionBody.
    // IP is currently at cancelation block.
    // We need to backtrack to the condition block to fetch
    // the exit block and create a branch from cancelation
    // to exit block.
    IRBuilder<>::InsertPointGuard IPG(Builder);
    Builder.restoreIP(IP);
    auto *CaseBB = Loc.IP.getBlock();
    auto *CondBB = CaseBB->getSinglePredecessor()->getSinglePredecessor();
    auto *ExitBB = CondBB->getTerminator()->getSuccessor(1);
    Instruction *I = Builder.CreateBr(ExitBB);
    IP = InsertPointTy(I->getParent(), I->getIterator());
    return FiniCB(IP);
  };

  Directive OMPD = Directive::OMPD_sections;
  // Since we are using Finalization Callback here, HasFinalize
  // and IsCancellable have to be true
  return EmitOMPInlinedRegion(OMPD, nullptr, nullptr, BodyGenCB, FiniCBWrapper,
                              /*Conditional*/ false, /*hasFinalize*/ true,
                              /*IsCancellable*/ true);
}

/// Create a function with a unique name and a "void (i8*, i8*)" signature in
/// the given module and return it.
Function *getFreshReductionFunc(Module &M) {
  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *Int8PtrTy = Type::getInt8PtrTy(M.getContext());
  auto *FuncTy =
      FunctionType::get(VoidTy, {Int8PtrTy, Int8PtrTy}, /* IsVarArg */ false);
  return Function::Create(FuncTy, GlobalVariable::InternalLinkage,
                          M.getDataLayout().getDefaultGlobalsAddressSpace(),
                          ".omp.reduction.func", &M);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createReductions(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<ReductionInfo> ReductionInfos, bool IsNoWait) {
  for (const ReductionInfo &RI : ReductionInfos) {
    (void)RI;
    assert(RI.Variable && "expected non-null variable");
    assert(RI.PrivateVariable && "expected non-null private variable");
    assert(RI.ReductionGen && "expected non-null reduction generator callback");
    assert(RI.Variable->getType() == RI.PrivateVariable->getType() &&
           "expected variables and their private equivalents to have the same "
           "type");
    assert(RI.Variable->getType()->isPointerTy() &&
           "expected variables to be pointers");
  }

  if (!updateToLocation(Loc))
    return InsertPointTy();

  BasicBlock *InsertBlock = Loc.IP.getBlock();
  BasicBlock *ContinuationBlock =
      InsertBlock->splitBasicBlock(Loc.IP.getPoint(), "reduce.finalize");
  InsertBlock->getTerminator()->eraseFromParent();

  // Create and populate array of type-erased pointers to private reduction
  // values.
  unsigned NumReductions = ReductionInfos.size();
  Type *RedArrayTy = ArrayType::get(Builder.getInt8PtrTy(), NumReductions);
  Builder.restoreIP(AllocaIP);
  Value *RedArray = Builder.CreateAlloca(RedArrayTy, nullptr, "red.array");

  Builder.SetInsertPoint(InsertBlock, InsertBlock->end());

  for (auto En : enumerate(ReductionInfos)) {
    unsigned Index = En.index();
    const ReductionInfo &RI = En.value();
    Value *RedArrayElemPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArrayTy, RedArray, 0, Index, "red.array.elem." + Twine(Index));
    Value *Casted =
        Builder.CreateBitCast(RI.PrivateVariable, Builder.getInt8PtrTy(),
                              "private.red.var." + Twine(Index) + ".casted");
    Builder.CreateStore(Casted, RedArrayElemPtr);
  }

  // Emit a call to the runtime function that orchestrates the reduction.
  // Declare the reduction function in the process.
  Function *Func = Builder.GetInsertBlock()->getParent();
  Module *Module = Func->getParent();
  Value *RedArrayPtr =
      Builder.CreateBitCast(RedArray, Builder.getInt8PtrTy(), "red.array.ptr");
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  bool CanGenerateAtomic =
      llvm::all_of(ReductionInfos, [](const ReductionInfo &RI) {
        return RI.AtomicReductionGen;
      });
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize,
                                  CanGenerateAtomic
                                      ? IdentFlag::OMP_IDENT_FLAG_ATOMIC_REDUCE
                                      : IdentFlag(0));
  Value *ThreadId = getOrCreateThreadID(Ident);
  Constant *NumVariables = Builder.getInt32(NumReductions);
  const DataLayout &DL = Module->getDataLayout();
  unsigned RedArrayByteSize = DL.getTypeStoreSize(RedArrayTy);
  Constant *RedArraySize = Builder.getInt64(RedArrayByteSize);
  Function *ReductionFunc = getFreshReductionFunc(*Module);
  Value *Lock = getOMPCriticalRegionLock(".reduction");
  Function *ReduceFunc = getOrCreateRuntimeFunctionPtr(
      IsNoWait ? RuntimeFunction::OMPRTL___kmpc_reduce_nowait
               : RuntimeFunction::OMPRTL___kmpc_reduce);
  CallInst *ReduceCall =
      Builder.CreateCall(ReduceFunc,
                         {Ident, ThreadId, NumVariables, RedArraySize,
                          RedArrayPtr, ReductionFunc, Lock},
                         "reduce");

  // Create final reduction entry blocks for the atomic and non-atomic case.
  // Emit IR that dispatches control flow to one of the blocks based on the
  // reduction supporting the atomic mode.
  BasicBlock *NonAtomicRedBlock =
      BasicBlock::Create(Module->getContext(), "reduce.switch.nonatomic", Func);
  BasicBlock *AtomicRedBlock =
      BasicBlock::Create(Module->getContext(), "reduce.switch.atomic", Func);
  SwitchInst *Switch =
      Builder.CreateSwitch(ReduceCall, ContinuationBlock, /* NumCases */ 2);
  Switch->addCase(Builder.getInt32(1), NonAtomicRedBlock);
  Switch->addCase(Builder.getInt32(2), AtomicRedBlock);

  // Populate the non-atomic reduction using the elementwise reduction function.
  // This loads the elements from the global and private variables and reduces
  // them before storing back the result to the global variable.
  Builder.SetInsertPoint(NonAtomicRedBlock);
  for (auto En : enumerate(ReductionInfos)) {
    const ReductionInfo &RI = En.value();
    Type *ValueType = RI.ElementType;
    Value *RedValue = Builder.CreateLoad(ValueType, RI.Variable,
                                         "red.value." + Twine(En.index()));
    Value *PrivateRedValue =
        Builder.CreateLoad(ValueType, RI.PrivateVariable,
                           "red.private.value." + Twine(En.index()));
    Value *Reduced;
    Builder.restoreIP(
        RI.ReductionGen(Builder.saveIP(), RedValue, PrivateRedValue, Reduced));
    if (!Builder.GetInsertBlock())
      return InsertPointTy();
    Builder.CreateStore(Reduced, RI.Variable);
  }
  Function *EndReduceFunc = getOrCreateRuntimeFunctionPtr(
      IsNoWait ? RuntimeFunction::OMPRTL___kmpc_end_reduce_nowait
               : RuntimeFunction::OMPRTL___kmpc_end_reduce);
  Builder.CreateCall(EndReduceFunc, {Ident, ThreadId, Lock});
  Builder.CreateBr(ContinuationBlock);

  // Populate the atomic reduction using the atomic elementwise reduction
  // function. There are no loads/stores here because they will be happening
  // inside the atomic elementwise reduction.
  Builder.SetInsertPoint(AtomicRedBlock);
  if (CanGenerateAtomic) {
    for (const ReductionInfo &RI : ReductionInfos) {
      Builder.restoreIP(RI.AtomicReductionGen(Builder.saveIP(), RI.ElementType,
                                              RI.Variable, RI.PrivateVariable));
      if (!Builder.GetInsertBlock())
        return InsertPointTy();
    }
    Builder.CreateBr(ContinuationBlock);
  } else {
    Builder.CreateUnreachable();
  }

  // Populate the outlined reduction function using the elementwise reduction
  // function. Partial values are extracted from the type-erased array of
  // pointers to private variables.
  BasicBlock *ReductionFuncBlock =
      BasicBlock::Create(Module->getContext(), "", ReductionFunc);
  Builder.SetInsertPoint(ReductionFuncBlock);
  Value *LHSArrayPtr = ReductionFunc->getArg(0);
  Value *RHSArrayPtr = ReductionFunc->getArg(1);

  for (auto En : enumerate(ReductionInfos)) {
    const ReductionInfo &RI = En.value();
    Value *LHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArrayTy, LHSArrayPtr, 0, En.index());
    Value *LHSI8Ptr = Builder.CreateLoad(Builder.getInt8PtrTy(), LHSI8PtrPtr);
    Value *LHSPtr = Builder.CreateBitCast(LHSI8Ptr, RI.Variable->getType());
    Value *LHS = Builder.CreateLoad(RI.ElementType, LHSPtr);
    Value *RHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArrayTy, RHSArrayPtr, 0, En.index());
    Value *RHSI8Ptr = Builder.CreateLoad(Builder.getInt8PtrTy(), RHSI8PtrPtr);
    Value *RHSPtr =
        Builder.CreateBitCast(RHSI8Ptr, RI.PrivateVariable->getType());
    Value *RHS = Builder.CreateLoad(RI.ElementType, RHSPtr);
    Value *Reduced;
    Builder.restoreIP(RI.ReductionGen(Builder.saveIP(), LHS, RHS, Reduced));
    if (!Builder.GetInsertBlock())
      return InsertPointTy();
    Builder.CreateStore(Reduced, LHSPtr);
  }
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(ContinuationBlock);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createMaster(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_master;
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId};

  Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_master);
  Instruction *EntryCall = Builder.CreateCall(EntryRTLFn, Args);

  Function *ExitRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_master);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ true, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createMasked(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB, Value *Filter) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_masked;
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId, Filter};
  Value *ArgsEnd[] = {Ident, ThreadId};

  Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_masked);
  Instruction *EntryCall = Builder.CreateCall(EntryRTLFn, Args);

  Function *ExitRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_masked);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, ArgsEnd);

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ true, /*hasFinalize*/ true);
}

CanonicalLoopInfo *OpenMPIRBuilder::createLoopSkeleton(
    DebugLoc DL, Value *TripCount, Function *F, BasicBlock *PreInsertBefore,
    BasicBlock *PostInsertBefore, const Twine &Name) {
  Module *M = F->getParent();
  LLVMContext &Ctx = M->getContext();
  Type *IndVarTy = TripCount->getType();

  // Create the basic block structure.
  BasicBlock *Preheader =
      BasicBlock::Create(Ctx, "omp_" + Name + ".preheader", F, PreInsertBefore);
  BasicBlock *Header =
      BasicBlock::Create(Ctx, "omp_" + Name + ".header", F, PreInsertBefore);
  BasicBlock *Cond =
      BasicBlock::Create(Ctx, "omp_" + Name + ".cond", F, PreInsertBefore);
  BasicBlock *Body =
      BasicBlock::Create(Ctx, "omp_" + Name + ".body", F, PreInsertBefore);
  BasicBlock *Latch =
      BasicBlock::Create(Ctx, "omp_" + Name + ".inc", F, PostInsertBefore);
  BasicBlock *Exit =
      BasicBlock::Create(Ctx, "omp_" + Name + ".exit", F, PostInsertBefore);
  BasicBlock *After =
      BasicBlock::Create(Ctx, "omp_" + Name + ".after", F, PostInsertBefore);

  // Use specified DebugLoc for new instructions.
  Builder.SetCurrentDebugLocation(DL);

  Builder.SetInsertPoint(Preheader);
  Builder.CreateBr(Header);

  Builder.SetInsertPoint(Header);
  PHINode *IndVarPHI = Builder.CreatePHI(IndVarTy, 2, "omp_" + Name + ".iv");
  IndVarPHI->addIncoming(ConstantInt::get(IndVarTy, 0), Preheader);
  Builder.CreateBr(Cond);

  Builder.SetInsertPoint(Cond);
  Value *Cmp =
      Builder.CreateICmpULT(IndVarPHI, TripCount, "omp_" + Name + ".cmp");
  Builder.CreateCondBr(Cmp, Body, Exit);

  Builder.SetInsertPoint(Body);
  Builder.CreateBr(Latch);

  Builder.SetInsertPoint(Latch);
  Value *Next = Builder.CreateAdd(IndVarPHI, ConstantInt::get(IndVarTy, 1),
                                  "omp_" + Name + ".next", /*HasNUW=*/true);
  Builder.CreateBr(Header);
  IndVarPHI->addIncoming(Next, Latch);

  Builder.SetInsertPoint(Exit);
  Builder.CreateBr(After);

  // Remember and return the canonical control flow.
  LoopInfos.emplace_front();
  CanonicalLoopInfo *CL = &LoopInfos.front();

  CL->Header = Header;
  CL->Cond = Cond;
  CL->Latch = Latch;
  CL->Exit = Exit;

#ifndef NDEBUG
  CL->assertOK();
#endif
  return CL;
}

CanonicalLoopInfo *
OpenMPIRBuilder::createCanonicalLoop(const LocationDescription &Loc,
                                     LoopBodyGenCallbackTy BodyGenCB,
                                     Value *TripCount, const Twine &Name) {
  BasicBlock *BB = Loc.IP.getBlock();
  BasicBlock *NextBB = BB->getNextNode();

  CanonicalLoopInfo *CL = createLoopSkeleton(Loc.DL, TripCount, BB->getParent(),
                                             NextBB, NextBB, Name);
  BasicBlock *After = CL->getAfter();

  // If location is not set, don't connect the loop.
  if (updateToLocation(Loc)) {
    // Split the loop at the insertion point: Branch to the preheader and move
    // every following instruction to after the loop (the After BB). Also, the
    // new successor is the loop's after block.
    spliceBB(Builder, After, /*CreateBranch=*/false);
    Builder.CreateBr(CL->getPreheader());
  }

  // Emit the body content. We do it after connecting the loop to the CFG to
  // avoid that the callback encounters degenerate BBs.
  BodyGenCB(CL->getBodyIP(), CL->getIndVar());

#ifndef NDEBUG
  CL->assertOK();
#endif
  return CL;
}

CanonicalLoopInfo *OpenMPIRBuilder::createCanonicalLoop(
    const LocationDescription &Loc, LoopBodyGenCallbackTy BodyGenCB,
    Value *Start, Value *Stop, Value *Step, bool IsSigned, bool InclusiveStop,
    InsertPointTy ComputeIP, const Twine &Name) {

  // Consider the following difficulties (assuming 8-bit signed integers):
  //  * Adding \p Step to the loop counter which passes \p Stop may overflow:
  //      DO I = 1, 100, 50
  ///  * A \p Step of INT_MIN cannot not be normalized to a positive direction:
  //      DO I = 100, 0, -128

  // Start, Stop and Step must be of the same integer type.
  auto *IndVarTy = cast<IntegerType>(Start->getType());
  assert(IndVarTy == Stop->getType() && "Stop type mismatch");
  assert(IndVarTy == Step->getType() && "Step type mismatch");

  LocationDescription ComputeLoc =
      ComputeIP.isSet() ? LocationDescription(ComputeIP, Loc.DL) : Loc;
  updateToLocation(ComputeLoc);

  ConstantInt *Zero = ConstantInt::get(IndVarTy, 0);
  ConstantInt *One = ConstantInt::get(IndVarTy, 1);

  // Like Step, but always positive.
  Value *Incr = Step;

  // Distance between Start and Stop; always positive.
  Value *Span;

  // Condition whether there are no iterations are executed at all, e.g. because
  // UB < LB.
  Value *ZeroCmp;

  if (IsSigned) {
    // Ensure that increment is positive. If not, negate and invert LB and UB.
    Value *IsNeg = Builder.CreateICmpSLT(Step, Zero);
    Incr = Builder.CreateSelect(IsNeg, Builder.CreateNeg(Step), Step);
    Value *LB = Builder.CreateSelect(IsNeg, Stop, Start);
    Value *UB = Builder.CreateSelect(IsNeg, Start, Stop);
    Span = Builder.CreateSub(UB, LB, "", false, true);
    ZeroCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_SLT : CmpInst::ICMP_SLE, UB, LB);
  } else {
    Span = Builder.CreateSub(Stop, Start, "", true);
    ZeroCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_ULT : CmpInst::ICMP_ULE, Stop, Start);
  }

  Value *CountIfLooping;
  if (InclusiveStop) {
    CountIfLooping = Builder.CreateAdd(Builder.CreateUDiv(Span, Incr), One);
  } else {
    // Avoid incrementing past stop since it could overflow.
    Value *CountIfTwo = Builder.CreateAdd(
        Builder.CreateUDiv(Builder.CreateSub(Span, One), Incr), One);
    Value *OneCmp = Builder.CreateICmp(CmpInst::ICMP_ULE, Span, Incr);
    CountIfLooping = Builder.CreateSelect(OneCmp, One, CountIfTwo);
  }
  Value *TripCount = Builder.CreateSelect(ZeroCmp, Zero, CountIfLooping,
                                          "omp_" + Name + ".tripcount");

  auto BodyGen = [=](InsertPointTy CodeGenIP, Value *IV) {
    Builder.restoreIP(CodeGenIP);
    Value *Span = Builder.CreateMul(IV, Step);
    Value *IndVar = Builder.CreateAdd(Span, Start);
    BodyGenCB(Builder.saveIP(), IndVar);
  };
  LocationDescription LoopLoc = ComputeIP.isSet() ? Loc.IP : Builder.saveIP();
  return createCanonicalLoop(LoopLoc, BodyGen, TripCount, Name);
}

// Returns an LLVM function to call for initializing loop bounds using OpenMP
// static scheduling depending on `type`. Only i32 and i64 are supported by the
// runtime. Always interpret integers as unsigned similarly to
// CanonicalLoopInfo.
static FunctionCallee getKmpcForStaticInitForType(Type *Ty, Module &M,
                                                  OpenMPIRBuilder &OMPBuilder) {
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_for_static_init_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::applyStaticWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                          InsertPointTy AllocaIP,
                                          bool NeedsBarrier) {
  assert(CLI->isValid() && "Requires a valid canonical loop");
  assert(!isConflictIP(AllocaIP, CLI->getPreheaderIP()) &&
         "Require dedicated allocate IP");

  // Set up the source location value for OpenMP runtime.
  Builder.restoreIP(CLI->getPreheaderIP());
  Builder.SetCurrentDebugLocation(DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(DL, SrcLocStrSize);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // Declare useful OpenMP runtime functions.
  Value *IV = CLI->getIndVar();
  Type *IVTy = IV->getType();
  FunctionCallee StaticInit = getKmpcForStaticInitForType(IVTy, M, *this);
  FunctionCallee StaticFini =
      getOrCreateRuntimeFunction(M, omp::OMPRTL___kmpc_for_static_fini);

  // Allocate space for computed loop bounds as expected by the "init" function.
  Builder.restoreIP(AllocaIP);
  Type *I32Type = Type::getInt32Ty(M.getContext());
  Value *PLastIter = Builder.CreateAlloca(I32Type, nullptr, "p.lastiter");
  Value *PLowerBound = Builder.CreateAlloca(IVTy, nullptr, "p.lowerbound");
  Value *PUpperBound = Builder.CreateAlloca(IVTy, nullptr, "p.upperbound");
  Value *PStride = Builder.CreateAlloca(IVTy, nullptr, "p.stride");

  // At the end of the preheader, prepare for calling the "init" function by
  // storing the current loop bounds into the allocated space. A canonical loop
  // always iterates from 0 to trip-count with step 1. Note that "init" expects
  // and produces an inclusive upper bound.
  Builder.SetInsertPoint(CLI->getPreheader()->getTerminator());
  Constant *Zero = ConstantInt::get(IVTy, 0);
  Constant *One = ConstantInt::get(IVTy, 1);
  Builder.CreateStore(Zero, PLowerBound);
  Value *UpperBound = Builder.CreateSub(CLI->getTripCount(), One);
  Builder.CreateStore(UpperBound, PUpperBound);
  Builder.CreateStore(One, PStride);

  Value *ThreadNum = getOrCreateThreadID(SrcLoc);

  Constant *SchedulingType = ConstantInt::get(
      I32Type, static_cast<int>(OMPScheduleType::UnorderedStatic));

  // Call the "init" function and update the trip count of the loop with the
  // value it produced.
  Builder.CreateCall(StaticInit,
                     {SrcLoc, ThreadNum, SchedulingType, PLastIter, PLowerBound,
                      PUpperBound, PStride, One, Zero});
  Value *LowerBound = Builder.CreateLoad(IVTy, PLowerBound);
  Value *InclusiveUpperBound = Builder.CreateLoad(IVTy, PUpperBound);
  Value *TripCountMinusOne = Builder.CreateSub(InclusiveUpperBound, LowerBound);
  Value *TripCount = Builder.CreateAdd(TripCountMinusOne, One);
  CLI->setTripCount(TripCount);

  // Update all uses of the induction variable except the one in the condition
  // block that compares it with the actual upper bound, and the increment in
  // the latch block.

  CLI->mapIndVar([&](Instruction *OldIV) -> Value * {
    Builder.SetInsertPoint(CLI->getBody(),
                           CLI->getBody()->getFirstInsertionPt());
    Builder.SetCurrentDebugLocation(DL);
    return Builder.CreateAdd(OldIV, LowerBound);
  });

  // In the "exit" block, call the "fini" function.
  Builder.SetInsertPoint(CLI->getExit(),
                         CLI->getExit()->getTerminator()->getIterator());
  Builder.CreateCall(StaticFini, {SrcLoc, ThreadNum});

  // Add the barrier if requested.
  if (NeedsBarrier)
    createBarrier(LocationDescription(Builder.saveIP(), DL),
                  omp::Directive::OMPD_for, /* ForceSimpleCall */ false,
                  /* CheckCancelFlag */ false);

  InsertPointTy AfterIP = CLI->getAfterIP();
  CLI->invalidate();

  return AfterIP;
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::applyStaticChunkedWorkshareLoop(
    DebugLoc DL, CanonicalLoopInfo *CLI, InsertPointTy AllocaIP,
    bool NeedsBarrier, Value *ChunkSize) {
  assert(CLI->isValid() && "Requires a valid canonical loop");
  assert(ChunkSize && "Chunk size is required");

  LLVMContext &Ctx = CLI->getFunction()->getContext();
  Value *IV = CLI->getIndVar();
  Value *OrigTripCount = CLI->getTripCount();
  Type *IVTy = IV->getType();
  assert(IVTy->getIntegerBitWidth() <= 64 &&
         "Max supported tripcount bitwidth is 64 bits");
  Type *InternalIVTy = IVTy->getIntegerBitWidth() <= 32 ? Type::getInt32Ty(Ctx)
                                                        : Type::getInt64Ty(Ctx);
  Type *I32Type = Type::getInt32Ty(M.getContext());
  Constant *Zero = ConstantInt::get(InternalIVTy, 0);
  Constant *One = ConstantInt::get(InternalIVTy, 1);

  // Declare useful OpenMP runtime functions.
  FunctionCallee StaticInit =
      getKmpcForStaticInitForType(InternalIVTy, M, *this);
  FunctionCallee StaticFini =
      getOrCreateRuntimeFunction(M, omp::OMPRTL___kmpc_for_static_fini);

  // Allocate space for computed loop bounds as expected by the "init" function.
  Builder.restoreIP(AllocaIP);
  Builder.SetCurrentDebugLocation(DL);
  Value *PLastIter = Builder.CreateAlloca(I32Type, nullptr, "p.lastiter");
  Value *PLowerBound =
      Builder.CreateAlloca(InternalIVTy, nullptr, "p.lowerbound");
  Value *PUpperBound =
      Builder.CreateAlloca(InternalIVTy, nullptr, "p.upperbound");
  Value *PStride = Builder.CreateAlloca(InternalIVTy, nullptr, "p.stride");

  // Set up the source location value for the OpenMP runtime.
  Builder.restoreIP(CLI->getPreheaderIP());
  Builder.SetCurrentDebugLocation(DL);

  // TODO: Detect overflow in ubsan or max-out with current tripcount.
  Value *CastedChunkSize =
      Builder.CreateZExtOrTrunc(ChunkSize, InternalIVTy, "chunksize");
  Value *CastedTripCount =
      Builder.CreateZExt(OrigTripCount, InternalIVTy, "tripcount");

  Constant *SchedulingType = ConstantInt::get(
      I32Type, static_cast<int>(OMPScheduleType::UnorderedStaticChunked));
  Builder.CreateStore(Zero, PLowerBound);
  Value *OrigUpperBound = Builder.CreateSub(CastedTripCount, One);
  Builder.CreateStore(OrigUpperBound, PUpperBound);
  Builder.CreateStore(One, PStride);

  // Call the "init" function and update the trip count of the loop with the
  // value it produced.
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(DL, SrcLocStrSize);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadNum = getOrCreateThreadID(SrcLoc);
  Builder.CreateCall(StaticInit,
                     {/*loc=*/SrcLoc, /*global_tid=*/ThreadNum,
                      /*schedtype=*/SchedulingType, /*plastiter=*/PLastIter,
                      /*plower=*/PLowerBound, /*pupper=*/PUpperBound,
                      /*pstride=*/PStride, /*incr=*/One,
                      /*chunk=*/CastedChunkSize});

  // Load values written by the "init" function.
  Value *FirstChunkStart =
      Builder.CreateLoad(InternalIVTy, PLowerBound, "omp_firstchunk.lb");
  Value *FirstChunkStop =
      Builder.CreateLoad(InternalIVTy, PUpperBound, "omp_firstchunk.ub");
  Value *FirstChunkEnd = Builder.CreateAdd(FirstChunkStop, One);
  Value *ChunkRange =
      Builder.CreateSub(FirstChunkEnd, FirstChunkStart, "omp_chunk.range");
  Value *NextChunkStride =
      Builder.CreateLoad(InternalIVTy, PStride, "omp_dispatch.stride");

  // Create outer "dispatch" loop for enumerating the chunks.
  BasicBlock *DispatchEnter = splitBB(Builder, true);
  Value *DispatchCounter;
  CanonicalLoopInfo *DispatchCLI = createCanonicalLoop(
      {Builder.saveIP(), DL},
      [&](InsertPointTy BodyIP, Value *Counter) { DispatchCounter = Counter; },
      FirstChunkStart, CastedTripCount, NextChunkStride,
      /*IsSigned=*/false, /*InclusiveStop=*/false, /*ComputeIP=*/{},
      "dispatch");

  // Remember the BasicBlocks of the dispatch loop we need, then invalidate to
  // not have to preserve the canonical invariant.
  BasicBlock *DispatchBody = DispatchCLI->getBody();
  BasicBlock *DispatchLatch = DispatchCLI->getLatch();
  BasicBlock *DispatchExit = DispatchCLI->getExit();
  BasicBlock *DispatchAfter = DispatchCLI->getAfter();
  DispatchCLI->invalidate();

  // Rewire the original loop to become the chunk loop inside the dispatch loop.
  redirectTo(DispatchAfter, CLI->getAfter(), DL);
  redirectTo(CLI->getExit(), DispatchLatch, DL);
  redirectTo(DispatchBody, DispatchEnter, DL);

  // Prepare the prolog of the chunk loop.
  Builder.restoreIP(CLI->getPreheaderIP());
  Builder.SetCurrentDebugLocation(DL);

  // Compute the number of iterations of the chunk loop.
  Builder.SetInsertPoint(CLI->getPreheader()->getTerminator());
  Value *ChunkEnd = Builder.CreateAdd(DispatchCounter, ChunkRange);
  Value *IsLastChunk =
      Builder.CreateICmpUGE(ChunkEnd, CastedTripCount, "omp_chunk.is_last");
  Value *CountUntilOrigTripCount =
      Builder.CreateSub(CastedTripCount, DispatchCounter);
  Value *ChunkTripCount = Builder.CreateSelect(
      IsLastChunk, CountUntilOrigTripCount, ChunkRange, "omp_chunk.tripcount");
  Value *BackcastedChunkTC =
      Builder.CreateTrunc(ChunkTripCount, IVTy, "omp_chunk.tripcount.trunc");
  CLI->setTripCount(BackcastedChunkTC);

  // Update all uses of the induction variable except the one in the condition
  // block that compares it with the actual upper bound, and the increment in
  // the latch block.
  Value *BackcastedDispatchCounter =
      Builder.CreateTrunc(DispatchCounter, IVTy, "omp_dispatch.iv.trunc");
  CLI->mapIndVar([&](Instruction *) -> Value * {
    Builder.restoreIP(CLI->getBodyIP());
    return Builder.CreateAdd(IV, BackcastedDispatchCounter);
  });

  // In the "exit" block, call the "fini" function.
  Builder.SetInsertPoint(DispatchExit, DispatchExit->getFirstInsertionPt());
  Builder.CreateCall(StaticFini, {SrcLoc, ThreadNum});

  // Add the barrier if requested.
  if (NeedsBarrier)
    createBarrier(LocationDescription(Builder.saveIP(), DL), OMPD_for,
                  /*ForceSimpleCall=*/false, /*CheckCancelFlag=*/false);

#ifndef NDEBUG
  // Even though we currently do not support applying additional methods to it,
  // the chunk loop should remain a canonical loop.
  CLI->assertOK();
#endif

  return {DispatchAfter, DispatchAfter->getFirstInsertionPt()};
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::applyWorkshareLoop(
    DebugLoc DL, CanonicalLoopInfo *CLI, InsertPointTy AllocaIP,
    bool NeedsBarrier, llvm::omp::ScheduleKind SchedKind,
    llvm::Value *ChunkSize, bool HasSimdModifier, bool HasMonotonicModifier,
    bool HasNonmonotonicModifier, bool HasOrderedClause) {
  OMPScheduleType EffectiveScheduleType = computeOpenMPScheduleType(
      SchedKind, ChunkSize, HasSimdModifier, HasMonotonicModifier,
      HasNonmonotonicModifier, HasOrderedClause);

  bool IsOrdered = (EffectiveScheduleType & OMPScheduleType::ModifierOrdered) ==
                   OMPScheduleType::ModifierOrdered;
  switch (EffectiveScheduleType & ~OMPScheduleType::ModifierMask) {
  case OMPScheduleType::BaseStatic:
    assert(!ChunkSize && "No chunk size with static-chunked schedule");
    if (IsOrdered)
      return applyDynamicWorkshareLoop(DL, CLI, AllocaIP, EffectiveScheduleType,
                                       NeedsBarrier, ChunkSize);
    // FIXME: Monotonicity ignored?
    return applyStaticWorkshareLoop(DL, CLI, AllocaIP, NeedsBarrier);

  case OMPScheduleType::BaseStaticChunked:
    if (IsOrdered)
      return applyDynamicWorkshareLoop(DL, CLI, AllocaIP, EffectiveScheduleType,
                                       NeedsBarrier, ChunkSize);
    // FIXME: Monotonicity ignored?
    return applyStaticChunkedWorkshareLoop(DL, CLI, AllocaIP, NeedsBarrier,
                                           ChunkSize);

  case OMPScheduleType::BaseRuntime:
  case OMPScheduleType::BaseAuto:
  case OMPScheduleType::BaseGreedy:
  case OMPScheduleType::BaseBalanced:
  case OMPScheduleType::BaseSteal:
  case OMPScheduleType::BaseGuidedSimd:
  case OMPScheduleType::BaseRuntimeSimd:
    assert(!ChunkSize &&
           "schedule type does not support user-defined chunk sizes");
    [[fallthrough]];
  case OMPScheduleType::BaseDynamicChunked:
  case OMPScheduleType::BaseGuidedChunked:
  case OMPScheduleType::BaseGuidedIterativeChunked:
  case OMPScheduleType::BaseGuidedAnalyticalChunked:
  case OMPScheduleType::BaseStaticBalancedChunked:
    return applyDynamicWorkshareLoop(DL, CLI, AllocaIP, EffectiveScheduleType,
                                     NeedsBarrier, ChunkSize);

  default:
    llvm_unreachable("Unknown/unimplemented schedule kind");
  }
}

/// Returns an LLVM function to call for initializing loop bounds using OpenMP
/// dynamic scheduling depending on `type`. Only i32 and i64 are supported by
/// the runtime. Always interpret integers as unsigned similarly to
/// CanonicalLoopInfo.
static FunctionCallee
getKmpcForDynamicInitForType(Type *Ty, Module &M, OpenMPIRBuilder &OMPBuilder) {
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_init_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_init_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

/// Returns an LLVM function to call for updating the next loop using OpenMP
/// dynamic scheduling depending on `type`. Only i32 and i64 are supported by
/// the runtime. Always interpret integers as unsigned similarly to
/// CanonicalLoopInfo.
static FunctionCallee
getKmpcForDynamicNextForType(Type *Ty, Module &M, OpenMPIRBuilder &OMPBuilder) {
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_next_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_next_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

/// Returns an LLVM function to call for finalizing the dynamic loop using
/// depending on `type`. Only i32 and i64 are supported by the runtime. Always
/// interpret integers as unsigned similarly to CanonicalLoopInfo.
static FunctionCallee
getKmpcForDynamicFiniForType(Type *Ty, Module &M, OpenMPIRBuilder &OMPBuilder) {
  unsigned Bitwidth = Ty->getIntegerBitWidth();
  if (Bitwidth == 32)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_fini_4u);
  if (Bitwidth == 64)
    return OMPBuilder.getOrCreateRuntimeFunction(
        M, omp::RuntimeFunction::OMPRTL___kmpc_dispatch_fini_8u);
  llvm_unreachable("unknown OpenMP loop iterator bitwidth");
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::applyDynamicWorkshareLoop(
    DebugLoc DL, CanonicalLoopInfo *CLI, InsertPointTy AllocaIP,
    OMPScheduleType SchedType, bool NeedsBarrier, Value *Chunk) {
  assert(CLI->isValid() && "Requires a valid canonical loop");
  assert(!isConflictIP(AllocaIP, CLI->getPreheaderIP()) &&
         "Require dedicated allocate IP");
  assert(isValidWorkshareLoopScheduleType(SchedType) &&
         "Require valid schedule type");

  bool Ordered = (SchedType & OMPScheduleType::ModifierOrdered) ==
                 OMPScheduleType::ModifierOrdered;

  // Set up the source location value for OpenMP runtime.
  Builder.SetCurrentDebugLocation(DL);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(DL, SrcLocStrSize);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr, SrcLocStrSize);

  // Declare useful OpenMP runtime functions.
  Value *IV = CLI->getIndVar();
  Type *IVTy = IV->getType();
  FunctionCallee DynamicInit = getKmpcForDynamicInitForType(IVTy, M, *this);
  FunctionCallee DynamicNext = getKmpcForDynamicNextForType(IVTy, M, *this);

  // Allocate space for computed loop bounds as expected by the "init" function.
  Builder.restoreIP(AllocaIP);
  Type *I32Type = Type::getInt32Ty(M.getContext());
  Value *PLastIter = Builder.CreateAlloca(I32Type, nullptr, "p.lastiter");
  Value *PLowerBound = Builder.CreateAlloca(IVTy, nullptr, "p.lowerbound");
  Value *PUpperBound = Builder.CreateAlloca(IVTy, nullptr, "p.upperbound");
  Value *PStride = Builder.CreateAlloca(IVTy, nullptr, "p.stride");

  // At the end of the preheader, prepare for calling the "init" function by
  // storing the current loop bounds into the allocated space. A canonical loop
  // always iterates from 0 to trip-count with step 1. Note that "init" expects
  // and produces an inclusive upper bound.
  BasicBlock *PreHeader = CLI->getPreheader();
  Builder.SetInsertPoint(PreHeader->getTerminator());
  Constant *One = ConstantInt::get(IVTy, 1);
  Builder.CreateStore(One, PLowerBound);
  Value *UpperBound = CLI->getTripCount();
  Builder.CreateStore(UpperBound, PUpperBound);
  Builder.CreateStore(One, PStride);

  BasicBlock *Header = CLI->getHeader();
  BasicBlock *Exit = CLI->getExit();
  BasicBlock *Cond = CLI->getCond();
  BasicBlock *Latch = CLI->getLatch();
  InsertPointTy AfterIP = CLI->getAfterIP();

  // The CLI will be "broken" in the code below, as the loop is no longer
  // a valid canonical loop.

  if (!Chunk)
    Chunk = One;

  Value *ThreadNum = getOrCreateThreadID(SrcLoc);

  Constant *SchedulingType =
      ConstantInt::get(I32Type, static_cast<int>(SchedType));

  // Call the "init" function.
  Builder.CreateCall(DynamicInit,
                     {SrcLoc, ThreadNum, SchedulingType, /* LowerBound */ One,
                      UpperBound, /* step */ One, Chunk});

  // An outer loop around the existing one.
  BasicBlock *OuterCond = BasicBlock::Create(
      PreHeader->getContext(), Twine(PreHeader->getName()) + ".outer.cond",
      PreHeader->getParent());
  // This needs to be 32-bit always, so can't use the IVTy Zero above.
  Builder.SetInsertPoint(OuterCond, OuterCond->getFirstInsertionPt());
  Value *Res =
      Builder.CreateCall(DynamicNext, {SrcLoc, ThreadNum, PLastIter,
                                       PLowerBound, PUpperBound, PStride});
  Constant *Zero32 = ConstantInt::get(I32Type, 0);
  Value *MoreWork = Builder.CreateCmp(CmpInst::ICMP_NE, Res, Zero32);
  Value *LowerBound =
      Builder.CreateSub(Builder.CreateLoad(IVTy, PLowerBound), One, "lb");
  Builder.CreateCondBr(MoreWork, Header, Exit);

  // Change PHI-node in loop header to use outer cond rather than preheader,
  // and set IV to the LowerBound.
  Instruction *Phi = &Header->front();
  auto *PI = cast<PHINode>(Phi);
  PI->setIncomingBlock(0, OuterCond);
  PI->setIncomingValue(0, LowerBound);

  // Then set the pre-header to jump to the OuterCond
  Instruction *Term = PreHeader->getTerminator();
  auto *Br = cast<BranchInst>(Term);
  Br->setSuccessor(0, OuterCond);

  // Modify the inner condition:
  // * Use the UpperBound returned from the DynamicNext call.
  // * jump to the loop outer loop when done with one of the inner loops.
  Builder.SetInsertPoint(Cond, Cond->getFirstInsertionPt());
  UpperBound = Builder.CreateLoad(IVTy, PUpperBound, "ub");
  Instruction *Comp = &*Builder.GetInsertPoint();
  auto *CI = cast<CmpInst>(Comp);
  CI->setOperand(1, UpperBound);
  // Redirect the inner exit to branch to outer condition.
  Instruction *Branch = &Cond->back();
  auto *BI = cast<BranchInst>(Branch);
  assert(BI->getSuccessor(1) == Exit);
  BI->setSuccessor(1, OuterCond);

  // Call the "fini" function if "ordered" is present in wsloop directive.
  if (Ordered) {
    Builder.SetInsertPoint(&Latch->back());
    FunctionCallee DynamicFini = getKmpcForDynamicFiniForType(IVTy, M, *this);
    Builder.CreateCall(DynamicFini, {SrcLoc, ThreadNum});
  }

  // Add the barrier if requested.
  if (NeedsBarrier) {
    Builder.SetInsertPoint(&Exit->back());
    createBarrier(LocationDescription(Builder.saveIP(), DL),
                  omp::Directive::OMPD_for, /* ForceSimpleCall */ false,
                  /* CheckCancelFlag */ false);
  }

  CLI->invalidate();
  return AfterIP;
}

/// Redirect all edges that branch to \p OldTarget to \p NewTarget. That is,
/// after this \p OldTarget will be orphaned.
static void redirectAllPredecessorsTo(BasicBlock *OldTarget,
                                      BasicBlock *NewTarget, DebugLoc DL) {
  for (BasicBlock *Pred : make_early_inc_range(predecessors(OldTarget)))
    redirectTo(Pred, NewTarget, DL);
}

/// Determine which blocks in \p BBs are reachable from outside and remove the
/// ones that are not reachable from the function.
static void removeUnusedBlocksFromParent(ArrayRef<BasicBlock *> BBs) {
  SmallPtrSet<BasicBlock *, 6> BBsToErase{BBs.begin(), BBs.end()};
  auto HasRemainingUses = [&BBsToErase](BasicBlock *BB) {
    for (Use &U : BB->uses()) {
      auto *UseInst = dyn_cast<Instruction>(U.getUser());
      if (!UseInst)
        continue;
      if (BBsToErase.count(UseInst->getParent()))
        continue;
      return true;
    }
    return false;
  };

  while (true) {
    bool Changed = false;
    for (BasicBlock *BB : make_early_inc_range(BBsToErase)) {
      if (HasRemainingUses(BB)) {
        BBsToErase.erase(BB);
        Changed = true;
      }
    }
    if (!Changed)
      break;
  }

  SmallVector<BasicBlock *, 7> BBVec(BBsToErase.begin(), BBsToErase.end());
  DeleteDeadBlocks(BBVec);
}

CanonicalLoopInfo *
OpenMPIRBuilder::collapseLoops(DebugLoc DL, ArrayRef<CanonicalLoopInfo *> Loops,
                               InsertPointTy ComputeIP) {
  assert(Loops.size() >= 1 && "At least one loop required");
  size_t NumLoops = Loops.size();

  // Nothing to do if there is already just one loop.
  if (NumLoops == 1)
    return Loops.front();

  CanonicalLoopInfo *Outermost = Loops.front();
  CanonicalLoopInfo *Innermost = Loops.back();
  BasicBlock *OrigPreheader = Outermost->getPreheader();
  BasicBlock *OrigAfter = Outermost->getAfter();
  Function *F = OrigPreheader->getParent();

  // Loop control blocks that may become orphaned later.
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);

  // Setup the IRBuilder for inserting the trip count computation.
  Builder.SetCurrentDebugLocation(DL);
  if (ComputeIP.isSet())
    Builder.restoreIP(ComputeIP);
  else
    Builder.restoreIP(Outermost->getPreheaderIP());

  // Derive the collapsed' loop trip count.
  // TODO: Find common/largest indvar type.
  Value *CollapsedTripCount = nullptr;
  for (CanonicalLoopInfo *L : Loops) {
    assert(L->isValid() &&
           "All loops to collapse must be valid canonical loops");
    Value *OrigTripCount = L->getTripCount();
    if (!CollapsedTripCount) {
      CollapsedTripCount = OrigTripCount;
      continue;
    }

    // TODO: Enable UndefinedSanitizer to diagnose an overflow here.
    CollapsedTripCount = Builder.CreateMul(CollapsedTripCount, OrigTripCount,
                                           {}, /*HasNUW=*/true);
  }

  // Create the collapsed loop control flow.
  CanonicalLoopInfo *Result =
      createLoopSkeleton(DL, CollapsedTripCount, F,
                         OrigPreheader->getNextNode(), OrigAfter, "collapsed");

  // Build the collapsed loop body code.
  // Start with deriving the input loop induction variables from the collapsed
  // one, using a divmod scheme. To preserve the original loops' order, the
  // innermost loop use the least significant bits.
  Builder.restoreIP(Result->getBodyIP());

  Value *Leftover = Result->getIndVar();
  SmallVector<Value *> NewIndVars;
  NewIndVars.resize(NumLoops);
  for (int i = NumLoops - 1; i >= 1; --i) {
    Value *OrigTripCount = Loops[i]->getTripCount();

    Value *NewIndVar = Builder.CreateURem(Leftover, OrigTripCount);
    NewIndVars[i] = NewIndVar;

    Leftover = Builder.CreateUDiv(Leftover, OrigTripCount);
  }
  // Outermost loop gets all the remaining bits.
  NewIndVars[0] = Leftover;

  // Construct the loop body control flow.
  // We progressively construct the branch structure following in direction of
  // the control flow, from the leading in-between code, the loop nest body, the
  // trailing in-between code, and rejoining the collapsed loop's latch.
  // ContinueBlock and ContinuePred keep track of the source(s) of next edge. If
  // the ContinueBlock is set, continue with that block. If ContinuePred, use
  // its predecessors as sources.
  BasicBlock *ContinueBlock = Result->getBody();
  BasicBlock *ContinuePred = nullptr;
  auto ContinueWith = [&ContinueBlock, &ContinuePred, DL](BasicBlock *Dest,
                                                          BasicBlock *NextSrc) {
    if (ContinueBlock)
      redirectTo(ContinueBlock, Dest, DL);
    else
      redirectAllPredecessorsTo(ContinuePred, Dest, DL);

    ContinueBlock = nullptr;
    ContinuePred = NextSrc;
  };

  // The code before the nested loop of each level.
  // Because we are sinking it into the nest, it will be executed more often
  // that the original loop. More sophisticated schemes could keep track of what
  // the in-between code is and instantiate it only once per thread.
  for (size_t i = 0; i < NumLoops - 1; ++i)
    ContinueWith(Loops[i]->getBody(), Loops[i + 1]->getHeader());

  // Connect the loop nest body.
  ContinueWith(Innermost->getBody(), Innermost->getLatch());

  // The code after the nested loop at each level.
  for (size_t i = NumLoops - 1; i > 0; --i)
    ContinueWith(Loops[i]->getAfter(), Loops[i - 1]->getLatch());

  // Connect the finished loop to the collapsed loop latch.
  ContinueWith(Result->getLatch(), nullptr);

  // Replace the input loops with the new collapsed loop.
  redirectTo(Outermost->getPreheader(), Result->getPreheader(), DL);
  redirectTo(Result->getAfter(), Outermost->getAfter(), DL);

  // Replace the input loop indvars with the derived ones.
  for (size_t i = 0; i < NumLoops; ++i)
    Loops[i]->getIndVar()->replaceAllUsesWith(NewIndVars[i]);

  // Remove unused parts of the input loops.
  removeUnusedBlocksFromParent(OldControlBBs);

  for (CanonicalLoopInfo *L : Loops)
    L->invalidate();

#ifndef NDEBUG
  Result->assertOK();
#endif
  return Result;
}

std::vector<CanonicalLoopInfo *>
OpenMPIRBuilder::tileLoops(DebugLoc DL, ArrayRef<CanonicalLoopInfo *> Loops,
                           ArrayRef<Value *> TileSizes) {
  assert(TileSizes.size() == Loops.size() &&
         "Must pass as many tile sizes as there are loops");
  int NumLoops = Loops.size();
  assert(NumLoops >= 1 && "At least one loop to tile required");

  CanonicalLoopInfo *OutermostLoop = Loops.front();
  CanonicalLoopInfo *InnermostLoop = Loops.back();
  Function *F = OutermostLoop->getBody()->getParent();
  BasicBlock *InnerEnter = InnermostLoop->getBody();
  BasicBlock *InnerLatch = InnermostLoop->getLatch();

  // Loop control blocks that may become orphaned later.
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);

  // Collect original trip counts and induction variable to be accessible by
  // index. Also, the structure of the original loops is not preserved during
  // the construction of the tiled loops, so do it before we scavenge the BBs of
  // any original CanonicalLoopInfo.
  SmallVector<Value *, 4> OrigTripCounts, OrigIndVars;
  for (CanonicalLoopInfo *L : Loops) {
    assert(L->isValid() && "All input loops must be valid canonical loops");
    OrigTripCounts.push_back(L->getTripCount());
    OrigIndVars.push_back(L->getIndVar());
  }

  // Collect the code between loop headers. These may contain SSA definitions
  // that are used in the loop nest body. To be usable with in the innermost
  // body, these BasicBlocks will be sunk into the loop nest body. That is,
  // these instructions may be executed more often than before the tiling.
  // TODO: It would be sufficient to only sink them into body of the
  // corresponding tile loop.
  SmallVector<std::pair<BasicBlock *, BasicBlock *>, 4> InbetweenCode;
  for (int i = 0; i < NumLoops - 1; ++i) {
    CanonicalLoopInfo *Surrounding = Loops[i];
    CanonicalLoopInfo *Nested = Loops[i + 1];

    BasicBlock *EnterBB = Surrounding->getBody();
    BasicBlock *ExitBB = Nested->getHeader();
    InbetweenCode.emplace_back(EnterBB, ExitBB);
  }

  // Compute the trip counts of the floor loops.
  Builder.SetCurrentDebugLocation(DL);
  Builder.restoreIP(OutermostLoop->getPreheaderIP());
  SmallVector<Value *, 4> FloorCount, FloorRems;
  for (int i = 0; i < NumLoops; ++i) {
    Value *TileSize = TileSizes[i];
    Value *OrigTripCount = OrigTripCounts[i];
    Type *IVType = OrigTripCount->getType();

    Value *FloorTripCount = Builder.CreateUDiv(OrigTripCount, TileSize);
    Value *FloorTripRem = Builder.CreateURem(OrigTripCount, TileSize);

    // 0 if tripcount divides the tilesize, 1 otherwise.
    // 1 means we need an additional iteration for a partial tile.
    //
    // Unfortunately we cannot just use the roundup-formula
    //   (tripcount + tilesize - 1)/tilesize
    // because the summation might overflow. We do not want introduce undefined
    // behavior when the untiled loop nest did not.
    Value *FloorTripOverflow =
        Builder.CreateICmpNE(FloorTripRem, ConstantInt::get(IVType, 0));

    FloorTripOverflow = Builder.CreateZExt(FloorTripOverflow, IVType);
    FloorTripCount =
        Builder.CreateAdd(FloorTripCount, FloorTripOverflow,
                          "omp_floor" + Twine(i) + ".tripcount", true);

    // Remember some values for later use.
    FloorCount.push_back(FloorTripCount);
    FloorRems.push_back(FloorTripRem);
  }

  // Generate the new loop nest, from the outermost to the innermost.
  std::vector<CanonicalLoopInfo *> Result;
  Result.reserve(NumLoops * 2);

  // The basic block of the surrounding loop that enters the nest generated
  // loop.
  BasicBlock *Enter = OutermostLoop->getPreheader();

  // The basic block of the surrounding loop where the inner code should
  // continue.
  BasicBlock *Continue = OutermostLoop->getAfter();

  // Where the next loop basic block should be inserted.
  BasicBlock *OutroInsertBefore = InnermostLoop->getExit();

  auto EmbeddNewLoop =
      [this, DL, F, InnerEnter, &Enter, &Continue, &OutroInsertBefore](
          Value *TripCount, const Twine &Name) -> CanonicalLoopInfo * {
    CanonicalLoopInfo *EmbeddedLoop = createLoopSkeleton(
        DL, TripCount, F, InnerEnter, OutroInsertBefore, Name);
    redirectTo(Enter, EmbeddedLoop->getPreheader(), DL);
    redirectTo(EmbeddedLoop->getAfter(), Continue, DL);

    // Setup the position where the next embedded loop connects to this loop.
    Enter = EmbeddedLoop->getBody();
    Continue = EmbeddedLoop->getLatch();
    OutroInsertBefore = EmbeddedLoop->getLatch();
    return EmbeddedLoop;
  };

  auto EmbeddNewLoops = [&Result, &EmbeddNewLoop](ArrayRef<Value *> TripCounts,
                                                  const Twine &NameBase) {
    for (auto P : enumerate(TripCounts)) {
      CanonicalLoopInfo *EmbeddedLoop =
          EmbeddNewLoop(P.value(), NameBase + Twine(P.index()));
      Result.push_back(EmbeddedLoop);
    }
  };

  EmbeddNewLoops(FloorCount, "floor");

  // Within the innermost floor loop, emit the code that computes the tile
  // sizes.
  Builder.SetInsertPoint(Enter->getTerminator());
  SmallVector<Value *, 4> TileCounts;
  for (int i = 0; i < NumLoops; ++i) {
    CanonicalLoopInfo *FloorLoop = Result[i];
    Value *TileSize = TileSizes[i];

    Value *FloorIsEpilogue =
        Builder.CreateICmpEQ(FloorLoop->getIndVar(), FloorCount[i]);
    Value *TileTripCount =
        Builder.CreateSelect(FloorIsEpilogue, FloorRems[i], TileSize);

    TileCounts.push_back(TileTripCount);
  }

  // Create the tile loops.
  EmbeddNewLoops(TileCounts, "tile");

  // Insert the inbetween code into the body.
  BasicBlock *BodyEnter = Enter;
  BasicBlock *BodyEntered = nullptr;
  for (std::pair<BasicBlock *, BasicBlock *> P : InbetweenCode) {
    BasicBlock *EnterBB = P.first;
    BasicBlock *ExitBB = P.second;

    if (BodyEnter)
      redirectTo(BodyEnter, EnterBB, DL);
    else
      redirectAllPredecessorsTo(BodyEntered, EnterBB, DL);

    BodyEnter = nullptr;
    BodyEntered = ExitBB;
  }

  // Append the original loop nest body into the generated loop nest body.
  if (BodyEnter)
    redirectTo(BodyEnter, InnerEnter, DL);
  else
    redirectAllPredecessorsTo(BodyEntered, InnerEnter, DL);
  redirectAllPredecessorsTo(InnerLatch, Continue, DL);

  // Replace the original induction variable with an induction variable computed
  // from the tile and floor induction variables.
  Builder.restoreIP(Result.back()->getBodyIP());
  for (int i = 0; i < NumLoops; ++i) {
    CanonicalLoopInfo *FloorLoop = Result[i];
    CanonicalLoopInfo *TileLoop = Result[NumLoops + i];
    Value *OrigIndVar = OrigIndVars[i];
    Value *Size = TileSizes[i];

    Value *Scale =
        Builder.CreateMul(Size, FloorLoop->getIndVar(), {}, /*HasNUW=*/true);
    Value *Shift =
        Builder.CreateAdd(Scale, TileLoop->getIndVar(), {}, /*HasNUW=*/true);
    OrigIndVar->replaceAllUsesWith(Shift);
  }

  // Remove unused parts of the original loops.
  removeUnusedBlocksFromParent(OldControlBBs);

  for (CanonicalLoopInfo *L : Loops)
    L->invalidate();

#ifndef NDEBUG
  for (CanonicalLoopInfo *GenL : Result)
    GenL->assertOK();
#endif
  return Result;
}

/// Attach metadata \p Properties to the basic block described by \p BB. If the
/// basic block already has metadata, the basic block properties are appended.
static void addBasicBlockMetadata(BasicBlock *BB,
                                  ArrayRef<Metadata *> Properties) {
  // Nothing to do if no property to attach.
  if (Properties.empty())
    return;

  LLVMContext &Ctx = BB->getContext();
  SmallVector<Metadata *> NewProperties;
  NewProperties.push_back(nullptr);

  // If the basic block already has metadata, prepend it to the new metadata.
  MDNode *Existing = BB->getTerminator()->getMetadata(LLVMContext::MD_loop);
  if (Existing)
    append_range(NewProperties, drop_begin(Existing->operands(), 1));

  append_range(NewProperties, Properties);
  MDNode *BasicBlockID = MDNode::getDistinct(Ctx, NewProperties);
  BasicBlockID->replaceOperandWith(0, BasicBlockID);

  BB->getTerminator()->setMetadata(LLVMContext::MD_loop, BasicBlockID);
}

/// Attach loop metadata \p Properties to the loop described by \p Loop. If the
/// loop already has metadata, the loop properties are appended.
static void addLoopMetadata(CanonicalLoopInfo *Loop,
                            ArrayRef<Metadata *> Properties) {
  assert(Loop->isValid() && "Expecting a valid CanonicalLoopInfo");

  // Attach metadata to the loop's latch
  BasicBlock *Latch = Loop->getLatch();
  assert(Latch && "A valid CanonicalLoopInfo must have a unique latch");
  addBasicBlockMetadata(Latch, Properties);
}

/// Attach llvm.access.group metadata to the memref instructions of \p Block
static void addSimdMetadata(BasicBlock *Block, MDNode *AccessGroup,
                            LoopInfo &LI) {
  for (Instruction &I : *Block) {
    if (I.mayReadOrWriteMemory()) {
      // TODO: This instruction may already have access group from
      // other pragmas e.g. #pragma clang loop vectorize.  Append
      // so that the existing metadata is not overwritten.
      I.setMetadata(LLVMContext::MD_access_group, AccessGroup);
    }
  }
}

void OpenMPIRBuilder::unrollLoopFull(DebugLoc, CanonicalLoopInfo *Loop) {
  LLVMContext &Ctx = Builder.getContext();
  addLoopMetadata(
      Loop, {MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable")),
             MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.full"))});
}

void OpenMPIRBuilder::unrollLoopHeuristic(DebugLoc, CanonicalLoopInfo *Loop) {
  LLVMContext &Ctx = Builder.getContext();
  addLoopMetadata(
      Loop, {
                MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable")),
            });
}

void OpenMPIRBuilder::createIfVersion(CanonicalLoopInfo *CanonicalLoop,
                                      Value *IfCond, ValueToValueMapTy &VMap,
                                      const Twine &NamePrefix) {
  Function *F = CanonicalLoop->getFunction();

  // Define where if branch should be inserted
  Instruction *SplitBefore;
  if (Instruction::classof(IfCond)) {
    SplitBefore = dyn_cast<Instruction>(IfCond);
  } else {
    SplitBefore = CanonicalLoop->getPreheader()->getTerminator();
  }

  // TODO: We should not rely on pass manager. Currently we use pass manager
  // only for getting llvm::Loop which corresponds to given CanonicalLoopInfo
  // object. We should have a method  which returns all blocks between
  // CanonicalLoopInfo::getHeader() and CanonicalLoopInfo::getAfter()
  FunctionAnalysisManager FAM;
  FAM.registerPass([]() { return DominatorTreeAnalysis(); });
  FAM.registerPass([]() { return LoopAnalysis(); });
  FAM.registerPass([]() { return PassInstrumentationAnalysis(); });

  // Get the loop which needs to be cloned
  LoopAnalysis LIA;
  LoopInfo &&LI = LIA.run(*F, FAM);
  Loop *L = LI.getLoopFor(CanonicalLoop->getHeader());

  // Create additional blocks for the if statement
  BasicBlock *Head = SplitBefore->getParent();
  Instruction *HeadOldTerm = Head->getTerminator();
  llvm::LLVMContext &C = Head->getContext();
  llvm::BasicBlock *ThenBlock = llvm::BasicBlock::Create(
      C, NamePrefix + ".if.then", Head->getParent(), Head->getNextNode());
  llvm::BasicBlock *ElseBlock = llvm::BasicBlock::Create(
      C, NamePrefix + ".if.else", Head->getParent(), CanonicalLoop->getExit());

  // Create if condition branch.
  Builder.SetInsertPoint(HeadOldTerm);
  Instruction *BrInstr =
      Builder.CreateCondBr(IfCond, ThenBlock, /*ifFalse*/ ElseBlock);
  InsertPointTy IP{BrInstr->getParent(), ++BrInstr->getIterator()};
  // Then block contains branch to omp loop which needs to be vectorized
  spliceBB(IP, ThenBlock, false);
  ThenBlock->replaceSuccessorsPhiUsesWith(Head, ThenBlock);

  Builder.SetInsertPoint(ElseBlock);

  // Clone loop for the else branch
  SmallVector<BasicBlock *, 8> NewBlocks;

  VMap[CanonicalLoop->getPreheader()] = ElseBlock;
  for (BasicBlock *Block : L->getBlocks()) {
    BasicBlock *NewBB = CloneBasicBlock(Block, VMap, "", F);
    NewBB->moveBefore(CanonicalLoop->getExit());
    VMap[Block] = NewBB;
    NewBlocks.push_back(NewBB);
  }
  remapInstructionsInBlocks(NewBlocks, VMap);
  Builder.CreateBr(NewBlocks.front());
}

unsigned
OpenMPIRBuilder::getOpenMPDefaultSimdAlign(const Triple &TargetTriple,
                                           const StringMap<bool> &Features) {
  if (TargetTriple.isX86()) {
    if (Features.lookup("avx512f"))
      return 512;
    else if (Features.lookup("avx"))
      return 256;
    return 128;
  }
  if (TargetTriple.isPPC())
    return 128;
  if (TargetTriple.isWasm())
    return 128;
  return 0;
}

void OpenMPIRBuilder::applySimd(CanonicalLoopInfo *CanonicalLoop,
                                MapVector<Value *, Value *> AlignedVars,
                                Value *IfCond, OrderKind Order,
                                ConstantInt *Simdlen, ConstantInt *Safelen) {
  LLVMContext &Ctx = Builder.getContext();

  Function *F = CanonicalLoop->getFunction();

  // TODO: We should not rely on pass manager. Currently we use pass manager
  // only for getting llvm::Loop which corresponds to given CanonicalLoopInfo
  // object. We should have a method  which returns all blocks between
  // CanonicalLoopInfo::getHeader() and CanonicalLoopInfo::getAfter()
  FunctionAnalysisManager FAM;
  FAM.registerPass([]() { return DominatorTreeAnalysis(); });
  FAM.registerPass([]() { return LoopAnalysis(); });
  FAM.registerPass([]() { return PassInstrumentationAnalysis(); });

  LoopAnalysis LIA;
  LoopInfo &&LI = LIA.run(*F, FAM);

  Loop *L = LI.getLoopFor(CanonicalLoop->getHeader());
  if (AlignedVars.size()) {
    InsertPointTy IP = Builder.saveIP();
    Builder.SetInsertPoint(CanonicalLoop->getPreheader()->getTerminator());
    for (auto &AlignedItem : AlignedVars) {
      Value *AlignedPtr = AlignedItem.first;
      Value *Alignment = AlignedItem.second;
      Builder.CreateAlignmentAssumption(F->getParent()->getDataLayout(),
                                        AlignedPtr, Alignment);
    }
    Builder.restoreIP(IP);
  }

  if (IfCond) {
    ValueToValueMapTy VMap;
    createIfVersion(CanonicalLoop, IfCond, VMap, "simd");
    // Add metadata to the cloned loop which disables vectorization
    Value *MappedLatch = VMap.lookup(CanonicalLoop->getLatch());
    assert(MappedLatch &&
           "Cannot find value which corresponds to original loop latch");
    assert(isa<BasicBlock>(MappedLatch) &&
           "Cannot cast mapped latch block value to BasicBlock");
    BasicBlock *NewLatchBlock = dyn_cast<BasicBlock>(MappedLatch);
    ConstantAsMetadata *BoolConst =
        ConstantAsMetadata::get(ConstantInt::getFalse(Type::getInt1Ty(Ctx)));
    addBasicBlockMetadata(
        NewLatchBlock,
        {MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.vectorize.enable"),
                           BoolConst})});
  }

  SmallSet<BasicBlock *, 8> Reachable;

  // Get the basic blocks from the loop in which memref instructions
  // can be found.
  // TODO: Generalize getting all blocks inside a CanonicalizeLoopInfo,
  // preferably without running any passes.
  for (BasicBlock *Block : L->getBlocks()) {
    if (Block == CanonicalLoop->getCond() ||
        Block == CanonicalLoop->getHeader())
      continue;
    Reachable.insert(Block);
  }

  SmallVector<Metadata *> LoopMDList;

  // In presence of finite 'safelen', it may be unsafe to mark all
  // the memory instructions parallel, because loop-carried
  // dependences of 'safelen' iterations are possible.
  // If clause order(concurrent) is specified then the memory instructions
  // are marked parallel even if 'safelen' is finite.
  if ((Safelen == nullptr) || (Order == OrderKind::OMP_ORDER_concurrent)) {
    // Add access group metadata to memory-access instructions.
    MDNode *AccessGroup = MDNode::getDistinct(Ctx, {});
    for (BasicBlock *BB : Reachable)
      addSimdMetadata(BB, AccessGroup, LI);
    // TODO:  If the loop has existing parallel access metadata, have
    // to combine two lists.
    LoopMDList.push_back(MDNode::get(
        Ctx, {MDString::get(Ctx, "llvm.loop.parallel_accesses"), AccessGroup}));
  }

  // Use the above access group metadata to create loop level
  // metadata, which should be distinct for each loop.
  ConstantAsMetadata *BoolConst =
      ConstantAsMetadata::get(ConstantInt::getTrue(Type::getInt1Ty(Ctx)));
  LoopMDList.push_back(MDNode::get(
      Ctx, {MDString::get(Ctx, "llvm.loop.vectorize.enable"), BoolConst}));

  if (Simdlen || Safelen) {
    // If both simdlen and safelen clauses are specified, the value of the
    // simdlen parameter must be less than or equal to the value of the safelen
    // parameter. Therefore, use safelen only in the absence of simdlen.
    ConstantInt *VectorizeWidth = Simdlen == nullptr ? Safelen : Simdlen;
    LoopMDList.push_back(
        MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.vectorize.width"),
                          ConstantAsMetadata::get(VectorizeWidth)}));
  }

  addLoopMetadata(CanonicalLoop, LoopMDList);
}

/// Create the TargetMachine object to query the backend for optimization
/// preferences.
///
/// Ideally, this would be passed from the front-end to the OpenMPBuilder, but
/// e.g. Clang does not pass it to its CodeGen layer and creates it only when
/// needed for the LLVM pass pipline. We use some default options to avoid
/// having to pass too many settings from the frontend that probably do not
/// matter.
///
/// Currently, TargetMachine is only used sometimes by the unrollLoopPartial
/// method. If we are going to use TargetMachine for more purposes, especially
/// those that are sensitive to TargetOptions, RelocModel and CodeModel, it
/// might become be worth requiring front-ends to pass on their TargetMachine,
/// or at least cache it between methods. Note that while fontends such as Clang
/// have just a single main TargetMachine per translation unit, "target-cpu" and
/// "target-features" that determine the TargetMachine are per-function and can
/// be overrided using __attribute__((target("OPTIONS"))).
static std::unique_ptr<TargetMachine>
createTargetMachine(Function *F, CodeGenOptLevel OptLevel) {
  Module *M = F->getParent();

  StringRef CPU = F->getFnAttribute("target-cpu").getValueAsString();
  StringRef Features = F->getFnAttribute("target-features").getValueAsString();
  const std::string &Triple = M->getTargetTriple();

  std::string Error;
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget)
    return {};

  llvm::TargetOptions Options;
  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      Triple, CPU, Features, Options, /*RelocModel=*/std::nullopt,
      /*CodeModel=*/std::nullopt, OptLevel));
}

/// Heuristically determine the best-performant unroll factor for \p CLI. This
/// depends on the target processor. We are re-using the same heuristics as the
/// LoopUnrollPass.
static int32_t computeHeuristicUnrollFactor(CanonicalLoopInfo *CLI) {
  Function *F = CLI->getFunction();

  // Assume the user requests the most aggressive unrolling, even if the rest of
  // the code is optimized using a lower setting.
  CodeGenOptLevel OptLevel = CodeGenOptLevel::Aggressive;
  std::unique_ptr<TargetMachine> TM = createTargetMachine(F, OptLevel);

  FunctionAnalysisManager FAM;
  FAM.registerPass([]() { return TargetLibraryAnalysis(); });
  FAM.registerPass([]() { return AssumptionAnalysis(); });
  FAM.registerPass([]() { return DominatorTreeAnalysis(); });
  FAM.registerPass([]() { return LoopAnalysis(); });
  FAM.registerPass([]() { return ScalarEvolutionAnalysis(); });
  FAM.registerPass([]() { return PassInstrumentationAnalysis(); });
  TargetIRAnalysis TIRA;
  if (TM)
    TIRA = TargetIRAnalysis(
        [&](const Function &F) { return TM->getTargetTransformInfo(F); });
  FAM.registerPass([&]() { return TIRA; });

  TargetIRAnalysis::Result &&TTI = TIRA.run(*F, FAM);
  ScalarEvolutionAnalysis SEA;
  ScalarEvolution &&SE = SEA.run(*F, FAM);
  DominatorTreeAnalysis DTA;
  DominatorTree &&DT = DTA.run(*F, FAM);
  LoopAnalysis LIA;
  LoopInfo &&LI = LIA.run(*F, FAM);
  AssumptionAnalysis ACT;
  AssumptionCache &&AC = ACT.run(*F, FAM);
  OptimizationRemarkEmitter ORE{F};

  Loop *L = LI.getLoopFor(CLI->getHeader());
  assert(L && "Expecting CanonicalLoopInfo to be recognized as a loop");

  TargetTransformInfo::UnrollingPreferences UP =
      gatherUnrollingPreferences(L, SE, TTI,
                                 /*BlockFrequencyInfo=*/nullptr,
                                 /*ProfileSummaryInfo=*/nullptr, ORE, static_cast<int>(OptLevel),
                                 /*UserThreshold=*/std::nullopt,
                                 /*UserCount=*/std::nullopt,
                                 /*UserAllowPartial=*/true,
                                 /*UserAllowRuntime=*/true,
                                 /*UserUpperBound=*/std::nullopt,
                                 /*UserFullUnrollMaxCount=*/std::nullopt);

  UP.Force = true;

  // Account for additional optimizations taking place before the LoopUnrollPass
  // would unroll the loop.
  UP.Threshold *= UnrollThresholdFactor;
  UP.PartialThreshold *= UnrollThresholdFactor;

  // Use normal unroll factors even if the rest of the code is optimized for
  // size.
  UP.OptSizeThreshold = UP.Threshold;
  UP.PartialOptSizeThreshold = UP.PartialThreshold;

  LLVM_DEBUG(dbgs() << "Unroll heuristic thresholds:\n"
                    << "  Threshold=" << UP.Threshold << "\n"
                    << "  PartialThreshold=" << UP.PartialThreshold << "\n"
                    << "  OptSizeThreshold=" << UP.OptSizeThreshold << "\n"
                    << "  PartialOptSizeThreshold="
                    << UP.PartialOptSizeThreshold << "\n");

  // Disable peeling.
  TargetTransformInfo::PeelingPreferences PP =
      gatherPeelingPreferences(L, SE, TTI,
                               /*UserAllowPeeling=*/false,
                               /*UserAllowProfileBasedPeeling=*/false,
                               /*UnrollingSpecficValues=*/false);

  SmallPtrSet<const Value *, 32> EphValues;
  CodeMetrics::collectEphemeralValues(L, &AC, EphValues);

  // Assume that reads and writes to stack variables can be eliminated by
  // Mem2Reg, SROA or LICM. That is, don't count them towards the loop body's
  // size.
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      Value *Ptr;
      if (auto *Load = dyn_cast<LoadInst>(&I)) {
        Ptr = Load->getPointerOperand();
      } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
        Ptr = Store->getPointerOperand();
      } else
        continue;

      Ptr = Ptr->stripPointerCasts();

      if (auto *Alloca = dyn_cast<AllocaInst>(Ptr)) {
        if (Alloca->getParent() == &F->getEntryBlock())
          EphValues.insert(&I);
      }
    }
  }

  UnrollCostEstimator UCE(L, TTI, EphValues, UP.BEInsns);

  // Loop is not unrollable if the loop contains certain instructions.
  if (!UCE.canUnroll() || UCE.Convergent) {
    LLVM_DEBUG(dbgs() << "Loop not considered unrollable\n");
    return 1;
  }

  unsigned LoopSize = UCE.getRolledLoopSize();
  LLVM_DEBUG(dbgs() << "Estimated loop size is " << LoopSize << "\n");

  // TODO: Determine trip count of \p CLI if constant, computeUnrollCount might
  // be able to use it.
  int TripCount = 0;
  int MaxTripCount = 0;
  bool MaxOrZero = false;
  unsigned TripMultiple = 0;

  bool UseUpperBound = false;
  computeUnrollCount(L, TTI, DT, &LI, &AC, SE, EphValues, &ORE, TripCount,
                     MaxTripCount, MaxOrZero, TripMultiple, UCE, UP, PP,
                     UseUpperBound);
  unsigned Factor = UP.Count;
  LLVM_DEBUG(dbgs() << "Suggesting unroll factor of " << Factor << "\n");

  // This function returns 1 to signal to not unroll a loop.
  if (Factor == 0)
    return 1;
  return Factor;
}

void OpenMPIRBuilder::unrollLoopPartial(DebugLoc DL, CanonicalLoopInfo *Loop,
                                        int32_t Factor,
                                        CanonicalLoopInfo **UnrolledCLI) {
  assert(Factor >= 0 && "Unroll factor must not be negative");

  Function *F = Loop->getFunction();
  LLVMContext &Ctx = F->getContext();

  // If the unrolled loop is not used for another loop-associated directive, it
  // is sufficient to add metadata for the LoopUnrollPass.
  if (!UnrolledCLI) {
    SmallVector<Metadata *, 2> LoopMetadata;
    LoopMetadata.push_back(
        MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable")));

    if (Factor >= 1) {
      ConstantAsMetadata *FactorConst = ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(Ctx), APInt(32, Factor)));
      LoopMetadata.push_back(MDNode::get(
          Ctx, {MDString::get(Ctx, "llvm.loop.unroll.count"), FactorConst}));
    }

    addLoopMetadata(Loop, LoopMetadata);
    return;
  }

  // Heuristically determine the unroll factor.
  if (Factor == 0)
    Factor = computeHeuristicUnrollFactor(Loop);

  // No change required with unroll factor 1.
  if (Factor == 1) {
    *UnrolledCLI = Loop;
    return;
  }

  assert(Factor >= 2 &&
         "unrolling only makes sense with a factor of 2 or larger");

  Type *IndVarTy = Loop->getIndVarType();

  // Apply partial unrolling by tiling the loop by the unroll-factor, then fully
  // unroll the inner loop.
  Value *FactorVal =
      ConstantInt::get(IndVarTy, APInt(IndVarTy->getIntegerBitWidth(), Factor,
                                       /*isSigned=*/false));
  std::vector<CanonicalLoopInfo *> LoopNest =
      tileLoops(DL, {Loop}, {FactorVal});
  assert(LoopNest.size() == 2 && "Expect 2 loops after tiling");
  *UnrolledCLI = LoopNest[0];
  CanonicalLoopInfo *InnerLoop = LoopNest[1];

  // LoopUnrollPass can only fully unroll loops with constant trip count.
  // Unroll by the unroll factor with a fallback epilog for the remainder
  // iterations if necessary.
  ConstantAsMetadata *FactorConst = ConstantAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(Ctx), APInt(32, Factor)));
  addLoopMetadata(
      InnerLoop,
      {MDNode::get(Ctx, MDString::get(Ctx, "llvm.loop.unroll.enable")),
       MDNode::get(
           Ctx, {MDString::get(Ctx, "llvm.loop.unroll.count"), FactorConst})});

#ifndef NDEBUG
  (*UnrolledCLI)->assertOK();
#endif
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createCopyPrivate(const LocationDescription &Loc,
                                   llvm::Value *BufSize, llvm::Value *CpyBuf,
                                   llvm::Value *CpyFn, llvm::Value *DidIt) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);

  llvm::Value *DidItLD = Builder.CreateLoad(Builder.getInt32Ty(), DidIt);

  Value *Args[] = {Ident, ThreadId, BufSize, CpyBuf, CpyFn, DidItLD};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_copyprivate);
  Builder.CreateCall(Fn, Args);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createSingle(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    FinalizeCallbackTy FiniCB, bool IsNowait, llvm::Value *DidIt) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  // If needed (i.e. not null), initialize `DidIt` with 0
  if (DidIt) {
    Builder.CreateStore(Builder.getInt32(0), DidIt);
  }

  Directive OMPD = Directive::OMPD_single;
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId};

  Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_single);
  Instruction *EntryCall = Builder.CreateCall(EntryRTLFn, Args);

  Function *ExitRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_single);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  // generates the following:
  // if (__kmpc_single()) {
  //		.... single region ...
  // 		__kmpc_end_single
  // }
  // __kmpc_barrier

  EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                       /*Conditional*/ true,
                       /*hasFinalize*/ true);
  if (!IsNowait)
    createBarrier(LocationDescription(Builder.saveIP(), Loc.DL),
                  omp::Directive::OMPD_unknown, /* ForceSimpleCall */ false,
                  /* CheckCancelFlag */ false);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createCritical(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    FinalizeCallbackTy FiniCB, StringRef CriticalName, Value *HintInst) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_critical;
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *LockVar = getOMPCriticalRegionLock(CriticalName);
  Value *Args[] = {Ident, ThreadId, LockVar};

  SmallVector<llvm::Value *, 4> EnterArgs(std::begin(Args), std::end(Args));
  Function *RTFn = nullptr;
  if (HintInst) {
    // Add Hint to entry Args and create call
    EnterArgs.push_back(HintInst);
    RTFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_critical_with_hint);
  } else {
    RTFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_critical);
  }
  Instruction *EntryCall = Builder.CreateCall(RTFn, EnterArgs);

  Function *ExitRTLFn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_critical);
  Instruction *ExitCall = Builder.CreateCall(ExitRTLFn, Args);

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ false, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createOrderedDepend(const LocationDescription &Loc,
                                     InsertPointTy AllocaIP, unsigned NumLoops,
                                     ArrayRef<llvm::Value *> StoreValues,
                                     const Twine &Name, bool IsDependSource) {
  assert(
      llvm::all_of(StoreValues,
                   [](Value *SV) { return SV->getType()->isIntegerTy(64); }) &&
      "OpenMP runtime requires depend vec with i64 type");

  if (!updateToLocation(Loc))
    return Loc.IP;

  // Allocate space for vector and generate alloc instruction.
  auto *ArrI64Ty = ArrayType::get(Int64, NumLoops);
  Builder.restoreIP(AllocaIP);
  AllocaInst *ArgsBase = Builder.CreateAlloca(ArrI64Ty, nullptr, Name);
  ArgsBase->setAlignment(Align(8));
  Builder.restoreIP(Loc.IP);

  // Store the index value with offset in depend vector.
  for (unsigned I = 0; I < NumLoops; ++I) {
    Value *DependAddrGEPIter = Builder.CreateInBoundsGEP(
        ArrI64Ty, ArgsBase, {Builder.getInt64(0), Builder.getInt64(I)});
    StoreInst *STInst = Builder.CreateStore(StoreValues[I], DependAddrGEPIter);
    STInst->setAlignment(Align(8));
  }

  Value *DependBaseAddrGEP = Builder.CreateInBoundsGEP(
      ArrI64Ty, ArgsBase, {Builder.getInt64(0), Builder.getInt64(0)});

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {Ident, ThreadId, DependBaseAddrGEP};

  Function *RTLFn = nullptr;
  if (IsDependSource)
    RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_doacross_post);
  else
    RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_doacross_wait);
  Builder.CreateCall(RTLFn, Args);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createOrderedThreadsSimd(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    FinalizeCallbackTy FiniCB, bool IsThreads) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_ordered;
  Instruction *EntryCall = nullptr;
  Instruction *ExitCall = nullptr;

  if (IsThreads) {
    uint32_t SrcLocStrSize;
    Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
    Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
    Value *ThreadId = getOrCreateThreadID(Ident);
    Value *Args[] = {Ident, ThreadId};

    Function *EntryRTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_ordered);
    EntryCall = Builder.CreateCall(EntryRTLFn, Args);

    Function *ExitRTLFn =
        getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_ordered);
    ExitCall = Builder.CreateCall(ExitRTLFn, Args);
  }

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ false, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::EmitOMPInlinedRegion(
    Directive OMPD, Instruction *EntryCall, Instruction *ExitCall,
    BodyGenCallbackTy BodyGenCB, FinalizeCallbackTy FiniCB, bool Conditional,
    bool HasFinalize, bool IsCancellable) {

  if (HasFinalize)
    FinalizationStack.push_back({FiniCB, OMPD, IsCancellable});

  // Create inlined region's entry and body blocks, in preparation
  // for conditional creation
  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Instruction *SplitPos = EntryBB->getTerminator();
  if (!isa_and_nonnull<BranchInst>(SplitPos))
    SplitPos = new UnreachableInst(Builder.getContext(), EntryBB);
  BasicBlock *ExitBB = EntryBB->splitBasicBlock(SplitPos, "omp_region.end");
  BasicBlock *FiniBB =
      EntryBB->splitBasicBlock(EntryBB->getTerminator(), "omp_region.finalize");

  Builder.SetInsertPoint(EntryBB->getTerminator());
  emitCommonDirectiveEntry(OMPD, EntryCall, ExitBB, Conditional);

  // generate body
  BodyGenCB(/* AllocaIP */ InsertPointTy(),
            /* CodeGenIP */ Builder.saveIP());

  // emit exit call and do any needed finalization.
  auto FinIP = InsertPointTy(FiniBB, FiniBB->getFirstInsertionPt());
  assert(FiniBB->getTerminator()->getNumSuccessors() == 1 &&
         FiniBB->getTerminator()->getSuccessor(0) == ExitBB &&
         "Unexpected control flow graph state!!");
  emitCommonDirectiveExit(OMPD, FinIP, ExitCall, HasFinalize);
  assert(FiniBB->getUniquePredecessor()->getUniqueSuccessor() == FiniBB &&
         "Unexpected Control Flow State!");
  MergeBlockIntoPredecessor(FiniBB);

  // If we are skipping the region of a non conditional, remove the exit
  // block, and clear the builder's insertion point.
  assert(SplitPos->getParent() == ExitBB &&
         "Unexpected Insertion point location!");
  auto merged = MergeBlockIntoPredecessor(ExitBB);
  BasicBlock *ExitPredBB = SplitPos->getParent();
  auto InsertBB = merged ? ExitPredBB : ExitBB;
  if (!isa_and_nonnull<BranchInst>(SplitPos))
    SplitPos->eraseFromParent();
  Builder.SetInsertPoint(InsertBB);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitCommonDirectiveEntry(
    Directive OMPD, Value *EntryCall, BasicBlock *ExitBB, bool Conditional) {
  // if nothing to do, Return current insertion point.
  if (!Conditional || !EntryCall)
    return Builder.saveIP();

  BasicBlock *EntryBB = Builder.GetInsertBlock();
  Value *CallBool = Builder.CreateIsNotNull(EntryCall);
  auto *ThenBB = BasicBlock::Create(M.getContext(), "omp_region.body");
  auto *UI = new UnreachableInst(Builder.getContext(), ThenBB);

  // Emit thenBB and set the Builder's insertion point there for
  // body generation next. Place the block after the current block.
  Function *CurFn = EntryBB->getParent();
  CurFn->insert(std::next(EntryBB->getIterator()), ThenBB);

  // Move Entry branch to end of ThenBB, and replace with conditional
  // branch (If-stmt)
  Instruction *EntryBBTI = EntryBB->getTerminator();
  Builder.CreateCondBr(CallBool, ThenBB, ExitBB);
  EntryBBTI->removeFromParent();
  Builder.SetInsertPoint(UI);
  Builder.Insert(EntryBBTI);
  UI->eraseFromParent();
  Builder.SetInsertPoint(ThenBB->getTerminator());

  // return an insertion point to ExitBB.
  return IRBuilder<>::InsertPoint(ExitBB, ExitBB->getFirstInsertionPt());
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::emitCommonDirectiveExit(
    omp::Directive OMPD, InsertPointTy FinIP, Instruction *ExitCall,
    bool HasFinalize) {

  Builder.restoreIP(FinIP);

  // If there is finalization to do, emit it before the exit call
  if (HasFinalize) {
    assert(!FinalizationStack.empty() &&
           "Unexpected finalization stack state!");

    FinalizationInfo Fi = FinalizationStack.pop_back_val();
    assert(Fi.DK == OMPD && "Unexpected Directive for Finalization call!");

    Fi.FiniCB(FinIP);

    BasicBlock *FiniBB = FinIP.getBlock();
    Instruction *FiniBBTI = FiniBB->getTerminator();

    // set Builder IP for call creation
    Builder.SetInsertPoint(FiniBBTI);
  }

  if (!ExitCall)
    return Builder.saveIP();

  // place the Exitcall as last instruction before Finalization block terminator
  ExitCall->removeFromParent();
  Builder.Insert(ExitCall);

  return IRBuilder<>::InsertPoint(ExitCall->getParent(),
                                  ExitCall->getIterator());
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createCopyinClauseBlocks(
    InsertPointTy IP, Value *MasterAddr, Value *PrivateAddr,
    llvm::IntegerType *IntPtrTy, bool BranchtoEnd) {
  if (!IP.isSet())
    return IP;

  IRBuilder<>::InsertPointGuard IPG(Builder);

  // creates the following CFG structure
  //	   OMP_Entry : (MasterAddr != PrivateAddr)?
  //       F     T
  //       |      \
  //       |     copin.not.master
  //       |      /
  //       v     /
  //   copyin.not.master.end
  //		     |
  //         v
  //   OMP.Entry.Next

  BasicBlock *OMP_Entry = IP.getBlock();
  Function *CurFn = OMP_Entry->getParent();
  BasicBlock *CopyBegin =
      BasicBlock::Create(M.getContext(), "copyin.not.master", CurFn);
  BasicBlock *CopyEnd = nullptr;

  // If entry block is terminated, split to preserve the branch to following
  // basic block (i.e. OMP.Entry.Next), otherwise, leave everything as is.
  if (isa_and_nonnull<BranchInst>(OMP_Entry->getTerminator())) {
    CopyEnd = OMP_Entry->splitBasicBlock(OMP_Entry->getTerminator(),
                                         "copyin.not.master.end");
    OMP_Entry->getTerminator()->eraseFromParent();
  } else {
    CopyEnd =
        BasicBlock::Create(M.getContext(), "copyin.not.master.end", CurFn);
  }

  Builder.SetInsertPoint(OMP_Entry);
  Value *MasterPtr = Builder.CreatePtrToInt(MasterAddr, IntPtrTy);
  Value *PrivatePtr = Builder.CreatePtrToInt(PrivateAddr, IntPtrTy);
  Value *cmp = Builder.CreateICmpNE(MasterPtr, PrivatePtr);
  Builder.CreateCondBr(cmp, CopyBegin, CopyEnd);

  Builder.SetInsertPoint(CopyBegin);
  if (BranchtoEnd)
    Builder.SetInsertPoint(Builder.CreateBr(CopyEnd));

  return Builder.saveIP();
}

CallInst *OpenMPIRBuilder::createOMPAlloc(const LocationDescription &Loc,
                                          Value *Size, Value *Allocator,
                                          std::string Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {ThreadId, Size, Allocator};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_alloc);

  return Builder.CreateCall(Fn, Args, Name);
}

CallInst *OpenMPIRBuilder::createOMPFree(const LocationDescription &Loc,
                                         Value *Addr, Value *Allocator,
                                         std::string Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {ThreadId, Addr, Allocator};
  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_free);
  return Builder.CreateCall(Fn, Args, Name);
}

CallInst *OpenMPIRBuilder::createOMPInteropInit(
    const LocationDescription &Loc, Value *InteropVar,
    omp::OMPInteropType InteropType, Value *Device, Value *NumDependences,
    Value *DependenceAddress, bool HaveNowaitClause) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  if (Device == nullptr)
    Device = ConstantInt::get(Int32, -1);
  Constant *InteropTypeVal = ConstantInt::get(Int32, (int)InteropType);
  if (NumDependences == nullptr) {
    NumDependences = ConstantInt::get(Int32, 0);
    PointerType *PointerTypeVar = Type::getInt8PtrTy(M.getContext());
    DependenceAddress = ConstantPointerNull::get(PointerTypeVar);
  }
  Value *HaveNowaitClauseVal = ConstantInt::get(Int32, HaveNowaitClause);
  Value *Args[] = {
      Ident,  ThreadId,       InteropVar,        InteropTypeVal,
      Device, NumDependences, DependenceAddress, HaveNowaitClauseVal};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___tgt_interop_init);

  return Builder.CreateCall(Fn, Args);
}

CallInst *OpenMPIRBuilder::createOMPInteropDestroy(
    const LocationDescription &Loc, Value *InteropVar, Value *Device,
    Value *NumDependences, Value *DependenceAddress, bool HaveNowaitClause) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  if (Device == nullptr)
    Device = ConstantInt::get(Int32, -1);
  if (NumDependences == nullptr) {
    NumDependences = ConstantInt::get(Int32, 0);
    PointerType *PointerTypeVar = Type::getInt8PtrTy(M.getContext());
    DependenceAddress = ConstantPointerNull::get(PointerTypeVar);
  }
  Value *HaveNowaitClauseVal = ConstantInt::get(Int32, HaveNowaitClause);
  Value *Args[] = {
      Ident,          ThreadId,          InteropVar,         Device,
      NumDependences, DependenceAddress, HaveNowaitClauseVal};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___tgt_interop_destroy);

  return Builder.CreateCall(Fn, Args);
}

CallInst *OpenMPIRBuilder::createOMPInteropUse(const LocationDescription &Loc,
                                               Value *InteropVar, Value *Device,
                                               Value *NumDependences,
                                               Value *DependenceAddress,
                                               bool HaveNowaitClause) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  if (Device == nullptr)
    Device = ConstantInt::get(Int32, -1);
  if (NumDependences == nullptr) {
    NumDependences = ConstantInt::get(Int32, 0);
    PointerType *PointerTypeVar = Type::getInt8PtrTy(M.getContext());
    DependenceAddress = ConstantPointerNull::get(PointerTypeVar);
  }
  Value *HaveNowaitClauseVal = ConstantInt::get(Int32, HaveNowaitClause);
  Value *Args[] = {
      Ident,          ThreadId,          InteropVar,         Device,
      NumDependences, DependenceAddress, HaveNowaitClauseVal};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___tgt_interop_use);

  return Builder.CreateCall(Fn, Args);
}

CallInst *OpenMPIRBuilder::createCachedThreadPrivate(
    const LocationDescription &Loc, llvm::Value *Pointer,
    llvm::ConstantInt *Size, const llvm::Twine &Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Constant *ThreadPrivateCache =
      getOrCreateInternalVariable(Int8PtrPtr, Name.str());
  llvm::Value *Args[] = {Ident, ThreadId, Pointer, Size, ThreadPrivateCache};

  Function *Fn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_threadprivate_cached);

  return Builder.CreateCall(Fn, Args);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createTargetInit(const LocationDescription &Loc, bool IsSPMD) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Constant *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  ConstantInt *IsSPMDVal = ConstantInt::getSigned(
      IntegerType::getInt8Ty(Int8->getContext()),
      IsSPMD ? OMP_TGT_EXEC_MODE_SPMD : OMP_TGT_EXEC_MODE_GENERIC);
  ConstantInt *UseGenericStateMachineVal = ConstantInt::getSigned(
      IntegerType::getInt8Ty(Int8->getContext()), !IsSPMD);
  ConstantInt *MayUseNestedParallelismVal =
      ConstantInt::getSigned(IntegerType::getInt8Ty(Int8->getContext()), true);
  ConstantInt *DebugIndentionLevelVal =
      ConstantInt::getSigned(IntegerType::getInt16Ty(Int8->getContext()), 0);

  // We need to strip the debug prefix to get the correct kernel name.
  Function *Kernel = Builder.GetInsertBlock()->getParent();
  StringRef KernelName = Kernel->getName();
  const std::string DebugPrefix = "_debug__";
  if (KernelName.ends_with(DebugPrefix))
    KernelName = KernelName.drop_back(DebugPrefix.length());

  Function *Fn = getOrCreateRuntimeFunctionPtr(
      omp::RuntimeFunction::OMPRTL___kmpc_target_init);
  const DataLayout &DL = Fn->getParent()->getDataLayout();

  Twine DynamicEnvironmentName = KernelName + "_dynamic_environment";
  Constant *DynamicEnvironmentInitializer =
      ConstantStruct::get(DynamicEnvironment, {DebugIndentionLevelVal});
  GlobalVariable *DynamicEnvironmentGV = new GlobalVariable(
      M, DynamicEnvironment, /*IsConstant=*/false, GlobalValue::WeakODRLinkage,
      DynamicEnvironmentInitializer, DynamicEnvironmentName,
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
      DL.getDefaultGlobalsAddressSpace());
  DynamicEnvironmentGV->setVisibility(GlobalValue::ProtectedVisibility);

  Constant *DynamicEnvironment =
      DynamicEnvironmentGV->getType() == DynamicEnvironmentPtr
          ? DynamicEnvironmentGV
          : ConstantExpr::getAddrSpaceCast(DynamicEnvironmentGV,
                                           DynamicEnvironmentPtr);

  Constant *ConfigurationEnvironmentInitializer = ConstantStruct::get(
      ConfigurationEnvironment, {
                                    UseGenericStateMachineVal,
                                    MayUseNestedParallelismVal,
                                    IsSPMDVal,
                                });
  Constant *KernelEnvironmentInitializer = ConstantStruct::get(
      KernelEnvironment, {
                             ConfigurationEnvironmentInitializer,
                             Ident,
                             DynamicEnvironment,
                         });
  Twine KernelEnvironmentName = KernelName + "_kernel_environment";
  GlobalVariable *KernelEnvironmentGV = new GlobalVariable(
      M, KernelEnvironment, /*IsConstant=*/true, GlobalValue::WeakODRLinkage,
      KernelEnvironmentInitializer, KernelEnvironmentName,
      /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal,
      DL.getDefaultGlobalsAddressSpace());
  KernelEnvironmentGV->setVisibility(GlobalValue::ProtectedVisibility);

  Constant *KernelEnvironment =
      KernelEnvironmentGV->getType() == KernelEnvironmentPtr
          ? KernelEnvironmentGV
          : ConstantExpr::getAddrSpaceCast(KernelEnvironmentGV,
                                           KernelEnvironmentPtr);
  CallInst *ThreadKind = Builder.CreateCall(Fn, {KernelEnvironment});

  Value *ExecUserCode = Builder.CreateICmpEQ(
      ThreadKind, ConstantInt::get(ThreadKind->getType(), -1),
      "exec_user_code");

  // ThreadKind = __kmpc_target_init(...)
  // if (ThreadKind == -1)
  //   user_code
  // else
  //   return;

  auto *UI = Builder.CreateUnreachable();
  BasicBlock *CheckBB = UI->getParent();
  BasicBlock *UserCodeEntryBB = CheckBB->splitBasicBlock(UI, "user_code.entry");

  BasicBlock *WorkerExitBB = BasicBlock::Create(
      CheckBB->getContext(), "worker.exit", CheckBB->getParent());
  Builder.SetInsertPoint(WorkerExitBB);
  Builder.CreateRetVoid();

  auto *CheckBBTI = CheckBB->getTerminator();
  Builder.SetInsertPoint(CheckBBTI);
  Builder.CreateCondBr(ExecUserCode, UI->getParent(), WorkerExitBB);

  CheckBBTI->eraseFromParent();
  UI->eraseFromParent();

  // Continue in the "user_code" block, see diagram above and in
  // openmp/libomptarget/deviceRTLs/common/include/target.h .
  return InsertPointTy(UserCodeEntryBB, UserCodeEntryBB->getFirstInsertionPt());
}

void OpenMPIRBuilder::createTargetDeinit(const LocationDescription &Loc) {
  if (!updateToLocation(Loc))
    return;

  Function *Fn = getOrCreateRuntimeFunctionPtr(
      omp::RuntimeFunction::OMPRTL___kmpc_target_deinit);

  Builder.CreateCall(Fn, {});
}

static const omp::GV &getGridValue(Function *Kernel) {
  if (Kernel->getCallingConv() == CallingConv::AMDGPU_KERNEL) {
    StringRef Features =
        Kernel->getFnAttribute("target-features").getValueAsString();
    if (Features.count("+wavefrontsize64"))
      return omp::getAMDGPUGridValues<64>();
    return omp::getAMDGPUGridValues<32>();
  }
  if (Triple(Kernel->getParent()->getTargetTriple()).isNVPTX())

    return omp::NVPTXGridValues;
  llvm_unreachable("No grid value available for this architecture!");
}

void OpenMPIRBuilder::setOutlinedTargetRegionFunctionAttributes(
    Function *OutlinedFn, int32_t NumTeams, int32_t NumThreads) {
  if (Config.isTargetDevice()) {
    OutlinedFn->setLinkage(GlobalValue::WeakODRLinkage);
    // TODO: Determine if DSO local can be set to true.
    OutlinedFn->setDSOLocal(false);
    OutlinedFn->setVisibility(GlobalValue::ProtectedVisibility);
    if (Triple(M.getTargetTriple()).isAMDGCN())
      OutlinedFn->setCallingConv(CallingConv::AMDGPU_KERNEL);
  }

  if (NumTeams > 0)
    OutlinedFn->addFnAttr("omp_target_num_teams", std::to_string(NumTeams));

  if (NumThreads == -1 && Config.isGPU())
    NumThreads = getGridValue(OutlinedFn).GV_Default_WG_Size;

  if (NumThreads > 0) {
    if (OutlinedFn->getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      OutlinedFn->addFnAttr("amdgpu-flat-work-group-size",
                            "1," + llvm::utostr(NumThreads));
    } else {
      // Update the "maxntidx" metadata for NVIDIA, or add it.
      NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");
      MDNode *ExistingOp = nullptr;
      for (auto *Op : MD->operands()) {
        if (Op->getNumOperands() != 3)
          continue;
        auto *Kernel = dyn_cast<ConstantAsMetadata>(Op->getOperand(0));
        if (!Kernel || Kernel->getValue() != OutlinedFn)
          continue;
        auto *Prop = dyn_cast<MDString>(Op->getOperand(1));
        if (!Prop || Prop->getString() != "maxntidx")
          continue;
        ExistingOp = Op;
        break;
      }
      if (ExistingOp) {
        auto *OldVal = dyn_cast<ConstantAsMetadata>(ExistingOp->getOperand(2));
        int32_t OldLimit =
            cast<ConstantInt>(OldVal->getValue())->getZExtValue();
        ExistingOp->replaceOperandWith(
            2, ConstantAsMetadata::get(
                   ConstantInt::get(OldVal->getValue()->getType(),
                                    std::min(OldLimit, NumThreads))));
      } else {
        LLVMContext &Ctx = M.getContext();
        Metadata *MDVals[] = {ConstantAsMetadata::get(OutlinedFn),
                              MDString::get(Ctx, "maxntidx"),
                              ConstantAsMetadata::get(ConstantInt::get(
                                  Type::getInt32Ty(Ctx), NumThreads))};
        // Append metadata to nvvm.annotations
        MD->addOperand(MDNode::get(Ctx, MDVals));
      }
    }
    OutlinedFn->addFnAttr("omp_target_thread_limit",
                          std::to_string(NumThreads));
  }
}

Constant *OpenMPIRBuilder::createOutlinedFunctionID(Function *OutlinedFn,
                                                    StringRef EntryFnIDName) {
  if (Config.isTargetDevice()) {
    assert(OutlinedFn && "The outlined function must exist if embedded");
    return ConstantExpr::getBitCast(OutlinedFn, Builder.getInt8PtrTy());
  }

  return new GlobalVariable(
      M, Builder.getInt8Ty(), /*isConstant=*/true, GlobalValue::WeakAnyLinkage,
      Constant::getNullValue(Builder.getInt8Ty()), EntryFnIDName);
}

Constant *OpenMPIRBuilder::createTargetRegionEntryAddr(Function *OutlinedFn,
                                                       StringRef EntryFnName) {
  if (OutlinedFn)
    return OutlinedFn;

  assert(!M.getGlobalVariable(EntryFnName, true) &&
         "Named kernel already exists?");
  return new GlobalVariable(
      M, Builder.getInt8Ty(), /*isConstant=*/true, GlobalValue::InternalLinkage,
      Constant::getNullValue(Builder.getInt8Ty()), EntryFnName);
}

void OpenMPIRBuilder::emitTargetRegionFunction(
    TargetRegionEntryInfo &EntryInfo,
    FunctionGenCallback &GenerateFunctionCallback, int32_t NumTeams,
    int32_t NumThreads, bool IsOffloadEntry, Function *&OutlinedFn,
    Constant *&OutlinedFnID) {

  SmallString<64> EntryFnName;
  OffloadInfoManager.getTargetRegionEntryFnName(EntryFnName, EntryInfo);

  OutlinedFn = Config.isTargetDevice() || !Config.openMPOffloadMandatory()
                   ? GenerateFunctionCallback(EntryFnName)
                   : nullptr;

  // If this target outline function is not an offload entry, we don't need to
  // register it. This may be in the case of a false if clause, or if there are
  // no OpenMP targets.
  if (!IsOffloadEntry)
    return;

  std::string EntryFnIDName =
      Config.isTargetDevice()
          ? std::string(EntryFnName)
          : createPlatformSpecificName({EntryFnName, "region_id"});

  OutlinedFnID = registerTargetRegionFunction(
      EntryInfo, OutlinedFn, EntryFnName, EntryFnIDName, NumTeams, NumThreads);
}

Constant *OpenMPIRBuilder::registerTargetRegionFunction(
    TargetRegionEntryInfo &EntryInfo, Function *OutlinedFn,
    StringRef EntryFnName, StringRef EntryFnIDName, int32_t NumTeams,
    int32_t NumThreads) {
  if (OutlinedFn)
    setOutlinedTargetRegionFunctionAttributes(OutlinedFn, NumTeams, NumThreads);
  auto OutlinedFnID = createOutlinedFunctionID(OutlinedFn, EntryFnIDName);
  auto EntryAddr = createTargetRegionEntryAddr(OutlinedFn, EntryFnName);
  OffloadInfoManager.registerTargetRegionEntryInfo(
      EntryInfo, EntryAddr, OutlinedFnID,
      OffloadEntriesInfoManager::OMPTargetRegionEntryTargetRegion);
  return OutlinedFnID;
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createTargetData(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    InsertPointTy CodeGenIP, Value *DeviceID, Value *IfCond,
    TargetDataInfo &Info, GenMapInfoCallbackTy GenMapInfoCB,
    omp::RuntimeFunction *MapperFunc,
    function_ref<InsertPointTy(InsertPointTy CodeGenIP, BodyGenTy BodyGenType)>
        BodyGenCB,
    function_ref<void(unsigned int, Value *)> DeviceAddrCB,
    function_ref<Value *(unsigned int)> CustomMapperCB, Value *SrcLocInfo) {
  if (!updateToLocation(Loc))
    return InsertPointTy();

  Builder.restoreIP(CodeGenIP);
  bool IsStandAlone = !BodyGenCB;
  MapInfosTy *MapInfo;
  // Generate the code for the opening of the data environment. Capture all the
  // arguments of the runtime call by reference because they are used in the
  // closing of the region.
  auto BeginThenGen = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    MapInfo = &GenMapInfoCB(Builder.saveIP());
    emitOffloadingArrays(AllocaIP, Builder.saveIP(), *MapInfo, Info,
                         /*IsNonContiguous=*/true, DeviceAddrCB,
                         CustomMapperCB);

    TargetDataRTArgs RTArgs;
    emitOffloadingArraysArgument(Builder, RTArgs, Info,
                                 !MapInfo->Names.empty());

    // Emit the number of elements in the offloading arrays.
    Value *PointerNum = Builder.getInt32(Info.NumberOfPtrs);

    // Source location for the ident struct
    if (!SrcLocInfo) {
      uint32_t SrcLocStrSize;
      Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
      SrcLocInfo = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
    }

    Value *OffloadingArgs[] = {SrcLocInfo,           DeviceID,
                               PointerNum,           RTArgs.BasePointersArray,
                               RTArgs.PointersArray, RTArgs.SizesArray,
                               RTArgs.MapTypesArray, RTArgs.MapNamesArray,
                               RTArgs.MappersArray};

    if (IsStandAlone) {
      assert(MapperFunc && "MapperFunc missing for standalone target data");
      Builder.CreateCall(getOrCreateRuntimeFunctionPtr(*MapperFunc),
                         OffloadingArgs);
    } else {
      Function *BeginMapperFunc = getOrCreateRuntimeFunctionPtr(
          omp::OMPRTL___tgt_target_data_begin_mapper);

      Builder.CreateCall(BeginMapperFunc, OffloadingArgs);

      for (auto DeviceMap : Info.DevicePtrInfoMap) {
        if (isa<AllocaInst>(DeviceMap.second.second)) {
          auto *LI =
              Builder.CreateLoad(Builder.getPtrTy(), DeviceMap.second.first);
          Builder.CreateStore(LI, DeviceMap.second.second);
        }
      }

      // If device pointer privatization is required, emit the body of the
      // region here. It will have to be duplicated: with and without
      // privatization.
      Builder.restoreIP(BodyGenCB(Builder.saveIP(), BodyGenTy::Priv));
    }
  };

  // If we need device pointer privatization, we need to emit the body of the
  // region with no privatization in the 'else' branch of the conditional.
  // Otherwise, we don't have to do anything.
  auto BeginElseGen = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    Builder.restoreIP(BodyGenCB(Builder.saveIP(), BodyGenTy::DupNoPriv));
  };

  // Generate code for the closing of the data region.
  auto EndThenGen = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {
    TargetDataRTArgs RTArgs;
    emitOffloadingArraysArgument(Builder, RTArgs, Info, !MapInfo->Names.empty(),
                                 /*ForEndCall=*/true);

    // Emit the number of elements in the offloading arrays.
    Value *PointerNum = Builder.getInt32(Info.NumberOfPtrs);

    // Source location for the ident struct
    if (!SrcLocInfo) {
      uint32_t SrcLocStrSize;
      Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
      SrcLocInfo = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
    }

    Value *OffloadingArgs[] = {SrcLocInfo,           DeviceID,
                               PointerNum,           RTArgs.BasePointersArray,
                               RTArgs.PointersArray, RTArgs.SizesArray,
                               RTArgs.MapTypesArray, RTArgs.MapNamesArray,
                               RTArgs.MappersArray};
    Function *EndMapperFunc =
        getOrCreateRuntimeFunctionPtr(omp::OMPRTL___tgt_target_data_end_mapper);

    Builder.CreateCall(EndMapperFunc, OffloadingArgs);
  };

  // We don't have to do anything to close the region if the if clause evaluates
  // to false.
  auto EndElseGen = [&](InsertPointTy AllocaIP, InsertPointTy CodeGenIP) {};

  if (BodyGenCB) {
    if (IfCond) {
      emitIfClause(IfCond, BeginThenGen, BeginElseGen, AllocaIP);
    } else {
      BeginThenGen(AllocaIP, Builder.saveIP());
    }

    // If we don't require privatization of device pointers, we emit the body in
    // between the runtime calls. This avoids duplicating the body code.
    Builder.restoreIP(BodyGenCB(Builder.saveIP(), BodyGenTy::NoPriv));

    if (IfCond) {
      emitIfClause(IfCond, EndThenGen, EndElseGen, AllocaIP);
    } else {
      EndThenGen(AllocaIP, Builder.saveIP());
    }
  } else {
    if (IfCond) {
      emitIfClause(IfCond, BeginThenGen, EndElseGen, AllocaIP);
    } else {
      BeginThenGen(AllocaIP, Builder.saveIP());
    }
  }

  return Builder.saveIP();
}

FunctionCallee
OpenMPIRBuilder::createForStaticInitFunction(unsigned IVSize, bool IVSigned,
                                             bool IsGPUDistribute) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  RuntimeFunction Name;
  if (IsGPUDistribute)
    Name = IVSize == 32
               ? (IVSigned ? omp::OMPRTL___kmpc_distribute_static_init_4
                           : omp::OMPRTL___kmpc_distribute_static_init_4u)
               : (IVSigned ? omp::OMPRTL___kmpc_distribute_static_init_8
                           : omp::OMPRTL___kmpc_distribute_static_init_8u);
  else
    Name = IVSize == 32 ? (IVSigned ? omp::OMPRTL___kmpc_for_static_init_4
                                    : omp::OMPRTL___kmpc_for_static_init_4u)
                        : (IVSigned ? omp::OMPRTL___kmpc_for_static_init_8
                                    : omp::OMPRTL___kmpc_for_static_init_8u);

  return getOrCreateRuntimeFunction(M, Name);
}

FunctionCallee OpenMPIRBuilder::createDispatchInitFunction(unsigned IVSize,
                                                           bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  RuntimeFunction Name = IVSize == 32
                             ? (IVSigned ? omp::OMPRTL___kmpc_dispatch_init_4
                                         : omp::OMPRTL___kmpc_dispatch_init_4u)
                             : (IVSigned ? omp::OMPRTL___kmpc_dispatch_init_8
                                         : omp::OMPRTL___kmpc_dispatch_init_8u);

  return getOrCreateRuntimeFunction(M, Name);
}

FunctionCallee OpenMPIRBuilder::createDispatchNextFunction(unsigned IVSize,
                                                           bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  RuntimeFunction Name = IVSize == 32
                             ? (IVSigned ? omp::OMPRTL___kmpc_dispatch_next_4
                                         : omp::OMPRTL___kmpc_dispatch_next_4u)
                             : (IVSigned ? omp::OMPRTL___kmpc_dispatch_next_8
                                         : omp::OMPRTL___kmpc_dispatch_next_8u);

  return getOrCreateRuntimeFunction(M, Name);
}

FunctionCallee OpenMPIRBuilder::createDispatchFiniFunction(unsigned IVSize,
                                                           bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  RuntimeFunction Name = IVSize == 32
                             ? (IVSigned ? omp::OMPRTL___kmpc_dispatch_fini_4
                                         : omp::OMPRTL___kmpc_dispatch_fini_4u)
                             : (IVSigned ? omp::OMPRTL___kmpc_dispatch_fini_8
                                         : omp::OMPRTL___kmpc_dispatch_fini_8u);

  return getOrCreateRuntimeFunction(M, Name);
}

// Copy input from pointer or i64 to the expected argument type.
static Value *copyInput(IRBuilderBase &Builder, unsigned AddrSpace,
                        Value *Input, Argument &Arg) {
  auto Addr = Builder.CreateAlloca(Arg.getType()->isPointerTy()
                                       ? Arg.getType()
                                       : Type::getInt64Ty(Builder.getContext()),
                                   AddrSpace);
  auto AddrAscast =
      Builder.CreatePointerBitCastOrAddrSpaceCast(Addr, Input->getType());
  Builder.CreateStore(&Arg, AddrAscast);
  auto Copy = Builder.CreateLoad(Arg.getType(), AddrAscast);

  return Copy;
}

static Function *
createOutlinedFunction(OpenMPIRBuilder &OMPBuilder, IRBuilderBase &Builder,
                       StringRef FuncName, SmallVectorImpl<Value *> &Inputs,
                       OpenMPIRBuilder::TargetBodyGenCallbackTy &CBFunc) {
  SmallVector<Type *> ParameterTypes;
  if (OMPBuilder.Config.isTargetDevice()) {
    // All parameters to target devices are passed as pointers
    // or i64. This assumes 64-bit address spaces/pointers.
    for (auto &Arg : Inputs)
      ParameterTypes.push_back(Arg->getType()->isPointerTy()
                                   ? Arg->getType()
                                   : Type::getInt64Ty(Builder.getContext()));
  } else {
    for (auto &Arg : Inputs)
      ParameterTypes.push_back(Arg->getType());
  }

  auto FuncType = FunctionType::get(Builder.getVoidTy(), ParameterTypes,
                                    /*isVarArg*/ false);
  auto Func = Function::Create(FuncType, GlobalValue::InternalLinkage, FuncName,
                               Builder.GetInsertBlock()->getModule());

  // Save insert point.
  auto OldInsertPoint = Builder.saveIP();

  // Generate the region into the function.
  BasicBlock *EntryBB = BasicBlock::Create(Builder.getContext(), "entry", Func);
  Builder.SetInsertPoint(EntryBB);

  // Insert target init call in the device compilation pass.
  if (OMPBuilder.Config.isTargetDevice())
    Builder.restoreIP(OMPBuilder.createTargetInit(Builder, /*IsSPMD*/ false));

  BasicBlock *UserCodeEntryBB = Builder.GetInsertBlock();

  // Insert target deinit call in the device compilation pass.
  Builder.restoreIP(CBFunc(Builder.saveIP(), Builder.saveIP()));
  if (OMPBuilder.Config.isTargetDevice())
    OMPBuilder.createTargetDeinit(Builder);

  // Insert return instruction.
  Builder.CreateRetVoid();

  // Rewrite uses of input valus to parameters.
  Builder.SetInsertPoint(UserCodeEntryBB->getFirstNonPHIOrDbg());
  for (auto InArg : zip(Inputs, Func->args())) {
    Value *Input = std::get<0>(InArg);
    Argument &Arg = std::get<1>(InArg);

    Value *InputCopy =
        OMPBuilder.Config.isTargetDevice()
            ? copyInput(Builder,
                        OMPBuilder.M.getDataLayout().getAllocaAddrSpace(),
                        Input, Arg)
            : &Arg;

    // Collect all the instructions
    for (User *User : make_early_inc_range(Input->users()))
      if (auto Instr = dyn_cast<Instruction>(User))
        if (Instr->getFunction() == Func)
          Instr->replaceUsesOfWith(Input, InputCopy);
  }

  // Restore insert point.
  Builder.restoreIP(OldInsertPoint);

  return Func;
}

static void
emitTargetOutlinedFunction(OpenMPIRBuilder &OMPBuilder, IRBuilderBase &Builder,
                           TargetRegionEntryInfo &EntryInfo,
                           Function *&OutlinedFn, Constant *&OutlinedFnID,
                           int32_t NumTeams, int32_t NumThreads,
                           SmallVectorImpl<Value *> &Inputs,
                           OpenMPIRBuilder::TargetBodyGenCallbackTy &CBFunc) {

  OpenMPIRBuilder::FunctionGenCallback &&GenerateOutlinedFunction =
      [&OMPBuilder, &Builder, &Inputs, &CBFunc](StringRef EntryFnName) {
        return createOutlinedFunction(OMPBuilder, Builder, EntryFnName, Inputs,
                                      CBFunc);
      };

  OMPBuilder.emitTargetRegionFunction(EntryInfo, GenerateOutlinedFunction,
                                      NumTeams, NumThreads, true, OutlinedFn,
                                      OutlinedFnID);
}

static void emitTargetCall(OpenMPIRBuilder &OMPBuilder, IRBuilderBase &Builder,
                           OpenMPIRBuilder::InsertPointTy AllocaIP,
                           Function *OutlinedFn, Constant *OutlinedFnID,
                           int32_t NumTeams, int32_t NumThreads,
                           SmallVectorImpl<Value *> &Args,
                           OpenMPIRBuilder::GenMapInfoCallbackTy GenMapInfoCB) {

  OpenMPIRBuilder::TargetDataInfo Info(
      /*RequiresDevicePointerInfo=*/false,
      /*SeparateBeginEndCalls=*/true);

  OpenMPIRBuilder::MapInfosTy &MapInfo = GenMapInfoCB(Builder.saveIP());
  OMPBuilder.emitOffloadingArrays(AllocaIP, Builder.saveIP(), MapInfo, Info,
                                  /*IsNonContiguous=*/true);

  OpenMPIRBuilder::TargetDataRTArgs RTArgs;
  OMPBuilder.emitOffloadingArraysArgument(Builder, RTArgs, Info,
                                          !MapInfo.Names.empty());

  //  emitKernelLaunch
  auto &&EmitTargetCallFallbackCB =
      [&](OpenMPIRBuilder::InsertPointTy IP) -> OpenMPIRBuilder::InsertPointTy {
    Builder.restoreIP(IP);
    Builder.CreateCall(OutlinedFn, Args);
    return Builder.saveIP();
  };

  unsigned NumTargetItems = MapInfo.BasePointers.size();
  // TODO: Use correct device ID
  Value *DeviceID = Builder.getInt64(OMP_DEVICEID_UNDEF);
  Value *NumTeamsVal = Builder.getInt32(NumTeams);
  Value *NumThreadsVal = Builder.getInt32(NumThreads);
  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = OMPBuilder.getOrCreateDefaultSrcLocStr(SrcLocStrSize);
  Value *RTLoc = OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize,
                                             llvm::omp::IdentFlag(0), 0);
  // TODO: Use correct NumIterations
  Value *NumIterations = Builder.getInt64(0);
  // TODO: Use correct DynCGGroupMem
  Value *DynCGGroupMem = Builder.getInt32(0);

  bool HasNoWait = false;

  OpenMPIRBuilder::TargetKernelArgs KArgs(NumTargetItems, RTArgs, NumIterations,
                                          NumTeamsVal, NumThreadsVal,
                                          DynCGGroupMem, HasNoWait);

  Builder.restoreIP(OMPBuilder.emitKernelLaunch(
      Builder, OutlinedFn, OutlinedFnID, EmitTargetCallFallbackCB, KArgs,
      DeviceID, RTLoc, AllocaIP));
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createTarget(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    InsertPointTy CodeGenIP, TargetRegionEntryInfo &EntryInfo, int32_t NumTeams,
    int32_t NumThreads, SmallVectorImpl<Value *> &Args,
    GenMapInfoCallbackTy GenMapInfoCB, TargetBodyGenCallbackTy CBFunc) {
  if (!updateToLocation(Loc))
    return InsertPointTy();

  Builder.restoreIP(CodeGenIP);

  Function *OutlinedFn;
  Constant *OutlinedFnID;
  emitTargetOutlinedFunction(*this, Builder, EntryInfo, OutlinedFn,
                             OutlinedFnID, NumTeams, NumThreads, Args, CBFunc);
  if (!Config.isTargetDevice())
    emitTargetCall(*this, Builder, AllocaIP, OutlinedFn, OutlinedFnID, NumTeams,
                   NumThreads, Args, GenMapInfoCB);

  return Builder.saveIP();
}

std::string OpenMPIRBuilder::getNameWithSeparators(ArrayRef<StringRef> Parts,
                                                   StringRef FirstSeparator,
                                                   StringRef Separator) {
  SmallString<128> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  StringRef Sep = FirstSeparator;
  for (StringRef Part : Parts) {
    OS << Sep << Part;
    Sep = Separator;
  }
  return OS.str().str();
}

std::string
OpenMPIRBuilder::createPlatformSpecificName(ArrayRef<StringRef> Parts) const {
  return OpenMPIRBuilder::getNameWithSeparators(Parts, Config.firstSeparator(),
                                                Config.separator());
}

GlobalVariable *
OpenMPIRBuilder::getOrCreateInternalVariable(Type *Ty, const StringRef &Name,
                                             unsigned AddressSpace) {
  auto &Elem = *InternalVars.try_emplace(Name, nullptr).first;
  if (Elem.second) {
    assert(Elem.second->getValueType() == Ty &&
           "OMP internal variable has different type than requested");
  } else {
    // TODO: investigate the appropriate linkage type used for the global
    // variable for possibly changing that to internal or private, or maybe
    // create different versions of the function for different OMP internal
    // variables.
    auto *GV = new GlobalVariable(
        M, Ty, /*IsConstant=*/false, GlobalValue::CommonLinkage,
        Constant::getNullValue(Ty), Elem.first(),
        /*InsertBefore=*/nullptr, GlobalValue::NotThreadLocal, AddressSpace);
    const DataLayout &DL = M.getDataLayout();
    const llvm::Align TypeAlign = DL.getABITypeAlign(Ty);
    const llvm::Align PtrAlign = DL.getPointerABIAlignment(AddressSpace);
    GV->setAlignment(std::max(TypeAlign, PtrAlign));
    Elem.second = GV;
  }

  return Elem.second;
}

Value *OpenMPIRBuilder::getOMPCriticalRegionLock(StringRef CriticalName) {
  std::string Prefix = Twine("gomp_critical_user_", CriticalName).str();
  std::string Name = getNameWithSeparators({Prefix, "var"}, ".", ".");
  return getOrCreateInternalVariable(KmpCriticalNameTy, Name);
}

Value *OpenMPIRBuilder::getSizeInBytes(Value *BasePtr) {
  LLVMContext &Ctx = Builder.getContext();
  Value *Null =
      Constant::getNullValue(PointerType::getUnqual(BasePtr->getContext()));
  Value *SizeGep =
      Builder.CreateGEP(BasePtr->getType(), Null, Builder.getInt32(1));
  Value *SizePtrToInt = Builder.CreatePtrToInt(SizeGep, Type::getInt64Ty(Ctx));
  return SizePtrToInt;
}

GlobalVariable *
OpenMPIRBuilder::createOffloadMaptypes(SmallVectorImpl<uint64_t> &Mappings,
                                       std::string VarName) {
  llvm::Constant *MaptypesArrayInit =
      llvm::ConstantDataArray::get(M.getContext(), Mappings);
  auto *MaptypesArrayGlobal = new llvm::GlobalVariable(
      M, MaptypesArrayInit->getType(),
      /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, MaptypesArrayInit,
      VarName);
  MaptypesArrayGlobal->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  return MaptypesArrayGlobal;
}

void OpenMPIRBuilder::createMapperAllocas(const LocationDescription &Loc,
                                          InsertPointTy AllocaIP,
                                          unsigned NumOperands,
                                          struct MapperAllocas &MapperAllocas) {
  if (!updateToLocation(Loc))
    return;

  auto *ArrI8PtrTy = ArrayType::get(Int8Ptr, NumOperands);
  auto *ArrI64Ty = ArrayType::get(Int64, NumOperands);
  Builder.restoreIP(AllocaIP);
  AllocaInst *ArgsBase = Builder.CreateAlloca(
      ArrI8PtrTy, /* ArraySize = */ nullptr, ".offload_baseptrs");
  AllocaInst *Args = Builder.CreateAlloca(ArrI8PtrTy, /* ArraySize = */ nullptr,
                                          ".offload_ptrs");
  AllocaInst *ArgSizes = Builder.CreateAlloca(
      ArrI64Ty, /* ArraySize = */ nullptr, ".offload_sizes");
  Builder.restoreIP(Loc.IP);
  MapperAllocas.ArgsBase = ArgsBase;
  MapperAllocas.Args = Args;
  MapperAllocas.ArgSizes = ArgSizes;
}

void OpenMPIRBuilder::emitMapperCall(const LocationDescription &Loc,
                                     Function *MapperFunc, Value *SrcLocInfo,
                                     Value *MaptypesArg, Value *MapnamesArg,
                                     struct MapperAllocas &MapperAllocas,
                                     int64_t DeviceID, unsigned NumOperands) {
  if (!updateToLocation(Loc))
    return;

  auto *ArrI8PtrTy = ArrayType::get(Int8Ptr, NumOperands);
  auto *ArrI64Ty = ArrayType::get(Int64, NumOperands);
  Value *ArgsBaseGEP =
      Builder.CreateInBoundsGEP(ArrI8PtrTy, MapperAllocas.ArgsBase,
                                {Builder.getInt32(0), Builder.getInt32(0)});
  Value *ArgsGEP =
      Builder.CreateInBoundsGEP(ArrI8PtrTy, MapperAllocas.Args,
                                {Builder.getInt32(0), Builder.getInt32(0)});
  Value *ArgSizesGEP =
      Builder.CreateInBoundsGEP(ArrI64Ty, MapperAllocas.ArgSizes,
                                {Builder.getInt32(0), Builder.getInt32(0)});
  Value *NullPtr =
      Constant::getNullValue(PointerType::getUnqual(Int8Ptr->getContext()));
  Builder.CreateCall(MapperFunc,
                     {SrcLocInfo, Builder.getInt64(DeviceID),
                      Builder.getInt32(NumOperands), ArgsBaseGEP, ArgsGEP,
                      ArgSizesGEP, MaptypesArg, MapnamesArg, NullPtr});
}

void OpenMPIRBuilder::emitOffloadingArraysArgument(IRBuilderBase &Builder,
                                                   TargetDataRTArgs &RTArgs,
                                                   TargetDataInfo &Info,
                                                   bool EmitDebug,
                                                   bool ForEndCall) {
  assert((!ForEndCall || Info.separateBeginEndCalls()) &&
         "expected region end call to runtime only when end call is separate");
  auto UnqualPtrTy = PointerType::getUnqual(M.getContext());
  auto VoidPtrTy = UnqualPtrTy;
  auto VoidPtrPtrTy = UnqualPtrTy;
  auto Int64Ty = Type::getInt64Ty(M.getContext());
  auto Int64PtrTy = UnqualPtrTy;

  if (!Info.NumberOfPtrs) {
    RTArgs.BasePointersArray = ConstantPointerNull::get(VoidPtrPtrTy);
    RTArgs.PointersArray = ConstantPointerNull::get(VoidPtrPtrTy);
    RTArgs.SizesArray = ConstantPointerNull::get(Int64PtrTy);
    RTArgs.MapTypesArray = ConstantPointerNull::get(Int64PtrTy);
    RTArgs.MapNamesArray = ConstantPointerNull::get(VoidPtrPtrTy);
    RTArgs.MappersArray = ConstantPointerNull::get(VoidPtrPtrTy);
    return;
  }

  RTArgs.BasePointersArray = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(VoidPtrTy, Info.NumberOfPtrs),
      Info.RTArgs.BasePointersArray,
      /*Idx0=*/0, /*Idx1=*/0);
  RTArgs.PointersArray = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(VoidPtrTy, Info.NumberOfPtrs), Info.RTArgs.PointersArray,
      /*Idx0=*/0,
      /*Idx1=*/0);
  RTArgs.SizesArray = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(Int64Ty, Info.NumberOfPtrs), Info.RTArgs.SizesArray,
      /*Idx0=*/0, /*Idx1=*/0);
  RTArgs.MapTypesArray = Builder.CreateConstInBoundsGEP2_32(
      ArrayType::get(Int64Ty, Info.NumberOfPtrs),
      ForEndCall && Info.RTArgs.MapTypesArrayEnd ? Info.RTArgs.MapTypesArrayEnd
                                                 : Info.RTArgs.MapTypesArray,
      /*Idx0=*/0,
      /*Idx1=*/0);

  // Only emit the mapper information arrays if debug information is
  // requested.
  if (!EmitDebug)
    RTArgs.MapNamesArray = ConstantPointerNull::get(VoidPtrPtrTy);
  else
    RTArgs.MapNamesArray = Builder.CreateConstInBoundsGEP2_32(
        ArrayType::get(VoidPtrTy, Info.NumberOfPtrs), Info.RTArgs.MapNamesArray,
        /*Idx0=*/0,
        /*Idx1=*/0);
  // If there is no user-defined mapper, set the mapper array to nullptr to
  // avoid an unnecessary data privatization
  if (!Info.HasMapper)
    RTArgs.MappersArray = ConstantPointerNull::get(VoidPtrPtrTy);
  else
    RTArgs.MappersArray =
        Builder.CreatePointerCast(Info.RTArgs.MappersArray, VoidPtrPtrTy);
}

void OpenMPIRBuilder::emitNonContiguousDescriptor(InsertPointTy AllocaIP,
                                                  InsertPointTy CodeGenIP,
                                                  MapInfosTy &CombinedInfo,
                                                  TargetDataInfo &Info) {
  MapInfosTy::StructNonContiguousInfo &NonContigInfo =
      CombinedInfo.NonContigInfo;

  // Build an array of struct descriptor_dim and then assign it to
  // offload_args.
  //
  // struct descriptor_dim {
  //  uint64_t offset;
  //  uint64_t count;
  //  uint64_t stride
  // };
  Type *Int64Ty = Builder.getInt64Ty();
  StructType *DimTy = StructType::create(
      M.getContext(), ArrayRef<Type *>({Int64Ty, Int64Ty, Int64Ty}),
      "struct.descriptor_dim");

  enum { OffsetFD = 0, CountFD, StrideFD };
  // We need two index variable here since the size of "Dims" is the same as
  // the size of Components, however, the size of offset, count, and stride is
  // equal to the size of base declaration that is non-contiguous.
  for (unsigned I = 0, L = 0, E = NonContigInfo.Dims.size(); I < E; ++I) {
    // Skip emitting ir if dimension size is 1 since it cannot be
    // non-contiguous.
    if (NonContigInfo.Dims[I] == 1)
      continue;
    Builder.restoreIP(AllocaIP);
    ArrayType *ArrayTy = ArrayType::get(DimTy, NonContigInfo.Dims[I]);
    AllocaInst *DimsAddr =
        Builder.CreateAlloca(ArrayTy, /* ArraySize = */ nullptr, "dims");
    Builder.restoreIP(CodeGenIP);
    for (unsigned II = 0, EE = NonContigInfo.Dims[I]; II < EE; ++II) {
      unsigned RevIdx = EE - II - 1;
      Value *DimsLVal = Builder.CreateInBoundsGEP(
          DimsAddr->getAllocatedType(), DimsAddr,
          {Builder.getInt64(0), Builder.getInt64(II)});
      // Offset
      Value *OffsetLVal = Builder.CreateStructGEP(DimTy, DimsLVal, OffsetFD);
      Builder.CreateAlignedStore(
          NonContigInfo.Offsets[L][RevIdx], OffsetLVal,
          M.getDataLayout().getPrefTypeAlign(OffsetLVal->getType()));
      // Count
      Value *CountLVal = Builder.CreateStructGEP(DimTy, DimsLVal, CountFD);
      Builder.CreateAlignedStore(
          NonContigInfo.Counts[L][RevIdx], CountLVal,
          M.getDataLayout().getPrefTypeAlign(CountLVal->getType()));
      // Stride
      Value *StrideLVal = Builder.CreateStructGEP(DimTy, DimsLVal, StrideFD);
      Builder.CreateAlignedStore(
          NonContigInfo.Strides[L][RevIdx], StrideLVal,
          M.getDataLayout().getPrefTypeAlign(CountLVal->getType()));
    }
    // args[I] = &dims
    Builder.restoreIP(CodeGenIP);
    Value *DAddr = Builder.CreatePointerBitCastOrAddrSpaceCast(
        DimsAddr, Builder.getInt8PtrTy());
    Value *P = Builder.CreateConstInBoundsGEP2_32(
        ArrayType::get(Builder.getInt8PtrTy(), Info.NumberOfPtrs),
        Info.RTArgs.PointersArray, 0, I);
    Builder.CreateAlignedStore(
        DAddr, P, M.getDataLayout().getPrefTypeAlign(Builder.getInt8PtrTy()));
    ++L;
  }
}

void OpenMPIRBuilder::emitOffloadingArrays(
    InsertPointTy AllocaIP, InsertPointTy CodeGenIP, MapInfosTy &CombinedInfo,
    TargetDataInfo &Info, bool IsNonContiguous,
    function_ref<void(unsigned int, Value *)> DeviceAddrCB,
    function_ref<Value *(unsigned int)> CustomMapperCB) {

  // Reset the array information.
  Info.clearArrayInfo();
  Info.NumberOfPtrs = CombinedInfo.BasePointers.size();

  if (Info.NumberOfPtrs == 0)
    return;

  Builder.restoreIP(AllocaIP);
  // Detect if we have any capture size requiring runtime evaluation of the
  // size so that a constant array could be eventually used.
  ArrayType *PointerArrayType =
      ArrayType::get(Builder.getInt8PtrTy(), Info.NumberOfPtrs);

  Info.RTArgs.BasePointersArray = Builder.CreateAlloca(
      PointerArrayType, /* ArraySize = */ nullptr, ".offload_baseptrs");

  Info.RTArgs.PointersArray = Builder.CreateAlloca(
      PointerArrayType, /* ArraySize = */ nullptr, ".offload_ptrs");
  AllocaInst *MappersArray = Builder.CreateAlloca(
      PointerArrayType, /* ArraySize = */ nullptr, ".offload_mappers");
  Info.RTArgs.MappersArray = MappersArray;

  // If we don't have any VLA types or other types that require runtime
  // evaluation, we can use a constant array for the map sizes, otherwise we
  // need to fill up the arrays as we do for the pointers.
  Type *Int64Ty = Builder.getInt64Ty();
  SmallVector<Constant *> ConstSizes(CombinedInfo.Sizes.size(),
                                     ConstantInt::get(Int64Ty, 0));
  SmallBitVector RuntimeSizes(CombinedInfo.Sizes.size());
  for (unsigned I = 0, E = CombinedInfo.Sizes.size(); I < E; ++I) {
    if (auto *CI = dyn_cast<Constant>(CombinedInfo.Sizes[I])) {
      if (!isa<ConstantExpr>(CI) && !isa<GlobalValue>(CI)) {
        if (IsNonContiguous &&
            static_cast<std::underlying_type_t<OpenMPOffloadMappingFlags>>(
                CombinedInfo.Types[I] &
                OpenMPOffloadMappingFlags::OMP_MAP_NON_CONTIG))
          ConstSizes[I] =
              ConstantInt::get(Int64Ty, CombinedInfo.NonContigInfo.Dims[I]);
        else
          ConstSizes[I] = CI;
        continue;
      }
    }
    RuntimeSizes.set(I);
  }

  if (RuntimeSizes.all()) {
    ArrayType *SizeArrayType = ArrayType::get(Int64Ty, Info.NumberOfPtrs);
    Info.RTArgs.SizesArray = Builder.CreateAlloca(
        SizeArrayType, /* ArraySize = */ nullptr, ".offload_sizes");
    Builder.restoreIP(CodeGenIP);
  } else {
    auto *SizesArrayInit = ConstantArray::get(
        ArrayType::get(Int64Ty, ConstSizes.size()), ConstSizes);
    std::string Name = createPlatformSpecificName({"offload_sizes"});
    auto *SizesArrayGbl =
        new GlobalVariable(M, SizesArrayInit->getType(), /*isConstant=*/true,
                           GlobalValue::PrivateLinkage, SizesArrayInit, Name);
    SizesArrayGbl->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    if (!RuntimeSizes.any()) {
      Info.RTArgs.SizesArray = SizesArrayGbl;
    } else {
      unsigned IndexSize = M.getDataLayout().getIndexSizeInBits(0);
      Align OffloadSizeAlign = M.getDataLayout().getABIIntegerTypeAlignment(64);
      ArrayType *SizeArrayType = ArrayType::get(Int64Ty, Info.NumberOfPtrs);
      AllocaInst *Buffer = Builder.CreateAlloca(
          SizeArrayType, /* ArraySize = */ nullptr, ".offload_sizes");
      Buffer->setAlignment(OffloadSizeAlign);
      Builder.restoreIP(CodeGenIP);
      Builder.CreateMemCpy(
          Buffer, M.getDataLayout().getPrefTypeAlign(Buffer->getType()),
          SizesArrayGbl, OffloadSizeAlign,
          Builder.getIntN(
              IndexSize,
              Buffer->getAllocationSize(M.getDataLayout())->getFixedValue()));

      Info.RTArgs.SizesArray = Buffer;
    }
    Builder.restoreIP(CodeGenIP);
  }

  // The map types are always constant so we don't need to generate code to
  // fill arrays. Instead, we create an array constant.
  SmallVector<uint64_t, 4> Mapping;
  for (auto mapFlag : CombinedInfo.Types)
    Mapping.push_back(
        static_cast<std::underlying_type_t<OpenMPOffloadMappingFlags>>(
            mapFlag));
  std::string MaptypesName = createPlatformSpecificName({"offload_maptypes"});
  auto *MapTypesArrayGbl = createOffloadMaptypes(Mapping, MaptypesName);
  Info.RTArgs.MapTypesArray = MapTypesArrayGbl;

  // The information types are only built if provided.
  if (!CombinedInfo.Names.empty()) {
    std::string MapnamesName = createPlatformSpecificName({"offload_mapnames"});
    auto *MapNamesArrayGbl =
        createOffloadMapnames(CombinedInfo.Names, MapnamesName);
    Info.RTArgs.MapNamesArray = MapNamesArrayGbl;
  } else {
    Info.RTArgs.MapNamesArray =
        Constant::getNullValue(PointerType::getUnqual(Builder.getContext()));
  }

  // If there's a present map type modifier, it must not be applied to the end
  // of a region, so generate a separate map type array in that case.
  if (Info.separateBeginEndCalls()) {
    bool EndMapTypesDiffer = false;
    for (uint64_t &Type : Mapping) {
      if (Type & static_cast<std::underlying_type_t<OpenMPOffloadMappingFlags>>(
                     OpenMPOffloadMappingFlags::OMP_MAP_PRESENT)) {
        Type &= ~static_cast<std::underlying_type_t<OpenMPOffloadMappingFlags>>(
            OpenMPOffloadMappingFlags::OMP_MAP_PRESENT);
        EndMapTypesDiffer = true;
      }
    }
    if (EndMapTypesDiffer) {
      MapTypesArrayGbl = createOffloadMaptypes(Mapping, MaptypesName);
      Info.RTArgs.MapTypesArrayEnd = MapTypesArrayGbl;
    }
  }

  PointerType *PtrTy = Builder.getPtrTy();
  for (unsigned I = 0; I < Info.NumberOfPtrs; ++I) {
    Value *BPVal = CombinedInfo.BasePointers[I];
    Value *BP = Builder.CreateConstInBoundsGEP2_32(
        ArrayType::get(PtrTy, Info.NumberOfPtrs), Info.RTArgs.BasePointersArray,
        0, I);
    Builder.CreateAlignedStore(BPVal, BP,
                               M.getDataLayout().getPrefTypeAlign(PtrTy));

    if (Info.requiresDevicePointerInfo()) {
      if (CombinedInfo.DevicePointers[I] == DeviceInfoTy::Pointer) {
        CodeGenIP = Builder.saveIP();
        Builder.restoreIP(AllocaIP);
        Info.DevicePtrInfoMap[BPVal] = {BP, Builder.CreateAlloca(PtrTy)};
        Builder.restoreIP(CodeGenIP);
        if (DeviceAddrCB)
          DeviceAddrCB(I, Info.DevicePtrInfoMap[BPVal].second);
      } else if (CombinedInfo.DevicePointers[I] == DeviceInfoTy::Address) {
        Info.DevicePtrInfoMap[BPVal] = {BP, BP};
        if (DeviceAddrCB)
          DeviceAddrCB(I, BP);
      }
    }

    Value *PVal = CombinedInfo.Pointers[I];
    Value *P = Builder.CreateConstInBoundsGEP2_32(
        ArrayType::get(PtrTy, Info.NumberOfPtrs), Info.RTArgs.PointersArray, 0,
        I);
    // TODO: Check alignment correct.
    Builder.CreateAlignedStore(PVal, P,
                               M.getDataLayout().getPrefTypeAlign(PtrTy));

    if (RuntimeSizes.test(I)) {
      Value *S = Builder.CreateConstInBoundsGEP2_32(
          ArrayType::get(Int64Ty, Info.NumberOfPtrs), Info.RTArgs.SizesArray,
          /*Idx0=*/0,
          /*Idx1=*/I);
      Builder.CreateAlignedStore(Builder.CreateIntCast(CombinedInfo.Sizes[I],
                                                       Int64Ty,
                                                       /*isSigned=*/true),
                                 S, M.getDataLayout().getPrefTypeAlign(PtrTy));
    }
    // Fill up the mapper array.
    unsigned IndexSize = M.getDataLayout().getIndexSizeInBits(0);
    Value *MFunc = ConstantPointerNull::get(PtrTy);
    if (CustomMapperCB)
      if (Value *CustomMFunc = CustomMapperCB(I))
        MFunc = Builder.CreatePointerCast(CustomMFunc, PtrTy);
    Value *MAddr = Builder.CreateInBoundsGEP(
        MappersArray->getAllocatedType(), MappersArray,
        {Builder.getIntN(IndexSize, 0), Builder.getIntN(IndexSize, I)});
    Builder.CreateAlignedStore(
        MFunc, MAddr, M.getDataLayout().getPrefTypeAlign(MAddr->getType()));
  }

  if (!IsNonContiguous || CombinedInfo.NonContigInfo.Offsets.empty() ||
      Info.NumberOfPtrs == 0)
    return;
  emitNonContiguousDescriptor(AllocaIP, CodeGenIP, CombinedInfo, Info);
}

void OpenMPIRBuilder::emitBranch(BasicBlock *Target) {
  BasicBlock *CurBB = Builder.GetInsertBlock();

  if (!CurBB || CurBB->getTerminator()) {
    // If there is no insert point or the previous block is already
    // terminated, don't touch it.
  } else {
    // Otherwise, create a fall-through branch.
    Builder.CreateBr(Target);
  }

  Builder.ClearInsertionPoint();
}

void OpenMPIRBuilder::emitBlock(BasicBlock *BB, Function *CurFn,
                                bool IsFinished) {
  BasicBlock *CurBB = Builder.GetInsertBlock();

  // Fall out of the current block (if necessary).
  emitBranch(BB);

  if (IsFinished && BB->use_empty()) {
    BB->eraseFromParent();
    return;
  }

  // Place the block after the current block, if possible, or else at
  // the end of the function.
  if (CurBB && CurBB->getParent())
    CurFn->insert(std::next(CurBB->getIterator()), BB);
  else
    CurFn->insert(CurFn->end(), BB);
  Builder.SetInsertPoint(BB);
}

void OpenMPIRBuilder::emitIfClause(Value *Cond, BodyGenCallbackTy ThenGen,
                                   BodyGenCallbackTy ElseGen,
                                   InsertPointTy AllocaIP) {
  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  if (auto *CI = dyn_cast<ConstantInt>(Cond)) {
    auto CondConstant = CI->getSExtValue();
    if (CondConstant)
      ThenGen(AllocaIP, Builder.saveIP());
    else
      ElseGen(AllocaIP, Builder.saveIP());
    return;
  }

  Function *CurFn = Builder.GetInsertBlock()->getParent();

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just
  // emit the conditional branch.
  BasicBlock *ThenBlock = BasicBlock::Create(M.getContext(), "omp_if.then");
  BasicBlock *ElseBlock = BasicBlock::Create(M.getContext(), "omp_if.else");
  BasicBlock *ContBlock = BasicBlock::Create(M.getContext(), "omp_if.end");
  Builder.CreateCondBr(Cond, ThenBlock, ElseBlock);
  // Emit the 'then' code.
  emitBlock(ThenBlock, CurFn);
  ThenGen(AllocaIP, Builder.saveIP());
  emitBranch(ContBlock);
  // Emit the 'else' code if present.
  // There is no need to emit line number for unconditional branch.
  emitBlock(ElseBlock, CurFn);
  ElseGen(AllocaIP, Builder.saveIP());
  // There is no need to emit line number for unconditional branch.
  emitBranch(ContBlock);
  // Emit the continuation block for code after the if.
  emitBlock(ContBlock, CurFn, /*IsFinished=*/true);
}

bool OpenMPIRBuilder::checkAndEmitFlushAfterAtomic(
    const LocationDescription &Loc, llvm::AtomicOrdering AO, AtomicKind AK) {
  assert(!(AO == AtomicOrdering::NotAtomic ||
           AO == llvm::AtomicOrdering::Unordered) &&
         "Unexpected Atomic Ordering.");

  bool Flush = false;
  llvm::AtomicOrdering FlushAO = AtomicOrdering::Monotonic;

  switch (AK) {
  case Read:
    if (AO == AtomicOrdering::Acquire || AO == AtomicOrdering::AcquireRelease ||
        AO == AtomicOrdering::SequentiallyConsistent) {
      FlushAO = AtomicOrdering::Acquire;
      Flush = true;
    }
    break;
  case Write:
  case Compare:
  case Update:
    if (AO == AtomicOrdering::Release || AO == AtomicOrdering::AcquireRelease ||
        AO == AtomicOrdering::SequentiallyConsistent) {
      FlushAO = AtomicOrdering::Release;
      Flush = true;
    }
    break;
  case Capture:
    switch (AO) {
    case AtomicOrdering::Acquire:
      FlushAO = AtomicOrdering::Acquire;
      Flush = true;
      break;
    case AtomicOrdering::Release:
      FlushAO = AtomicOrdering::Release;
      Flush = true;
      break;
    case AtomicOrdering::AcquireRelease:
    case AtomicOrdering::SequentiallyConsistent:
      FlushAO = AtomicOrdering::AcquireRelease;
      Flush = true;
      break;
    default:
      // do nothing - leave silently.
      break;
    }
  }

  if (Flush) {
    // Currently Flush RT call still doesn't take memory_ordering, so for when
    // that happens, this tries to do the resolution of which atomic ordering
    // to use with but issue the flush call
    // TODO: pass `FlushAO` after memory ordering support is added
    (void)FlushAO;
    emitFlush(Loc);
  }

  // for AO == AtomicOrdering::Monotonic and  all other case combinations
  // do nothing
  return Flush;
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createAtomicRead(const LocationDescription &Loc,
                                  AtomicOpValue &X, AtomicOpValue &V,
                                  AtomicOrdering AO) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  assert(X.Var->getType()->isPointerTy() &&
         "OMP Atomic expects a pointer to target memory");
  Type *XElemTy = X.ElemTy;
  assert((XElemTy->isFloatingPointTy() || XElemTy->isIntegerTy() ||
          XElemTy->isPointerTy()) &&
         "OMP atomic read expected a scalar type");

  Value *XRead = nullptr;

  if (XElemTy->isIntegerTy()) {
    LoadInst *XLD =
        Builder.CreateLoad(XElemTy, X.Var, X.IsVolatile, "omp.atomic.read");
    XLD->setAtomic(AO);
    XRead = cast<Value>(XLD);
  } else {
    // We need to perform atomic op as integer
    IntegerType *IntCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    LoadInst *XLoad =
        Builder.CreateLoad(IntCastTy, X.Var, X.IsVolatile, "omp.atomic.load");
    XLoad->setAtomic(AO);
    if (XElemTy->isFloatingPointTy()) {
      XRead = Builder.CreateBitCast(XLoad, XElemTy, "atomic.flt.cast");
    } else {
      XRead = Builder.CreateIntToPtr(XLoad, XElemTy, "atomic.ptr.cast");
    }
  }
  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Read);
  Builder.CreateStore(XRead, V.Var, V.IsVolatile);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createAtomicWrite(const LocationDescription &Loc,
                                   AtomicOpValue &X, Value *Expr,
                                   AtomicOrdering AO) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Type *XTy = X.Var->getType();
  assert(XTy->isPointerTy() && "OMP Atomic expects a pointer to target memory");
  Type *XElemTy = X.ElemTy;
  assert((XElemTy->isFloatingPointTy() || XElemTy->isIntegerTy() ||
          XElemTy->isPointerTy()) &&
         "OMP atomic write expected a scalar type");

  if (XElemTy->isIntegerTy()) {
    StoreInst *XSt = Builder.CreateStore(Expr, X.Var, X.IsVolatile);
    XSt->setAtomic(AO);
  } else {
    // We need to bitcast and perform atomic op as integers
    unsigned Addrspace = cast<PointerType>(XTy)->getAddressSpace();
    IntegerType *IntCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    Value *XBCast = Builder.CreateBitCast(
        X.Var, IntCastTy->getPointerTo(Addrspace), "atomic.dst.int.cast");
    Value *ExprCast =
        Builder.CreateBitCast(Expr, IntCastTy, "atomic.src.int.cast");
    StoreInst *XSt = Builder.CreateStore(ExprCast, XBCast, X.IsVolatile);
    XSt->setAtomic(AO);
  }

  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Write);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createAtomicUpdate(
    const LocationDescription &Loc, InsertPointTy AllocaIP, AtomicOpValue &X,
    Value *Expr, AtomicOrdering AO, AtomicRMWInst::BinOp RMWOp,
    AtomicUpdateCallbackTy &UpdateOp, bool IsXBinopExpr) {
  assert(!isConflictIP(Loc.IP, AllocaIP) && "IPs must not be ambiguous");
  if (!updateToLocation(Loc))
    return Loc.IP;

  LLVM_DEBUG({
    Type *XTy = X.Var->getType();
    assert(XTy->isPointerTy() &&
           "OMP Atomic expects a pointer to target memory");
    Type *XElemTy = X.ElemTy;
    assert((XElemTy->isFloatingPointTy() || XElemTy->isIntegerTy() ||
            XElemTy->isPointerTy()) &&
           "OMP atomic update expected a scalar type");
    assert((RMWOp != AtomicRMWInst::Max) && (RMWOp != AtomicRMWInst::Min) &&
           (RMWOp != AtomicRMWInst::UMax) && (RMWOp != AtomicRMWInst::UMin) &&
           "OpenMP atomic does not support LT or GT operations");
  });

  emitAtomicUpdate(AllocaIP, X.Var, X.ElemTy, Expr, AO, RMWOp, UpdateOp,
                   X.IsVolatile, IsXBinopExpr);
  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Update);
  return Builder.saveIP();
}

// FIXME: Duplicating AtomicExpand
Value *OpenMPIRBuilder::emitRMWOpAsInstruction(Value *Src1, Value *Src2,
                                               AtomicRMWInst::BinOp RMWOp) {
  switch (RMWOp) {
  case AtomicRMWInst::Add:
    return Builder.CreateAdd(Src1, Src2);
  case AtomicRMWInst::Sub:
    return Builder.CreateSub(Src1, Src2);
  case AtomicRMWInst::And:
    return Builder.CreateAnd(Src1, Src2);
  case AtomicRMWInst::Nand:
    return Builder.CreateNeg(Builder.CreateAnd(Src1, Src2));
  case AtomicRMWInst::Or:
    return Builder.CreateOr(Src1, Src2);
  case AtomicRMWInst::Xor:
    return Builder.CreateXor(Src1, Src2);
  case AtomicRMWInst::Xchg:
  case AtomicRMWInst::FAdd:
  case AtomicRMWInst::FSub:
  case AtomicRMWInst::BAD_BINOP:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
  case AtomicRMWInst::FMax:
  case AtomicRMWInst::FMin:
  case AtomicRMWInst::UIncWrap:
  case AtomicRMWInst::UDecWrap:
    llvm_unreachable("Unsupported atomic update operation");
  }
  llvm_unreachable("Unsupported atomic update operation");
}

std::pair<Value *, Value *> OpenMPIRBuilder::emitAtomicUpdate(
    InsertPointTy AllocaIP, Value *X, Type *XElemTy, Value *Expr,
    AtomicOrdering AO, AtomicRMWInst::BinOp RMWOp,
    AtomicUpdateCallbackTy &UpdateOp, bool VolatileX, bool IsXBinopExpr) {
  // TODO: handle the case where XElemTy is not byte-sized or not a power of 2
  // or a complex datatype.
  bool emitRMWOp = false;
  switch (RMWOp) {
  case AtomicRMWInst::Add:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Nand:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Xchg:
    emitRMWOp = XElemTy;
    break;
  case AtomicRMWInst::Sub:
    emitRMWOp = (IsXBinopExpr && XElemTy);
    break;
  default:
    emitRMWOp = false;
  }
  emitRMWOp &= XElemTy->isIntegerTy();

  std::pair<Value *, Value *> Res;
  if (emitRMWOp) {
    Res.first = Builder.CreateAtomicRMW(RMWOp, X, Expr, llvm::MaybeAlign(), AO);
    // not needed except in case of postfix captures. Generate anyway for
    // consistency with the else part. Will be removed with any DCE pass.
    // AtomicRMWInst::Xchg does not have a coressponding instruction.
    if (RMWOp == AtomicRMWInst::Xchg)
      Res.second = Res.first;
    else
      Res.second = emitRMWOpAsInstruction(Res.first, Expr, RMWOp);
  } else {
    IntegerType *IntCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    LoadInst *OldVal =
        Builder.CreateLoad(IntCastTy, X, X->getName() + ".atomic.load");
    OldVal->setAtomic(AO);
    // CurBB
    // |     /---\
		// ContBB    |
    // |     \---/
    // ExitBB
    BasicBlock *CurBB = Builder.GetInsertBlock();
    Instruction *CurBBTI = CurBB->getTerminator();
    CurBBTI = CurBBTI ? CurBBTI : Builder.CreateUnreachable();
    BasicBlock *ExitBB =
        CurBB->splitBasicBlock(CurBBTI, X->getName() + ".atomic.exit");
    BasicBlock *ContBB = CurBB->splitBasicBlock(CurBB->getTerminator(),
                                                X->getName() + ".atomic.cont");
    ContBB->getTerminator()->eraseFromParent();
    Builder.restoreIP(AllocaIP);
    AllocaInst *NewAtomicAddr = Builder.CreateAlloca(XElemTy);
    NewAtomicAddr->setName(X->getName() + "x.new.val");
    Builder.SetInsertPoint(ContBB);
    llvm::PHINode *PHI = Builder.CreatePHI(OldVal->getType(), 2);
    PHI->addIncoming(OldVal, CurBB);
    bool IsIntTy = XElemTy->isIntegerTy();
    Value *OldExprVal = PHI;
    if (!IsIntTy) {
      if (XElemTy->isFloatingPointTy()) {
        OldExprVal = Builder.CreateBitCast(PHI, XElemTy,
                                           X->getName() + ".atomic.fltCast");
      } else {
        OldExprVal = Builder.CreateIntToPtr(PHI, XElemTy,
                                            X->getName() + ".atomic.ptrCast");
      }
    }

    Value *Upd = UpdateOp(OldExprVal, Builder);
    Builder.CreateStore(Upd, NewAtomicAddr);
    LoadInst *DesiredVal = Builder.CreateLoad(IntCastTy, NewAtomicAddr);
    AtomicOrdering Failure =
        llvm::AtomicCmpXchgInst::getStrongestFailureOrdering(AO);
    AtomicCmpXchgInst *Result = Builder.CreateAtomicCmpXchg(
        X, PHI, DesiredVal, llvm::MaybeAlign(), AO, Failure);
    Result->setVolatile(VolatileX);
    Value *PreviousVal = Builder.CreateExtractValue(Result, /*Idxs=*/0);
    Value *SuccessFailureVal = Builder.CreateExtractValue(Result, /*Idxs=*/1);
    PHI->addIncoming(PreviousVal, Builder.GetInsertBlock());
    Builder.CreateCondBr(SuccessFailureVal, ExitBB, ContBB);

    Res.first = OldExprVal;
    Res.second = Upd;

    // set Insertion point in exit block
    if (UnreachableInst *ExitTI =
            dyn_cast<UnreachableInst>(ExitBB->getTerminator())) {
      CurBBTI->eraseFromParent();
      Builder.SetInsertPoint(ExitBB);
    } else {
      Builder.SetInsertPoint(ExitTI);
    }
  }

  return Res;
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createAtomicCapture(
    const LocationDescription &Loc, InsertPointTy AllocaIP, AtomicOpValue &X,
    AtomicOpValue &V, Value *Expr, AtomicOrdering AO,
    AtomicRMWInst::BinOp RMWOp, AtomicUpdateCallbackTy &UpdateOp,
    bool UpdateExpr, bool IsPostfixUpdate, bool IsXBinopExpr) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  LLVM_DEBUG({
    Type *XTy = X.Var->getType();
    assert(XTy->isPointerTy() &&
           "OMP Atomic expects a pointer to target memory");
    Type *XElemTy = X.ElemTy;
    assert((XElemTy->isFloatingPointTy() || XElemTy->isIntegerTy() ||
            XElemTy->isPointerTy()) &&
           "OMP atomic capture expected a scalar type");
    assert((RMWOp != AtomicRMWInst::Max) && (RMWOp != AtomicRMWInst::Min) &&
           "OpenMP atomic does not support LT or GT operations");
  });

  // If UpdateExpr is 'x' updated with some `expr` not based on 'x',
  // 'x' is simply atomically rewritten with 'expr'.
  AtomicRMWInst::BinOp AtomicOp = (UpdateExpr ? RMWOp : AtomicRMWInst::Xchg);
  std::pair<Value *, Value *> Result =
      emitAtomicUpdate(AllocaIP, X.Var, X.ElemTy, Expr, AO, AtomicOp, UpdateOp,
                       X.IsVolatile, IsXBinopExpr);

  Value *CapturedVal = (IsPostfixUpdate ? Result.first : Result.second);
  Builder.CreateStore(CapturedVal, V.Var, V.IsVolatile);

  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Capture);
  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createAtomicCompare(
    const LocationDescription &Loc, AtomicOpValue &X, AtomicOpValue &V,
    AtomicOpValue &R, Value *E, Value *D, AtomicOrdering AO,
    omp::OMPAtomicCompareOp Op, bool IsXBinopExpr, bool IsPostfixUpdate,
    bool IsFailOnly) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  assert(X.Var->getType()->isPointerTy() &&
         "OMP atomic expects a pointer to target memory");
  // compare capture
  if (V.Var) {
    assert(V.Var->getType()->isPointerTy() && "v.var must be of pointer type");
    assert(V.ElemTy == X.ElemTy && "x and v must be of same type");
  }

  bool IsInteger = E->getType()->isIntegerTy();

  if (Op == OMPAtomicCompareOp::EQ) {
    AtomicOrdering Failure = AtomicCmpXchgInst::getStrongestFailureOrdering(AO);
    AtomicCmpXchgInst *Result = nullptr;
    if (!IsInteger) {
      IntegerType *IntCastTy =
          IntegerType::get(M.getContext(), X.ElemTy->getScalarSizeInBits());
      Value *EBCast = Builder.CreateBitCast(E, IntCastTy);
      Value *DBCast = Builder.CreateBitCast(D, IntCastTy);
      Result = Builder.CreateAtomicCmpXchg(X.Var, EBCast, DBCast, MaybeAlign(),
                                           AO, Failure);
    } else {
      Result =
          Builder.CreateAtomicCmpXchg(X.Var, E, D, MaybeAlign(), AO, Failure);
    }

    if (V.Var) {
      Value *OldValue = Builder.CreateExtractValue(Result, /*Idxs=*/0);
      if (!IsInteger)
        OldValue = Builder.CreateBitCast(OldValue, X.ElemTy);
      assert(OldValue->getType() == V.ElemTy &&
             "OldValue and V must be of same type");
      if (IsPostfixUpdate) {
        Builder.CreateStore(OldValue, V.Var, V.IsVolatile);
      } else {
        Value *SuccessOrFail = Builder.CreateExtractValue(Result, /*Idxs=*/1);
        if (IsFailOnly) {
          // CurBB----
          //   |     |
          //   v     |
          // ContBB  |
          //   |     |
          //   v     |
          // ExitBB <-
          //
          // where ContBB only contains the store of old value to 'v'.
          BasicBlock *CurBB = Builder.GetInsertBlock();
          Instruction *CurBBTI = CurBB->getTerminator();
          CurBBTI = CurBBTI ? CurBBTI : Builder.CreateUnreachable();
          BasicBlock *ExitBB = CurBB->splitBasicBlock(
              CurBBTI, X.Var->getName() + ".atomic.exit");
          BasicBlock *ContBB = CurBB->splitBasicBlock(
              CurBB->getTerminator(), X.Var->getName() + ".atomic.cont");
          ContBB->getTerminator()->eraseFromParent();
          CurBB->getTerminator()->eraseFromParent();

          Builder.CreateCondBr(SuccessOrFail, ExitBB, ContBB);

          Builder.SetInsertPoint(ContBB);
          Builder.CreateStore(OldValue, V.Var);
          Builder.CreateBr(ExitBB);

          if (UnreachableInst *ExitTI =
                  dyn_cast<UnreachableInst>(ExitBB->getTerminator())) {
            CurBBTI->eraseFromParent();
            Builder.SetInsertPoint(ExitBB);
          } else {
            Builder.SetInsertPoint(ExitTI);
          }
        } else {
          Value *CapturedValue =
              Builder.CreateSelect(SuccessOrFail, E, OldValue);
          Builder.CreateStore(CapturedValue, V.Var, V.IsVolatile);
        }
      }
    }
    // The comparison result has to be stored.
    if (R.Var) {
      assert(R.Var->getType()->isPointerTy() &&
             "r.var must be of pointer type");
      assert(R.ElemTy->isIntegerTy() && "r must be of integral type");

      Value *SuccessFailureVal = Builder.CreateExtractValue(Result, /*Idxs=*/1);
      Value *ResultCast = R.IsSigned
                              ? Builder.CreateSExt(SuccessFailureVal, R.ElemTy)
                              : Builder.CreateZExt(SuccessFailureVal, R.ElemTy);
      Builder.CreateStore(ResultCast, R.Var, R.IsVolatile);
    }
  } else {
    assert((Op == OMPAtomicCompareOp::MAX || Op == OMPAtomicCompareOp::MIN) &&
           "Op should be either max or min at this point");
    assert(!IsFailOnly && "IsFailOnly is only valid when the comparison is ==");

    // Reverse the ordop as the OpenMP forms are different from LLVM forms.
    // Let's take max as example.
    // OpenMP form:
    // x = x > expr ? expr : x;
    // LLVM form:
    // *ptr = *ptr > val ? *ptr : val;
    // We need to transform to LLVM form.
    // x = x <= expr ? x : expr;
    AtomicRMWInst::BinOp NewOp;
    if (IsXBinopExpr) {
      if (IsInteger) {
        if (X.IsSigned)
          NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::Min
                                                : AtomicRMWInst::Max;
        else
          NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::UMin
                                                : AtomicRMWInst::UMax;
      } else {
        NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::FMin
                                              : AtomicRMWInst::FMax;
      }
    } else {
      if (IsInteger) {
        if (X.IsSigned)
          NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::Max
                                                : AtomicRMWInst::Min;
        else
          NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::UMax
                                                : AtomicRMWInst::UMin;
      } else {
        NewOp = Op == OMPAtomicCompareOp::MAX ? AtomicRMWInst::FMax
                                              : AtomicRMWInst::FMin;
      }
    }

    AtomicRMWInst *OldValue =
        Builder.CreateAtomicRMW(NewOp, X.Var, E, MaybeAlign(), AO);
    if (V.Var) {
      Value *CapturedValue = nullptr;
      if (IsPostfixUpdate) {
        CapturedValue = OldValue;
      } else {
        CmpInst::Predicate Pred;
        switch (NewOp) {
        case AtomicRMWInst::Max:
          Pred = CmpInst::ICMP_SGT;
          break;
        case AtomicRMWInst::UMax:
          Pred = CmpInst::ICMP_UGT;
          break;
        case AtomicRMWInst::FMax:
          Pred = CmpInst::FCMP_OGT;
          break;
        case AtomicRMWInst::Min:
          Pred = CmpInst::ICMP_SLT;
          break;
        case AtomicRMWInst::UMin:
          Pred = CmpInst::ICMP_ULT;
          break;
        case AtomicRMWInst::FMin:
          Pred = CmpInst::FCMP_OLT;
          break;
        default:
          llvm_unreachable("unexpected comparison op");
        }
        Value *NonAtomicCmp = Builder.CreateCmp(Pred, OldValue, E);
        CapturedValue = Builder.CreateSelect(NonAtomicCmp, E, OldValue);
      }
      Builder.CreateStore(CapturedValue, V.Var, V.IsVolatile);
    }
  }

  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Compare);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createTeams(const LocationDescription &Loc,
                             BodyGenCallbackTy BodyGenCB) {
  if (!updateToLocation(Loc))
    return InsertPointTy();

  uint32_t SrcLocStrSize;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc, SrcLocStrSize);
  Value *Ident = getOrCreateIdent(SrcLocStr, SrcLocStrSize);
  Function *CurrentFunction = Builder.GetInsertBlock()->getParent();

  // Outer allocation basicblock is the entry block of the current function.
  BasicBlock &OuterAllocaBB = CurrentFunction->getEntryBlock();
  if (&OuterAllocaBB == Builder.GetInsertBlock()) {
    BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "teams.entry");
    Builder.SetInsertPoint(BodyBB, BodyBB->begin());
  }

  // The current basic block is split into four basic blocks. After outlining,
  // they will be mapped as follows:
  // ```
  // def current_fn() {
  //   current_basic_block:
  //     br label %teams.exit
  //   teams.exit:
  //     ; instructions after teams
  // }
  //
  // def outlined_fn() {
  //   teams.alloca:
  //     br label %teams.body
  //   teams.body:
  //     ; instructions within teams body
  // }
  // ```
  BasicBlock *ExitBB = splitBB(Builder, /*CreateBranch=*/true, "teams.exit");
  BasicBlock *BodyBB = splitBB(Builder, /*CreateBranch=*/true, "teams.body");
  BasicBlock *AllocaBB =
      splitBB(Builder, /*CreateBranch=*/true, "teams.alloca");

  OutlineInfo OI;
  OI.EntryBB = AllocaBB;
  OI.ExitBB = ExitBB;
  OI.OuterAllocaBB = &OuterAllocaBB;
  OI.PostOutlineCB = [this, Ident](Function &OutlinedFn) {
    // The input IR here looks like the following-
    // ```
    // func @current_fn() {
    //   outlined_fn(%args)
    // }
    // func @outlined_fn(%args) { ... }
    // ```
    //
    // This is changed to the following-
    //
    // ```
    // func @current_fn() {
    //   runtime_call(..., wrapper_fn, ...)
    // }
    // func @wrapper_fn(..., %args) {
    //   outlined_fn(%args)
    // }
    // func @outlined_fn(%args) { ... }
    // ```

    // The stale call instruction will be replaced with a new call instruction
    // for runtime call with a wrapper function.

    assert(OutlinedFn.getNumUses() == 1 &&
           "there must be a single user for the outlined function");
    CallInst *StaleCI = cast<CallInst>(OutlinedFn.user_back());

    // Create the wrapper function.
    SmallVector<Type *> WrapperArgTys{Builder.getPtrTy(), Builder.getPtrTy()};
    for (auto &Arg : OutlinedFn.args())
      WrapperArgTys.push_back(Arg.getType());
    FunctionCallee WrapperFuncVal = M.getOrInsertFunction(
        (Twine(OutlinedFn.getName()) + ".teams").str(),
        FunctionType::get(Builder.getVoidTy(), WrapperArgTys, false));
    Function *WrapperFunc = dyn_cast<Function>(WrapperFuncVal.getCallee());
    WrapperFunc->getArg(0)->setName("global_tid");
    WrapperFunc->getArg(1)->setName("bound_tid");
    if (WrapperFunc->arg_size() > 2)
      WrapperFunc->getArg(2)->setName("data");

    // Emit the body of the wrapper function - just a call to outlined function
    // and return statement.
    BasicBlock *WrapperEntryBB =
        BasicBlock::Create(M.getContext(), "entrybb", WrapperFunc);
    Builder.SetInsertPoint(WrapperEntryBB);
    SmallVector<Value *> Args;
    for (size_t ArgIndex = 2; ArgIndex < WrapperFunc->arg_size(); ArgIndex++)
      Args.push_back(WrapperFunc->getArg(ArgIndex));
    Builder.CreateCall(&OutlinedFn, Args);
    Builder.CreateRetVoid();

    OutlinedFn.addFnAttr(Attribute::AttrKind::AlwaysInline);

    // Call to the runtime function for teams in the current function.
    assert(StaleCI && "Error while outlining - no CallInst user found for the "
                      "outlined function.");
    Builder.SetInsertPoint(StaleCI);
    Args = {Ident, Builder.getInt32(StaleCI->arg_size()), WrapperFunc};
    for (Use &Arg : StaleCI->args())
      Args.push_back(Arg);
    Builder.CreateCall(getOrCreateRuntimeFunctionPtr(
                           omp::RuntimeFunction::OMPRTL___kmpc_fork_teams),
                       Args);
    StaleCI->eraseFromParent();
  };

  addOutlineInfo(std::move(OI));

  // Generate the body of teams.
  InsertPointTy AllocaIP(AllocaBB, AllocaBB->begin());
  InsertPointTy CodeGenIP(BodyBB, BodyBB->begin());
  BodyGenCB(AllocaIP, CodeGenIP);

  Builder.SetInsertPoint(ExitBB, ExitBB->begin());

  return Builder.saveIP();
}

GlobalVariable *
OpenMPIRBuilder::createOffloadMapnames(SmallVectorImpl<llvm::Constant *> &Names,
                                       std::string VarName) {
  llvm::Constant *MapNamesArrayInit = llvm::ConstantArray::get(
      llvm::ArrayType::get(llvm::PointerType::getUnqual(M.getContext()),
                           Names.size()),
      Names);
  auto *MapNamesArrayGlobal = new llvm::GlobalVariable(
      M, MapNamesArrayInit->getType(),
      /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage, MapNamesArrayInit,
      VarName);
  return MapNamesArrayGlobal;
}

// Create all simple and struct types exposed by the runtime and remember
// the llvm::PointerTypes of them for easy access later.
void OpenMPIRBuilder::initializeTypes(Module &M) {
  LLVMContext &Ctx = M.getContext();
  StructType *T;
#define OMP_TYPE(VarName, InitValue) VarName = InitValue;
#define OMP_ARRAY_TYPE(VarName, ElemTy, ArraySize)                             \
  VarName##Ty = ArrayType::get(ElemTy, ArraySize);                             \
  VarName##PtrTy = PointerType::getUnqual(VarName##Ty);
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  VarName = FunctionType::get(ReturnType, {__VA_ARGS__}, IsVarArg);            \
  VarName##Ptr = PointerType::getUnqual(VarName);
#define OMP_STRUCT_TYPE(VarName, StructName, Packed, ...)                      \
  T = StructType::getTypeByName(Ctx, StructName);                              \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName, Packed);            \
  VarName = T;                                                                 \
  VarName##Ptr = PointerType::getUnqual(T);
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

void OpenMPIRBuilder::OutlineInfo::collectBlocks(
    SmallPtrSetImpl<BasicBlock *> &BlockSet,
    SmallVectorImpl<BasicBlock *> &BlockVector) {
  SmallVector<BasicBlock *, 32> Worklist;
  BlockSet.insert(EntryBB);
  BlockSet.insert(ExitBB);

  Worklist.push_back(EntryBB);
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    BlockVector.push_back(BB);
    for (BasicBlock *SuccBB : successors(BB))
      if (BlockSet.insert(SuccBB).second)
        Worklist.push_back(SuccBB);
  }
}

void OpenMPIRBuilder::createOffloadEntry(Constant *ID, Constant *Addr,
                                         uint64_t Size, int32_t Flags,
                                         GlobalValue::LinkageTypes,
                                         StringRef Name) {
  if (!Config.isGPU()) {
    emitOffloadingEntry(ID, Name.empty() ? Addr->getName() : Name, Size, Flags);
    return;
  }
  // TODO: Add support for global variables on the device after declare target
  // support.
  Function *Fn = dyn_cast<Function>(Addr);
  if (!Fn)
    return;

  Module &M = *(Fn->getParent());
  LLVMContext &Ctx = M.getContext();

  // Get "nvvm.annotations" metadata node.
  NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");

  Metadata *MDVals[] = {
      ConstantAsMetadata::get(Fn), MDString::get(Ctx, "kernel"),
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1))};
  // Append metadata to nvvm.annotations.
  MD->addOperand(MDNode::get(Ctx, MDVals));

  // Add a function attribute for the kernel.
  Fn->addFnAttr(Attribute::get(Ctx, "kernel"));
  if (Triple(M.getTargetTriple()).isAMDGCN())
    Fn->addFnAttr("uniform-work-group-size", "true");
  Fn->addFnAttr(Attribute::MustProgress);
}

// We only generate metadata for function that contain target regions.
void OpenMPIRBuilder::createOffloadEntriesAndInfoMetadata(
    EmitMetadataErrorReportFunctionTy &ErrorFn) {

  // If there are no entries, we don't need to do anything.
  if (OffloadInfoManager.empty())
    return;

  LLVMContext &C = M.getContext();
  SmallVector<std::pair<const OffloadEntriesInfoManager::OffloadEntryInfo *,
                        TargetRegionEntryInfo>,
              16>
      OrderedEntries(OffloadInfoManager.size());

  // Auxiliary methods to create metadata values and strings.
  auto &&GetMDInt = [this](unsigned V) {
    return ConstantAsMetadata::get(ConstantInt::get(Builder.getInt32Ty(), V));
  };

  auto &&GetMDString = [&C](StringRef V) { return MDString::get(C, V); };

  // Create the offloading info metadata node.
  NamedMDNode *MD = M.getOrInsertNamedMetadata("omp_offload.info");
  auto &&TargetRegionMetadataEmitter =
      [&C, MD, &OrderedEntries, &GetMDInt, &GetMDString](
          const TargetRegionEntryInfo &EntryInfo,
          const OffloadEntriesInfoManager::OffloadEntryInfoTargetRegion &E) {
        // Generate metadata for target regions. Each entry of this metadata
        // contains:
        // - Entry 0 -> Kind of this type of metadata (0).
        // - Entry 1 -> Device ID of the file where the entry was identified.
        // - Entry 2 -> File ID of the file where the entry was identified.
        // - Entry 3 -> Mangled name of the function where the entry was
        // identified.
        // - Entry 4 -> Line in the file where the entry was identified.
        // - Entry 5 -> Count of regions at this DeviceID/FilesID/Line.
        // - Entry 6 -> Order the entry was created.
        // The first element of the metadata node is the kind.
        Metadata *Ops[] = {
            GetMDInt(E.getKind()),      GetMDInt(EntryInfo.DeviceID),
            GetMDInt(EntryInfo.FileID), GetMDString(EntryInfo.ParentName),
            GetMDInt(EntryInfo.Line),   GetMDInt(EntryInfo.Count),
            GetMDInt(E.getOrder())};

        // Save this entry in the right position of the ordered entries array.
        OrderedEntries[E.getOrder()] = std::make_pair(&E, EntryInfo);

        // Add metadata to the named metadata node.
        MD->addOperand(MDNode::get(C, Ops));
      };

  OffloadInfoManager.actOnTargetRegionEntriesInfo(TargetRegionMetadataEmitter);

  // Create function that emits metadata for each device global variable entry;
  auto &&DeviceGlobalVarMetadataEmitter =
      [&C, &OrderedEntries, &GetMDInt, &GetMDString, MD](
          StringRef MangledName,
          const OffloadEntriesInfoManager::OffloadEntryInfoDeviceGlobalVar &E) {
        // Generate metadata for global variables. Each entry of this metadata
        // contains:
        // - Entry 0 -> Kind of this type of metadata (1).
        // - Entry 1 -> Mangled name of the variable.
        // - Entry 2 -> Declare target kind.
        // - Entry 3 -> Order the entry was created.
        // The first element of the metadata node is the kind.
        Metadata *Ops[] = {GetMDInt(E.getKind()), GetMDString(MangledName),
                           GetMDInt(E.getFlags()), GetMDInt(E.getOrder())};

        // Save this entry in the right position of the ordered entries array.
        TargetRegionEntryInfo varInfo(MangledName, 0, 0, 0);
        OrderedEntries[E.getOrder()] = std::make_pair(&E, varInfo);

        // Add metadata to the named metadata node.
        MD->addOperand(MDNode::get(C, Ops));
      };

  OffloadInfoManager.actOnDeviceGlobalVarEntriesInfo(
      DeviceGlobalVarMetadataEmitter);

  for (const auto &E : OrderedEntries) {
    assert(E.first && "All ordered entries must exist!");
    if (const auto *CE =
            dyn_cast<OffloadEntriesInfoManager::OffloadEntryInfoTargetRegion>(
                E.first)) {
      if (!CE->getID() || !CE->getAddress()) {
        // Do not blame the entry if the parent funtion is not emitted.
        TargetRegionEntryInfo EntryInfo = E.second;
        StringRef FnName = EntryInfo.ParentName;
        if (!M.getNamedValue(FnName))
          continue;
        ErrorFn(EMIT_MD_TARGET_REGION_ERROR, EntryInfo);
        continue;
      }
      createOffloadEntry(CE->getID(), CE->getAddress(),
                         /*Size=*/0, CE->getFlags(),
                         GlobalValue::WeakAnyLinkage);
    } else if (const auto *CE = dyn_cast<
                   OffloadEntriesInfoManager::OffloadEntryInfoDeviceGlobalVar>(
                   E.first)) {
      OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind Flags =
          static_cast<OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind>(
              CE->getFlags());
      switch (Flags) {
      case OffloadEntriesInfoManager::OMPTargetGlobalVarEntryEnter:
      case OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo:
        if (Config.isTargetDevice() && Config.hasRequiresUnifiedSharedMemory())
          continue;
        if (!CE->getAddress()) {
          ErrorFn(EMIT_MD_DECLARE_TARGET_ERROR, E.second);
          continue;
        }
        // The vaiable has no definition - no need to add the entry.
        if (CE->getVarSize() == 0)
          continue;
        break;
      case OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink:
        assert(((Config.isTargetDevice() && !CE->getAddress()) ||
                (!Config.isTargetDevice() && CE->getAddress())) &&
               "Declaret target link address is set.");
        if (Config.isTargetDevice())
          continue;
        if (!CE->getAddress()) {
          ErrorFn(EMIT_MD_GLOBAL_VAR_LINK_ERROR, TargetRegionEntryInfo());
          continue;
        }
        break;
      default:
        break;
      }

      // Hidden or internal symbols on the device are not externally visible.
      // We should not attempt to register them by creating an offloading
      // entry. Indirect variables are handled separately on the device.
      if (auto *GV = dyn_cast<GlobalValue>(CE->getAddress()))
        if ((GV->hasLocalLinkage() || GV->hasHiddenVisibility()) &&
            Flags != OffloadEntriesInfoManager::OMPTargetGlobalVarEntryIndirect)
          continue;

      // Indirect globals need to use a special name that doesn't match the name
      // of the associated host global.
      if (Flags == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryIndirect)
        createOffloadEntry(CE->getAddress(), CE->getAddress(), CE->getVarSize(),
                           Flags, CE->getLinkage(), CE->getVarName());
      else
        createOffloadEntry(CE->getAddress(), CE->getAddress(), CE->getVarSize(),
                           Flags, CE->getLinkage());

    } else {
      llvm_unreachable("Unsupported entry kind.");
    }
  }
}

void TargetRegionEntryInfo::getTargetRegionEntryFnName(
    SmallVectorImpl<char> &Name, StringRef ParentName, unsigned DeviceID,
    unsigned FileID, unsigned Line, unsigned Count) {
  raw_svector_ostream OS(Name);
  OS << "__omp_offloading" << llvm::format("_%x", DeviceID)
     << llvm::format("_%x_", FileID) << ParentName << "_l" << Line;
  if (Count)
    OS << "_" << Count;
}

void OffloadEntriesInfoManager::getTargetRegionEntryFnName(
    SmallVectorImpl<char> &Name, const TargetRegionEntryInfo &EntryInfo) {
  unsigned NewCount = getTargetRegionEntryInfoCount(EntryInfo);
  TargetRegionEntryInfo::getTargetRegionEntryFnName(
      Name, EntryInfo.ParentName, EntryInfo.DeviceID, EntryInfo.FileID,
      EntryInfo.Line, NewCount);
}

TargetRegionEntryInfo
OpenMPIRBuilder::getTargetEntryUniqueInfo(FileIdentifierInfoCallbackTy CallBack,
                                          StringRef ParentName) {
  sys::fs::UniqueID ID;
  auto FileIDInfo = CallBack();
  if (auto EC = sys::fs::getUniqueID(std::get<0>(FileIDInfo), ID)) {
    report_fatal_error(("Unable to get unique ID for file, during "
                        "getTargetEntryUniqueInfo, error message: " +
                        EC.message())
                           .c_str());
  }

  return TargetRegionEntryInfo(ParentName, ID.getDevice(), ID.getFile(),
                               std::get<1>(FileIDInfo));
}

Constant *OpenMPIRBuilder::getAddrOfDeclareTargetVar(
    OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind CaptureClause,
    OffloadEntriesInfoManager::OMPTargetDeviceClauseKind DeviceClause,
    bool IsDeclaration, bool IsExternallyVisible,
    TargetRegionEntryInfo EntryInfo, StringRef MangledName,
    std::vector<GlobalVariable *> &GeneratedRefs, bool OpenMPSIMD,
    std::vector<Triple> TargetTriple, Type *LlvmPtrTy,
    std::function<Constant *()> GlobalInitializer,
    std::function<GlobalValue::LinkageTypes()> VariableLinkage) {
  // TODO: convert this to utilise the IRBuilder Config rather than
  // a passed down argument.
  if (OpenMPSIMD)
    return nullptr;

  if (CaptureClause == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink ||
      ((CaptureClause == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo ||
        CaptureClause ==
            OffloadEntriesInfoManager::OMPTargetGlobalVarEntryEnter) &&
       Config.hasRequiresUnifiedSharedMemory())) {
    SmallString<64> PtrName;
    {
      raw_svector_ostream OS(PtrName);
      OS << MangledName;
      if (!IsExternallyVisible)
        OS << format("_%x", EntryInfo.FileID);
      OS << "_decl_tgt_ref_ptr";
    }

    Value *Ptr = M.getNamedValue(PtrName);

    if (!Ptr) {
      GlobalValue *GlobalValue = M.getNamedValue(MangledName);
      Ptr = getOrCreateInternalVariable(LlvmPtrTy, PtrName);

      auto *GV = cast<GlobalVariable>(Ptr);
      GV->setLinkage(GlobalValue::WeakAnyLinkage);

      if (!Config.isTargetDevice()) {
        if (GlobalInitializer)
          GV->setInitializer(GlobalInitializer());
        else
          GV->setInitializer(GlobalValue);
      }

      registerTargetGlobalVariable(
          CaptureClause, DeviceClause, IsDeclaration, IsExternallyVisible,
          EntryInfo, MangledName, GeneratedRefs, OpenMPSIMD, TargetTriple,
          GlobalInitializer, VariableLinkage, LlvmPtrTy, cast<Constant>(Ptr));
    }

    return cast<Constant>(Ptr);
  }

  return nullptr;
}

void OpenMPIRBuilder::registerTargetGlobalVariable(
    OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind CaptureClause,
    OffloadEntriesInfoManager::OMPTargetDeviceClauseKind DeviceClause,
    bool IsDeclaration, bool IsExternallyVisible,
    TargetRegionEntryInfo EntryInfo, StringRef MangledName,
    std::vector<GlobalVariable *> &GeneratedRefs, bool OpenMPSIMD,
    std::vector<Triple> TargetTriple,
    std::function<Constant *()> GlobalInitializer,
    std::function<GlobalValue::LinkageTypes()> VariableLinkage, Type *LlvmPtrTy,
    Constant *Addr) {
  if (DeviceClause != OffloadEntriesInfoManager::OMPTargetDeviceClauseAny ||
      (TargetTriple.empty() && !Config.isTargetDevice()))
    return;

  OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind Flags;
  StringRef VarName;
  int64_t VarSize;
  GlobalValue::LinkageTypes Linkage;

  if ((CaptureClause == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo ||
       CaptureClause ==
           OffloadEntriesInfoManager::OMPTargetGlobalVarEntryEnter) &&
      !Config.hasRequiresUnifiedSharedMemory()) {
    Flags = OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo;
    VarName = MangledName;
    GlobalValue *LlvmVal = M.getNamedValue(VarName);

    if (!IsDeclaration)
      VarSize = divideCeil(
          M.getDataLayout().getTypeSizeInBits(LlvmVal->getValueType()), 8);
    else
      VarSize = 0;
    Linkage = (VariableLinkage) ? VariableLinkage() : LlvmVal->getLinkage();

    // This is a workaround carried over from Clang which prevents undesired
    // optimisation of internal variables.
    if (Config.isTargetDevice() &&
        (!IsExternallyVisible || Linkage == GlobalValue::LinkOnceODRLinkage)) {
      // Do not create a "ref-variable" if the original is not also available
      // on the host.
      if (!OffloadInfoManager.hasDeviceGlobalVarEntryInfo(VarName))
        return;

      std::string RefName = createPlatformSpecificName({VarName, "ref"});

      if (!M.getNamedValue(RefName)) {
        Constant *AddrRef =
            getOrCreateInternalVariable(Addr->getType(), RefName);
        auto *GvAddrRef = cast<GlobalVariable>(AddrRef);
        GvAddrRef->setConstant(true);
        GvAddrRef->setLinkage(GlobalValue::InternalLinkage);
        GvAddrRef->setInitializer(Addr);
        GeneratedRefs.push_back(GvAddrRef);
      }
    }
  } else {
    if (CaptureClause == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink)
      Flags = OffloadEntriesInfoManager::OMPTargetGlobalVarEntryLink;
    else
      Flags = OffloadEntriesInfoManager::OMPTargetGlobalVarEntryTo;

    if (Config.isTargetDevice()) {
      VarName = (Addr) ? Addr->getName() : "";
      Addr = nullptr;
    } else {
      Addr = getAddrOfDeclareTargetVar(
          CaptureClause, DeviceClause, IsDeclaration, IsExternallyVisible,
          EntryInfo, MangledName, GeneratedRefs, OpenMPSIMD, TargetTriple,
          LlvmPtrTy, GlobalInitializer, VariableLinkage);
      VarName = (Addr) ? Addr->getName() : "";
    }
    VarSize = M.getDataLayout().getPointerSize();
    Linkage = GlobalValue::WeakAnyLinkage;
  }

  OffloadInfoManager.registerDeviceGlobalVarEntryInfo(VarName, Addr, VarSize,
                                                      Flags, Linkage);
}

/// Loads all the offload entries information from the host IR
/// metadata.
void OpenMPIRBuilder::loadOffloadInfoMetadata(Module &M) {
  // If we are in target mode, load the metadata from the host IR. This code has
  // to match the metadata creation in createOffloadEntriesAndInfoMetadata().

  NamedMDNode *MD = M.getNamedMetadata(ompOffloadInfoName);
  if (!MD)
    return;

  for (MDNode *MN : MD->operands()) {
    auto &&GetMDInt = [MN](unsigned Idx) {
      auto *V = cast<ConstantAsMetadata>(MN->getOperand(Idx));
      return cast<ConstantInt>(V->getValue())->getZExtValue();
    };

    auto &&GetMDString = [MN](unsigned Idx) {
      auto *V = cast<MDString>(MN->getOperand(Idx));
      return V->getString();
    };

    switch (GetMDInt(0)) {
    default:
      llvm_unreachable("Unexpected metadata!");
      break;
    case OffloadEntriesInfoManager::OffloadEntryInfo::
        OffloadingEntryInfoTargetRegion: {
      TargetRegionEntryInfo EntryInfo(/*ParentName=*/GetMDString(3),
                                      /*DeviceID=*/GetMDInt(1),
                                      /*FileID=*/GetMDInt(2),
                                      /*Line=*/GetMDInt(4),
                                      /*Count=*/GetMDInt(5));
      OffloadInfoManager.initializeTargetRegionEntryInfo(EntryInfo,
                                                         /*Order=*/GetMDInt(6));
      break;
    }
    case OffloadEntriesInfoManager::OffloadEntryInfo::
        OffloadingEntryInfoDeviceGlobalVar:
      OffloadInfoManager.initializeDeviceGlobalVarEntryInfo(
          /*MangledName=*/GetMDString(1),
          static_cast<OffloadEntriesInfoManager::OMPTargetGlobalVarEntryKind>(
              /*Flags=*/GetMDInt(2)),
          /*Order=*/GetMDInt(3));
      break;
    }
  }
}

void OpenMPIRBuilder::loadOffloadInfoMetadata(StringRef HostFilePath) {
  if (HostFilePath.empty())
    return;

  auto Buf = MemoryBuffer::getFile(HostFilePath);
  if (std::error_code Err = Buf.getError()) {
    report_fatal_error(("error opening host file from host file path inside of "
                        "OpenMPIRBuilder: " +
                        Err.message())
                           .c_str());
  }

  LLVMContext Ctx;
  auto M = expectedToErrorOrAndEmitErrors(
      Ctx, parseBitcodeFile(Buf.get()->getMemBufferRef(), Ctx));
  if (std::error_code Err = M.getError()) {
    report_fatal_error(
        ("error parsing host file inside of OpenMPIRBuilder: " + Err.message())
            .c_str());
  }

  loadOffloadInfoMetadata(*M.get());
}

Function *OpenMPIRBuilder::createRegisterRequires(StringRef Name) {
  // Skip the creation of the registration function if this is device codegen
  if (Config.isTargetDevice())
    return nullptr;

  Builder.ClearInsertionPoint();

  // Create registration function prototype
  auto *RegFnTy = FunctionType::get(Builder.getVoidTy(), {});
  auto *RegFn = Function::Create(
      RegFnTy, GlobalVariable::LinkageTypes::InternalLinkage, Name, M);
  RegFn->setSection(".text.startup");
  RegFn->addFnAttr(Attribute::NoInline);
  RegFn->addFnAttr(Attribute::NoUnwind);

  // Create registration function body
  auto *BB = BasicBlock::Create(M.getContext(), "entry", RegFn);
  ConstantInt *FlagsVal =
      ConstantInt::getSigned(Builder.getInt64Ty(), Config.getRequiresFlags());
  Function *RTLRegFn = getOrCreateRuntimeFunctionPtr(
      omp::RuntimeFunction::OMPRTL___tgt_register_requires);

  Builder.SetInsertPoint(BB);
  Builder.CreateCall(RTLRegFn, {FlagsVal});
  Builder.CreateRetVoid();

  return RegFn;
}

//===----------------------------------------------------------------------===//
// OffloadEntriesInfoManager
//===----------------------------------------------------------------------===//

bool OffloadEntriesInfoManager::empty() const {
  return OffloadEntriesTargetRegion.empty() &&
         OffloadEntriesDeviceGlobalVar.empty();
}

unsigned OffloadEntriesInfoManager::getTargetRegionEntryInfoCount(
    const TargetRegionEntryInfo &EntryInfo) const {
  auto It = OffloadEntriesTargetRegionCount.find(
      getTargetRegionEntryCountKey(EntryInfo));
  if (It == OffloadEntriesTargetRegionCount.end())
    return 0;
  return It->second;
}

void OffloadEntriesInfoManager::incrementTargetRegionEntryInfoCount(
    const TargetRegionEntryInfo &EntryInfo) {
  OffloadEntriesTargetRegionCount[getTargetRegionEntryCountKey(EntryInfo)] =
      EntryInfo.Count + 1;
}

/// Initialize target region entry.
void OffloadEntriesInfoManager::initializeTargetRegionEntryInfo(
    const TargetRegionEntryInfo &EntryInfo, unsigned Order) {
  OffloadEntriesTargetRegion[EntryInfo] =
      OffloadEntryInfoTargetRegion(Order, /*Addr=*/nullptr, /*ID=*/nullptr,
                                   OMPTargetRegionEntryTargetRegion);
  ++OffloadingEntriesNum;
}

void OffloadEntriesInfoManager::registerTargetRegionEntryInfo(
    TargetRegionEntryInfo EntryInfo, Constant *Addr, Constant *ID,
    OMPTargetRegionEntryKind Flags) {
  assert(EntryInfo.Count == 0 && "expected default EntryInfo");

  // Update the EntryInfo with the next available count for this location.
  EntryInfo.Count = getTargetRegionEntryInfoCount(EntryInfo);

  // If we are emitting code for a target, the entry is already initialized,
  // only has to be registered.
  if (OMPBuilder->Config.isTargetDevice()) {
    // This could happen if the device compilation is invoked standalone.
    if (!hasTargetRegionEntryInfo(EntryInfo)) {
      return;
    }
    auto &Entry = OffloadEntriesTargetRegion[EntryInfo];
    Entry.setAddress(Addr);
    Entry.setID(ID);
    Entry.setFlags(Flags);
  } else {
    if (Flags == OffloadEntriesInfoManager::OMPTargetRegionEntryTargetRegion &&
        hasTargetRegionEntryInfo(EntryInfo, /*IgnoreAddressId*/ true))
      return;
    assert(!hasTargetRegionEntryInfo(EntryInfo) &&
           "Target region entry already registered!");
    OffloadEntryInfoTargetRegion Entry(OffloadingEntriesNum, Addr, ID, Flags);
    OffloadEntriesTargetRegion[EntryInfo] = Entry;
    ++OffloadingEntriesNum;
  }
  incrementTargetRegionEntryInfoCount(EntryInfo);
}

bool OffloadEntriesInfoManager::hasTargetRegionEntryInfo(
    TargetRegionEntryInfo EntryInfo, bool IgnoreAddressId) const {

  // Update the EntryInfo with the next available count for this location.
  EntryInfo.Count = getTargetRegionEntryInfoCount(EntryInfo);

  auto It = OffloadEntriesTargetRegion.find(EntryInfo);
  if (It == OffloadEntriesTargetRegion.end()) {
    return false;
  }
  // Fail if this entry is already registered.
  if (!IgnoreAddressId && (It->second.getAddress() || It->second.getID()))
    return false;
  return true;
}

void OffloadEntriesInfoManager::actOnTargetRegionEntriesInfo(
    const OffloadTargetRegionEntryInfoActTy &Action) {
  // Scan all target region entries and perform the provided action.
  for (const auto &It : OffloadEntriesTargetRegion) {
    Action(It.first, It.second);
  }
}

void OffloadEntriesInfoManager::initializeDeviceGlobalVarEntryInfo(
    StringRef Name, OMPTargetGlobalVarEntryKind Flags, unsigned Order) {
  OffloadEntriesDeviceGlobalVar.try_emplace(Name, Order, Flags);
  ++OffloadingEntriesNum;
}

void OffloadEntriesInfoManager::registerDeviceGlobalVarEntryInfo(
    StringRef VarName, Constant *Addr, int64_t VarSize,
    OMPTargetGlobalVarEntryKind Flags, GlobalValue::LinkageTypes Linkage) {
  if (OMPBuilder->Config.isTargetDevice()) {
    // This could happen if the device compilation is invoked standalone.
    if (!hasDeviceGlobalVarEntryInfo(VarName))
      return;
    auto &Entry = OffloadEntriesDeviceGlobalVar[VarName];
    if (Entry.getAddress() && hasDeviceGlobalVarEntryInfo(VarName)) {
      if (Entry.getVarSize() == 0) {
        Entry.setVarSize(VarSize);
        Entry.setLinkage(Linkage);
      }
      return;
    }
    Entry.setVarSize(VarSize);
    Entry.setLinkage(Linkage);
    Entry.setAddress(Addr);
  } else {
    if (hasDeviceGlobalVarEntryInfo(VarName)) {
      auto &Entry = OffloadEntriesDeviceGlobalVar[VarName];
      assert(Entry.isValid() && Entry.getFlags() == Flags &&
             "Entry not initialized!");
      if (Entry.getVarSize() == 0) {
        Entry.setVarSize(VarSize);
        Entry.setLinkage(Linkage);
      }
      return;
    }
    if (Flags == OffloadEntriesInfoManager::OMPTargetGlobalVarEntryIndirect)
      OffloadEntriesDeviceGlobalVar.try_emplace(VarName, OffloadingEntriesNum,
                                                Addr, VarSize, Flags, Linkage,
                                                VarName.str());
    else
      OffloadEntriesDeviceGlobalVar.try_emplace(
          VarName, OffloadingEntriesNum, Addr, VarSize, Flags, Linkage, "");
    ++OffloadingEntriesNum;
  }
}

void OffloadEntriesInfoManager::actOnDeviceGlobalVarEntriesInfo(
    const OffloadDeviceGlobalVarEntryInfoActTy &Action) {
  // Scan all target region entries and perform the provided action.
  for (const auto &E : OffloadEntriesDeviceGlobalVar)
    Action(E.getKey(), E.getValue());
}

//===----------------------------------------------------------------------===//
// CanonicalLoopInfo
//===----------------------------------------------------------------------===//

void CanonicalLoopInfo::collectControlBlocks(
    SmallVectorImpl<BasicBlock *> &BBs) {
  // We only count those BBs as control block for which we do not need to
  // reverse the CFG, i.e. not the loop body which can contain arbitrary control
  // flow. For consistency, this also means we do not add the Body block, which
  // is just the entry to the body code.
  BBs.reserve(BBs.size() + 6);
  BBs.append({getPreheader(), Header, Cond, Latch, Exit, getAfter()});
}

BasicBlock *CanonicalLoopInfo::getPreheader() const {
  assert(isValid() && "Requires a valid canonical loop");
  for (BasicBlock *Pred : predecessors(Header)) {
    if (Pred != Latch)
      return Pred;
  }
  llvm_unreachable("Missing preheader");
}

void CanonicalLoopInfo::setTripCount(Value *TripCount) {
  assert(isValid() && "Requires a valid canonical loop");

  Instruction *CmpI = &getCond()->front();
  assert(isa<CmpInst>(CmpI) && "First inst must compare IV with TripCount");
  CmpI->setOperand(1, TripCount);

#ifndef NDEBUG
  assertOK();
#endif
}

void CanonicalLoopInfo::mapIndVar(
    llvm::function_ref<Value *(Instruction *)> Updater) {
  assert(isValid() && "Requires a valid canonical loop");

  Instruction *OldIV = getIndVar();

  // Record all uses excluding those introduced by the updater. Uses by the
  // CanonicalLoopInfo itself to keep track of the number of iterations are
  // excluded.
  SmallVector<Use *> ReplacableUses;
  for (Use &U : OldIV->uses()) {
    auto *User = dyn_cast<Instruction>(U.getUser());
    if (!User)
      continue;
    if (User->getParent() == getCond())
      continue;
    if (User->getParent() == getLatch())
      continue;
    ReplacableUses.push_back(&U);
  }

  // Run the updater that may introduce new uses
  Value *NewIV = Updater(OldIV);

  // Replace the old uses with the value returned by the updater.
  for (Use *U : ReplacableUses)
    U->set(NewIV);

#ifndef NDEBUG
  assertOK();
#endif
}

void CanonicalLoopInfo::assertOK() const {
#ifndef NDEBUG
  // No constraints if this object currently does not describe a loop.
  if (!isValid())
    return;

  BasicBlock *Preheader = getPreheader();
  BasicBlock *Body = getBody();
  BasicBlock *After = getAfter();

  // Verify standard control-flow we use for OpenMP loops.
  assert(Preheader);
  assert(isa<BranchInst>(Preheader->getTerminator()) &&
         "Preheader must terminate with unconditional branch");
  assert(Preheader->getSingleSuccessor() == Header &&
         "Preheader must jump to header");

  assert(Header);
  assert(isa<BranchInst>(Header->getTerminator()) &&
         "Header must terminate with unconditional branch");
  assert(Header->getSingleSuccessor() == Cond &&
         "Header must jump to exiting block");

  assert(Cond);
  assert(Cond->getSinglePredecessor() == Header &&
         "Exiting block only reachable from header");

  assert(isa<BranchInst>(Cond->getTerminator()) &&
         "Exiting block must terminate with conditional branch");
  assert(size(successors(Cond)) == 2 &&
         "Exiting block must have two successors");
  assert(cast<BranchInst>(Cond->getTerminator())->getSuccessor(0) == Body &&
         "Exiting block's first successor jump to the body");
  assert(cast<BranchInst>(Cond->getTerminator())->getSuccessor(1) == Exit &&
         "Exiting block's second successor must exit the loop");

  assert(Body);
  assert(Body->getSinglePredecessor() == Cond &&
         "Body only reachable from exiting block");
  assert(!isa<PHINode>(Body->front()));

  assert(Latch);
  assert(isa<BranchInst>(Latch->getTerminator()) &&
         "Latch must terminate with unconditional branch");
  assert(Latch->getSingleSuccessor() == Header && "Latch must jump to header");
  // TODO: To support simple redirecting of the end of the body code that has
  // multiple; introduce another auxiliary basic block like preheader and after.
  assert(Latch->getSinglePredecessor() != nullptr);
  assert(!isa<PHINode>(Latch->front()));

  assert(Exit);
  assert(isa<BranchInst>(Exit->getTerminator()) &&
         "Exit block must terminate with unconditional branch");
  assert(Exit->getSingleSuccessor() == After &&
         "Exit block must jump to after block");

  assert(After);
  assert(After->getSinglePredecessor() == Exit &&
         "After block only reachable from exit block");
  assert(After->empty() || !isa<PHINode>(After->front()));

  Instruction *IndVar = getIndVar();
  assert(IndVar && "Canonical induction variable not found?");
  assert(isa<IntegerType>(IndVar->getType()) &&
         "Induction variable must be an integer");
  assert(cast<PHINode>(IndVar)->getParent() == Header &&
         "Induction variable must be a PHI in the loop header");
  assert(cast<PHINode>(IndVar)->getIncomingBlock(0) == Preheader);
  assert(
      cast<ConstantInt>(cast<PHINode>(IndVar)->getIncomingValue(0))->isZero());
  assert(cast<PHINode>(IndVar)->getIncomingBlock(1) == Latch);

  auto *NextIndVar = cast<PHINode>(IndVar)->getIncomingValue(1);
  assert(cast<Instruction>(NextIndVar)->getParent() == Latch);
  assert(cast<BinaryOperator>(NextIndVar)->getOpcode() == BinaryOperator::Add);
  assert(cast<BinaryOperator>(NextIndVar)->getOperand(0) == IndVar);
  assert(cast<ConstantInt>(cast<BinaryOperator>(NextIndVar)->getOperand(1))
             ->isOne());

  Value *TripCount = getTripCount();
  assert(TripCount && "Loop trip count not found?");
  assert(IndVar->getType() == TripCount->getType() &&
         "Trip count and induction variable must have the same type");

  auto *CmpI = cast<CmpInst>(&Cond->front());
  assert(CmpI->getPredicate() == CmpInst::ICMP_ULT &&
         "Exit condition must be a signed less-than comparison");
  assert(CmpI->getOperand(0) == IndVar &&
         "Exit condition must compare the induction variable");
  assert(CmpI->getOperand(1) == TripCount &&
         "Exit condition must compare with the trip count");
#endif
}

void CanonicalLoopInfo::invalidate() {
  Header = nullptr;
  Cond = nullptr;
  Latch = nullptr;
  Exit = nullptr;
}
