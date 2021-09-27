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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include <sstream>

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

void OpenMPIRBuilder::addAttributes(omp::RuntimeFunction FnID, Function &Fn) {
  LLVMContext &Ctx = Fn.getContext();

  // Get the function's current attributes.
  auto Attrs = Fn.getAttributes();
  auto FnAttrs = Attrs.getFnAttrs();
  auto RetAttrs = Attrs.getRetAttrs();
  SmallVector<AttributeSet, 4> ArgAttrs;
  for (size_t ArgNo = 0; ArgNo < Fn.arg_size(); ++ArgNo)
    ArgAttrs.emplace_back(Attrs.getParamAttrs(ArgNo));

#define OMP_ATTRS_SET(VarName, AttrSet) AttributeSet VarName = AttrSet;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // Add attributes to the function declaration.
  switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    FnAttrs = FnAttrs.addAttributes(Ctx, FnAttrSet);                           \
    RetAttrs = RetAttrs.addAttributes(Ctx, RetAttrSet);                        \
    for (size_t ArgNo = 0; ArgNo < ArgAttrSets.size(); ++ArgNo)                \
      ArgAttrs[ArgNo] =                                                        \
          ArgAttrs[ArgNo].addAttributes(Ctx, ArgAttrSets[ArgNo]);              \
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

  // Cast the function to the expected type if necessary
  Constant *C = ConstantExpr::getBitCast(Fn, FnTy->getPointerTo());
  return {FnTy, C};
}

Function *OpenMPIRBuilder::getOrCreateRuntimeFunctionPtr(RuntimeFunction FnID) {
  FunctionCallee RTLFn = getOrCreateRuntimeFunction(M, FnID);
  auto *Fn = dyn_cast<llvm::Function>(RTLFn.getCallee());
  assert(Fn && "Failed to create OpenMP runtime function pointer");
  return Fn;
}

void OpenMPIRBuilder::initialize() { initializeTypes(M); }

void OpenMPIRBuilder::finalize(Function *Fn, bool AllowExtractorSinking) {
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
                            /* AggregateArgs */ false,
                            /* BlockFrequencyInfo */ nullptr,
                            /* BranchProbabilityInfo */ nullptr,
                            /* AssumptionCache */ nullptr,
                            /* AllowVarArgs */ true,
                            /* AllowAlloca */ true,
                            /* Suffix */ ".omp_par");

    LLVM_DEBUG(dbgs() << "Before     outlining: " << *OuterFn << "\n");
    LLVM_DEBUG(dbgs() << "Entry " << OI.EntryBB->getName()
                      << " Exit: " << OI.ExitBB->getName() << "\n");
    assert(Extractor.isEligible() &&
           "Expected OpenMP outlining to be possible!");

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
      if (AllowExtractorSinking) {
        // Move instructions from the to-be-deleted ArtificialEntry to the entry
        // basic block of the parallel region. CodeExtractor may have sunk
        // allocas/bitcasts for values that are solely used in the outlined
        // region and do not escape.
        assert(!ArtificialEntry.empty() &&
               "Expected instructions to sink in the outlined region");
        for (BasicBlock::iterator It = ArtificialEntry.begin(),
                                  End = ArtificialEntry.end();
             It != End;) {
          Instruction &I = *It;
          It++;

          if (I.isTerminator())
            continue;

          I.moveBefore(*OI.EntryBB, OI.EntryBB->getFirstInsertionPt());
        }
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
}

OpenMPIRBuilder::~OpenMPIRBuilder() {
  assert(OutlineInfos.empty() && "There must be no outstanding outlinings");
}

GlobalValue *OpenMPIRBuilder::createDebugKind(unsigned DebugKind) {
  IntegerType *I32Ty = Type::getInt32Ty(M.getContext());
  auto *GV = new GlobalVariable(
      M, I32Ty,
      /* isConstant = */ true, GlobalValue::WeakODRLinkage,
      ConstantInt::get(I32Ty, DebugKind), "__omp_rtl_debug_kind");

  return GV;
}

Value *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                         IdentFlag LocFlags,
                                         unsigned Reserve2Flags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  Value *&Ident =
      IdentMap[{SrcLocStr, uint64_t(LocFlags) << 31 | Reserve2Flags}];
  if (!Ident) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {
        I32Null, ConstantInt::get(Int32, uint32_t(LocFlags)),
        ConstantInt::get(Int32, Reserve2Flags), I32Null, SrcLocStr};
    Constant *Initializer =
        ConstantStruct::get(OpenMPIRBuilder::Ident, IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
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

  return Builder.CreatePointerCast(Ident, IdentPtr);
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(StringRef LocStr) {
  Constant *&SrcLocStr = SrcLocStrMap[LocStr];
  if (!SrcLocStr) {
    Constant *Initializer =
        ConstantDataArray::getString(M.getContext(), LocStr);

    // Look for existing encoding of the location, not needed but minimizes the
    // difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
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
                                                unsigned Line,
                                                unsigned Column) {
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
  return getOrCreateSrcLocStr(Buffer.str());
}

Constant *OpenMPIRBuilder::getOrCreateDefaultSrcLocStr() {
  return getOrCreateSrcLocStr(";unknown;unknown;0;0;;");
}

Constant *OpenMPIRBuilder::getOrCreateSrcLocStr(DebugLoc DL, Function *F) {
  DILocation *DIL = DL.get();
  if (!DIL)
    return getOrCreateDefaultSrcLocStr();
  StringRef FileName = M.getName();
  if (DIFile *DIF = DIL->getFile())
    if (Optional<StringRef> Source = DIF->getSource())
      FileName = *Source;
  StringRef Function = DIL->getScope()->getSubprogram()->getName();
  if (Function.empty() && F)
    Function = F->getName();
  return getOrCreateSrcLocStr(Function, FileName, DIL->getLine(),
                              DIL->getColumn());
}

Constant *
OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc) {
  return getOrCreateSrcLocStr(Loc.DL, Loc.IP.getBlock()->getParent());
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

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr, BarrierLocFlags),
                   getOrCreateThreadID(getOrCreateIdent(SrcLocStr))};

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

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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
  if (!updateToLocation(Loc))
    return Loc.IP;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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

  // If there is an if condition we actually use the TIDAddr and ZeroAddr in the
  // program, otherwise we only need them for modeling purposes to get the
  // associated arguments in the outlined function. In the former case,
  // initialize the allocas properly, in the latter case, delete them later.
  if (IfCondition) {
    Builder.CreateStore(Constant::getNullValue(Int32), TIDAddr);
    Builder.CreateStore(Constant::getNullValue(Int32), ZeroAddr);
  } else {
    ToBeDeleted.push_back(TIDAddr);
    ToBeDeleted.push_back(ZeroAddr);
  }

  // Create an artificial insertion point that will also ensure the blocks we
  // are about to split are not degenerated.
  auto *UI = new UnreachableInst(Builder.getContext(), InsertBB);

  Instruction *ThenTI = UI, *ElseTI = nullptr;
  if (IfCondition)
    SplitBlockAndInsertIfThenElse(IfCondition, UI, &ThenTI, &ElseTI);

  BasicBlock *ThenBB = ThenTI->getParent();
  BasicBlock *PRegEntryBB = ThenBB->splitBasicBlock(ThenTI, "omp.par.entry");
  BasicBlock *PRegBodyBB =
      PRegEntryBB->splitBasicBlock(ThenTI, "omp.par.region");
  BasicBlock *PRegPreFiniBB =
      PRegBodyBB->splitBasicBlock(ThenTI, "omp.par.pre_finalize");
  BasicBlock *PRegExitBB =
      PRegPreFiniBB->splitBasicBlock(ThenTI, "omp.par.exit");

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

  // ThenBB
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
  BodyGenCB(InnerAllocaIP, CodeGenIP, *PRegPreFiniBB);

  LLVM_DEBUG(dbgs() << "After  body codegen: " << *OuterFn << "\n");

  FunctionCallee RTLFn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_fork_call);
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

    // Build call __kmpc_fork_call(Ident, n, microtask, var1, .., varn);
    Value *ForkCallArgs[] = {
        Ident, Builder.getInt32(NumCapturedVars),
        Builder.CreateBitCast(&OutlinedFn, ParallelTaskPtr)};

    SmallVector<Value *, 16> RealArgs;
    RealArgs.append(std::begin(ForkCallArgs), std::end(ForkCallArgs));
    RealArgs.append(CI->arg_begin() + /* tid & bound tid */ 2, CI->arg_end());

    Builder.CreateCall(RTLFn, RealArgs);

    LLVM_DEBUG(dbgs() << "With fork_call placed: "
                      << *Builder.GetInsertBlock()->getParent() << "\n");

    InsertPointTy ExitIP(PRegExitBB, PRegExitBB->end());

    // Initialize the local TID stack location with the argument value.
    Builder.SetInsertPoint(PrivTID);
    Function::arg_iterator OutlinedAI = OutlinedFn.arg_begin();
    Builder.CreateStore(Builder.CreateLoad(Int32, OutlinedAI), PrivTIDAddr);

    // If no "if" clause was present we do not need the call created during
    // outlining, otherwise we reuse it in the serialized parallel region.
    if (!ElseTI) {
      CI->eraseFromParent();
    } else {

      // If an "if" clause was present we are now generating the serialized
      // version into the "else" branch.
      Builder.SetInsertPoint(ElseTI);

      // Build calls __kmpc_serialized_parallel(&Ident, GTid);
      Value *SerializedParallelCallArgs[] = {Ident, ThreadID};
      Builder.CreateCall(
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_serialized_parallel),
          SerializedParallelCallArgs);

      // OutlinedFn(&GTid, &zero, CapturedStruct);
      CI->removeFromParent();
      Builder.Insert(CI);

      // __kmpc_end_serialized_parallel(&Ident, GTid);
      Value *EndArgs[] = {Ident, ThreadID};
      Builder.CreateCall(
          getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_end_serialized_parallel),
          EndArgs);

      LLVM_DEBUG(dbgs() << "With serialized parallel region: "
                        << *Builder.GetInsertBlock()->getParent() << "\n");
    }

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
    if (&V == TIDAddr || &V == ZeroAddr)
      return;

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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Args[] = {getOrCreateIdent(SrcLocStr)};

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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createSections(
    const LocationDescription &Loc, InsertPointTy AllocaIP,
    ArrayRef<StorableBodyGenCallbackTy> SectionCBs, PrivatizeCallbackTy PrivCB,
    FinalizeCallbackTy FiniCB, bool IsCancellable, bool IsNowait) {
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
    auto *CurFn = CodeGenIP.getBlock()->getParent();
    auto *ForIncBB = CodeGenIP.getBlock()->getSingleSuccessor();
    auto *ForExitBB = CodeGenIP.getBlock()
                          ->getSinglePredecessor()
                          ->getTerminator()
                          ->getSuccessor(1);
    SwitchInst *SwitchStmt = Builder.CreateSwitch(IndVar, ForIncBB);
    Builder.restoreIP(CodeGenIP);
    unsigned CaseNumber = 0;
    for (auto SectionCB : SectionCBs) {
      auto *CaseBB = BasicBlock::Create(M.getContext(),
                                        "omp_section_loop.body.case", CurFn);
      SwitchStmt->addCase(Builder.getInt32(CaseNumber), CaseBB);
      Builder.SetInsertPoint(CaseBB);
      SectionCB(InsertPointTy(), Builder.saveIP(), *ForExitBB);
      CaseNumber++;
    }
    // remove the existing terminator from body BB since there can be no
    // terminators after switch/case
    CodeGenIP.getBlock()->getTerminator()->eraseFromParent();
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
      applyStaticWorkshareLoop(Loc.DL, LoopInfo, AllocaIP, true);
  BasicBlock *LoopAfterBB = AfterIP.getBlock();
  Instruction *SplitPos = LoopAfterBB->getTerminator();
  if (!isa_and_nonnull<BranchInst>(SplitPos))
    SplitPos = new UnreachableInst(Builder.getContext(), LoopAfterBB);
  // ExitBB after LoopAfterBB because LoopAfterBB is used for FinalizationCB,
  // which requires a BB with branch
  BasicBlock *ExitBB =
      LoopAfterBB->splitBasicBlock(SplitPos, "omp_sections.end");
  SplitPos->eraseFromParent();

  // Apply the finalization callback in LoopAfterBB
  auto FiniInfo = FinalizationStack.pop_back_val();
  assert(FiniInfo.DK == OMPD_sections &&
         "Unexpected finalization stack state!");
  Builder.SetInsertPoint(LoopAfterBB->getTerminator());
  FiniInfo.FiniCB(Builder.saveIP());
  Builder.SetInsertPoint(ExitBB);

  return Builder.saveIP();
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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  bool CanGenerateAtomic =
      llvm::all_of(ReductionInfos, [](const ReductionInfo &RI) {
        return RI.AtomicReductionGen;
      });
  Value *Ident = getOrCreateIdent(
      SrcLocStr, CanGenerateAtomic ? IdentFlag::OMP_IDENT_FLAG_ATOMIC_REDUCE
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
    Type *ValueType = RI.getElementType();
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
      Builder.restoreIP(RI.AtomicReductionGen(Builder.saveIP(), RI.Variable,
                                              RI.PrivateVariable));
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
  Value *LHSArrayPtr = Builder.CreateBitCast(ReductionFunc->getArg(0),
                                             RedArrayTy->getPointerTo());
  Value *RHSArrayPtr = Builder.CreateBitCast(ReductionFunc->getArg(1),
                                             RedArrayTy->getPointerTo());
  for (auto En : enumerate(ReductionInfos)) {
    const ReductionInfo &RI = En.value();
    Value *LHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArrayTy, LHSArrayPtr, 0, En.index());
    Value *LHSI8Ptr = Builder.CreateLoad(Builder.getInt8PtrTy(), LHSI8PtrPtr);
    Value *LHSPtr = Builder.CreateBitCast(LHSI8Ptr, RI.Variable->getType());
    Value *LHS = Builder.CreateLoad(RI.getElementType(), LHSPtr);
    Value *RHSI8PtrPtr = Builder.CreateConstInBoundsGEP2_64(
        RedArrayTy, RHSArrayPtr, 0, En.index());
    Value *RHSI8Ptr = Builder.CreateLoad(Builder.getInt8PtrTy(), RHSI8PtrPtr);
    Value *RHSPtr =
        Builder.CreateBitCast(RHSI8Ptr, RI.PrivateVariable->getType());
    Value *RHS = Builder.CreateLoad(RI.getElementType(), RHSPtr);
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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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

  CL->Preheader = Preheader;
  CL->Header = Header;
  CL->Cond = Cond;
  CL->Body = Body;
  CL->Latch = Latch;
  CL->Exit = Exit;
  CL->After = After;

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
    Builder.CreateBr(CL->Preheader);
    After->getInstList().splice(After->begin(), BB->getInstList(),
                                Builder.GetInsertPoint(), BB->end());
    After->replaceSuccessorsPhiUsesWith(BB, After);
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
    Value *OneCmp = Builder.CreateICmp(
        InclusiveStop ? CmpInst::ICMP_ULT : CmpInst::ICMP_ULE, Span, Incr);
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

// Sets the number of loop iterations to the given value. This value must be
// valid in the condition block (i.e., defined in the preheader) and is
// interpreted as an unsigned integer.
void setCanonicalLoopTripCount(CanonicalLoopInfo *CLI, Value *TripCount) {
  Instruction *CmpI = &CLI->getCond()->front();
  assert(isa<CmpInst>(CmpI) && "First inst must compare IV with TripCount");
  CmpI->setOperand(1, TripCount);
  CLI->assertOK();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::applyStaticWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                          InsertPointTy AllocaIP,
                                          bool NeedsBarrier, Value *Chunk) {
  assert(CLI->isValid() && "Requires a valid canonical loop");

  // Set up the source location value for OpenMP runtime.
  Builder.restoreIP(CLI->getPreheaderIP());
  Builder.SetCurrentDebugLocation(DL);

  Constant *SrcLocStr = getOrCreateSrcLocStr(DL);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr);

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

  // FIXME: schedule(static) is NOT the same as schedule(static,1)
  if (!Chunk)
    Chunk = One;

  Value *ThreadNum = getOrCreateThreadID(SrcLoc);

  Constant *SchedulingType =
      ConstantInt::get(I32Type, static_cast<int>(OMPScheduleType::Static));

  // Call the "init" function and update the trip count of the loop with the
  // value it produced.
  Builder.CreateCall(StaticInit,
                     {SrcLoc, ThreadNum, SchedulingType, PLastIter, PLowerBound,
                      PUpperBound, PStride, One, Chunk});
  Value *LowerBound = Builder.CreateLoad(IVTy, PLowerBound);
  Value *InclusiveUpperBound = Builder.CreateLoad(IVTy, PUpperBound);
  Value *TripCountMinusOne = Builder.CreateSub(InclusiveUpperBound, LowerBound);
  Value *TripCount = Builder.CreateAdd(TripCountMinusOne, One);
  setCanonicalLoopTripCount(CLI, TripCount);

  // Update all uses of the induction variable except the one in the condition
  // block that compares it with the actual upper bound, and the increment in
  // the latch block.
  // TODO: this can eventually move to CanonicalLoopInfo or to a new
  // CanonicalLoopInfoUpdater interface.
  Builder.SetInsertPoint(CLI->getBody(), CLI->getBody()->getFirstInsertionPt());
  Value *UpdatedIV = Builder.CreateAdd(IV, LowerBound);
  IV->replaceUsesWithIf(UpdatedIV, [&](Use &U) {
    auto *Instr = dyn_cast<Instruction>(U.getUser());
    return !Instr ||
           (Instr->getParent() != CLI->getCond() &&
            Instr->getParent() != CLI->getLatch() && Instr != UpdatedIV);
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

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::applyWorkshareLoop(DebugLoc DL, CanonicalLoopInfo *CLI,
                                    InsertPointTy AllocaIP, bool NeedsBarrier) {
  // Currently only supports static schedules.
  return applyStaticWorkshareLoop(DL, CLI, AllocaIP, NeedsBarrier);
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

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::applyDynamicWorkshareLoop(
    DebugLoc DL, CanonicalLoopInfo *CLI, InsertPointTy AllocaIP,
    OMPScheduleType SchedType, bool NeedsBarrier, Value *Chunk) {
  assert(CLI->isValid() && "Requires a valid canonical loop");

  // Set up the source location value for OpenMP runtime.
  Builder.SetCurrentDebugLocation(DL);

  Constant *SrcLocStr = getOrCreateSrcLocStr(DL);
  Value *SrcLoc = getOrCreateIdent(SrcLocStr);

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
  NewIndVars.set_size(NumLoops);
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
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);
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
  SmallVector<BasicBlock *, 12> OldControlBBs;
  OldControlBBs.reserve(6 * Loops.size());
  for (CanonicalLoopInfo *Loop : Loops)
    Loop->collectControlBlocks(OldControlBBs);
  removeUnusedBlocksFromParent(OldControlBBs);

  for (CanonicalLoopInfo *L : Loops)
    L->invalidate();

#ifndef NDEBUG
  for (CanonicalLoopInfo *GenL : Result)
    GenL->assertOK();
#endif
  return Result;
}

/// Attach loop metadata \p Properties to the loop described by \p Loop. If the
/// loop already has metadata, the loop properties are appended.
static void addLoopMetadata(CanonicalLoopInfo *Loop,
                            ArrayRef<Metadata *> Properties) {
  assert(Loop->isValid() && "Expecting a valid CanonicalLoopInfo");

  // Nothing to do if no property to attach.
  if (Properties.empty())
    return;

  LLVMContext &Ctx = Loop->getFunction()->getContext();
  SmallVector<Metadata *> NewLoopProperties;
  NewLoopProperties.push_back(nullptr);

  // If the loop already has metadata, prepend it to the new metadata.
  BasicBlock *Latch = Loop->getLatch();
  assert(Latch && "A valid CanonicalLoopInfo must have a unique latch");
  MDNode *Existing = Latch->getTerminator()->getMetadata(LLVMContext::MD_loop);
  if (Existing)
    append_range(NewLoopProperties, drop_begin(Existing->operands(), 1));

  append_range(NewLoopProperties, Properties);
  MDNode *LoopID = MDNode::getDistinct(Ctx, NewLoopProperties);
  LoopID->replaceOperandWith(0, LoopID);

  Latch->getTerminator()->setMetadata(LLVMContext::MD_loop, LoopID);
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
createTargetMachine(Function *F, CodeGenOpt::Level OptLevel) {
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
      Triple, CPU, Features, Options, /*RelocModel=*/None, /*CodeModel=*/None,
      OptLevel));
}

/// Heuristically determine the best-performant unroll factor for \p CLI. This
/// depends on the target processor. We are re-using the same heuristics as the
/// LoopUnrollPass.
static int32_t computeHeuristicUnrollFactor(CanonicalLoopInfo *CLI) {
  Function *F = CLI->getFunction();

  // Assume the user requests the most aggressive unrolling, even if the rest of
  // the code is optimized using a lower setting.
  CodeGenOpt::Level OptLevel = CodeGenOpt::Aggressive;
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
                                 /*ProfileSummaryInfo=*/nullptr, ORE, OptLevel,
                                 /*UserThreshold=*/None,
                                 /*UserCount=*/None,
                                 /*UserAllowPartial=*/true,
                                 /*UserAllowRuntime=*/true,
                                 /*UserUpperBound=*/None,
                                 /*UserFullUnrollMaxCount=*/None);

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
                               /*UserUnrollingSpecficValues=*/false);

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

  unsigned NumInlineCandidates;
  bool NotDuplicatable;
  bool Convergent;
  unsigned LoopSize =
      ApproximateLoopSize(L, NumInlineCandidates, NotDuplicatable, Convergent,
                          TTI, EphValues, UP.BEInsns);
  LLVM_DEBUG(dbgs() << "Estimated loop size is " << LoopSize << "\n");

  // Loop is not unrollable if the loop contains certain instructions.
  if (NotDuplicatable || Convergent) {
    LLVM_DEBUG(dbgs() << "Loop not considered unrollable\n");
    return 1;
  }

  // TODO: Determine trip count of \p CLI if constant, computeUnrollCount might
  // be able to use it.
  int TripCount = 0;
  int MaxTripCount = 0;
  bool MaxOrZero = false;
  unsigned TripMultiple = 0;

  bool UseUpperBound = false;
  computeUnrollCount(L, TTI, DT, &LI, SE, EphValues, &ORE, TripCount,
                     MaxTripCount, MaxOrZero, TripMultiple, LoopSize, UP, PP,
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

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);

  llvm::Value *DidItLD = Builder.CreateLoad(Builder.getInt32Ty(), DidIt);

  Value *Args[] = {Ident, ThreadId, BufSize, CpyBuf, CpyFn, DidItLD};

  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_copyprivate);
  Builder.CreateCall(Fn, Args);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createSingle(const LocationDescription &Loc,
                              BodyGenCallbackTy BodyGenCB,
                              FinalizeCallbackTy FiniCB, llvm::Value *DidIt) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  // If needed (i.e. not null), initialize `DidIt` with 0
  if (DidIt) {
    Builder.CreateStore(Builder.getInt32(0), DidIt);
  }

  Directive OMPD = Directive::OMPD_single;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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

  return EmitOMPInlinedRegion(OMPD, EntryCall, ExitCall, BodyGenCB, FiniCB,
                              /*Conditional*/ true, /*hasFinalize*/ true);
}

OpenMPIRBuilder::InsertPointTy OpenMPIRBuilder::createCritical(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    FinalizeCallbackTy FiniCB, StringRef CriticalName, Value *HintInst) {

  if (!updateToLocation(Loc))
    return Loc.IP;

  Directive OMPD = Directive::OMPD_critical;
  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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
    Builder.CreateStore(StoreValues[I], DependAddrGEPIter);
  }

  Value *DependBaseAddrGEP = Builder.CreateInBoundsGEP(
      ArrI64Ty, ArgsBase, {Builder.getInt64(0), Builder.getInt64(0)});

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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
    Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
    Value *Ident = getOrCreateIdent(SrcLocStr);
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
            /* CodeGenIP */ Builder.saveIP(), *FiniBB);

  // If we didn't emit a branch to FiniBB during body generation, it means
  // FiniBB is unreachable (e.g. while(1);). stop generating all the
  // unreachable blocks, and remove anything we are not going to use.
  auto SkipEmittingRegion = FiniBB->hasNPredecessors(0);
  if (SkipEmittingRegion) {
    FiniBB->eraseFromParent();
    ExitCall->eraseFromParent();
    // Discard finalization if we have it.
    if (HasFinalize) {
      assert(!FinalizationStack.empty() &&
             "Unexpected finalization stack state!");
      FinalizationStack.pop_back();
    }
  } else {
    // emit exit call and do any needed finalization.
    auto FinIP = InsertPointTy(FiniBB, FiniBB->getFirstInsertionPt());
    assert(FiniBB->getTerminator()->getNumSuccessors() == 1 &&
           FiniBB->getTerminator()->getSuccessor(0) == ExitBB &&
           "Unexpected control flow graph state!!");
    emitCommonDirectiveExit(OMPD, FinIP, ExitCall, HasFinalize);
    assert(FiniBB->getUniquePredecessor()->getUniqueSuccessor() == FiniBB &&
           "Unexpected Control Flow State!");
    MergeBlockIntoPredecessor(FiniBB);
  }

  // If we are skipping the region of a non conditional, remove the exit
  // block, and clear the builder's insertion point.
  assert(SplitPos->getParent() == ExitBB &&
         "Unexpected Insertion point location!");
  if (!Conditional && SkipEmittingRegion) {
    ExitBB->eraseFromParent();
    Builder.ClearInsertionPoint();
  } else {
    auto merged = MergeBlockIntoPredecessor(ExitBB);
    BasicBlock *ExitPredBB = SplitPos->getParent();
    auto InsertBB = merged ? ExitPredBB : ExitBB;
    if (!isa_and_nonnull<BranchInst>(SplitPos))
      SplitPos->eraseFromParent();
    Builder.SetInsertPoint(InsertBB);
  }

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
  CurFn->getBasicBlockList().insertAfter(EntryBB->getIterator(), ThenBB);

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

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
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

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Value *Args[] = {ThreadId, Addr, Allocator};
  Function *Fn = getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_free);
  return Builder.CreateCall(Fn, Args, Name);
}

CallInst *OpenMPIRBuilder::createCachedThreadPrivate(
    const LocationDescription &Loc, llvm::Value *Pointer,
    llvm::ConstantInt *Size, const llvm::Twine &Name) {
  IRBuilder<>::InsertPointGuard IPG(Builder);
  Builder.restoreIP(Loc.IP);

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  Value *ThreadId = getOrCreateThreadID(Ident);
  Constant *ThreadPrivateCache =
      getOrCreateOMPInternalVariable(Int8PtrPtr, Name);
  llvm::Value *Args[] = {Ident, ThreadId, Pointer, Size, ThreadPrivateCache};

  Function *Fn =
      getOrCreateRuntimeFunctionPtr(OMPRTL___kmpc_threadprivate_cached);

  return Builder.CreateCall(Fn, Args);
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::createTargetInit(const LocationDescription &Loc, bool IsSPMD,
                                  bool RequiresFullRuntime) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  ConstantInt *IsSPMDVal = ConstantInt::getSigned(
      IntegerType::getInt8Ty(Int8->getContext()),
      IsSPMD ? OMP_TGT_EXEC_MODE_SPMD : OMP_TGT_EXEC_MODE_GENERIC);
  ConstantInt *UseGenericStateMachine =
      ConstantInt::getBool(Int32->getContext(), !IsSPMD);
  ConstantInt *RequiresFullRuntimeVal =
      ConstantInt::getBool(Int32->getContext(), RequiresFullRuntime);

  Function *Fn = getOrCreateRuntimeFunctionPtr(
      omp::RuntimeFunction::OMPRTL___kmpc_target_init);

  CallInst *ThreadKind = Builder.CreateCall(
      Fn, {Ident, IsSPMDVal, UseGenericStateMachine, RequiresFullRuntimeVal});

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

void OpenMPIRBuilder::createTargetDeinit(const LocationDescription &Loc,
                                         bool IsSPMD,
                                         bool RequiresFullRuntime) {
  if (!updateToLocation(Loc))
    return;

  Constant *SrcLocStr = getOrCreateSrcLocStr(Loc);
  Value *Ident = getOrCreateIdent(SrcLocStr);
  ConstantInt *IsSPMDVal = ConstantInt::getSigned(
      IntegerType::getInt8Ty(Int8->getContext()),
      IsSPMD ? OMP_TGT_EXEC_MODE_SPMD : OMP_TGT_EXEC_MODE_GENERIC);
  ConstantInt *RequiresFullRuntimeVal =
      ConstantInt::getBool(Int32->getContext(), RequiresFullRuntime);

  Function *Fn = getOrCreateRuntimeFunctionPtr(
      omp::RuntimeFunction::OMPRTL___kmpc_target_deinit);

  Builder.CreateCall(Fn, {Ident, IsSPMDVal, RequiresFullRuntimeVal});
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

Constant *OpenMPIRBuilder::getOrCreateOMPInternalVariable(
    llvm::Type *Ty, const llvm::Twine &Name, unsigned AddressSpace) {
  // TODO: Replace the twine arg with stringref to get rid of the conversion
  // logic. However This is taken from current implementation in clang as is.
  // Since this method is used in many places exclusively for OMP internal use
  // we will keep it as is for temporarily until we move all users to the
  // builder and then, if possible, fix it everywhere in one go.
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name;
  StringRef RuntimeName = Out.str();
  auto &Elem = *InternalVars.try_emplace(RuntimeName, nullptr).first;
  if (Elem.second) {
    assert(Elem.second->getType()->getPointerElementType() == Ty &&
           "OMP internal variable has different type than requested");
  } else {
    // TODO: investigate the appropriate linkage type used for the global
    // variable for possibly changing that to internal or private, or maybe
    // create different versions of the function for different OMP internal
    // variables.
    Elem.second = new llvm::GlobalVariable(
        M, Ty, /*IsConstant*/ false, llvm::GlobalValue::CommonLinkage,
        llvm::Constant::getNullValue(Ty), Elem.first(),
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
        AddressSpace);
  }

  return Elem.second;
}

Value *OpenMPIRBuilder::getOMPCriticalRegionLock(StringRef CriticalName) {
  std::string Prefix = Twine("gomp_critical_user_", CriticalName).str();
  std::string Name = getNameWithSeparators({Prefix, "var"}, ".", ".");
  return getOrCreateOMPInternalVariable(KmpCriticalNameTy, Name);
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
  AllocaInst *ArgsBase = Builder.CreateAlloca(ArrI8PtrTy);
  AllocaInst *Args = Builder.CreateAlloca(ArrI8PtrTy);
  AllocaInst *ArgSizes = Builder.CreateAlloca(ArrI64Ty);
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
  Value *NullPtr = Constant::getNullValue(Int8Ptr->getPointerTo());
  Builder.CreateCall(MapperFunc,
                     {SrcLocInfo, Builder.getInt64(DeviceID),
                      Builder.getInt32(NumOperands), ArgsBaseGEP, ArgsGEP,
                      ArgSizesGEP, MaptypesArg, MapnamesArg, NullPtr});
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

  Type *XTy = X.Var->getType();
  assert(XTy->isPointerTy() && "OMP Atomic expects a pointer to target memory");
  Type *XElemTy = XTy->getPointerElementType();
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
    // We need to bitcast and perform atomic op as integer
    unsigned Addrspace = cast<PointerType>(XTy)->getAddressSpace();
    IntegerType *IntCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    Value *XBCast = Builder.CreateBitCast(
        X.Var, IntCastTy->getPointerTo(Addrspace), "atomic.src.int.cast");
    LoadInst *XLoad =
        Builder.CreateLoad(IntCastTy, XBCast, X.IsVolatile, "omp.atomic.load");
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
  Type *XElemTy = XTy->getPointerElementType();
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
    const LocationDescription &Loc, Instruction *AllocIP, AtomicOpValue &X,
    Value *Expr, AtomicOrdering AO, AtomicRMWInst::BinOp RMWOp,
    AtomicUpdateCallbackTy &UpdateOp, bool IsXLHSInRHSPart) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  LLVM_DEBUG({
    Type *XTy = X.Var->getType();
    assert(XTy->isPointerTy() &&
           "OMP Atomic expects a pointer to target memory");
    Type *XElemTy = XTy->getPointerElementType();
    assert((XElemTy->isFloatingPointTy() || XElemTy->isIntegerTy() ||
            XElemTy->isPointerTy()) &&
           "OMP atomic update expected a scalar type");
    assert((RMWOp != AtomicRMWInst::Max) && (RMWOp != AtomicRMWInst::Min) &&
           (RMWOp != AtomicRMWInst::UMax) && (RMWOp != AtomicRMWInst::UMin) &&
           "OpenMP atomic does not support LT or GT operations");
  });

  emitAtomicUpdate(AllocIP, X.Var, Expr, AO, RMWOp, UpdateOp, X.IsVolatile,
                   IsXLHSInRHSPart);
  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Update);
  return Builder.saveIP();
}

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
    llvm_unreachable("Unsupported atomic update operation");
  }
  llvm_unreachable("Unsupported atomic update operation");
}

std::pair<Value *, Value *>
OpenMPIRBuilder::emitAtomicUpdate(Instruction *AllocIP, Value *X, Value *Expr,
                                  AtomicOrdering AO, AtomicRMWInst::BinOp RMWOp,
                                  AtomicUpdateCallbackTy &UpdateOp,
                                  bool VolatileX, bool IsXLHSInRHSPart) {
  Type *XElemTy = X->getType()->getPointerElementType();

  bool DoCmpExch =
      ((RMWOp == AtomicRMWInst::BAD_BINOP) || (RMWOp == AtomicRMWInst::FAdd)) ||
      (RMWOp == AtomicRMWInst::FSub) ||
      (RMWOp == AtomicRMWInst::Sub && !IsXLHSInRHSPart);

  std::pair<Value *, Value *> Res;
  if (XElemTy->isIntegerTy() && !DoCmpExch) {
    Res.first = Builder.CreateAtomicRMW(RMWOp, X, Expr, llvm::MaybeAlign(), AO);
    // not needed except in case of postfix captures. Generate anyway for
    // consistency with the else part. Will be removed with any DCE pass.
    Res.second = emitRMWOpAsInstruction(Res.first, Expr, RMWOp);
  } else {
    unsigned Addrspace = cast<PointerType>(X->getType())->getAddressSpace();
    IntegerType *IntCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    Value *XBCast =
        Builder.CreateBitCast(X, IntCastTy->getPointerTo(Addrspace));
    LoadInst *OldVal =
        Builder.CreateLoad(IntCastTy, XBCast, X->getName() + ".atomic.load");
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
    Builder.SetInsertPoint(ContBB);
    llvm::PHINode *PHI = Builder.CreatePHI(OldVal->getType(), 2);
    PHI->addIncoming(OldVal, CurBB);
    AllocaInst *NewAtomicAddr = Builder.CreateAlloca(XElemTy);
    NewAtomicAddr->setName(X->getName() + "x.new.val");
    NewAtomicAddr->moveBefore(AllocIP);
    IntegerType *NewAtomicCastTy =
        IntegerType::get(M.getContext(), XElemTy->getScalarSizeInBits());
    bool IsIntTy = XElemTy->isIntegerTy();
    Value *NewAtomicIntAddr =
        (IsIntTy)
            ? NewAtomicAddr
            : Builder.CreateBitCast(NewAtomicAddr,
                                    NewAtomicCastTy->getPointerTo(Addrspace));
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
    LoadInst *DesiredVal = Builder.CreateLoad(XElemTy, NewAtomicIntAddr);
    Value *XAddr =
        (IsIntTy)
            ? X
            : Builder.CreateBitCast(X, IntCastTy->getPointerTo(Addrspace));
    AtomicOrdering Failure =
        llvm::AtomicCmpXchgInst::getStrongestFailureOrdering(AO);
    AtomicCmpXchgInst *Result = Builder.CreateAtomicCmpXchg(
        XAddr, OldExprVal, DesiredVal, llvm::MaybeAlign(), AO, Failure);
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
    const LocationDescription &Loc, Instruction *AllocIP, AtomicOpValue &X,
    AtomicOpValue &V, Value *Expr, AtomicOrdering AO,
    AtomicRMWInst::BinOp RMWOp, AtomicUpdateCallbackTy &UpdateOp,
    bool UpdateExpr, bool IsPostfixUpdate, bool IsXLHSInRHSPart) {
  if (!updateToLocation(Loc))
    return Loc.IP;

  LLVM_DEBUG({
    Type *XTy = X.Var->getType();
    assert(XTy->isPointerTy() &&
           "OMP Atomic expects a pointer to target memory");
    Type *XElemTy = XTy->getPointerElementType();
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
      emitAtomicUpdate(AllocIP, X.Var, Expr, AO, AtomicOp, UpdateOp,
                       X.IsVolatile, IsXLHSInRHSPart);

  Value *CapturedVal = (IsPostfixUpdate ? Result.first : Result.second);
  Builder.CreateStore(CapturedVal, V.Var, V.IsVolatile);

  checkAndEmitFlushAfterAtomic(Loc, AO, AtomicKind::Capture);
  return Builder.saveIP();
}

GlobalVariable *
OpenMPIRBuilder::createOffloadMapnames(SmallVectorImpl<llvm::Constant *> &Names,
                                       std::string VarName) {
  llvm::Constant *MapNamesArrayInit = llvm::ConstantArray::get(
      llvm::ArrayType::get(
          llvm::Type::getInt8Ty(M.getContext())->getPointerTo(), Names.size()),
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
#define OMP_STRUCT_TYPE(VarName, StructName, ...)                              \
  T = StructType::getTypeByName(Ctx, StructName);                              \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName);                    \
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

void CanonicalLoopInfo::collectControlBlocks(
    SmallVectorImpl<BasicBlock *> &BBs) {
  // We only count those BBs as control block for which we do not need to
  // reverse the CFG, i.e. not the loop body which can contain arbitrary control
  // flow. For consistency, this also means we do not add the Body block, which
  // is just the entry to the body code.
  BBs.reserve(BBs.size() + 6);
  BBs.append({Preheader, Header, Cond, Latch, Exit, After});
}

void CanonicalLoopInfo::assertOK() const {
#ifndef NDEBUG
  // No constraints if this object currently does not describe a loop.
  if (!isValid())
    return;

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
  Preheader = nullptr;
  Header = nullptr;
  Cond = nullptr;
  Body = nullptr;
  Latch = nullptr;
  Exit = nullptr;
  After = nullptr;
}
