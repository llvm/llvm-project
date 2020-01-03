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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include <sstream>

#define DEBUG_TYPE "openmp-ir-builder"

using namespace llvm;
using namespace omp;
using namespace types;

static cl::opt<bool>
    OptimisticAttributes("openmp-ir-builder-optimistic-attributes", cl::Hidden,
                         cl::desc("Use optimistic attributes describing "
                                  "'as-if' properties of runtime calls."),
                         cl::init(false));

void OpenMPIRBuilder::addAttributes(omp::RuntimeFunction FnID, Function &Fn) {
  LLVMContext &Ctx = Fn.getContext();

#define OMP_ATTRS_SET(VarName, AttrSet) AttributeSet VarName = AttrSet;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // Add attributes to the new declaration.
  switch (FnID) {
#define OMP_RTL_ATTRS(Enum, FnAttrSet, RetAttrSet, ArgAttrSets)                \
  case Enum:                                                                   \
    Fn.setAttributes(                                                          \
        AttributeList::get(Ctx, FnAttrSet, RetAttrSet, ArgAttrSets));          \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  default:
    // Attributes are optional.
    break;
  }
}

Function *OpenMPIRBuilder::getOrCreateRuntimeFunction(RuntimeFunction FnID) {
  Function *Fn = nullptr;

  // Try to find the declation in the module first.
  switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = M.getFunction(Str);                                                   \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }

  if (!Fn) {
    // Create a new declaration if we need one.
    switch (FnID) {
#define OMP_RTL(Enum, Str, IsVarArg, ReturnType, ...)                          \
  case Enum:                                                                   \
    Fn = Function::Create(FunctionType::get(ReturnType,                        \
                                            ArrayRef<Type *>{__VA_ARGS__},     \
                                            IsVarArg),                         \
                          GlobalValue::ExternalLinkage, Str, M);               \
    break;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
    }

    addAttributes(FnID, *Fn);
  }

  assert(Fn && "Failed to create OpenMP runtime function");
  return Fn;
}

void OpenMPIRBuilder::initialize() { initializeTypes(M); }

Value *OpenMPIRBuilder::getOrCreateIdent(Constant *SrcLocStr,
                                         IdentFlag LocFlags) {
  // Enable "C-mode".
  LocFlags |= OMP_IDENT_FLAG_KMPC;

  GlobalVariable *&DefaultIdent = IdentMap[{SrcLocStr, uint64_t(LocFlags)}];
  if (!DefaultIdent) {
    Constant *I32Null = ConstantInt::getNullValue(Int32);
    Constant *IdentData[] = {I32Null,
                             ConstantInt::get(Int32, uint64_t(LocFlags)),
                             I32Null, I32Null, SrcLocStr};
    Constant *Initializer = ConstantStruct::get(
        cast<StructType>(IdentPtr->getPointerElementType()), IdentData);

    // Look for existing encoding of the location + flags, not needed but
    // minimizes the difference to the existing solution while we transition.
    for (GlobalVariable &GV : M.getGlobalList())
      if (GV.getType() == IdentPtr && GV.hasInitializer())
        if (GV.getInitializer() == Initializer)
          return DefaultIdent = &GV;

    DefaultIdent = new GlobalVariable(M, IdentPtr->getPointerElementType(),
                                      /* isConstant = */ false,
                                      GlobalValue::PrivateLinkage, Initializer);
    DefaultIdent->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    DefaultIdent->setAlignment(Align(8));
  }
  return DefaultIdent;
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

    SrcLocStr = Builder.CreateGlobalStringPtr(LocStr);
  }
  return SrcLocStr;
}

Constant *OpenMPIRBuilder::getOrCreateDefaultSrcLocStr() {
  return getOrCreateSrcLocStr(";unknown;unknown;0;0;;");
}

Constant *
OpenMPIRBuilder::getOrCreateSrcLocStr(const LocationDescription &Loc) {
  DILocation *DIL = Loc.DL.get();
  if (!DIL)
    return getOrCreateDefaultSrcLocStr();
  StringRef Filename =
      !DIL->getFilename().empty() ? DIL->getFilename() : M.getName();
  StringRef Function = DIL->getScope()->getSubprogram()->getName();
  Function =
      !Function.empty() ? Function : Loc.IP.getBlock()->getParent()->getName();
  std::string LineStr = std::to_string(DIL->getLine());
  std::string ColumnStr = std::to_string(DIL->getColumn());
  std::stringstream SrcLocStr;
  SrcLocStr << ";" << Filename.data() << ";" << Function.data() << ";"
            << LineStr << ";" << ColumnStr << ";;";
  return getOrCreateSrcLocStr(SrcLocStr.str());
}

Value *OpenMPIRBuilder::getOrCreateThreadID(Value *Ident) {
  return Builder.CreateCall(
      getOrCreateRuntimeFunction(OMPRTL___kmpc_global_thread_num), Ident,
      "omp_global_thread_num");
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::CreateBarrier(const LocationDescription &Loc, Directive DK,
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

  Value *Result = Builder.CreateCall(
      getOrCreateRuntimeFunction(UseCancelBarrier ? OMPRTL___kmpc_cancel_barrier
                                                  : OMPRTL___kmpc_barrier),
      Args);

  if (UseCancelBarrier && CheckCancelFlag)
    emitCancelationCheckImpl(Result, OMPD_parallel);

  return Builder.saveIP();
}

OpenMPIRBuilder::InsertPointTy
OpenMPIRBuilder::CreateCancel(const LocationDescription &Loc,
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
      getOrCreateRuntimeFunction(OMPRTL___kmpc_cancel), Args);

  // The actual cancel logic is shared with others, e.g., cancel_barriers.
  emitCancelationCheckImpl(Result, CanceledDirective);

  // Update the insertion point and remove the terminator we introduced.
  Builder.SetInsertPoint(UI->getParent());
  UI->eraseFromParent();

  return Builder.saveIP();
}

void OpenMPIRBuilder::emitCancelationCheckImpl(
    Value *CancelFlag, omp::Directive CanceledDirective) {
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
  auto &FI = FinalizationStack.back();
  FI.FiniCB(Builder.saveIP());

  // The continuation block is where code generation continues.
  Builder.SetInsertPoint(NonCancellationBlock, NonCancellationBlock->begin());
}

IRBuilder<>::InsertPoint OpenMPIRBuilder::CreateParallel(
    const LocationDescription &Loc, BodyGenCallbackTy BodyGenCB,
    PrivatizeCallbackTy PrivCB, FinalizeCallbackTy FiniCB, Value *IfCondition,
    Value *NumThreads, omp::ProcBindKind ProcBind, bool IsCancellable) {
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
        getOrCreateRuntimeFunction(OMPRTL___kmpc_push_num_threads), Args);
  }

  if (ProcBind != OMP_PROC_BIND_default) {
    // Build call __kmpc_push_proc_bind(&Ident, global_tid, proc_bind)
    Value *Args[] = {
        Ident, ThreadID,
        ConstantInt::get(Int32, unsigned(ProcBind), /*isSigned=*/true)};
    Builder.CreateCall(getOrCreateRuntimeFunction(OMPRTL___kmpc_push_proc_bind),
                       Args);
  }

  BasicBlock *InsertBB = Builder.GetInsertBlock();
  Function *OuterFn = InsertBB->getParent();

  // Vector to remember instructions we used only during the modeling but which
  // we want to delete at the end.
  SmallVector<Instruction *, 4> ToBeDeleted;

  Builder.SetInsertPoint(OuterFn->getEntryBlock().getFirstNonPHI());
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
  InsertPointTy AllocaIP(PRegEntryBB,
                         PRegEntryBB->getTerminator()->getIterator());
  Builder.restoreIP(AllocaIP);
  AllocaInst *PrivTIDAddr =
      Builder.CreateAlloca(Int32, nullptr, "tid.addr.local");
  Instruction *PrivTID = Builder.CreateLoad(PrivTIDAddr, "tid");

  // Add some fake uses for OpenMP provided arguments.
  ToBeDeleted.push_back(Builder.CreateLoad(TIDAddr, "tid.addr.use"));
  ToBeDeleted.push_back(Builder.CreateLoad(ZeroAddr, "zero.addr.use"));

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

  LLVM_DEBUG(dbgs() << "Before body codegen: " << *UI->getFunction() << "\n");

  // Let the caller create the body.
  assert(BodyGenCB && "Expected body generation callback!");
  InsertPointTy CodeGenIP(PRegBodyBB, PRegBodyBB->begin());
  BodyGenCB(AllocaIP, CodeGenIP, *PRegPreFiniBB);

  LLVM_DEBUG(dbgs() << "After  body codegen: " << *UI->getFunction() << "\n");

  SmallPtrSet<BasicBlock *, 32> ParallelRegionBlockSet;
  SmallVector<BasicBlock *, 32> ParallelRegionBlocks, Worklist;
  ParallelRegionBlockSet.insert(PRegEntryBB);
  ParallelRegionBlockSet.insert(PRegExitBB);

  // Collect all blocks in-between PRegEntryBB and PRegExitBB.
  Worklist.push_back(PRegEntryBB);
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    ParallelRegionBlocks.push_back(BB);
    for (BasicBlock *SuccBB : successors(BB))
      if (ParallelRegionBlockSet.insert(SuccBB).second)
        Worklist.push_back(SuccBB);
  }

  CodeExtractorAnalysisCache CEAC(*OuterFn);
  CodeExtractor Extractor(ParallelRegionBlocks, /* DominatorTree */ nullptr,
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

  LLVM_DEBUG(dbgs() << "Before privatization: " << *UI->getFunction() << "\n");

  FunctionCallee TIDRTLFn =
      getOrCreateRuntimeFunction(OMPRTL___kmpc_global_thread_num);

  auto PrivHelper = [&](Value &V) {
    if (&V == TIDAddr || &V == ZeroAddr)
      return;

    SmallVector<Use *, 8> Uses;
    for (Use &U : V.uses())
      if (auto *UserI = dyn_cast<Instruction>(U.getUser()))
        if (ParallelRegionBlockSet.count(UserI->getParent()))
          Uses.push_back(&U);

    Value *ReplacementValue = nullptr;
    CallInst *CI = dyn_cast<CallInst>(&V);
    if (CI && CI->getCalledFunction() == TIDRTLFn.getCallee()) {
      ReplacementValue = PrivTID;
    } else {
      Builder.restoreIP(
          PrivCB(AllocaIP, Builder.saveIP(), V, ReplacementValue));
      assert(ReplacementValue &&
             "Expected copy/create callback to set replacement value!");
      if (ReplacementValue == &V)
        return;
    }

    for (Use *UPtr : Uses)
      UPtr->set(ReplacementValue);
  };

  for (Value *Input : Inputs) {
    LLVM_DEBUG(dbgs() << "Captured input: " << *Input << "\n");
    PrivHelper(*Input);
  }
  for (Value *Output : Outputs) {
    LLVM_DEBUG(dbgs() << "Captured output: " << *Output << "\n");
    PrivHelper(*Output);
  }

  LLVM_DEBUG(dbgs() << "After  privatization: " << *UI->getFunction() << "\n");
  LLVM_DEBUG({
    for (auto *BB : ParallelRegionBlocks)
      dbgs() << " PBR: " << BB->getName() << "\n";
  });

  // Add some known attributes to the outlined function.
  Function *OutlinedFn = Extractor.extractCodeRegion(CEAC);
  OutlinedFn->addParamAttr(0, Attribute::NoAlias);
  OutlinedFn->addParamAttr(1, Attribute::NoAlias);
  OutlinedFn->addFnAttr(Attribute::NoUnwind);
  OutlinedFn->addFnAttr(Attribute::NoRecurse);

  LLVM_DEBUG(dbgs() << "After      outlining: " << *UI->getFunction() << "\n");
  LLVM_DEBUG(dbgs() << "   Outlined function: " << *OutlinedFn << "\n");

  // For compability with the clang CG we move the outlined function after the
  // one with the parallel region.
  OutlinedFn->removeFromParent();
  M.getFunctionList().insertAfter(OuterFn->getIterator(), OutlinedFn);

  // Remove the artificial entry introduced by the extractor right away, we
  // made our own entry block after all.
  {
    BasicBlock &ArtificialEntry = OutlinedFn->getEntryBlock();
    assert(ArtificialEntry.getUniqueSuccessor() == PRegEntryBB);
    assert(PRegEntryBB->getUniquePredecessor() == &ArtificialEntry);
    PRegEntryBB->moveBefore(&ArtificialEntry);
    ArtificialEntry.eraseFromParent();
  }
  LLVM_DEBUG(dbgs() << "PP Outlined function: " << *OutlinedFn << "\n");
  assert(&OutlinedFn->getEntryBlock() == PRegEntryBB);

  assert(OutlinedFn && OutlinedFn->getNumUses() == 1);
  assert(OutlinedFn->arg_size() >= 2 &&
         "Expected at least tid and bounded tid as arguments");
  unsigned NumCapturedVars = OutlinedFn->arg_size() - /* tid & bounded tid */ 2;

  CallInst *CI = cast<CallInst>(OutlinedFn->user_back());
  CI->getParent()->setName("omp_parallel");
  Builder.SetInsertPoint(CI);

  // Build call __kmpc_fork_call(Ident, n, microtask, var1, .., varn);
  Value *ForkCallArgs[] = {Ident, Builder.getInt32(NumCapturedVars),
                           Builder.CreateBitCast(OutlinedFn, ParallelTaskPtr)};

  SmallVector<Value *, 16> RealArgs;
  RealArgs.append(std::begin(ForkCallArgs), std::end(ForkCallArgs));
  RealArgs.append(CI->arg_begin() + /* tid & bound tid */ 2, CI->arg_end());

  FunctionCallee RTLFn = getOrCreateRuntimeFunction(OMPRTL___kmpc_fork_call);
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
          *llvm::MDNode::get(Ctx, {MDB.createCallbackEncoding(
                                      2, {-1, -1},
                                      /* VarArgsArePassed */ true)}));
    }
  }

  Builder.CreateCall(RTLFn, RealArgs);

  LLVM_DEBUG(dbgs() << "With fork_call placed: "
                    << *Builder.GetInsertBlock()->getParent() << "\n");

  InsertPointTy AfterIP(UI->getParent(), UI->getParent()->end());
  InsertPointTy ExitIP(PRegExitBB, PRegExitBB->end());
  UI->eraseFromParent();

  // Initialize the local TID stack location with the argument value.
  Builder.SetInsertPoint(PrivTID);
  Function::arg_iterator OutlinedAI = OutlinedFn->arg_begin();
  Builder.CreateStore(Builder.CreateLoad(OutlinedAI), PrivTIDAddr);

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
        getOrCreateRuntimeFunction(OMPRTL___kmpc_serialized_parallel),
        SerializedParallelCallArgs);

    // OutlinedFn(&GTid, &zero, CapturedStruct);
    CI->removeFromParent();
    Builder.Insert(CI);

    // __kmpc_end_serialized_parallel(&Ident, GTid);
    Value *EndArgs[] = {Ident, ThreadID};
    Builder.CreateCall(
        getOrCreateRuntimeFunction(OMPRTL___kmpc_end_serialized_parallel),
        EndArgs);

    LLVM_DEBUG(dbgs() << "With serialized parallel region: "
                      << *Builder.GetInsertBlock()->getParent() << "\n");
  }

  // Adjust the finalization stack, verify the adjustment, and call the
  // finalize function a last time to finalize values between the pre-fini block
  // and the exit block if we left the parallel "the normal way".
  auto FiniInfo = FinalizationStack.pop_back_val();
  (void)FiniInfo;
  assert(FiniInfo.DK == OMPD_parallel &&
         "Unexpected finalization stack state!");

  Instruction *PreFiniTI = PRegPreFiniBB->getTerminator();
  assert(PreFiniTI->getNumSuccessors() == 1 &&
         PreFiniTI->getSuccessor(0)->size() == 1 &&
         isa<ReturnInst>(PreFiniTI->getSuccessor(0)->getTerminator()) &&
         "Unexpected CFG structure!");

  InsertPointTy PreFiniIP(PRegPreFiniBB, PreFiniTI->getIterator());
  FiniCB(PreFiniIP);

  for (Instruction *I : ToBeDeleted)
    I->eraseFromParent();

  return AfterIP;
}
