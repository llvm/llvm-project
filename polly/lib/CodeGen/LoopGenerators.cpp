//===------ LoopGenerators.cpp -  IR helper to create loops ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar loops and orchestrate the
// creation of parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGenerators.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

int polly::PollyNumThreads;
OMPGeneralSchedulingType polly::PollyScheduling;
int polly::PollyChunkSize;

static cl::opt<int, true>
    XPollyNumThreads("polly-num-threads",
                     cl::desc("Number of threads to use (0 = auto)"),
                     cl::Hidden, cl::location(polly::PollyNumThreads),
                     cl::init(0), cl::cat(PollyCategory));

static cl::opt<OMPGeneralSchedulingType, true> XPollyScheduling(
    "polly-scheduling",
    cl::desc("Scheduling type of parallel OpenMP for loops"),
    cl::values(clEnumValN(OMPGeneralSchedulingType::StaticChunked, "static",
                          "Static scheduling"),
               clEnumValN(OMPGeneralSchedulingType::Dynamic, "dynamic",
                          "Dynamic scheduling"),
               clEnumValN(OMPGeneralSchedulingType::Guided, "guided",
                          "Guided scheduling"),
               clEnumValN(OMPGeneralSchedulingType::Runtime, "runtime",
                          "Runtime determined (OMP_SCHEDULE)")),
    cl::Hidden, cl::location(polly::PollyScheduling),
    cl::init(OMPGeneralSchedulingType::Runtime), cl::Optional,
    cl::cat(PollyCategory));

static cl::opt<int, true>
    XPollyChunkSize("polly-scheduling-chunksize",
                    cl::desc("Chunksize to use by the OpenMP runtime calls"),
                    cl::Hidden, cl::location(polly::PollyChunkSize),
                    cl::init(0), cl::Optional, cl::cat(PollyCategory));

// We generate a loop of either of the following structures:
//
//              BeforeBB                      BeforeBB
//                 |                             |
//                 v                             v
//              GuardBB                      PreHeaderBB
//              /      |                         |   _____
//     __  PreHeaderBB  |                        v  \/    |
//    /  \    /         |                     HeaderBB  latch
// latch  HeaderBB      |                        |\       |
//    \  /    \         /                        | \------/
//     <       \       /                         |
//              \     /                          v
//              ExitBB                         ExitBB
//
// depending on whether or not we know that it is executed at least once. If
// not, GuardBB checks if the loop is executed at least once. If this is the
// case we branch to PreHeaderBB and subsequently to the HeaderBB, which
// contains the loop iv 'polly.indvar', the incremented loop iv
// 'polly.indvar_next' as well as the condition to check if we execute another
// iteration of the loop. After the loop has finished, we branch to ExitBB.
// We expect the type of UB, LB, UB+Stride to be large enough for values that
// UB may take throughout the execution of the loop, including the computation
// of indvar + Stride before the final abort.
Value *polly::createLoop(Value *LB, Value *UB, Value *Stride,
                         PollyIRBuilder &Builder, LoopInfo &LI,
                         DominatorTree &DT, BasicBlock *&ExitBB,
                         ICmpInst::Predicate Predicate,
                         ScopAnnotator *Annotator, bool Parallel, bool UseGuard,
                         bool LoopVectDisabled) {
  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  assert(LB->getType() == UB->getType() && "Types of loop bounds do not match");
  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  BasicBlock *BeforeBB = Builder.GetInsertBlock();
  BasicBlock *GuardBB =
      UseGuard ? BasicBlock::Create(Context, "polly.loop_if", F) : nullptr;
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *PreHeaderBB =
      BasicBlock::Create(Context, "polly.loop_preheader", F);

  // Update LoopInfo
  Loop *OuterLoop = LI.getLoopFor(BeforeBB);
  Loop *NewLoop = LI.AllocateLoop();

  if (OuterLoop)
    OuterLoop->addChildLoop(NewLoop);
  else
    LI.addTopLevelLoop(NewLoop);

  if (OuterLoop) {
    if (GuardBB)
      OuterLoop->addBasicBlockToLoop(GuardBB, LI);
    OuterLoop->addBasicBlockToLoop(PreHeaderBB, LI);
  }

  NewLoop->addBasicBlockToLoop(HeaderBB, LI);

  // Notify the annotator (if present) that we have a new loop, but only
  // after the header block is set.
  if (Annotator)
    Annotator->pushLoop(NewLoop, Parallel);

  // ExitBB
  ExitBB = SplitBlock(BeforeBB, &*Builder.GetInsertPoint(), &DT, &LI);
  ExitBB->setName("polly.loop_exit");

  // BeforeBB
  if (GuardBB) {
    BeforeBB->getTerminator()->setSuccessor(0, GuardBB);
    DT.addNewBlock(GuardBB, BeforeBB);

    // GuardBB
    Builder.SetInsertPoint(GuardBB);
    Value *LoopGuard;
    LoopGuard = Builder.CreateICmp(Predicate, LB, UB);
    LoopGuard->setName("polly.loop_guard");
    Builder.CreateCondBr(LoopGuard, PreHeaderBB, ExitBB);
    DT.addNewBlock(PreHeaderBB, GuardBB);
  } else {
    BeforeBB->getTerminator()->setSuccessor(0, PreHeaderBB);
    DT.addNewBlock(PreHeaderBB, BeforeBB);
  }

  // PreHeaderBB
  Builder.SetInsertPoint(PreHeaderBB);
  Builder.CreateBr(HeaderBB);

  // HeaderBB
  DT.addNewBlock(HeaderBB, PreHeaderBB);
  Builder.SetInsertPoint(HeaderBB);
  PHINode *IV = Builder.CreatePHI(LoopIVType, 2, "polly.indvar");
  IV->addIncoming(LB, PreHeaderBB);
  Stride = Builder.CreateZExtOrBitCast(Stride, LoopIVType);
  Value *IncrementedIV = Builder.CreateNSWAdd(IV, Stride, "polly.indvar_next");
  Value *LoopCondition =
      Builder.CreateICmp(Predicate, IncrementedIV, UB, "polly.loop_cond");

  // Create the loop latch and annotate it as such.
  BranchInst *B = Builder.CreateCondBr(LoopCondition, HeaderBB, ExitBB);
  if (Annotator)
    Annotator->annotateLoopLatch(B, NewLoop, Parallel, LoopVectDisabled);

  IV->addIncoming(IncrementedIV, HeaderBB);
  if (GuardBB)
    DT.changeImmediateDominator(ExitBB, GuardBB);
  else
    DT.changeImmediateDominator(ExitBB, HeaderBB);

  // The loop body should be added here.
  Builder.SetInsertPoint(HeaderBB->getFirstNonPHI());
  return IV;
}

Value *ParallelLoopGenerator::createParallelLoop(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {

  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();

  Value *IV;
  Function *SubFn;
  std::tie(IV, SubFn) = createSubFn(Stride, Struct, UsedValues, Map);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  // Add one as the upper bound provided by OpenMP is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Execute the prepared subfunction in parallel.
  deployParallelExecution(SubFn, SubFnParam, LB, UB, Stride);

  return IV;
}

Function *ParallelLoopGenerator::createSubFnDefinition() {
  Function *F = Builder.GetInsertBlock()->getParent();
  Function *SubFn = prepareSubFnDefinition(F);

  // Certain backends (e.g., NVPTX) do not support '.'s in function names.
  // Hence, we ensure that all '.'s are replaced by '_'s.
  std::string FunctionName = SubFn->getName().str();
  std::replace(FunctionName.begin(), FunctionName.end(), '.', '_');
  SubFn->setName(FunctionName);

  // Do not run any polly pass on the new function.
  SubFn->addFnAttr(PollySkipFnAttr);

  return SubFn;
}

AllocaInst *
ParallelLoopGenerator::storeValuesIntoStruct(SetVector<Value *> &Values) {
  SmallVector<Type *, 8> Members;

  for (Value *V : Values)
    Members.push_back(V->getType());

  const DataLayout &DL = Builder.GetInsertBlock()->getModule()->getDataLayout();

  // We do not want to allocate the alloca inside any loop, thus we allocate it
  // in the entry block of the function and use annotations to denote the actual
  // live span (similar to clang).
  BasicBlock &EntryBB = Builder.GetInsertBlock()->getParent()->getEntryBlock();
  Instruction *IP = &*EntryBB.getFirstInsertionPt();
  StructType *Ty = StructType::get(Builder.getContext(), Members);
  AllocaInst *Struct = new AllocaInst(Ty, DL.getAllocaAddrSpace(), nullptr,
                                      "polly.par.userContext", IP);

  for (unsigned i = 0; i < Values.size(); i++) {
    Value *Address = Builder.CreateStructGEP(Ty, Struct, i);
    Address->setName("polly.subfn.storeaddr." + Values[i]->getName());
    Builder.CreateStore(Values[i], Address);
  }

  return Struct;
}

void ParallelLoopGenerator::extractValuesFromStruct(
    SetVector<Value *> OldValues, Type *Ty, Value *Struct, ValueMapT &Map) {
  for (unsigned i = 0; i < OldValues.size(); i++) {
    Value *Address = Builder.CreateStructGEP(Ty, Struct, i);
    Type *ElemTy = cast<GetElementPtrInst>(Address)->getResultElementType();
    Value *NewValue = Builder.CreateLoad(ElemTy, Address);
    NewValue->setName("polly.subfunc.arg." + OldValues[i]->getName());
    Map[OldValues[i]] = NewValue;
  }
}

DebugLoc polly::createDebugLocForGeneratedCode(Function *F) {
  if (!F)
    return DebugLoc();

  LLVMContext &Ctx = F->getContext();
  DISubprogram *DILScope =
      dyn_cast_or_null<DISubprogram>(F->getMetadata(LLVMContext::MD_dbg));
  if (!DILScope)
    return DebugLoc();
  return DILocation::get(Ctx, 0, 0, DILScope);
}
