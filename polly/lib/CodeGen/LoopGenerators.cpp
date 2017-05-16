//===------ LoopGenerators.cpp -  IR helper to create loops ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGenerators.h"
#include "polly/ScopDetection.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

static cl::opt<int>
    PollyNumThreads("polly-num-threads",
                    cl::desc("Number of threads to use (0 = auto)"), cl::Hidden,
                    cl::init(0));

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
                         ScopAnnotator *Annotator, bool Parallel,
                         bool UseGuard) {
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
  Loop *NewLoop = new Loop();

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
    Annotator->annotateLoopLatch(B, NewLoop, Parallel);

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
  Function *SubFn;

  AllocaInst *Struct = storeValuesIntoStruct(UsedValues);
  BasicBlock::iterator BeforeLoop = Builder.GetInsertPoint();
  Value *IV = createSubFn(Stride, Struct, UsedValues, Map, &SubFn);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(&*BeforeLoop);

  Value *SubFnParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                            "polly.par.userContext");

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateAdd(UB, ConstantInt::get(LongType, 1));

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();

  return IV;
}

void ParallelLoopGenerator::createCallSpawnThreads(Value *SubFn,
                                                   Value *SubFnParam, Value *LB,
                                                   Value *UB, Value *Stride) {
  const std::string Name = "GOMP_parallel_loop_runtime_start";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                          Builder.getVoidTy(), Builder.getInt8PtrTy(), false)),
                      Builder.getInt8PtrTy(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *NumberOfThreads = Builder.getInt32(PollyNumThreads);
  Value *Args[] = {SubFn, SubFnParam, NumberOfThreads, LB, UB, Stride};

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGenerator::createCallGetWorkItem(Value *LBPtr,
                                                    Value *UBPtr) {
  const std::string Name = "GOMP_loop_runtime_next";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {LongType->getPointerTo(), LongType->getPointerTo()};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(
      Return, Builder.CreateZExt(Builder.getFalse(), Return->getType()));
  return Return;
}

void ParallelLoopGenerator::createCallJoinThreads() {
  const std::string Name = "GOMP_parallel_end";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}

void ParallelLoopGenerator::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}

Function *ParallelLoopGenerator::createSubFnDefinition() {
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type *> Arguments(1, Builder.getInt8PtrTy());
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);

  // Certain backends (e.g., NVPTX) do not support '.'s in function names.
  // Hence, we ensure that all '.'s are replaced by '_'s.
  std::string FunctionName = SubFn->getName();
  std::replace(FunctionName.begin(), FunctionName.end(), '.', '_');
  SubFn->setName(FunctionName);

  // Do not run any polly pass on the new function.
  SubFn->addFnAttr(PollySkipFnAttr);

  Function::arg_iterator AI = SubFn->arg_begin();
  AI->setName("polly.par.userContext");

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
    Value *NewValue = Builder.CreateLoad(Address);
    NewValue->setName("polly.subfunc.arg." + OldValues[i]->getName());
    Map[OldValues[i]] = NewValue;
  }
}

Value *ParallelLoopGenerator::createSubFn(Value *Stride, AllocaInst *StructData,
                                          SetVector<Value *> Data,
                                          ValueMapT &Map, Function **SubFnPtr) {
  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *PreHeaderBB, *AfterBB;
  Value *LBPtr, *UBPtr, *UserContext, *Ret1, *HasNextSchedule, *LB, *UB, *IV;
  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Store the previous basic block.
  PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  CheckNextBB = BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  PreHeaderBB = BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  UserContext = Builder.CreateBitCast(
      &*SubFn->arg_begin(), StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Ret1 = createCallGetWorkItem(LBPtr, UBPtr);
  HasNextSchedule = Builder.CreateTrunc(Ret1, Builder.getInt1Ty(),
                                        "polly.par.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, PreHeaderBB, ExitBB);

  // Add code to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(PreHeaderBB);
  LB = Builder.CreateLoad(LBPtr, "polly.par.LB");
  UB = Builder.CreateLoad(UBPtr, "polly.par.UB");

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  IV = createLoop(LB, UB, Stride, Builder, LI, DT, AfterBB, ICmpInst::ICMP_SLE,
                  nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);
  *SubFnPtr = SubFn;

  return IV;
}
