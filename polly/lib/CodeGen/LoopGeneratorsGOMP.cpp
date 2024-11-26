//===------ LoopGeneratorsGOMP.cpp - IR helper to create loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGeneratorsGOMP.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace polly;

void ParallelLoopGeneratorGOMP::createCallSpawnThreads(Value *SubFn,
                                                       Value *SubFnParam,
                                                       Value *LB, Value *UB,
                                                       Value *Stride) {
  const std::string Name = "GOMP_parallel_loop_runtime_start";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                          Builder.getVoidTy(), Builder.getPtrTy(), false)),
                      Builder.getPtrTy(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SubFn, SubFnParam, Builder.getInt32(PollyNumThreads),
                   LB,    UB,         Stride};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorGOMP::deployParallelExecution(Function *SubFn,
                                                        Value *SubFnParam,
                                                        Value *LB, Value *UB,
                                                        Value *Stride) {
  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  CallInst *Call = Builder.CreateCall(SubFn, SubFnParam);
  Call->setDebugLoc(DLGenerated);
  createCallJoinThreads();
}

Function *ParallelLoopGeneratorGOMP::prepareSubFnDefinition(Function *F) const {
  FunctionType *FT =
      FunctionType::get(Builder.getVoidTy(), {Builder.getPtrTy()}, false);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);
  // Name the function's arguments
  SubFn->arg_begin()->setName("polly.par.userContext");
  return SubFn;
}

// Create a subfunction of the following (preliminary) structure:
//
//    PrevBB
//       |
//       v
//    HeaderBB
//       |   _____
//       v  v    |
//   CheckNextBB  PreHeaderBB
//       |\       |
//       | \______/
//       |
//       v
//     ExitBB
//
// HeaderBB will hold allocations and loading of variables.
// CheckNextBB will check for more work.
// If there is more work to do: go to PreHeaderBB, otherwise go to ExitBB.
// PreHeaderBB loads the new boundaries (& will lead to the loop body later on).
// ExitBB marks the end of the parallel execution.
std::tuple<Value *, Function *>
ParallelLoopGeneratorGOMP::createSubFn(Value *Stride, AllocaInst *StructData,
                                       SetVector<Value *> Data,
                                       ValueMapT &Map) {
  if (PollyScheduling != OMPGeneralSchedulingType::Runtime) {
    // User tried to influence the scheduling type (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the scheduling type 'runtime'.\n";
  }

  if (PollyChunkSize != 0) {
    // User tried to influence the chunk size (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the default chunk size.\n";
  }

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  SubFnDT = std::make_unique<DominatorTree>(*SubFn);
  SubFnLI = std::make_unique<LoopInfo>(*SubFnDT);

  BasicBlock *ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  BasicBlock *CheckNextBB =
      BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  BasicBlock *PreHeaderBB =
      BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  SubFnDT->addNewBlock(ExitBB, HeaderBB);
  SubFnDT->addNewBlock(CheckNextBB, HeaderBB);
  SubFnDT->addNewBlock(PreHeaderBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  Value *LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  Value *UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  Value *UserContext = &*SubFn->arg_begin();

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Value *Next = createCallGetWorkItem(LBPtr, UBPtr);
  Value *HasNextSchedule = Builder.CreateTrunc(
      Next, Builder.getInt1Ty(), "polly.par.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, PreHeaderBB, ExitBB);

  // Add code to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(PreHeaderBB);
  Value *LB = Builder.CreateLoad(LongType, LBPtr, "polly.par.LB");
  Value *UB = Builder.CreateLoad(LongType, UBPtr, "polly.par.UB");

  // Subtract one as the upper bound provided by OpenMP is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  BasicBlock *AfterBB;
  Value *IV =
      createLoop(LB, UB, Stride, Builder, *SubFnLI, *SubFnDT, AfterBB,
                 ICmpInst::ICMP_SLE, nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);

  // FIXME: Call SubFnDT->verify() and SubFnLI->verify() to check that the
  // DominatorTree/LoopInfo has been created correctly. Alternatively, recreate
  // from scratch since it is not needed here directly.

  return std::make_tuple(IV, SubFn);
}

Value *ParallelLoopGeneratorGOMP::createCallGetWorkItem(Value *LBPtr,
                                                        Value *UBPtr) {
  const std::string Name = "GOMP_loop_runtime_next";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {Builder.getPtrTy(0), Builder.getPtrTy(0)};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
  Value *Return = Builder.CreateICmpNE(
      Call, Builder.CreateZExt(Builder.getFalse(), Call->getType()));
  return Return;
}

void ParallelLoopGeneratorGOMP::createCallJoinThreads() {
  const std::string Name = "GOMP_parallel_end";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  CallInst *Call = Builder.CreateCall(F, {});
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorGOMP::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  CallInst *Call = Builder.CreateCall(F, {});
  Call->setDebugLoc(DLGenerated);
}
