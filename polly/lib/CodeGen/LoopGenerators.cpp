//===------ LoopGenerators.cpp -  IR helper to create loops ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopDetection.h"
#include "polly/CodeGen/LoopGenerators.h"

#include "llvm/Module.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/DataLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

Value *polly::createLoop(Value *LB, Value *UB, Value *Stride,
                         IRBuilder<> &Builder, Pass *P,
                         BasicBlock *&AfterBlock,
                         ICmpInst::Predicate Predicate) {
  DominatorTree &DT = P->getAnalysis<DominatorTree>();
  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  BasicBlock *PreheaderBB = Builder.GetInsertBlock();
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.loop_header", F);
  BasicBlock *BodyBB = BasicBlock::Create(Context, "polly.loop_body", F);
  BasicBlock *AfterBB = SplitBlock(PreheaderBB, Builder.GetInsertPoint()++, P);
  AfterBB->setName("polly.loop_after");

  PreheaderBB->getTerminator()->setSuccessor(0, HeaderBB);
  DT.addNewBlock(HeaderBB, PreheaderBB);

  Builder.SetInsertPoint(HeaderBB);

  // Use the type of upper and lower bound.
  assert(LB->getType() == UB->getType()
         && "Different types for upper and lower bound.");

  IntegerType *LoopIVType = dyn_cast<IntegerType>(UB->getType());
  assert(LoopIVType && "UB is not integer?");

  // IV
  PHINode *IV = Builder.CreatePHI(LoopIVType, 2, "polly.loopiv");
  IV->addIncoming(LB, PreheaderBB);

  Stride = Builder.CreateZExtOrBitCast(Stride, LoopIVType);
  Value *IncrementedIV = Builder.CreateNSWAdd(IV, Stride, "polly.next_loopiv");

  // Exit condition.
  Value *CMP;
  CMP = Builder.CreateICmp(Predicate, IV, UB);

  Builder.CreateCondBr(CMP, BodyBB, AfterBB);
  DT.addNewBlock(BodyBB, HeaderBB);

  Builder.SetInsertPoint(BodyBB);
  Builder.CreateBr(HeaderBB);
  IV->addIncoming(IncrementedIV, BodyBB);
  DT.changeImmediateDominator(AfterBB, HeaderBB);

  Builder.SetInsertPoint(BodyBB->begin());
  AfterBlock = AfterBB;

  return IV;
}

void OMPGenerator::createCallParallelLoopStart(Value *SubFunction,
                                               Value *SubfunctionParam,
                                               Value *NumberOfThreads,
                                               Value *LowerBound,
                                               Value *UpperBound,
                                               Value *Stride) {
  Module *M = getModule();
  const char *Name = "GOMP_parallel_loop_runtime_start";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    Type *LongTy = getIntPtrTy();
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {
      PointerType::getUnqual(FunctionType::get(Builder.getVoidTy(),
                                               Builder.getInt8PtrTy(),
                                               false)),
      Builder.getInt8PtrTy(),
      Builder.getInt32Ty(),
      LongTy,
      LongTy,
      LongTy,
    };

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {
    SubFunction,
    SubfunctionParam,
    NumberOfThreads,
    LowerBound,
    UpperBound,
    Stride,
  };

  Builder.CreateCall(F, Args);
}

Value *OMPGenerator::createCallLoopNext(Value *LowerBoundPtr,
                                        Value *UpperBoundPtr) {
  Module *M = getModule();
  const char *Name = "GOMP_loop_runtime_next";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    Type *LongPtrTy = PointerType::getUnqual(getIntPtrTy());
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {
      LongPtrTy,
      LongPtrTy,
    };

    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {
    LowerBoundPtr,
    UpperBoundPtr,
  };

  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(Return, Builder.CreateZExt(Builder.getFalse(),
                                                           Return->getType()));
  return Return;
}

void OMPGenerator::createCallParallelEnd() {
  const char *Name = "GOMP_parallel_end";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F);
}

void OMPGenerator::createCallLoopEndNowait() {
  const char *Name = "GOMP_loop_end_nowait";
  Module *M = getModule();
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F);
}

IntegerType *OMPGenerator::getIntPtrTy() {
  return P->getAnalysis<DataLayout>().getIntPtrType(Builder.getContext());
}

Module *OMPGenerator::getModule() {
  return Builder.GetInsertBlock()->getParent()->getParent();
}

Function *OMPGenerator::createSubfunctionDefinition() {
  Module *M = getModule();
  Function *F = Builder.GetInsertBlock()->getParent();
  std::vector<Type*> Arguments(1, Builder.getInt8PtrTy());
  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *FN = Function::Create(FT, Function::InternalLinkage,
                                  F->getName() + ".omp_subfn", M);
  // Do not run any polly pass on the new function.
  P->getAnalysis<polly::ScopDetection>().markFunctionAsInvalid(FN);

  Function::arg_iterator AI = FN->arg_begin();
  AI->setName("omp.userContext");

  return FN;
}

Value *OMPGenerator::loadValuesIntoStruct(SetVector<Value*> &Values) {
  std::vector<Type*> Members;

  for (unsigned i = 0; i < Values.size(); i++)
    Members.push_back(Values[i]->getType());

  StructType *Ty = StructType::get(Builder.getContext(), Members);
  Value *Struct = Builder.CreateAlloca(Ty, 0, "omp.userContext");

  for (unsigned i = 0; i < Values.size(); i++) {
    Value *Address = Builder.CreateStructGEP(Struct, i);
    Builder.CreateStore(Values[i], Address);
  }

  return Struct;
}

void OMPGenerator::extractValuesFromStruct(SetVector<Value*> OldValues,
                                           Value *Struct,
                                           ValueToValueMapTy &Map) {
  for (unsigned i = 0; i < OldValues.size(); i++) {
    Value *Address = Builder.CreateStructGEP(Struct, i);
    Value *NewValue = Builder.CreateLoad(Address);
    Map.insert(std::make_pair<Value*, Value*>(OldValues[i], NewValue));
  }
}

Value *OMPGenerator::createSubfunction(Value *Stride, Value *StructData,
                                       SetVector<Value*> Data,
                                       ValueToValueMapTy &Map,
                                       Function **SubFunction) {
  Function *FN = createSubfunctionDefinition();

  BasicBlock *PrevBB, *HeaderBB, *ExitBB, *CheckNextBB, *LoadIVBoundsBB,
             *AfterBB;
  Value *LowerBoundPtr, *UpperBoundPtr, *UserContext, *Ret1, *HasNextSchedule,
        *LowerBound, *UpperBound, *IV;
  Type *IntPtrTy = getIntPtrTy();
  LLVMContext &Context = FN->getContext();

  // Store the previous basic block.
  PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  HeaderBB = BasicBlock::Create(Context, "omp.setup", FN);
  ExitBB = BasicBlock::Create(Context, "omp.exit", FN);
  CheckNextBB = BasicBlock::Create(Context, "omp.checkNext", FN);
  LoadIVBoundsBB = BasicBlock::Create(Context, "omp.loadIVBounds", FN);

  DominatorTree &DT = P->getAnalysis<DominatorTree>();
  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(LoadIVBoundsBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  LowerBoundPtr = Builder.CreateAlloca(IntPtrTy, 0, "omp.lowerBoundPtr");
  UpperBoundPtr = Builder.CreateAlloca(IntPtrTy, 0, "omp.upperBoundPtr");
  UserContext = Builder.CreateBitCast(FN->arg_begin(), StructData->getType(),
                                      "omp.userContext");

  extractValuesFromStruct(Data, UserContext, Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Ret1 = createCallLoopNext(LowerBoundPtr, UpperBoundPtr);
  HasNextSchedule = Builder.CreateTrunc(Ret1, Builder.getInt1Ty(),
                                        "omp.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, LoadIVBoundsBB, ExitBB);

  // Add code to to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(LoadIVBoundsBB);
  LowerBound = Builder.CreateLoad(LowerBoundPtr, "omp.lowerBound");
  UpperBound = Builder.CreateLoad(UpperBoundPtr, "omp.upperBound");

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UpperBound = Builder.CreateSub(UpperBound, ConstantInt::get(IntPtrTy, 1),
                                 "omp.upperBoundAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(--Builder.GetInsertPoint());
  IV = createLoop(LowerBound, UpperBound, Stride, Builder, P, AfterBB,
                  ICmpInst::ICMP_SLE);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(AfterBB->begin());

  // Add code to terminate this openmp subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallLoopEndNowait();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(LoopBody);
  *SubFunction = FN;

  return IV;
}

Value *OMPGenerator::createParallelLoop(Value *LowerBound, Value *UpperBound,
                                        Value *Stride,
                                        SetVector<Value*> &Values,
                                        ValueToValueMapTy &Map,
                                        BasicBlock::iterator *LoopBody) {
  Value *Struct, *IV, *SubfunctionParam, *NumberOfThreads;
  Function *SubFunction;

  Struct = loadValuesIntoStruct(Values);

  BasicBlock::iterator PrevInsertPoint = Builder.GetInsertPoint();
  IV = createSubfunction(Stride, Struct, Values, Map, &SubFunction);
  *LoopBody = Builder.GetInsertPoint();
  Builder.SetInsertPoint(PrevInsertPoint);

  // Create call for GOMP_parallel_loop_runtime_start.
  SubfunctionParam = Builder.CreateBitCast(Struct, Builder.getInt8PtrTy(),
                                           "omp_data");

  NumberOfThreads = Builder.getInt32(0);

  // Add one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UpperBound = Builder.CreateAdd(UpperBound,
                                 ConstantInt::get(getIntPtrTy(), 1));

  createCallParallelLoopStart(SubFunction, SubfunctionParam, NumberOfThreads,
                              LowerBound, UpperBound, Stride);
  Builder.CreateCall(SubFunction, SubfunctionParam);
  createCallParallelEnd();

  return IV;
}
