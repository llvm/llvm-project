//===------ LoopGeneratorsKMP.cpp - IR helper to create loops -------------===//
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

#include "polly/CodeGen/LoopGeneratorsKMP.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace polly;

void ParallelLoopGeneratorKMP::createCallSpawnThreads(Value *SubFn,
                                                      Value *SubFnParam,
                                                      Value *LB, Value *UB,
                                                      Value *Stride) {
  const std::string Name = "__kmpc_fork_call";
  Function *F = M->getFunction(Name);
  Type *KMPCMicroTy = StructType::getTypeByName(M->getContext(), "kmpc_micro");

  if (!KMPCMicroTy) {
    // void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid, ...)
    Type *MicroParams[] = {Builder.getPtrTy(0), Builder.getPtrTy(0)};

    KMPCMicroTy = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
  }

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {Builder.getPtrTy(0), Builder.getInt32Ty(),
                      Builder.getPtrTy(0)};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Task =
      Builder.CreatePointerBitCastOrAddrSpaceCast(SubFn, Builder.getPtrTy(0));

  Value *Args[] = {SourceLocationInfo,
                   Builder.getInt32(4) /* Number of arguments (w/o Task) */,
                   Task,
                   LB,
                   UB,
                   Stride,
                   SubFnParam};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorKMP::deployParallelExecution(Function *SubFn,
                                                       Value *SubFnParam,
                                                       Value *LB, Value *UB,
                                                       Value *Stride) {
  // Inform OpenMP runtime about the number of threads if greater than zero
  if (PollyNumThreads > 0) {
    Value *GlobalThreadID = createCallGlobalThreadNum();
    createCallPushNumThreads(GlobalThreadID, Builder.getInt32(PollyNumThreads));
  }

  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
}

Function *ParallelLoopGeneratorKMP::prepareSubFnDefinition(Function *F) const {
  std::vector<Type *> Arguments = {
      Builder.getPtrTy(0), Builder.getPtrTy(0), LongType, LongType, LongType,
      Builder.getPtrTy()};

  FunctionType *FT = FunctionType::get(Builder.getVoidTy(), Arguments, false);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);
  // Name the function's arguments
  Function::arg_iterator AI = SubFn->arg_begin();
  AI->setName("polly.kmpc.global_tid");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.bound_tid");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.lb");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.ub");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.inc");
  std::advance(AI, 1);
  AI->setName("polly.kmpc.shared");

  return SubFn;
}

// Create a subfunction of the following (preliminary) structure:
//
//        PrevBB
//           |
//           v
//        HeaderBB
//       /   |    _____
//      /    v   v     |
//     / PreHeaderBB   |
//    |      |         |
//    |      v         |
//    |  CheckNextBB   |
//     \     |   \_____/
//      \    |
//       v   v
//       ExitBB
//
// HeaderBB will hold allocations, loading of variables and kmp-init calls.
// CheckNextBB will check for more work (dynamic / static chunked) or will be
// empty (static non chunked).
// If there is more work to do: go to PreHeaderBB, otherwise go to ExitBB.
// PreHeaderBB loads the new boundaries (& will lead to the loop body later on).
// Just like CheckNextBB: PreHeaderBB is (preliminary) empty in the static non
// chunked scheduling case. ExitBB marks the end of the parallel execution.
// The possibly empty BasicBlocks will automatically be removed.
std::tuple<Value *, Function *>
ParallelLoopGeneratorKMP::createSubFn(Value *SequentialLoopStride,
                                      AllocaInst *StructData,
                                      SetVector<Value *> Data, ValueMapT &Map) {
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
  Value *IsLastPtr = Builder.CreateAlloca(Builder.getInt32Ty(), nullptr,
                                          "polly.par.lastIterPtr");
  Value *StridePtr =
      Builder.CreateAlloca(LongType, nullptr, "polly.par.StridePtr");

  // Get iterator for retrieving the previously defined parameters.
  Function::arg_iterator AI = SubFn->arg_begin();
  // First argument holds "global thread ID".
  Value *IDPtr = &*AI;
  // Skip "bound thread ID" since it is not used (but had to be defined).
  std::advance(AI, 2);
  // Move iterator to: LB, UB, Stride, Shared variable struct.
  Value *LB = &*AI;
  std::advance(AI, 1);
  Value *UB = &*AI;
  std::advance(AI, 1);
  Value *Stride = &*AI;
  std::advance(AI, 1);
  Value *Shared = &*AI;

  extractValuesFromStruct(Data, StructData->getAllocatedType(), Shared, Map);

  const auto Alignment = llvm::Align(is64BitArch() ? 8 : 4);
  Value *ID = Builder.CreateAlignedLoad(Builder.getInt32Ty(), IDPtr, Alignment,
                                        "polly.par.global_tid");

  Builder.CreateAlignedStore(LB, LBPtr, Alignment);
  Builder.CreateAlignedStore(UB, UBPtr, Alignment);
  Builder.CreateAlignedStore(Builder.getInt32(0), IsLastPtr, Alignment);
  Builder.CreateAlignedStore(Stride, StridePtr, Alignment);

  // Subtract one as the upper bound provided by openmp is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  Value *AdjustedUB = Builder.CreateAdd(UB, ConstantInt::get(LongType, -1),
                                        "polly.indvar.UBAdjusted");

  Value *ChunkSize =
      ConstantInt::get(LongType, std::max<int>(PollyChunkSize, 1));

  OMPGeneralSchedulingType Scheduling =
      getSchedType(PollyChunkSize, PollyScheduling);

  switch (Scheduling) {
  case OMPGeneralSchedulingType::Dynamic:
  case OMPGeneralSchedulingType::Guided:
  case OMPGeneralSchedulingType::Runtime:
    // "DYNAMIC" scheduling types are handled below (including 'runtime')
    {
      UB = AdjustedUB;
      createCallDispatchInit(ID, LB, UB, Stride, ChunkSize);
      Value *HasWork =
          createCallDispatchNext(ID, IsLastPtr, LBPtr, UBPtr, StridePtr);
      Value *HasIteration =
          Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, HasWork,
                             Builder.getInt32(1), "polly.hasIteration");
      Builder.CreateCondBr(HasIteration, PreHeaderBB, ExitBB);

      Builder.SetInsertPoint(CheckNextBB);
      HasWork = createCallDispatchNext(ID, IsLastPtr, LBPtr, UBPtr, StridePtr);
      HasIteration =
          Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, HasWork,
                             Builder.getInt32(1), "polly.hasWork");
      Builder.CreateCondBr(HasIteration, PreHeaderBB, ExitBB);

      Builder.SetInsertPoint(PreHeaderBB);
      LB = Builder.CreateAlignedLoad(LongType, LBPtr, Alignment,
                                     "polly.indvar.LB");
      UB = Builder.CreateAlignedLoad(LongType, UBPtr, Alignment,
                                     "polly.indvar.UB");
    }
    break;
  case OMPGeneralSchedulingType::StaticChunked:
  case OMPGeneralSchedulingType::StaticNonChunked:
    // "STATIC" scheduling types are handled below
    {
      Builder.CreateAlignedStore(AdjustedUB, UBPtr, Alignment);
      createCallStaticInit(ID, IsLastPtr, LBPtr, UBPtr, StridePtr, ChunkSize);

      Value *ChunkedStride = Builder.CreateAlignedLoad(
          LongType, StridePtr, Alignment, "polly.kmpc.stride");

      LB = Builder.CreateAlignedLoad(LongType, LBPtr, Alignment,
                                     "polly.indvar.LB");
      UB = Builder.CreateAlignedLoad(LongType, UBPtr, Alignment,
                                     "polly.indvar.UB.temp");

      Value *UBInRange =
          Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE, UB, AdjustedUB,
                             "polly.indvar.UB.inRange");
      UB = Builder.CreateSelect(UBInRange, UB, AdjustedUB, "polly.indvar.UB");
      Builder.CreateAlignedStore(UB, UBPtr, Alignment);

      Value *HasIteration = Builder.CreateICmp(
          llvm::CmpInst::Predicate::ICMP_SLE, LB, UB, "polly.hasIteration");
      Builder.CreateCondBr(HasIteration, PreHeaderBB, ExitBB);

      if (Scheduling == OMPGeneralSchedulingType::StaticChunked) {
        Builder.SetInsertPoint(PreHeaderBB);
        LB = Builder.CreateAlignedLoad(LongType, LBPtr, Alignment,
                                       "polly.indvar.LB.entry");
        UB = Builder.CreateAlignedLoad(LongType, UBPtr, Alignment,
                                       "polly.indvar.UB.entry");
      }

      Builder.SetInsertPoint(CheckNextBB);

      if (Scheduling == OMPGeneralSchedulingType::StaticChunked) {
        Value *NextLB =
            Builder.CreateAdd(LB, ChunkedStride, "polly.indvar.nextLB");
        Value *NextUB = Builder.CreateAdd(UB, ChunkedStride);

        Value *NextUBOutOfBounds =
            Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SGT, NextUB,
                               AdjustedUB, "polly.indvar.nextUB.outOfBounds");
        NextUB = Builder.CreateSelect(NextUBOutOfBounds, AdjustedUB, NextUB,
                                      "polly.indvar.nextUB");

        Builder.CreateAlignedStore(NextLB, LBPtr, Alignment);
        Builder.CreateAlignedStore(NextUB, UBPtr, Alignment);

        Value *HasWork =
            Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLE, NextLB,
                               AdjustedUB, "polly.hasWork");
        Builder.CreateCondBr(HasWork, PreHeaderBB, ExitBB);
      } else {
        Builder.CreateBr(ExitBB);
      }

      Builder.SetInsertPoint(PreHeaderBB);
    }
    break;
  }

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  BasicBlock *AfterBB;
  Value *IV = createLoop(LB, UB, SequentialLoopStride, Builder, *SubFnLI,
                         *SubFnDT, AfterBB, ICmpInst::ICMP_SLE, nullptr, true,
                         /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  // Static (i.e. non-dynamic) scheduling types, are terminated with a fini-call
  if (Scheduling == OMPGeneralSchedulingType::StaticChunked ||
      Scheduling == OMPGeneralSchedulingType::StaticNonChunked) {
    createCallStaticFini(ID);
  }
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(&*LoopBody);

  // FIXME: Call SubFnDT->verify() and SubFnLI->verify() to check that the
  // DominatorTree/LoopInfo has been created correctly. Alternatively, recreate
  // from scratch since it is not needed here directly.

  return std::make_tuple(IV, SubFn);
}

Value *ParallelLoopGeneratorKMP::createCallGlobalThreadNum() {
  const std::string Name = "__kmpc_global_thread_num";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {Builder.getPtrTy(0)};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  CallInst *Call = Builder.CreateCall(F, {SourceLocationInfo});
  Call->setDebugLoc(DLGenerated);
  return Call;
}

void ParallelLoopGeneratorKMP::createCallPushNumThreads(Value *GlobalThreadID,
                                                        Value *NumThreads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {Builder.getPtrTy(0), Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID, NumThreads};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorKMP::createCallStaticInit(Value *GlobalThreadID,
                                                    Value *IsLastPtr,
                                                    Value *LBPtr, Value *UBPtr,
                                                    Value *StridePtr,
                                                    Value *ChunkSize) {
  const std::string Name =
      is64BitArch() ? "__kmpc_for_static_init_8" : "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {Builder.getPtrTy(0),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getPtrTy(0),
                      Builder.getPtrTy(0),
                      Builder.getPtrTy(0),
                      Builder.getPtrTy(0),
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // The parameter 'ChunkSize' will hold strictly positive integer values,
  // regardless of PollyChunkSize's value
  Value *Args[] = {
      SourceLocationInfo,
      GlobalThreadID,
      Builder.getInt32(int(getSchedType(PollyChunkSize, PollyScheduling))),
      IsLastPtr,
      LBPtr,
      UBPtr,
      StridePtr,
      ConstantInt::get(LongType, 1),
      ChunkSize};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorKMP::createCallStaticFini(Value *GlobalThreadID) {
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {Builder.getPtrTy(0), Builder.getInt32Ty()};
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

void ParallelLoopGeneratorKMP::createCallDispatchInit(Value *GlobalThreadID,
                                                      Value *LB, Value *UB,
                                                      Value *Inc,
                                                      Value *ChunkSize) {
  const std::string Name =
      is64BitArch() ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_4";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {Builder.getPtrTy(0),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // The parameter 'ChunkSize' will hold strictly positive integer values,
  // regardless of PollyChunkSize's value
  Value *Args[] = {
      SourceLocationInfo,
      GlobalThreadID,
      Builder.getInt32(int(getSchedType(PollyChunkSize, PollyScheduling))),
      LB,
      UB,
      Inc,
      ChunkSize};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
}

Value *ParallelLoopGeneratorKMP::createCallDispatchNext(Value *GlobalThreadID,
                                                        Value *IsLastPtr,
                                                        Value *LBPtr,
                                                        Value *UBPtr,
                                                        Value *StridePtr) {
  const std::string Name =
      is64BitArch() ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_4";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {Builder.getPtrTy(0), Builder.getInt32Ty(),
                      Builder.getPtrTy(0), Builder.getPtrTy(0),
                      Builder.getPtrTy(0), Builder.getPtrTy(0)};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID, IsLastPtr, LBPtr, UBPtr,
                   StridePtr};

  CallInst *Call = Builder.CreateCall(F, Args);
  Call->setDebugLoc(DLGenerated);
  return Call;
}

// TODO: This function currently creates a source location dummy. It might be
// necessary to (actually) provide information, in the future.
GlobalVariable *ParallelLoopGeneratorKMP::createSourceLocation() {
  const std::string LocName = ".loc.dummy";
  GlobalVariable *SourceLocDummy = M->getGlobalVariable(LocName);

  if (SourceLocDummy == nullptr) {
    const std::string StructName = "struct.ident_t";
    StructType *IdentTy =
        StructType::getTypeByName(M->getContext(), StructName);

    // If the ident_t StructType is not available, declare it.
    // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
    if (!IdentTy) {
      Type *LocMembers[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                            Builder.getInt32Ty(), Builder.getInt32Ty(),
                            Builder.getPtrTy()};

      IdentTy =
          StructType::create(M->getContext(), LocMembers, StructName, false);
    }

    const auto ArrayType =
        llvm::ArrayType::get(Builder.getInt8Ty(), /* Length */ 23);

    // Global Variable Definitions
    GlobalVariable *StrVar =
        new GlobalVariable(*M, ArrayType, true, GlobalValue::PrivateLinkage,
                           nullptr, ".str.ident");
    StrVar->setAlignment(llvm::Align(1));

    SourceLocDummy = new GlobalVariable(
        *M, IdentTy, true, GlobalValue::PrivateLinkage, nullptr, LocName);
    SourceLocDummy->setAlignment(llvm::Align(8));

    // Constant Definitions
    Constant *InitStr = ConstantDataArray::getString(
        M->getContext(), "Source location dummy.", true);

    Constant *StrPtr = static_cast<Constant *>(Builder.CreateInBoundsGEP(
        ArrayType, StrVar, {Builder.getInt32(0), Builder.getInt32(0)}));

    Constant *LocInitStruct = ConstantStruct::get(
        IdentTy, {Builder.getInt32(0), Builder.getInt32(0), Builder.getInt32(0),
                  Builder.getInt32(0), StrPtr});

    // Initialize variables
    StrVar->setInitializer(InitStr);
    SourceLocDummy->setInitializer(LocInitStruct);
  }

  return SourceLocDummy;
}

bool ParallelLoopGeneratorKMP::is64BitArch() {
  return (LongType->getIntegerBitWidth() == 64);
}

OMPGeneralSchedulingType ParallelLoopGeneratorKMP::getSchedType(
    int ChunkSize, OMPGeneralSchedulingType Scheduling) const {
  if (ChunkSize == 0 && Scheduling == OMPGeneralSchedulingType::StaticChunked)
    return OMPGeneralSchedulingType::StaticNonChunked;

  return Scheduling;
}
