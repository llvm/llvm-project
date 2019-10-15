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
  Type *KMPCMicroTy = M->getTypeByName("kmpc_micro");

  if (!KMPCMicroTy) {
    // void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid, ...)
    Type *MicroParams[] = {Builder.getInt32Ty()->getPointerTo(),
                           Builder.getInt32Ty()->getPointerTo()};

    KMPCMicroTy = FunctionType::get(Builder.getVoidTy(), MicroParams, true);
  }

  // If F is not available, declare it.
  if (!F) {
    StructType *IdentTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {IdentTy->getPointerTo(), Builder.getInt32Ty(),
                      KMPCMicroTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Task = Builder.CreatePointerBitCastOrAddrSpaceCast(
      SubFn, KMPCMicroTy->getPointerTo());

  Value *Args[] = {SourceLocationInfo,
                   Builder.getInt32(4) /* Number of arguments (w/o Task) */,
                   Task,
                   LB,
                   UB,
                   Stride,
                   SubFnParam};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorKMP::deployParallelExecution(Value *SubFn,
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
  std::vector<Type *> Arguments = {Builder.getInt32Ty()->getPointerTo(),
                                   Builder.getInt32Ty()->getPointerTo(),
                                   LongType,
                                   LongType,
                                   LongType,
                                   Builder.getInt8PtrTy()};

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
// HeaderBB will hold allocations, loading of variables and kmp-init calls.
// CheckNextBB will check for more work (dynamic) or will be "empty" (static).
// If there is more work to do: go to PreHeaderBB, otherwise go to ExitBB.
// PreHeaderBB loads the new boundaries (& will lead to the loop body later on).
// Just like CheckNextBB: PreHeaderBB is empty in the static scheduling case.
// ExitBB marks the end of the parallel execution.
// The possibly empty BasicBlocks will automatically be removed.
std::tuple<Value *, Function *>
ParallelLoopGeneratorKMP::createSubFn(Value *StrideNotUsed,
                                      AllocaInst *StructData,
                                      SetVector<Value *> Data, ValueMapT &Map) {
  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Store the previous basic block.
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  BasicBlock *CheckNextBB =
      BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  BasicBlock *PreHeaderBB =
      BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB);

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

  Value *UserContext = Builder.CreateBitCast(Shared, StructData->getType(),
                                             "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);

  const int Alignment = (is64BitArch()) ? 8 : 4;
  Value *ID =
      Builder.CreateAlignedLoad(IDPtr, Alignment, "polly.par.global_tid");

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

  switch (PollyScheduling) {
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
      LB = Builder.CreateAlignedLoad(LBPtr, Alignment, "polly.indvar.LB");
      UB = Builder.CreateAlignedLoad(UBPtr, Alignment, "polly.indvar.UB");
    }
    break;
  case OMPGeneralSchedulingType::StaticChunked:
  case OMPGeneralSchedulingType::StaticNonChunked:
    // "STATIC" scheduling types are handled below
    {
      createCallStaticInit(ID, IsLastPtr, LBPtr, UBPtr, StridePtr, ChunkSize);

      LB = Builder.CreateAlignedLoad(LBPtr, Alignment, "polly.indvar.LB");
      UB = Builder.CreateAlignedLoad(UBPtr, Alignment, "polly.indvar.UB");

      Value *AdjUBOutOfBounds =
          Builder.CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, UB, AdjustedUB,
                             "polly.adjustedUBOutOfBounds");

      UB = Builder.CreateSelect(AdjUBOutOfBounds, UB, AdjustedUB);
      Builder.CreateAlignedStore(UB, UBPtr, Alignment);

      Value *HasIteration = Builder.CreateICmp(
          llvm::CmpInst::Predicate::ICMP_SLE, LB, UB, "polly.hasIteration");
      Builder.CreateCondBr(HasIteration, PreHeaderBB, ExitBB);

      Builder.SetInsertPoint(CheckNextBB);
      Builder.CreateBr(ExitBB);

      Builder.SetInsertPoint(PreHeaderBB);
    }
    break;
  }

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  BasicBlock *AfterBB;
  Value *IV = createLoop(LB, UB, Stride, Builder, LI, DT, AfterBB,
                         ICmpInst::ICMP_SLE, nullptr, true,
                         /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  // Static (i.e. non-dynamic) scheduling types, are terminated with a fini-call
  if (PollyScheduling == OMPGeneralSchedulingType::StaticChunked) {
    createCallStaticFini(ID);
  }
  Builder.CreateRetVoid();
  Builder.SetInsertPoint(&*LoopBody);

  return std::make_tuple(IV, SubFn);
}

Value *ParallelLoopGeneratorKMP::createCallGlobalThreadNum() {
  const std::string Name = "__kmpc_global_thread_num";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *IdentTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {IdentTy->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return Builder.CreateCall(F, {SourceLocationInfo});
}

void ParallelLoopGeneratorKMP::createCallPushNumThreads(Value *GlobalThreadID,
                                                        Value *NumThreads) {
  const std::string Name = "__kmpc_push_num_threads";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    StructType *IdentTy = M->getTypeByName("struct.ident_t");

    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {IdentTy->getPointerTo(), Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID, NumThreads};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorKMP::createCallStaticInit(Value *GlobalThreadID,
                                                    Value *IsLastPtr,
                                                    Value *LBPtr, Value *UBPtr,
                                                    Value *StridePtr,
                                                    Value *ChunkSize) {
  const std::string Name =
      is64BitArch() ? "__kmpc_for_static_init_8" : "__kmpc_for_static_init_4";
  Function *F = M->getFunction(Name);
  StructType *IdentTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {IdentTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
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

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorKMP::createCallStaticFini(Value *GlobalThreadID) {
  const std::string Name = "__kmpc_for_static_fini";
  Function *F = M->getFunction(Name);
  StructType *IdentTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {IdentTy->getPointerTo(), Builder.getInt32Ty()};
    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorKMP::createCallDispatchInit(Value *GlobalThreadID,
                                                      Value *LB, Value *UB,
                                                      Value *Inc,
                                                      Value *ChunkSize) {
  const std::string Name =
      is64BitArch() ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_4";
  Function *F = M->getFunction(Name);
  StructType *IdentTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {IdentTy->getPointerTo(),
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

  Builder.CreateCall(F, Args);
}

Value *ParallelLoopGeneratorKMP::createCallDispatchNext(Value *GlobalThreadID,
                                                        Value *IsLastPtr,
                                                        Value *LBPtr,
                                                        Value *UBPtr,
                                                        Value *StridePtr) {
  const std::string Name =
      is64BitArch() ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_4";
  Function *F = M->getFunction(Name);
  StructType *IdentTy = M->getTypeByName("struct.ident_t");

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {IdentTy->getPointerTo(),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo(),
                      LongType->getPointerTo()};

    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SourceLocationInfo, GlobalThreadID, IsLastPtr, LBPtr, UBPtr,
                   StridePtr};

  return Builder.CreateCall(F, Args);
}

// TODO: This function currently creates a source location dummy. It might be
// necessary to (actually) provide information, in the future.
GlobalVariable *ParallelLoopGeneratorKMP::createSourceLocation() {
  const std::string LocName = ".loc.dummy";
  GlobalVariable *SourceLocDummy = M->getGlobalVariable(LocName);

  if (SourceLocDummy == nullptr) {
    const std::string StructName = "struct.ident_t";
    StructType *IdentTy = M->getTypeByName(StructName);

    // If the ident_t StructType is not available, declare it.
    // in LLVM-IR: ident_t = type { i32, i32, i32, i32, i8* }
    if (!IdentTy) {
      Type *LocMembers[] = {Builder.getInt32Ty(), Builder.getInt32Ty(),
                            Builder.getInt32Ty(), Builder.getInt32Ty(),
                            Builder.getInt8PtrTy()};

      IdentTy =
          StructType::create(M->getContext(), LocMembers, StructName, false);
    }

    const auto ArrayType =
        llvm::ArrayType::get(Builder.getInt8Ty(), /* Length */ 23);

    // Global Variable Definitions
    GlobalVariable *StrVar = new GlobalVariable(
        *M, ArrayType, true, GlobalValue::PrivateLinkage, 0, ".str.ident");
    StrVar->setAlignment(llvm::Align::None());

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
