//===------ PerfMonitor.cpp - Generate a run-time performance monitor. -======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/PerfMonitor.h"
#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;
using namespace polly;

Function *PerfMonitor::getAtExit() {
  const char *Name = "atexit";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(),
                                         {Builder.getInt8PtrTy()}, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

void PerfMonitor::addToGlobalConstructors(Function *Fn) {
  const char *Name = "llvm.global_ctors";
  GlobalVariable *GV = M->getGlobalVariable(Name);
  std::vector<Constant *> V;

  if (GV) {
    Constant *Array = GV->getInitializer();
    for (Value *X : Array->operand_values())
      V.push_back(cast<Constant>(X));
    GV->eraseFromParent();
  }

  StructType *ST = StructType::get(Builder.getInt32Ty(), Fn->getType(),
                                   Builder.getInt8PtrTy());

  V.push_back(
      ConstantStruct::get(ST, Builder.getInt32(10), Fn,
                          ConstantPointerNull::get(Builder.getInt8PtrTy())));
  ArrayType *Ty = ArrayType::get(ST, V.size());

  GV = new GlobalVariable(*M, Ty, true, GlobalValue::AppendingLinkage,
                          ConstantArray::get(Ty, V), Name, nullptr,
                          GlobalVariable::NotThreadLocal);
}

Function *PerfMonitor::getRDTSCP() {
  return Intrinsic::getDeclaration(M, Intrinsic::x86_rdtscp);
}

PerfMonitor::PerfMonitor(Module *M) : M(M), Builder(M->getContext()) {
  if (Triple(M->getTargetTriple()).getArch() == llvm::Triple::x86_64)
    Supported = true;
  else
    Supported = false;
}

void PerfMonitor::addGlobalVariables() {
  auto TryRegisterGlobal = [=](const char *Name, Constant *InitialValue,
                               Value **Location) {
    *Location = M->getGlobalVariable(Name);

    if (!*Location)
      *Location = new GlobalVariable(
          *M, InitialValue->getType(), true, GlobalValue::WeakAnyLinkage,
          InitialValue, Name, nullptr, GlobalVariable::InitialExecTLSModel);
  };

  TryRegisterGlobal("__polly_perf_cycles_total_start", Builder.getInt64(0),
                    &CyclesTotalStartPtr);

  TryRegisterGlobal("__polly_perf_initialized", Builder.getInt1(0),
                    &AlreadyInitializedPtr);

  TryRegisterGlobal("__polly_perf_cycles_in_scops", Builder.getInt64(0),
                    &CyclesInScopsPtr);

  TryRegisterGlobal("__polly_perf_cycles_in_scop_start", Builder.getInt64(0),
                    &CyclesInScopStartPtr);

  TryRegisterGlobal("__polly_perf_write_loation", Builder.getInt32(0),
                    &RDTSCPWriteLocation);
}

static const char *InitFunctionName = "__polly_perf_init";
static const char *FinalReportingFunctionName = "__polly_perf_final";

Function *PerfMonitor::insertFinalReporting() {
  // Create new function.
  GlobalValue::LinkageTypes Linkage = Function::WeakODRLinkage;
  FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), {}, false);
  Function *ExitFn =
      Function::Create(Ty, Linkage, FinalReportingFunctionName, M);
  BasicBlock *Start = BasicBlock::Create(M->getContext(), "start", ExitFn);
  Builder.SetInsertPoint(Start);

  if (!Supported) {
    RuntimeDebugBuilder::createCPUPrinter(
        Builder, "Polly runtime information generation not supported\n");
    Builder.CreateRetVoid();
    return ExitFn;
  }

  // Measure current cycles and compute final timings.
  Function *RDTSCPFn = getRDTSCP();
  Value *CurrentCycles = Builder.CreateCall(
      RDTSCPFn,
      Builder.CreatePointerCast(RDTSCPWriteLocation, Builder.getInt8PtrTy()));
  Value *CyclesStart = Builder.CreateLoad(CyclesTotalStartPtr, true);
  Value *CyclesTotal = Builder.CreateSub(CurrentCycles, CyclesStart);
  Value *CyclesInScops = Builder.CreateLoad(CyclesInScopsPtr, true);

  // Print the runtime information.
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Polly runtime information\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "-------------------------\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Total: ", CyclesTotal, "\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Scops: ", CyclesInScops,
                                        "\n");

  // Finalize function.
  Builder.CreateRetVoid();
  return ExitFn;
}

void PerfMonitor::initialize() {
  addGlobalVariables();

  Function *F = M->getFunction(InitFunctionName);
  if (F)
    return;

  // initialize
  Function *FinalReporting = insertFinalReporting();
  Function *InitFn = insertInitFunction(FinalReporting);
  addToGlobalConstructors(InitFn);
}

Function *PerfMonitor::insertInitFunction(Function *FinalReporting) {
  // Insert function definition and BBs.
  GlobalValue::LinkageTypes Linkage = Function::WeakODRLinkage;
  FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), {}, false);
  Function *InitFn = Function::Create(Ty, Linkage, InitFunctionName, M);
  BasicBlock *Start = BasicBlock::Create(M->getContext(), "start", InitFn);
  BasicBlock *EarlyReturn =
      BasicBlock::Create(M->getContext(), "earlyreturn", InitFn);
  BasicBlock *InitBB = BasicBlock::Create(M->getContext(), "initbb", InitFn);

  Builder.SetInsertPoint(Start);

  // Check if this function was already run. If yes, return.
  //
  // In case profiling has been enabled in multiple translation units, the
  // initializer function will be added to the global constructors list of
  // each translation unit. When merging translation units, the global
  // constructor lists are just appended, such that the initializer will appear
  // multiple times. To avoid initializations being run multiple times (and
  // especially to avoid that atExitFn is called more than once), we bail
  // out if the intializer is run more than once.
  Value *HasRunBefore = Builder.CreateLoad(AlreadyInitializedPtr);
  Builder.CreateCondBr(HasRunBefore, EarlyReturn, InitBB);
  Builder.SetInsertPoint(EarlyReturn);
  Builder.CreateRetVoid();

  // Keep track that this function has been run once.
  Builder.SetInsertPoint(InitBB);
  Value *True = Builder.getInt1(true);
  Builder.CreateStore(True, AlreadyInitializedPtr);

  // Register the final reporting function with atexit().
  Value *FinalReportingPtr =
      Builder.CreatePointerCast(FinalReporting, Builder.getInt8PtrTy());
  Function *AtExitFn = getAtExit();
  Builder.CreateCall(AtExitFn, {FinalReportingPtr});

  if (Supported) {
    // Read the currently cycle counter and store the result for later.
    Function *RDTSCPFn = getRDTSCP();
    Value *CurrentCycles = Builder.CreateCall(
        RDTSCPFn,
        Builder.CreatePointerCast(RDTSCPWriteLocation, Builder.getInt8PtrTy()));
    Builder.CreateStore(CurrentCycles, CyclesTotalStartPtr, true);
  }
  Builder.CreateRetVoid();

  return InitFn;
}

void PerfMonitor::insertRegionStart(Instruction *InsertBefore) {
  if (!Supported)
    return;

  Builder.SetInsertPoint(InsertBefore);
  Function *RDTSCPFn = getRDTSCP();
  Value *CurrentCycles = Builder.CreateCall(
      RDTSCPFn,
      Builder.CreatePointerCast(RDTSCPWriteLocation, Builder.getInt8PtrTy()));
  Builder.CreateStore(CurrentCycles, CyclesInScopStartPtr, true);
}

void PerfMonitor::insertRegionEnd(Instruction *InsertBefore) {
  if (!Supported)
    return;

  Builder.SetInsertPoint(InsertBefore);
  Function *RDTSCPFn = getRDTSCP();
  LoadInst *CyclesStart = Builder.CreateLoad(CyclesInScopStartPtr, true);
  Value *CurrentCycles = Builder.CreateCall(
      RDTSCPFn,
      Builder.CreatePointerCast(RDTSCPWriteLocation, Builder.getInt8PtrTy()));
  Value *CyclesInScop = Builder.CreateSub(CurrentCycles, CyclesStart);
  Value *CyclesInScops = Builder.CreateLoad(CyclesInScopsPtr, true);
  CyclesInScops = Builder.CreateAdd(CyclesInScops, CyclesInScop);
  Builder.CreateStore(CyclesInScops, CyclesInScopsPtr, true);
}
