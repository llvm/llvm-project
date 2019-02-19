//===------ PerfMonitor.cpp - Generate a run-time performance monitor. -======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/PerfMonitor.h"
#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "polly/ScopInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Intrinsics.h"
#include <sstream>

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

PerfMonitor::PerfMonitor(const Scop &S, Module *M)
    : M(M), Builder(M->getContext()), S(S) {
  if (Triple(M->getTargetTriple()).getArch() == llvm::Triple::x86_64)
    Supported = true;
  else
    Supported = false;
}

static void TryRegisterGlobal(Module *M, const char *Name,
                              Constant *InitialValue, Value **Location) {
  *Location = M->getGlobalVariable(Name);

  if (!*Location)
    *Location = new GlobalVariable(
        *M, InitialValue->getType(), true, GlobalValue::WeakAnyLinkage,
        InitialValue, Name, nullptr, GlobalVariable::InitialExecTLSModel);
}

// Generate a unique name that is usable as a LLVM name for a scop to name its
// performance counter.
static std::string GetScopUniqueVarname(const Scop &S) {
  std::stringstream Name;
  std::string EntryString, ExitString;
  std::tie(EntryString, ExitString) = S.getEntryExitStr();

  Name << "__polly_perf_in_" << std::string(S.getFunction().getName())
       << "_from__" << EntryString << "__to__" << ExitString;
  return Name.str();
}

void PerfMonitor::addScopCounter() {
  const std::string varname = GetScopUniqueVarname(S);
  TryRegisterGlobal(M, (varname + "_cycles").c_str(), Builder.getInt64(0),
                    &CyclesInCurrentScopPtr);

  TryRegisterGlobal(M, (varname + "_trip_count").c_str(), Builder.getInt64(0),
                    &TripCountForCurrentScopPtr);
}

void PerfMonitor::addGlobalVariables() {
  TryRegisterGlobal(M, "__polly_perf_cycles_total_start", Builder.getInt64(0),
                    &CyclesTotalStartPtr);

  TryRegisterGlobal(M, "__polly_perf_initialized", Builder.getInt1(0),
                    &AlreadyInitializedPtr);

  TryRegisterGlobal(M, "__polly_perf_cycles_in_scops", Builder.getInt64(0),
                    &CyclesInScopsPtr);

  TryRegisterGlobal(M, "__polly_perf_cycles_in_scop_start", Builder.getInt64(0),
                    &CyclesInScopStartPtr);
}

static const char *InitFunctionName = "__polly_perf_init";
static const char *FinalReportingFunctionName = "__polly_perf_final";

static BasicBlock *FinalStartBB = nullptr;
static ReturnInst *ReturnFromFinal = nullptr;

Function *PerfMonitor::insertFinalReporting() {
  // Create new function.
  GlobalValue::LinkageTypes Linkage = Function::WeakODRLinkage;
  FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), {}, false);
  Function *ExitFn =
      Function::Create(Ty, Linkage, FinalReportingFunctionName, M);
  FinalStartBB = BasicBlock::Create(M->getContext(), "start", ExitFn);
  Builder.SetInsertPoint(FinalStartBB);

  if (!Supported) {
    RuntimeDebugBuilder::createCPUPrinter(
        Builder, "Polly runtime information generation not supported\n");
    Builder.CreateRetVoid();
    return ExitFn;
  }

  // Measure current cycles and compute final timings.
  Function *RDTSCPFn = getRDTSCP();

  Value *CurrentCycles =
      Builder.CreateExtractValue(Builder.CreateCall(RDTSCPFn), {0});
  Value *CyclesStart = Builder.CreateLoad(CyclesTotalStartPtr, true);
  Value *CyclesTotal = Builder.CreateSub(CurrentCycles, CyclesStart);
  Value *CyclesInScops = Builder.CreateLoad(CyclesInScopsPtr, true);

  // Print the runtime information.
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Polly runtime information\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "-------------------------\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Total: ", CyclesTotal, "\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Scops: ", CyclesInScops,
                                        "\n");

  // Print the preamble for per-scop information.
  RuntimeDebugBuilder::createCPUPrinter(Builder, "\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "Per SCoP information\n");
  RuntimeDebugBuilder::createCPUPrinter(Builder, "--------------------\n");

  RuntimeDebugBuilder::createCPUPrinter(
      Builder, "scop function, "
               "entry block name, exit block name, total time, trip count\n");
  ReturnFromFinal = Builder.CreateRetVoid();
  return ExitFn;
}

void PerfMonitor::AppendScopReporting() {
  if (!Supported)
    return;

  assert(FinalStartBB && "Expected FinalStartBB to be initialized by "
                         "PerfMonitor::insertFinalReporting.");
  assert(ReturnFromFinal && "Expected ReturnFromFinal to be initialized by "
                            "PerfMonitor::insertFinalReporting.");

  Builder.SetInsertPoint(FinalStartBB);
  ReturnFromFinal->eraseFromParent();

  Value *CyclesInCurrentScop =
      Builder.CreateLoad(this->CyclesInCurrentScopPtr, true);

  Value *TripCountForCurrentScop =
      Builder.CreateLoad(this->TripCountForCurrentScopPtr, true);

  std::string EntryName, ExitName;
  std::tie(EntryName, ExitName) = S.getEntryExitStr();

  // print in CSV for easy parsing with other tools.
  RuntimeDebugBuilder::createCPUPrinter(
      Builder, S.getFunction().getName(), ", ", EntryName, ", ", ExitName, ", ",
      CyclesInCurrentScop, ", ", TripCountForCurrentScop, "\n");

  ReturnFromFinal = Builder.CreateRetVoid();
}

static Function *FinalReporting = nullptr;

void PerfMonitor::initialize() {
  addGlobalVariables();
  addScopCounter();

  // Ensure that we only add the final reporting function once.
  // On later invocations, append to the reporting function.
  if (!FinalReporting) {
    FinalReporting = insertFinalReporting();

    Function *InitFn = insertInitFunction(FinalReporting);
    addToGlobalConstructors(InitFn);
  }

  AppendScopReporting();
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
  // out if the initializer is run more than once.
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
    Value *CurrentCycles =
        Builder.CreateExtractValue(Builder.CreateCall(RDTSCPFn), {0});
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
  Value *CurrentCycles =
      Builder.CreateExtractValue(Builder.CreateCall(RDTSCPFn), {0});
  Builder.CreateStore(CurrentCycles, CyclesInScopStartPtr, true);
}

void PerfMonitor::insertRegionEnd(Instruction *InsertBefore) {
  if (!Supported)
    return;

  Builder.SetInsertPoint(InsertBefore);
  Function *RDTSCPFn = getRDTSCP();
  LoadInst *CyclesStart = Builder.CreateLoad(CyclesInScopStartPtr, true);
  Value *CurrentCycles =
      Builder.CreateExtractValue(Builder.CreateCall(RDTSCPFn), {0});
  Value *CyclesInScop = Builder.CreateSub(CurrentCycles, CyclesStart);
  Value *CyclesInScops = Builder.CreateLoad(CyclesInScopsPtr, true);
  CyclesInScops = Builder.CreateAdd(CyclesInScops, CyclesInScop);
  Builder.CreateStore(CyclesInScops, CyclesInScopsPtr, true);

  Value *CyclesInCurrentScop = Builder.CreateLoad(CyclesInCurrentScopPtr, true);
  CyclesInCurrentScop = Builder.CreateAdd(CyclesInCurrentScop, CyclesInScop);
  Builder.CreateStore(CyclesInCurrentScop, CyclesInCurrentScopPtr, true);

  Value *TripCountForCurrentScop =
      Builder.CreateLoad(TripCountForCurrentScopPtr, true);
  TripCountForCurrentScop =
      Builder.CreateAdd(TripCountForCurrentScop, Builder.getInt64(1));
  Builder.CreateStore(TripCountForCurrentScop, TripCountForCurrentScopPtr,
                      true);
}
