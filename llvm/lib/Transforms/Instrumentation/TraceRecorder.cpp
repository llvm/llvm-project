//===-- TraceRecorder.cpp - race detector -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder, a race detector.
//
// The tool is under development, for the details about previous versions see
// http://code.google.com/p/data-race-test
//
// The instrumentation phase is quite simple:
//   - Insert calls to run-time library before every memory access.
//      - Optimizations may apply to avoid instrumenting some of the accesses.
//   - Insert calls at function entry/exit.
// The rest is handled by the run-time library.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/TraceRecorder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "trec"

static cl::opt<bool>
    ClInstrumentFuncEntryExit("trec-instrument-func-entry-exit", cl::init(true),
                              cl::desc("Instrument function entry and exit"),
                              cl::Hidden);
static cl::opt<bool>
    ClTrecAddDebugInfo("trec-add-debug-info", cl::init(true),
                       cl::desc("Instrument to record debug information"),
                       cl::Hidden);
const char kTrecModuleCtorName[] = "trec.module_ctor";
const char kTrecInitName[] = "__trec_init";

namespace {
std::map<std::string, Value *> TraceRecorderModuleVarNames;
static enum Mode { Eagle, Verification, Unknown } mode;
/// TraceRecorder: instrument the code in module to record traces.
///
/// Instantiating TraceRecorder inserts the trec runtime library API
/// function declarations into the module if they don't exist already.
/// Instantiating ensures the __trec_init function is in the list of global
/// constructors for the module.
struct TraceRecorder {
  TraceRecorder(std::map<std::string, Value *> &VN) : VarNames(VN) {
    // Sanity check options and warn user.
    if (getenv("TREC_COMPILE_MODE") == nullptr)
      mode = Mode::Unknown;
    else if (strcmp(getenv("TREC_COMPILE_MODE"), "eagle") == 0)
      mode = Mode::Eagle;
    else if (strcmp(getenv("TREC_COMPILE_MODE"), "verification") == 0)
      mode = Mode::Verification;
    else
      mode = Mode::Unknown;
    if (mode == Mode::Unknown) {
      printf("Error: Unknown TraceRecorder mode: ENV variable "
             "`TREC_COMPILE_MODE` has "
             "not been set!\n");
      exit(-1);
    }
  }

  bool sanitizeFunction(Function &F, const TargetLibraryInfo &TLI);
  std::map<std::string, Value *> &VarNames;

private:
  void initialize(Module &M);
  bool instrumentFunctionCall(Instruction *I);
  inline std::string concatFileName(std::string dir, std::string file) {
    return dir + "/" + file;
  }

  FunctionCallee TrecFuncEntry;
  FunctionCallee TrecFuncExit;
  FunctionCallee TrecInstDebugInfo;
  FunctionCallee TrecBBLEntry;
};

void insertModuleCtor(Module &M) {
  getOrCreateSanitizerCtorAndInitFunctions(
      M, kTrecModuleCtorName, kTrecInitName, /*InitArgTypes=*/{},
      /*InitArgs=*/{},
      // This callback is invoked when the functions are created the first
      // time. Hook them into the global ctors list in that case:
      [&](Function *Ctor, FunctionCallee) { appendToGlobalCtors(M, Ctor, 0); });
}
} // namespace

PreservedAnalyses TraceRecorderPass::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  TraceRecorder TRec(TraceRecorderModuleVarNames);
  if (TRec.sanitizeFunction(F, FAM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

PreservedAnalyses ModuleTraceRecorderPass::run(Module &M,
                                               ModuleAnalysisManager &MAM) {
  insertModuleCtor(M);
  TraceRecorderModuleVarNames.clear();
  return PreservedAnalyses::none();
}

void TraceRecorder::initialize(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  IRBuilder<> IRB(M.getContext());
  AttributeList Attr;
  Attr = Attr.addFnAttribute(M.getContext(), Attribute::NoUnwind);
  // Initialize the callbacks.
  TrecFuncEntry =
      M.getOrInsertFunction("__trec_func_entry", Attr, IRB.getVoidTy());

  TrecFuncExit =
      M.getOrInsertFunction("__trec_func_exit", Attr, IRB.getVoidTy());

  TrecInstDebugInfo = M.getOrInsertFunction(
      "__trec_inst_debug_info", Attr, IRB.getVoidTy(), IRB.getInt64Ty(),
      IRB.getInt32Ty(), IRB.getInt16Ty(), IRB.getInt64Ty(), IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy());
  TrecBBLEntry =
      M.getOrInsertFunction("__trec_bbl_entry", Attr, IRB.getVoidTy());
}

bool TraceRecorder::sanitizeFunction(Function &F,
                                     const TargetLibraryInfo &TLI) {
  // This is required to prevent instrumenting call to __trec_init from
  // within the module constructor.
  if (F.getName() == kTrecModuleCtorName)
    return false;
  // If we cannot find the source file, then this function must not be written
  // by user. Do not instrument it.
  if (F.getSubprogram() == nullptr || F.getSubprogram()->getFile() == nullptr)
    return false;

  // Some cases that we do not instrument

  // Naked functions can not have prologue/epilogue
  // (__trec_func_entry/__trec_func_exit) generated, so don't
  // instrument them at all.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  initialize(*F.getParent());
  SmallVector<Instruction *> FuncCalls;
  bool Res = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // printf("f_name:%s,%d\n", F.getName().str().c_str(), F.getType());
  // BasicBlock *block = BasicBlock::Create(F.getParent()->getContext());
  // block->insertInto(&F);


  SmallVector<BasicBlock *, 8> NewBlocks;
  ValueToValueMapTy VMap;
  std::map<BasicBlock *, BasicBlock *> bbMap;
  for (auto &BB : F) {
    NewBlocks.push_back(&BB);
  }
  for (auto &BB : NewBlocks) {
    BasicBlock *block = CloneBasicBlock(BB, VMap, "", &F);
    bbMap[BB] = block;
  }
  for (auto &entry : bbMap) {
    BasicBlock *bb = entry.first;
    BasicBlock *copy = entry.second;
    for (auto &Inst : *copy) {
      if (auto *jumpInst = dyn_cast<BranchInst>(&Inst)) {
        for (unsigned i = 0; i < jumpInst->getNumSuccessors(); i++) {
          BasicBlock *succ = jumpInst->getSuccessor(i);
          if (bbMap.count(succ)) {
            BasicBlock *target = bbMap[succ];
            jumpInst->setSuccessor(i, target);
          }
        }
      } else if (auto *switchInst = dyn_cast<SwitchInst>(&Inst)) {
        for (unsigned i = 0; i < switchInst->getNumSuccessors(); i++) {
          BasicBlock *succ = switchInst->getSuccessor(i);
          if (bbMap.count(succ)) {
            BasicBlock *target = bbMap[succ];
            switchInst->setSuccessor(i, target);
          }
        }
      }
    }
  }

  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (isa<CallInst>(Inst) || isa<InvokeInst>(Inst) ||
          isa<CallBrInst>(Inst)) {
        FuncCalls.push_back(&Inst);
      }
    }
  }

  for (auto Inst : FuncCalls) {
    instrumentFunctionCall(Inst);
  }
  // Instrument function entry/exit points if there were instrumented
  // accesses.
  if (ClInstrumentFuncEntryExit) {
    uint64_t fid = (uint64_t)(F.getGUID());
    IRBuilder<> IRB(F.getEntryBlock().getFirstNonPHI());
    StringRef FuncName = F.getName();

    if (ClTrecAddDebugInfo && F.getSubprogram() &&
        F.getSubprogram()->getFile()) {
      std::string CurrentFileName =
          concatFileName(F.getSubprogram()->getFile()->getDirectory().str(),
                         F.getSubprogram()->getFile()->getFilename().str());
      FuncName = F.getSubprogram()->getName();
      if (FuncName == "main") {
        Value *func_name, *file_name;
        if (VarNames.count(FuncName.str())) {
          func_name = VarNames.find(FuncName.str())->second;
        } else {
          func_name = IRB.CreateGlobalStringPtr(FuncName);
          VarNames.insert(std::make_pair(FuncName.str(), func_name));
        }
        if (VarNames.count(CurrentFileName)) {
          file_name = VarNames.find(CurrentFileName)->second;
        } else {
          file_name = IRB.CreateGlobalStringPtr(CurrentFileName);
          VarNames.insert(std::make_pair(CurrentFileName, file_name));
        }
        IRB.CreateCall(
            TrecInstDebugInfo,
            {IRB.getInt64(fid), IRB.getInt32(F.getSubprogram()->getLine()),
             IRB.getInt16(0), IRB.getInt64(0), func_name, file_name});
      }
    }

    Value *name_ptr;
    if (VarNames.count(FuncName.str())) {
      name_ptr = VarNames.find(FuncName.str())->second;
    } else {
      name_ptr = IRB.CreateGlobalStringPtr(FuncName.str());
      VarNames.insert(std::make_pair(FuncName.str(), name_ptr));
    }

    IRB.CreateCall(TrecFuncEntry, {name_ptr});
    // IRB.CreateCall(TrecBBLEntry);

    EscapeEnumerator EE(F);
    while (IRBuilder<> *AtExit = EE.Next()) {
      AtExit->CreateCall(TrecFuncExit, {});
    }
    Res |= true;
  }

  return Res;
}

bool TraceRecorder::instrumentFunctionCall(Instruction *I) {
  IRBuilder<> IRB(I);
  CallBase *CI = dyn_cast<CallBase>(I);

  if (CI->getCalledFunction() &&
      (CI->getCalledFunction()->hasFnAttribute(Attribute::Naked) ||
       CI->getCalledFunction()->getName().startswith("llvm.dbg"))) {
    return false;
  }
  Function *F = CI->getCalledFunction();
  uint64_t fid = (uint64_t)(F->getGUID());
  if (ClTrecAddDebugInfo) {
    std::string CurrentFileName = "";
    if (CI->getDebugLoc().get())
      CurrentFileName =
          concatFileName((CI->getDebugLoc()->getScope()->getDirectory().str()),
                         (CI->getDebugLoc()->getScope()->getFilename().str()));
    StringRef FuncName = "";
    if (F)
      FuncName =
          (F->getSubprogram()) ? F->getSubprogram()->getName() : F->getName();

    if (FuncName == "pthread_create") {
      Function *called = dyn_cast<Function>(CI->getArgOperand(2));
      FuncName = called ? called->getSubprogram()->getName()
                        : CI->getArgOperand(2)->getName();
    }
    Value *func_name, *file_name;
    if (VarNames.count(FuncName.str())) {
      func_name = VarNames.find(FuncName.str())->second;
    } else {
      func_name = IRB.CreateGlobalStringPtr(FuncName);
      VarNames.insert(std::make_pair(FuncName.str(), func_name));
    }
    if (VarNames.count(CurrentFileName)) {
      file_name = VarNames.find(CurrentFileName)->second;
    } else {
      file_name = IRB.CreateGlobalStringPtr(CurrentFileName);
      VarNames.insert(std::make_pair(CurrentFileName, file_name));
    }
    IRB.CreateCall(
        TrecInstDebugInfo,
        {IRB.getInt64(fid),
         IRB.getInt32(I->getDebugLoc().get() ? I->getDebugLoc().getLine() : 0),
         IRB.getInt16(I->getDebugLoc().get() ? I->getDebugLoc().getCol() : 0),
         IRB.getInt64(0), func_name, file_name});
  }
  return true;
}
