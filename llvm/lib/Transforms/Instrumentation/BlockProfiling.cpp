//===- BlockProfiling.cpp - Insert counters for block profiling -----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments the specified program with counters for basic block or
// function profiling.  This is the most basic form of profiling, which can tell
// which blocks are hot, but cannot reliably detect hot paths through the CFG.
// Block profiling counts the number of times each basic block executes, and
// function profiling counts the number of times each function is called.
//
// Note that this implementation is very naive.  Control equivalent regions of
// the CFG should not require duplicate counters, but we do put duplicate
// counters in.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Transforms/Instrumentation.h"
#include "RSProfiling.h"
#include "ProfilingUtils.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN FunctionProfiler : public RSProfilers_std {
  public:
    static char ID;
    bool runOnModule(Module &M);
  };
}

char FunctionProfiler::ID = 0;

static RegisterPass<FunctionProfiler>
X("insert-function-profiling",
  "Insert instrumentation for function profiling");
static RegisterAnalysisGroup<RSProfilers> XG(X);

ModulePass *llvm::createFunctionProfilerPass() {
  return new FunctionProfiler();
}

bool FunctionProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    cerr << "WARNING: cannot insert function profiling into a module"
         << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  unsigned NumFunctions = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration())
      ++NumFunctions;

  const Type *ATy = ArrayType::get(Type::Int32Ty, NumFunctions);
  GlobalVariable *Counters =
    new GlobalVariable(ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "FuncProfCounters", &M);

  // Instrument all of the functions...
  unsigned i = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration())
      // Insert counter at the start of the function
      IncrementCounterInBlock(&I->getEntryBlock(), i++, Counters);

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_func_profiling", Counters);
  return true;
}


namespace {
  class BlockProfiler : public RSProfilers_std {
    bool runOnModule(Module &M);
  public:
    static char ID;
  };
}

char BlockProfiler::ID = 0;
static RegisterPass<BlockProfiler>
Y("insert-block-profiling", "Insert instrumentation for block profiling");
static RegisterAnalysisGroup<RSProfilers> YG(Y);

ModulePass *llvm::createBlockProfilerPass() { return new BlockProfiler(); }

bool BlockProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    cerr << "WARNING: cannot insert block profiling into a module"
         << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  unsigned NumBlocks = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    NumBlocks += I->size();

  const Type *ATy = ArrayType::get(Type::Int32Ty, NumBlocks);
  GlobalVariable *Counters =
    new GlobalVariable(ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "BlockProfCounters", &M);

  // Instrument all of the blocks...
  unsigned i = 0;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    for (Function::iterator BB = I->begin(), E = I->end(); BB != E; ++BB)
      // Insert counter at the start of the block
      IncrementCounterInBlock(BB, i++, Counters);

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_block_profiling", Counters);
  return true;
}

