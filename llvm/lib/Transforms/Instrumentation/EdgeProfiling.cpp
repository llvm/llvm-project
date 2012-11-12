//===- EdgeProfiling.cpp - Insert counters for edge profiling -------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass instruments the specified program with counters for edge profiling.
// Edge profiling can give a reasonable approximation of the hot paths through a
// program, and is used for a wide variety of program transformations.
//
// Note that this implementation is very naive.  We insert a counter for *every*
// edge in the program, instead of using control flow information to prune the
// number of counters inserted.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "insert-edge-profiling"

#include "ProfilingUtils.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/Statistic.h"
#include <set>
using namespace llvm;

STATISTIC(NumEdgesInserted, "The # of edges inserted.");

namespace {
  class EdgeProfiler : public ModulePass {
    bool runOnModule(Module &M);
  public:
    static char ID; // Pass identification, replacement for typeid
    EdgeProfiler() : ModulePass(ID) {
      initializeEdgeProfilerPass(*PassRegistry::getPassRegistry());
    }

    virtual const char *getPassName() const {
      return "Edge Profiler";
    }
  };
}

char EdgeProfiler::ID = 0;
INITIALIZE_PASS(EdgeProfiler, "insert-edge-profiling",
                "Insert instrumentation for edge profiling", false, false)

ModulePass *llvm::createEdgeProfilerPass() { return new EdgeProfiler(); }

bool EdgeProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    errs() << "WARNING: cannot insert edge profiling into a module"
           << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  std::set<BasicBlock*> BlocksToInstrument;
  unsigned NumEdges = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    // Reserve space for (0,entry) edge.
    ++NumEdges;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      // Keep track of which blocks need to be instrumented.  We don't want to
      // instrument blocks that are added as the result of breaking critical
      // edges!
      BlocksToInstrument.insert(BB);
      NumEdges += BB->getTerminator()->getNumSuccessors();
    }
  }

  Type *ATy = ArrayType::get(Type::getInt32Ty(M.getContext()), NumEdges);
  GlobalVariable *Counters =
    new GlobalVariable(M, ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "EdgeProfCounters");
  NumEdgesInserted = NumEdges;

  // Instrument all of the edges...
  unsigned i = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    // Create counter for (0,entry) edge.
    IncrementCounterInBlock(&F->getEntryBlock(), i++, Counters);
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (BlocksToInstrument.count(BB)) {  // Don't instrument inserted blocks
        // Okay, we have to add a counter of each outgoing edge.  If the
        // outgoing edge is not critical don't split it, just insert the counter
        // in the source or destination of the edge.
        TerminatorInst *TI = BB->getTerminator();
        for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
          // If the edge is critical, split it.
          SplitCriticalEdge(TI, s, this);

          // Okay, we are guaranteed that the edge is no longer critical.  If we
          // only have a single successor, insert the counter in this block,
          // otherwise insert it in the successor block.
          if (TI->getNumSuccessors() == 1) {
            // Insert counter at the start of the block
            IncrementCounterInBlock(BB, i++, Counters, false);
          } else {
            // Insert counter at the start of the block
            IncrementCounterInBlock(TI->getSuccessor(s), i++, Counters);
          }
        }
      }
  }

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_edge_profiling", Counters);
  return true;
}

