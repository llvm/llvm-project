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

#include "ProfilingUtils.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include <set>
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN EdgeProfiler : public ModulePass {
    bool runOnModule(Module &M);
  public:
    static char ID; // Pass identification, replacement for typeid
    EdgeProfiler() : ModulePass(&ID) {}
  };
}

char EdgeProfiler::ID = 0;
static RegisterPass<EdgeProfiler>
X("insert-edge-profiling", "Insert instrumentation for edge profiling");

ModulePass *llvm::createEdgeProfilerPass() { return new EdgeProfiler(); }

bool EdgeProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    cerr << "WARNING: cannot insert edge profiling into a module"
         << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  std::set<BasicBlock*> BlocksToInstrument;
  unsigned NumEdges = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      // Keep track of which blocks need to be instrumented.  We don't want to
      // instrument blocks that are added as the result of breaking critical
      // edges!
      BlocksToInstrument.insert(BB);
      NumEdges += BB->getTerminator()->getNumSuccessors();
    }

  const Type *ATy = ArrayType::get(Type::Int32Ty, NumEdges);
  GlobalVariable *Counters =
    new GlobalVariable(ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "EdgeProfCounters", &M);

  // Instrument all of the edges...
  unsigned i = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
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
            IncrementCounterInBlock(BB, i++, Counters);
          } else {
            // Insert counter at the start of the block
            IncrementCounterInBlock(TI->getSuccessor(s), i++, Counters);
          }
        }
      }

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_edge_profiling", Counters);
  return true;
}

