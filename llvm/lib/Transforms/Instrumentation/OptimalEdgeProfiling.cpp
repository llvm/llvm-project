//===- OptimalEdgeProfiling.cpp - Insert counters for opt. edge profiling -===//
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
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "insert-optimal-edge-profiling"
#include "ProfilingUtils.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "MaximumSpanningTree.h"
using namespace llvm;

STATISTIC(NumEdgesInserted, "The # of edges inserted.");

namespace {
  class OptimalEdgeProfiler : public ModulePass {
    bool runOnModule(Module &M);
  public:
    static char ID; // Pass identification, replacement for typeid
    OptimalEdgeProfiler() : ModulePass(ID) {
      initializeOptimalEdgeProfilerPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(ProfileEstimatorPassID);
      AU.addRequired<ProfileInfo>();
    }

    virtual const char *getPassName() const {
      return "Optimal Edge Profiler";
    }
  };
}

char OptimalEdgeProfiler::ID = 0;
INITIALIZE_PASS_BEGIN(OptimalEdgeProfiler, "insert-optimal-edge-profiling",
                "Insert optimal instrumentation for edge profiling",
                false, false)
INITIALIZE_PASS_DEPENDENCY(ProfileEstimatorPass)
INITIALIZE_AG_DEPENDENCY(ProfileInfo)
INITIALIZE_PASS_END(OptimalEdgeProfiler, "insert-optimal-edge-profiling",
                "Insert optimal instrumentation for edge profiling",
                false, false)

ModulePass *llvm::createOptimalEdgeProfilerPass() {
  return new OptimalEdgeProfiler();
}

inline static void printEdgeCounter(ProfileInfo::Edge e,
                                    BasicBlock* b,
                                    unsigned i) {
  DEBUG(dbgs() << "--Edge Counter for " << (e) << " in " \
               << ((b)?(b)->getName():"0") << " (# " << (i) << ")\n");
}

bool OptimalEdgeProfiler::runOnModule(Module &M) {
  Function *Main = M.getFunction("main");
  if (Main == 0) {
    errs() << "WARNING: cannot insert edge profiling into a module"
           << " with no main function!\n";
    return false;  // No main, no instrumentation!
  }

  // NumEdges counts all the edges that may be instrumented. Later on its
  // decided which edges to actually instrument, to achieve optimal profiling.
  // For the entry block a virtual edge (0,entry) is reserved, for each block
  // with no successors an edge (BB,0) is reserved. These edges are necessary
  // to calculate a truly optimal maximum spanning tree and thus an optimal
  // instrumentation.
  unsigned NumEdges = 0;

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    // Reserve space for (0,entry) edge.
    ++NumEdges;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      // Keep track of which blocks need to be instrumented.  We don't want to
      // instrument blocks that are added as the result of breaking critical
      // edges!
      if (BB->getTerminator()->getNumSuccessors() == 0) {
        // Reserve space for (BB,0) edge.
        ++NumEdges;
      } else {
        NumEdges += BB->getTerminator()->getNumSuccessors();
      }
    }
  }

  // In the profiling output a counter for each edge is reserved, but only few
  // are used. This is done to be able to read back in the profile without
  // calulating the maximum spanning tree again, instead each edge counter that
  // is not used is initialised with -1 to signal that this edge counter has to
  // be calculated from other edge counters on reading the profile info back
  // in.

  Type *Int32 = Type::getInt32Ty(M.getContext());
  ArrayType *ATy = ArrayType::get(Int32, NumEdges);
  GlobalVariable *Counters =
    new GlobalVariable(M, ATy, false, GlobalValue::InternalLinkage,
                       Constant::getNullValue(ATy), "OptEdgeProfCounters");
  NumEdgesInserted = 0;

  std::vector<Constant*> Initializer(NumEdges);
  Constant *Zero = ConstantInt::get(Int32, 0);
  Constant *Uncounted = ConstantInt::get(Int32, ProfileInfoLoader::Uncounted);

  // Instrument all of the edges not in MST...
  unsigned i = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    if (F->isDeclaration()) continue;
    DEBUG(dbgs() << "Working on " << F->getName() << "\n");

    // Calculate a Maximum Spanning Tree with the edge weights determined by
    // ProfileEstimator. ProfileEstimator also assign weights to the virtual
    // edges (0,entry) and (BB,0) (for blocks with no successors) and this
    // edges also participate in the maximum spanning tree calculation.
    // The third parameter of MaximumSpanningTree() has the effect that not the
    // actual MST is returned but the edges _not_ in the MST.

    ProfileInfo::EdgeWeights ECs =
      getAnalysis<ProfileInfo>(*F).getEdgeWeights(F);
    std::vector<ProfileInfo::EdgeWeight> EdgeVector(ECs.begin(), ECs.end());
    MaximumSpanningTree<BasicBlock> MST(EdgeVector);
    std::stable_sort(MST.begin(), MST.end());

    // Check if (0,entry) not in the MST. If not, instrument edge
    // (IncrementCounterInBlock()) and set the counter initially to zero, if
    // the edge is in the MST the counter is initialised to -1.

    BasicBlock *entry = &(F->getEntryBlock());
    ProfileInfo::Edge edge = ProfileInfo::getEdge(0, entry);
    if (!std::binary_search(MST.begin(), MST.end(), edge)) {
      printEdgeCounter(edge, entry, i);
      IncrementCounterInBlock(entry, i, Counters); ++NumEdgesInserted;
      Initializer[i++] = (Zero);
    } else{
      Initializer[i++] = (Uncounted);
    }

    // InsertedBlocks contains all blocks that were inserted for splitting an
    // edge, this blocks do not have to be instrumented.
    DenseSet<BasicBlock*> InsertedBlocks;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      // Check if block was not inserted and thus does not have to be
      // instrumented.
      if (InsertedBlocks.count(BB)) continue;

      // Okay, we have to add a counter of each outgoing edge not in MST. If
      // the outgoing edge is not critical don't split it, just insert the
      // counter in the source or destination of the edge. Also, if the block
      // has no successors, the virtual edge (BB,0) is processed.
      TerminatorInst *TI = BB->getTerminator();
      if (TI->getNumSuccessors() == 0) {
        ProfileInfo::Edge edge = ProfileInfo::getEdge(BB, 0);
        if (!std::binary_search(MST.begin(), MST.end(), edge)) {
          printEdgeCounter(edge, BB, i);
          IncrementCounterInBlock(BB, i, Counters); ++NumEdgesInserted;
          Initializer[i++] = (Zero);
        } else{
          Initializer[i++] = (Uncounted);
        }
      }
      for (unsigned s = 0, e = TI->getNumSuccessors(); s != e; ++s) {
        BasicBlock *Succ = TI->getSuccessor(s);
        ProfileInfo::Edge edge = ProfileInfo::getEdge(BB,Succ);
        if (!std::binary_search(MST.begin(), MST.end(), edge)) {

          // If the edge is critical, split it.
          bool wasInserted = SplitCriticalEdge(TI, s, this);
          Succ = TI->getSuccessor(s);
          if (wasInserted)
            InsertedBlocks.insert(Succ);

          // Okay, we are guaranteed that the edge is no longer critical.  If
          // we only have a single successor, insert the counter in this block,
          // otherwise insert it in the successor block.
          if (TI->getNumSuccessors() == 1) {
            // Insert counter at the start of the block
            printEdgeCounter(edge, BB, i);
            IncrementCounterInBlock(BB, i, Counters); ++NumEdgesInserted;
          } else {
            // Insert counter at the start of the block
            printEdgeCounter(edge, Succ, i);
            IncrementCounterInBlock(Succ, i, Counters); ++NumEdgesInserted;
          }
          Initializer[i++] = (Zero);
        } else {
          Initializer[i++] = (Uncounted);
        }
      }
    }
  }

  // Check if the number of edges counted at first was the number of edges we
  // considered for instrumentation.
  assert(i == NumEdges && "the number of edges in counting array is wrong");

  // Assign the now completely defined initialiser to the array.
  Constant *init = ConstantArray::get(ATy, Initializer);
  Counters->setInitializer(init);

  // Add the initialization call to main.
  InsertProfilingInitCall(Main, "llvm_start_opt_edge_profiling", Counters);
  return true;
}

