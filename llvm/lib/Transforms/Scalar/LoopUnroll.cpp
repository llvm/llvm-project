//===-- LoopUnroll.cpp - Loop unroller pass -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a simple loop unroller.  It works best when loops have
// been canonicalized by the -indvars pass, allowing it to determine the trip
// counts of loops easily.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-unroll"
#include "llvm/IntrinsicInst.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <climits>

using namespace llvm;

static cl::opt<unsigned>
UnrollThreshold("unroll-threshold", cl::init(100), cl::Hidden,
  cl::desc("The cut-off point for automatic loop unrolling"));

static cl::opt<unsigned>
UnrollCount("unroll-count", cl::init(0), cl::Hidden,
  cl::desc("Use this unroll count for all loops, for testing purposes"));

static cl::opt<bool>
UnrollAllowPartial("unroll-allow-partial", cl::init(false), cl::Hidden,
  cl::desc("Allows loops to be partially unrolled until "
           "-unroll-threshold loop size is reached."));

namespace {
  class VISIBILITY_HIDDEN LoopUnroll : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopUnroll() : LoopPass(&ID) {}

    /// A magic value for use with the Threshold parameter to indicate
    /// that the loop unroll should be performed regardless of how much
    /// code expansion would result.
    static const unsigned NoThreshold = UINT_MAX;

    bool runOnLoop(Loop *L, LPPassManager &LPM);

    /// This transformation requires natural loop information & requires that
    /// loop preheaders be inserted into the CFG...
    ///
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addRequired<LoopInfo>();
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<LoopInfo>();
      // FIXME: Loop unroll requires LCSSA. And LCSSA requires dom info.
      // If loop unroll does not preserve dom info then LCSSA pass on next
      // loop will receive invalid dom info.
      // For now, recreate dom info, if loop is unrolled.
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }
  };
}

char LoopUnroll::ID = 0;
static RegisterPass<LoopUnroll> X("loop-unroll", "Unroll loops");

Pass *llvm::createLoopUnrollPass() { return new LoopUnroll(); }

/// ApproximateLoopSize - Approximate the size of the loop.
static unsigned ApproximateLoopSize(const Loop *L) {
  unsigned Size = 0;
  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    Instruction *Term = BB->getTerminator();
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (isa<PHINode>(I) && BB == L->getHeader()) {
        // Ignore PHI nodes in the header.
      } else if (I->hasOneUse() && I->use_back() == Term) {
        // Ignore instructions only used by the loop terminator.
      } else if (isa<DbgInfoIntrinsic>(I)) {
        // Ignore debug instructions
      } else if (isa<CallInst>(I)) {
        // Estimate size overhead introduced by call instructions which
        // is higher than other instructions. Here 3 and 10 are magic
        // numbers that help one isolated test case from PR2067 without
        // negatively impacting measured benchmarks.
        if (isa<IntrinsicInst>(I))
          Size = Size + 3;
        else
          Size = Size + 10;
      } else {
        ++Size;
      }

      // TODO: Ignore expressions derived from PHI and constants if inval of phi
      // is a constant, or if operation is associative.  This will get induction
      // variables.
    }
  }

  return Size;
}

bool LoopUnroll::runOnLoop(Loop *L, LPPassManager &LPM) {
  assert(L->isLCSSAForm());
  LoopInfo *LI = &getAnalysis<LoopInfo>();

  BasicBlock *Header = L->getHeader();
  DOUT << "Loop Unroll: F[" << Header->getParent()->getName()
       << "] Loop %" << Header->getName() << "\n";

  // Find trip count
  unsigned TripCount = L->getSmallConstantTripCount();
  unsigned Count = UnrollCount;
 
  // Automatically select an unroll count.
  if (Count == 0) {
    // Conservative heuristic: if we know the trip count, see if we can
    // completely unroll (subject to the threshold, checked below); otherwise
    // try to find greatest modulo of the trip count which is still under 
    // threshold value.
    if (TripCount != 0) {
      Count = TripCount;
    } else {
      return false;
    }
  }

  // Enforce the threshold.
  if (UnrollThreshold != NoThreshold) {
    unsigned LoopSize = ApproximateLoopSize(L);
    DOUT << "  Loop Size = " << LoopSize << "\n";
    uint64_t Size = (uint64_t)LoopSize*Count;
    if (TripCount != 1 && Size > UnrollThreshold) {
      DOUT << "  Too large to fully unroll with count: " << Count
           << " because size: " << Size << ">" << UnrollThreshold << "\n";
      if (UnrollAllowPartial) {
        // Reduce unroll count to be modulo of TripCount for partial unrolling
        Count = UnrollThreshold / LoopSize;        
        while (Count != 0 && TripCount%Count != 0) {
          Count--;
        }        
        if (Count < 2) {
          DOUT << "  could not unroll partially\n";
          return false;
        } else {
          DOUT << "  partially unrolling with count: " << Count << "\n";
        }
      } else {
        DOUT << "  will not try to unroll partially because "
             << "-unroll-allow-partial not given\n";
        return false;
      }
    }
  }

  // Unroll the loop.
  Function *F = L->getHeader()->getParent();
  if (!UnrollLoop(L, Count, LI, &LPM))
    return false;

  // FIXME: Reconstruct dom info, because it is not preserved properly.
  DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
  if (DT) {
    DT->runOnFunction(*F);
    DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>();
    if (DF)
      DF->runOnFunction(*F);
  }
  return true;
}
