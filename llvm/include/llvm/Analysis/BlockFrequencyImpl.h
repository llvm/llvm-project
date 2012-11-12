//===---- BlockFrequencyImpl.h - Machine Block Frequency Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Shared implementation of BlockFrequency for IR and Machine Instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYIMPL_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYIMPL_H

#include "llvm/BasicBlock.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <string>

namespace llvm {


class BlockFrequencyInfo;
class MachineBlockFrequencyInfo;

/// BlockFrequencyImpl implements block frequency algorithm for IR and
/// Machine Instructions. Algorithm starts with value 1024 (START_FREQ)
/// for the entry block and then propagates frequencies using branch weights
/// from (Machine)BranchProbabilityInfo. LoopInfo is not required because
/// algorithm can find "backedges" by itself.
template<class BlockT, class FunctionT, class BlockProbInfoT>
class BlockFrequencyImpl {

  DenseMap<const BlockT *, BlockFrequency> Freqs;

  BlockProbInfoT *BPI;

  FunctionT *Fn;

  typedef GraphTraits< Inverse<BlockT *> > GT;

  const uint32_t EntryFreq;

  std::string getBlockName(BasicBlock *BB) const {
    return BB->getName().str();
  }

  std::string getBlockName(MachineBasicBlock *MBB) const {
    std::string str;
    raw_string_ostream ss(str);
    ss << "BB#" << MBB->getNumber();

    if (const BasicBlock *BB = MBB->getBasicBlock())
      ss << " derived from LLVM BB " << BB->getName();

    return ss.str();
  }

  void setBlockFreq(BlockT *BB, BlockFrequency Freq) {
    Freqs[BB] = Freq;
    DEBUG(dbgs() << "Frequency(" << getBlockName(BB) << ") = " << Freq << "\n");
  }

  /// getEdgeFreq - Return edge frequency based on SRC frequency and Src -> Dst
  /// edge probability.
  BlockFrequency getEdgeFreq(BlockT *Src, BlockT *Dst) const {
    BranchProbability Prob = BPI->getEdgeProbability(Src, Dst);
    return getBlockFreq(Src) * Prob;
  }

  /// incBlockFreq - Increase BB block frequency by FREQ.
  ///
  void incBlockFreq(BlockT *BB, BlockFrequency Freq) {
    Freqs[BB] += Freq;
    DEBUG(dbgs() << "Frequency(" << getBlockName(BB) << ") += " << Freq
                 << " --> " << Freqs[BB] << "\n");
  }

  /// divBlockFreq - Divide BB block frequency by PROB. If Prob = 0 do nothing.
  ///
  void divBlockFreq(BlockT *BB, BranchProbability Prob) {
    uint64_t N = Prob.getNumerator();
    assert(N && "Illegal division by zero!");
    uint64_t D = Prob.getDenominator();
    uint64_t Freq = (Freqs[BB].getFrequency() * D) / N;

    // Should we assert it?
    if (Freq > UINT32_MAX)
      Freq = UINT32_MAX;

    Freqs[BB] = BlockFrequency(Freq);
    DEBUG(dbgs() << "Frequency(" << getBlockName(BB) << ") /= (" << Prob
                 << ") --> " << Freqs[BB] << "\n");
  }

  // All blocks in postorder.
  std::vector<BlockT *> POT;

  // Map Block -> Position in reverse-postorder list.
  DenseMap<BlockT *, unsigned> RPO;

  // Cycle Probability for each bloch.
  DenseMap<BlockT *, uint32_t> CycleProb;

  // (reverse-)postorder traversal iterators.
  typedef typename std::vector<BlockT *>::iterator pot_iterator;
  typedef typename std::vector<BlockT *>::reverse_iterator rpot_iterator;

  pot_iterator pot_begin() { return POT.begin(); }
  pot_iterator pot_end() { return POT.end(); }

  rpot_iterator rpot_begin() { return POT.rbegin(); }
  rpot_iterator rpot_end() { return POT.rend(); }

  rpot_iterator rpot_at(BlockT *BB) {
    rpot_iterator I = rpot_begin();
    unsigned idx = RPO[BB];
    assert(idx);
    std::advance(I, idx - 1);

    assert(*I == BB);
    return I;
  }


  /// isReachable - Returns if BB block is reachable from the entry.
  ///
  bool isReachable(BlockT *BB) {
    return RPO.count(BB);
  }

  /// isBackedge - Return if edge Src -> Dst is a backedge.
  ///
  bool isBackedge(BlockT *Src, BlockT *Dst) {
    assert(isReachable(Src));
    assert(isReachable(Dst));

    unsigned a = RPO[Src];
    unsigned b = RPO[Dst];

    return a >= b;
  }

  /// getSingleBlockPred - return single BB block predecessor or NULL if
  /// BB has none or more predecessors.
  BlockT *getSingleBlockPred(BlockT *BB) {
    typename GT::ChildIteratorType
      PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
      PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);

    if (PI == PE)
      return 0;

    BlockT *Pred = *PI;

    ++PI;
    if (PI != PE)
      return 0;

    return Pred;
  }

  void doBlock(BlockT *BB, BlockT *LoopHead,
               SmallPtrSet<BlockT *, 8> &BlocksInLoop) {

    DEBUG(dbgs() << "doBlock(" << getBlockName(BB) << ")\n");
    setBlockFreq(BB, 0);

    if (BB == LoopHead) {
      setBlockFreq(BB, EntryFreq);
      return;
    }

    if(BlockT *Pred = getSingleBlockPred(BB)) {
      if (BlocksInLoop.count(Pred))
        setBlockFreq(BB, getEdgeFreq(Pred, BB));
      // TODO: else? irreducible, ignore it for now.
      return;
    }

    bool isInLoop = false;
    bool isLoopHead = false;

    for (typename GT::ChildIteratorType
         PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
         PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);
         PI != PE; ++PI) {
      BlockT *Pred = *PI;

      if (isReachable(Pred) && isBackedge(Pred, BB)) {
        isLoopHead = true;
      } else if (BlocksInLoop.count(Pred)) {
        incBlockFreq(BB, getEdgeFreq(Pred, BB));
        isInLoop = true;
      }
      // TODO: else? irreducible.
    }

    if (!isInLoop)
      return;

    if (!isLoopHead)
      return;

    assert(EntryFreq >= CycleProb[BB]);
    uint32_t CProb = CycleProb[BB];
    uint32_t Numerator = EntryFreq - CProb ? EntryFreq - CProb : 1;
    divBlockFreq(BB, BranchProbability(Numerator, EntryFreq));
  }

  /// doLoop - Propagate block frequency down through the loop.
  void doLoop(BlockT *Head, BlockT *Tail) {
    DEBUG(dbgs() << "doLoop(" << getBlockName(Head) << ", "
                 << getBlockName(Tail) << ")\n");

    SmallPtrSet<BlockT *, 8> BlocksInLoop;

    for (rpot_iterator I = rpot_at(Head), E = rpot_at(Tail); ; ++I) {
      BlockT *BB = *I;
      doBlock(BB, Head, BlocksInLoop);

      BlocksInLoop.insert(BB);
      if (I == E)
        break;
    }

    // Compute loop's cyclic probability using backedges probabilities.
    for (typename GT::ChildIteratorType
         PI = GraphTraits< Inverse<BlockT *> >::child_begin(Head),
         PE = GraphTraits< Inverse<BlockT *> >::child_end(Head);
         PI != PE; ++PI) {
      BlockT *Pred = *PI;
      assert(Pred);
      if (isReachable(Pred) && isBackedge(Pred, Head)) {
        uint64_t N = getEdgeFreq(Pred, Head).getFrequency();
        uint64_t D = getBlockFreq(Head).getFrequency();
        assert(N <= EntryFreq && "Backedge frequency must be <= EntryFreq!");
        uint64_t Res = (N * EntryFreq) / D;

        assert(Res <= UINT32_MAX);
        CycleProb[Head] += (uint32_t) Res;
        DEBUG(dbgs() << "  CycleProb[" << getBlockName(Head) << "] += " << Res
                     << " --> " << CycleProb[Head] << "\n");
      }
    }
  }

  friend class BlockFrequencyInfo;
  friend class MachineBlockFrequencyInfo;

  BlockFrequencyImpl() : EntryFreq(BlockFrequency::getEntryFrequency()) { }

  void doFunction(FunctionT *fn, BlockProbInfoT *bpi) {
    Fn = fn;
    BPI = bpi;

    // Clear everything.
    RPO.clear();
    POT.clear();
    CycleProb.clear();
    Freqs.clear();

    BlockT *EntryBlock = fn->begin();

    copy(po_begin(EntryBlock), po_end(EntryBlock), back_inserter(POT));

    unsigned RPOidx = 0;
    for (rpot_iterator I = rpot_begin(), E = rpot_end(); I != E; ++I) {
      BlockT *BB = *I;
      RPO[BB] = ++RPOidx;
      DEBUG(dbgs() << "RPO[" << getBlockName(BB) << "] = " << RPO[BB] << "\n");
    }

    // Travel over all blocks in postorder.
    for (pot_iterator I = pot_begin(), E = pot_end(); I != E; ++I) {
      BlockT *BB = *I;
      BlockT *LastTail = 0;
      DEBUG(dbgs() << "POT: " << getBlockName(BB) << "\n");

      for (typename GT::ChildIteratorType
           PI = GraphTraits< Inverse<BlockT *> >::child_begin(BB),
           PE = GraphTraits< Inverse<BlockT *> >::child_end(BB);
           PI != PE; ++PI) {

        BlockT *Pred = *PI;
        if (isReachable(Pred) && isBackedge(Pred, BB)
            && (!LastTail || RPO[Pred] > RPO[LastTail]))
          LastTail = Pred;
      }

      if (LastTail)
        doLoop(BB, LastTail);
    }

    // At the end assume the whole function as a loop, and travel over it once
    // again.
    doLoop(*(rpot_begin()), *(pot_begin()));
  }

public:
  /// getBlockFreq - Return block frequency. Return 0 if we don't have it.
  BlockFrequency getBlockFreq(const BlockT *BB) const {
    typename DenseMap<const BlockT *, BlockFrequency>::const_iterator
      I = Freqs.find(BB);
    if (I != Freqs.end())
      return I->second;
    return 0;
  }

  void print(raw_ostream &OS) const {
    OS << "\n\n---- Block Freqs ----\n";
    for (typename FunctionT::iterator I = Fn->begin(), E = Fn->end(); I != E;) {
      BlockT *BB = I++;
      OS << " " << getBlockName(BB) << " = " << getBlockFreq(BB) << "\n";

      for (typename GraphTraits<BlockT *>::ChildIteratorType
           SI = GraphTraits<BlockT *>::child_begin(BB),
           SE = GraphTraits<BlockT *>::child_end(BB); SI != SE; ++SI) {
        BlockT *Succ = *SI;
        OS << "  " << getBlockName(BB) << " -> " << getBlockName(Succ)
           << " = " << getEdgeFreq(BB, Succ) << "\n";
      }
    }
  }

  void dump() const {
    print(dbgs());
  }
};

}

#endif
