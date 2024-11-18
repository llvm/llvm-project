//===- SpillPlacement.cpp - Optimal Spill Code Placement ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the spill code placement analysis.
//
// Each edge bundle corresponds to a node in a Hopfield network. Constraints on
// basic blocks are weighted by the block frequency and added to become the node
// bias.
//
// Transparent basic blocks have the variable live through, but don't care if it
// is spilled or in a register. These blocks become connections in the Hopfield
// network, again weighted by block frequency.
//
// The Hopfield network minimizes (possibly locally) its energy function:
//
//   E = -sum_n V_n * ( B_n + sum_{n, m linked by b} V_m * F_b )
//
// The energy function represents the expected spill code execution frequency,
// or the cost of spilling. This is a Lyapunov function which never increases
// when a node is updated. It is guaranteed to converge to a local minimum.
//
//===----------------------------------------------------------------------===//

#include "SpillPlacement.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "spill-code-placement"

char SpillPlacementWrapperLegacy::ID = 0;

char &llvm::SpillPlacementID = SpillPlacementWrapperLegacy::ID;

INITIALIZE_PASS_BEGIN(SpillPlacementWrapperLegacy, DEBUG_TYPE,
                      "Spill Code Placement Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(EdgeBundlesWrapperLegacy)
INITIALIZE_PASS_END(SpillPlacementWrapperLegacy, DEBUG_TYPE,
                    "Spill Code Placement Analysis", true, true)

void SpillPlacementWrapperLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequiredTransitive<EdgeBundlesWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Node - Each edge bundle corresponds to a Hopfield node.
///
/// The node contains precomputed frequency data that only depends on the CFG,
/// but Bias and Links are computed each time placeSpills is called.
///
/// The node Value is positive when the variable should be in a register. The
/// value can change when linked nodes change, but convergence is very fast
/// because all weights are positive.
struct SpillPlacement::Node {
  /// BiasN - Sum of blocks that prefer a spill.
  BlockFrequency BiasN;

  /// BiasP - Sum of blocks that prefer a register.
  BlockFrequency BiasP;

  /// Value - Output value of this node computed from the Bias and links.
  /// This is always on of the values {-1, 0, 1}. A positive number means the
  /// variable should go in a register through this bundle.
  int Value;

  using LinkVector = SmallVector<std::pair<BlockFrequency, unsigned>, 4>;

  /// Links - (Weight, BundleNo) for all transparent blocks connecting to other
  /// bundles. The weights are all positive block frequencies.
  LinkVector Links;

  /// SumLinkWeights - Cached sum of the weights of all links + ThresHold.
  BlockFrequency SumLinkWeights;

  /// preferReg - Return true when this node prefers to be in a register.
  bool preferReg() const {
    // Undecided nodes (Value==0) go on the stack.
    return Value > 0;
  }

  /// mustSpill - Return True if this node is so biased that it must spill.
  bool mustSpill() const {
    // We must spill if Bias < -sum(weights) or the MustSpill flag was set.
    // BiasN is saturated when MustSpill is set, make sure this still returns
    // true when the RHS saturates. Note that SumLinkWeights includes Threshold.
    return BiasN >= BiasP + SumLinkWeights;
  }

  /// clear - Reset per-query data, but preserve frequencies that only depend on
  /// the CFG.
  void clear(BlockFrequency Threshold) {
    BiasN = BlockFrequency(0);
    BiasP = BlockFrequency(0);
    Value = 0;
    SumLinkWeights = Threshold;
    Links.clear();
  }

  /// addLink - Add a link to bundle b with weight w.
  void addLink(unsigned b, BlockFrequency w) {
    // Update cached sum.
    SumLinkWeights += w;

    // There can be multiple links to the same bundle, add them up.
    for (std::pair<BlockFrequency, unsigned> &L : Links)
      if (L.second == b) {
        L.first += w;
        return;
      }
    // This must be the first link to b.
    Links.push_back(std::make_pair(w, b));
  }

  /// addBias - Bias this node.
  void addBias(BlockFrequency freq, BorderConstraint direction) {
    switch (direction) {
    default:
      break;
    case PrefReg:
      BiasP += freq;
      break;
    case PrefSpill:
      BiasN += freq;
      break;
    case MustSpill:
      BiasN = BlockFrequency::max();
      break;
    }
  }

  /// update - Recompute Value from Bias and Links. Return true when node
  /// preference changes.
  bool update(const Node nodes[], BlockFrequency Threshold) {
    // Compute the weighted sum of inputs.
    BlockFrequency SumN = BiasN;
    BlockFrequency SumP = BiasP;
    for (std::pair<BlockFrequency, unsigned> &L : Links) {
      if (nodes[L.second].Value == -1)
        SumN += L.first;
      else if (nodes[L.second].Value == 1)
        SumP += L.first;
    }

    // Each weighted sum is going to be less than the total frequency of the
    // bundle. Ideally, we should simply set Value = sign(SumP - SumN), but we
    // will add a dead zone around 0 for two reasons:
    //
    //  1. It avoids arbitrary bias when all links are 0 as is possible during
    //     initial iterations.
    //  2. It helps tame rounding errors when the links nominally sum to 0.
    //
    bool Before = preferReg();
    if (SumN >= SumP + Threshold)
      Value = -1;
    else if (SumP >= SumN + Threshold)
      Value = 1;
    else
      Value = 0;
    return Before != preferReg();
  }

  void getDissentingNeighbors(SparseSet<unsigned> &List,
                              const Node nodes[]) const {
    for (const auto &Elt : Links) {
      unsigned n = Elt.second;
      // Neighbors that already have the same value are not going to
      // change because of this node changing.
      if (Value != nodes[n].Value)
        List.insert(n);
    }
  }
};

bool SpillPlacementWrapperLegacy::runOnMachineFunction(MachineFunction &MF) {
  auto *Bundles = &getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();
  auto *MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();

  Impl.reset(new SpillPlacement(Bundles, MBFI));
  Impl->run(MF);
  return false;
}

AnalysisKey SpillPlacementAnalysis::Key;

SpillPlacement
SpillPlacementAnalysis::run(MachineFunction &MF,
                            MachineFunctionAnalysisManager &MFAM) {
  auto *Bundles = &MFAM.getResult<EdgeBundlesAnalysis>(MF);
  auto *MBFI = &MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
  SpillPlacement Impl(Bundles, MBFI);
  Impl.run(MF);
  return Impl;
}

bool SpillPlacementAnalysis::Result::invalidate(
    MachineFunction &MF, const PreservedAnalyses &PA,
    MachineFunctionAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<SpillPlacementAnalysis>();
  return !(PAC.preserved() ||
           PAC.preservedSet<AllAnalysesOn<MachineFunction>>()) ||
         Inv.invalidate<EdgeBundlesAnalysis>(MF, PA) ||
         Inv.invalidate<MachineBlockFrequencyAnalysis>(MF, PA);
}

void SpillPlacement::arrayDeleter(Node *N) {
  if (N)
    delete[] N;
}

void SpillPlacement::run(MachineFunction &mf) {
  MF = &mf;

  assert(!nodes && "Leaking node array");
  nodes.reset(new Node[bundles->getNumBundles()]);
  TodoList.clear();
  TodoList.setUniverse(bundles->getNumBundles());

  // Compute total ingoing and outgoing block frequencies for all bundles.
  BlockFrequencies.resize(mf.getNumBlockIDs());
  setThreshold(MBFI->getEntryFreq());
  for (auto &I : mf) {
    unsigned Num = I.getNumber();
    BlockFrequencies[Num] = MBFI->getBlockFreq(&I);
  }
}

/// activate - mark node n as active if it wasn't already.
void SpillPlacement::activate(unsigned n) {
  TodoList.insert(n);
  if (ActiveNodes->test(n))
    return;
  ActiveNodes->set(n);
  nodes.get()[n].clear(Threshold);

  // Very large bundles usually come from big switches, indirect branches,
  // landing pads, or loops with many 'continue' statements. It is difficult to
  // allocate registers when so many different blocks are involved.
  //
  // Give a small negative bias to large bundles such that a substantial
  // fraction of the connected blocks need to be interested before we consider
  // expanding the region through the bundle. This helps compile time by
  // limiting the number of blocks visited and the number of links in the
  // Hopfield network.
  if (bundles->getBlocks(n).size() > 100) {
    nodes.get()[n].BiasP = BlockFrequency(0);
    BlockFrequency BiasN = MBFI->getEntryFreq();
    BiasN >>= 4;
    nodes.get()[n].BiasN = BiasN;
  }
}

/// Set the threshold for a given entry frequency.
///
/// Set the threshold relative to \c Entry.  Since the threshold is used as a
/// bound on the open interval (-Threshold;Threshold), 1 is the minimum
/// threshold.
void SpillPlacement::setThreshold(BlockFrequency Entry) {
  // Apparently 2 is a good threshold when Entry==2^14, but we need to scale
  // it.  Divide by 2^13, rounding as appropriate.
  uint64_t Freq = Entry.getFrequency();
  uint64_t Scaled = (Freq >> 13) + bool(Freq & (1 << 12));
  Threshold = BlockFrequency(std::max(UINT64_C(1), Scaled));
}

/// addConstraints - Compute node biases and weights from a set of constraints.
/// Set a bit in NodeMask for each active node.
void SpillPlacement::addConstraints(ArrayRef<BlockConstraint> LiveBlocks) {
  for (const BlockConstraint &LB : LiveBlocks) {
    BlockFrequency Freq = BlockFrequencies[LB.Number];

    // Live-in to block?
    if (LB.Entry != DontCare) {
      unsigned ib = bundles->getBundle(LB.Number, false);
      activate(ib);
      nodes.get()[ib].addBias(Freq, LB.Entry);
    }

    // Live-out from block?
    if (LB.Exit != DontCare) {
      unsigned ob = bundles->getBundle(LB.Number, true);
      activate(ob);
      nodes.get()[ob].addBias(Freq, LB.Exit);
    }
  }
}

/// addPrefSpill - Same as addConstraints(PrefSpill)
void SpillPlacement::addPrefSpill(ArrayRef<unsigned> Blocks, bool Strong) {
  for (unsigned B : Blocks) {
    BlockFrequency Freq = BlockFrequencies[B];
    if (Strong)
      Freq += Freq;
    unsigned ib = bundles->getBundle(B, false);
    unsigned ob = bundles->getBundle(B, true);
    activate(ib);
    activate(ob);
    nodes.get()[ib].addBias(Freq, PrefSpill);
    nodes.get()[ob].addBias(Freq, PrefSpill);
  }
}

void SpillPlacement::addLinks(ArrayRef<unsigned> Links) {
  for (unsigned Number : Links) {
    unsigned ib = bundles->getBundle(Number, false);
    unsigned ob = bundles->getBundle(Number, true);

    // Ignore self-loops.
    if (ib == ob)
      continue;
    activate(ib);
    activate(ob);
    BlockFrequency Freq = BlockFrequencies[Number];
    nodes.get()[ib].addLink(ob, Freq);
    nodes.get()[ob].addLink(ib, Freq);
  }
}

bool SpillPlacement::scanActiveBundles() {
  RecentPositive.clear();
  for (unsigned n : ActiveNodes->set_bits()) {
    update(n);
    // A node that must spill, or a node without any links is not going to
    // change its value ever again, so exclude it from iterations.
    if (nodes.get()[n].mustSpill())
      continue;
    if (nodes.get()[n].preferReg())
      RecentPositive.push_back(n);
  }
  return !RecentPositive.empty();
}

bool SpillPlacement::update(unsigned n) {
  if (!nodes.get()[n].update(nodes.get(), Threshold))
    return false;
  nodes.get()[n].getDissentingNeighbors(TodoList, nodes.get());
  return true;
}

/// iterate - Repeatedly update the Hopfield nodes until stability or the
/// maximum number of iterations is reached.
void SpillPlacement::iterate() {
  // We do not need to push those node in the todolist.
  // They are already been proceeded as part of the previous iteration.
  RecentPositive.clear();

  // Since the last iteration, the todolist have been augmented by calls
  // to addConstraints, addLinks, and co.
  // Update the network energy starting at this new frontier.
  // The call to ::update will add the nodes that changed into the todolist.
  unsigned Limit = bundles->getNumBundles() * 10;
  while(Limit-- > 0 && !TodoList.empty()) {
    unsigned n = TodoList.pop_back_val();
    if (!update(n))
      continue;
    if (nodes.get()[n].preferReg())
      RecentPositive.push_back(n);
  }
}

void SpillPlacement::prepare(BitVector &RegBundles) {
  RecentPositive.clear();
  TodoList.clear();
  // Reuse RegBundles as our ActiveNodes vector.
  ActiveNodes = &RegBundles;
  ActiveNodes->clear();
  ActiveNodes->resize(bundles->getNumBundles());
}

bool
SpillPlacement::finish() {
  assert(ActiveNodes && "Call prepare() first");

  // Write preferences back to ActiveNodes.
  bool Perfect = true;
  for (unsigned n : ActiveNodes->set_bits())
    if (!nodes.get()[n].preferReg()) {
      ActiveNodes->reset(n);
      Perfect = false;
    }
  ActiveNodes = nullptr;
  return Perfect;
}

void SpillPlacement::BlockConstraint::print(raw_ostream &OS) const {
  auto toString = [](BorderConstraint C) -> StringRef {
    switch(C) {
    case DontCare: return "DontCare";
    case PrefReg: return "PrefReg";
    case PrefSpill: return "PrefSpill";
    case PrefBoth: return "PrefBoth";
    case MustSpill: return "MustSpill";
    };
    llvm_unreachable("uncovered switch");
  };

  dbgs() << "{" << Number << ", "
         << toString(Entry) << ", "
         << toString(Exit) << ", "
         << (ChangesValue ? "changes" : "no change") << "}";
}

void SpillPlacement::BlockConstraint::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
