//===- Transforms/Utils/SampleProfileInference.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for the profile inference algorithm, profi.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SAMPLEPROFILEINFERENCE_H
#define LLVM_TRANSFORMS_UTILS_SAMPLEPROFILEINFERENCE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

namespace llvm {

class Function;
class MachineBasicBlock;
class MachineFunction;

namespace afdo_detail {

template <class BlockT> struct TypeMap {};
template <> struct TypeMap<BasicBlock> {
  using BasicBlockT = BasicBlock;
  using FunctionT = Function;
};
template <> struct TypeMap<MachineBasicBlock> {
  using BasicBlockT = MachineBasicBlock;
  using FunctionT = MachineFunction;
};

} // end namespace afdo_detail

struct FlowJump;

/// A wrapper of a binary basic block.
struct FlowBlock {
  uint64_t Index;
  uint64_t Weight{0};
  bool UnknownWeight{false};
  uint64_t Flow{0};
  bool HasSelfEdge{false};
  std::vector<FlowJump *> SuccJumps;
  std::vector<FlowJump *> PredJumps;

  /// Check if it is the entry block in the function.
  bool isEntry() const { return PredJumps.empty(); }

  /// Check if it is an exit block in the function.
  bool isExit() const { return SuccJumps.empty(); }
};

/// A wrapper of a jump between two basic blocks.
struct FlowJump {
  uint64_t Source;
  uint64_t Target;
  uint64_t Flow{0};
  bool IsUnlikely{false};
};

/// A wrapper of binary function with basic blocks and jumps.
struct FlowFunction {
  std::vector<FlowBlock> Blocks;
  std::vector<FlowJump> Jumps;
  /// The index of the entry block.
  uint64_t Entry;
};

void applyFlowInference(FlowFunction &Func);

/// Sample profile inference pass.
template <typename BT> class SampleProfileInference {
public:
  using BasicBlockT = typename afdo_detail::TypeMap<BT>::BasicBlockT;
  using FunctionT = typename afdo_detail::TypeMap<BT>::FunctionT;
  using Edge = std::pair<const BasicBlockT *, const BasicBlockT *>;
  using BlockWeightMap = DenseMap<const BasicBlockT *, uint64_t>;
  using EdgeWeightMap = DenseMap<Edge, uint64_t>;
  using BlockEdgeMap =
      DenseMap<const BasicBlockT *, SmallVector<const BasicBlockT *, 8>>;

  SampleProfileInference(FunctionT &F, BlockEdgeMap &Successors,
                         BlockWeightMap &SampleBlockWeights)
      : F(F), Successors(Successors), SampleBlockWeights(SampleBlockWeights) {}

  /// Apply the profile inference algorithm for a given function
  void apply(BlockWeightMap &BlockWeights, EdgeWeightMap &EdgeWeights);

private:
  /// Try to infer branch probabilities mimicking implementation of
  /// BranchProbabilityInfo. Unlikely taken branches are marked so that the
  /// inference algorithm can avoid sending flow along corresponding edges.
  void findUnlikelyJumps(const std::vector<const BasicBlockT *> &BasicBlocks,
                         BlockEdgeMap &Successors, FlowFunction &Func);

  /// Determine whether the block is an exit in the CFG.
  bool isExit(const BasicBlockT *BB);

  /// Function.
  const FunctionT &F;

  /// Successors for each basic block in the CFG.
  BlockEdgeMap &Successors;

  /// Map basic blocks to their sampled weights.
  BlockWeightMap &SampleBlockWeights;
};

template <typename BT>
void SampleProfileInference<BT>::apply(BlockWeightMap &BlockWeights,
                                       EdgeWeightMap &EdgeWeights) {
  // Find all forwards reachable blocks which the inference algorithm will be
  // applied on.
  df_iterator_default_set<const BasicBlockT *> Reachable;
  for (auto *BB : depth_first_ext(&F, Reachable))
    (void)BB /* Mark all reachable blocks */;

  // Find all backwards reachable blocks which the inference algorithm will be
  // applied on.
  df_iterator_default_set<const BasicBlockT *> InverseReachable;
  for (const auto &BB : F) {
    // An exit block is a block without any successors.
    if (isExit(&BB)) {
      for (auto *RBB : inverse_depth_first_ext(&BB, InverseReachable))
        (void)RBB;
    }
  }

  // Keep a stable order for reachable blocks
  DenseMap<const BasicBlockT *, uint64_t> BlockIndex;
  std::vector<const BasicBlockT *> BasicBlocks;
  BlockIndex.reserve(Reachable.size());
  BasicBlocks.reserve(Reachable.size());
  for (const auto &BB : F) {
    if (Reachable.count(&BB) && InverseReachable.count(&BB)) {
      BlockIndex[&BB] = BasicBlocks.size();
      BasicBlocks.push_back(&BB);
    }
  }

  BlockWeights.clear();
  EdgeWeights.clear();
  bool HasSamples = false;
  for (const auto *BB : BasicBlocks) {
    auto It = SampleBlockWeights.find(BB);
    if (It != SampleBlockWeights.end() && It->second > 0) {
      HasSamples = true;
      BlockWeights[BB] = It->second;
    }
  }
  // Quit early for functions with a single block or ones w/o samples
  if (BasicBlocks.size() <= 1 || !HasSamples) {
    return;
  }

  // Create necessary objects
  FlowFunction Func;
  Func.Blocks.reserve(BasicBlocks.size());
  // Create FlowBlocks
  for (const auto *BB : BasicBlocks) {
    FlowBlock Block;
    if (SampleBlockWeights.find(BB) != SampleBlockWeights.end()) {
      Block.UnknownWeight = false;
      Block.Weight = SampleBlockWeights[BB];
    } else {
      Block.UnknownWeight = true;
      Block.Weight = 0;
    }
    Block.Index = Func.Blocks.size();
    Func.Blocks.push_back(Block);
  }
  // Create FlowEdges
  for (const auto *BB : BasicBlocks) {
    for (auto *Succ : Successors[BB]) {
      if (!BlockIndex.count(Succ))
        continue;
      FlowJump Jump;
      Jump.Source = BlockIndex[BB];
      Jump.Target = BlockIndex[Succ];
      Func.Jumps.push_back(Jump);
      if (BB == Succ) {
        Func.Blocks[BlockIndex[BB]].HasSelfEdge = true;
      }
    }
  }
  for (auto &Jump : Func.Jumps) {
    Func.Blocks[Jump.Source].SuccJumps.push_back(&Jump);
    Func.Blocks[Jump.Target].PredJumps.push_back(&Jump);
  }

  // Try to infer probabilities of jumps based on the content of basic block
  findUnlikelyJumps(BasicBlocks, Successors, Func);

  // Find the entry block
  for (size_t I = 0; I < Func.Blocks.size(); I++) {
    if (Func.Blocks[I].isEntry()) {
      Func.Entry = I;
      break;
    }
  }

  // Create and apply the inference network model.
  applyFlowInference(Func);

  // Extract the resulting weights from the control flow
  // All weights are increased by one to avoid propagation errors introduced by
  // zero weights.
  for (const auto *BB : BasicBlocks) {
    BlockWeights[BB] = Func.Blocks[BlockIndex[BB]].Flow;
  }
  for (auto &Jump : Func.Jumps) {
    Edge E = std::make_pair(BasicBlocks[Jump.Source], BasicBlocks[Jump.Target]);
    EdgeWeights[E] = Jump.Flow;
  }

#ifndef NDEBUG
  // Unreachable blocks and edges should not have a weight.
  for (auto &I : BlockWeights) {
    assert(Reachable.contains(I.first));
    assert(InverseReachable.contains(I.first));
  }
  for (auto &I : EdgeWeights) {
    assert(Reachable.contains(I.first.first) &&
           Reachable.contains(I.first.second));
    assert(InverseReachable.contains(I.first.first) &&
           InverseReachable.contains(I.first.second));
  }
#endif
}

template <typename BT>
inline void SampleProfileInference<BT>::findUnlikelyJumps(
    const std::vector<const BasicBlockT *> &BasicBlocks,
    BlockEdgeMap &Successors, FlowFunction &Func) {}

template <>
inline void SampleProfileInference<BasicBlock>::findUnlikelyJumps(
    const std::vector<const BasicBlockT *> &BasicBlocks,
    BlockEdgeMap &Successors, FlowFunction &Func) {
  for (auto &Jump : Func.Jumps) {
    const auto *BB = BasicBlocks[Jump.Source];
    const auto *Succ = BasicBlocks[Jump.Target];
    const Instruction *TI = BB->getTerminator();
    // Check if a block ends with InvokeInst and mark non-taken branch unlikely.
    // In that case block Succ should be a landing pad
    if (Successors[BB].size() == 2 && Successors[BB].back() == Succ) {
      if (isa<InvokeInst>(TI)) {
        Jump.IsUnlikely = true;
      }
    }
    const Instruction *SuccTI = Succ->getTerminator();
    // Check if the target block contains UnreachableInst and mark it unlikely
    if (SuccTI->getNumSuccessors() == 0) {
      if (isa<UnreachableInst>(SuccTI)) {
        Jump.IsUnlikely = true;
      }
    }
  }
}

template <typename BT>
inline bool SampleProfileInference<BT>::isExit(const BasicBlockT *BB) {
  return BB->succ_empty();
}

template <>
inline bool SampleProfileInference<BasicBlock>::isExit(const BasicBlock *BB) {
  return succ_empty(BB);
}

} // end namespace llvm
#endif // LLVM_TRANSFORMS_UTILS_SAMPLEPROFILEINFERENCE_H
