//===---- SLPVectorizer.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SLPVECTORIZER_H
#define LLVM_TRANSFORMS_VECTORIZE_SLPVECTORIZER_H

#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// A private "module" namespace for types and utilities used by this pass.
/// These are implementation details and should not be used by clients.
namespace slpvectorizer {
class BoUpSLP;
}

struct SLPVectorizerPass : public PassInfoMixin<SLPVectorizerPass> {
  typedef SmallVector<StoreInst *, 8> StoreList;
  typedef MapVector<Value *, StoreList> StoreListMap;
  typedef SmallVector<WeakTrackingVH, 8> WeakTrackingVHList;
  typedef MapVector<Value *, WeakTrackingVHList> WeakTrackingVHListMap;

  ScalarEvolution *SE = nullptr;
  TargetTransformInfo *TTI = nullptr;
  TargetLibraryInfo *TLI = nullptr;
  AliasAnalysis *AA = nullptr;
  LoopInfo *LI = nullptr;
  DominatorTree *DT = nullptr;
  AssumptionCache *AC = nullptr;
  DemandedBits *DB = nullptr;
  const DataLayout *DL = nullptr;

public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  // Glue for old PM.
  bool runImpl(Function &F, ScalarEvolution *SE_, TargetTransformInfo *TTI_,
               TargetLibraryInfo *TLI_, AliasAnalysis *AA_, LoopInfo *LI_,
               DominatorTree *DT_, AssumptionCache *AC_, DemandedBits *DB_);

private:
  /// \brief Collect store and getelementptr instructions and organize them
  /// according to the underlying object of their pointer operands. We sort the
  /// instructions by their underlying objects to reduce the cost of
  /// consecutive access queries.
  ///
  /// TODO: We can further reduce this cost if we flush the chain creation
  ///       every time we run into a memory barrier.
  void collectSeedInstructions(BasicBlock *BB);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B, slpvectorizer::BoUpSLP &R);

  /// \brief Try to vectorize a list of operands.
  /// \@param BuildVector A list of users to ignore for the purpose of
  ///                     scheduling and that don't need extracting.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, slpvectorizer::BoUpSLP &R,
                          ArrayRef<Value *> BuildVector = None,
                          bool AllowReorder = false);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V, slpvectorizer::BoUpSLP &R);

  /// \brief Vectorize the store instructions collected in Stores.
  bool vectorizeStoreChains(slpvectorizer::BoUpSLP &R);

  /// \brief Vectorize the index computations of the getelementptr instructions
  /// collected in GEPs.
  bool vectorizeGEPIndices(BasicBlock *BB, slpvectorizer::BoUpSLP &R);

  /// Try to find horizontal reduction or otherwise vectorize a chain of binary
  /// operators.
  bool vectorizeRootInstruction(PHINode *P, Value *V, BasicBlock *BB,
                                slpvectorizer::BoUpSLP &R,
                                TargetTransformInfo *TTI);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, slpvectorizer::BoUpSLP &R);

  bool vectorizeStoreChain(ArrayRef<Value *> Chain, slpvectorizer::BoUpSLP &R,
                           unsigned VecRegSize);

  bool vectorizeStores(ArrayRef<StoreInst *> Stores, slpvectorizer::BoUpSLP &R);

  /// The store instructions in a basic block organized by base pointer.
  StoreListMap Stores;

  /// The getelementptr instructions in a basic block organized by base pointer.
  WeakTrackingVHListMap GEPs;
};
}

#endif // LLVM_TRANSFORMS_VECTORIZE_SLPVECTORIZER_H
