//===- GVN.h - Eliminate redundant values and loads -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Global Value Numbering pass
/// which eliminates fully redundant instructions. It also does somewhat Ad-Hoc
/// PRE and dead load elimination.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_GVN_H
#define LLVM_TRANSFORMS_SCALAR_GVN_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/PHITransAddr.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Transforms/Scalar/GVNLeaderMap.h"
#include "llvm/Transforms/Scalar/GVNValueTable.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace llvm {

class AAResults;
class AssumeInst;
class AssumptionCache;
class BasicBlock;
class BatchAAResults;
class CallInst;
class CondBrInst;
class EarliestEscapeAnalysis;
class ExtractValueInst;
class Function;
class FunctionPass;
class GVNLegacyPass;
class GetElementPtrInst;
class ImplicitControlFlowTracking;
class LoadInst;
class LoopInfo;
class MemDepResult;
class MemoryAccess;
class MemoryDependenceResults;
class MemoryLocation;
class MemorySSA;
class MemorySSAUpdater;
class NonLocalDepResult;
class OptimizationRemarkEmitter;
class PHINode;
class TargetLibraryInfo;
class Value;
class IntrinsicInst;

/// A set of parameters to control various transforms performed by GVN pass.
//  Each of the optional boolean parameters can be set to:
///      true - enabling the transformation.
///      false - disabling the transformation.
///      None - relying on a global default.
/// Intended use is to create a default object, modify parameters with
/// additional setters and then pass it to GVN.
struct GVNOptions {
  std::optional<bool> AllowScalarPRE;
  std::optional<bool> AllowLoadPRE;
  std::optional<bool> AllowLoadInLoopPRE;
  std::optional<bool> AllowLoadPRESplitBackedge;
  std::optional<bool> AllowMemDep;
  std::optional<bool> AllowMemorySSA;

  GVNOptions() = default;

  /// Enables or disables PRE of scalars in GVN.
  GVNOptions &setScalarPRE(bool ScalarPRE) {
    AllowScalarPRE = ScalarPRE;
    return *this;
  }

  /// Enables or disables PRE of loads in GVN.
  GVNOptions &setLoadPRE(bool LoadPRE) {
    AllowLoadPRE = LoadPRE;
    return *this;
  }

  GVNOptions &setLoadInLoopPRE(bool LoadInLoopPRE) {
    AllowLoadInLoopPRE = LoadInLoopPRE;
    return *this;
  }

  /// Enables or disables PRE of loads in GVN.
  GVNOptions &setLoadPRESplitBackedge(bool LoadPRESplitBackedge) {
    AllowLoadPRESplitBackedge = LoadPRESplitBackedge;
    return *this;
  }

  /// Enables or disables use of MemDepAnalysis.
  GVNOptions &setMemDep(bool MemDep) {
    AllowMemDep = MemDep;
    return *this;
  }

  /// Enables or disables use of MemorySSA.
  GVNOptions &setMemorySSA(bool MemSSA) {
    AllowMemorySSA = MemSSA;
    return *this;
  }
};

/// The core GVN pass object.
///
/// FIXME: We should have a good summary of the GVN algorithm implemented by
/// this particular pass here.
class GVNPass : public OptionalPassInfoMixin<GVNPass> {
public:
  struct AvailableValue;
  struct AvailableValueInBlock;
  struct ReachingMemVal;
  struct DependencyBlockInfo;

  friend class GVNValueTable;
  friend class GVNLegacyPass;

private:
  GVNOptions Options;
  MemoryDependenceResults *MD = nullptr;
  DominatorTree *DT = nullptr;
  const TargetLibraryInfo *TLI = nullptr;
  AssumptionCache *AC = nullptr;
  SetVector<BasicBlock *> DeadBlocks;
  OptimizationRemarkEmitter *ORE = nullptr;
  ImplicitControlFlowTracking *ICF = nullptr;
  LoopInfo *LI = nullptr;
  AAResults *AA = nullptr;
  MemorySSAUpdater *MSSAU = nullptr;
  GVNValueTable VN;
  GVNLeaderMap LeaderTable;

  // Map the block to reversed postorder traversal number. It is used to
  // find back edge easily.
  DenseMap<AssertingVH<BasicBlock>, uint32_t> BlockRPONumber;

  // This is set 'true' initially and also when new blocks have been added to
  // the function being analyzed. This boolean is used to control the updating
  // of BlockRPONumber prior to accessing the contents of BlockRPONumber.
  bool InvalidBlockRPONumbers = true;

  // List of critical edges to be split between iterations.
  SmallVector<std::pair<Instruction *, unsigned>, 4> ToSplit;

public:
  GVNPass(GVNOptions Options = {}) : Options(Options) {}

  /// Run the pass over the function.
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  LLVM_ABI void
  printPipeline(raw_ostream &OS,
                function_ref<StringRef(StringRef)> MapClassName2PassName);

private:
  DominatorTree &getDominatorTree() const { return *DT; }
  AAResults *getAliasAnalysis() const { return VN.getAliasAnalysis(); }
  MemoryDependenceResults &getMemDep() const { return *MD; }

  bool isScalarPREEnabled() const;
  bool isLoadPREEnabled() const;
  bool isLoadInLoopPREEnabled() const;
  bool isLoadPRESplitBackedgeEnabled() const;
  bool isMemDepEnabled() const;
  bool isMemorySSAEnabled() const;

  using LoadDepVect = SmallVector<NonLocalDepResult, 64>;
  using AvailValInBlkVect = SmallVector<AvailableValueInBlock, 64>;
  using UnavailBlkVect = SmallVector<BasicBlock *, 64>;

  using DependencyBlockSet = DenseMap<BasicBlock *, DependencyBlockInfo>;

  /// Given a select-dependency for the load (the load address is a select of
  /// \p TrueAddr and \p FalseAddr guarded by \p Cond), determine whether a
  /// value is available by finding dominating values for both addresses.  If
  /// so, the load can be rematerialized as a select of those two values.
  std::optional<AvailableValue>
  analyzeSelectAvailability(LoadInst *Load, Value *Cond, Value *TrueAddr,
                            Value *FalseAddr, Instruction *From);

  /// Given a local dependency (Def or Clobber) determine if a value is
  /// available for the load.
  std::optional<AvailableValue>
  analyzeLoadAvailability(LoadInst *Load, const ReachingMemVal &Dep,
                          Value *Address);

  /// Given a list of non-local dependencies, determine if a value is
  /// available for the load in each specified block.  If it is, add it to
  /// ValuesPerBlock.  If not, add it to UnavailableBlocks.
  void analyzeLoadAvailability(LoadInst *Load,
                               SmallVectorImpl<ReachingMemVal> &Deps,
                               AvailValInBlkVect &ValuesPerBlock,
                               UnavailBlkVect &UnavailableBlocks);

  /// Given a critical edge from Pred to LoadBB, find a load instruction
  /// which is identical to Load from another successor of Pred.
  LoadInst *findLoadToHoistIntoPred(BasicBlock *Pred, BasicBlock *LoadBB,
                                    LoadInst *Load);

  /// Eliminates partially redundant \p Load, replacing it with \p
  /// AvailableLoads (connected by Phis if needed).
  void eliminatePartiallyRedundantLoad(
      LoadInst *Load, AvailValInBlkVect &ValuesPerBlock,
      MapVector<BasicBlock *, Value *> &AvailableLoads,
      MapVector<BasicBlock *, LoadInst *> *CriticalEdgePredAndLoad);

  // Helper functions for d etermining load dependencies.
  std::optional<GVNPass::ReachingMemVal> scanMemoryAccessesUsers(
      const MemoryLocation &Loc, bool IsInvariantLoad, BasicBlock *BB,
      const SmallVectorImpl<MemoryAccess *> &ClobbersList, MemorySSA &MSSA,
      BatchAAResults &AA, LoadInst *L = nullptr);

  std::optional<GVNPass::ReachingMemVal>
  accessMayModifyLocation(MemoryAccess *ClobberMA, const MemoryLocation &Loc,
                          bool IsInvariantLoad, BasicBlock *BB, MemorySSA &MSSA,
                          BatchAAResults &AA);

  bool collectPredecessors(BasicBlock *BB, const PHITransAddr &Addr,
                           MemoryAccess *ClobberMA, DependencyBlockSet &Blocks,
                           SmallVectorImpl<BasicBlock *> &Worklist);

  void collectClobberList(SmallVectorImpl<MemoryAccess *> &Clobbers,
                          BasicBlock *BB, const DependencyBlockInfo &StartInfo,
                          const DependencyBlockSet &Blocks, MemorySSA &MSSA);

  bool findReachingValuesForLoad(LoadInst *Inst,
                                 SmallVectorImpl<ReachingMemVal> &Values,
                                 MemorySSA &MSSA, AAResults &AA);

  bool performLoadPRE(LoadInst *Load, AvailValInBlkVect &ValuesPerBlock,
                      UnavailBlkVect &UnavailableBlocks);

  /// Try to replace a load which executes on each loop iteraiton with Phi
  /// translation of load in preheader and load(s) in conditionally executed
  /// paths.
  bool performLoopLoadPRE(LoadInst *Load, AvailValInBlkVect &ValuesPerBlock,
                          UnavailBlkVect &UnavailableBlocks);

  // Try to eliminate redundent loades with non-local dependencies.
  bool processNonLocalLoad(LoadInst *L);
  bool processNonLocalLoad(LoadInst *L, SmallVectorImpl<ReachingMemVal> &Deps);

  /// Add any blocks determined to be unreachable by a conditional branch with a
  /// constant condition to the dead blocks.
  bool processFoldableCondBr(CondBrInst *BI);

  /// Propagate equalities derived from llvm.assume intrinsics.
  bool processAssumeIntrinsic(AssumeInst *II);

  /// Try to eliminate redundant loads.
  bool processLoad(LoadInst *L);

  /// Try to eliminate masked loads which have loaded from
  /// masked stores with the same mask.
  bool processMaskedLoad(IntrinsicInst *I);

  /// Propagate value of a condition to blocks dominated by "then" and "else"
  /// edges, as well as certains derived equalities.
  bool
  propagateEquality(Value *LHS, Value *RHS,
                    const std::variant<BasicBlockEdge, Instruction *> &Root);

  // Pass iteration helper functions.
  bool processInstruction(Instruction *I);
  bool processBlock(BasicBlock *BB);
  bool iterateOnFunction(Function &F);

  // Scalar PRE helper functions
  bool performScalarPREInsertion(Instruction *Instr, BasicBlock *Pred,
                                 BasicBlock *Curr, unsigned int ValNo);
  bool performScalarPRE(Instruction *I);
  bool performPRE(Function &F);

  /// Main entry point for the GVN pass. Also used by the GVNLegacyPass.
  bool runImpl(Function &F, AssumptionCache &RunAC, DominatorTree &RunDT,
               const TargetLibraryInfo &RunTLI, AAResults &RunAA,
               MemoryDependenceResults *RunMD, LoopInfo &LI,
               OptimizationRemarkEmitter *ORE, MemorySSA *MSSA = nullptr);

  // Other helper routines.

  Value *findLeader(const BasicBlock *BB, uint32_t Num);
  void cleanupGlobalSets();

  void removeInstruction(Instruction *I);

  /// This removes the specified instruction from
  /// our various maps and marks it for deletion.
  void salvageAndRemoveInstruction(Instruction *I);

  void verifyRemoved(const Instruction *I) const;

  bool splitCriticalEdges();
  BasicBlock *splitCriticalEdges(BasicBlock *Pred, BasicBlock *Succ);

  void addDeadBlock(BasicBlock *BB);
  void assignValNumForDeadCode();
  void assignBlockRPONumber(Function &F);
};

/// Create a legacy GVN pass.
LLVM_ABI FunctionPass *createGVNPass(bool ScalarPRE);
LLVM_ABI FunctionPass *createGVNPass();

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_GVN_H
