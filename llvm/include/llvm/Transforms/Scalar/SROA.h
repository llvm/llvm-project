//===- SROA.h - Scalar Replacement Of Aggregates ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for LLVM's Scalar Replacement of
/// Aggregates pass. This pass provides both aggregate splitting and the
/// primary SSA formation used in the compiler.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_SROA_H
#define LLVM_TRANSFORMS_SCALAR_SROA_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include <vector>

namespace llvm {

class AllocaInst;
class LoadInst;
class AssumptionCache;
class DominatorTree;
class DomTreeUpdater;
class Function;
class LLVMContext;
class PHINode;
class SelectInst;
class Use;

/// A private "module" namespace for types and utilities used by SROA. These
/// are implementation details and should not be used by clients.
namespace sroa LLVM_LIBRARY_VISIBILITY {

class AllocaSliceRewriter;
class AllocaSlices;
class Partition;
class SROALegacyPass;

class SelectHandSpeculativity {
  unsigned char Storage = 0;
  using TrueVal = Bitfield::Element<bool, 0, 1>;  // Low 0'th bit.
  using FalseVal = Bitfield::Element<bool, 1, 1>; // Low 1'th bit.
public:
  SelectHandSpeculativity() = default;
  SelectHandSpeculativity &setAsSpeculatable(bool isTrueVal);
  bool isSpeculatable(bool isTrueVal) const;
  bool areAllSpeculatable() const;
  bool areAnySpeculatable() const;
  bool areNoneSpeculatable() const;
  // For interop as int half of PointerIntPair.
  explicit operator intptr_t() const { return static_cast<intptr_t>(Storage); }
  explicit SelectHandSpeculativity(intptr_t Storage_) : Storage(Storage_) {}
};
static_assert(sizeof(SelectHandSpeculativity) == sizeof(unsigned char));

using PossiblySpeculatableLoad =
    PointerIntPair<LoadInst *, 2, sroa::SelectHandSpeculativity>;
using PossiblySpeculatableLoads = SmallVector<PossiblySpeculatableLoad, 2>;

} // end namespace sroa

enum class SROAOptions : bool { ModifyCFG, PreserveCFG };

/// An optimization pass providing Scalar Replacement of Aggregates.
///
/// This pass takes allocations which can be completely analyzed (that is, they
/// don't escape) and tries to turn them into scalar SSA values. There are
/// a few steps to this process.
///
/// 1) It takes allocations of aggregates and analyzes the ways in which they
///    are used to try to split them into smaller allocations, ideally of
///    a single scalar data type. It will split up memcpy and memset accesses
///    as necessary and try to isolate individual scalar accesses.
/// 2) It will transform accesses into forms which are suitable for SSA value
///    promotion. This can be replacing a memset with a scalar store of an
///    integer value, or it can involve speculating operations on a PHI or
///    select to be a PHI or select of the results.
/// 3) Finally, this will try to detect a pattern of accesses which map cleanly
///    onto insert and extract operations on a vector value, and convert them to
///    this form. By doing so, it will enable promotion of vector aggregates to
///    SSA vector values.
class SROAPass : public PassInfoMixin<SROAPass> {
  LLVMContext *C = nullptr;
  DomTreeUpdater *DTU = nullptr;
  AssumptionCache *AC = nullptr;
  const bool PreserveCFG;

  /// Worklist of alloca instructions to simplify.
  ///
  /// Each alloca in the function is added to this. Each new alloca formed gets
  /// added to it as well to recursively simplify unless that alloca can be
  /// directly promoted. Finally, each time we rewrite a use of an alloca other
  /// the one being actively rewritten, we add it back onto the list if not
  /// already present to ensure it is re-visited.
  SetVector<AllocaInst *, SmallVector<AllocaInst *, 16>> Worklist;

  /// A collection of instructions to delete.
  /// We try to batch deletions to simplify code and make things a bit more
  /// efficient. We also make sure there is no dangling pointers.
  SmallVector<WeakVH, 8> DeadInsts;

  /// Post-promotion worklist.
  ///
  /// Sometimes we discover an alloca which has a high probability of becoming
  /// viable for SROA after a round of promotion takes place. In those cases,
  /// the alloca is enqueued here for re-processing.
  ///
  /// Note that we have to be very careful to clear allocas out of this list in
  /// the event they are deleted.
  SetVector<AllocaInst *, SmallVector<AllocaInst *, 16>> PostPromotionWorklist;

  /// A collection of alloca instructions we can directly promote.
  std::vector<AllocaInst *> PromotableAllocas;

  /// A worklist of PHIs to speculate prior to promoting allocas.
  ///
  /// All of these PHIs have been checked for the safety of speculation and by
  /// being speculated will allow promoting allocas currently in the promotable
  /// queue.
  SetVector<PHINode *, SmallVector<PHINode *, 8>> SpeculatablePHIs;

  /// A worklist of select instructions to rewrite prior to promoting
  /// allocas.
  SmallMapVector<SelectInst *, sroa::PossiblySpeculatableLoads, 8>
      SelectsToRewrite;

  /// Select instructions that use an alloca and are subsequently loaded can be
  /// rewritten to load both input pointers and then select between the result,
  /// allowing the load of the alloca to be promoted.
  /// From this:
  ///   %P2 = select i1 %cond, ptr %Alloca, ptr %Other
  ///   %V = load <type>, ptr %P2
  /// to:
  ///   %V1 = load <type>, ptr %Alloca      -> will be mem2reg'd
  ///   %V2 = load <type>, ptr %Other
  ///   %V = select i1 %cond, <type> %V1, <type> %V2
  ///
  /// We can do this to a select if its only uses are loads
  /// and if either the operand to the select can be loaded unconditionally,
  ///        or if we are allowed to perform CFG modifications.
  /// If found an intervening bitcast with a single use of the load,
  /// allow the promotion.
  static std::optional<sroa::PossiblySpeculatableLoads>
  isSafeSelectToSpeculate(SelectInst &SI, bool PreserveCFG);

public:
  /// If \p PreserveCFG is set, then the pass is not allowed to modify CFG
  /// in any way, even if it would update CFG analyses.
  SROAPass(SROAOptions PreserveCFG);

  /// Run the pass over the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  void printPipeline(raw_ostream &OS,
                     function_ref<StringRef(StringRef)> MapClassName2PassName);

private:
  friend class sroa::AllocaSliceRewriter;
  friend class sroa::SROALegacyPass;

  /// Helper used by both the public run method and by the legacy pass.
  PreservedAnalyses runImpl(Function &F, DomTreeUpdater &RunDTU,
                            AssumptionCache &RunAC);
  PreservedAnalyses runImpl(Function &F, DominatorTree &RunDT,
                            AssumptionCache &RunAC);

  bool presplitLoadsAndStores(AllocaInst &AI, sroa::AllocaSlices &AS);
  AllocaInst *rewritePartition(AllocaInst &AI, sroa::AllocaSlices &AS,
                               sroa::Partition &P);
  bool splitAlloca(AllocaInst &AI, sroa::AllocaSlices &AS);
  std::pair<bool /*Changed*/, bool /*CFGChanged*/> runOnAlloca(AllocaInst &AI);
  void clobberUse(Use &U);
  bool deleteDeadInstructions(SmallPtrSetImpl<AllocaInst *> &DeletedAllocas);
  bool promoteAllocas(Function &F);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_SROA_H
