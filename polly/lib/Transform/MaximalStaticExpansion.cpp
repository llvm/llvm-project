//===- MaximalStaticExpansion.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass fully expand the memory accesses of a Scop to get rid of
// dependencies.
//
//===----------------------------------------------------------------------===//

#include "polly/MaximalStaticExpansion.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/InitializePasses.h"
#include "isl/isl-noexceptions.h"
#include "isl/union_map.h"
#include <cassert>
#include <limits>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-mse"

namespace {

class MaximalStaticExpanderWrapperPass final : public ScopPass {
public:
  static char ID;

  explicit MaximalStaticExpanderWrapperPass() : ScopPass(ID) {}

  ~MaximalStaticExpanderWrapperPass() override = default;

  /// Expand the accesses of the SCoP.
  ///
  /// @param S The SCoP that must be expanded.
  bool runOnScop(Scop &S) override;

  /// Print the SCoP.
  ///
  /// @param OS The stream where to print.
  /// @param S The SCop that must be printed.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformations required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

#ifndef NDEBUG
/// Whether a dimension of a set is bounded (lower and upper) by a constant,
/// i.e. there are two constants Min and Max, such that every value x of the
/// chosen dimensions is Min <= x <= Max.
static bool isDimBoundedByConstant(isl::set Set, unsigned dim) {
  auto ParamDims = unsignedFromIslSize(Set.dim(isl::dim::param));
  Set = Set.project_out(isl::dim::param, 0, ParamDims);
  Set = Set.project_out(isl::dim::set, 0, dim);
  auto SetDims = unsignedFromIslSize(Set.tuple_dim());
  assert(SetDims >= 1);
  Set = Set.project_out(isl::dim::set, 1, SetDims - 1);
  return bool(Set.is_bounded());
}
#endif

class MaximalStaticExpansionImpl {
  OptimizationRemarkEmitter &ORE;
  Scop &S;
  isl::union_map &Dependences;

  /// Emit remark
  void emitRemark(StringRef Msg, Instruction *Inst) {
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ExpansionRejection", Inst)
             << Msg);
  }

  /// Filter the dependences to have only one related to current memory access.
  ///
  /// @param S The SCop in which the memory access appears in.
  /// @param MapDependences The dependences to filter.
  /// @param MA The memory access that need to be expanded.
  isl::union_map filterDependences(const isl::union_map &Dependences,
                                   MemoryAccess *MA) {
    auto SAI = MA->getLatestScopArrayInfo();

    auto AccessDomainSet = MA->getAccessRelation().domain();
    auto AccessDomainId = AccessDomainSet.get_tuple_id();

    isl::union_map MapDependences = isl::union_map::empty(S.getIslCtx());

    for (isl::map Map : Dependences.get_map_list()) {
      // Filter out Statement to Statement dependences.
      if (!Map.can_curry())
        continue;

      // Intersect with the relevant SAI.
      auto TmpMapDomainId =
          Map.get_space().domain().unwrap().range().get_tuple_id(isl::dim::set);

      ScopArrayInfo *UserSAI =
          static_cast<ScopArrayInfo *>(TmpMapDomainId.get_user());

      if (SAI != UserSAI)
        continue;

      // Get the correct S1[] -> S2[] dependence.
      auto NewMap = Map.factor_domain();
      auto NewMapDomainId = NewMap.domain().get_tuple_id();

      if (AccessDomainId.get() != NewMapDomainId.get())
        continue;

      // Add the corresponding map to MapDependences.
      MapDependences = MapDependences.unite(NewMap);
    }

    return MapDependences;
  }

  /// Return true if the SAI in parameter is expandable.
  ///
  /// @param SAI the SAI that need to be checked.
  /// @param Writes A set that will contains all the write accesses.
  /// @param Reads A set that will contains all the read accesses.
  /// @param S The SCop in which the SAI is in.
  /// @param Dependences The RAW dependences of the SCop.
  bool isExpandable(const ScopArrayInfo *SAI,
                    SmallPtrSetImpl<MemoryAccess *> &Writes,
                    SmallPtrSetImpl<MemoryAccess *> &Reads, Scop &S) {
    if (SAI->isValueKind()) {
      Writes.insert(S.getValueDef(SAI));
      for (auto MA : S.getValueUses(SAI))
        Reads.insert(MA);
      return true;
    } else if (SAI->isPHIKind()) {
      auto Read = S.getPHIRead(SAI);

      auto StmtDomain = isl::union_set(Read->getStatement()->getDomain());

      auto Writes = S.getPHIIncomings(SAI);

      // Get the domain where all the writes are writing to.
      auto WriteDomain = isl::union_set::empty(S.getIslCtx());

      for (auto Write : Writes) {
        auto MapDeps = filterDependences(Dependences, Write);
        for (isl::map Map : MapDeps.get_map_list())
          WriteDomain = WriteDomain.unite(Map.range());
      }

      // For now, read from original scalar is not possible.
      if (!StmtDomain.is_equal(WriteDomain)) {
        emitRemark(SAI->getName() + " read from its original value.",
                   Read->getAccessInstruction());
        return false;
      }

      return true;
    } else if (SAI->isExitPHIKind()) {
      // For now, we are not able to expand ExitPhi.
      emitRemark(SAI->getName() + " is a ExitPhi node.",
                 &*S.getEnteringBlock()->getFirstNonPHIIt());
      return false;
    }

    int NumberWrites = 0;
    for (ScopStmt &Stmt : S) {
      auto StmtReads = isl::union_map::empty(S.getIslCtx());
      auto StmtWrites = isl::union_map::empty(S.getIslCtx());

      for (MemoryAccess *MA : Stmt) {
        // Check if the current MemoryAccess involved the current SAI.
        if (SAI != MA->getLatestScopArrayInfo())
          continue;

        // For now, we are not able to expand array where read come after write
        // (to the same location) in a same statement.
        auto AccRel = isl::union_map(MA->getAccessRelation());
        if (MA->isRead()) {
          // Reject load after store to same location.
          if (!StmtWrites.is_disjoint(AccRel)) {
            emitRemark(SAI->getName() + " has read after write to the same "
                                        "element in same statement. The "
                                        "dependences found during analysis may "
                                        "be wrong because Polly is not able to "
                                        "handle such case for now.",
                       MA->getAccessInstruction());
            return false;
          }

          StmtReads = StmtReads.unite(AccRel);
        } else {
          StmtWrites = StmtWrites.unite(AccRel);
        }

        // For now, we are not able to expand MayWrite.
        if (MA->isMayWrite()) {
          emitRemark(SAI->getName() + " has a maywrite access.",
                     MA->getAccessInstruction());
          return false;
        }

        // For now, we are not able to expand SAI with more than one write.
        if (MA->isMustWrite()) {
          Writes.insert(MA);
          NumberWrites++;
          if (NumberWrites > 1) {
            emitRemark(SAI->getName() + " has more than 1 write access.",
                       MA->getAccessInstruction());
            return false;
          }
        }

        // Check if it is possible to expand this read.
        if (MA->isRead()) {
          // Get the domain of the current ScopStmt.
          auto StmtDomain = Stmt.getDomain();

          // Get the domain of the future Read access.
          auto ReadDomainSet = MA->getAccessRelation().domain();
          auto ReadDomain = isl::union_set(ReadDomainSet);

          // Get the dependences relevant for this MA
          auto MapDependences = filterDependences(Dependences.reverse(), MA);
          unsigned NumberElementMap = isl_union_map_n_map(MapDependences.get());

          if (NumberElementMap == 0) {
            emitRemark("The expansion of " + SAI->getName() +
                           " would lead to a read from the original array.",
                       MA->getAccessInstruction());
            return false;
          }

          auto DepsDomain = MapDependences.domain();

          // If there are multiple maps in the Deps, we cannot handle this case
          // for now.
          if (NumberElementMap != 1) {
            emitRemark(SAI->getName() +
                           " has too many dependences to be handle for now.",
                       MA->getAccessInstruction());
            return false;
          }

          auto DepsDomainSet = isl::set(DepsDomain);

          // For now, read from the original array is not possible.
          if (!StmtDomain.is_subset(DepsDomainSet)) {
            emitRemark("The expansion of " + SAI->getName() +
                           " would lead to a read from the original array.",
                       MA->getAccessInstruction());
            return false;
          }

          Reads.insert(MA);
        }
      }
    }

    // No need to expand SAI with no write.
    if (NumberWrites == 0) {
      emitRemark(SAI->getName() + " has 0 write access.",
                 &*S.getEnteringBlock()->getFirstNonPHIIt());
      return false;
    }

    return true;
  }

  /// Expand the MemoryAccess according to Dependences and already expanded
  /// MemoryAccesses.
  ///
  /// @param The SCop in which the memory access appears in.
  /// @param The memory access that need to be expanded.
  /// @param Dependences The RAW dependences of the SCop.
  /// @param ExpandedSAI The expanded SAI created during write expansion.
  /// @param Reverse if true, the Dependences union_map is reversed before
  /// intersection.
  void mapAccess(SmallPtrSetImpl<MemoryAccess *> &Accesses,
                 const isl::union_map &Dependences, ScopArrayInfo *ExpandedSAI,
                 bool Reverse) {
    for (auto MA : Accesses) {
      // Get the current AM.
      auto CurrentAccessMap = MA->getAccessRelation();

      // Get RAW dependences for the current WA.
      auto DomainSet = MA->getAccessRelation().domain();
      auto Domain = isl::union_set(DomainSet);

      // Get the dependences relevant for this MA.
      isl::union_map MapDependences =
          filterDependences(Reverse ? Dependences.reverse() : Dependences, MA);

      // If no dependences, no need to modify anything.
      if (MapDependences.is_empty())
        return;

      assert(isl_union_map_n_map(MapDependences.get()) == 1 &&
             "There are more than one RAW dependencies in the union map.");
      auto NewAccessMap = isl::map::from_union_map(MapDependences);

      auto Id = ExpandedSAI->getBasePtrId();

      // Replace the out tuple id with the one of the access array.
      NewAccessMap = NewAccessMap.set_tuple_id(isl::dim::out, Id);

      // Set the new access relation.
      MA->setNewAccessRelation(NewAccessMap);
    }
  }

  /// Expand the MemoryAccess according to its domain.
  ///
  /// @param S The SCop in which the memory access appears in.
  /// @param MA The memory access that need to be expanded.
  ScopArrayInfo *expandAccess(MemoryAccess *MA) {
    // Get the current AM.
    auto CurrentAccessMap = MA->getAccessRelation();

    unsigned in_dimensions =
        unsignedFromIslSize(CurrentAccessMap.domain_tuple_dim());

    // Get domain from the current AM.
    auto Domain = CurrentAccessMap.domain();

    // Create a new AM from the domain.
    auto NewAccessMap = isl::map::from_domain(Domain);

    // Add dimensions to the new AM according to the current in_dim.
    NewAccessMap = NewAccessMap.add_dims(isl::dim::out, in_dimensions);

    // Create the string representing the name of the new SAI.
    // One new SAI for each statement so that each write go to a different
    // memory cell.
    auto CurrentStmtDomain = MA->getStatement()->getDomain();
    auto CurrentStmtName = CurrentStmtDomain.get_tuple_name();
    auto CurrentOutId = CurrentAccessMap.get_tuple_id(isl::dim::out);
    std::string CurrentOutIdString =
        MA->getScopArrayInfo()->getName() + "_" + CurrentStmtName + "_expanded";

    // Set the tuple id for the out dimension.
    NewAccessMap = NewAccessMap.set_tuple_id(isl::dim::out, CurrentOutId);

    // Create the size vector.
    std::vector<unsigned> Sizes;
    for (unsigned i = 0; i < in_dimensions; i++) {
      assert(isDimBoundedByConstant(CurrentStmtDomain, i) &&
             "Domain boundary are not constant.");
      auto UpperBound = getConstant(CurrentStmtDomain.dim_max(i), true, false);
      assert(!UpperBound.is_null() && UpperBound.is_pos() &&
             !UpperBound.is_nan() &&
             "The upper bound is not a positive integer.");
      assert(UpperBound.le(isl::val(CurrentAccessMap.ctx(),
                                    std::numeric_limits<int>::max() - 1)) &&
             "The upper bound overflow a int.");
      Sizes.push_back(UpperBound.get_num_si() + 1);
    }

    // Get the ElementType of the current SAI.
    auto ElementType = MA->getLatestScopArrayInfo()->getElementType();

    // Create (or get if already existing) the new expanded SAI.
    auto ExpandedSAI =
        S.createScopArrayInfo(ElementType, CurrentOutIdString, Sizes);
    ExpandedSAI->setIsOnHeap(true);

    // Get the out Id of the expanded Array.
    auto NewOutId = ExpandedSAI->getBasePtrId();

    // Set the out id of the new AM to the new SAI id.
    NewAccessMap = NewAccessMap.set_tuple_id(isl::dim::out, NewOutId);

    // Add constraints to linked output with input id.
    auto SpaceMap = NewAccessMap.get_space();
    auto ConstraintBasicMap = isl::basic_map::equal(
        SpaceMap, unsignedFromIslSize(SpaceMap.dim(isl::dim::in)));
    NewAccessMap = isl::map(ConstraintBasicMap);

    // Set the new access relation map.
    MA->setNewAccessRelation(NewAccessMap);

    return ExpandedSAI;
  }

  /// Expand PHI memory accesses.
  ///
  /// @param The SCop in which the memory access appears in.
  /// @param The ScopArrayInfo representing the PHI accesses to expand.
  /// @param Dependences The RAW dependences of the SCop.
  void expandPhi(Scop &S, const ScopArrayInfo *SAI,
                 const isl::union_map &Dependences) {
    SmallPtrSet<MemoryAccess *, 4> Writes;
    for (auto MA : S.getPHIIncomings(SAI))
      Writes.insert(MA);
    auto Read = S.getPHIRead(SAI);
    auto ExpandedSAI = expandAccess(Read);

    mapAccess(Writes, Dependences, ExpandedSAI, false);
  }

public:
  MaximalStaticExpansionImpl(Scop &S, isl::union_map &Dependences,
                             OptimizationRemarkEmitter &ORE)
      : ORE(ORE), S(S), Dependences(Dependences) {}

  /// Expand the accesses of the SCoP
  ///
  /// @param S The SCoP that must be expanded
  /// @param D The dependencies information of SCoP
  void expand() {
    SmallVector<ScopArrayInfo *, 4> CurrentSAI(S.arrays().begin(),
                                               S.arrays().end());
    for (auto SAI : CurrentSAI) {
      SmallPtrSet<MemoryAccess *, 4> AllWrites;
      SmallPtrSet<MemoryAccess *, 4> AllReads;
      if (!isExpandable(SAI, AllWrites, AllReads, S))
        continue;

      if (SAI->isValueKind() || SAI->isArrayKind()) {
        assert(AllWrites.size() == 1 || SAI->isValueKind());

        auto TheWrite = *(AllWrites.begin());
        ScopArrayInfo *ExpandedArray = expandAccess(TheWrite);

        mapAccess(AllReads, Dependences, ExpandedArray, true);
      } else if (SAI->isPHIKind()) {
        expandPhi(S, SAI, Dependences);
      }
    }
  }

  /// Dump the internal information about a performed MSE to @p OS.
  void print(llvm::raw_ostream &OS) {
    OS << "After arrays {\n";

    for (auto &Array : S.arrays())
      Array->print(OS);

    OS << "}\n";

    OS << "After accesses {\n";
    for (auto &Stmt : S) {
      OS.indent(4) << Stmt.getBaseName() << "{\n";
      for (auto *MA : Stmt)
        MA->print(OS);
      OS.indent(4) << "}\n";
    }
    OS << "}\n";
  }
};

static std::unique_ptr<MaximalStaticExpansionImpl>
runMaximalStaticExpansion(Scop &S, OptimizationRemarkEmitter &ORE,
                          const Dependences &D) {
  auto Dependences = D.getDependences(Dependences::TYPE_RAW);

  std::unique_ptr<MaximalStaticExpansionImpl> Impl =
      std::make_unique<MaximalStaticExpansionImpl>(S, Dependences, ORE);

  Impl->expand();
  return Impl;
}

static PreservedAnalyses runMSEUsingNPM(Scop &S, ScopAnalysisManager &SAM,
                                        ScopStandardAnalysisResults &SAR,
                                        raw_ostream *OS) {
  OptimizationRemarkEmitter ORE(&S.getFunction());

  auto &DI = SAM.getResult<DependenceAnalysis>(S, SAR);
  auto &D = DI.getDependences(Dependences::AL_Reference);

  std::unique_ptr<MaximalStaticExpansionImpl> Impl =
      runMaximalStaticExpansion(S, ORE, D);

  if (OS) {
    *OS << "Printing analysis 'Polly - Maximal static expansion of SCoP' for "
           "region: '"
        << S.getName() << "' in function '" << S.getFunction().getName()
        << "':\n";

    if (Impl) {
      *OS << "MSE result:\n";
      Impl->print(*OS);
    }
  }

  return PreservedAnalyses::all();
}

} // namespace

PreservedAnalyses
MaximalStaticExpansionPass::run(Scop &S, ScopAnalysisManager &SAM,
                                ScopStandardAnalysisResults &SAR,
                                SPMUpdater &) {
  return runMSEUsingNPM(S, SAM, SAR, nullptr);
}

PreservedAnalyses
MaximalStaticExpansionPrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                                       ScopStandardAnalysisResults &SAR,
                                       SPMUpdater &) {
  return runMSEUsingNPM(S, SAM, SAR, &OS);
}

char MaximalStaticExpanderWrapperPass::ID = 0;

bool MaximalStaticExpanderWrapperPass::runOnScop(Scop &S) {
  // Get the ORE from OptimizationRemarkEmitterWrapperPass.
  OptimizationRemarkEmitter *ORE =
      &getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  // Get the RAW Dependences.
  auto &DI = getAnalysis<DependenceInfo>();
  auto &D = DI.getDependences(Dependences::AL_Reference);

  std::unique_ptr<MaximalStaticExpansionImpl> Impl =
      runMaximalStaticExpansion(S, *ORE, D);

  return false;
}

void MaximalStaticExpanderWrapperPass::printScop(raw_ostream &OS,
                                                 Scop &S) const {
  S.print(OS, false);
}

void MaximalStaticExpanderWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
}

Pass *polly::createMaximalStaticExpansionPass() {
  return new MaximalStaticExpanderWrapperPass();
}

INITIALIZE_PASS_BEGIN(MaximalStaticExpanderWrapperPass, "polly-mse",
                      "Polly - Maximal static expansion of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(OptimizationRemarkEmitterWrapperPass);
INITIALIZE_PASS_END(MaximalStaticExpanderWrapperPass, "polly-mse",
                    "Polly - Maximal static expansion of SCoP", false, false)
